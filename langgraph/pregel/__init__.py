from __future__ import annotations

import asyncio
import concurrent.futures
from collections import defaultdict, deque
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    overload,
)

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.globals import get_debug
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator
from langchain_core.runnables import (
    Runnable,
    RunnableSerializable,
)
from langchain_core.runnables.base import Input, Output, coerce_to_runnable
from langchain_core.runnables.config import (
    RunnableConfig,
    get_executor_for_config,
    patch_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    get_unique_config_specs,
)
from langchain_core.tracers.log_stream import LogStreamCallbackHandler

from langgraph.channels.base import (
    AsyncChannelsManager,
    BaseChannel,
    ChannelsManager,
    EmptyChannelError,
    create_checkpoint,
)
from langgraph.channels.last_value import LastValue
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointAt,
    empty_checkpoint,
)
from langgraph.constants import CONFIG_KEY_READ, CONFIG_KEY_SEND
from langgraph.pregel.debug import print_checkpoint, print_step_start
from langgraph.pregel.io import map_input, map_output
from langgraph.pregel.log import logger
from langgraph.pregel.read import ChannelBatch, ChannelInvoke
from langgraph.pregel.reserved import ReservedChannels
from langgraph.pregel.validate import validate_graph
from langgraph.pregel.write import ChannelWrite

WriteValue = Union[
    Runnable[Input, Output],
    Callable[[Input], Output],
    Callable[[Input], Awaitable[Output]],
    Any,
]


def _coerce_write_value(value: WriteValue) -> Runnable[Input, Output]:
    if not isinstance(value, Runnable) and not callable(value):
        return coerce_to_runnable(lambda _: value)
    return coerce_to_runnable(value)


class Channel:
    @overload
    @classmethod
    def subscribe_to(
        cls,
        channels: str,
        *,
        key: Optional[str] = None,
        when: Optional[Callable[[Any], bool]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> ChannelInvoke:
        ...

    @overload
    @classmethod
    def subscribe_to(
        cls,
        channels: Sequence[str],
        *,
        key: None = None,
        when: Optional[Callable[[Any], bool]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> ChannelInvoke:
        ...

    @classmethod
    def subscribe_to(
        cls,
        channels: Union[str, Sequence[str]],
        *,
        key: Optional[str] = None,
        when: Optional[Callable[[Any], bool]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> ChannelInvoke:
        """Runs process.invoke() each time channels are updated,
        with a dict of the channel values as input."""
        if not isinstance(channels, str) and key is not None:
            raise ValueError(
                "Can't specify a key when subscribing to multiple channels"
            )
        return ChannelInvoke(
            channels=cast(
                Union[Mapping[None, str], Mapping[str, str]],
                {key: channels}
                if isinstance(channels, str)
                else {chan: chan for chan in channels},
            ),
            triggers=[channels] if isinstance(channels, str) else channels,
            when=when,
            tags=tags,
        )

    @classmethod
    def subscribe_to_each(cls, inbox: str, key: Optional[str] = None) -> ChannelBatch:
        """Runs process.batch() with the content of inbox each time it is updated."""
        return ChannelBatch(channel=inbox, key=key)

    @classmethod
    def write_to(
        cls,
        *channels: str,
        **kwargs: WriteValue,
    ) -> ChannelWrite:
        """Writes to channels the result of the lambda, or None to skip writing."""
        return ChannelWrite(
            channels=(
                [(c, None) for c in channels]
                + [(k, _coerce_write_value(v)) for k, v in kwargs.items()]
            )
        )


class Pregel(
    RunnableSerializable[Union[dict[str, Any], Any], Union[dict[str, Any], Any]]
):
    nodes: Mapping[str, Union[ChannelInvoke, ChannelBatch]]

    channels: Mapping[str, BaseChannel] = Field(default_factory=dict)

    output: Union[str, Sequence[str]] = "output"

    hidden: Sequence[str] = Field(default_factory=list)

    interrupt: Sequence[str] = Field(default_factory=list)

    input: Union[str, Sequence[str]] = "input"

    step_timeout: Optional[float] = None

    debug: bool = Field(default_factory=get_debug)

    checkpointer: Optional[BaseCheckpointSaver] = None

    name: str = "LangGraph"

    class Config:
        arbitrary_types_allowed = True

    @root_validator(skip_on_failure=True)
    def validate_pregel(cls, values: dict[str, Any]) -> dict[str, Any]:
        validate_graph(
            values["nodes"], values["channels"], values["input"], values["output"]
        )
        return values

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            [spec for node in self.nodes.values() for spec in node.config_specs]
            + (self.checkpointer.config_specs if self.checkpointer is not None else [])
        )

    @property
    def InputType(self) -> Any:
        if isinstance(self.input, str):
            return self.channels[self.input].UpdateType

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        if isinstance(self.input, str):
            return super().get_input_schema(config)
        else:
            return create_model(  # type: ignore[call-overload]
                self.get_name("Input"),
                **{
                    k: (self.channels[k].UpdateType, None)
                    for k in self.input or self.channels.keys()
                },
            )

    @property
    def OutputType(self) -> Any:
        if isinstance(self.output, str):
            return self.channels[self.output].ValueType

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        if isinstance(self.output, str):
            return super().get_output_schema(config)
        else:
            return create_model(  # type: ignore[call-overload]
                self.get_name("Output"),
                **{k: (self.channels[k].ValueType, None) for k in self.output},
            )

    def _transform(
        self,
        input: Iterator[Union[dict[str, Any], Any]],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        *,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
    ) -> Iterator[Union[dict[str, Any], Any]]:
        if config["recursion_limit"] < 1:
            raise ValueError("recursion_limit must be at least 1")
        # assign defaults
        if output_keys is None:
            output_keys = [chan for chan in self.channels if chan not in self.hidden]
        if input_keys is None:
            input_keys = self.input
        # copy nodes to ignore mutations during execution
        processes = {**self.nodes}
        # get checkpoint from saver, or create an empty one
        checkpoint = self.checkpointer.get(config) if self.checkpointer else None
        checkpoint = checkpoint or empty_checkpoint()
        # create channels from checkpoint
        with ChannelsManager(
            self.channels, checkpoint
        ) as channels, get_executor_for_config(config) as executor:
            # map inputs to channel updates
            _apply_writes(
                checkpoint,
                channels,
                deque(w for c in input for w in map_input(input_keys, c)),
                config,
                0,
            )

            read = partial(_read_channel, channels)

            # Similarly to Bulk Synchronous Parallel / Pregel model
            # computation proceeds in steps, while there are channel updates
            # channel updates from step N are only visible in step N+1
            # channels are guaranteed to be immutable for the duration of the step,
            # with channel updates applied only at the transition between steps
            for step in range(config["recursion_limit"]):
                next_tasks = _prepare_next_tasks(checkpoint, processes, channels)

                # if no more tasks, we're done
                if not next_tasks:
                    break

                if self.debug:
                    print_step_start(step, next_tasks)

                # collect all writes to channels, without applying them yet
                pending_writes = deque[tuple[str, Any]]()

                # prepare tasks with config
                tasks_w_config = [
                    (
                        proc,
                        input,
                        patch_config(
                            config,
                            run_name=name,
                            callbacks=run_manager.get_child(f"graph:step:{step}"),
                            configurable={
                                # deque.extend is thread-safe
                                CONFIG_KEY_SEND: pending_writes.extend,
                                CONFIG_KEY_READ: read,
                            },
                        ),
                    )
                    for proc, input, name in next_tasks
                ]

                # execute tasks, and wait for one to fail or all to finish.
                # each task is independent from all other concurrent tasks
                done, inflight = concurrent.futures.wait(
                    [
                        executor.submit(proc.invoke, input, config)
                        for proc, input, config in tasks_w_config
                    ],
                    return_when=concurrent.futures.FIRST_EXCEPTION,
                    timeout=self.step_timeout,
                )

                # interrupt on failure or timeout
                _interrupt_or_proceed(done, inflight, step)

                # apply writes to channels
                _apply_writes(checkpoint, channels, pending_writes, config, step + 1)

                if self.debug:
                    print_checkpoint(step, channels)

                # yield current value and checkpoint view
                if step_output := map_output(output_keys, pending_writes, channels):
                    yield step_output
                    # we can detect updates when output is multiple channels (ie. dict)
                    if not isinstance(output_keys, str):
                        # if view was updated, apply writes to channels
                        _apply_writes_from_view(checkpoint, channels, step_output)

                # save end of step checkpoint
                if (
                    self.checkpointer is not None
                    and self.checkpointer.at == CheckpointAt.END_OF_STEP
                ):
                    checkpoint = create_checkpoint(checkpoint, channels)
                    self.checkpointer.put(config, checkpoint)

                # interrupt if any channel written to is in interrupt list
                if any(chan for chan, _ in pending_writes if chan in self.interrupt):
                    break

            # save end of run checkpoint
            if (
                self.checkpointer is not None
                and self.checkpointer.at == CheckpointAt.END_OF_RUN
            ):
                checkpoint = create_checkpoint(checkpoint, channels)
                self.checkpointer.put(config, checkpoint)

    async def _atransform(
        self,
        input: AsyncIterator[Union[dict[str, Any], Any]],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        *,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
    ) -> AsyncIterator[Union[dict[str, Any], Any]]:
        if config["recursion_limit"] < 1:
            raise ValueError("recursion_limit must be at least 1")
        # if running from astream_log() run each proc with streaming
        do_stream = next(
            (
                h
                for h in run_manager.handlers
                if isinstance(h, LogStreamCallbackHandler)
            ),
            None,
        )
        # assign defaults
        if output_keys is None:
            output_keys = [chan for chan in self.channels if chan not in self.hidden]
        if input_keys is None:
            input_keys = self.input
        # copy nodes to ignore mutations during execution
        processes = {**self.nodes}
        # get checkpoint from saver, or create an empty one
        checkpoint = await self.checkpointer.aget(config) if self.checkpointer else None
        checkpoint = checkpoint or empty_checkpoint()
        # create channels from checkpoint
        async with AsyncChannelsManager(self.channels, checkpoint) as channels:
            # map inputs to channel updates
            _apply_writes(
                checkpoint,
                channels,
                deque([w async for c in input for w in map_input(input_keys, c)]),
                config,
                0,
            )

            read = partial(_read_channel, channels)

            # Similarly to Bulk Synchronous Parallel / Pregel model
            # computation proceeds in steps, while there are channel updates
            # channel updates from step N are only visible in step N+1,
            # channels are guaranteed to be immutable for the duration of the step,
            # channel updates being applied only at the transition between steps
            for step in range(config["recursion_limit"]):
                next_tasks = _prepare_next_tasks(checkpoint, processes, channels)

                # if no more tasks, we're done
                if not next_tasks:
                    break

                if self.debug:
                    print_step_start(step, next_tasks)

                # collect all writes to channels, without applying them yet
                pending_writes = deque[tuple[str, Any]]()

                # prepare tasks with config
                tasks_w_config = [
                    (
                        proc,
                        input,
                        patch_config(
                            config,
                            run_name=name,
                            callbacks=run_manager.get_child(f"graph:step:{step}"),
                            configurable={
                                # deque.extend is thread-safe
                                CONFIG_KEY_SEND: pending_writes.extend,
                                CONFIG_KEY_READ: read,
                            },
                        ),
                    )
                    for proc, input, name in next_tasks
                ]

                # execute tasks, and wait for one to fail or all to finish.
                # each task is independent from all other concurrent tasks
                done, inflight = await asyncio.wait(
                    [
                        asyncio.create_task(_aconsume(proc.astream(input, config)))
                        for proc, input, config in tasks_w_config
                    ]
                    if do_stream
                    else [
                        asyncio.create_task(proc.ainvoke(input, config))
                        for proc, input, config in tasks_w_config
                    ],
                    return_when=asyncio.FIRST_EXCEPTION,
                    timeout=self.step_timeout,
                )

                # interrupt on failure or timeout
                _interrupt_or_proceed(done, inflight, step)

                # apply writes to channels
                _apply_writes(checkpoint, channels, pending_writes, config, step + 1)

                if self.debug:
                    print_checkpoint(step, channels)

                # yield current value and checkpoint view
                if step_output := map_output(output_keys, pending_writes, channels):
                    yield step_output
                    # we can detect updates when output is multiple channels (ie. dict)
                    if not isinstance(output_keys, str):
                        # if view was updated, apply writes to channels
                        _apply_writes_from_view(checkpoint, channels, step_output)

                # save end of step checkpoint
                if (
                    self.checkpointer is not None
                    and self.checkpointer.at == CheckpointAt.END_OF_STEP
                ):
                    checkpoint = create_checkpoint(checkpoint, channels)
                    await self.checkpointer.aput(config, checkpoint)

                # interrupt if any channel written to is in interrupt list
                if any(chan for chan, _ in pending_writes if chan in self.interrupt):
                    break

            # save end of run checkpoint
            if (
                self.checkpointer is not None
                and self.checkpointer.at == CheckpointAt.END_OF_RUN
            ):
                checkpoint = create_checkpoint(checkpoint, channels)
                await self.checkpointer.aput(config, checkpoint)

    def invoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        latest: Union[dict[str, Any], Any] = None
        for chunk in self.stream(
            input,
            config,
            output_keys=output_keys if output_keys is not None else self.output,
            input_keys=input_keys,
            **kwargs,
        ):
            latest = chunk
        return latest

    def stream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[dict[str, Any], Any]]:
        return self.transform(
            iter([input]),
            config,
            output_keys=output_keys,
            input_keys=input_keys,
            **kwargs,
        )

    def transform(
        self,
        input: Iterator[Union[dict[str, Any], Any]],
        config: Optional[RunnableConfig] = None,
        *,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[dict[str, Any], Any]]:
        for chunk in self._transform_stream_with_config(
            input,
            self._transform,
            config,
            output_keys=output_keys,
            input_keys=input_keys,
            **kwargs,
        ):
            yield chunk

    async def ainvoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        latest: Union[dict[str, Any], Any] = None
        async for chunk in self.astream(
            input,
            config,
            output_keys=output_keys if output_keys is not None else self.output,
            input_keys=input_keys,
            **kwargs,
        ):
            latest = chunk
        return latest

    async def astream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[dict[str, Any], Any]]:
        async def input_stream() -> AsyncIterator[Union[dict[str, Any], Any]]:
            yield input

        async for chunk in self.atransform(
            input_stream(),
            config,
            output_keys=output_keys,
            input_keys=input_keys,
            **kwargs,
        ):
            yield chunk

    async def atransform(
        self,
        input: AsyncIterator[Union[dict[str, Any], Any]],
        config: Optional[RunnableConfig] = None,
        *,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[dict[str, Any], Any]]:
        async for chunk in self._atransform_stream_with_config(
            input,
            self._atransform,
            config,
            output_keys=output_keys,
            input_keys=input_keys,
            **kwargs,
        ):
            yield chunk


def _interrupt_or_proceed(
    done: Union[set[concurrent.futures.Future[Any]], set[asyncio.Task[Any]]],
    inflight: Union[set[concurrent.futures.Future[Any]], set[asyncio.Task[Any]]],
    step: int,
) -> None:
    while done:
        # if any task failed
        if exc := done.pop().exception():
            # cancel all pending tasks
            while inflight:
                inflight.pop().cancel()
            # raise the exception
            raise exc
            # TODO this is where retry of an entire step would happen

    if inflight:
        # if we got here means we timed out
        while inflight:
            # cancel all pending tasks
            inflight.pop().cancel()
        # raise timeout error
        raise TimeoutError(f"Timed out at step {step}")


def _read_channel(
    channels: Mapping[str, BaseChannel], chan: str, catch: bool = True
) -> Any:
    try:
        return channels[chan].get()
    except EmptyChannelError:
        if catch:
            return None
        else:
            raise


def _apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    pending_writes: Sequence[tuple[str, Any]],
    config: RunnableConfig,
    for_step: int,
) -> None:
    pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
    # Group writes by channel
    for chan, val in pending_writes:
        if chan in [c.value for c in ReservedChannels]:
            raise ValueError(f"Can't write to reserved channel {chan}")
        pending_writes_by_channel[chan].append(val)

    # Update reserved channels
    pending_writes_by_channel[ReservedChannels.is_last_step] = [
        for_step + 1 == config["recursion_limit"]
    ]

    updated_channels: set[str] = set()
    # Apply writes to channels
    for chan, vals in pending_writes_by_channel.items():
        if chan in channels:
            channels[chan].update(vals)
            checkpoint["channel_versions"][chan] += 1
            updated_channels.add(chan)
        else:
            logger.warning(f"Skipping write for channel {chan} which has no readers")
    # Channels that weren't updated in this step are notified of a new step
    for chan in channels:
        if chan not in updated_channels:
            channels[chan].update([])


def _apply_writes_from_view(
    checkpoint: Checkpoint, channels: Mapping[str, BaseChannel], values: dict[str, Any]
) -> None:
    for chan, value in values.items():
        if value == _read_channel(channels, chan):
            continue

        assert isinstance(channels[chan], LastValue), (
            f"Can't modify channel {chan} of type "
            f"{channels[chan].__class__.__name__}"
        )
        checkpoint["channel_versions"][chan] += 1
        channels[chan].update([values[chan]])


def _prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, Union[ChannelInvoke, ChannelBatch]],
    channels: Mapping[str, BaseChannel],
) -> list[tuple[Runnable, Any, str]]:
    tasks: list[tuple[Runnable, Any, str]] = []
    # Check if any processes should be run in next step
    # If so, prepare the values to be passed to them
    for name, proc in processes.items():
        seen = checkpoint["versions_seen"][name]
        if isinstance(proc, ChannelInvoke):
            # If any of the channels read by this process were updated
            if any(
                checkpoint["channel_versions"][chan] > seen[chan]
                for chan in proc.triggers
            ):
                # If all channels subscribed by this process have been initialized
                try:
                    val: Any = {
                        k: _read_channel(
                            channels, chan, catch=chan not in proc.triggers
                        )
                        for k, chan in proc.channels.items()
                    }
                except EmptyChannelError:
                    continue

                # Processes that subscribe to a single keyless channel get
                # the value directly, instead of a dict
                if list(proc.channels.keys()) == [None]:
                    val = val[None]

                # update seen versions
                seen.update(
                    {
                        chan: checkpoint["channel_versions"][chan]
                        for chan in proc.triggers
                    }
                )

                # skip if condition is not met
                if proc.when is None or proc.when(val):
                    tasks.append((proc, val, name))
        elif isinstance(proc, ChannelBatch):
            # If the channel read by this process was updated
            if checkpoint["channel_versions"][proc.channel] > seen[proc.channel]:
                # Here we don't catch EmptyChannelError because the channel
                # must be intialized if the previous `if` condition is true
                val = channels[proc.channel].get()
                if proc.key is not None:
                    val = [{proc.key: v} for v in val]

                tasks.append((proc, val, name))
                seen[proc.channel] = checkpoint["channel_versions"][proc.channel]

    return tasks


async def _aconsume(iterator: AsyncIterator[Any]) -> None:
    """Consume an async iterator."""
    async for _ in iterator:
        pass
