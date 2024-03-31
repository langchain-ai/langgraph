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
    Literal,
    Mapping,
    NamedTuple,
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
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
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
    create_model,
    get_unique_config_specs,
)
from langchain_core.tracers.log_stream import LogStreamCallbackHandler

from langgraph.channels.base import (
    AsyncChannelsManager,
    BaseChannel,
    ChannelsManager,
    EmptyChannelError,
    InvalidUpdateError,
    create_checkpoint,
)
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointAt,
    copy_checkpoint,
    empty_checkpoint,
)
from langgraph.constants import CONFIG_KEY_READ, CONFIG_KEY_SEND, INTERRUPT
from langgraph.pregel.debug import print_checkpoint, print_step_start
from langgraph.pregel.io import map_input, map_output_updates, map_output_values
from langgraph.pregel.log import logger
from langgraph.pregel.read import ChannelInvoke
from langgraph.pregel.reserved import AllReservedChannels, ReservedChannels
from langgraph.pregel.validate import validate_graph, validate_keys
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry

WriteValue = Union[
    Runnable[Input, Output],
    Callable[[Input], Output],
    Callable[[Input], Awaitable[Output]],
    Any,
]


class GraphRecursionError(RecursionError):
    pass


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
        tags: Optional[list[str]] = None,
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
        tags: Optional[list[str]] = None,
    ) -> ChannelInvoke:
        ...

    @classmethod
    def subscribe_to(
        cls,
        channels: Union[str, Sequence[str]],
        *,
        key: Optional[str] = None,
        when: Optional[Callable[[Any], bool]] = None,
        tags: Optional[list[str]] = None,
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
    def write_to(
        cls,
        *channels: str,
        **kwargs: WriteValue,
    ) -> ChannelWrite:
        """Writes to channels the result of the lambda, or None to skip writing."""
        return ChannelWrite(
            channels=(
                [ChannelWriteEntry(c, None, False) for c in channels]
                + [
                    ChannelWriteEntry(k, _coerce_write_value(v), True)
                    for k, v in kwargs.items()
                ]
            )
        )


StreamMode = Literal["values", "updates"]


class StateSnapshot(NamedTuple):
    values: dict[str, Any] | Any
    """Current values of channels"""
    next: tuple[str]
    """Nodes to execute in the next step, if any"""
    config: RunnableConfig
    """Config used to fetch this snapshot"""
    parent_config: Optional[RunnableConfig] = None
    """Config used to fetch the parent snapshot, if any"""


class Pregel(
    RunnableSerializable[Union[dict[str, Any], Any], Union[dict[str, Any], Any]]
):
    nodes: Mapping[str, ChannelInvoke]

    channels: Mapping[str, BaseChannel] = Field(default_factory=dict)

    stream_mode: StreamMode = "values"

    output_channels: Union[str, Sequence[str]] = "output"
    """Channels to output, defaults to channel named 'output'."""

    stream_channels: Optional[Union[str, Sequence[str]]] = None
    """Channels to stream, defaults to all channels not in reserved channels"""

    interrupt_after_nodes: Sequence[str] = Field(default_factory=list)

    interrupt_before_nodes: Sequence[str] = Field(default_factory=list)

    input_channels: Union[str, Sequence[str]] = "input"

    step_timeout: Optional[float] = None

    debug: bool = Field(default_factory=get_debug)

    checkpointer: Optional[BaseCheckpointSaver] = None

    name: str = "LangGraph"

    class Config:
        arbitrary_types_allowed = True

    @root_validator(skip_on_failure=True)
    def validate_pregel(cls, values: dict[str, Any]) -> dict[str, Any]:
        validate_graph(
            values["nodes"],
            values["channels"],
            values["input_channels"],
            values["output_channels"],
            values["stream_channels"],
            values["interrupt_after_nodes"],
            values["interrupt_before_nodes"],
        )
        if values["interrupt_after_nodes"] or values["interrupt_before_nodes"]:
            if not values["checkpointer"]:
                raise ValueError("Interrupts require a checkpointer")
        return values

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [
            spec
            for spec in get_unique_config_specs(
                [spec for node in self.nodes.values() for spec in node.config_specs]
                + (
                    self.checkpointer.config_specs
                    if self.checkpointer is not None
                    else []
                )
            )
            # these are provided by the Pregel class
            if spec.id not in [CONFIG_KEY_READ, CONFIG_KEY_SEND]
        ]

    @property
    def InputType(self) -> Any:
        if isinstance(self.input_channels, str):
            return self.channels[self.input_channels].UpdateType

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        if isinstance(self.input_channels, str):
            return super().get_input_schema(config)
        else:
            return create_model(  # type: ignore[call-overload]
                self.get_name("Input"),
                **{
                    k: (self.channels[k].UpdateType, None)
                    for k in self.input_channels or self.channels.keys()
                },
            )

    @property
    def OutputType(self) -> Any:
        if isinstance(self.output_channels, str):
            return self.channels[self.output_channels].ValueType

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        if isinstance(self.output_channels, str):
            return super().get_output_schema(config)
        else:
            return create_model(  # type: ignore[call-overload]
                self.get_name("Output"),
                **{k: (self.channels[k].ValueType, None) for k in self.output_channels},
            )

    @property
    def snapshot_channels_list(self) -> Sequence[str]:
        return (
            [self.stream_channels]
            if isinstance(self.stream_channels, str)
            else self.stream_channels
            or [k for k in self.channels if k not in AllReservedChannels]
        )

    def get_state(self, config: RunnableConfig) -> StateSnapshot:
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        saved = self.checkpointer.get_tuple(config)
        checkpoint = saved.checkpoint if saved else empty_checkpoint()
        config = saved.config if saved else config
        with ChannelsManager(self.channels, checkpoint) as channels:
            _, next_tasks = _prepare_next_tasks(
                checkpoint, self.nodes, channels, for_execution=False
            )
            values = {
                k: _read_channel(channels, k)
                for k in channels
                if k in self.snapshot_channels_list
            }
            return StateSnapshot(
                values[self.stream_channels]
                if isinstance(self.stream_channels, str)
                else values,
                tuple(name for _, _, name in next_tasks),
                config,
            )

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        saved = await self.checkpointer.aget_tuple(config)
        checkpoint = saved.checkpoint if saved else empty_checkpoint()
        config = saved.config if saved else config
        async with AsyncChannelsManager(self.channels, checkpoint) as channels:
            _, next_tasks = _prepare_next_tasks(
                checkpoint, self.nodes, channels, for_execution=False
            )
            values = {
                k: _read_channel(channels, k)
                for k in channels
                if k in self.snapshot_channels_list
            }
            return StateSnapshot(
                values[self.stream_channels]
                if isinstance(self.stream_channels, str)
                else values,
                tuple(name for _, _, name in next_tasks),
                config,
            )

    def get_state_history(self, config: RunnableConfig) -> Iterator[StateSnapshot]:
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        for config, checkpoint, parent_config in self.checkpointer.list(config):
            with ChannelsManager(self.channels, checkpoint) as channels:
                _, next_tasks = _prepare_next_tasks(
                    checkpoint, self.nodes, channels, for_execution=False
                )
                values = {
                    k: _read_channel(channels, k)
                    for k in channels
                    if k in self.snapshot_channels_list
                }
                yield StateSnapshot(
                    values[self.stream_channels]
                    if isinstance(self.stream_channels, str)
                    else values,
                    tuple(name for _, _, name in next_tasks),
                    config,
                    parent_config,
                )

    async def aget_state_history(
        self, config: RunnableConfig
    ) -> AsyncIterator[StateSnapshot]:
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        async for config, checkpoint, parent_config in self.checkpointer.alist(config):
            async with AsyncChannelsManager(self.channels, checkpoint) as channels:
                _, next_tasks = _prepare_next_tasks(
                    checkpoint, self.nodes, channels, for_execution=False
                )
                values = {
                    k: _read_channel(channels, k)
                    for k in channels
                    if k in self.snapshot_channels_list
                }
                yield StateSnapshot(
                    values[self.stream_channels]
                    if isinstance(self.stream_channels, str)
                    else values,
                    tuple(name for _, _, name in next_tasks),
                    config,
                    parent_config,
                )

    def update_state(
        self, config: RunnableConfig, values: dict[str, Any] | Any
    ) -> RunnableConfig:
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        values = (
            {self.stream_channels: values}
            if isinstance(self.stream_channels, str)
            else values
        )
        checkpoint = self.checkpointer.get(config)
        checkpoint = copy_checkpoint(checkpoint) if checkpoint else empty_checkpoint()
        with ChannelsManager(self.channels, checkpoint) as channels:
            for k, v in values.items():
                channels[k].update([v])
                checkpoint["channel_versions"][k] += 1
            for k in self.snapshot_channels_list:
                version = checkpoint["channel_versions"][k]
                checkpoint["versions_seen"][INTERRUPT][k] = version
            return self.checkpointer.put(
                config, create_checkpoint(checkpoint, channels)
            )

    async def aupdate_state(
        self, config: RunnableConfig, values: dict[str, Any] | Any
    ) -> RunnableConfig:
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        values = (
            {self.stream_channels: values}
            if isinstance(self.stream_channels, str)
            else values
        )
        checkpoint = await self.checkpointer.aget(config)
        checkpoint = copy_checkpoint(checkpoint) if checkpoint else empty_checkpoint()
        async with AsyncChannelsManager(self.channels, checkpoint) as channels:
            for k, v in values.items():
                channels[k].update([v])
                checkpoint["channel_versions"][k] += 1
            for k in self.stream_channels or self.channels:
                version = checkpoint["channel_versions"][k]
                checkpoint["versions_seen"][INTERRUPT][k] = version
            return await self.checkpointer.aput(
                config, create_checkpoint(checkpoint, channels)
            )

    def _defaults(
        self,
        *,
        stream_mode: Optional[StreamMode] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before_nodes: Optional[Sequence[str]] = None,
        interrupt_after_nodes: Optional[Sequence[str]] = None,
        debug: Optional[bool] = None,
    ) -> tuple[
        bool,
        StreamMode,
        Union[str, Sequence[str]],
        Union[str, Sequence[str]],
        Optional[Sequence[str]],
        Optional[Sequence[str]],
    ]:
        debug = debug if debug is not None else self.debug
        if output_keys is None:
            output_keys = (
                [chan for chan in self.channels]
                if self.stream_channels is None
                else self.stream_channels
            )
        else:
            validate_keys(output_keys, self.channels)
        if input_keys is None:
            input_keys = self.input_channels
        else:
            validate_keys(input_keys, self.channels)
        interrupt_before_nodes = interrupt_before_nodes or self.interrupt_before_nodes
        interrupt_after_nodes = interrupt_after_nodes or self.interrupt_after_nodes
        return (
            debug,
            stream_mode if stream_mode is not None else self.stream_mode,
            input_keys,
            output_keys,
            interrupt_before_nodes,
            interrupt_after_nodes,
        )

    def _transform(
        self,
        input: Iterator[Union[dict[str, Any], Any]],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[Union[dict[str, Any], Any]]:
        try:
            if config["recursion_limit"] < 1:
                raise ValueError("recursion_limit must be at least 1")
            # assign defaults
            (
                debug,
                stream_mode,
                input_keys,
                output_keys,
                interrupt_before_nodes,
                interrupt_after_nodes,
            ) = self._defaults(**kwargs)
            # copy nodes to ignore mutations during execution
            processes = {**self.nodes}
            # get checkpoint from saver, or create an empty one
            checkpoint_config = config
            checkpoint = (
                self.checkpointer.get(checkpoint_config) if self.checkpointer else None
            )
            checkpoint = checkpoint or empty_checkpoint()
            # create channels from checkpoint
            with ChannelsManager(
                self.channels, checkpoint
            ) as channels, get_executor_for_config(config) as executor:
                # map inputs to channel updates
                if input_writes := deque(
                    w for c in input for w in map_input(input_keys, c)
                ):
                    # discard any unfinished tasks from previous checkpoint
                    checkpoint, _ = _prepare_next_tasks(
                        checkpoint, processes, channels, for_execution=True
                    )
                    # apply input writes
                    _apply_writes(
                        checkpoint,
                        channels,
                        input_writes,
                        config,
                        0,
                    )
                else:
                    # if received no input, take that as signal to proceed
                    # past previous interrupt, if any
                    checkpoint = copy_checkpoint(checkpoint)
                    for k in self.snapshot_channels_list:
                        version = checkpoint["channel_versions"][k]
                        checkpoint["versions_seen"][INTERRUPT][k] = version

                read = partial(_read_channel, channels)

                # Similarly to Bulk Synchronous Parallel / Pregel model
                # computation proceeds in steps, while there are channel updates
                # channel updates from step N are only visible in step N+1
                # channels are guaranteed to be immutable for the duration of the step,
                # with channel updates applied only at the transition between steps
                for step in range(config["recursion_limit"] + 1):
                    checkpoint, next_tasks = _prepare_next_tasks(
                        checkpoint, processes, channels, for_execution=True
                    )

                    # if no more tasks, we're done
                    if not next_tasks:
                        if step == 0:
                            raise ValueError("No tasks to run in graph.")
                        else:
                            break
                    elif step == config["recursion_limit"]:
                        raise GraphRecursionError(
                            f"Recursion limit of {config['recursion_limit']} reached"
                            "without hitting a stop condition. You can increase the "
                            "limit by setting the `recursion_limit` config key."
                        )

                    if debug:
                        print_step_start(step, next_tasks)

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
                                    CONFIG_KEY_SEND: writes.extend,
                                    CONFIG_KEY_READ: read,
                                },
                            ),
                        )
                        for proc, input, name, writes in next_tasks
                    ]

                    futures = [
                        executor.submit(proc.invoke, input, config)
                        for proc, input, config in tasks_w_config
                    ]

                    # execute tasks, and wait for one to fail or all to finish.
                    # each task is independent from all other concurrent tasks
                    done, inflight = concurrent.futures.wait(
                        futures,
                        return_when=concurrent.futures.FIRST_EXCEPTION,
                        timeout=self.step_timeout,
                    )

                    # panic on failure or timeout
                    _panic_or_proceed(done, inflight, step)

                    # combine pending writes from all tasks
                    pending_writes = deque[tuple[str, Any]]()
                    for _, _, _, writes in next_tasks:
                        pending_writes.extend(writes)

                    # apply writes to channels
                    _apply_writes(
                        checkpoint, channels, pending_writes, config, step + 1
                    )

                    if debug:
                        print_checkpoint(step, channels)

                    # yield current value or updates
                    if stream_mode == "values":
                        if step_output := map_output_values(
                            output_keys, pending_writes, channels
                        ):
                            yield step_output
                    else:
                        if step_output := map_output_updates(output_keys, next_tasks):
                            yield step_output

                    # with previous step's checkpoint
                    if _should_interrupt(
                        checkpoint,
                        interrupt_before_nodes,
                        self.snapshot_channels_list,
                        pending_writes,
                    ):
                        break

                    # save end of step checkpoint
                    if self.checkpointer is not None and (
                        self.checkpointer.at == CheckpointAt.END_OF_STEP
                        or interrupt_before_nodes
                    ):
                        checkpoint = create_checkpoint(checkpoint, channels)
                        checkpoint_config = self.checkpointer.put(
                            checkpoint_config, checkpoint
                        )

                    # with this step's checkpoint,
                    if _should_interrupt(
                        checkpoint,
                        interrupt_after_nodes,
                        self.snapshot_channels_list,
                        pending_writes,
                    ):
                        break

                # save end of run checkpoint
                if (
                    self.checkpointer is not None
                    and self.checkpointer.at == CheckpointAt.END_OF_RUN
                    and not interrupt_before_nodes
                ):
                    checkpoint = create_checkpoint(checkpoint, channels)
                    self.checkpointer.put(checkpoint_config, checkpoint)
        finally:
            # cancel any pending tasks when generator is interrupted
            try:
                for task in futures:
                    task.cancel()
            except NameError:
                pass

    async def _atransform(
        self,
        input: AsyncIterator[Union[dict[str, Any], Any]],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[Union[dict[str, Any], Any]]:
        try:
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
            (
                debug,
                stream_mode,
                input_keys,
                output_keys,
                interrupt_before_nodes,
                interrupt_after_nodes,
            ) = self._defaults(**kwargs)
            # copy nodes to ignore mutations during execution
            processes = {**self.nodes}
            # get checkpoint from saver, or create an empty one
            checkpoint_config = config
            checkpoint = (
                await self.checkpointer.aget(checkpoint_config)
                if self.checkpointer
                else None
            )
            checkpoint = checkpoint or empty_checkpoint()
            # create channels from checkpoint
            async with AsyncChannelsManager(self.channels, checkpoint) as channels:
                # map inputs to channel updates
                if input_writes := deque(
                    [w async for c in input for w in map_input(input_keys, c)]
                ):
                    # discard any unfinished tasks from previous checkpoint
                    checkpoint, _ = _prepare_next_tasks(
                        checkpoint, processes, channels, for_execution=True
                    )
                    # apply input writes
                    _apply_writes(
                        checkpoint,
                        channels,
                        input_writes,
                        config,
                        0,
                    )
                else:
                    # if received no input, take that as signal to proceed
                    # past previous interrupt, if any
                    checkpoint = copy_checkpoint(checkpoint)
                    for k in self.snapshot_channels_list:
                        version = checkpoint["channel_versions"][k]
                        checkpoint["versions_seen"][INTERRUPT][k] = version

                read = partial(_read_channel, channels)

                # Similarly to Bulk Synchronous Parallel / Pregel model
                # computation proceeds in steps, while there are channel updates
                # channel updates from step N are only visible in step N+1,
                # channels are guaranteed to be immutable for the duration of the step,
                # channel updates being applied only at the transition between steps
                for step in range(config["recursion_limit"] + 1):
                    checkpoint, next_tasks = _prepare_next_tasks(
                        checkpoint, processes, channels, for_execution=True
                    )

                    # if no more tasks, we're done
                    if not next_tasks:
                        break
                    elif step == config["recursion_limit"]:
                        raise GraphRecursionError(
                            f"Recursion limit of {config['recursion_limit']} reached"
                            "without hitting a stop condition. You can increase the limit"
                            "by setting the `recursion_limit` config key."
                        )

                    if debug:
                        print_step_start(step, next_tasks)

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
                                    CONFIG_KEY_SEND: writes.extend,
                                    CONFIG_KEY_READ: read,
                                },
                            ),
                        )
                        for proc, input, name, writes in next_tasks
                    ]

                    futures = (
                        [
                            asyncio.create_task(_aconsume(proc.astream(input, config)))
                            for proc, input, config in tasks_w_config
                        ]
                        if do_stream
                        else [
                            asyncio.create_task(proc.ainvoke(input, config))
                            for proc, input, config in tasks_w_config
                        ]
                    )

                    # execute tasks, and wait for one to fail or all to finish.
                    # each task is independent from all other concurrent tasks
                    done, inflight = await asyncio.wait(
                        futures,
                        return_when=asyncio.FIRST_EXCEPTION,
                        timeout=self.step_timeout,
                    )

                    # panic on failure or timeout
                    _panic_or_proceed(done, inflight, step)

                    # combine pending writes from all tasks
                    pending_writes = deque[tuple[str, Any]]()
                    for _, _, _, writes in next_tasks:
                        pending_writes.extend(writes)

                    # apply writes to channels
                    _apply_writes(
                        checkpoint, channels, pending_writes, config, step + 1
                    )

                    if debug:
                        print_checkpoint(step, channels)

                    # yield current value or updates
                    if stream_mode == "values":
                        if step_output := map_output_values(
                            output_keys, pending_writes, channels
                        ):
                            yield step_output
                    else:
                        if step_output := map_output_updates(output_keys, next_tasks):
                            yield step_output

                    # with previous step's checkpoint
                    if _should_interrupt(
                        checkpoint,
                        interrupt_before_nodes,
                        self.snapshot_channels_list,
                        pending_writes,
                    ):
                        break

                    # save end of step checkpoint
                    if self.checkpointer is not None and (
                        self.checkpointer.at == CheckpointAt.END_OF_STEP
                        or interrupt_before_nodes
                    ):
                        checkpoint = create_checkpoint(checkpoint, channels)
                        checkpoint_config = await self.checkpointer.aput(
                            checkpoint_config, checkpoint
                        )

                    # with this step's checkpoint
                    if _should_interrupt(
                        checkpoint,
                        interrupt_after_nodes,
                        self.snapshot_channels_list,
                        pending_writes,
                    ):
                        break

                # save end of run checkpoint
                if (
                    self.checkpointer is not None
                    and self.checkpointer.at == CheckpointAt.END_OF_RUN
                    and not interrupt_before_nodes
                ):
                    checkpoint = create_checkpoint(checkpoint, channels)
                    await self.checkpointer.aput(checkpoint_config, checkpoint)
        finally:
            # cancel any pending tasks when generator is interrupted
            try:
                for task in futures:
                    task.cancel()
            except NameError:
                pass

    def invoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before_nodes: Optional[Sequence[str]] = None,
        interrupt_after_nodes: Optional[Sequence[str]] = None,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        output_keys = output_keys if output_keys is not None else self.output_channels
        output_is_dict = not isinstance(output_keys, str)
        latest: Union[dict[str, Any], Any] = {} if output_is_dict else None
        for chunk in self.stream(
            input,
            config,
            stream_mode="values",
            output_keys=output_keys,
            input_keys=input_keys,
            interrupt_before_nodes=interrupt_before_nodes,
            interrupt_after_nodes=interrupt_after_nodes,
            debug=debug,
            **kwargs,
        ):
            latest = {**latest, **chunk} if output_is_dict else chunk
        return latest

    def stream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[StreamMode] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before_nodes: Optional[Sequence[str]] = None,
        interrupt_after_nodes: Optional[Sequence[str]] = None,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[Union[dict[str, Any], Any]]:
        return self.transform(
            iter([input]),
            config,
            stream_mode=stream_mode,
            output_keys=output_keys,
            input_keys=input_keys,
            interrupt_before_nodes=interrupt_before_nodes,
            interrupt_after_nodes=interrupt_after_nodes,
            debug=debug,
            **kwargs,
        )

    def transform(
        self,
        input: Iterator[Union[dict[str, Any], Any]],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[StreamMode] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before_nodes: Optional[Sequence[str]] = None,
        interrupt_after_nodes: Optional[Sequence[str]] = None,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[Union[dict[str, Any], Any]]:
        for chunk in self._transform_stream_with_config(
            input,
            self._transform,
            config,
            stream_mode=stream_mode,
            output_keys=output_keys,
            input_keys=input_keys,
            interrupt_before_nodes=interrupt_before_nodes,
            interrupt_after_nodes=interrupt_after_nodes,
            debug=debug,
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
        interrupt_before_nodes: Optional[Sequence[str]] = None,
        interrupt_after_nodes: Optional[Sequence[str]] = None,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        output_keys = output_keys if output_keys is not None else self.output_channels
        output_is_dict = not isinstance(output_keys, str)
        latest: Union[dict[str, Any], Any] = {} if output_is_dict else None
        async for chunk in self.astream(
            input,
            config,
            stream_mode="values",
            output_keys=output_keys,
            input_keys=input_keys,
            interrupt_before_nodes=interrupt_before_nodes,
            interrupt_after_nodes=interrupt_after_nodes,
            debug=debug,
            **kwargs,
        ):
            latest = {**latest, **chunk} if output_is_dict else chunk
        return latest

    async def astream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[StreamMode] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before_nodes: Optional[Sequence[str]] = None,
        interrupt_after_nodes: Optional[Sequence[str]] = None,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[dict[str, Any], Any]]:
        async def input_stream() -> AsyncIterator[Union[dict[str, Any], Any]]:
            yield input

        async for chunk in self.atransform(
            input_stream(),
            config,
            stream_mode=stream_mode,
            output_keys=output_keys,
            input_keys=input_keys,
            interrupt_before_nodes=interrupt_before_nodes,
            interrupt_after_nodes=interrupt_after_nodes,
            debug=debug,
            **kwargs,
        ):
            yield chunk

    async def atransform(
        self,
        input: AsyncIterator[Union[dict[str, Any], Any]],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[StreamMode] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before_nodes: Optional[Sequence[str]] = None,
        interrupt_after_nodes: Optional[Sequence[str]] = None,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[dict[str, Any], Any]]:
        async for chunk in self._atransform_stream_with_config(
            input,
            self._atransform,
            config,
            stream_mode=stream_mode,
            output_keys=output_keys,
            input_keys=input_keys,
            interrupt_before_nodes=interrupt_before_nodes,
            interrupt_after_nodes=interrupt_after_nodes,
            debug=debug,
            **kwargs,
        ):
            yield chunk


def _panic_or_proceed(
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


def _should_interrupt(
    checkpoint: Checkpoint,
    interrupt_nodes: Sequence[str],
    snapshot_channels: Sequence[str],
    pending_writes: Sequence[tuple[str, Any]],
) -> bool:
    return (
        # interrupt if any of snapshopt_channels has been updated since last interrupt
        any(
            checkpoint["channel_versions"][chan]
            > checkpoint["versions_seen"][INTERRUPT][chan]
            for chan in snapshot_channels
        )
        # and any channel written to is in interrupt_nodes list
        and any(chan for chan, _ in pending_writes if chan in interrupt_nodes)
    )


def _read_channel(
    channels: Mapping[str, BaseChannel],
    chan: str,
    *,
    catch: bool = True,
    return_exception: bool = False,
) -> Any:
    try:
        return channels[chan].get()
    except EmptyChannelError as exc:
        if return_exception:
            return exc
        elif catch:
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
        if chan in AllReservedChannels:
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
            try:
                channels[chan].update(vals)
            except InvalidUpdateError as e:
                raise InvalidUpdateError(
                    f"Invalid update for channel {chan}: {e}"
                ) from e
            checkpoint["channel_versions"][chan] += 1
            updated_channels.add(chan)
        else:
            logger.warning(f"Skipping write for channel {chan} which has no readers")
    # Channels that weren't updated in this step are notified of a new step
    for chan in channels:
        if chan not in updated_channels:
            channels[chan].update([])


@overload
def _prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, ChannelInvoke],
    channels: Mapping[str, BaseChannel],
    for_execution: Literal[False],
) -> tuple[Checkpoint, list[tuple[Runnable, Any, str]]]:
    ...


@overload
def _prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, ChannelInvoke],
    channels: Mapping[str, BaseChannel],
    for_execution: Literal[True],
) -> tuple[Checkpoint, list[tuple[Runnable, Any, str, deque[tuple[str, Any]]]]]:
    ...


def _prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, ChannelInvoke],
    channels: Mapping[str, BaseChannel],
    *,
    for_execution: bool,
) -> tuple[
    Checkpoint,
    Union[
        list[tuple[Runnable, Any, str]],
        list[tuple[Runnable, Any, str, deque[tuple[str, Any]]]],
    ],
]:
    checkpoint = copy_checkpoint(checkpoint)
    tasks: Union[
        list[tuple[Runnable, Any, str]],
        list[tuple[Runnable, Any, str, deque[tuple[str, Any]]]],
    ] = []
    # Check if any processes should be run in next step
    # If so, prepare the values to be passed to them
    for name, proc in processes.items():
        seen = checkpoint["versions_seen"][name]
        # If any of the channels read by this process were updated
        if any(
            checkpoint["channel_versions"][chan] > seen[chan]
            for chan in proc.triggers
            if not isinstance(
                _read_channel(channels, chan, return_exception=True), EmptyChannelError
            )
        ):
            # If all trigger channels subscribed by this process are not empty
            # then invoke the process with the values of all non-empty channels
            try:
                val: Any = {
                    k: _read_channel(channels, chan, catch=chan not in proc.triggers)
                    for k, chan in proc.channels.items()
                }
            except EmptyChannelError:
                continue

            # If the process has a mapper, apply it to the value
            if proc.mapper is not None:
                val = proc.mapper(val)

            # Processes that subscribe to a single keyless channel get
            # the value directly, instead of a dict
            if list(proc.channels.keys()) == [None]:
                val = val[None]

            # update seen versions
            if for_execution:
                seen.update(
                    {
                        chan: checkpoint["channel_versions"][chan]
                        for chan in proc.triggers
                    }
                )

            # skip if condition is not met
            if proc.when is None or proc.when(val):
                if for_execution:
                    tasks.append((proc, val, name, deque()))
                else:
                    tasks.append((proc, val, name))
    return checkpoint, tasks


async def _aconsume(iterator: AsyncIterator[Any]) -> None:
    """Consume an async iterator."""
    async for _ in iterator:
        pass
