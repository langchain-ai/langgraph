from __future__ import annotations

import asyncio
import concurrent.futures
import enum
import logging
from collections import defaultdict, deque
from typing import Any, AsyncIterator, Iterator, Mapping, Optional, Sequence, Type, cast

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.globals import get_debug
from langchain.pydantic_v1 import BaseModel, create_model
from langchain.schema.runnable import (
    Runnable,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain.schema.runnable.base import RunnableLike, coerce_to_runnable
from langchain.schema.runnable.config import (
    RunnableConfig,
    get_executor_for_config,
    patch_config,
)

from permchain.channels.base import (
    AsyncChannelsManager,
    Channel,
    ChannelsManager,
    EmptyChannelError,
)
from permchain.channels.last_value import LastValue
from permchain.pregel.constants import (
    CHAINS_MAIN,
    CONFIG_KEY_READ,
    CONFIG_KEY_SEND,
    CONFIG_KEY_STEP,
)
from permchain.pregel.debug import print_step_start
from permchain.pregel.read import PregelBatch, PregelInvoke
from permchain.pregel.validate import validate_chains_channels
from permchain.pregel.write import PregelSink

logger = logging.getLogger(__name__)


# Before Python 3.11 native StrEnum is not available
class StrEnum(str, enum.Enum):
    """A string enum."""

    pass


class PregelIO(StrEnum):
    """Pregel IO channels."""

    IN = "__pregel_input"
    OUT = "__pregel_output"


class Pregel(RunnableSerializable[dict[str, Any] | Any, dict[str, Any] | Any]):
    channels: Mapping[str, Channel]

    chains: Mapping[str, PregelInvoke | PregelBatch]

    output: str | Sequence[str]

    input: str | Sequence[str]

    step_timeout: Optional[float] = None

    debug: bool

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        main: Optional[Runnable] = None,
        *,
        chains: Mapping[str, PregelInvoke | PregelBatch],
        channels: Mapping[str, Channel],
        output: str | Sequence[str] = PregelIO.OUT,
        input: str | Sequence[str] = PregelIO.IN,
        step_timeout: Optional[float] = None,
        debug: Optional[bool] = None,
    ) -> None:
        chains = {**chains}
        channels = {**channels}

        if main is not None:
            chains[CHAINS_MAIN] = (
                Pregel.subscribe_to(PregelIO.IN) | main | Pregel.send_to(PregelIO.OUT)
            )
        elif output is PregelIO.OUT:
            raise ValueError(
                f"When no main runnable is provided, output must be one or more of the channels in {channels.keys()}"
            )

        if input is PregelIO.IN:
            channels[PregelIO.IN] = LastValue(
                main.input_schema if main is not None else Any  # type: ignore[arg-type]
            )
        if output is PregelIO.OUT:
            channels[PregelIO.OUT] = LastValue(
                main.output_schema if main is not None else Any  # type: ignore[arg-type]
            )

        validate_chains_channels(chains, channels, input, output)

        super().__init__(
            chains=chains,
            channels=channels,
            output=output,
            input=input,
            step_timeout=step_timeout,
            debug=debug if debug is not None else get_debug(),
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
                "PregelInput",
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
                "PregelOutput",
                **{k: (self.channels[k].ValueType, None) for k in self.output},
            )

    @classmethod
    def subscribe_to(cls, channels: str | Sequence[str]) -> PregelInvoke:
        """Runs process.invoke() each time channels are updated,
        with a dict of the channel values as input."""
        return PregelInvoke(
            channels=cast(
                Mapping[None, str] | Mapping[str, str],
                {None: channels}
                if isinstance(channels, str)
                else {chan: chan for chan in channels},
            )
        )

    @classmethod
    def subscribe_to_each(cls, inbox: str, key: Optional[str] = None) -> PregelBatch:
        """Runs process.batch() with the content of inbox each time it is updated."""
        return PregelBatch(channel=inbox, key=key)

    @classmethod
    def send_to(
        cls,
        *channels: str,
        _max_steps: Optional[int] = None,
        **kwargs: RunnableLike,
    ) -> PregelSink:
        """Writes to channels the result of the lambda, or None to skip writing."""
        return PregelSink(
            channels=(
                [(c, RunnablePassthrough()) for c in channels]
                + [(k, coerce_to_runnable(v)) for k, v in kwargs.items()]
            ),
            max_steps=_max_steps,
        )

    def _transform(
        self,
        input: Iterator[dict[str, Any] | Any],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Iterator[dict[str, Any] | Any]:
        processes = {**self.chains}
        # TODO this is where we'd restore from checkpoint
        with ChannelsManager(self.channels) as channels, get_executor_for_config(
            config
        ) as executor:
            next_tasks = _apply_writes_and_prepare_next_tasks(
                processes,
                channels,
                deque((self.input, chunk) for chunk in input)
                if isinstance(self.input, str)
                else deque(
                    (k, v)
                    for chunk in input
                    for k, v in chunk.items()
                    if k in self.input
                ),
            )

            def read(chan: str) -> Any:
                try:
                    return channels[chan].get()
                except EmptyChannelError:
                    return None

            # Similarly to Bulk Synchronous Parallel / Pregel model
            # computation proceeds in steps, while there are channel updates
            # channel updates from step N are only visible in step N+1
            # channels are guaranteed to be immutable for the duration of the step,
            # with channel updates applied only at the transition between steps
            for step in range(config["recursion_limit"]):
                if self.debug:
                    print_step_start(step, next_tasks)

                # collect all writes to channels, without applying them yet
                pending_writes = deque[tuple[str, Any]]()

                # execute tasks, and wait for one to fail or all to finish.
                # each task is independent from all other concurrent tasks
                done, inflight = concurrent.futures.wait(
                    (
                        executor.submit(
                            proc.invoke,
                            input,
                            patch_config(
                                config,
                                callbacks=run_manager.get_child(f"pregel:step:{step}"),
                                configurable={
                                    # deque.extend is thread-safe
                                    CONFIG_KEY_SEND: pending_writes.extend,
                                    CONFIG_KEY_READ: read,
                                    CONFIG_KEY_STEP: step,
                                },
                            ),
                        )
                        for proc, input, _ in next_tasks
                    ),
                    return_when=concurrent.futures.FIRST_EXCEPTION,
                    timeout=self.step_timeout,
                )

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

                # apply writes to channels, decide on next step
                next_tasks = _apply_writes_and_prepare_next_tasks(
                    processes, channels, pending_writes
                )

                # if any write to output channel in this step, yield current value
                if isinstance(self.output, str):
                    if any(chan == self.output for chan, _ in pending_writes):
                        yield channels[self.output].get()
                else:
                    if updated := {c for c, _ in pending_writes if c in self.output}:
                        yield {chan: channels[chan].get() for chan in updated}

                # TODO this is where we'd save checkpoint

                # if no more tasks, we're done
                if not next_tasks:
                    break

    async def _atransform(
        self,
        input: AsyncIterator[dict[str, Any] | Any],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        processes = {**self.chains}
        # TODO this is where we'd restore from checkpoint
        async with AsyncChannelsManager(self.channels) as channels:
            next_tasks = _apply_writes_and_prepare_next_tasks(
                processes,
                channels,
                deque([(self.input, chunk) async for chunk in input])
                if isinstance(self.input, str)
                else deque(
                    [
                        (k, v)
                        async for chunk in input
                        for k, v in chunk.items()
                        if k in self.input
                    ]
                ),
            )

            def read(chan: str) -> Any:
                try:
                    return channels[chan].get()
                except EmptyChannelError:
                    return None

            # Similarly to Bulk Synchronous Parallel / Pregel model
            # computation proceeds in steps, while there are channel updates
            # channel updates from step N are only visible in step N+1,
            # channels are guaranteed to be immutable for the duration of the step,
            # channel updates being applied only at the transition between steps
            for step in range(config["recursion_limit"]):
                if self.debug:
                    print_step_start(step, next_tasks)

                # collect all writes to channels, without applying them yet
                pending_writes = deque[tuple[str, Any]]()

                # execute tasks, and wait for one to fail or all to finish.
                # each task is independent from all other concurrent tasks
                done, inflight = await asyncio.wait(
                    [
                        asyncio.create_task(
                            proc.ainvoke(
                                input,
                                patch_config(
                                    config,
                                    callbacks=run_manager.get_child(
                                        f"pregel:step:{step}"
                                    ),
                                    configurable={
                                        # deque.extend is thread-safe
                                        CONFIG_KEY_SEND: pending_writes.extend,
                                        CONFIG_KEY_READ: read,
                                        CONFIG_KEY_STEP: step,
                                    },
                                ),
                            )
                        )
                        for proc, input, _ in next_tasks
                    ],
                    return_when=asyncio.FIRST_EXCEPTION,
                    timeout=self.step_timeout,
                )

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

                # apply writes to channels, decide on next step
                next_tasks = _apply_writes_and_prepare_next_tasks(
                    processes, channels, pending_writes
                )

                # if any write to output channel in this step, yield current value
                if isinstance(self.output, str):
                    if any(chan == self.output for chan, _ in pending_writes):
                        yield channels[self.output].get()
                else:
                    if updated := {c for c, _ in pending_writes if c in self.output}:
                        yield {chan: channels[chan].get() for chan in updated}

                # if no more tasks, we're done
                if not next_tasks:
                    break

    def invoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        latest: dict[str, Any] | Any = None
        for chunk in self.stream(input, config, **kwargs):
            latest = chunk
        return latest

    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any] | Any]:
        return self.transform(iter([input]), config, **kwargs)

    def transform(
        self,
        input: Iterator[dict[str, Any] | Any],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[dict[str, Any] | Any]:
        return self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def ainvoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        latest: dict[str, Any] | Any = None
        async for chunk in self.astream(input, config, **kwargs):
            latest = chunk
        return latest

    async def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        async def input_stream() -> AsyncIterator[dict[str, Any] | Any]:
            yield input

        async for chunk in self.atransform(input_stream(), config, **kwargs):
            yield chunk

    async def atransform(
        self,
        input: AsyncIterator[dict[str, Any] | Any],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk


def _apply_writes_and_prepare_next_tasks(
    processes: Mapping[str, PregelInvoke | PregelBatch],
    channels: Mapping[str, Channel],
    pending_writes: Sequence[tuple[str, Any]],
) -> list[tuple[Runnable, Any, str]]:
    pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
    # Group writes by channel
    for chan, val in pending_writes:
        pending_writes_by_channel[chan].append(val)

    updated_channels: set[str] = set()
    # Apply writes to channels
    for chan, vals in pending_writes_by_channel.items():
        if chan in channels:
            channels[chan].update(vals)
            updated_channels.add(chan)
        elif chan != PregelIO.OUT:
            logger.warning(f"Skipping write for channel {chan} which has no readers")

    tasks: list[tuple[Runnable, Any, str]] = []
    # Check if any processes should be run in next step
    # If so, prepare the values to be passed to them
    for name, proc in processes.items():
        if isinstance(proc, PregelInvoke):
            # If any of the channels read by this process were updated
            if any(chan in updated_channels for chan in proc.channels.values()):
                # If all channels read by this process have been initialized
                try:
                    val = {k: channels[chan].get() for k, chan in proc.channels.items()}
                except EmptyChannelError:
                    continue

                # Processes that subscribe to a single keyless channel get
                # the value directly, instead of a dict
                if list(proc.channels.keys()) == [None]:
                    tasks.append((proc, val[None], name))
                else:
                    tasks.append((proc, val, name))
        elif isinstance(proc, PregelBatch):
            # If the channel read by this process was updated
            if proc.channel in updated_channels:
                # Here we don't catch EmptyChannelError because the channel
                # must be intialized if the previous `if` condition is true
                val = channels[proc.channel].get()
                if proc.key is not None:
                    val = [{proc.key: v} for v in val]

                tasks.append((proc, val, name))

    return tasks
