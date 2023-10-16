from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections import defaultdict, deque
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    cast,
    overload,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.pydantic_v1 import Field
from langchain.schema.runnable import (
    Runnable,
    RunnableBinding,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain.schema.runnable.base import (
    Other,
    RunnableEach,
    RunnableLike,
    coerce_to_runnable,
)
from langchain.schema.runnable.config import (
    RunnableConfig,
    get_executor_for_config,
    patch_config,
)
from langchain.schema.runnable.utils import ConfigurableFieldSpec, Input, Output

from permchain.channels import Channel, EmptyChannelError, Inbox

logger = logging.getLogger(__name__)


CONFIG_KEY_STEP = "__pregel_step"
CONFIG_KEY_SEND = "__pregel_send"
CONFIG_KEY_READ = "__pregel_read"

TYPE_SEND = Callable[[Sequence[tuple[str, Any]]], None]


class PregelRead(RunnableLambda):
    channel: str

    @property
    def config_specs(self) -> Sequence[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id=CONFIG_KEY_READ,
                name=CONFIG_KEY_READ,
                description=None,
                default=None,
                annotation=Callable[[Channel], Any],
            ),
        ]

    def __init__(self, channel: str) -> None:
        super().__init__(func=self._read, afunc=self._aread)  # type: ignore[arg-type]
        self.channel = channel

    def _read(self, _: Any, config: RunnableConfig) -> Any:
        try:
            read: Callable[[str], Any] = config["configurable"][CONFIG_KEY_READ]
        except KeyError:
            raise RuntimeError(
                f"Runnable {self} is not configured with a read function"
                "Make sure to call in the context of a Pregel process"
            )
        return read(self.channel)

    async def _aread(self, _: Any, config: RunnableConfig) -> Any:
        try:
            read: Callable[[str], Any] = config["configurable"][CONFIG_KEY_READ]
        except KeyError:
            raise RuntimeError(
                f"Runnable {self} is not configured with a read function"
                "Make sure to call in the context of a Pregel process"
            )
        return read(self.channel)


class PregelInvoke(RunnableBinding):
    channels: Mapping[None, str] | Mapping[str, str]

    bound: Runnable[Any, Any] = Field(default_factory=RunnablePassthrough)

    kwargs: Mapping[str, Any] = Field(default_factory=dict)

    def join(self, channels: Sequence[str]) -> PregelInvoke:
        joiner = RunnablePassthrough.assign(
            **{chan: PregelRead(chan) for chan in channels}
        )
        if isinstance(self.bound, RunnablePassthrough):
            return PregelInvoke(channels=self.channels, bound=joiner)
        else:
            return PregelInvoke(channels=self.channels, bound=self.bound | joiner)

    def __or__(
        self,
        other: Runnable[Any, Other]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other]],
    ) -> Runnable:
        if isinstance(self.bound, RunnablePassthrough):
            return PregelInvoke(channels=self.channels, bound=coerce_to_runnable(other))
        else:
            return PregelInvoke(channels=self.channels, bound=self.bound | other)

    def __ror__(
        self,
        other: Runnable[Other, Any]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any]],
    ) -> Runnable:
        raise NotImplementedError()


class PregelBatch(RunnableEach):
    channel: str

    bound: Runnable[Any, Any] = Field(default_factory=RunnablePassthrough)

    def __or__(
        self,
        other: Runnable[Any, Other]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other]],
    ) -> Runnable:
        if isinstance(self.bound, RunnablePassthrough):
            return PregelBatch(channel=self.channel, bound=coerce_to_runnable(other))
        else:
            return PregelBatch(channel=self.channel, bound=self.bound | other)

    def __ror__(
        self,
        other: Runnable[Other, Any]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any]],
    ) -> Runnable:
        raise NotImplementedError()


class PregelSink(RunnableLambda):
    channels: Sequence[tuple[str, Runnable]]
    """
    Mapping of write channels to Runnables that return the value to be written,
    or None to skip writing.
    """

    max_steps: Optional[int]

    def __init__(
        self,
        *,
        channels: Sequence[tuple[str, Runnable]],
        max_steps: Optional[int] = None,
    ):
        super().__init__(func=self._write, afunc=self._awrite)
        self.channels = channels
        self.max_steps = max_steps

    @property
    def config_specs(self) -> Sequence[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id=CONFIG_KEY_STEP,
                name=CONFIG_KEY_STEP,
                description=None,
                default=None,
                annotation=int,
            ),
            ConfigurableFieldSpec(
                id=CONFIG_KEY_SEND,
                name=CONFIG_KEY_SEND,
                description=None,
                default=None,
                annotation=TYPE_SEND,
            ),
        ]

    def _write(self, input: Any, config: RunnableConfig) -> None:
        step: int = config["configurable"][CONFIG_KEY_STEP]

        if self.max_steps is not None and step >= self.max_steps:
            return

        write: TYPE_SEND = config["configurable"][CONFIG_KEY_SEND]

        values = [(chan, r.invoke(input, config)) for chan, r in self.channels]

        write([(chan, val) for chan, val in values if val is not None])

        return input

    async def _awrite(self, input: Any, config: RunnableConfig) -> None:
        step: int = config["configurable"][CONFIG_KEY_STEP]

        if self.max_steps is not None and step >= self.max_steps:
            return

        write: TYPE_SEND = config["configurable"][CONFIG_KEY_SEND]

        values = [(chan, await r.ainvoke(input, config)) for chan, r in self.channels]

        write([(chan, val) for chan, val in values if val is not None])

        return input


class Pregel(Generic[Output], RunnableSerializable[dict[str, Any] | Any, Output]):
    channels: Mapping[str, Channel]

    chains: Sequence[PregelInvoke | PregelBatch]

    output: str | Sequence[str]

    input: str | None

    step_timeout: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        *chains: Sequence[PregelInvoke | PregelBatch] | PregelInvoke | PregelBatch,
        channels: Mapping[str, Channel],
        output: str | Sequence[str],
        input: str | None = None,
        step_timeout: Optional[float] = None,
    ) -> None:
        chains_flat: list[PregelInvoke | PregelBatch] = []
        for chain in chains:
            if isinstance(chain, (list, tuple)):
                chains_flat.extend(chain)
            else:
                chains_flat.append(chain)

        validate_chains_channels(chains_flat, channels, input, output)

        super().__init__(
            chains=chains_flat,
            channels=channels,
            output=output,
            input=input,
            step_timeout=step_timeout,
        )

    @classmethod
    def subscribe_to(cls, channels: str | Sequence[str]) -> PregelInvoke:
        """Runs process.invoke() each time channels are updated,
        with a dict of the channel values as input."""
        return PregelInvoke(
            channels={None: channels}
            if isinstance(channels, str)
            else {chan: chan for chan in channels}
        )

    @classmethod
    def subscribe_to_each(cls, inbox: Inbox) -> PregelBatch:
        """Runs process.batch() with the content of inbox each time it is updated."""
        return PregelBatch(channel=inbox)

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
    ) -> Iterator[Output]:
        processes = tuple(self.chains)
        # TODO this is where we'd restore from checkpoint
        channels = {k: v._empty() for k, v in self.channels.items()}
        next_tasks = _apply_writes_and_prepare_next_tasks(
            processes,
            channels,
            deque((self.input, chunk) for chunk in input)
            if self.input is not None
            else deque((k, v) for chunk in input for k, v in chunk.items()),
        )

        def read(chan: Channel) -> Any:
            try:
                return channels[chan]._get()
            except EmptyChannelError:
                return None

        with get_executor_for_config(config) as executor:
            # Similarly to Bulk Synchronous Parallel / Pregel model
            # computation proceeds in steps, while there are channel updates
            # channel updates from step N are only visible in step N+1
            # channels are guaranteed to be immutable for the duration of the step,
            # with channel updates applied only at the transition between steps
            for step in range(config["recursion_limit"]):
                # collect all writes to channels, without applying them yet
                pending_writes = deque[tuple[Channel, Any]]()

                # execute tasks, and wait for one to fail or all to finish
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
                        for proc, input in next_tasks
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
                if any(chan is self.output for chan, _ in pending_writes):
                    yield channels[self.output]._get()

                # TODO this is where we'd save checkpoint

                # if no more tasks, we're done
                if not next_tasks:
                    break

    async def _atransform(
        self,
        input: AsyncIterator[dict[str, Any] | Any],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> AsyncIterator[Output]:
        processes = tuple(self.chains)
        channels = {k: v._empty() for k, v in self.channels.items()}
        next_tasks = _apply_writes_and_prepare_next_tasks(
            processes,
            channels,
            deque((self.input, chunk) async for chunk in input)
            if self.input is not None
            else deque((k, v) async for chunk in input for k, v in chunk.items()),
        )

        def read(chan: Channel) -> Any:
            try:
                return channels[chan]._get()
            except EmptyChannelError:
                return None

        # Similarly to Bulk Synchronous Parallel / Pregel model
        # computation proceeds in steps, while there are channel updates
        # channel updates from step N are only visible in step N+1,
        # channels are guaranteed to be immutable for the duration of the step,
        # channel updates being applied only at the transition between steps
        for step in range(config["recursion_limit"]):
            # collect all writes to channels, without applying them yet
            pending_writes = deque[tuple[Channel, Any]]()

            # execute tasks, and wait for one to fail or all to finish
            # each task is independent from all other concurrent tasks
            done, inflight = await asyncio.wait(
                [
                    asyncio.create_task(
                        proc.ainvoke(
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
                    )
                    for proc, input in next_tasks
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
                if any(chan is self.output for chan, _ in pending_writes):
                    yield channels[self.output]._get()
            else:
                if updated := {c for c, _ in pending_writes if c in self.output}:
                    yield {chan: channels[chan]._get() for chan in updated}

            # if no more tasks, we're done
            if not next_tasks:
                break

    def invoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        latest: Output | None = None
        for chunk in self.stream(input, config, **kwargs):
            latest = chunk
        return latest

    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        return self.transform(iter([input]), config, **kwargs)

    def transform(
        self,
        input: Iterator[dict[str, Any] | Any],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        return self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def ainvoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        latest: Output | None = None
        async for chunk in self.astream(input, config, **kwargs):
            latest = chunk
        return latest

    async def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        async def input_stream() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_stream(), config, **kwargs):
            yield chunk

    async def atransform(
        self,
        input: AsyncIterator[dict[str, Any] | Any],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk


def _apply_writes_and_prepare_next_tasks(
    processes: Sequence[PregelInvoke | PregelBatch],
    channels: Mapping[str, Channel],
    pending_writes: Sequence[tuple[str, Any]],
) -> list[tuple[Runnable, Any]]:
    pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
    # Group writes by channel
    for chan, val in pending_writes:
        pending_writes_by_channel[chan].append(val)

    updated_channels: set[Channel] = set()
    # Apply writes to channels
    for chan, vals in pending_writes_by_channel.items():
        if chan in channels:
            channels[chan]._update(vals)
            updated_channels.add(chan)
        else:
            logger.warning(f"Skipping write for channel {chan} which has no readers")

    tasks: list[tuple[Runnable, Any]] = []
    # Check if any processes should be run in next step
    # If so, prepare the values to be passed to them
    for proc in processes:
        if isinstance(proc, PregelInvoke):
            # If any of the channels read by this process were updated
            if any(chan in updated_channels for chan in proc.channels.values()):
                # If all channels read by this process have been initialized
                try:
                    val = {
                        k: channels[chan]._get() for k, chan in proc.channels.items()
                    }
                except EmptyChannelError:
                    continue

                # Processes that subscribe to a single keyless channel get
                # the value directly, instead of a dict
                if list(proc.channels.keys()) == [None]:
                    tasks.append((proc, val[None]))
                else:
                    tasks.append((proc, val))
        elif isinstance(proc, PregelBatch):
            # If the channel read by this process was updated
            if proc.channel in updated_channels:
                # Here we don't catch EmptyChannelError because the channel
                # must be intialized if the previous `if` condition is true
                val = channels[proc.channel]._get()

                tasks.append((proc, val))

    return tasks


def validate_chains_channels(
    chains: Sequence[PregelInvoke | PregelBatch],
    channels: Mapping[str, Channel],
    input: str | None,
    output: str | Sequence[str],
) -> None:
    subscribed_channels = set()
    for chain in chains:
        if isinstance(chain, PregelInvoke):
            subscribed_channels.update(chain.channels.values())
        elif isinstance(chain, PregelBatch):
            subscribed_channels.add(chain.channel)
        else:
            raise TypeError(
                f"Invalid chain type {type(chain)}, expected Pregel.subscribe_to() or Pregel.subscribe_to_each()"
            )

    if input is not None and input not in subscribed_channels:
        raise ValueError(f"Input channel {input} is not subscribed to by any chain")

    for chan in subscribed_channels:
        if chan not in channels:
            raise ValueError(f"Channel {chan} is subscribed to, but not initialized")

    if isinstance(output, str):
        if output not in channels:
            raise ValueError(f"Output channel {output} is not initialized")
