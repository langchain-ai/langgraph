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

TYPE_SEND = Callable[[Sequence[tuple[Channel, Any]]], None]


class PregelRead(RunnableLambda):
    channel: Channel

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

    def __init__(self, channel: Channel) -> None:
        super().__init__(func=self._read, afunc=self._aread)  # type: ignore[arg-type]
        self.channel = channel

    def _read(self, _: Any, config: RunnableConfig) -> Any:
        try:
            read: Callable[[Channel], Any] = config["configurable"][CONFIG_KEY_READ]
        except KeyError:
            raise RuntimeError(
                f"Runnable {self} is not configured with a read function"
                "Make sure to call in the context of a Pregel process"
            )
        return read(self.channel)

    async def _aread(self, _: Any, config: RunnableConfig) -> Any:
        try:
            read: Callable[[Channel], Any] = config["configurable"][CONFIG_KEY_READ]
        except KeyError:
            raise RuntimeError(
                f"Runnable {self} is not configured with a read function"
                "Make sure to call in the context of a Pregel process"
            )
        return read(self.channel)


class PregelInvoke(RunnableBinding):
    channels: Mapping[None, Channel] | Mapping[str, Channel]

    bound: Runnable[Any, Any] = Field(default_factory=RunnablePassthrough)

    kwargs: Mapping[str, Any] = Field(default_factory=dict)

    def join(self, **channels: Channel) -> PregelInvoke:
        joiner = RunnablePassthrough.assign(
            **{k: PregelRead(chan) for k, chan in channels.items()}
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
    channel: Inbox

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
    channels: Sequence[tuple[Channel, Runnable]]
    """
    Mapping of write channels to Runnables that return the value to be written,
    or None to skip writing.
    """

    max_steps: Optional[int]

    def __init__(
        self,
        *,
        channels: Sequence[tuple[Channel, Runnable]],
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


class Pregel(Generic[Input, Output], RunnableSerializable[Input, Output]):
    input: Channel[Any, Input]

    output: Channel[Output, Any]

    processes: Sequence[PregelInvoke | PregelBatch]

    step_timeout: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        *processes: PregelInvoke | PregelBatch,
        input: Channel[Input, Any],
        output: Channel[Output, Any],
        step_timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(
            processes=processes,
            input=input,
            output=output,
            step_timeout=step_timeout,
            **kwargs,
        )

    @overload
    @classmethod
    def subscribe_to(cls, __channel: Channel) -> PregelInvoke:
        ...

    @overload
    @classmethod
    def subscribe_to(
        cls, __channel: Mapping[str, Channel] | None = None, **kwargs: Channel
    ) -> PregelInvoke:
        ...

    @classmethod
    def subscribe_to(
        cls, __channel: Channel | Mapping[str, Channel] | None = None, **kwargs: Channel
    ) -> PregelInvoke:
        """Runs process.invoke() each time channels are updated."""
        __channel = __channel or {}
        return (
            PregelInvoke(channels={None: __channel})
            if isinstance(__channel, Channel)
            else PregelInvoke(channels={**__channel, **kwargs})
        )

    @classmethod
    def subscribe_to_each(cls, inbox: Inbox) -> PregelBatch:
        """Runs process.batch() with the content of inbox each time it is updated."""
        return PregelBatch(channel=inbox)

    @classmethod
    def send_to(
        cls,
        channels: Channel | Mapping[Channel, RunnableLike],
        *,
        max_steps: Optional[int] = None,
    ) -> PregelSink:
        """Writes to channels the result of the lambda, or None to skip writing."""
        return PregelSink(
            channels=(
                [(channels, RunnablePassthrough())]
                if isinstance(channels, Channel)
                else [(k, coerce_to_runnable(v)) for k, v in channels.items()]
            ),
            max_steps=max_steps,
        )

    def _prepare_channels(self) -> Mapping[Channel, Channel]:
        channels: dict[Channel, Channel] = {self.output: self.output._empty()}
        for proc in self.processes:
            if isinstance(proc, PregelInvoke):
                for chan in proc.channels.values():
                    if chan not in channels:
                        channels[chan] = chan._empty()
            elif isinstance(proc, PregelBatch):
                if proc.channel not in channels:
                    channels[proc.channel] = proc.channel._empty()
            else:
                raise TypeError(
                    f"Received process {proc}, expected instance of PregelInvoke or PregelBatch"
                )

        if not channels:
            raise ValueError("Found 0 channels for Pregel run")

        if self.input not in channels:
            raise ValueError("Input channel not being read from")

        return channels

    def _transform(
        self,
        input: Iterator[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Iterator[Output]:
        processes = tuple(self.processes)
        # TODO this is where we'd restore from checkpoint
        channels = self._prepare_channels()
        next_tasks = _apply_writes_and_prepare_next_tasks(
            processes, channels, deque((self.input, chunk) for chunk in input)
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
        input: AsyncIterator[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> AsyncIterator[Output]:
        processes = tuple(self.processes)
        channels = self._prepare_channels()
        next_tasks = _apply_writes_and_prepare_next_tasks(
            processes, channels, [(self.input, chunk) async for chunk in input]
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
            if any(chan is self.output for chan, _ in pending_writes):
                yield channels[self.output]._get()

            # if no more tasks, we're done
            if not next_tasks:
                break

    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        latest: Output | None = None
        for chunk in self.stream(input, config, **kwargs):
            latest = chunk
        return latest

    def stream(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[Output]:
        return self.transform(iter([input]), config, **kwargs)

    def transform(
        self,
        input: Iterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        return self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        latest: Output | None = None
        async for chunk in self.astream(input, config, **kwargs):
            latest = chunk
        return latest

    async def astream(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Output]:
        async def input_stream() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_stream(), config, **kwargs):
            yield chunk

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk


def _apply_writes_and_prepare_next_tasks(
    processes: Sequence[PregelInvoke | PregelBatch],
    channels: Mapping[Channel, Channel],
    pending_writes: Sequence[tuple[Channel, Any]],
) -> list[tuple[Runnable, Any]]:
    pending_writes_by_channel: dict[Channel, list[Any]] = defaultdict(list)
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
