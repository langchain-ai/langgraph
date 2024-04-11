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
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    overload,
)

from langchain_core.globals import get_debug
from langchain_core.load.dump import dumpd
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import (
    Runnable,
    RunnableSequence,
    RunnableSerializable,
)
from langchain_core.runnables.base import Input, Output, coerce_to_runnable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_executor_for_config,
    merge_configs,
    patch_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    create_model,
    get_unique_config_specs,
)
from langchain_core.tracers.log_stream import LogStreamCallbackHandler
from typing_extensions import Self

from langgraph.channels.base import (
    AsyncChannelsManager,
    BaseChannel,
    ChannelsManager,
    EmptyChannelError,
    InvalidUpdateError,
    create_checkpoint,
)
from langgraph.channels.last_value import LastValue
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointAt,
    copy_checkpoint,
    empty_checkpoint,
)
from langgraph.constants import (
    CONFIG_KEY_READ,
    CONFIG_KEY_SEND,
    INTERRUPT,
)
from langgraph.pregel.debug import print_checkpoint, print_step_start
from langgraph.pregel.io import (
    map_input,
    map_output_updates,
    map_output_values,
    read_channel,
    read_channels,
)
from langgraph.pregel.log import logger
from langgraph.pregel.read import PregelNode
from langgraph.pregel.types import (
    PregelExecutableTask,
    PregelTaskDescription,
    StateSnapshot,
)
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
        tags: Optional[list[str]] = None,
    ) -> PregelNode:
        ...

    @overload
    @classmethod
    def subscribe_to(
        cls,
        channels: Sequence[str],
        *,
        key: None = None,
        tags: Optional[list[str]] = None,
    ) -> PregelNode:
        ...

    @classmethod
    def subscribe_to(
        cls,
        channels: Union[str, Sequence[str]],
        *,
        key: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> PregelNode:
        """Runs process.invoke() each time channels are updated,
        with a dict of the channel values as input."""
        if not isinstance(channels, str) and key is not None:
            raise ValueError(
                "Can't specify a key when subscribing to multiple channels"
            )
        return PregelNode(
            channels=cast(
                Union[Mapping[None, str], Mapping[str, str]],
                {key: channels}
                if isinstance(channels, str) and key is not None
                else [channels]
                if isinstance(channels, str)
                else {chan: chan for chan in channels},
            ),
            triggers=[channels] if isinstance(channels, str) else channels,
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
            [ChannelWriteEntry(c) for c in channels]
            + [
                ChannelWriteEntry(k, _coerce_write_value(v), True)
                for k, v in kwargs.items()
            ]
        )


StreamMode = Literal["values", "updates"]


class Pregel(
    RunnableSerializable[Union[dict[str, Any], Any], Union[dict[str, Any], Any]]
):
    nodes: Mapping[str, PregelNode]

    channels: Mapping[str, BaseChannel] = Field(default_factory=dict)

    default_channel_cls: Type[BaseChannel] = Field(default=LastValue)

    auto_validate: bool = True

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
    def validate_on_init(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values["auto_validate"]:
            return values
        validate_graph(
            values["nodes"],
            values["channels"],
            values["input_channels"],
            values["output_channels"],
            values["stream_channels"],
            values["interrupt_after_nodes"],
            values["interrupt_before_nodes"],
            values["default_channel_cls"],
        )
        if values["interrupt_after_nodes"] or values["interrupt_before_nodes"]:
            if not values["checkpointer"]:
                raise ValueError("Interrupts require a checkpointer")
        return values

    def validate(self) -> Self:
        validate_graph(
            self.nodes,
            self.channels,
            self.input_channels,
            self.output_channels,
            self.stream_channels,
            self.interrupt_after_nodes,
            self.interrupt_before_nodes,
            self.default_channel_cls,
        )
        if self.interrupt_after_nodes or self.interrupt_before_nodes:
            if not self.checkpointer:
                raise ValueError("Interrupts require a checkpointer")
        return self

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
    def stream_channels_list(self) -> Sequence[str]:
        return (
            [self.stream_channels]
            if isinstance(self.stream_channels, str)
            else self.stream_channels or [k for k in self.channels]
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
            values = read_channels(channels, self.stream_channels_list)
            return StateSnapshot(
                values[self.stream_channels]
                if isinstance(self.stream_channels, str)
                else values,
                tuple(name for name, _ in next_tasks),
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
            values = read_channels(channels, self.stream_channels_list)
            return StateSnapshot(
                values[self.stream_channels]
                if isinstance(self.stream_channels, str)
                else values,
                tuple(name for name, _ in next_tasks),
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
                values = read_channels(channels, self.stream_channels_list)
                yield StateSnapshot(
                    values[self.stream_channels]
                    if isinstance(self.stream_channels, str)
                    else values,
                    tuple(name for name, _ in next_tasks),
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
                values = read_channels(channels, self.stream_channels_list)
                yield StateSnapshot(
                    values[self.stream_channels]
                    if isinstance(self.stream_channels, str)
                    else values,
                    tuple(name for name, _ in next_tasks),
                    config,
                    parent_config,
                )

    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any,
        as_node: Optional[str] = None,
    ) -> RunnableConfig:
        """Update the state of the graph with the given values, as if they came from
        node `as_node`. If `as_node` is not provided, it will be set to the last node
        that updated the state, if not ambiguous.
        """
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        # get last checkpoint
        checkpoint = self.checkpointer.get(config)
        checkpoint = copy_checkpoint(checkpoint) if checkpoint else empty_checkpoint()
        # find last node that updated the state, if not provided
        if as_node is None:
            last_seen_by_node = sorted(
                (v, n)
                for n, seen in checkpoint["versions_seen"].items()
                for v in seen.values()
            )
            # if two nodes updated the state at the same time, it's ambiguous
            if last_seen_by_node:
                if len(last_seen_by_node) == 1:
                    as_node = last_seen_by_node[0][1]
                elif last_seen_by_node[-1][0] != last_seen_by_node[-2][0]:
                    as_node = last_seen_by_node[-1][1]
        if as_node is None:
            raise InvalidUpdateError("Ambiguous update, specify as_node")
        # update channels
        with ChannelsManager(self.channels, checkpoint) as channels:
            # create task to run all writers of the chosen node
            writers = self.nodes[as_node].get_writers()
            if not writers:
                raise InvalidUpdateError(f"Node {as_node} has no writers")
            task = PregelExecutableTask(
                as_node,
                values,
                RunnableSequence(*writers) if len(writers) > 1 else writers[0],
                deque(),
            )
            # execute task
            task.proc.invoke(
                task.input,
                patch_config(
                    config,
                    run_name=self.name + "UpdateState",
                    configurable={
                        # deque.extend is thread-safe
                        CONFIG_KEY_SEND: task.writes.extend,
                        CONFIG_KEY_READ: partial(
                            _local_read, checkpoint, channels, task.writes
                        ),
                    },
                ),
            )
            # apply to checkpoint and save
            _apply_writes(checkpoint, channels, task.writes)
            return self.checkpointer.put(
                config, create_checkpoint(checkpoint, channels)
            )

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any,
        as_node: Optional[str] = None,
    ) -> RunnableConfig:
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        # get last checkpoint
        checkpoint = await self.checkpointer.aget(config)
        checkpoint = copy_checkpoint(checkpoint) if checkpoint else empty_checkpoint()
        # find last node that updated the state, if not provided
        if as_node is None:
            last_seen_by_node = sorted(
                (v, n)
                for n, seen in checkpoint["versions_seen"].items()
                for v in seen.values()
            )
            # if two nodes updated the state at the same time, it's ambiguous
            if last_seen_by_node:
                if len(last_seen_by_node) == 1:
                    as_node = last_seen_by_node[0][1]
                elif last_seen_by_node[-1][0] != last_seen_by_node[-2][0]:
                    as_node = last_seen_by_node[-1][1]
        if as_node is None:
            raise InvalidUpdateError("Ambiguous update, specify as_node")
        # update channels, acting as the chosen node
        async with AsyncChannelsManager(self.channels, checkpoint) as channels:
            # create task to run all writers of the chosen node
            writers = self.nodes[as_node].get_writers()
            if not writers:
                raise InvalidUpdateError(f"Node {as_node} has no writers")
            task = PregelExecutableTask(
                as_node,
                values,
                RunnableSequence(*writers) if len(writers) > 1 else writers[0],
                deque(),
            )
            # execute task
            await task.proc.ainvoke(
                task.input,
                patch_config(
                    config,
                    run_name=self.name + "UpdateState",
                    configurable={
                        # deque.extend is thread-safe
                        CONFIG_KEY_SEND: task.writes.extend,
                        CONFIG_KEY_READ: partial(
                            _local_read, checkpoint, channels, task.writes
                        ),
                    },
                ),
            )
            # apply to checkpoint and save
            _apply_writes(checkpoint, channels, task.writes)
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
    ) -> Iterator[Union[dict[str, Any], Any]]:
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name", self.get_name())
        )
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
            ) = self._defaults(
                stream_mode=stream_mode,
                input_keys=input_keys,
                output_keys=output_keys,
                interrupt_before_nodes=interrupt_before_nodes,
                interrupt_after_nodes=interrupt_after_nodes,
                debug=debug,
            )
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
                if input_writes := deque(map_input(input_keys, input)):
                    # discard any unfinished tasks from previous checkpoint
                    checkpoint, _ = _prepare_next_tasks(
                        checkpoint, processes, channels, for_execution=True
                    )
                    # apply input writes
                    _apply_writes(checkpoint, channels, input_writes)
                else:
                    # if received no input, take that as signal to proceed
                    # past previous interrupt, if any
                    checkpoint = copy_checkpoint(checkpoint)
                    for k in self.stream_channels_list:
                        version = checkpoint["channel_versions"][k]
                        checkpoint["versions_seen"][INTERRUPT][k] = version

                # Similarly to Bulk Synchronous Parallel / Pregel model
                # computation proceeds in steps, while there are channel updates
                # channel updates from step N are only visible in step N+1
                # channels are guaranteed to be immutable for the duration of the step,
                # with channel updates applied only at the transition between steps
                for step in range(config["recursion_limit"] + 1):
                    next_checkpoint, next_tasks = _prepare_next_tasks(
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

                    # before execution, check if we should interrupt
                    if _should_interrupt(
                        checkpoint,
                        interrupt_before_nodes,
                        self.stream_channels_list,
                        next_tasks,
                    ):
                        break
                    else:
                        checkpoint = next_checkpoint

                    if debug:
                        print_step_start(step, next_tasks)

                    # prepare tasks with config
                    tasks_w_config = [
                        (
                            proc,
                            input,
                            patch_config(
                                merge_configs(config, proc_config),
                                run_name=name,
                                callbacks=run_manager.get_child(f"graph:step:{step}"),
                                configurable={
                                    # deque.extend is thread-safe
                                    CONFIG_KEY_SEND: writes.extend,
                                    CONFIG_KEY_READ: partial(
                                        _local_read, checkpoint, channels, writes
                                    ),
                                },
                            ),
                        )
                        for name, input, proc, writes, proc_config in next_tasks
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
                    for _, _, _, writes, _ in next_tasks:
                        pending_writes.extend(writes)

                    # apply writes to channels
                    _apply_writes(checkpoint, channels, pending_writes)

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

                    # save end of step checkpoint
                    if self.checkpointer is not None and (
                        self.checkpointer.at == CheckpointAt.END_OF_STEP
                    ):
                        checkpoint = create_checkpoint(checkpoint, channels)
                        checkpoint_config = self.checkpointer.put(
                            checkpoint_config, checkpoint
                        )

                    # after execution, check if we should interrupt
                    if _should_interrupt(
                        checkpoint,
                        interrupt_after_nodes,
                        self.stream_channels_list,
                        next_tasks,
                    ):
                        break

                # set final channel values as run output
                run_manager.on_chain_end(read_channels(channels, output_keys))

                # save end of run checkpoint
                if (
                    self.checkpointer is not None
                    and self.checkpointer.at == CheckpointAt.END_OF_RUN
                ):
                    checkpoint = create_checkpoint(checkpoint, channels)
                    self.checkpointer.put(checkpoint_config, checkpoint)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        finally:
            # cancel any pending tasks when generator is interrupted
            try:
                for task in futures:
                    task.cancel()
            except NameError:
                pass

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
    ) -> AsyncIterator[Union[dict[str, Any], Any]]:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name", self.get_name())
        )
        # if running from astream_log() run each proc with streaming
        do_stream = next(
            (
                h
                for h in run_manager.handlers
                if isinstance(h, LogStreamCallbackHandler)
            ),
            None,
        )
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
            ) = self._defaults(
                stream_mode=stream_mode,
                input_keys=input_keys,
                output_keys=output_keys,
                interrupt_before_nodes=interrupt_before_nodes,
                interrupt_after_nodes=interrupt_after_nodes,
                debug=debug,
            )
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
                if input_writes := deque(map_input(input_keys, input)):
                    # discard any unfinished tasks from previous checkpoint
                    checkpoint, _ = _prepare_next_tasks(
                        checkpoint, processes, channels, for_execution=True
                    )
                    # apply input writes
                    _apply_writes(checkpoint, channels, input_writes)
                else:
                    # if received no input, take that as signal to proceed
                    # past previous interrupt, if any
                    checkpoint = copy_checkpoint(checkpoint)
                    for k in self.stream_channels_list:
                        version = checkpoint["channel_versions"][k]
                        checkpoint["versions_seen"][INTERRUPT][k] = version

                # Similarly to Bulk Synchronous Parallel / Pregel model
                # computation proceeds in steps, while there are channel updates
                # channel updates from step N are only visible in step N+1,
                # channels are guaranteed to be immutable for the duration of the step,
                # channel updates being applied only at the transition between steps
                for step in range(config["recursion_limit"] + 1):
                    next_checkpoint, next_tasks = _prepare_next_tasks(
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
                            "without hitting a stop condition. You can increase the limit"
                            "by setting the `recursion_limit` config key."
                        )

                    # before execution, check if we should interrupt
                    if _should_interrupt(
                        checkpoint,
                        interrupt_before_nodes,
                        self.stream_channels_list,
                        next_tasks,
                    ):
                        break
                    else:
                        checkpoint = next_checkpoint

                    if debug:
                        print_step_start(step, next_tasks)

                    # prepare tasks with config
                    tasks_w_config = [
                        (
                            proc,
                            input,
                            patch_config(
                                merge_configs(config, proc_config),
                                run_name=name,
                                callbacks=run_manager.get_child(f"graph:step:{step}"),
                                configurable={
                                    # deque.extend is thread-safe
                                    CONFIG_KEY_SEND: writes.extend,
                                    CONFIG_KEY_READ: partial(
                                        _local_read, checkpoint, channels, writes
                                    ),
                                },
                            ),
                        )
                        for name, input, proc, writes, proc_config in next_tasks
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
                    for _, _, _, writes, _ in next_tasks:
                        pending_writes.extend(writes)

                    # apply writes to channels
                    _apply_writes(checkpoint, channels, pending_writes)

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

                    # save end of step checkpoint
                    if self.checkpointer is not None and (
                        self.checkpointer.at == CheckpointAt.END_OF_STEP
                    ):
                        checkpoint = create_checkpoint(checkpoint, channels)
                        checkpoint_config = await self.checkpointer.aput(
                            checkpoint_config, checkpoint
                        )

                    # after execution, check if we should interrupt
                    if _should_interrupt(
                        checkpoint,
                        interrupt_after_nodes,
                        self.stream_channels_list,
                        next_tasks,
                    ):
                        break

                # set final channel values as run output
                await run_manager.on_chain_end(read_channels(channels, output_keys))

                # save end of run checkpoint
                if (
                    self.checkpointer is not None
                    and self.checkpointer.at == CheckpointAt.END_OF_RUN
                ):
                    checkpoint = create_checkpoint(checkpoint, channels)
                    await self.checkpointer.aput(checkpoint_config, checkpoint)
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
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
        stream_mode: StreamMode = "values",
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before_nodes: Optional[Sequence[str]] = None,
        interrupt_after_nodes: Optional[Sequence[str]] = None,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        output_keys = output_keys if output_keys is not None else self.output_channels
        output_is_dict = not isinstance(output_keys, str)
        if stream_mode == "values":
            latest: Union[dict[str, Any], Any] = {} if output_is_dict else None
        else:
            chunks = []
        for chunk in self.stream(
            input,
            config,
            stream_mode=stream_mode,
            output_keys=output_keys,
            input_keys=input_keys,
            interrupt_before_nodes=interrupt_before_nodes,
            interrupt_after_nodes=interrupt_after_nodes,
            debug=debug,
            **kwargs,
        ):
            if stream_mode == "values":
                latest = {**latest, **chunk} if output_is_dict else chunk
            else:
                chunks.append(chunk)
        if stream_mode == "values":
            return latest
        else:
            return chunks

    async def ainvoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: StreamMode = "values",
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before_nodes: Optional[Sequence[str]] = None,
        interrupt_after_nodes: Optional[Sequence[str]] = None,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        output_keys = output_keys if output_keys is not None else self.output_channels
        output_is_dict = not isinstance(output_keys, str)
        if stream_mode == "values":
            latest: Union[dict[str, Any], Any] = {} if output_is_dict else None
        else:
            chunks = []
        async for chunk in self.astream(
            input,
            config,
            stream_mode=stream_mode,
            output_keys=output_keys,
            input_keys=input_keys,
            interrupt_before_nodes=interrupt_before_nodes,
            interrupt_after_nodes=interrupt_after_nodes,
            debug=debug,
            **kwargs,
        ):
            if stream_mode == "values":
                latest = {**latest, **chunk} if output_is_dict else chunk
            else:
                chunks.append(chunk)
        if stream_mode == "values":
            return latest
        else:
            return chunks


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
    tasks: list[PregelExecutableTask],
) -> bool:
    # defaultdicts are mutated on access :( so we need to copy
    seen = checkpoint["versions_seen"].copy()[INTERRUPT].copy()
    return (
        # interrupt if any of snapshopt_channels has been updated since last interrupt
        any(
            checkpoint["channel_versions"][chan] > seen[chan]
            for chan in snapshot_channels
        )
        # and any channel written to is in interrupt_nodes list
        and any(node for node, _, _, _, _ in tasks if node in interrupt_nodes)
    )


def _local_read(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    writes: Sequence[tuple[str, Any]],
    select: Union[list[str], str],
    fresh: bool = False,
) -> Union[dict[str, Any], Any]:
    if fresh:
        checkpoint = create_checkpoint(checkpoint, channels)
        with ChannelsManager(channels, checkpoint) as channels:
            _apply_writes(copy_checkpoint(checkpoint), channels, writes)
            return read_channels(channels, select)
    else:
        return read_channels(channels, select)


def _apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    pending_writes: Sequence[tuple[str, Any]],
) -> None:
    pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
    # Group writes by channel
    for chan, val in pending_writes:
        pending_writes_by_channel[chan].append(val)

    # Find the highest version of all channels
    if checkpoint["channel_versions"]:
        max_version = max(checkpoint["channel_versions"].values())
    else:
        max_version = 0

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
            checkpoint["channel_versions"][chan] = max_version + 1
            updated_channels.add(chan)
        else:
            logger.warning(f"Skipping write for channel '{chan}' which has no readers")
    # Channels that weren't updated in this step are notified of a new step
    for chan in channels:
        if chan not in updated_channels:
            channels[chan].update([])


@overload
def _prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    for_execution: Literal[False],
) -> tuple[Checkpoint, list[PregelTaskDescription]]:
    ...


@overload
def _prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    for_execution: Literal[True],
) -> tuple[Checkpoint, list[PregelExecutableTask]]:
    ...


def _prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    *,
    for_execution: bool,
) -> tuple[Checkpoint, Union[list[PregelTaskDescription], list[PregelExecutableTask]]]:
    checkpoint = copy_checkpoint(checkpoint)
    tasks: Union[list[PregelTaskDescription], list[PregelExecutableTask]] = []
    # Check if any processes should be run in next step
    # If so, prepare the values to be passed to them
    for name, proc in processes.items():
        seen = checkpoint["versions_seen"][name]
        # If any of the channels read by this process were updated
        if any(
            checkpoint["channel_versions"][chan] > seen[chan]
            for chan in proc.triggers
            if not isinstance(
                read_channel(channels, chan, return_exception=True), EmptyChannelError
            )
        ):
            # If all trigger channels subscribed by this process are not empty
            # then invoke the process with the values of all non-empty channels
            if isinstance(proc.channels, dict):
                try:
                    val: Any = {
                        k: read_channel(channels, chan, catch=chan not in proc.triggers)
                        for k, chan in proc.channels.items()
                    }
                except EmptyChannelError:
                    continue
            elif isinstance(proc.channels, list):
                for chan in proc.channels:
                    try:
                        val = read_channel(channels, chan, catch=False)
                        break
                    except EmptyChannelError:
                        pass
                else:
                    continue
            else:
                raise RuntimeError(
                    "Invalid channels type, expected list or dict, got {proc.channels}"
                )

            # If the process has a mapper, apply it to the value
            if proc.mapper is not None:
                val = proc.mapper(val)

            # update seen versions
            if for_execution:
                seen.update(
                    {
                        chan: checkpoint["channel_versions"][chan]
                        for chan in proc.triggers
                    }
                )

            if for_execution:
                if node := proc.get_node():
                    tasks.append(
                        PregelExecutableTask(name, val, node, deque(), proc.config)
                    )
            else:
                tasks.append(PregelTaskDescription(name, val))
    return checkpoint, tasks


async def _aconsume(iterator: AsyncIterator[Any]) -> None:
    """Consume an async iterator."""
    async for _ in iterator:
        pass
