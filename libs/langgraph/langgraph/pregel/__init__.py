from __future__ import annotations

import asyncio
import concurrent.futures
import json
import time
from collections import defaultdict, deque
from functools import partial
from inspect import signature
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    get_type_hints,
    overload,
)
from uuid import UUID, uuid5

from langchain_core.callbacks.manager import AsyncParentRunManager, ParentRunManager
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
    merge_configs,
    patch_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    create_model,
    get_unique_config_specs,
)
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from typing_extensions import Self

from langgraph.channels.base import (
    BaseChannel,
    EmptyChannelError,
)
from langgraph.channels.context import Context
from langgraph.channels.manager import (
    AsyncChannelsManager,
    ChannelsManager,
    create_checkpoint,
)
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    copy_checkpoint,
    empty_checkpoint,
)
from langgraph.constants import (
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_READ,
    CONFIG_KEY_SEND,
    INTERRUPT,
    TAG_HIDDEN,
    TASKS,
    Send,
)
from langgraph.errors import GraphInterrupt, GraphRecursionError, InvalidUpdateError
from langgraph.managed.base import (
    AsyncManagedValuesManager,
    ManagedValueMapping,
    ManagedValuesManager,
    ManagedValueSpec,
    is_managed_value,
)
from langgraph.pregel.debug import (
    map_debug_checkpoint,
    map_debug_task_results,
    map_debug_tasks,
    print_step_checkpoint,
    print_step_tasks,
    print_step_writes,
)
from langgraph.pregel.executor import AsyncBackgroundExecutor, BackgroundExecutor
from langgraph.pregel.io import (
    map_input,
    map_output_updates,
    map_output_values,
    read_channel,
    read_channels,
    single,
)
from langgraph.pregel.log import logger
from langgraph.pregel.read import PregelNode
from langgraph.pregel.retry import RetryPolicy, arun_with_retry, run_with_retry
from langgraph.pregel.types import (
    All,
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
                (
                    {key: channels}
                    if isinstance(channels, str) and key is not None
                    else (
                        [channels]
                        if isinstance(channels, str)
                        else {chan: chan for chan in channels}
                    )
                ),
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
                (
                    ChannelWriteEntry(k, skip_none=True, mapper=coerce_to_runnable(v))
                    if isinstance(v, Runnable) or callable(v)
                    else ChannelWriteEntry(k, value=v)
                )
                for k, v in kwargs.items()
            ]
        )


StreamMode = Literal["values", "updates", "debug"]
"""How the stream method should emit outputs.

- 'values': Emit all values of the state for each step.
- 'updates': Emit only the node name(s) and updates
    that were returned by the node(s) **after** each step.
- 'debug': Emit debug events for each step.
"""


class Pregel(
    RunnableSerializable[Union[dict[str, Any], Any], Union[dict[str, Any], Any]]
):
    nodes: Mapping[str, PregelNode]

    channels: Mapping[str, BaseChannel] = Field(default_factory=dict)

    auto_validate: bool = True

    stream_mode: StreamMode = "values"
    """Mode to stream output, defaults to 'values'."""

    output_channels: Union[str, Sequence[str]]

    stream_channels: Optional[Union[str, Sequence[str]]] = None
    """Channels to stream, defaults to all channels not in reserved channels"""

    interrupt_after_nodes: Union[All, Sequence[str]] = Field(default_factory=list)

    interrupt_before_nodes: Union[All, Sequence[str]] = Field(default_factory=list)

    input_channels: Union[str, Sequence[str]]

    step_timeout: Optional[float] = None
    """Maximum time to wait for a step to complete, in seconds. Defaults to None."""

    debug: bool = Field(default_factory=get_debug)
    """Whether to print debug information during execution. Defaults to False."""

    checkpointer: Optional[BaseCheckpointSaver] = None
    """Checkpointer used to save and load graph state. Defaults to None."""

    retry_policy: Optional[RetryPolicy] = None
    """Retry policy to use when running tasks. Set to None to disable."""

    config_type: Optional[Type[Any]] = None

    name: str = "LangGraph"

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether the graph can be serialized by Langchain."""
        return True

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
        )
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
                + (
                    [
                        ConfigurableFieldSpec(id=name, annotation=typ)
                        for name, typ in get_type_hints(self.config_type).items()
                    ]
                    if self.config_type is not None
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
        stream_channels = self.stream_channels_asis
        return (
            [stream_channels] if isinstance(stream_channels, str) else stream_channels
        )

    @property
    def stream_channels_asis(self) -> Union[str, Sequence[str]]:
        return self.stream_channels or [
            k for k in self.channels if not isinstance(self.channels[k], Context)
        ]

    @property
    def managed_values_dict(self) -> dict[str, ManagedValueSpec]:
        return {
            k: v
            for node in self.nodes.values()
            if isinstance(node.channels, dict)
            for k, v in node.channels.items()
            if is_managed_value(v)
        }

    def get_state(self, config: RunnableConfig) -> StateSnapshot:
        """Get the current state of the graph."""
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        saved = self.checkpointer.get_tuple(config)
        checkpoint = saved.checkpoint if saved else empty_checkpoint()
        config = saved.config if saved else config
        with ChannelsManager(
            self.channels, checkpoint, config
        ) as channels, ManagedValuesManager(
            self.managed_values_dict, ensure_config(config), self
        ) as managed:
            _, next_tasks = _prepare_next_tasks(
                checkpoint,
                self.nodes,
                channels,
                managed,
                config,
                -1,
                for_execution=False,
            )
            return StateSnapshot(
                read_channels(channels, self.stream_channels_asis),
                tuple(name for name, _ in next_tasks),
                saved.config if saved else config,
                saved.metadata if saved else None,
                saved.checkpoint["ts"] if saved else None,
                saved.parent_config if saved else None,
            )

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        """Get the current state of the graph."""
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        saved = await self.checkpointer.aget_tuple(config)
        checkpoint = saved.checkpoint if saved else empty_checkpoint()

        config = saved.config if saved else config
        async with AsyncChannelsManager(
            self.channels, checkpoint, config
        ) as channels, AsyncManagedValuesManager(
            self.managed_values_dict, ensure_config(config), self
        ) as managed:
            _, next_tasks = _prepare_next_tasks(
                checkpoint,
                self.nodes,
                channels,
                managed,
                config,
                -1,
                for_execution=False,
            )
            return StateSnapshot(
                read_channels(channels, self.stream_channels_asis),
                tuple(name for name, _ in next_tasks),
                saved.config if saved else config,
                saved.metadata if saved else None,
                saved.checkpoint["ts"] if saved else None,
                saved.parent_config if saved else None,
            )

    def get_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[StateSnapshot]:
        """Get the history of the state of the graph."""
        if not self.checkpointer:
            raise ValueError("No checkpointer set")
        if (
            filter is not None
            and signature(self.checkpointer.list).parameters.get("filter") is None
        ):
            raise ValueError("Checkpointer does not support filtering")
        for config, checkpoint, metadata, parent_config, _ in self.checkpointer.list(
            config, before=before, limit=limit, filter=filter
        ):
            with ChannelsManager(
                self.channels, checkpoint, config
            ) as channels, ManagedValuesManager(
                self.managed_values_dict, ensure_config(config), self
            ) as managed:
                _, next_tasks = _prepare_next_tasks(
                    checkpoint,
                    self.nodes,
                    channels,
                    managed,
                    config,
                    -1,
                    for_execution=False,
                )
                yield StateSnapshot(
                    read_channels(channels, self.stream_channels_asis),
                    tuple(name for name, _ in next_tasks),
                    config,
                    metadata,
                    checkpoint["ts"],
                    parent_config,
                )

    async def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[StateSnapshot]:
        """Get the history of the state of the graph."""
        if not self.checkpointer:
            raise ValueError("No checkpointer set")
        if (
            filter is not None
            and signature(self.checkpointer.list).parameters.get("filter") is None
        ):
            raise ValueError("Checkpointer does not support filtering")
        async for (
            config,
            checkpoint,
            metadata,
            parent_config,
            _,
        ) in self.checkpointer.alist(config, before=before, limit=limit, filter=filter):
            async with AsyncChannelsManager(
                self.channels, checkpoint, config
            ) as channels, AsyncManagedValuesManager(
                self.managed_values_dict, ensure_config(config), self
            ) as managed:
                _, next_tasks = _prepare_next_tasks(
                    checkpoint,
                    self.nodes,
                    channels,
                    managed,
                    config,
                    -1,
                    for_execution=False,
                )
                yield StateSnapshot(
                    read_channels(channels, self.stream_channels_asis),
                    tuple(name for name, _ in next_tasks),
                    config,
                    metadata,
                    checkpoint["ts"],
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
        saved = self.checkpointer.get_tuple(config)
        checkpoint = copy_checkpoint(saved.checkpoint) if saved else empty_checkpoint()
        # find last node that updated the state, if not provided
        if as_node is None and not any(
            v for vv in checkpoint["versions_seen"].values() for v in vv.values()
        ):
            if (
                isinstance(self.input_channels, str)
                and self.input_channels in self.nodes
            ):
                as_node = self.input_channels
        elif as_node is None:
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
        if as_node not in self.nodes:
            raise InvalidUpdateError(f"Node {as_node} does not exist")
        # update channels
        with ChannelsManager(self.channels, checkpoint, config) as channels:
            # create task to run all writers of the chosen node
            writers = self.nodes[as_node].get_writers()
            if not writers:
                raise InvalidUpdateError(f"Node {as_node} has no writers")
            task = PregelExecutableTask(
                as_node,
                values,
                RunnableSequence(*writers) if len(writers) > 1 else writers[0],
                deque(),
                None,
                [INTERRUPT],
                None,
                str(uuid5(UUID(checkpoint["id"]), INTERRUPT)),
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
                            _local_read, checkpoint, channels, task.writes, config
                        ),
                    },
                ),
            )
            # apply to checkpoint and save
            _apply_writes(
                checkpoint, channels, task.writes, self.checkpointer.get_next_version
            )
            step = saved.metadata.get("step", -2) + 1 if saved else -1

            # merge configurable fields with previous checkpoint config
            checkpoint_config = config
            if saved:
                checkpoint_config = {
                    "configurable": {
                        **config.get("configurable", {}),
                        **saved.config["configurable"],
                    }
                }

            return self.checkpointer.put(
                checkpoint_config,
                create_checkpoint(checkpoint, channels, step),
                {
                    "source": "update",
                    "step": step,
                    "writes": {as_node: values},
                },
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
        saved = await self.checkpointer.aget_tuple(config)
        checkpoint = copy_checkpoint(saved.checkpoint) if saved else empty_checkpoint()
        # find last node that updated the state, if not provided
        if as_node is None and not saved:
            if (
                isinstance(self.input_channels, str)
                and self.input_channels in self.nodes
            ):
                as_node = self.input_channels
        elif as_node is None:
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
        if as_node not in self.nodes:
            raise InvalidUpdateError(f"Node {as_node} does not exist")
        # update channels, acting as the chosen node
        async with AsyncChannelsManager(self.channels, checkpoint, config) as channels:
            # create task to run all writers of the chosen node
            writers = self.nodes[as_node].get_writers()
            if not writers:
                raise InvalidUpdateError(f"Node {as_node} has no writers")
            task = PregelExecutableTask(
                as_node,
                values,
                RunnableSequence(*writers) if len(writers) > 1 else writers[0],
                deque(),
                None,
                [INTERRUPT],
                None,
                str(uuid5(UUID(checkpoint["id"]), INTERRUPT)),
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
                            _local_read, checkpoint, channels, task.writes, config
                        ),
                    },
                ),
            )
            # apply to checkpoint and save
            _apply_writes(
                checkpoint, channels, task.writes, self.checkpointer.get_next_version
            )
            step = saved.metadata.get("step", -2) + 1 if saved else -1

            # merge configurable fields with previous checkpoint config
            checkpoint_config = config
            if saved:
                checkpoint_config = {
                    "configurable": {
                        **config.get("configurable", {}),
                        **saved.config["configurable"],
                    }
                }

            return await self.checkpointer.aput(
                checkpoint_config,
                create_checkpoint(checkpoint, channels, step),
                {
                    "source": "update",
                    "step": step,
                    "writes": {as_node: values},
                },
            )

    def _defaults(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: Optional[bool] = None,
    ) -> tuple[
        bool,
        Sequence[StreamMode],
        Union[str, Sequence[str]],
        Union[str, Sequence[str]],
        Optional[Sequence[str]],
        Optional[Sequence[str]],
    ]:
        debug = debug if debug is not None else self.debug
        if output_keys is None:
            output_keys = self.stream_channels_asis
        else:
            validate_keys(output_keys, self.channels)
        if input_keys is None:
            input_keys = self.input_channels
        else:
            validate_keys(input_keys, self.channels)
        interrupt_before = interrupt_before or self.interrupt_before_nodes
        interrupt_after = interrupt_after or self.interrupt_after_nodes
        stream_mode = stream_mode if stream_mode is not None else self.stream_mode
        if not isinstance(stream_mode, list):
            stream_mode = [stream_mode]
        if config is not None and config.get("configurable", {}).get(CONFIG_KEY_READ):
            # if being called as a node in another graph, always use values mode
            stream_mode = ["values"]
        if config is not None and config.get("configurable", {}).get(
            CONFIG_KEY_CHECKPOINTER
        ):
            checkpointer: Optional[BaseCheckpointSaver] = config["configurable"][
                CONFIG_KEY_CHECKPOINTER
            ]
        else:
            checkpointer = self.checkpointer
        return (
            debug,
            stream_mode,
            input_keys,
            output_keys,
            interrupt_before,
            interrupt_after,
            checkpointer,
        )

    def stream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: Optional[bool] = None,
    ) -> Iterator[Union[dict[str, Any], Any]]:
        """Stream graph steps for a single input.

        Args:
            input: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to self.stream_mode.
                Options are 'values', 'updates', and 'debug'.
                values: Emit the current values of the state for each step.
                updates: Emit only the updates to the state for each step.
                    Output is a dict with the node name as key and the updated values as value.
                debug: Emit debug events for each step.
            output_keys: The keys to stream, defaults to all non-context channels.
            input_keys: The keys to use from the input, defaults to all input channels.
            interrupt_before: Nodes to interrupt before, defaults to all nodes in the graph.
            interrupt_after: Nodes to interrupt after, defaults to all nodes in the graph.
            debug: Whether to print debug information during execution, defaults to False.

        Yields:
            The output of each step in the graph. The output shape depends on the stream_mode.

        Examples:
            Using different stream modes with a graph:
            ```pycon
            >>> import operator
            >>> from typing_extensions import Annotated, TypedDict
            >>> from langgraph.graph import StateGraph
            >>> from langgraph.constants import START
            ...
            >>> class State(TypedDict):
            ...     alist: Annotated[list, operator.add]
            ...     another_list: Annotated[list, operator.add]
            ...
            >>> builder = StateGraph(State)
            >>> builder.add_node("a", lambda _state: {"another_list": ["hi"]})
            >>> builder.add_node("b", lambda _state: {"alist": ["there"]})
            >>> builder.add_edge("a", "b")
            >>> builder.add_edge(START, "a")
            >>> graph = builder.compile()
            ```
            With stream_mode="values":

            ```pycon
            >>> for event in graph.stream({"alist": ['Ex for stream_mode="values"']}, stream_mode="values"):
            ...     print(event)
            {'alist': ['Ex for stream_mode="values"'], 'another_list': []}
            {'alist': ['Ex for stream_mode="values"'], 'another_list': ['hi']}
            {'alist': ['Ex for stream_mode="values"', 'there'], 'another_list': ['hi']}
            ```
            With stream_mode="updates":

            ```pycon
            >>> for event in graph.stream({"alist": ['Ex for stream_mode="updates"']}, stream_mode="updates"):
            ...     print(event)
            {'a': {'another_list': ['hi']}}
            {'b': {'alist': ['there']}}
            ```
            With stream_mode="debug":

            ```pycon
            >>> for event in graph.stream({"alist": ['Ex for stream_mode="debug"']}, stream_mode="debug"):
            ...     print(event)
            {'type': 'task', 'timestamp': '2024-06-23T...+00:00', 'step': 1, 'payload': {'id': '...', 'name': 'a', 'input': {'alist': ['Ex for stream_mode="debug"'], 'another_list': []}, 'triggers': ['start:a']}}
            {'type': 'task_result', 'timestamp': '2024-06-23T...+00:00', 'step': 1, 'payload': {'id': '...', 'name': 'a', 'result': [('another_list', ['hi'])]}}
            {'type': 'task', 'timestamp': '2024-06-23T...+00:00', 'step': 2, 'payload': {'id': '...', 'name': 'b', 'input': {'alist': ['Ex for stream_mode="debug"'], 'another_list': ['hi']}, 'triggers': ['a']}}
            {'type': 'task_result', 'timestamp': '2024-06-23T...+00:00', 'step': 2, 'payload': {'id': '...', 'name': 'b', 'result': [('alist', ['there'])]}}
            ```
        """
        config = ensure_config(config)
        is_subgraph = config.get("configurable", {}).get(CONFIG_KEY_READ) is not None
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name", self.get_name()),
            run_id=config.get("run_id"),
        )
        try:
            if config["recursion_limit"] < 1:
                raise ValueError("recursion_limit must be at least 1")
            if self.checkpointer and not config.get("configurable"):
                raise ValueError(
                    f"Checkpointer requires one or more of the following 'configurable' keys: {[s.id for s in self.checkpointer.config_specs]}"
                )
            # assign defaults
            (
                debug,
                stream_modes,
                input_keys,
                output_keys,
                interrupt_before,
                interrupt_after,
                checkpointer,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                input_keys=input_keys,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
            )
            # copy nodes to ignore mutations during execution
            processes = {**self.nodes}
            # get checkpoint from saver, or create an empty one
            saved = checkpointer.get_tuple(config) if checkpointer else None
            checkpoint = saved.checkpoint if saved else empty_checkpoint()

            # merge configurable fields with previous checkpoint config
            checkpoint_config = config
            if saved:
                checkpoint_config = {
                    **config,
                    **saved.config,
                    "configurable": {
                        **config.get("configurable", {}),
                        **saved.config["configurable"],
                    },
                }

            start = saved.metadata.get("step", -2) + 1 if saved else -1
            # create channels from checkpoint
            with BackgroundExecutor(config) as submit, ChannelsManager(
                self.channels, checkpoint, config
            ) as channels, ManagedValuesManager(
                self.managed_values_dict, config, self
            ) as managed:

                def put_writes(task_id: str, writes: Sequence[tuple[str, Any]]) -> None:
                    if checkpointer is not None:
                        submit(
                            checkpointer.put_writes,
                            {
                                **checkpoint_config,
                                "configurable": {
                                    **checkpoint_config["configurable"],
                                    "thread_ts": checkpoint["id"],
                                },
                            },
                            writes,
                            task_id,
                        )

                def put_checkpoint(metadata: CheckpointMetadata) -> Iterator[Any]:
                    nonlocal checkpoint, checkpoint_config, channels

                    if checkpointer is None:
                        return
                    if debug:
                        print_step_checkpoint(
                            metadata["step"], channels, self.stream_channels_list
                        )

                    # create new checkpoint
                    checkpoint = create_checkpoint(
                        checkpoint, channels, metadata["step"]
                    )
                    # save it, without blocking
                    submit(
                        checkpointer.put,
                        checkpoint_config,
                        copy_checkpoint(checkpoint),
                        metadata,
                    )
                    # update checkpoint config
                    checkpoint_config = {
                        **checkpoint_config,
                        "configurable": {
                            **checkpoint_config["configurable"],
                            "thread_ts": checkpoint["id"],
                        },
                    }
                    # yield debug checkpoint event
                    if "debug" in stream_modes:
                        yield from _with_mode(
                            "debug",
                            isinstance(stream_mode, list),
                            map_debug_checkpoint(
                                metadata["step"],
                                checkpoint_config,
                                channels,
                                self.stream_channels_asis,
                                metadata,
                            ),
                        )

                # map inputs to channel updates
                if input_writes := deque(map_input(input_keys, input)):
                    # discard any unfinished tasks from previous checkpoint
                    checkpoint, _ = _prepare_next_tasks(
                        checkpoint,
                        processes,
                        channels,
                        managed,
                        config,
                        -1,
                        for_execution=True,
                        get_next_version=(
                            checkpointer.get_next_version
                            if checkpointer
                            else _increment
                        ),
                        checkpointer=checkpointer,
                    )
                    # apply input writes
                    _apply_writes(
                        checkpoint,
                        channels,
                        input_writes,
                        (checkpointer.get_next_version if checkpointer else _increment),
                    )
                    # save input checkpoint
                    yield from put_checkpoint(
                        {
                            "source": "input",
                            "step": start,
                            "writes": input,
                        }
                    )
                    # increment start to 0
                    start += 1
                else:
                    # no input is taken as signal to proceed past previous interrupt
                    checkpoint = copy_checkpoint(checkpoint)
                    for k in channels:
                        if k in checkpoint["channel_versions"]:
                            version = checkpoint["channel_versions"][k]
                            checkpoint["versions_seen"][INTERRUPT][k] = version

                # Similarly to Bulk Synchronous Parallel / Pregel model
                # computation proceeds in steps, while there are channel updates
                # channel updates from step N are only visible in step N+1
                # channels are guaranteed to be immutable for the duration of the step,
                # with channel updates applied only at the transition between steps
                stop = start + config["recursion_limit"] + 1
                for step in range(start, stop):
                    next_checkpoint, next_tasks = _prepare_next_tasks(
                        checkpoint,
                        processes,
                        channels,
                        managed,
                        config,
                        step,
                        for_execution=True,
                        manager=run_manager,
                        get_next_version=(
                            checkpointer.get_next_version
                            if checkpointer
                            else _increment
                        ),
                        checkpointer=checkpointer,
                    )

                    # assign pending writes to tasks
                    if saved and saved.pending_writes:
                        for task in next_tasks:
                            task.writes.extend(
                                (c, v)
                                for tid, c, v in saved.pending_writes
                                if tid == task.id
                            )

                    # if no more tasks, we're done
                    if not next_tasks:
                        if step == start:
                            raise ValueError("No tasks to run in graph.")
                        else:
                            break

                    # before execution, check if we should interrupt
                    if _should_interrupt(
                        checkpoint,
                        interrupt_before,
                        self.stream_channels_list,
                        next_tasks,
                    ):
                        if is_subgraph:
                            raise GraphInterrupt()
                        else:
                            break
                    else:
                        checkpoint = next_checkpoint

                    if debug:
                        print_step_tasks(step, next_tasks)
                    if "debug" in stream_modes:
                        yield from _with_mode(
                            "debug",
                            isinstance(stream_mode, list),
                            map_debug_tasks(step, next_tasks),
                        )

                    # execute tasks, and wait for one to fail or all to finish.
                    # each task is independent from all other concurrent tasks
                    # yield updates/debug output as each task finishes
                    futures = {
                        submit(run_with_retry, task, self.retry_policy): task
                        for task in next_tasks
                        if not task.writes
                    }
                    end_time = (
                        self.step_timeout + time.monotonic()
                        if self.step_timeout
                        else None
                    )
                    if not futures:
                        done, inflight = set(), set()
                    while futures:
                        done, inflight = concurrent.futures.wait(
                            futures,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                            timeout=(
                                max(0, end_time - time.monotonic())
                                if end_time
                                else None
                            ),
                        )
                        if not done:
                            break  # timed out
                        for fut in done:
                            task = futures.pop(fut)
                            if fut.exception() is not None:
                                # we got an exception, break out of while loop
                                # exception will be handled in panic_or_proceed
                                futures.clear()
                            else:
                                # save task writes to checkpointer, unless this
                                # is the single or last task in this step
                                if futures:
                                    put_writes(task.id, task.writes)
                                # yield updates output for the finished task
                                if "updates" in stream_modes:
                                    yield from _with_mode(
                                        "updates",
                                        isinstance(stream_mode, list),
                                        map_output_updates(output_keys, [task]),
                                    )
                                if "debug" in stream_modes:
                                    yield from _with_mode(
                                        "debug",
                                        isinstance(stream_mode, list),
                                        map_debug_task_results(
                                            step, [task], self.stream_channels_list
                                        ),
                                    )
                        else:
                            # remove references to loop vars
                            del fut, task

                    # panic on failure or timeout
                    # NOTE: for subgraphs we'll raise GraphInterrupt exception on interrupt
                    exceptions_to_handle = () if is_subgraph else (GraphInterrupt,)
                    try:
                        _panic_or_proceed(done, inflight, step)
                    except exceptions_to_handle:
                        break

                    # don't keep futures around in memory longer than needed
                    del done, inflight, futures

                    # combine pending writes from all tasks
                    pending_writes = deque[tuple[str, Any]]()
                    for task in next_tasks:
                        pending_writes.extend(task.writes)

                    if debug:
                        print_step_writes(
                            step, pending_writes, self.stream_channels_list
                        )

                    # apply writes to channels
                    _apply_writes(
                        checkpoint,
                        channels,
                        pending_writes,
                        (checkpointer.get_next_version if checkpointer else _increment),
                    )

                    # yield values output
                    if "values" in stream_modes:
                        yield from _with_mode(
                            "values",
                            isinstance(stream_mode, list),
                            map_output_values(output_keys, pending_writes, channels),
                        )

                    # save end of step checkpoint
                    yield from put_checkpoint(
                        {
                            "source": "loop",
                            "step": step,
                            "writes": (
                                single(map_output_updates(output_keys, next_tasks))
                                if self.stream_mode == "updates"
                                else single(
                                    map_output_values(
                                        output_keys, pending_writes, channels
                                    ),
                                )
                            ),
                        }
                    )

                    # after execution, check if we should interrupt
                    if _should_interrupt(
                        checkpoint,
                        interrupt_after,
                        self.stream_channels_list,
                        next_tasks,
                    ):
                        if is_subgraph:
                            raise GraphInterrupt()
                        else:
                            break
                else:
                    raise GraphRecursionError(
                        f"Recursion limit of {config['recursion_limit']} reached"
                        "without hitting a stop condition. You can increase the "
                        "limit by setting the `recursion_limit` config key."
                    )

                # set final channel values as run output
                run_manager.on_chain_end(read_channels(channels, output_keys))
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise

    async def astream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: Optional[bool] = None,
    ) -> AsyncIterator[Union[dict[str, Any], Any]]:
        """Stream graph steps for a single input.

        Args:
            input: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to self.stream_mode.
                Options are 'values', 'updates', and 'debug'.
                values: Emit the current values of the state for each step.
                updates: Emit only the updates to the state for each step.
                    Output is a dict with the node name as key and the updated values as value.
                debug: Emit debug events for each step.
            output_keys: The keys to stream, defaults to all non-context channels.
            input_keys: The keys to use from the input, defaults to all input channels.
            interrupt_before: Nodes to interrupt before, defaults to all nodes in the graph.
            interrupt_after: Nodes to interrupt after, defaults to all nodes in the graph.
            debug: Whether to print debug information during execution, defaults to False.

        Yields:
            The output of each step in the graph. The output shape depends on the stream_mode.

        Examples:
            Using different stream modes with a graph:
            ```pycon
            >>> import operator
            >>> from typing_extensions import Annotated, TypedDict
            >>> from langgraph.graph import StateGraph
            >>> from langgraph.constants import START
            ...
            >>> class State(TypedDict):
            ...     alist: Annotated[list, operator.add]
            ...     another_list: Annotated[list, operator.add]
            ...
            >>> builder = StateGraph(State)
            >>> builder.add_node("a", lambda _state: {"another_list": ["hi"]})
            >>> builder.add_node("b", lambda _state: {"alist": ["there"]})
            >>> builder.add_edge("a", "b")
            >>> builder.add_edge(START, "a")
            >>> graph = builder.compile()
            ```
            With stream_mode="values":

            ```pycon
            >>> async for event in graph.astream({"alist": ['Ex for stream_mode="values"']}, stream_mode="values"):
            ...     print(event)
            {'alist': ['Ex for stream_mode="values"'], 'another_list': []}
            {'alist': ['Ex for stream_mode="values"'], 'another_list': ['hi']}
            {'alist': ['Ex for stream_mode="values"', 'there'], 'another_list': ['hi']}
            ```
            With stream_mode="updates":

            ```pycon
            >>> async for event in graph.astream({"alist": ['Ex for stream_mode="updates"']}, stream_mode="updates"):
            ...     print(event)
            {'a': {'another_list': ['hi']}}
            {'b': {'alist': ['there']}}
            ```
            With stream_mode="debug":

            ```pycon
            >>> async for event in graph.astream({"alist": ['Ex for stream_mode="debug"']}, stream_mode="debug"):
            ...     print(event)
            {'type': 'task', 'timestamp': '2024-06-23T...+00:00', 'step': 1, 'payload': {'id': '...', 'name': 'a', 'input': {'alist': ['Ex for stream_mode="debug"'], 'another_list': []}, 'triggers': ['start:a']}}
            {'type': 'task_result', 'timestamp': '2024-06-23T...+00:00', 'step': 1, 'payload': {'id': '...', 'name': 'a', 'result': [('another_list', ['hi'])]}}
            {'type': 'task', 'timestamp': '2024-06-23T...+00:00', 'step': 2, 'payload': {'id': '...', 'name': 'b', 'input': {'alist': ['Ex for stream_mode="debug"'], 'another_list': ['hi']}, 'triggers': ['a']}}
            {'type': 'task_result', 'timestamp': '2024-06-23T...+00:00', 'step': 2, 'payload': {'id': '...', 'name': 'b', 'result': [('alist', ['there'])]}}
            ```
        """
        config = ensure_config(config)
        is_subgraph = config.get("configurable", {}).get(CONFIG_KEY_READ) is not None
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name", self.get_name()),
            run_id=config.get("run_id"),
        )
        # if running from astream_log() run each proc with streaming
        do_stream = next(
            (
                h
                for h in run_manager.handlers
                if isinstance(h, _StreamingCallbackHandler)
            ),
            None,
        )
        try:
            loop = asyncio.get_event_loop()
            if config["recursion_limit"] < 1:
                raise ValueError("recursion_limit must be at least 1")
            if self.checkpointer and not config.get("configurable"):
                raise ValueError(
                    f"Checkpointer requires one or more of the following 'configurable' keys: {[s.id for s in self.checkpointer.config_specs]}"
                )
            # assign defaults
            (
                debug,
                stream_modes,
                input_keys,
                output_keys,
                interrupt_before,
                interrupt_after,
                checkpointer,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                input_keys=input_keys,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
            )
            # copy nodes to ignore mutations during execution
            processes = {**self.nodes}
            # get checkpoint from saver, or create an empty one
            saved = await checkpointer.aget_tuple(config) if checkpointer else None
            checkpoint = saved.checkpoint if saved else empty_checkpoint()

            # merge configurable fields with previous checkpoint config
            checkpoint_config = config
            if saved:
                checkpoint_config = {
                    **config,
                    **saved.config,
                    "configurable": {
                        **config.get("configurable", {}),
                        **saved.config["configurable"],
                    },
                }

            start = saved.metadata.get("step", -2) + 1 if saved else -1

            # create channels from checkpoint
            async with AsyncBackgroundExecutor() as submit, AsyncChannelsManager(
                self.channels, checkpoint, config
            ) as channels, AsyncManagedValuesManager(
                self.managed_values_dict, config, self
            ) as managed:

                def put_writes(task_id: str, writes: Sequence[tuple[str, Any]]) -> None:
                    if checkpointer is not None:
                        submit(
                            checkpointer.aput_writes,
                            {
                                **checkpoint_config,
                                "configurable": {
                                    **checkpoint_config["configurable"],
                                    "thread_ts": checkpoint["id"],
                                },
                            },
                            writes,
                            task_id,
                        )

                def put_checkpoint(metadata: CheckpointMetadata) -> Iterator[Any]:
                    nonlocal checkpoint, checkpoint_config, channels

                    if checkpointer is None:
                        return
                    if debug:
                        print_step_checkpoint(
                            metadata["step"], channels, self.stream_channels_list
                        )

                    # create new checkpoint
                    checkpoint = create_checkpoint(
                        checkpoint, channels, metadata["step"]
                    )
                    # save it, without blocking
                    submit(
                        checkpointer.aput,
                        checkpoint_config,
                        copy_checkpoint(checkpoint),
                        metadata,
                    )

                    # update checkpoint config
                    checkpoint_config = {
                        **checkpoint_config,
                        "configurable": {
                            **checkpoint_config["configurable"],
                            "thread_ts": checkpoint["id"],
                        },
                    }
                    # yield debug checkpoint event
                    if "debug" in stream_modes:
                        yield from _with_mode(
                            "debug",
                            isinstance(stream_mode, list),
                            map_debug_checkpoint(
                                metadata["step"],
                                checkpoint_config,
                                channels,
                                self.stream_channels_asis,
                                metadata,
                            ),
                        )

                # map inputs to channel updates
                if input_writes := deque(map_input(input_keys, input)):
                    # discard any unfinished tasks from previous checkpoint
                    checkpoint, _ = _prepare_next_tasks(
                        checkpoint,
                        processes,
                        channels,
                        managed,
                        config,
                        -1,
                        for_execution=True,
                        get_next_version=(
                            checkpointer.get_next_version
                            if checkpointer
                            else _increment
                        ),
                        checkpointer=checkpointer,
                    )
                    # apply input writes
                    _apply_writes(
                        checkpoint,
                        channels,
                        input_writes,
                        (checkpointer.get_next_version if checkpointer else _increment),
                    )
                    # save input checkpoint
                    for chunk in put_checkpoint(
                        {"source": "input", "step": start, "writes": input}
                    ):
                        yield chunk
                    # increment start to 0
                    start += 1
                else:
                    # no input is taken as signal to proceed past previous interrupt
                    checkpoint = copy_checkpoint(checkpoint)
                    for k in channels:
                        if k in checkpoint["channel_versions"]:
                            version = checkpoint["channel_versions"][k]
                            checkpoint["versions_seen"][INTERRUPT][k] = version

                # Similarly to Bulk Synchronous Parallel / Pregel model
                # computation proceeds in steps, while there are channel updates
                # channel updates from step N are only visible in step N+1,
                # channels are guaranteed to be immutable for the duration of the step,
                # channel updates being applied only at the transition between steps
                stop = start + config["recursion_limit"] + 1
                for step in range(start, stop):
                    next_checkpoint, next_tasks = _prepare_next_tasks(
                        checkpoint,
                        processes,
                        channels,
                        managed,
                        config,
                        step,
                        for_execution=True,
                        manager=run_manager,
                        get_next_version=(
                            checkpointer.get_next_version
                            if checkpointer
                            else _increment
                        ),
                        checkpointer=checkpointer,
                    )

                    # assign pending writes to tasks
                    if saved and saved.pending_writes:
                        for task in next_tasks:
                            task.writes.extend(
                                (c, v)
                                for tid, c, v in saved.pending_writes
                                if tid == task.id
                            )

                    # if no more tasks, we're done
                    if not next_tasks:
                        if step == start:
                            raise ValueError("No tasks to run in graph.")
                        else:
                            break

                    # before execution, check if we should interrupt
                    if _should_interrupt(
                        checkpoint,
                        interrupt_before,
                        self.stream_channels_list,
                        next_tasks,
                    ):
                        if is_subgraph:
                            raise GraphInterrupt()
                        else:
                            break
                    else:
                        checkpoint = next_checkpoint

                    if debug:
                        print_step_tasks(step, next_tasks)
                    if "debug" in stream_modes:
                        for chunk in _with_mode(
                            "debug",
                            isinstance(stream_mode, list),
                            map_debug_tasks(step, next_tasks),
                        ):
                            yield chunk

                    # execute tasks, and wait for one to fail or all to finish.
                    # each task is independent from all other concurrent tasks
                    # yield updates/debug output as each task finishes
                    futures = {
                        submit(
                            arun_with_retry,
                            task,
                            self.retry_policy,
                            do_stream,
                            __name__=task.name,
                            __cancel_on_exit__=True,
                        ): task
                        for task in next_tasks
                        if not task.writes
                    }
                    end_time = (
                        self.step_timeout + loop.time() if self.step_timeout else None
                    )
                    if not futures:
                        done, inflight = set(), set()
                    while futures:
                        done, inflight = await asyncio.wait(
                            futures,
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=(
                                max(0, end_time - loop.time()) if end_time else None
                            ),
                        )
                        if not done:
                            break  # timed out
                        for fut in done:
                            task = futures.pop(fut)
                            if fut.exception() is not None:
                                # we got an exception, break out of while loop
                                # exception will be handle in panic_or_proceed
                                futures.clear()
                            else:
                                # save task writes to checkpointer, unless this
                                # is the single or last task in this step
                                if futures:
                                    put_writes(task.id, task.writes)
                                # yield updates output for the finished task
                                if "updates" in stream_modes:
                                    for chunk in _with_mode(
                                        "updates",
                                        isinstance(stream_mode, list),
                                        map_output_updates(output_keys, [task]),
                                    ):
                                        yield chunk
                                if "debug" in stream_modes:
                                    for chunk in _with_mode(
                                        "debug",
                                        isinstance(stream_mode, list),
                                        map_debug_task_results(
                                            step, [task], self.stream_channels_list
                                        ),
                                    ):
                                        yield chunk
                        else:
                            # remove references to loop vars
                            del fut, task

                    # panic on failure or timeout
                    # NOTE: for subgraphs we'll raise GraphInterrupt exception on interrupt
                    exceptions_to_handle = () if is_subgraph else (GraphInterrupt,)
                    try:
                        _panic_or_proceed(done, inflight, step)
                    except exceptions_to_handle:
                        break

                    # don't keep futures around in memory longer than needed
                    del done, inflight, futures

                    # combine pending writes from all tasks
                    pending_writes = deque[tuple[str, Any]]()
                    for task in next_tasks:
                        pending_writes.extend(task.writes)

                    if debug:
                        print_step_writes(
                            step, pending_writes, self.stream_channels_list
                        )

                    # apply writes to channels
                    _apply_writes(
                        checkpoint,
                        channels,
                        pending_writes,
                        (checkpointer.get_next_version if checkpointer else _increment),
                    )

                    # yield current values
                    if "values" in stream_modes:
                        for chunk in _with_mode(
                            "values",
                            isinstance(stream_mode, list),
                            map_output_values(output_keys, pending_writes, channels),
                        ):
                            yield chunk

                    # save end of step checkpoint
                    for chunk in put_checkpoint(
                        {
                            "source": "loop",
                            "step": step,
                            "writes": (
                                single(map_output_updates(output_keys, next_tasks))
                                if self.stream_mode == "updates"
                                else single(
                                    map_output_values(
                                        output_keys, pending_writes, channels
                                    )
                                )
                            ),
                        }
                    ):
                        yield chunk

                    # after execution, check if we should interrupt
                    if _should_interrupt(
                        checkpoint,
                        interrupt_after,
                        self.stream_channels_list,
                        next_tasks,
                    ):
                        if is_subgraph:
                            raise GraphInterrupt()
                        else:
                            break
                else:
                    raise GraphRecursionError(
                        f"Recursion limit of {config['recursion_limit']} reached"
                        "without hitting a stop condition. You can increase the limit"
                        "by setting the `recursion_limit` config key."
                    )

                # set final channel values as run output
                await run_manager.on_chain_end(read_channels(channels, output_keys))
        except BaseException as e:
            await asyncio.shield(run_manager.on_chain_error(e))
            raise

    def invoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: StreamMode = "values",
        output_keys: Optional[Union[str, Sequence[str]]] = None,
        input_keys: Optional[Union[str, Sequence[str]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        """Run the graph with a single input and config.

        Args:
            input: The input data for the graph. It can be a dictionary or any other type.
            config: Optional. The configuration for the graph run.
            stream_mode: Optional[str]. The stream mode for the graph run. Default is "values".
            output_keys: Optional. The output keys to retrieve from the graph run.
            input_keys: Optional. The input keys to provide for the graph run.
            interrupt_before: Optional. The nodes to interrupt the graph run before.
            interrupt_after: Optional. The nodes to interrupt the graph run after.
            debug: Optional. Enable debug mode for the graph run.
            **kwargs: Additional keyword arguments to pass to the graph run.

        Returns:
            The output of the graph run. If stream_mode is "values", it returns the latest output.
            If stream_mode is not "values", it returns a list of output chunks.
        """
        output_keys = output_keys if output_keys is not None else self.output_channels
        if stream_mode == "values":
            latest: Union[dict[str, Any], Any] = None
        else:
            chunks = []
        for chunk in self.stream(
            input,
            config,
            stream_mode=stream_mode,
            output_keys=output_keys,
            input_keys=input_keys,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            **kwargs,
        ):
            if stream_mode == "values":
                latest = chunk
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
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        """Asynchronously invoke the graph on a single input.

        Args:
            input: The input data for the computation. It can be a dictionary or any other type.
            config: Optional. The configuration for the computation.
            stream_mode: Optional. The stream mode for the computation. Default is "values".
            output_keys: Optional. The output keys to include in the result. Default is None.
            input_keys: Optional. The input keys to include in the result. Default is None.
            interrupt_before: Optional. The nodes to interrupt before. Default is None.
            interrupt_after: Optional. The nodes to interrupt after. Default is None.
            debug: Optional. Whether to enable debug mode. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the computation. If stream_mode is "values", it returns the latest value.
            If stream_mode is "chunks", it returns a list of chunks.
        """

        output_keys = output_keys if output_keys is not None else self.output_channels
        if stream_mode == "values":
            latest: Union[dict[str, Any], Any] = None
        else:
            chunks = []
        async for chunk in self.astream(
            input,
            config,
            stream_mode=stream_mode,
            output_keys=output_keys,
            input_keys=input_keys,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            **kwargs,
        ):
            if stream_mode == "values":
                latest = chunk
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
    timeout_exc_cls: Type[Exception] = TimeoutError,
) -> None:
    while done:
        # if any task failed
        if exc := done.pop().exception():
            # cancel all pending tasks
            while inflight:
                inflight.pop().cancel()
            # raise the exception
            raise exc

    if inflight:
        # if we got here means we timed out
        while inflight:
            # cancel all pending tasks
            inflight.pop().cancel()
        # raise timeout error
        raise timeout_exc_cls(f"Timed out at step {step}")


def _should_interrupt(
    checkpoint: Checkpoint,
    interrupt_nodes: Union[All, Sequence[str]],
    snapshot_channels: Sequence[str],
    tasks: list[PregelExecutableTask],
) -> Sequence[str]:
    version_type = type(next(iter(checkpoint["channel_versions"].values()), None))
    null_version = version_type()
    # defaultdicts are mutated on access :( so we need to copy
    seen = checkpoint["versions_seen"].copy()[INTERRUPT]
    return (
        # interrupt if any channel has been updated since last interrupt, otherwise return
        any(
            version > seen.get(chan, null_version)
            for chan, version in checkpoint["channel_versions"].items()
        )
        # and any triggered node is in interrupt_nodes list
        and any(
            task.name
            for task in tasks
            if (
                (not task.config or TAG_HIDDEN not in task.config.get("tags"))
                if interrupt_nodes == "*"
                else task.name in interrupt_nodes
            )
        )
    )


def _local_read(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    writes: Sequence[tuple[str, Any]],
    config: RunnableConfig,
    select: Union[list[str], str],
    fresh: bool = False,
) -> Union[dict[str, Any], Any]:
    if fresh:
        checkpoint = create_checkpoint(checkpoint, channels, -1)
        context_channels = {k: v for k, v in channels.items() if isinstance(v, Context)}
        with ChannelsManager(
            {k: v for k, v in channels.items() if k not in context_channels},
            checkpoint,
            config,
        ) as channels:
            all_channels = {**channels, **context_channels}
            _apply_writes(copy_checkpoint(checkpoint), all_channels, writes, None)
            return read_channels(all_channels, select)
    else:
        return read_channels(channels, select)


def _local_write(
    commit: Callable[[Sequence[tuple[str, Any]]], None],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    writes: Sequence[tuple[str, Any]],
) -> None:
    for chan, value in writes:
        if chan == TASKS:
            if not isinstance(value, Send):
                raise InvalidUpdateError(
                    f"Invalid packet type, expected Packet, got {value}"
                )
            if value.node not in processes:
                raise InvalidUpdateError(f"Invalid node name {value.node} in packet")
        elif chan not in channels:
            logger.warning(f"Skipping write for channel '{chan}' which has no readers")
    commit(writes)


def _increment(current: Optional[int], channel: BaseChannel) -> int:
    return current + 1 if current is not None else 1


def _apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    pending_writes: Sequence[tuple[str, Any]],
    get_next_version: Optional[Callable[[int, BaseChannel], int]],
) -> None:
    if checkpoint["pending_sends"]:
        checkpoint["pending_sends"].clear()

    pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
    # Group writes by channel
    for chan, val in pending_writes:
        if chan == TASKS:
            checkpoint["pending_sends"].append(val)
        else:
            pending_writes_by_channel[chan].append(val)

    # Find the highest version of all channels
    if checkpoint["channel_versions"]:
        max_version = max(checkpoint["channel_versions"].values())
    else:
        max_version = None

    updated_channels: set[str] = set()
    # Apply writes to channels
    for chan, vals in pending_writes_by_channel.items():
        if chan in channels:
            try:
                updated = channels[chan].update(vals)
            except InvalidUpdateError as e:
                raise InvalidUpdateError(
                    f"Invalid update for channel {chan} with values {vals}"
                ) from e
            if updated and get_next_version is not None:
                checkpoint["channel_versions"][chan] = get_next_version(
                    max_version, channels[chan]
                )
            updated_channels.add(chan)
    # Channels that weren't updated in this step are notified of a new step
    for chan in channels:
        if chan not in updated_channels:
            if channels[chan].update([]) and get_next_version is not None:
                checkpoint["channel_versions"][chan] = get_next_version(
                    max_version, channels[chan]
                )


@overload
def _prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    for_execution: Literal[False],
    get_next_version: Literal[None] = None,
    manager: Literal[None] = None,
    checkpointer: Literal[None] = None,
    interrupted_before_nodes: Literal[None] = None,
    is_resuming: Literal[False] = False
) -> tuple[Checkpoint, list[PregelTaskDescription]]:
    ...


@overload
def _prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    for_execution: Literal[True],
    get_next_version: Callable[[int, BaseChannel], int],
    manager: Union[None, ParentRunManager, AsyncParentRunManager],
    checkpointer: Optional[BaseCheckpointSaver],
    interrupted_before_nodes: Optional[Sequence[str]],
    is_resuming: bool = False
) -> tuple[Checkpoint, list[PregelExecutableTask]]:
    ...


def _prepare_next_tasks(
    checkpoint: Checkpoint,
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    *,
    for_execution: bool,
    get_next_version: Union[None, Callable[[int, BaseChannel], int]] = None,
    manager: Union[None, ParentRunManager, AsyncParentRunManager] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    is_resuming: bool = False
) -> tuple[Checkpoint, Union[list[PregelTaskDescription], list[PregelExecutableTask]]]:
    checkpoint = copy_checkpoint(checkpoint)
    tasks: Union[list[PregelTaskDescription], list[PregelExecutableTask]] = []
    # Consume pending packets
    for packet in checkpoint["pending_sends"]:
        if not isinstance(packet, Send):
            logger.warn(f"Ignoring invalid packet type {type(packet)} in pending sends")
            continue
        if for_execution:
            proc = processes[packet.node]
            if node := proc.get_node():
                triggers = [TASKS]
                metadata = {
                    "langgraph_step": step,
                    "langgraph_node": packet.node,
                    "langgraph_triggers": triggers,
                    "langgraph_task_idx": len(tasks),
                }
                task_id = str(uuid5(UUID(checkpoint["id"]), json.dumps(metadata)))
                # in Send we can't checkpoint nested graphs
                # as they could be running in parallel
                writes = deque()
                tasks.append(
                    PregelExecutableTask(
                        packet.node,
                        packet.arg,
                        node,
                        writes,
                        patch_config(
                            merge_configs(
                                config,
                                processes[packet.node].config,
                                {"metadata": metadata},
                            ),
                            run_name=packet.node,
                            callbacks=(
                                manager.get_child(f"graph:step:{step}")
                                if manager
                                else None
                            ),
                            configurable={
                                # deque.extend is thread-safe
                                CONFIG_KEY_SEND: partial(
                                    _local_write, writes.extend, processes, channels
                                ),
                                CONFIG_KEY_READ: partial(
                                    _local_read, checkpoint, channels, writes, config
                                ),
                            },
                        ),
                        triggers,
                        proc.retry_policy,
                        task_id,
                    )
                )
        else:
            tasks.append(PregelTaskDescription(packet.node, packet.arg))
    if for_execution:
        checkpoint["pending_sends"].clear()
    # Collect channels to consume
    channels_to_consume = set()
    # Check if any processes should be run in next step
    # If so, prepare the values to be passed to them
    version_type = type(next(iter(checkpoint["channel_versions"].values()), None))
    null_version = version_type()
    if null_version is None:
        return checkpoint, tasks
    for name, proc in processes.items():
        seen = checkpoint["versions_seen"][name]
        # If any of the channels read by this process were updated
        if triggers := sorted(
            chan
            for chan in proc.triggers
            if not isinstance(
                read_channel(channels, chan, return_exception=True), EmptyChannelError
            )
            and checkpoint["channel_versions"].get(chan, null_version)
            > seen.get(chan, null_version)
        ):
            channels_to_consume.update(triggers)
            try:
                val = next(
                    _proc_input(
                        step,
                        name,
                        proc,
                        managed,
                        channels,
                    )
                )
            except StopIteration:
                continue

            # update seen versions
            if for_execution:
                seen.update(
                    {
                        chan: checkpoint["channel_versions"][chan]
                        for chan in proc.triggers
                        if chan in checkpoint["channel_versions"]
                    }
                )

            if for_execution:
                if node := proc.get_node():
                    metadata = {
                        "langgraph_step": step,
                        "langgraph_node": name,
                        "langgraph_triggers": triggers,
                        "langgraph_task_idx": len(tasks),
                    }
                    task_id = str(uuid5(UUID(checkpoint["id"]), json.dumps(metadata)))
                    if parent_thread_id := config.get("configurable", {}).get(
                        "thread_id"
                    ):
                        thread_id: Optional[str] = f"{parent_thread_id}-{name}"
                    else:
                        thread_id = None
                    writes = deque()
                    tasks.append(
                        PregelExecutableTask(
                            name,
                            val,
                            node,
                            writes,
                            patch_config(
                                merge_configs(
                                    config,
                                    proc.config,
                                    {"metadata": metadata},
                                ),
                                run_name=name,
                                callbacks=(
                                    manager.get_child(f"graph:step:{step}")
                                    if manager
                                    else None
                                ),
                                configurable={
                                    # deque.extend is thread-safe
                                    CONFIG_KEY_SEND: partial(
                                        _local_write, writes.extend, processes, channels
                                    ),
                                    CONFIG_KEY_READ: partial(
                                        _local_read,
                                        checkpoint,
                                        channels,
                                        writes,
                                        config,
                                    ),
                                    CONFIG_KEY_CHECKPOINTER: checkpointer,
                                    CONFIG_KEY_RESUMING: is_resuming,
                                    "thread_id": thread_id,
                                },
                            ),
                            triggers,
                            proc.retry_policy,
                            task_id,
                        )
                    )
            else:
                tasks.append(PregelTaskDescription(name, val))
    # Find the highest version of all channels
    if checkpoint["channel_versions"]:
        max_version = max(checkpoint["channel_versions"].values())
    else:
        max_version = None
    # Consume all channels that were read
    if for_execution:
        for chan in channels_to_consume:
            if channels[chan].consume():
                checkpoint["channel_versions"][chan] = get_next_version(
                    max_version, channels[chan]
                )
    return checkpoint, tasks


def _is_subgraph_proc(proc: PregelNode) -> bool:
    node = proc.get_node()
    if not isinstance(node, RunnableSequence):
        return False

    return isinstance(node.steps[0], Pregel)


def _proc_input(
    step: int,
    name: str,
    proc: PregelNode,
    managed: ManagedValueMapping,
    channels: Mapping[str, BaseChannel],
    should_continue_from_interrupt: bool = False,
) -> Iterator[Any]:
    if _is_subgraph_proc(proc) and should_continue_from_interrupt:
        yield None

    # If all trigger channels subscribed by this process are not empty
    # then invoke the process with the values of all non-empty channels
    if isinstance(proc.channels, dict):
        try:
            val: dict = {
                k: read_channel(
                    channels,
                    chan,
                    catch=chan not in proc.triggers,
                )
                for k, chan in proc.channels.items()
                if isinstance(chan, str)
            }

            managed_values = {}
            for key, chan in proc.channels.items():
                if is_managed_value(chan):
                    managed_values[key] = managed[key](
                        step, PregelTaskDescription(name, val)
                    )

            val.update(managed_values)
        except EmptyChannelError:
            return
    elif isinstance(proc.channels, list):
        for chan in proc.channels:
            try:
                val = read_channel(channels, chan, catch=False)
                break
            except EmptyChannelError:
                pass
        else:
            return
    else:
        raise RuntimeError(
            "Invalid channels type, expected list or dict, got {proc.channels}"
        )

    # If the process has a mapper, apply it to the value
    if proc.mapper is not None:
        val = proc.mapper(val)

    yield val


def _with_mode(mode: StreamMode, on: bool, iter: Iterator[Any]) -> Iterator[Any]:
    if on:
        for chunk in iter:
            yield (mode, chunk)
    else:
        yield from iter
