from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections import deque
from functools import partial
from inspect import signature
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
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
)
from langgraph.channels.context import Context
from langgraph.channels.last_value import LastValue
from langgraph.channels.manager import (
    AsyncChannelsManager,
    ChannelsManager,
)
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
    copy_checkpoint,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.constants import (
    CHECKPOINT_NAMESPACE_SEPARATOR,
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_READ,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_SEND,
    ERROR,
    INTERRUPT,
    SEND_CHECKPOINT_NAMESPACE_SEPARATOR,
    Interrupt,
)
from langgraph.errors import GraphInterrupt, GraphRecursionError, InvalidUpdateError
from langgraph.managed.base import (
    AsyncManagedValuesManager,
    ManagedValuesManager,
    ManagedValueSpec,
    is_managed_value,
)
from langgraph.pregel.algo import (
    apply_writes,
    local_read,
    prepare_next_tasks,
    should_interrupt,
)
from langgraph.pregel.debug import (
    map_debug_task_results,
    print_step_checkpoint,
    print_step_tasks,
    print_step_writes,
    tasks_w_writes,
)
from langgraph.pregel.io import (
    map_output_updates,
    read_channels,
)
from langgraph.pregel.loop import AsyncPregelLoop, SyncPregelLoop
from langgraph.pregel.read import PregelNode
from langgraph.pregel.retry import RetryPolicy, arun_with_retry, run_with_retry
from langgraph.pregel.types import (
    All,
    PregelExecutableTask,
    StateSnapshot,
    StreamMode,
)
from langgraph.pregel.utils import get_new_channel_versions
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


def _get_subgraph(graph: Pregel, checkpoint_ns: str) -> Pregel:
    if checkpoint_ns == "":
        return graph

    path = checkpoint_ns.split(CHECKPOINT_NAMESPACE_SEPARATOR)
    nodes = graph.nodes
    for subgraph_node_name in path:
        # if we have this separator it means we have a node that was triggered by Send
        if SEND_CHECKPOINT_NAMESPACE_SEPARATOR in subgraph_node_name:
            name_parts = subgraph_node_name.split(SEND_CHECKPOINT_NAMESPACE_SEPARATOR)
            if len(name_parts) != 2:
                raise ValueError(f"Malformed node name '{subgraph_node_name}'")

            subgraph_node_name = name_parts[0]
        if subgraph_node_name not in nodes:
            raise ValueError(f"Couldn't find node '{subgraph_node_name}'.")

        subgraph_node = nodes[subgraph_node_name]
        if isinstance(subgraph_node.bound, Pregel):
            nodes = subgraph_node.bound.nodes
        elif isinstance(subgraph_node.bound, RunnableSequence):
            for runnable in subgraph_node.bound.steps:
                if isinstance(runnable, Pregel):
                    nodes = runnable.nodes
                    break
        else:
            continue
    return subgraph_node.bound


def _assemble_state_snapshot_hierarchy(
    root_checkpoint_ns: str,
    checkpoint_ns_to_state_snapshots: dict[str, StateSnapshot],
) -> StateSnapshot:
    checkpoint_ns_list_to_visit = sorted(
        checkpoint_ns_to_state_snapshots.keys(),
        key=lambda x: len(x.split(CHECKPOINT_NAMESPACE_SEPARATOR)),
    )
    while checkpoint_ns_list_to_visit:
        checkpoint_ns = checkpoint_ns_list_to_visit.pop()
        state_snapshot = checkpoint_ns_to_state_snapshots[checkpoint_ns]
        *path, subgraph_node = checkpoint_ns.split(CHECKPOINT_NAMESPACE_SEPARATOR)
        parent_checkpoint_ns = CHECKPOINT_NAMESPACE_SEPARATOR.join(path)
        if subgraph_node and (
            parent_state_snapshot := checkpoint_ns_to_state_snapshots.get(
                parent_checkpoint_ns
            )
        ):
            parent_subgraph_snapshots = {
                **(parent_state_snapshot.subgraph_state_snapshots or {}),
                subgraph_node: state_snapshot,
            }
            checkpoint_ns_to_state_snapshots[
                parent_checkpoint_ns
            ] = checkpoint_ns_to_state_snapshots[parent_checkpoint_ns]._replace(
                subgraph_state_snapshots=parent_subgraph_snapshots
            )

    state_snapshot = checkpoint_ns_to_state_snapshots.pop(root_checkpoint_ns, None)
    if state_snapshot is None:
        raise ValueError(f"Missing checkpoint for checkpoint NS '{root_checkpoint_ns}'")
    return state_snapshot


def _prepare_state_snapshot(
    saved: CheckpointTuple,
    graph: Pregel,
) -> StateSnapshot:
    with ChannelsManager(
        {
            k: LastValue(None) if isinstance(c, Context) else c
            for k, c in graph.channels.items()
        },
        saved.checkpoint,
        saved.config,
    ) as channels, ManagedValuesManager(
        graph.managed_values_dict, ensure_config(saved.config)
    ) as managed:
        next_tasks = prepare_next_tasks(
            saved.checkpoint,
            graph.nodes,
            channels,
            managed,
            saved.config,
            saved.metadata.get("step", -1) + 1,
            for_execution=False,
        )
        return StateSnapshot(
            values=read_channels(channels, graph.stream_channels_asis),
            next=tuple(t.name for t in next_tasks),
            config=saved.config,
            metadata=saved.metadata,
            created_at=saved.checkpoint["ts"],
            parent_config=saved.parent_config,
            tasks=tasks_w_writes(next_tasks, saved.pending_writes),
        )


async def _prepare_state_snapshot_async(
    saved: CheckpointTuple, graph: Pregel
) -> StateSnapshot:
    async with AsyncChannelsManager(
        {
            k: LastValue(None) if isinstance(c, Context) else c
            for k, c in graph.channels.items()
        },
        saved.checkpoint,
        saved.config,
    ) as channels, AsyncManagedValuesManager(
        graph.managed_values_dict, ensure_config(saved.config)
    ) as managed:
        next_tasks = prepare_next_tasks(
            saved.checkpoint,
            graph.nodes,
            channels,
            managed,
            saved.config,
            saved.metadata.get("step", -1) + 1,
            for_execution=False,
        )
        return StateSnapshot(
            values=read_channels(channels, graph.stream_channels_asis),
            next=tuple(t.name for t in next_tasks),
            config=saved.config,
            metadata=saved.metadata,
            created_at=saved.checkpoint["ts"],
            parent_config=saved.parent_config,
            tasks=tasks_w_writes(next_tasks, saved.pending_writes),
        )


def _has_nested_interrupts(
    graph: Pregel,
) -> bool:
    for child in graph.subgraphs:
        if child.interrupt_after_nodes or child.interrupt_before_nodes:
            return True
    else:
        return False


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
            if spec.id
            not in [
                CONFIG_KEY_READ,
                CONFIG_KEY_SEND,
                CONFIG_KEY_CHECKPOINTER,
                CONFIG_KEY_RESUMING,
            ]
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

    @property
    def subgraphs(self) -> Iterator[Pregel]:
        for node in self.nodes.values():
            if isinstance(node.bound, Pregel):
                yield node.bound
                yield from node.bound.subgraphs
            elif isinstance(node.bound, RunnableSequence):
                for runnable in node.bound.steps:
                    if isinstance(runnable, Pregel):
                        yield runnable
                        yield from runnable.subgraphs

    def get_state(self, config: RunnableConfig) -> StateSnapshot:
        """Get the current state of the graph."""
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        checkpoint_tuple = self.checkpointer.get_tuple(config)
        checkpoint_config = checkpoint_tuple.config if checkpoint_tuple else config
        checkpoint_ns = checkpoint_config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint_config["configurable"].get("checkpoint_id")
        checkpoint_ns_to_checkpoint_id: dict[str, str] = {}
        checkpoint_ns_to_state_snapshots: dict[str, StateSnapshot] = {}
        checkpoint_ns_to_graph: dict[str, Pregel] = {}
        for checkpoint_tuple in self.checkpointer.list(config):
            saved_checkpoint_ns = checkpoint_tuple.config["configurable"][
                "checkpoint_ns"
            ]
            saved_checkpoint_id = checkpoint_tuple.config["configurable"][
                "checkpoint_id"
            ]
            if checkpoint_id != saved_checkpoint_id:
                continue

            existing_checkpoint_id = checkpoint_ns_to_checkpoint_id.get(
                saved_checkpoint_ns
            )
            # keep only most recent checkpoint_id
            if (
                existing_checkpoint_id is None
                or saved_checkpoint_id > existing_checkpoint_id
            ):
                if saved_checkpoint_ns not in checkpoint_ns_to_graph:
                    checkpoint_ns_to_graph[saved_checkpoint_ns] = _get_subgraph(
                        self, saved_checkpoint_ns
                    )

                state_snapshot = _prepare_state_snapshot(
                    checkpoint_tuple,
                    checkpoint_ns_to_graph[saved_checkpoint_ns],
                )
                checkpoint_ns_to_state_snapshots[saved_checkpoint_ns] = state_snapshot
                checkpoint_ns_to_checkpoint_id[
                    saved_checkpoint_ns
                ] = saved_checkpoint_id

        if not checkpoint_ns_to_state_snapshots:
            return StateSnapshot(
                values={}, next=(), config=config, metadata=None, created_at=None, tasks=()
            )

        state_snapshot = _assemble_state_snapshot_hierarchy(
            checkpoint_ns, checkpoint_ns_to_state_snapshots
        )
        return state_snapshot

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        """Get the current state of the graph."""
        if not self.checkpointer:
            raise ValueError("No checkpointer set")

        checkpoint_tuple = await self.checkpointer.aget_tuple(config)
        checkpoint_config = checkpoint_tuple.config if checkpoint_tuple else config
        checkpoint_ns = checkpoint_config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint_config["configurable"].get("checkpoint_id")
        checkpoint_ns_to_checkpoint_id: dict[str, str] = {}
        checkpoint_ns_to_state_snapshots: dict[str, StateSnapshot] = {}
        checkpoint_ns_to_graph: dict[str, Pregel] = {}
        async for checkpoint_tuple in self.checkpointer.alist(config):
            saved_checkpoint_ns = checkpoint_tuple.config["configurable"][
                "checkpoint_ns"
            ]
            saved_checkpoint_id = checkpoint_tuple.config["configurable"][
                "checkpoint_id"
            ]
            if checkpoint_id != saved_checkpoint_id:
                continue

            existing_checkpoint_id = checkpoint_ns_to_checkpoint_id.get(
                saved_checkpoint_ns
            )
            # keep only most recent checkpoint_id
            if (
                existing_checkpoint_id is None
                or saved_checkpoint_id > existing_checkpoint_id
            ):
                if saved_checkpoint_ns not in checkpoint_ns_to_graph:
                    checkpoint_ns_to_graph[saved_checkpoint_ns] = _get_subgraph(
                        self, saved_checkpoint_ns
                    )

                state_snapshot = await _prepare_state_snapshot_async(
                    checkpoint_tuple,
                    checkpoint_ns_to_graph[saved_checkpoint_ns],
                )
                checkpoint_ns_to_state_snapshots[saved_checkpoint_ns] = state_snapshot
                checkpoint_ns_to_checkpoint_id[
                    saved_checkpoint_ns
                ] = saved_checkpoint_id

        if not checkpoint_ns_to_state_snapshots:
            return StateSnapshot(
                values={},
                next=(),
                config=config,
                metadata=None,
                created_at=None,
                tasks=(),
            )

        state_snapshot = _assemble_state_snapshot_hierarchy(
            checkpoint_ns, checkpoint_ns_to_state_snapshots
        )
        return state_snapshot

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

        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        for checkpoint_tuple in self.checkpointer.list(
            config, before=before, limit=limit, filter=filter
        ):
            if (
                checkpoint_tuple.config["configurable"]["checkpoint_ns"]
                != checkpoint_ns
            ):
                # only list root checkpoints here
                continue

            state_snapshot = self.get_state(checkpoint_tuple.config)
            yield state_snapshot

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

        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        async for checkpoint_tuple in self.checkpointer.alist(
            config, before=before, limit=limit, filter=filter
        ):
            if (
                checkpoint_tuple.config["configurable"]["checkpoint_ns"]
                != checkpoint_ns
            ):
                # only list root checkpoints here
                continue

            state_snapshot = await self.aget_state(checkpoint_tuple.config)
            yield state_snapshot

    def update_state(
        self,
        config: RunnableConfig,
        values: Optional[Union[dict[str, Any], Any]],
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
        checkpoint_previous_versions = (
            saved.checkpoint["channel_versions"].copy() if saved else {}
        )
        step = saved.metadata.get("step", -1) if saved else -1
        # merge configurable fields with previous checkpoint config
        checkpoint_config = {
            **config,
            "configurable": {
                **config["configurable"],
                # TODO: add proper support for updating nested subgraph state
                "checkpoint_ns": "",
            },
        }
        if saved:
            checkpoint_config = {
                "configurable": {
                    **config.get("configurable", {}),
                    **saved.config["configurable"],
                }
            }
        # find last node that updated the state, if not provided
        if values is None and as_node is None:
            return self.checkpointer.put(
                checkpoint_config,
                create_checkpoint(checkpoint, None, step),
                {
                    "source": "update",
                    "step": step + 1,
                    "writes": {},
                },
                {},
            )
        elif as_node is None and not any(
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
        with ChannelsManager(
            self.channels, checkpoint, config
        ) as channels, ManagedValuesManager(
            self.managed_values_dict, ensure_config(config)
        ) as managed:
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
                            local_read, checkpoint, channels, task, config
                        ),
                    },
                ),
            )
            # apply to checkpoint and save
            apply_writes(
                checkpoint, channels, [task], self.checkpointer.get_next_version
            )
            checkpoint = create_checkpoint(checkpoint, channels, step + 1)
            # check interrupt before
            if tasks := should_interrupt(
                checkpoint,
                self.interrupt_before_nodes,
                prepare_next_tasks(
                    checkpoint,
                    self.nodes,
                    channels,
                    managed,
                    config,
                    step + 2,
                    for_execution=False,
                ),
            ):
                for t in tasks:
                    self.checkpointer.put_writes(
                        {
                            "configurable": {
                                **checkpoint_config["configurable"],
                                "checkpoint_id": checkpoint["id"],
                            }
                        },
                        [(INTERRUPT, Interrupt("before"))],
                        t.id,
                    )
            return self.checkpointer.put(
                checkpoint_config,
                checkpoint,
                {
                    "source": "update",
                    "step": step + 1,
                    "writes": {as_node: values},
                },
                get_new_channel_versions(
                    checkpoint_previous_versions, checkpoint["channel_versions"]
                ),
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
        checkpoint_previous_versions = (
            saved.checkpoint["channel_versions"].copy() if saved else {}
        )
        step = saved.metadata.get("step", -1) if saved else -1
        # merge configurable fields with previous checkpoint config
        checkpoint_config = {
            **config,
            "configurable": {
                **config["configurable"],
                # TODO: add proper support for updating nested subgraph state
                "checkpoint_ns": "",
            },
        }
        if saved:
            checkpoint_config = {
                "configurable": {
                    **config.get("configurable", {}),
                    **saved.config["configurable"],
                }
            }
        # find last node that updated the state, if not provided
        if values is None and as_node is None:
            return await self.checkpointer.aput(
                checkpoint_config,
                create_checkpoint(checkpoint, None, step),
                {
                    "source": "update",
                    "step": step + 1,
                    "writes": {},
                },
                {},
            )
        elif as_node is None and not saved:
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
        async with AsyncChannelsManager(
            self.channels, checkpoint, config
        ) as channels, AsyncManagedValuesManager(
            self.managed_values_dict, ensure_config(config)
        ) as managed:
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
                            local_read, checkpoint, channels, task, config
                        ),
                    },
                ),
            )
            # apply to checkpoint and save
            apply_writes(
                checkpoint, channels, [task], self.checkpointer.get_next_version
            )
            checkpoint = create_checkpoint(checkpoint, channels, step + 1)
            # check interrupt before
            if tasks := should_interrupt(
                checkpoint,
                self.interrupt_before_nodes,
                prepare_next_tasks(
                    checkpoint,
                    self.nodes,
                    channels,
                    managed,
                    config,
                    step + 2,
                    for_execution=False,
                ),
            ):
                await asyncio.gather(
                    *(
                        self.checkpointer.aput_writes(
                            {
                                "configurable": {
                                    **checkpoint_config["configurable"],
                                    "checkpoint_id": checkpoint["id"],
                                }
                            },
                            [(INTERRUPT, Interrupt("before"))],
                            t.id,
                        )
                        for t in tasks
                    )
                )
            return await self.checkpointer.aput(
                checkpoint_config,
                checkpoint,
                {
                    "source": "update",
                    "step": step + 1,
                    "writes": {as_node: values},
                },
                get_new_channel_versions(
                    checkpoint_previous_versions, checkpoint["channel_versions"]
                ),
            )

    def _defaults(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
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
        Optional[BaseCheckpointSaver],
    ]:
        debug = debug if debug is not None else self.debug
        if output_keys is None:
            output_keys = self.stream_channels_asis
        else:
            validate_keys(output_keys, self.channels)
        interrupt_before = interrupt_before or self.interrupt_before_nodes
        interrupt_after = interrupt_after or self.interrupt_after_nodes
        stream_mode = stream_mode if stream_mode is not None else self.stream_mode
        if not isinstance(stream_mode, list):
            stream_mode = [stream_mode]
        if config and config.get("configurable", {}).get(CONFIG_KEY_READ) is not None:
            # if being called as a node in another graph, always use values mode
            stream_mode = ["values"]
        if (
            config is not None
            and config.get("configurable", {}).get(CONFIG_KEY_CHECKPOINTER)
            and (interrupt_after or interrupt_before or _has_nested_interrupts(self))
        ):
            checkpointer: Optional[BaseCheckpointSaver] = config["configurable"][
                CONFIG_KEY_CHECKPOINTER
            ]
        else:
            checkpointer = self.checkpointer
        return (
            debug,
            stream_mode,
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
                output_keys,
                interrupt_before,
                interrupt_after,
                checkpointer,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
            )

            with SyncPregelLoop(
                input, config=config, checkpointer=checkpointer, graph=self
            ) as loop:
                # Similarly to Bulk Synchronous Parallel / Pregel model
                # computation proceeds in steps, while there are channel updates
                # channel updates from step N are only visible in step N+1
                # channels are guaranteed to be immutable for the duration of the step,
                # with channel updates applied only at the transition between steps
                while loop.tick(
                    output_keys=output_keys,
                    interrupt_before=interrupt_before,
                    interrupt_after=interrupt_after,
                    manager=run_manager,
                ):
                    # debug flag
                    if debug:
                        print_step_checkpoint(
                            loop.checkpoint_metadata,
                            loop.channels,
                            self.stream_channels_list,
                        )
                    # emit output
                    while loop.stream:
                        mode, payload = loop.stream.popleft()
                        if mode in stream_modes:
                            if isinstance(stream_mode, list):
                                yield (mode, payload)
                            else:
                                yield payload
                    # debug flag
                    if debug:
                        print_step_tasks(loop.step, loop.tasks)

                    # execute tasks, and wait for one to fail or all to finish.
                    # each task is independent from all other concurrent tasks
                    # yield updates/debug output as each task finishes
                    futures = {
                        loop.submit(
                            run_with_retry,
                            task,
                            self.retry_policy,
                        ): task
                        for task in loop.tasks
                        if not task.writes
                    }
                    all_futures = futures.copy()
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
                        for fut, task in zip(done, [futures.pop(fut) for fut in done]):
                            if exc := _exception(fut):
                                # save error to checkpointer
                                if isinstance(exc, GraphInterrupt):
                                    loop.put_writes(
                                        task.id, [(INTERRUPT, i) for i in exc.args[0]]
                                    )
                                else:
                                    loop.put_writes(task.id, [(ERROR, exc)])

                                futures.clear()

                            else:
                                # save task writes to checkpointer
                                loop.put_writes(task.id, task.writes)
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
                                            loop.step,
                                            [task],
                                            self.stream_channels_list,
                                        ),
                                    )
                        else:
                            # remove references to loop vars
                            del fut, task
                        if _should_stop_others(done):
                            break

                    # panic on failure or timeout
                    _panic_or_proceed(all_futures, loop.step)
                    # don't keep futures around in memory longer than needed
                    del done, inflight, futures
                    # debug flag
                    if debug:
                        print_step_writes(
                            loop.step,
                            [w for t in loop.tasks for w in t.writes],
                            self.stream_channels_list,
                        )
                # emit output
                while loop.stream:
                    mode, payload = loop.stream.popleft()
                    if mode in stream_modes:
                        if isinstance(stream_mode, list):
                            yield (mode, payload)
                        else:
                            yield payload
                # handle exit
                if loop.status == "out_of_steps":
                    raise GraphRecursionError(
                        f"Recursion limit of {config['recursion_limit']} reached "
                        "without hitting a stop condition. You can increase the "
                        "limit by setting the `recursion_limit` config key."
                    )
                # set final channel values as run output
                run_manager.on_chain_end(read_channels(loop.channels, output_keys))
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
                output_keys,
                interrupt_before,
                interrupt_after,
                checkpointer,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
            )
            async with AsyncPregelLoop(
                input, config=config, checkpointer=checkpointer, graph=self
            ) as loop:
                aioloop = asyncio.get_event_loop()
                # Similarly to Bulk Synchronous Parallel / Pregel model
                # computation proceeds in steps, while there are channel updates
                # channel updates from step N are only visible in step N+1
                # channels are guaranteed to be immutable for the duration of the step,
                # with channel updates applied only at the transition between steps
                while loop.tick(
                    output_keys=output_keys,
                    interrupt_before=interrupt_before,
                    interrupt_after=interrupt_after,
                    manager=run_manager,
                ):
                    # debug flag
                    if debug:
                        print_step_checkpoint(
                            loop.checkpoint_metadata,
                            loop.channels,
                            self.stream_channels_list,
                        )
                    # emit output
                    while loop.stream:
                        mode, payload = loop.stream.popleft()
                        if mode in stream_modes:
                            if isinstance(stream_mode, list):
                                yield (mode, payload)
                            else:
                                yield payload
                    # debug flag
                    if debug:
                        print_step_tasks(loop.step, loop.tasks)

                    # execute tasks, and wait for one to fail or all to finish.
                    # each task is independent from all other concurrent tasks
                    # yield updates/debug output as each task finishes
                    futures = {
                        loop.submit(
                            arun_with_retry,
                            task,
                            self.retry_policy,
                            stream=do_stream,
                            __name__=task.name,
                            __cancel_on_exit__=True,
                        ): task
                        for task in loop.tasks
                        if not task.writes
                    }
                    all_futures = futures.copy()
                    end_time = (
                        self.step_timeout + aioloop.time()
                        if self.step_timeout
                        else None
                    )
                    if not futures:
                        done, inflight = set(), set()
                    while futures:
                        done, inflight = await asyncio.wait(
                            futures,
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=(
                                max(0, end_time - aioloop.time()) if end_time else None
                            ),
                        )
                        if not done:
                            break  # timed out

                        for fut, task in zip(done, [futures.pop(fut) for fut in done]):
                            if exc := _exception(fut):
                                # save error to checkpointer
                                if isinstance(exc, GraphInterrupt):
                                    loop.put_writes(
                                        task.id, [(INTERRUPT, i) for i in exc.args[0]]
                                    )
                                else:
                                    loop.put_writes(task.id, [(ERROR, exc)])

                                futures.clear()
                            else:
                                # save task writes to checkpointer
                                loop.put_writes(task.id, task.writes)
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
                                            loop.step,
                                            [task],
                                            self.stream_channels_list,
                                        ),
                                    ):
                                        yield chunk
                        else:
                            # remove references to loop vars
                            del fut, task
                        if _should_stop_others(done):
                            break

                    # panic on failure or timeout
                    _panic_or_proceed(all_futures, loop.step, asyncio.TimeoutError)
                    # don't keep futures around in memory longer than needed
                    del done, inflight, futures
                    # debug flag
                    if debug:
                        print_step_writes(
                            loop.step,
                            [w for t in loop.tasks for w in t.writes],
                            self.stream_channels_list,
                        )
                # emit output
                while loop.stream:
                    mode, payload = loop.stream.popleft()
                    if mode in stream_modes:
                        if isinstance(stream_mode, list):
                            yield (mode, payload)
                        else:
                            yield payload
                # handle exit
                if loop.status == "out_of_steps":
                    raise GraphRecursionError(
                        f"Recursion limit of {config['recursion_limit']} reached "
                        "without hitting a stop condition. You can increase the "
                        "limit by setting the `recursion_limit` config key."
                    )

                # set final channel values as run output
                await run_manager.on_chain_end(
                    read_channels(loop.channels, output_keys)
                )
        except BaseException as e:
            # TODO use on_chain_end if exc is GraphInterrupt
            await asyncio.shield(run_manager.on_chain_error(e))
            raise

    def invoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: StreamMode = "values",
        output_keys: Optional[Union[str, Sequence[str]]] = None,
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


def _should_stop_others(
    done: Union[set[concurrent.futures.Future[Any]], set[asyncio.Task[Any]]],
) -> bool:
    for fut in done:
        if fut.cancelled():
            return True
        if exc := fut.exception():
            return not isinstance(exc, GraphInterrupt)
    else:
        return False


def _exception(
    fut: Union[concurrent.futures.Future[Any], asyncio.Task[Any]],
) -> Optional[BaseException]:
    if fut.cancelled():
        if isinstance(fut, asyncio.Task):
            return asyncio.CancelledError()
        else:
            return concurrent.futures.CancelledError()
    else:
        return fut.exception()


def _panic_or_proceed(
    futs: Union[set[concurrent.futures.Future[Any]], set[asyncio.Task[Any]]],
    step: int,
    timeout_exc_cls: Type[Exception] = TimeoutError,
) -> None:
    done: set[Union[concurrent.futures.Future[Any], asyncio.Task[Any]]] = set()
    inflight: set[Union[concurrent.futures.Future[Any], asyncio.Task[Any]]] = set()
    for fut in futs:
        if fut.done():
            done.add(fut)
        else:
            inflight.add(fut)
    while done:
        # if any task failed
        if exc := _exception(done.pop()):
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


def _with_mode(mode: StreamMode, on: bool, iter: Iterator[Any]) -> Iterator[Any]:
    if on:
        for chunk in iter:
            yield (mode, chunk)
    else:
        yield from iter
