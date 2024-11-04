from __future__ import annotations

import asyncio
import concurrent
import concurrent.futures
import queue
from collections import deque
from functools import partial
from typing import (
    Any,
    AsyncIterator,
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
from langchain_core.runnables import (
    RunnableSequence,
)
from langchain_core.runnables.base import Input, Output
from langchain_core.runnables.config import (
    RunnableConfig,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
)
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    get_unique_config_specs,
)
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from pydantic import BaseModel
from typing_extensions import Self

from langgraph.channels.base import (
    BaseChannel,
)
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
    copy_checkpoint,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_NODE_FINISHED,
    CONFIG_KEY_READ,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_SEND,
    CONFIG_KEY_STORE,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_STREAM_WRITER,
    CONFIG_KEY_TASK_ID,
    INTERRUPT,
    NS_END,
    NS_SEP,
)
from langgraph.errors import (
    ErrorCode,
    GraphRecursionError,
    InvalidUpdateError,
    create_error_message,
)
from langgraph.managed.base import ManagedValueSpec
from langgraph.pregel.algo import (
    PregelTaskWrites,
    apply_writes,
    local_read,
    local_write,
    prepare_next_tasks,
)
from langgraph.pregel.debug import tasks_w_writes
from langgraph.pregel.io import read_channels
from langgraph.pregel.loop import AsyncPregelLoop, StreamProtocol, SyncPregelLoop
from langgraph.pregel.manager import AsyncChannelsManager, ChannelsManager
from langgraph.pregel.messages import StreamMessagesHandler
from langgraph.pregel.protocol import PregelProtocol
from langgraph.pregel.read import PregelNode
from langgraph.pregel.retry import RetryPolicy
from langgraph.pregel.runner import PregelRunner
from langgraph.pregel.utils import find_subgraph_pregel, get_new_channel_versions
from langgraph.pregel.validate import validate_graph, validate_keys
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
from langgraph.store.base import BaseStore
from langgraph.types import All, Checkpointer, LoopProtocol, StateSnapshot, StreamMode
from langgraph.utils.config import (
    ensure_config,
    merge_configs,
    patch_checkpoint_map,
    patch_config,
    patch_configurable,
)
from langgraph.utils.pydantic import create_model
from langgraph.utils.queue import AsyncQueue, SyncQueue  # type: ignore[attr-defined]

WriteValue = Union[Callable[[Input], Output], Any]


class Channel:
    @overload
    @classmethod
    def subscribe_to(
        cls,
        channels: str,
        *,
        key: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> PregelNode: ...

    @overload
    @classmethod
    def subscribe_to(
        cls,
        channels: Sequence[str],
        *,
        key: None = None,
        tags: Optional[list[str]] = None,
    ) -> PregelNode: ...

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
                Union[list[str], Mapping[str, str]],
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
                    ChannelWriteEntry(k, mapper=v)
                    if callable(v)
                    else ChannelWriteEntry(k, value=v)
                )
                for k, v in kwargs.items()
            ]
        )


class Pregel(PregelProtocol):
    nodes: dict[str, PregelNode]

    channels: dict[str, Union[BaseChannel, ManagedValueSpec]]

    stream_mode: StreamMode = "values"
    """Mode to stream output, defaults to 'values'."""

    output_channels: Union[str, Sequence[str]]

    stream_channels: Optional[Union[str, Sequence[str]]] = None
    """Channels to stream, defaults to all channels not in reserved channels"""

    interrupt_after_nodes: Union[All, Sequence[str]]

    interrupt_before_nodes: Union[All, Sequence[str]]

    input_channels: Union[str, Sequence[str]]

    step_timeout: Optional[float] = None
    """Maximum time to wait for a step to complete, in seconds. Defaults to None."""

    debug: bool
    """Whether to print debug information during execution. Defaults to False."""

    checkpointer: Checkpointer = None
    """Checkpointer used to save and load graph state. Defaults to None."""

    store: Optional[BaseStore] = None
    """Memory store to use for SharedValues. Defaults to None."""

    retry_policy: Optional[RetryPolicy] = None
    """Retry policy to use when running tasks. Set to None to disable."""

    config_type: Optional[Type[Any]] = None

    config: Optional[RunnableConfig] = None

    name: str = "LangGraph"

    def __init__(
        self,
        *,
        nodes: dict[str, PregelNode],
        channels: Optional[dict[str, Union[BaseChannel, ManagedValueSpec]]],
        auto_validate: bool = True,
        stream_mode: StreamMode = "values",
        output_channels: Union[str, Sequence[str]],
        stream_channels: Optional[Union[str, Sequence[str]]] = None,
        interrupt_after_nodes: Union[All, Sequence[str]] = (),
        interrupt_before_nodes: Union[All, Sequence[str]] = (),
        input_channels: Union[str, Sequence[str]],
        step_timeout: Optional[float] = None,
        debug: Optional[bool] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        store: Optional[BaseStore] = None,
        retry_policy: Optional[RetryPolicy] = None,
        config_type: Optional[Type[Any]] = None,
        config: Optional[RunnableConfig] = None,
        name: str = "LangGraph",
    ) -> None:
        self.nodes = nodes
        self.channels = channels or {}
        self.stream_mode = stream_mode
        self.output_channels = output_channels
        self.stream_channels = stream_channels
        self.interrupt_after_nodes = interrupt_after_nodes
        self.interrupt_before_nodes = interrupt_before_nodes
        self.input_channels = input_channels
        self.step_timeout = step_timeout
        self.debug = debug if debug is not None else get_debug()
        self.checkpointer = checkpointer
        self.store = store
        self.retry_policy = retry_policy
        self.config_type = config_type
        self.config = config
        self.name = name
        if auto_validate:
            self.validate()

    def get_graph(
        self, config: RunnableConfig | None = None, *, xray: int | bool = False
    ) -> Graph:
        raise NotImplementedError

    async def aget_graph(
        self, config: RunnableConfig | None = None, *, xray: int | bool = False
    ) -> Graph:
        raise NotImplementedError

    def copy(self, update: dict[str, Any] | None = None) -> Self:
        attrs = {**self.__dict__, **(update or {})}
        return self.__class__(**attrs)

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Self:
        return self.copy(
            {"config": merge_configs(self.config, config, cast(RunnableConfig, kwargs))}
        )

    def validate(self) -> Self:
        validate_graph(
            self.nodes,
            {k: v for k, v in self.channels.items() if isinstance(v, BaseChannel)},
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
                    if isinstance(self.checkpointer, BaseCheckpointSaver)
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
            channel = self.channels[self.input_channels]
            if isinstance(channel, BaseChannel):
                return channel.UpdateType

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        config = merge_configs(self.config, config)
        if isinstance(self.input_channels, str):
            return super().get_input_schema(config)
        else:
            return create_model(
                self.get_name("Input"),
                field_definitions={
                    k: (c.UpdateType, None)
                    for k in self.input_channels or self.channels.keys()
                    if (c := self.channels[k]) and isinstance(c, BaseChannel)
                },
            )

    def get_input_jsonschema(
        self, config: Optional[RunnableConfig] = None
    ) -> Dict[All, Any]:
        schema = self.get_input_schema(config)
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        else:
            return schema.schema()

    @property
    def OutputType(self) -> Any:
        if isinstance(self.output_channels, str):
            channel = self.channels[self.output_channels]
            if isinstance(channel, BaseChannel):
                return channel.ValueType

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        config = merge_configs(self.config, config)
        if isinstance(self.output_channels, str):
            return super().get_output_schema(config)
        else:
            return create_model(
                self.get_name("Output"),
                field_definitions={
                    k: (c.ValueType, None)
                    for k in self.output_channels
                    if (c := self.channels[k]) and isinstance(c, BaseChannel)
                },
            )

    def get_output_jsonschema(
        self, config: Optional[RunnableConfig] = None
    ) -> Dict[All, Any]:
        schema = self.get_output_schema(config)
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        else:
            return schema.schema()

    @property
    def stream_channels_list(self) -> Sequence[str]:
        stream_channels = self.stream_channels_asis
        return (
            [stream_channels] if isinstance(stream_channels, str) else stream_channels
        )

    @property
    def stream_channels_asis(self) -> Union[str, Sequence[str]]:
        return self.stream_channels or [
            k for k in self.channels if isinstance(self.channels[k], BaseChannel)
        ]

    def get_subgraphs(
        self, *, namespace: Optional[str] = None, recurse: bool = False
    ) -> Iterator[tuple[str, Pregel]]:
        for name, node in self.nodes.items():
            # filter by prefix
            if namespace is not None:
                if not namespace.startswith(name):
                    continue

            # find the subgraph, if any
            graph = cast(Optional[Pregel], find_subgraph_pregel(node.bound))

            # if found, yield recursively
            if graph:
                if name == namespace:
                    yield name, graph
                    return  # we found it, stop searching
                if namespace is None:
                    yield name, graph
                if recurse:
                    if namespace is not None:
                        namespace = namespace[len(name) + 1 :]
                    yield from (
                        (f"{name}{NS_SEP}{n}", s)
                        for n, s in graph.get_subgraphs(
                            namespace=namespace, recurse=recurse
                        )
                    )

    async def aget_subgraphs(
        self, *, namespace: Optional[str] = None, recurse: bool = False
    ) -> AsyncIterator[tuple[str, Pregel]]:
        for name, node in self.get_subgraphs(namespace=namespace, recurse=recurse):
            yield name, node

    def _prepare_state_snapshot(
        self,
        config: RunnableConfig,
        saved: Optional[CheckpointTuple],
        recurse: Optional[BaseCheckpointSaver] = None,
    ) -> StateSnapshot:
        if not saved:
            return StateSnapshot(
                values={},
                next=(),
                config=config,
                metadata=None,
                created_at=None,
                parent_config=None,
                tasks=(),
            )

        with ChannelsManager(
            self.channels,
            saved.checkpoint,
            LoopProtocol(
                config=saved.config,
                step=saved.metadata.get("step", -1) + 1,
                stop=saved.metadata.get("step", -1) + 2,
            ),
            skip_context=True,
        ) as (channels, managed):
            # tasks for this checkpoint
            next_tasks = prepare_next_tasks(
                saved.checkpoint,
                self.nodes,
                channels,
                managed,
                saved.config,
                saved.metadata.get("step", -1) + 1,
                for_execution=False,
            )
            # get the subgraphs
            subgraphs = dict(self.get_subgraphs())
            parent_ns = saved.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
            task_states: dict[str, Union[RunnableConfig, StateSnapshot]] = {}
            for task in next_tasks.values():
                if task.name not in subgraphs:
                    continue
                # assemble checkpoint_ns for this task
                task_ns = f"{task.name}{NS_END}{task.id}"
                if parent_ns:
                    task_ns = f"{parent_ns}{NS_SEP}{task_ns}"
                if not recurse:
                    # set config as signal that subgraph checkpoints exist
                    config = {
                        CONF: {
                            "thread_id": saved.config[CONF]["thread_id"],
                            CONFIG_KEY_CHECKPOINT_NS: task_ns,
                        }
                    }
                    task_states[task.id] = config
                else:
                    # get the state of the subgraph
                    config = {
                        CONF: {
                            CONFIG_KEY_CHECKPOINTER: recurse,
                            "thread_id": saved.config[CONF]["thread_id"],
                            CONFIG_KEY_CHECKPOINT_NS: task_ns,
                        }
                    }
                    task_states[task.id] = subgraphs[task.name].get_state(
                        config, subgraphs=True
                    )
            # assemble the state snapshot
            return StateSnapshot(
                read_channels(channels, self.stream_channels_asis),
                tuple(t.name for t in next_tasks.values()),
                patch_checkpoint_map(saved.config, saved.metadata),
                saved.metadata,
                saved.checkpoint["ts"],
                patch_checkpoint_map(saved.parent_config, saved.metadata),
                tasks_w_writes(
                    next_tasks.values(),
                    saved.pending_writes,
                    task_states,
                    self.stream_channels_asis,
                ),
            )

    async def _aprepare_state_snapshot(
        self,
        config: RunnableConfig,
        saved: Optional[CheckpointTuple],
        recurse: Optional[BaseCheckpointSaver] = None,
    ) -> StateSnapshot:
        if not saved:
            return StateSnapshot(
                values={},
                next=(),
                config=config,
                metadata=None,
                created_at=None,
                parent_config=None,
                tasks=(),
            )

        async with AsyncChannelsManager(
            self.channels,
            saved.checkpoint,
            LoopProtocol(
                config=saved.config,
                step=saved.metadata.get("step", -1) + 1,
                stop=saved.metadata.get("step", -1) + 2,
            ),
            skip_context=True,
        ) as (
            channels,
            managed,
        ):
            # tasks for this checkpoint
            next_tasks = prepare_next_tasks(
                saved.checkpoint,
                self.nodes,
                channels,
                managed,
                saved.config,
                saved.metadata.get("step", -1) + 1,
                for_execution=False,
            )
            # get the subgraphs
            subgraphs = {n: g async for n, g in self.aget_subgraphs()}
            parent_ns = saved.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
            task_states: dict[str, Union[RunnableConfig, StateSnapshot]] = {}
            for task in next_tasks.values():
                if task.name not in subgraphs:
                    continue
                # assemble checkpoint_ns for this task
                task_ns = f"{task.name}{NS_END}{task.id}"
                if parent_ns:
                    task_ns = f"{parent_ns}{NS_SEP}{task_ns}"
                if not recurse:
                    # set config as signal that subgraph checkpoints exist
                    config = {
                        CONF: {
                            "thread_id": saved.config[CONF]["thread_id"],
                            CONFIG_KEY_CHECKPOINT_NS: task_ns,
                        }
                    }
                    task_states[task.id] = config
                else:
                    # get the state of the subgraph
                    config = {
                        CONF: {
                            CONFIG_KEY_CHECKPOINTER: recurse,
                            "thread_id": saved.config[CONF]["thread_id"],
                            CONFIG_KEY_CHECKPOINT_NS: task_ns,
                        }
                    }
                    task_states[task.id] = await subgraphs[task.name].aget_state(
                        config, subgraphs=True
                    )
            # assemble the state snapshot
            return StateSnapshot(
                read_channels(channels, self.stream_channels_asis),
                tuple(t.name for t in next_tasks.values()),
                patch_checkpoint_map(saved.config, saved.metadata),
                saved.metadata,
                saved.checkpoint["ts"],
                patch_checkpoint_map(saved.parent_config, saved.metadata),
                tasks_w_writes(
                    next_tasks.values(),
                    saved.pending_writes,
                    task_states,
                    self.stream_channels_asis,
                ),
            )

    def get_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        """Get the current state of the graph."""
        checkpointer: Optional[BaseCheckpointSaver] = config[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # remove task_ids from checkpoint_ns
            recast_checkpoint_ns = NS_SEP.join(
                part.split(NS_END)[0] for part in checkpoint_ns.split(NS_SEP)
            )
            # find the subgraph with the matching name
            for _, pregel in self.get_subgraphs(
                namespace=recast_checkpoint_ns, recurse=True
            ):
                return pregel.get_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    subgraphs=subgraphs,
                )
            else:
                raise ValueError(f"Subgraph {recast_checkpoint_ns} not found")

        config = merge_configs(self.config, config) if self.config else config
        saved = checkpointer.get_tuple(config)
        return self._prepare_state_snapshot(
            config, saved, recurse=checkpointer if subgraphs else None
        )

    async def aget_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        """Get the current state of the graph."""
        checkpointer: Optional[BaseCheckpointSaver] = config[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # remove task_ids from checkpoint_ns
            recast_checkpoint_ns = NS_SEP.join(
                part.split(NS_END)[0] for part in checkpoint_ns.split(NS_SEP)
            )
            # find the subgraph with the matching name
            async for _, pregel in self.aget_subgraphs(
                namespace=recast_checkpoint_ns, recurse=True
            ):
                return await pregel.aget_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    subgraphs=subgraphs,
                )
            else:
                raise ValueError(f"Subgraph {recast_checkpoint_ns} not found")

        config = merge_configs(self.config, config) if self.config else config
        saved = await checkpointer.aget_tuple(config)
        return await self._aprepare_state_snapshot(
            config, saved, recurse=checkpointer if subgraphs else None
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
        checkpointer: Optional[BaseCheckpointSaver] = config[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # remove task_ids from checkpoint_ns
            recast_checkpoint_ns = NS_SEP.join(
                part.split(NS_END)[0] for part in checkpoint_ns.split(NS_SEP)
            )
            # find the subgraph with the matching name
            for _, pregel in self.get_subgraphs(
                namespace=recast_checkpoint_ns, recurse=True
            ):
                yield from pregel.get_state_history(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    filter=filter,
                    before=before,
                    limit=limit,
                )
                return
            else:
                raise ValueError(f"Subgraph {recast_checkpoint_ns} not found")

        config = merge_configs(
            self.config,
            config,
            {CONF: {CONFIG_KEY_CHECKPOINT_NS: checkpoint_ns}},
        )
        # eagerly consume list() to avoid holding up the db cursor
        for checkpoint_tuple in list(
            checkpointer.list(config, before=before, limit=limit, filter=filter)
        ):
            yield self._prepare_state_snapshot(
                checkpoint_tuple.config, checkpoint_tuple
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
        checkpointer: Optional[BaseCheckpointSaver] = config[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # remove task_ids from checkpoint_ns
            recast_checkpoint_ns = NS_SEP.join(
                part.split(NS_END)[0] for part in checkpoint_ns.split(NS_SEP)
            )
            # find the subgraph with the matching name
            async for _, pregel in self.aget_subgraphs(
                namespace=recast_checkpoint_ns, recurse=True
            ):
                async for state in pregel.aget_state_history(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    filter=filter,
                    before=before,
                    limit=limit,
                ):
                    yield state
                return
            else:
                raise ValueError(f"Subgraph {recast_checkpoint_ns} not found")

        config = merge_configs(
            self.config,
            config,
            {CONF: {CONFIG_KEY_CHECKPOINT_NS: checkpoint_ns}},
        )
        # eagerly consume list() to avoid holding up the db cursor
        for checkpoint_tuple in [
            c
            async for c in checkpointer.alist(
                config, before=before, limit=limit, filter=filter
            )
        ]:
            yield await self._aprepare_state_snapshot(
                checkpoint_tuple.config, checkpoint_tuple
            )

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
        checkpointer: Optional[BaseCheckpointSaver] = config[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        # delegate to subgraph
        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # remove task_ids from checkpoint_ns
            recast_checkpoint_ns = NS_SEP.join(
                part.split(NS_END)[0] for part in checkpoint_ns.split(NS_SEP)
            )
            # find the subgraph with the matching name
            for _, pregel in self.get_subgraphs(
                namespace=recast_checkpoint_ns, recurse=True
            ):
                return pregel.update_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    values,
                    as_node,
                )
            else:
                raise ValueError(f"Subgraph {recast_checkpoint_ns} not found")

        # get last checkpoint
        config = ensure_config(self.config, config)
        saved = checkpointer.get_tuple(config)
        checkpoint = copy_checkpoint(saved.checkpoint) if saved else empty_checkpoint()
        checkpoint_previous_versions = (
            saved.checkpoint["channel_versions"].copy() if saved else {}
        )
        step = saved.metadata.get("step", -1) if saved else -1
        # merge configurable fields with previous checkpoint config
        checkpoint_config = patch_configurable(
            config,
            {CONFIG_KEY_CHECKPOINT_NS: config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")},
        )
        checkpoint_metadata = config["metadata"]
        if saved:
            checkpoint_config = patch_configurable(config, saved.config[CONF])
            checkpoint_metadata = {**saved.metadata, **checkpoint_metadata}
        # find last node that updated the state, if not provided
        if values is None and as_node is None:
            next_config = checkpointer.put(
                checkpoint_config,
                create_checkpoint(checkpoint, None, step),
                {
                    **checkpoint_metadata,
                    "source": "update",
                    "step": step + 1,
                    "writes": {},
                    "parents": saved.metadata.get("parents", {}) if saved else {},
                },
                {},
            )
            return patch_checkpoint_map(next_config, saved.metadata if saved else None)
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
                if n in self.nodes
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
            self.channels,
            checkpoint,
            LoopProtocol(config=config, step=step + 1, stop=step + 2),
        ) as (
            channels,
            managed,
        ):
            # create task to run all writers of the chosen node
            writers = self.nodes[as_node].flat_writers
            if not writers:
                raise InvalidUpdateError(f"Node {as_node} has no writers")
            writes: deque[tuple[str, Any]] = deque()
            task = PregelTaskWrites(as_node, writes, [INTERRUPT])
            task_id = str(uuid5(UUID(checkpoint["id"]), INTERRUPT))
            run = RunnableSequence(*writers) if len(writers) > 1 else writers[0]
            # execute task
            run.invoke(
                values,
                patch_config(
                    config,
                    run_name=self.name + "UpdateState",
                    configurable={
                        # deque.extend is thread-safe
                        CONFIG_KEY_SEND: partial(
                            local_write,
                            writes.extend,
                            self.nodes.keys(),
                        ),
                        CONFIG_KEY_READ: partial(
                            local_read,
                            step + 1,
                            checkpoint,
                            channels,
                            managed,
                            task,
                            config,
                        ),
                    },
                ),
            )
            # save task writes
            if saved:
                checkpointer.put_writes(checkpoint_config, task.writes, task_id)
            # apply to checkpoint and save
            mv_writes = apply_writes(
                checkpoint, channels, [task], checkpointer.get_next_version
            )
            assert not mv_writes, "Can't write to SharedValues from update_state"
            checkpoint = create_checkpoint(checkpoint, channels, step + 1)
            next_config = checkpointer.put(
                checkpoint_config,
                checkpoint,
                {
                    **checkpoint_metadata,
                    "source": "update",
                    "step": step + 1,
                    "writes": {as_node: values},
                    "parents": saved.metadata.get("parents", {}) if saved else {},
                },
                get_new_channel_versions(
                    checkpoint_previous_versions, checkpoint["channel_versions"]
                ),
            )
            return patch_checkpoint_map(next_config, saved.metadata if saved else None)

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any,
        as_node: Optional[str] = None,
    ) -> RunnableConfig:
        checkpointer: Optional[BaseCheckpointSaver] = config[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        # delegate to subgraph
        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # remove task_ids from checkpoint_ns
            recast_checkpoint_ns = NS_SEP.join(
                part.split(NS_END)[0] for part in checkpoint_ns.split(NS_SEP)
            )
            # find the subgraph with the matching name
            async for _, pregel in self.aget_subgraphs(
                namespace=recast_checkpoint_ns, recurse=True
            ):
                return await pregel.aupdate_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    values,
                    as_node,
                )
            else:
                raise ValueError(f"Subgraph {recast_checkpoint_ns} not found")

        # get last checkpoint
        config = ensure_config(self.config, config)
        saved = await checkpointer.aget_tuple(config)
        checkpoint = copy_checkpoint(saved.checkpoint) if saved else empty_checkpoint()
        checkpoint_previous_versions = (
            saved.checkpoint["channel_versions"].copy() if saved else {}
        )
        step = saved.metadata.get("step", -1) if saved else -1
        # merge configurable fields with previous checkpoint config
        checkpoint_config = patch_configurable(
            config,
            {CONFIG_KEY_CHECKPOINT_NS: config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")},
        )
        checkpoint_metadata = config["metadata"]
        if saved:
            checkpoint_config = patch_configurable(config, saved.config[CONF])
            checkpoint_metadata = {**saved.metadata, **checkpoint_metadata}
        # find last node that updated the state, if not provided
        if values is None and as_node is None:
            next_config = await checkpointer.aput(
                checkpoint_config,
                create_checkpoint(checkpoint, None, step),
                {
                    **checkpoint_metadata,
                    "source": "update",
                    "step": step + 1,
                    "writes": {},
                    "parents": saved.metadata.get("parents", {}) if saved else {},
                },
                {},
            )
            return patch_checkpoint_map(next_config, saved.metadata if saved else None)
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
                if n in self.nodes
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
            self.channels,
            checkpoint,
            LoopProtocol(config=config, step=step + 1, stop=step + 2),
        ) as (
            channels,
            managed,
        ):
            # create task to run all writers of the chosen node
            writers = self.nodes[as_node].flat_writers
            if not writers:
                raise InvalidUpdateError(f"Node {as_node} has no writers")
            writes: deque[tuple[str, Any]] = deque()
            task = PregelTaskWrites(as_node, writes, [INTERRUPT])
            task_id = str(uuid5(UUID(checkpoint["id"]), INTERRUPT))
            run = RunnableSequence(*writers) if len(writers) > 1 else writers[0]
            # execute task
            await run.ainvoke(
                values,
                patch_config(
                    config,
                    run_name=self.name + "UpdateState",
                    configurable={
                        # deque.extend is thread-safe
                        CONFIG_KEY_SEND: partial(
                            local_write,
                            writes.extend,
                            self.nodes.keys(),
                        ),
                        CONFIG_KEY_READ: partial(
                            local_read,
                            step + 1,
                            checkpoint,
                            channels,
                            managed,
                            task,
                            config,
                        ),
                    },
                ),
            )
            # save task writes
            if saved:
                await checkpointer.aput_writes(checkpoint_config, writes, task_id)
            # apply to checkpoint and save
            mv_writes = apply_writes(
                checkpoint, channels, [task], checkpointer.get_next_version
            )
            assert not mv_writes, "Can't write to SharedValues from update_state"
            checkpoint = create_checkpoint(checkpoint, channels, step + 1)
            next_config = await checkpointer.aput(
                checkpoint_config,
                checkpoint,
                {
                    **checkpoint_metadata,
                    "source": "update",
                    "step": step + 1,
                    "writes": {as_node: values},
                    "parents": saved.metadata.get("parents", {}) if saved else {},
                },
                get_new_channel_versions(
                    checkpoint_previous_versions, checkpoint["channel_versions"]
                ),
            )
            return patch_checkpoint_map(next_config, saved.metadata if saved else None)

    def _defaults(
        self,
        config: RunnableConfig,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]],
        output_keys: Optional[Union[str, Sequence[str]]],
        interrupt_before: Optional[Union[All, Sequence[str]]],
        interrupt_after: Optional[Union[All, Sequence[str]]],
        debug: Optional[bool],
    ) -> tuple[
        bool,
        set[StreamMode],
        Union[str, Sequence[str]],
        Union[All, Sequence[str]],
        Union[All, Sequence[str]],
        Optional[BaseCheckpointSaver],
        Optional[BaseStore],
    ]:
        if config["recursion_limit"] < 1:
            raise ValueError("recursion_limit must be at least 1")
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
        if CONFIG_KEY_TASK_ID in config.get(CONF, {}):
            # if being called as a node in another graph, always use values mode
            stream_mode = ["values"]
        if self.checkpointer is False:
            checkpointer: Optional[BaseCheckpointSaver] = None
        elif CONFIG_KEY_CHECKPOINTER in config.get(CONF, {}):
            checkpointer = config[CONF][CONFIG_KEY_CHECKPOINTER]
        else:
            checkpointer = self.checkpointer
        if checkpointer and not config.get(CONF):
            raise ValueError(
                f"Checkpointer requires one or more of the following 'configurable' keys: {[s.id for s in checkpointer.config_specs]}"
            )
        if CONFIG_KEY_STORE in config.get(CONF, {}):
            store: Optional[BaseStore] = config[CONF][CONFIG_KEY_STORE]
        else:
            store = self.store
        return (
            debug,
            set(stream_mode),
            output_keys,
            interrupt_before,
            interrupt_after,
            checkpointer,
            store,
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
        subgraphs: bool = False,
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
            subgraphs: Whether to stream subgraphs, defaults to False.

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

        stream = SyncQueue()

        def output() -> Iterator:
            while True:
                try:
                    ns, mode, payload = stream.get(block=False)
                except queue.Empty:
                    break
                if subgraphs and isinstance(stream_mode, list):
                    yield (ns, mode, payload)
                elif isinstance(stream_mode, list):
                    yield (mode, payload)
                elif subgraphs:
                    yield (ns, payload)
                else:
                    yield payload

        config = ensure_config(self.config, config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name", self.get_name()),
            run_id=config.get("run_id"),
        )
        try:
            # assign defaults
            (
                debug,
                stream_modes,
                output_keys,
                interrupt_before_,
                interrupt_after_,
                checkpointer,
                store,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
            )
            # set up messages stream mode
            if "messages" in stream_modes:
                run_manager.inheritable_handlers.append(
                    StreamMessagesHandler(stream.put)
                )
            # set up custom stream mode
            if "custom" in stream_modes:
                config[CONF][CONFIG_KEY_STREAM_WRITER] = lambda c: stream.put(
                    ((), "custom", c)
                )
            with SyncPregelLoop(
                input,
                stream=StreamProtocol(stream.put, stream_modes),
                config=config,
                store=store,
                checkpointer=checkpointer,
                nodes=self.nodes,
                specs=self.channels,
                output_keys=output_keys,
                stream_keys=self.stream_channels_asis,
                debug=debug,
            ) as loop:
                # create runner
                runner = PregelRunner(
                    submit=loop.submit,
                    put_writes=loop.put_writes,
                    node_finished=config[CONF].get(CONFIG_KEY_NODE_FINISHED),
                )
                # enable subgraph streaming
                if subgraphs:
                    loop.config[CONF][CONFIG_KEY_STREAM] = loop.stream
                # enable concurrent streaming
                if subgraphs or "messages" in stream_modes or "custom" in stream_modes:
                    # we are careful to have a single waiter live at any one time
                    # because on exit we increment semaphore count by exactly 1
                    waiter: Optional[concurrent.futures.Future] = None
                    # because sync futures cannot be cancelled, we instead
                    # release the stream semaphore on exit, which will cause
                    # a pending waiter to return immediately
                    loop.stack.callback(stream._count.release)

                    def get_waiter() -> concurrent.futures.Future[None]:
                        nonlocal waiter
                        if waiter is None or waiter.done():
                            waiter = loop.submit(stream.wait)
                            return waiter
                        else:
                            return waiter

                else:
                    get_waiter = None  # type: ignore[assignment]
                # Similarly to Bulk Synchronous Parallel / Pregel model
                # computation proceeds in steps, while there are channel updates
                # channel updates from step N are only visible in step N+1
                # channels are guaranteed to be immutable for the duration of the step,
                # with channel updates applied only at the transition between steps
                while loop.tick(
                    input_keys=self.input_channels,
                    interrupt_before=interrupt_before_,
                    interrupt_after=interrupt_after_,
                    manager=run_manager,
                ):
                    for _ in runner.tick(
                        loop.tasks.values(),
                        timeout=self.step_timeout,
                        retry_policy=self.retry_policy,
                        get_waiter=get_waiter,
                    ):
                        # emit output
                        yield from output()
            # emit output
            yield from output()
            # handle exit
            if loop.status == "out_of_steps":
                msg = create_error_message(
                    message=(
                        f"Recursion limit of {config['recursion_limit']} reached "
                        "without hitting a stop condition. You can increase the "
                        "limit by setting the `recursion_limit` config key."
                    ),
                    error_code=ErrorCode.GRAPH_RECURSION_LIMIT,
                )
                raise GraphRecursionError(msg)
            # set final channel values as run output
            run_manager.on_chain_end(loop.output)
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
        subgraphs: bool = False,
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
            subgraphs: Whether to stream subgraphs, defaults to False.

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

        stream = AsyncQueue()
        aioloop = asyncio.get_running_loop()

        def output() -> Iterator:
            while True:
                try:
                    ns, mode, payload = stream.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if subgraphs and isinstance(stream_mode, list):
                    yield (ns, mode, payload)
                elif isinstance(stream_mode, list):
                    yield (mode, payload)
                elif subgraphs:
                    yield (ns, payload)
                else:
                    yield payload

        config = ensure_config(self.config, config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name", self.get_name()),
            run_id=config.get("run_id"),
        )
        # if running from astream_log() run each proc with streaming
        do_stream = next(
            (
                cast(_StreamingCallbackHandler, h)
                for h in run_manager.handlers
                if isinstance(h, _StreamingCallbackHandler)
            ),
            None,
        )
        try:
            # assign defaults
            (
                debug,
                stream_modes,
                output_keys,
                interrupt_before_,
                interrupt_after_,
                checkpointer,
                store,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                debug=debug,
            )
            # set up messages stream mode
            if "messages" in stream_modes:
                run_manager.inheritable_handlers.append(
                    StreamMessagesHandler(stream.put_nowait)
                )
            # set up custom stream mode
            if "custom" in stream_modes:
                config[CONF][CONFIG_KEY_STREAM_WRITER] = lambda c: stream.put_nowait(
                    ((), "custom", c)
                )
            async with AsyncPregelLoop(
                input,
                stream=StreamProtocol(stream.put_nowait, stream_modes),
                config=config,
                store=store,
                checkpointer=checkpointer,
                nodes=self.nodes,
                specs=self.channels,
                output_keys=output_keys,
                stream_keys=self.stream_channels_asis,
            ) as loop:
                # create runner
                runner = PregelRunner(
                    submit=loop.submit,
                    put_writes=loop.put_writes,
                    use_astream=do_stream is not None,
                    node_finished=config[CONF].get(CONFIG_KEY_NODE_FINISHED),
                )
                # enable subgraph streaming
                if subgraphs:
                    loop.config[CONF][CONFIG_KEY_STREAM] = loop.stream
                # enable concurrent streaming
                if subgraphs or "messages" in stream_modes or "custom" in stream_modes:

                    def get_waiter() -> asyncio.Task[None]:
                        return aioloop.create_task(stream.wait())

                else:
                    get_waiter = None  # type: ignore[assignment]
                # Similarly to Bulk Synchronous Parallel / Pregel model
                # computation proceeds in steps, while there are channel updates
                # channel updates from step N are only visible in step N+1
                # channels are guaranteed to be immutable for the duration of the step,
                # with channel updates applied only at the transition between steps
                while loop.tick(
                    input_keys=self.input_channels,
                    interrupt_before=interrupt_before_,
                    interrupt_after=interrupt_after_,
                    manager=run_manager,
                ):
                    async for _ in runner.atick(
                        loop.tasks.values(),
                        timeout=self.step_timeout,
                        retry_policy=self.retry_policy,
                        get_waiter=get_waiter,
                    ):
                        # emit output
                        for o in output():
                            yield o
            # emit output
            for o in output():
                yield o
            # handle exit
            if loop.status == "out_of_steps":
                msg = create_error_message(
                    message=(
                        f"Recursion limit of {config['recursion_limit']} reached "
                        "without hitting a stop condition. You can increase the "
                        "limit by setting the `recursion_limit` config key."
                    ),
                    error_code=ErrorCode.GRAPH_RECURSION_LIMIT,
                )
                raise GraphRecursionError(msg)
            # set final channel values as run output
            await run_manager.on_chain_end(loop.output)
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
