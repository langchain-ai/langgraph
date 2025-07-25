from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import asdict
from typing import (
    Any,
    Literal,
    cast,
)

import langsmith as ls
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import (
    Edge as DrawableEdge,
)
from langchain_core.runnables.graph import (
    Graph as DrawableGraph,
)
from langchain_core.runnables.graph import (
    Node as DrawableNode,
)
from langgraph_sdk.client import (
    LangGraphClient,
    SyncLangGraphClient,
    get_client,
    get_sync_client,
)
from langgraph_sdk.schema import Checkpoint, ThreadState
from langgraph_sdk.schema import Command as CommandSDK
from langgraph_sdk.schema import StreamMode as StreamModeSDK
from typing_extensions import Self

from langgraph._internal._config import merge_configs
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_TASK_ID,
    INTERRUPT,
    NS_SEP,
)
from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.errors import GraphInterrupt, ParentCommand
from langgraph.pregel.protocol import PregelProtocol, StreamProtocol
from langgraph.types import (
    All,
    Command,
    Interrupt,
    PregelTask,
    StateSnapshot,
    StreamMode,
)

__all__ = ("RemoteGraph", "RemoteException")

_CONF_DROPLIST = frozenset(
    (
        CONFIG_KEY_CHECKPOINT_MAP,
        CONFIG_KEY_CHECKPOINT_ID,
        CONFIG_KEY_CHECKPOINT_NS,
        CONFIG_KEY_TASK_ID,
    ),
)


def _sanitize_config_value(v: Any) -> Any:
    """Recursively sanitize a config value to ensure it contains only primitives."""
    if isinstance(v, (str, int, float, bool)):
        return v
    elif isinstance(v, dict):
        sanitized_dict = {}
        for k, val in v.items():
            if isinstance(k, str):
                sanitized_value = _sanitize_config_value(val)
                if sanitized_value is not None:
                    sanitized_dict[k] = sanitized_value
        return sanitized_dict
    elif isinstance(v, (list, tuple)):
        sanitized_list = []
        for item in v:
            sanitized_item = _sanitize_config_value(item)
            if sanitized_item is not None:
                sanitized_list.append(sanitized_item)
        return sanitized_list
    return None


class RemoteException(Exception):
    """Exception raised when an error occurs in the remote graph."""

    pass


class RemoteGraph(PregelProtocol):
    """The `RemoteGraph` class is a client implementation for calling remote
    APIs that implement the LangGraph Server API specification.

    For example, the `RemoteGraph` class can be used to call APIs from deployments
    on LangGraph Platform.

    `RemoteGraph` behaves the same way as a `Graph` and can be used directly as
    a node in another `Graph`.
    """

    assistant_id: str
    name: str | None

    def __init__(
        self,
        assistant_id: str,  # graph_id
        /,
        *,
        url: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        client: LangGraphClient | None = None,
        sync_client: SyncLangGraphClient | None = None,
        config: RunnableConfig | None = None,
        name: str | None = None,
        distributed_tracing: bool = False,
    ):
        """Specify `url`, `api_key`, and/or `headers` to create default sync and async clients.

        If `client` or `sync_client` are provided, they will be used instead of the default clients.
        See `LangGraphClient` and `SyncLangGraphClient` for details on the default clients. At least
        one of `url`, `client`, or `sync_client` must be provided.

        Args:
            assistant_id: The assistant ID or graph name of the remote graph to use.
            url: The URL of the remote API.
            api_key: The API key to use for authentication. If not provided, it will be read from the environment (`LANGGRAPH_API_KEY`, `LANGSMITH_API_KEY`, or `LANGCHAIN_API_KEY`).
            headers: Additional headers to include in the requests.
            client: A `LangGraphClient` instance to use instead of creating a default client.
            sync_client: A `SyncLangGraphClient` instance to use instead of creating a default client.
            config: An optional `RunnableConfig` instance with additional configuration.
            name: Human-readable name to attach to the RemoteGraph instance.
                This is useful for adding `RemoteGraph` as a subgraph via `graph.add_node(remote_graph)`.
                If not provided, defaults to the assistant ID.
            distributed_tracing: Whether to enable sending LangSmith distributed tracing headers.
        """
        self.assistant_id = assistant_id
        if name is None:
            self.name = assistant_id
        else:
            self.name = name
        self.config = config
        self.distributed_tracing = distributed_tracing

        if client is None and url is not None:
            client = get_client(url=url, api_key=api_key, headers=headers)
        self.client = client

        if sync_client is None and url is not None:
            sync_client = get_sync_client(url=url, api_key=api_key, headers=headers)
        self.sync_client = sync_client

    def _validate_client(self) -> LangGraphClient:
        if self.client is None:
            raise ValueError(
                "Async client is not initialized: please provide `url` or `client` when initializing `RemoteGraph`."
            )
        return self.client

    def _validate_sync_client(self) -> SyncLangGraphClient:
        if self.sync_client is None:
            raise ValueError(
                "Sync client is not initialized: please provide `url` or `sync_client` when initializing `RemoteGraph`."
            )
        return self.sync_client

    def copy(self, update: dict[str, Any]) -> Self:
        attrs = {**self.__dict__, **update}
        return self.__class__(attrs.pop("assistant_id"), **attrs)

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Self:
        return self.copy(
            {"config": merge_configs(self.config, config, cast(RunnableConfig, kwargs))}
        )

    def _get_drawable_nodes(
        self, graph: dict[str, list[dict[str, Any]]]
    ) -> dict[str, DrawableNode]:
        nodes = {}
        for node in graph["nodes"]:
            node_id = str(node["id"])
            node_data = node.get("data", {})

            # Get node name from node_data if available. If not, use node_id.
            node_name = node.get("name")
            if node_name is None:
                if isinstance(node_data, dict):
                    node_name = node_data.get("name", node_id)
                else:
                    node_name = node_id

            nodes[node_id] = DrawableNode(
                id=node_id,
                name=node_name,
                data=node_data,
                metadata=node.get("metadata"),
            )
        return nodes

    def get_graph(
        self,
        config: RunnableConfig | None = None,
        *,
        xray: int | bool = False,
    ) -> DrawableGraph:
        """Get graph by graph name.

        This method calls `GET /assistants/{assistant_id}/graph`.

        Args:
            config: This parameter is not used.
            xray: Include graph representation of subgraphs. If an integer
                value is provided, only subgraphs with a depth less than or
                equal to the value will be included.

        Returns:
            The graph information for the assistant in JSON format.
        """
        sync_client = self._validate_sync_client()
        graph = sync_client.assistants.get_graph(
            assistant_id=self.assistant_id,
            xray=xray,
        )
        return DrawableGraph(
            nodes=self._get_drawable_nodes(graph),
            edges=[DrawableEdge(**edge) for edge in graph["edges"]],
        )

    async def aget_graph(
        self,
        config: RunnableConfig | None = None,
        *,
        xray: int | bool = False,
    ) -> DrawableGraph:
        """Get graph by graph name.

        This method calls `GET /assistants/{assistant_id}/graph`.

        Args:
            config: This parameter is not used.
            xray: Include graph representation of subgraphs. If an integer
                value is provided, only subgraphs with a depth less than or
                equal to the value will be included.

        Returns:
            The graph information for the assistant in JSON format.
        """
        client = self._validate_client()
        graph = await client.assistants.get_graph(
            assistant_id=self.assistant_id,
            xray=xray,
        )
        return DrawableGraph(
            nodes=self._get_drawable_nodes(graph),
            edges=[DrawableEdge(**edge) for edge in graph["edges"]],
        )

    def _create_state_snapshot(self, state: ThreadState) -> StateSnapshot:
        tasks: list[PregelTask] = []
        for task in state["tasks"]:
            interrupts = tuple(
                Interrupt(**interrupt) for interrupt in task["interrupts"]
            )

            tasks.append(
                PregelTask(
                    id=task["id"],
                    name=task["name"],
                    path=tuple(),
                    error=Exception(task["error"]) if task["error"] else None,
                    interrupts=interrupts,
                    state=(
                        self._create_state_snapshot(task["state"])
                        if task["state"]
                        else (
                            cast(RunnableConfig, {"configurable": task["checkpoint"]})
                            if task["checkpoint"]
                            else None
                        )
                    ),
                    result=task.get("result"),
                )
            )

        return StateSnapshot(
            values=state["values"],
            next=tuple(state["next"]) if state["next"] else tuple(),
            config={
                "configurable": {
                    "thread_id": state["checkpoint"]["thread_id"],
                    "checkpoint_ns": state["checkpoint"]["checkpoint_ns"],
                    "checkpoint_id": state["checkpoint"]["checkpoint_id"],
                    "checkpoint_map": state["checkpoint"].get("checkpoint_map", {}),
                }
            },
            metadata=CheckpointMetadata(**state["metadata"]),
            created_at=state["created_at"],
            parent_config=(
                {
                    "configurable": {
                        "thread_id": state["parent_checkpoint"]["thread_id"],
                        "checkpoint_ns": state["parent_checkpoint"]["checkpoint_ns"],
                        "checkpoint_id": state["parent_checkpoint"]["checkpoint_id"],
                        "checkpoint_map": state["parent_checkpoint"].get(
                            "checkpoint_map", {}
                        ),
                    }
                }
                if state["parent_checkpoint"]
                else None
            ),
            tasks=tuple(tasks),
            interrupts=tuple([i for task in tasks for i in task.interrupts]),
        )

    def _get_checkpoint(self, config: RunnableConfig | None) -> Checkpoint | None:
        if config is None:
            return None

        checkpoint = {}

        if "thread_id" in config["configurable"]:
            checkpoint["thread_id"] = config["configurable"]["thread_id"]
        if "checkpoint_ns" in config["configurable"]:
            checkpoint["checkpoint_ns"] = config["configurable"]["checkpoint_ns"]
        if "checkpoint_id" in config["configurable"]:
            checkpoint["checkpoint_id"] = config["configurable"]["checkpoint_id"]
        if "checkpoint_map" in config["configurable"]:
            checkpoint["checkpoint_map"] = config["configurable"]["checkpoint_map"]

        return checkpoint if checkpoint else None

    def _get_config(self, checkpoint: Checkpoint) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": checkpoint["thread_id"],
                "checkpoint_ns": checkpoint["checkpoint_ns"],
                "checkpoint_id": checkpoint["checkpoint_id"],
                "checkpoint_map": checkpoint.get("checkpoint_map", {}),
            }
        }

    def _sanitize_config(self, config: RunnableConfig) -> RunnableConfig:
        """Sanitize the config to remove non-serializable fields."""
        sanitized: RunnableConfig = {}
        if "recursion_limit" in config:
            sanitized["recursion_limit"] = config["recursion_limit"]
        if "tags" in config:
            sanitized["tags"] = [tag for tag in config["tags"] if isinstance(tag, str)]

        if "metadata" in config:
            sanitized["metadata"] = {}
            for k, v in config["metadata"].items():
                if (
                    isinstance(k, str)
                    and (sanitized_value := _sanitize_config_value(v)) is not None
                ):
                    sanitized["metadata"][k] = sanitized_value

        if "configurable" in config:
            sanitized["configurable"] = {}
            for k, v in config["configurable"].items():
                if (
                    isinstance(k, str)
                    and k not in _CONF_DROPLIST
                    and (sanitized_value := _sanitize_config_value(v)) is not None
                ):
                    sanitized["configurable"][k] = sanitized_value

        return sanitized

    def get_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        """Get the state of a thread.

        This method calls `POST /threads/{thread_id}/state/checkpoint` if a
        checkpoint is specified in the config or `GET /threads/{thread_id}/state`
        if no checkpoint is specified.

        Args:
            config: A `RunnableConfig` that includes `thread_id` in the
                `configurable` field.
            subgraphs: Include subgraphs in the state.

        Returns:
            The latest state of the thread.
        """
        sync_client = self._validate_sync_client()
        merged_config = merge_configs(self.config, config)

        state = sync_client.threads.get_state(
            thread_id=merged_config["configurable"]["thread_id"],
            checkpoint=self._get_checkpoint(merged_config),
            subgraphs=subgraphs,
        )
        return self._create_state_snapshot(state)

    async def aget_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        """Get the state of a thread.

        This method calls `POST /threads/{thread_id}/state/checkpoint` if a
        checkpoint is specified in the config or `GET /threads/{thread_id}/state`
        if no checkpoint is specified.

        Args:
            config: A `RunnableConfig` that includes `thread_id` in the
                `configurable` field.
            subgraphs: Include subgraphs in the state.

        Returns:
            The latest state of the thread.
        """
        client = self._validate_client()
        merged_config = merge_configs(self.config, config)

        state = await client.threads.get_state(
            thread_id=merged_config["configurable"]["thread_id"],
            checkpoint=self._get_checkpoint(merged_config),
            subgraphs=subgraphs,
        )
        return self._create_state_snapshot(state)

    def get_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[StateSnapshot]:
        """Get the state history of a thread.

        This method calls `POST /threads/{thread_id}/history`.

        Args:
            config: A `RunnableConfig` that includes `thread_id` in the
                `configurable` field.
            filter: Metadata to filter on.
            before: A `RunnableConfig` that includes checkpoint metadata.
            limit: Max number of states to return.

        Returns:
            States of the thread.
        """
        sync_client = self._validate_sync_client()
        merged_config = merge_configs(self.config, config)

        states = sync_client.threads.get_history(
            thread_id=merged_config["configurable"]["thread_id"],
            limit=limit if limit else 10,
            before=self._get_checkpoint(before),
            metadata=filter,
            checkpoint=self._get_checkpoint(merged_config),
        )
        for state in states:
            yield self._create_state_snapshot(state)

    async def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[StateSnapshot]:
        """Get the state history of a thread.

        This method calls `POST /threads/{thread_id}/history`.

        Args:
            config: A `RunnableConfig` that includes `thread_id` in the
                `configurable` field.
            filter: Metadata to filter on.
            before: A `RunnableConfig` that includes checkpoint metadata.
            limit: Max number of states to return.

        Returns:
            States of the thread.
        """
        client = self._validate_client()
        merged_config = merge_configs(self.config, config)

        states = await client.threads.get_history(
            thread_id=merged_config["configurable"]["thread_id"],
            limit=limit if limit else 10,
            before=self._get_checkpoint(before),
            metadata=filter,
            checkpoint=self._get_checkpoint(merged_config),
        )
        for state in states:
            yield self._create_state_snapshot(state)

    def bulk_update_state(
        self,
        config: RunnableConfig,
        updates: list[tuple[dict[str, Any] | None, str | None]],
    ) -> RunnableConfig:
        raise NotImplementedError

    async def abulk_update_state(
        self,
        config: RunnableConfig,
        updates: list[tuple[dict[str, Any] | None, str | None]],
    ) -> RunnableConfig:
        raise NotImplementedError

    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
    ) -> RunnableConfig:
        """Update the state of a thread.

        This method calls `POST /threads/{thread_id}/state`.

        Args:
            config: A `RunnableConfig` that includes `thread_id` in the
                `configurable` field.
            values: Values to update to the state.
            as_node: Update the state as if this node had just executed.

        Returns:
            `RunnableConfig` for the updated thread.
        """
        sync_client = self._validate_sync_client()
        merged_config = merge_configs(self.config, config)

        response: dict = sync_client.threads.update_state(  # type: ignore
            thread_id=merged_config["configurable"]["thread_id"],
            values=values,
            as_node=as_node,
            checkpoint=self._get_checkpoint(merged_config),
        )
        return self._get_config(response["checkpoint"])

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
    ) -> RunnableConfig:
        """Update the state of a thread.

        This method calls `POST /threads/{thread_id}/state`.

        Args:
            config: A `RunnableConfig` that includes `thread_id` in the
                `configurable` field.
            values: Values to update to the state.
            as_node: Update the state as if this node had just executed.

        Returns:
            `RunnableConfig` for the updated thread.
        """
        client = self._validate_client()
        merged_config = merge_configs(self.config, config)

        response: dict = await client.threads.update_state(  # type: ignore
            thread_id=merged_config["configurable"]["thread_id"],
            values=values,
            as_node=as_node,
            checkpoint=self._get_checkpoint(merged_config),
        )
        return self._get_config(response["checkpoint"])

    def _get_stream_modes(
        self,
        stream_mode: StreamMode | list[StreamMode] | None,
        config: RunnableConfig | None,
        default: StreamMode = "updates",
    ) -> tuple[list[StreamModeSDK], list[StreamModeSDK], bool, StreamProtocol | None]:
        """Return a tuple of the final list of stream modes sent to the
        remote graph and a boolean flag indicating if stream mode 'updates'
        was present in the original list of stream modes.

        'updates' mode is added to the list of stream modes so that interrupts
        can be detected in the remote graph.
        """
        updated_stream_modes: list[StreamModeSDK] = []
        req_single = True
        # coerce to list, or add default stream mode
        if stream_mode:
            if isinstance(stream_mode, str):
                updated_stream_modes.append(stream_mode)
            else:
                req_single = False
                updated_stream_modes.extend(stream_mode)
        else:
            updated_stream_modes.append(default)
        requested_stream_modes = updated_stream_modes.copy()
        # add any from parent graph
        stream: StreamProtocol | None = (
            (config or {}).get(CONF, {}).get(CONFIG_KEY_STREAM)
        )
        if stream:
            updated_stream_modes.extend(stream.modes)
        # map "messages" to "messages-tuple"
        if "messages" in updated_stream_modes:
            updated_stream_modes.remove("messages")
            updated_stream_modes.append("messages-tuple")

        # if requested "messages-tuple",
        # map to "messages" in requested_stream_modes
        if "messages-tuple" in requested_stream_modes:
            requested_stream_modes.remove("messages-tuple")
            requested_stream_modes.append("messages")

        # add 'updates' mode if not present
        if "updates" not in updated_stream_modes:
            updated_stream_modes.append("updates")

        # remove 'events', as it's not supported in Pregel
        if "events" in updated_stream_modes:
            updated_stream_modes.remove("events")
        return (updated_stream_modes, requested_stream_modes, req_single, stream)

    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any] | Any]:
        """Create a run and stream the results.

        This method calls `POST /threads/{thread_id}/runs/stream` if a `thread_id`
        is speciffed in the `configurable` field of the config or
        `POST /runs/stream` otherwise.

        Args:
            input: Input to the graph.
            config: A `RunnableConfig` for graph invocation.
            stream_mode: Stream mode(s) to use.
            interrupt_before: Interrupt the graph before these nodes.
            interrupt_after: Interrupt the graph after these nodes.
            subgraphs: Stream from subgraphs.
            headers: Additional headers to pass to the request.
            **kwargs: Additional params to pass to client.runs.stream.

        Yields:
            The output of the graph.
        """
        sync_client = self._validate_sync_client()
        merged_config = merge_configs(self.config, config)
        sanitized_config = self._sanitize_config(merged_config)
        stream_modes, requested, req_single, stream = self._get_stream_modes(
            stream_mode, config
        )
        if isinstance(input, Command):
            command: CommandSDK | None = cast(CommandSDK, asdict(input))
            input = None
        else:
            command = None

        for chunk in sync_client.runs.stream(
            thread_id=sanitized_config["configurable"].get("thread_id"),
            assistant_id=self.assistant_id,
            input=input,
            command=command,
            config=sanitized_config,
            stream_mode=stream_modes,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            stream_subgraphs=subgraphs or stream is not None,
            if_not_exists="create",
            headers=_merge_tracing_headers(headers)
            if self.distributed_tracing
            else headers,
            **kwargs,
        ):
            # split mode and ns
            if NS_SEP in chunk.event:
                mode, ns_ = chunk.event.split(NS_SEP, 1)
                ns = tuple(ns_.split(NS_SEP))
            else:
                mode, ns = chunk.event, ()
            # raise ParentCommand exception for command events
            if mode == "command" and chunk.data.get("graph") == Command.PARENT:
                raise ParentCommand(Command(**chunk.data))
            # prepend caller ns (as it is not passed to remote graph)
            if caller_ns := (config or {}).get(CONF, {}).get(CONFIG_KEY_CHECKPOINT_NS):
                caller_ns = tuple(caller_ns.split(NS_SEP))
                ns = caller_ns + ns
            # stream to parent stream
            if stream is not None and mode in stream.modes:
                stream((ns, mode, chunk.data))
            # raise interrupt or errors
            if chunk.event.startswith("updates"):
                if isinstance(chunk.data, dict) and INTERRUPT in chunk.data:
                    if caller_ns:
                        raise GraphInterrupt(
                            [Interrupt(**i) for i in chunk.data[INTERRUPT]]
                        )
            elif chunk.event.startswith("error"):
                raise RemoteException(chunk.data)
            # filter for what was actually requested
            if mode not in requested:
                continue

            if chunk.event.startswith("messages"):
                chunk = chunk._replace(data=tuple(chunk.data))  # type: ignore

            # emit chunk
            if subgraphs:
                if NS_SEP in chunk.event:
                    mode, ns_ = chunk.event.split(NS_SEP, 1)
                    ns = tuple(ns_.split(NS_SEP))
                else:
                    mode, ns = chunk.event, ()
                if req_single:
                    yield ns, chunk.data
                else:
                    yield ns, mode, chunk.data
            elif req_single:
                yield chunk.data
            else:
                yield chunk

    async def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        """Create a run and stream the results.

        This method calls `POST /threads/{thread_id}/runs/stream` if a `thread_id`
        is speciffed in the `configurable` field of the config or
        `POST /runs/stream` otherwise.

        Args:
            input: Input to the graph.
            config: A `RunnableConfig` for graph invocation.
            stream_mode: Stream mode(s) to use.
            interrupt_before: Interrupt the graph before these nodes.
            interrupt_after: Interrupt the graph after these nodes.
            subgraphs: Stream from subgraphs.
            headers: Additional headers to pass to the request.
            **kwargs: Additional params to pass to client.runs.stream.

        Yields:
            The output of the graph.
        """
        client = self._validate_client()
        merged_config = merge_configs(self.config, config)
        sanitized_config = self._sanitize_config(merged_config)
        stream_modes, requested, req_single, stream = self._get_stream_modes(
            stream_mode, config
        )
        if isinstance(input, Command):
            command: CommandSDK | None = cast(CommandSDK, asdict(input))
            input = None
        else:
            command = None

        async for chunk in client.runs.stream(
            thread_id=sanitized_config["configurable"].get("thread_id"),
            assistant_id=self.assistant_id,
            input=input,
            command=command,
            config=sanitized_config,
            stream_mode=stream_modes,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            stream_subgraphs=subgraphs or stream is not None,
            if_not_exists="create",
            headers=_merge_tracing_headers(headers)
            if self.distributed_tracing
            else headers,
            **kwargs,
        ):
            # split mode and ns
            if NS_SEP in chunk.event:
                mode, ns_ = chunk.event.split(NS_SEP, 1)
                ns = tuple(ns_.split(NS_SEP))
            else:
                mode, ns = chunk.event, ()
            # raise ParentCommand exception for command events
            if mode == "command" and chunk.data.get("graph") == Command.PARENT:
                raise ParentCommand(Command(**chunk.data))
            # prepend caller ns (as it is not passed to remote graph)
            if caller_ns := (config or {}).get(CONF, {}).get(CONFIG_KEY_CHECKPOINT_NS):
                caller_ns = tuple(caller_ns.split(NS_SEP))
                ns = caller_ns + ns
            # stream to parent stream
            if stream is not None and mode in stream.modes:
                stream((ns, mode, chunk.data))
            # raise interrupt or errors
            if chunk.event.startswith("updates"):
                if isinstance(chunk.data, dict) and INTERRUPT in chunk.data:
                    if caller_ns:
                        raise GraphInterrupt(
                            [Interrupt(**i) for i in chunk.data[INTERRUPT]]
                        )
            elif chunk.event.startswith("error"):
                raise RemoteException(chunk.data)
            # filter for what was actually requested
            if mode not in requested:
                continue

            if chunk.event.startswith("messages"):
                chunk = chunk._replace(data=tuple(chunk.data))  # type: ignore

            # emit chunk
            if subgraphs:
                if NS_SEP in chunk.event:
                    mode, ns_ = chunk.event.split(NS_SEP, 1)
                    ns = tuple(ns_.split(NS_SEP))
                else:
                    mode, ns = chunk.event, ()
                if req_single:
                    yield ns, chunk.data
                else:
                    yield ns, mode, chunk.data
            elif req_single:
                yield chunk.data
            else:
                yield chunk

    async def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"],
        include_names: Sequence[All] | None = None,
        include_types: Sequence[All] | None = None,
        include_tags: Sequence[All] | None = None,
        exclude_names: Sequence[All] | None = None,
        exclude_types: Sequence[All] | None = None,
        exclude_tags: Sequence[All] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError

    def invoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        """Create a run, wait until it finishes and return the final state.

        Args:
            input: Input to the graph.
            config: A `RunnableConfig` for graph invocation.
            interrupt_before: Interrupt the graph before these nodes.
            interrupt_after: Interrupt the graph after these nodes.
            headers: Additional headers to pass to the request.
            **kwargs: Additional params to pass to RemoteGraph.stream.

        Returns:
            The output of the graph.
        """
        for chunk in self.stream(
            input,
            config=config,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            headers=headers,
            stream_mode="values",
            **kwargs,
        ):
            pass
        try:
            return chunk
        except UnboundLocalError:
            return None

    async def ainvoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        """Create a run, wait until it finishes and return the final state.

        Args:
            input: Input to the graph.
            config: A `RunnableConfig` for graph invocation.
            interrupt_before: Interrupt the graph before these nodes.
            interrupt_after: Interrupt the graph after these nodes.
            headers: Additional headers to pass to the request.
            **kwargs: Additional params to pass to RemoteGraph.astream.

        Returns:
            The output of the graph.
        """
        async for chunk in self.astream(
            input,
            config=config,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            headers=headers,
            stream_mode="values",
            **kwargs,
        ):
            pass
        try:
            return chunk
        except UnboundLocalError:
            return None


def _merge_tracing_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    if rt := ls.get_current_run_tree():
        tracing_headers = rt.to_headers()
        baggage = tracing_headers.pop("baggage")
        if headers:
            if "baggage" in headers:
                baggage = headers["baggage"] + "," + baggage
            tracing_headers["baggage"] = baggage
            headers.update(tracing_headers)
        else:
            headers = tracing_headers
    return headers
