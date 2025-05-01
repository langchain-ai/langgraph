from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import asdict
from typing import (
    Any,
    Literal,
    Optional,
    Union,
    cast,
)

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

from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_STREAM,
    INTERRUPT,
    NS_SEP,
)
from langgraph.errors import GraphInterrupt
from langgraph.pregel.protocol import PregelProtocol
from langgraph.pregel.types import All, PregelTask, StateSnapshot, StreamMode
from langgraph.types import Command, Interrupt, StreamProtocol
from langgraph.utils.config import merge_configs

CONF_DROPLIST = frozenset(
    (
        CONFIG_KEY_CHECKPOINT_MAP,
        CONFIG_KEY_CHECKPOINT_ID,
        CONFIG_KEY_CHECKPOINT_NS,
    ),
)


def sanitize_config_value(v: Any) -> Any:
    """Recursively sanitize a config value to ensure it contains only primitives."""
    if isinstance(v, (str, int, float, bool)):
        return v
    elif isinstance(v, dict):
        sanitized_dict = {}
        for k, val in v.items():
            if isinstance(k, str):
                sanitized_value = sanitize_config_value(val)
                if sanitized_value is not None:
                    sanitized_dict[k] = sanitized_value
        return sanitized_dict
    elif isinstance(v, (list, tuple)):
        sanitized_list = []
        for item in v:
            sanitized_item = sanitize_config_value(item)
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
    on LangGraph Cloud.

    `RemoteGraph` behaves the same way as a `Graph` and can be used directly as
    a node in another `Graph`.
    """

    name: str

    def __init__(
        self,
        name: str,  # graph_id
        /,
        *,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        client: Optional[LangGraphClient] = None,
        sync_client: Optional[SyncLangGraphClient] = None,
        config: Optional[RunnableConfig] = None,
    ):
        """Specify `url`, `api_key`, and/or `headers` to create default sync and async clients.

        If `client` or `sync_client` are provided, they will be used instead of the default clients.
        See `LangGraphClient` and `SyncLangGraphClient` for details on the default clients. At least
        one of `url`, `client`, or `sync_client` must be provided.

        Args:
            name: The name of the graph.
            url: The URL of the remote API.
            api_key: The API key to use for authentication. If not provided, it will be read from the environment (`LANGGRAPH_API_KEY`, `LANGSMITH_API_KEY`, or `LANGCHAIN_API_KEY`).
            headers: Additional headers to include in the requests.
            client: A `LangGraphClient` instance to use instead of creating a default client.
            sync_client: A `SyncLangGraphClient` instance to use instead of creating a default client.
            config: An optional `RunnableConfig` instance with additional configuration.
        """
        self.name = name
        self.config = config

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
        return self.__class__(attrs.pop("name"), **attrs)

    def with_config(
        self, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Self:
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
        config: Optional[RunnableConfig] = None,
        *,
        xray: Union[int, bool] = False,
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
            assistant_id=self.name,
            xray=xray,
        )
        return DrawableGraph(
            nodes=self._get_drawable_nodes(graph),
            edges=[DrawableEdge(**edge) for edge in graph["edges"]],
        )

    async def aget_graph(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        xray: Union[int, bool] = False,
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
            assistant_id=self.name,
            xray=xray,
        )
        return DrawableGraph(
            nodes=self._get_drawable_nodes(graph),
            edges=[DrawableEdge(**edge) for edge in graph["edges"]],
        )

    def _create_state_snapshot(self, state: ThreadState) -> StateSnapshot:
        tasks: list[PregelTask] = []
        for task in state["tasks"]:
            interrupts = []
            for interrupt in task["interrupts"]:
                interrupts.append(Interrupt(**interrupt))

            tasks.append(
                PregelTask(
                    id=task["id"],
                    name=task["name"],
                    path=tuple(),
                    error=Exception(task["error"]) if task["error"] else None,
                    interrupts=tuple(interrupts),
                    state=self._create_state_snapshot(task["state"])
                    if task["state"]
                    else cast(RunnableConfig, {"configurable": task["checkpoint"]})
                    if task["checkpoint"]
                    else None,
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
            parent_config={
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
            else None,
            tasks=tuple(tasks),
            interrupts=tuple([i for task in tasks for i in task.interrupts]),
        )

    def _get_checkpoint(self, config: Optional[RunnableConfig]) -> Optional[Checkpoint]:
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
                    and (sanitized_value := sanitize_config_value(v)) is not None
                ):
                    sanitized["metadata"][k] = sanitized_value

        if "configurable" in config:
            sanitized["configurable"] = {}
            for k, v in config["configurable"].items():
                if (
                    isinstance(k, str)
                    and k not in CONF_DROPLIST
                    and (sanitized_value := sanitize_config_value(v)) is not None
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
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
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
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
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
        updates: list[tuple[Optional[dict[str, Any]], Optional[str]]],
    ) -> RunnableConfig:
        raise NotImplementedError

    async def abulk_update_state(
        self,
        config: RunnableConfig,
        updates: list[tuple[Optional[dict[str, Any]], Optional[str]]],
    ) -> RunnableConfig:
        raise NotImplementedError

    def update_state(
        self,
        config: RunnableConfig,
        values: Optional[Union[dict[str, Any], Any]],
        as_node: Optional[str] = None,
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
        values: Optional[Union[dict[str, Any], Any]],
        as_node: Optional[str] = None,
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
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]],
        config: Optional[RunnableConfig],
        default: StreamMode = "updates",
    ) -> tuple[
        list[StreamModeSDK], list[StreamModeSDK], bool, Optional[StreamProtocol]
    ]:
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
        stream: Optional[StreamProtocol] = (
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
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        subgraphs: bool = False,
        **kwargs: Any,
    ) -> Iterator[Union[dict[str, Any], Any]]:
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
            command: Optional[CommandSDK] = cast(CommandSDK, asdict(input))
            input = None
        else:
            command = None

        for chunk in sync_client.runs.stream(
            thread_id=sanitized_config["configurable"].get("thread_id"),
            assistant_id=self.name,
            input=input,
            command=command,
            config=sanitized_config,
            stream_mode=stream_modes,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            stream_subgraphs=subgraphs or stream is not None,
            if_not_exists="create",
            **kwargs,
        ):
            # split mode and ns
            if NS_SEP in chunk.event:
                mode, ns_ = chunk.event.split(NS_SEP, 1)
                ns = tuple(ns_.split(NS_SEP))
            else:
                mode, ns = chunk.event, ()
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
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        subgraphs: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[Union[dict[str, Any], Any]]:
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
            command: Optional[CommandSDK] = cast(CommandSDK, asdict(input))
            input = None
        else:
            command = None

        async for chunk in client.runs.stream(
            thread_id=sanitized_config["configurable"].get("thread_id"),
            assistant_id=self.name,
            input=input,
            command=command,
            config=sanitized_config,
            stream_mode=stream_modes,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            stream_subgraphs=subgraphs or stream is not None,
            if_not_exists="create",
            **kwargs,
        ):
            # split mode and ns
            if NS_SEP in chunk.event:
                mode, ns_ = chunk.event.split(NS_SEP, 1)
                ns = tuple(ns_.split(NS_SEP))
            else:
                mode, ns = chunk.event, ()
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
        config: Optional[RunnableConfig] = None,
        *,
        version: Literal["v1", "v2"],
        include_names: Optional[Sequence[All]] = None,
        include_types: Optional[Sequence[All]] = None,
        include_tags: Optional[Sequence[All]] = None,
        exclude_names: Optional[Sequence[All]] = None,
        exclude_types: Optional[Sequence[All]] = None,
        exclude_tags: Optional[Sequence[All]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError

    def invoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        """Create a run, wait until it finishes and return the final state.

        Args:
            input: Input to the graph.
            config: A `RunnableConfig` for graph invocation.
            interrupt_before: Interrupt the graph before these nodes.
            interrupt_after: Interrupt the graph after these nodes.
            **kwargs: Additional params to pass to RemoteGraph.stream.

        Returns:
            The output of the graph.
        """
        for chunk in self.stream(
            input,
            config=config,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
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
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        *,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> Union[dict[str, Any], Any]:
        """Create a run, wait until it finishes and return the final state.

        Args:
            input: Input to the graph.
            config: A `RunnableConfig` for graph invocation.
            interrupt_before: Interrupt the graph before these nodes.
            interrupt_after: Interrupt the graph after these nodes.
            **kwargs: Additional params to pass to RemoteGraph.astream.

        Returns:
            The output of the graph.
        """
        async for chunk in self.astream(
            input,
            config=config,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            stream_mode="values",
            **kwargs,
        ):
            pass
        try:
            return chunk
        except UnboundLocalError:
            return None
