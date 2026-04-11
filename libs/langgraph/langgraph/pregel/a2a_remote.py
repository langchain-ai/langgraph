"""A2ARemoteGraph — LangGraph adapter for remote A2A agents.

This module bridges two fundamentally different state models:
    - LangGraph: checkpoint-based, with thread_id / checkpoint_id / messages
    - A2A: task-based, with context_id / task_id / parts

Key mappings:
    LangGraph thread_id      <-> A2A context_id
    LangGraph checkpoint_id  <-> A2A task_id
    LangGraph messages       <-> A2A parts
    LangGraph GraphInterrupt <-> A2A INPUT_REQUIRED / AUTH_REQUIRED
    LangGraph StreamMode     <-> A2A SSE events

Implements `PregelProtocol` so it can be used anywhere a `CompiledGraph`
or `RemoteGraph` is accepted — including `builder.add_node("agent", a2a)`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Literal, overload

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
from typing_extensions import Self

from langgraph._internal._a2a import (
    A2AClientError,
    A2AMessage,
    A2ATask,
    AgentCard,
    Artifact,
    AsyncA2AClient,
    Part,
    Role,
    SyncA2AClient,
    TaskState,
    TaskStatus,
)
from langgraph._internal._config import merge_configs, sanitize_config_value
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
from langgraph.errors import GraphInterrupt
from langgraph.pregel.protocol import PregelProtocol, StreamProtocol
from langgraph.types import (
    All,
    Command,
    GraphOutput,
    Interrupt,
    PregelTask,
    StateSnapshot,
    StreamMode,
)
# Note: StreamPart comes from langgraph_sdk.schema (NamedTuple, instantiable),
# NOT from langgraph.types.StreamPart (TypeAliasType, not instantiable).
# RemoteGraph uses the same import.
from langgraph_sdk.schema import StreamPart

logger = logging.getLogger(__name__)

__all__ = ("A2ARemoteGraph",)

# LangGraph-internal config keys that must be stripped before sending to
# remote A2A agents — they are LangGraph-specific and the remote agent
# would not understand them.
_CONF_DROPLIST = frozenset(
    (
        CONFIG_KEY_CHECKPOINT_MAP,
        CONFIG_KEY_CHECKPOINT_ID,
        CONFIG_KEY_CHECKPOINT_NS,
        CONFIG_KEY_TASK_ID,
    ),
)


# --------------------------------------------------------------------------- #
#  State conversion: LangGraph <-> A2A
# --------------------------------------------------------------------------- #


def _messages_to_parts(messages: list[Any]) -> list[Part]:
    """Convert LangGraph messages into A2A Parts.

    LangGraph messages come in at least 4 formats, all handled:
        1. dict:   {"role": "user", "content": "hi"}
        2. tuple:  ("user", "hi")
        3. object: HumanMessage(content="hi")
        4. multimodal: {"content": [{"type": "text", "text": "hi"}, ...]}
    """
    parts: list[Part] = []
    for msg in messages:
        # Step 1: Extract content from different message formats
        if isinstance(msg, dict):
            content = msg.get("content", "")
        elif isinstance(msg, tuple):
            # ("user", "hi") format
            content = msg[1] if len(msg) > 1 else ""
        elif hasattr(msg, "content"):
            # BaseMessage objects (HumanMessage, AIMessage, etc.)
            content = msg.content
        else:
            content = str(msg)

        # Step 2: Content itself may also come in multiple formats
        if isinstance(content, str):
            # Plain string -> single text Part
            parts.append(Part(text=content))
        elif isinstance(content, list):
            # Multimodal content list -> process each item
            for item in content:
                if isinstance(item, str):
                    parts.append(Part(text=item))
                elif isinstance(item, dict) and "text" in item:
                    # {"type": "text", "text": "..."}
                    parts.append(Part(text=item["text"]))
                elif isinstance(item, dict):
                    # Structured data (image URLs, etc.)
                    parts.append(Part(data=item))
        else:
            parts.append(Part(text=str(content)))
    return parts


def _last_user_messages(messages: list[Any]) -> list[Any]:
    """Extract the trailing contiguous block of user messages.

    A2A uses a "send message" model rather than sending the full history.
    Only the last round of user messages is sent.

    Examples:
        [user, assistant, user, user] -> [user, user] (last two)
        [user, assistant, user]       -> [user]       (last one)
        [user, assistant]             -> [assistant]  (no trailing user, take last msg)
    """
    result: list[Any] = []
    for msg in reversed(messages):
        # Extract role from different message formats
        role = None
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("type"))
        elif isinstance(msg, tuple):
            role = msg[0]
        elif hasattr(msg, "type"):
            role = msg.type

        if role in ("user", "human"):
            result.insert(0, msg)            # Preserve original order
        elif result:
            break                            # Hit non-user msg after collecting some -> stop
    # Fall back to the last message (regardless of role) if no trailing user messages
    return result if result else messages[-1:] if messages else []


def _task_to_ai_messages(task: A2ATask) -> list[dict[str, Any]]:
    """Convert an A2A Task into a list of LangGraph AI message dicts.

    Extracts content from two sources:
        1. task.status.message — the latest status message
        2. task.artifacts      — output artifacts produced by the agent

    Uses a `seen` set to deduplicate: some agents put the same text in
    both places. Falls back to a placeholder message if both are empty.
    """
    messages: list[dict[str, Any]] = []
    seen: set[str] = set()                   # For deduplication

    def _add(text: str) -> None:
        """Add an AI message, automatically deduplicating."""
        if text and text not in seen:
            seen.add(text)
            messages.append({"role": "assistant", "type": "ai", "content": text})

    # Source 1: Status message (only process AGENT role)
    if task.status.message and task.status.message.role == Role.AGENT:
        for part in task.status.message.parts:
            if part.text:
                _add(part.text)
            elif part.data:
                _add(json.dumps(part.data))  # Structured data -> JSON string

    # Source 2: Artifacts
    for artifact in task.artifacts:
        for part in artifact.parts:
            if part.text:
                _add(part.text)
            elif part.data:
                _add(json.dumps(part.data))

    # Fallback: generate a placeholder if both sources are empty
    if not messages:
        messages.append({
            "role": "assistant",
            "type": "ai",
            "content": f"[A2A task {task.status.state.value}]",
        })
    return messages


# --------------------------------------------------------------------------- #
#  A2ARemoteGraph
# --------------------------------------------------------------------------- #


class A2ARemoteGraph(PregelProtocol):
    """Client for calling remote A2A-compliant agents.

    Unlike `RemoteGraph` which only talks to LangGraph Server,
    `A2ARemoteGraph` uses the A2A JSON-RPC 2.0 protocol and can
    communicate with agents built on any framework.

    Can be used directly as a node in `StateGraph`:

        a2a = A2ARemoteGraph("https://agent.example.com/a2a", name="translator")
        builder = StateGraph(MessagesState)
        builder.add_node("translator", a2a)
    """

    name: str

    def __init__(
        self,
        url: str,
        /,
        *,
        name: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        client: AsyncA2AClient | None = None,
        sync_client: SyncA2AClient | None = None,
        config: RunnableConfig | None = None,
        agent_card: AgentCard | None = None,
    ):
        """Create an A2A remote graph.

        Specify ``url``, ``api_key`` and/or ``headers`` to create default
        sync and async clients. Pre-built ``client`` / ``sync_client``
        instances are used when provided (mainly for injecting mocks in tests).

        Args:
            url: The A2A JSON-RPC endpoint URL.
            name: Human-readable node name. Defaults to ``"a2a_agent"``.
            api_key: Bearer token for the ``Authorization`` header.
            headers: Extra HTTP headers forwarded to both clients.
            client: An existing async A2A client.
            sync_client: An existing sync A2A client.
            config: Optional base ``RunnableConfig``.
            agent_card: A pre-fetched ``AgentCard``. When provided the
                agent's capabilities are used to auto-configure streaming.
        """
        self.name = name or "a2a_agent"
        self.url = url
        self.config = config
        self.agent_card = agent_card

        # Auto-create clients if not injected (standard production usage)
        if client is None:
            client = AsyncA2AClient(url, api_key=api_key, headers=headers)
        self.client = client

        if sync_client is None:
            sync_client = SyncA2AClient(url, api_key=api_key, headers=headers)
        self.sync_client = sync_client

    # -- factory ------------------------------------------------------------ #

    @classmethod
    def from_agent_card_sync(
        cls,
        card_url: str,
        *,
        name: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> A2ARemoteGraph:
        """Discover an agent via its Agent Card URL (sync).

        Flow:
            1. Create a temporary client to fetch the Agent Card (GET card_url)
            2. Find the JSON-RPC endpoint from the Card's supported_interfaces
            3. Create an A2ARemoteGraph instance with the resolved endpoint

        Falls back to the Card URL's base URL if no JSONRPC interface is declared.
        """
        tmp = SyncA2AClient(card_url, api_key=api_key, headers=headers)
        try:
            card = tmp.fetch_agent_card(card_url)
        finally:
            tmp.close()                      # Temp client closed after use
        # Prefer the JSONRPC URL declared in the Card; derive from Card URL otherwise
        endpoint = card.jsonrpc_url or card_url.rsplit("/.well-known/", 1)[0]
        return cls(
            endpoint,
            name=name or card.name,          # Default to the Card's agent name
            api_key=api_key,
            headers=headers,
            agent_card=card,
        )

    @classmethod
    async def from_agent_card(
        cls,
        card_url: str,
        *,
        name: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> A2ARemoteGraph:
        """Discover an agent via its Agent Card URL (async).

        Async version; logic mirrors from_agent_card_sync.
        """
        tmp = AsyncA2AClient(card_url, api_key=api_key, headers=headers)
        try:
            card = await tmp.fetch_agent_card(card_url)
        finally:
            await tmp.close()
        endpoint = card.jsonrpc_url or card_url.rsplit("/.well-known/", 1)[0]
        return cls(
            endpoint,
            name=name or card.name,
            api_key=api_key,
            headers=headers,
            agent_card=card,
        )

    # -- copy / config ------------------------------------------------------ #

    def copy(self, update: dict[str, Any]) -> Self:
        """Create a shallow copy of this instance, overriding specified attrs."""
        attrs = {**self.__dict__, **update}
        return self.__class__(attrs.pop("url"), **attrs)

    def with_config(
        self, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Self:
        """Create a new instance with merged config (original is not mutated).

        Consistent with RemoteGraph.with_config behavior:
            copy = graph.with_config({"configurable": {"thread_id": "t1"}})
            assert copy is not graph  # New instance
        """
        return self.copy(
            {"config": merge_configs(self.config, config, dict(kwargs))}
        )

    # -- graph introspection ------------------------------------------------ #

    def get_graph(
        self,
        config: RunnableConfig | None = None,
        *,
        xray: int | bool = False,
    ) -> DrawableGraph:
        """Return a minimal drawable graph.

        A2A agents are opaque — their internal topology is not exposed.
        Returns a single-node graph: __start__ -> agent -> __end__
        This is the correct semantic representation: from the outside it's a black box.
        """
        node = DrawableNode(id="0", name=self.name, data={}, metadata=None)
        start = DrawableNode(id="__start__", name="__start__", data={}, metadata=None)
        end = DrawableNode(id="__end__", name="__end__", data={}, metadata=None)
        return DrawableGraph(
            nodes={"__start__": start, "0": node, "__end__": end},
            edges=[
                DrawableEdge(source="__start__", target="0"),
                DrawableEdge(source="0", target="__end__"),
            ],
        )

    async def aget_graph(
        self,
        config: RunnableConfig | None = None,
        *,
        xray: int | bool = False,
    ) -> DrawableGraph:
        """Async get_graph; delegates to sync version (no I/O needed)."""
        return self.get_graph(config, xray=xray)

    # -- state management --------------------------------------------------- #

    def _context_id(self, config: RunnableConfig | None) -> str:
        """Extract thread_id from LangGraph config as A2A context_id.

        Generates a UUID if no thread_id is present (creates a new conversation).
        """
        if config and "configurable" in config:
            return config["configurable"].get("thread_id", str(uuid.uuid4()))
        return str(uuid.uuid4())

    def _task_id(self, config: RunnableConfig | None) -> str | None:
        """Extract checkpoint_id from LangGraph config as A2A task_id.

        Returns None if no checkpoint_id (new conversation, no existing task).
        """
        if config and "configurable" in config:
            return config["configurable"].get("checkpoint_id")
        return None

    def _task_to_snapshot(self, task: A2ATask) -> StateSnapshot:
        """Convert an A2A Task into a LangGraph StateSnapshot.

        This is the core conversion in the adapter layer:
            - A2A terminal (COMPLETED/FAILED/...)    -> next=(), graph execution done
            - A2A interrupted (INPUT_REQUIRED/...)    -> next=(), with interrupts
            - A2A in-progress (WORKING/SUBMITTED)     -> next=(self.name,), still running
            - A2A FAILED                              -> pregel_task.error is set

        metadata.source="a2a" marks the data source; step=-1 because A2A
        doesn't expose the step concept.
        """
        is_done = task.status.state.is_terminal
        is_interrupted = task.status.state.is_interrupted

        # Build interrupt info for INPUT_REQUIRED / AUTH_REQUIRED
        interrupts: tuple[Interrupt, ...] = ()
        if is_interrupted:
            msg = ""
            if task.status.message:
                texts = [p.text for p in task.status.message.parts if p.text]
                msg = "\n".join(texts)
            interrupts = (Interrupt(value=msg or task.status.state.value),)

        # Determine next nodes: only non-terminal + non-interrupted have next
        next_nodes: tuple[str, ...] = ()
        if not is_done and not is_interrupted:
            next_nodes = (self.name,)

        values = {"messages": _task_to_ai_messages(task)}

        # PregelTask is LangGraph's internal task representation
        pregel_task = PregelTask(
            id=task.id,
            name=self.name,
            path=(),
            error=(
                # Extract error message from status.message if FAILED
                Exception(task.status.message.parts[0].text)
                if task.status.state == TaskState.FAILED
                and task.status.message
                and task.status.message.parts
                and task.status.message.parts[0].text
                else None
            ),
            interrupts=interrupts,
        )

        return StateSnapshot(
            values=values,
            next=next_nodes,
            config={
                "configurable": {
                    "thread_id": task.context_id or "",
                    "checkpoint_ns": "",
                    "checkpoint_id": task.id,    # Map task_id back to checkpoint_id
                }
            },
            metadata=CheckpointMetadata(
                source="a2a",                    # Mark origin
                step=-1,                         # A2A doesn't expose step
                writes={},
                parents={},
            ),
            created_at=task.status.timestamp,
            parent_config=None,
            tasks=(pregel_task,),
            interrupts=interrupts,
        )

    def get_state(
        self,
        config: RunnableConfig,
        *,
        subgraphs: bool = False,
    ) -> StateSnapshot:
        """Fetch the current A2A task state.

        Maps ``configurable.checkpoint_id`` to the A2A task ID and calls
        ``tasks/get`` via JSON-RPC. Requires checkpoint_id to be present
        in the config.
        """
        merged = merge_configs(self.config, config)
        task_id = self._task_id(merged)
        if not task_id:
            raise ValueError(
                "No task ID found: set `checkpoint_id` in the config's "
                "`configurable` dict to the A2A task ID."
            )
        raw = self.sync_client.get_task(task_id)
        return self._task_to_snapshot(A2ATask.from_dict(raw))

    async def aget_state(
        self,
        config: RunnableConfig,
        *,
        subgraphs: bool = False,
    ) -> StateSnapshot:
        """Async version of `get_state`."""
        merged = merge_configs(self.config, config)
        task_id = self._task_id(merged)
        if not task_id:
            raise ValueError(
                "No task ID found: set `checkpoint_id` in the config's "
                "`configurable` dict to the A2A task ID."
            )
        raw = await self.client.get_task(task_id)
        return self._task_to_snapshot(A2ATask.from_dict(raw))

    def get_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[StateSnapshot]:
        """Yield a single state snapshot.

        A2A protocol does not expose a checkpoint history — only the current
        task state is available. Yields one entry for interface compatibility.
        """
        yield self.get_state(config)

    async def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[StateSnapshot]:
        """Async version of get_state_history."""
        yield await self.aget_state(config)

    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
    ) -> RunnableConfig:
        """Not supported — A2A agents are opaque.

        A2A agents are black boxes; their internal state cannot be modified
        directly. To continue a conversation, send a new message via invoke().
        """
        raise NotImplementedError(
            "A2A agents do not support direct state updates. "
            "Send a new message via invoke() instead."
        )

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
    ) -> RunnableConfig:
        """Async update_state — also not supported."""
        raise NotImplementedError(
            "A2A agents do not support direct state updates. "
            "Send a new message via ainvoke() instead."
        )

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

    # -- helpers ------------------------------------------------------------ #

    def _sanitize_config(self, config: RunnableConfig) -> RunnableConfig:
        """Sanitize config to make it safe for sending to remote A2A agents.

        Does three things:
            1. tags:         keep only string-typed tags
            2. metadata:     recursively drop non-JSON-serializable values
            3. configurable: drop LangGraph-internal keys + non-serializable values

        Aligned with RemoteGraph._sanitize_config logic.
        """
        sanitized: RunnableConfig = {}
        if "tags" in config:
            sanitized["tags"] = [t for t in config["tags"] if isinstance(t, str)]
        if "metadata" in config:
            sanitized["metadata"] = {
                k: sv
                for k, v in config["metadata"].items()
                if isinstance(k, str)
                and (sv := sanitize_config_value(v)) is not None
            }
        if "configurable" in config:
            sanitized["configurable"] = {
                k: sv
                for k, v in config["configurable"].items()
                if isinstance(k, str)
                and k not in _CONF_DROPLIST   # Drop internal keys
                and (sv := sanitize_config_value(v)) is not None  # Drop non-serializable
            }
        return sanitized

    def _build_a2a_message(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig,
    ) -> A2AMessage:
        """Build an A2A message from LangGraph input.

        Flow:
            1. Extract messages list from input
            2. Get the trailing user messages (_last_user_messages)
            3. Convert to A2A Parts (_messages_to_parts)
            4. Attach context_id (from thread_id) and task_id (from checkpoint_id)
        """
        if isinstance(input, dict):
            messages = input.get("messages", [])
        else:
            messages = [input] if input else []
        user_msgs = _last_user_messages(messages)
        parts = _messages_to_parts(user_msgs)
        return A2AMessage(
            message_id=str(uuid.uuid4()),
            role=Role.USER,
            parts=parts,
            context_id=self._context_id(config),
            task_id=self._task_id(config),
        )

    def _parse_send_response(self, result: dict[str, Any]) -> A2ATask:
        """Parse the message/send response.

        A2A spec allows two response formats:
            1. Task dict (has id + status) — standard format
            2. Message dict (has message_id) — some agents return a message directly

        For format 2, wraps it into a COMPLETED Task.
        """
        if "id" in result and "status" in result:
            return A2ATask.from_dict(result)
        if "message_id" in result:
            # Received a Message instead of a Task — wrap it
            msg = A2AMessage.from_dict(result)
            return A2ATask(
                id=msg.task_id or str(uuid.uuid4()),
                status=TaskStatus(state=TaskState.COMPLETED, message=msg),
                context_id=msg.context_id,
            )
        return A2ATask.from_dict(result)

    def _poll_until_done(
        self,
        task: A2ATask,
        *,
        initial_interval: float = 0.5,
        backoff_factor: float = 2.0,
        max_interval: float = 30.0,
        max_elapsed: float = 300.0,
    ) -> A2ATask:
        """Synchronously poll tasks/get until the task reaches a terminal or interrupted state.

        Uses exponential backoff:
            Wait sequence: 0.5s -> 1s -> 2s -> 4s -> 8s -> 16s -> 30s -> 30s -> ...
            Total timeout: 300s (5 minutes)

        Exponential backoff avoids hammering the agent during long-running tasks.
        """
        interval = initial_interval
        elapsed = 0.0
        while (
            not task.status.state.is_terminal
            and not task.status.state.is_interrupted
        ):
            if elapsed >= max_elapsed:
                raise TimeoutError(
                    f"A2A task {task.id} did not complete within {max_elapsed}s"
                )
            time.sleep(interval)
            elapsed += interval
            interval = min(interval * backoff_factor, max_interval)
            raw = self.sync_client.get_task(task.id)
            task = A2ATask.from_dict(raw)
        return task

    async def _apoll_until_done(
        self,
        task: A2ATask,
        *,
        initial_interval: float = 0.5,
        backoff_factor: float = 2.0,
        max_interval: float = 30.0,
        max_elapsed: float = 300.0,
    ) -> A2ATask:
        """Async polling; logic mirrors _poll_until_done, uses asyncio.sleep."""
        interval = initial_interval
        elapsed = 0.0
        while (
            not task.status.state.is_terminal
            and not task.status.state.is_interrupted
        ):
            if elapsed >= max_elapsed:
                raise TimeoutError(
                    f"A2A task {task.id} did not complete within {max_elapsed}s"
                )
            await asyncio.sleep(interval)
            elapsed += interval
            interval = min(interval * backoff_factor, max_interval)
            raw = await self.client.get_task(task.id)
            task = A2ATask.from_dict(raw)
        return task

    def _task_to_output(
        self,
        task: A2ATask,
        config: RunnableConfig | None,
    ) -> dict[str, Any]:
        """Process a completed A2A Task and convert to LangGraph output format.

        Three cases:
            1. Interrupted + called as subgraph -> raise GraphInterrupt
               (only raises in subgraph context, matching RemoteGraph behavior)
            2. Failed -> raise A2AClientError
            3. Normal -> return {"messages": [...]}
        """
        if task.status.state.is_interrupted:
            # Check if this is a subgraph call (checkpoint_ns present = called by parent)
            caller_ns = (config or {}).get(CONF, {}).get(CONFIG_KEY_CHECKPOINT_NS)
            if caller_ns:
                msg = ""
                if task.status.message:
                    texts = [p.text for p in task.status.message.parts if p.text]
                    msg = "\n".join(texts)
                raise GraphInterrupt(
                    [Interrupt(value=msg or task.status.state.value)]
                )
        if task.status.state == TaskState.FAILED:
            error_msg = "A2A task failed"
            if task.status.message and task.status.message.parts:
                texts = [p.text for p in task.status.message.parts if p.text]
                if texts:
                    error_msg = "\n".join(texts)
            raise A2AClientError(-1, error_msg)
        return {"messages": _task_to_ai_messages(task)}

    # -- stream ------------------------------------------------------------- #

    def _get_stream_modes(
        self,
        stream_mode: StreamMode | list[StreamMode] | None,
        config: RunnableConfig | None,
        default: StreamMode = "updates",
    ) -> tuple[list[StreamMode], list[StreamMode], bool, StreamProtocol | None]:
        """Parse stream_mode parameter. Aligned with RemoteGraph._get_stream_modes.

        Returns:
            - updated:    all modes to handle (including parent graph requirements)
            - requested:  modes explicitly requested by the user
            - req_single: True if user requested only one mode; affects output format:
                          single mode: yield data
                          multi mode:  yield StreamPart(event=mode, data=data)
            - stream:     parent graph's StreamProtocol (for propagating events upstream)
        """
        updated: list[StreamMode] = []
        req_single = True
        if stream_mode:
            if isinstance(stream_mode, str):
                updated.append(stream_mode)
            else:
                req_single = False           # Multiple modes
                updated.extend(stream_mode)
        else:
            updated.append(default)
        requested = updated.copy()
        # Check if parent graph requires additional stream modes
        stream: StreamProtocol | None = (
            (config or {}).get(CONF, {}).get(CONFIG_KEY_STREAM)
        )
        if stream:
            updated.extend(stream.modes)
        # Ensure "updates" is always present (needed internally)
        if "updates" not in updated:
            updated.append("updates")
        return updated, requested, req_single, stream

    # -- stream overloads (type signatures: v1 and v2 return different types) - #

    @overload
    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v2"],
        **kwargs: Any,
    ) -> Iterator[StreamPart]: ...

    @overload
    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v1"] = ...,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any] | Any]: ...

    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v1", "v2"] = "v1",
        **kwargs: Any,
    ) -> Iterator[dict[str, Any] | Any]:
        """Send a message and stream A2A events back.

        Selects strategy based on agent's capabilities.streaming:
            - True  -> _stream_sync (SSE streaming: message/stream)
            - False -> _stream_fallback_sync (fallback: message/send + polling)

        Defaults to True when no agent_card is available (optimistic strategy).
        """
        merged = merge_configs(self.config, config)
        sanitized = self._sanitize_config(merged)
        stream_modes, requested, req_single, parent_stream = (
            self._get_stream_modes(stream_mode, config)
        )

        msg = self._build_a2a_message(input, merged)
        supports_streaming = (
            self.agent_card.capabilities.streaming if self.agent_card else True
        )

        if supports_streaming:
            yield from self._stream_sync(
                msg, merged, sanitized, requested, req_single, parent_stream, version
            )
        else:
            yield from self._stream_fallback_sync(
                msg, merged, sanitized, requested, req_single, parent_stream, version
            )

    def _stream_sync(
        self,
        msg: A2AMessage,
        config: RunnableConfig,
        sanitized_config: RunnableConfig,
        requested: list[StreamMode],
        req_single: bool,
        parent_stream: StreamProtocol | None,
        version: str,
    ) -> Iterator[dict[str, Any] | Any]:
        """Process SSE streaming events (sync version).

        Three types of SSE events:
            1. Full Task (has id + status)        -> emit "values" event
            2. Status update (has task_id + status, no id) -> emit "updates" event
            3. Artifact (has task_id + artifact)   -> emit "updates" event

        Output format depends on version and req_single:
            - v2:          {"type": "values", "ns": (), "data": ..., "interrupts": ()}
            - v1 single:   data (yield bare data)
            - v1 multi:    StreamPart(event="values", data=data)
        """
        metadata = sanitized_config.get("metadata")
        last_task: A2ATask | None = None

        for event in self.sync_client.send_streaming_message(
            msg, metadata=metadata
        ):
            # SSE event may be wrapped in a "result" field (JSON-RPC format)
            event_data = event.get("result", event)

            # Case 1: Full Task update (contains id + status)
            if "id" in event_data and "status" in event_data:
                last_task = A2ATask.from_dict(event_data)
                output = {"messages": _task_to_ai_messages(last_task)}
                # Propagate to parent graph if running as subgraph
                if parent_stream and "values" in parent_stream.modes:
                    parent_stream(((), "values", output))
                if "values" in requested:
                    if version == "v2":
                        yield {"type": "values", "ns": (), "data": output, "interrupts": ()}
                    elif req_single:
                        yield output
                    else:
                        yield StreamPart(event="values", data=output)

            # Case 2: Status update (has task_id + status, but no full id -> not a full Task)
            elif "status" in event_data and "task_id" in event_data:
                status = TaskStatus.from_dict(event_data["status"])
                update = {self.name: {"status": status.state.value}}
                if parent_stream and "updates" in parent_stream.modes:
                    parent_stream(((), "updates", update))
                if "updates" in requested:
                    if version == "v2":
                        yield {"type": "updates", "ns": (), "data": update, "interrupts": ()}
                    elif req_single:
                        yield update
                    else:
                        yield StreamPart(event="updates", data=update)

            # Case 3: Artifact event (agent produced an output artifact)
            elif "artifact" in event_data and "task_id" in event_data:
                artifact = Artifact.from_dict(event_data["artifact"])
                texts = [p.text for p in artifact.parts if p.text]
                if texts:
                    ai_msg = {
                        "role": "assistant",
                        "type": "ai",
                        "content": "\n".join(texts),
                    }
                    update = {self.name: {"messages": [ai_msg]}}
                    if parent_stream and "updates" in parent_stream.modes:
                        parent_stream(((), "updates", update))
                    if "updates" in requested:
                        if version == "v2":
                            yield {"type": "updates", "ns": (), "data": update, "interrupts": ()}
                        elif req_single:
                            yield update
                        else:
                            yield StreamPart(event="updates", data=update)

        # After stream ends: if last task is interrupted + called as subgraph -> raise
        if last_task and last_task.status.state.is_interrupted:
            caller_ns = config.get(CONF, {}).get(CONFIG_KEY_CHECKPOINT_NS)
            if caller_ns:
                raise GraphInterrupt(
                    [Interrupt(value=last_task.status.state.value)]
                )

    def _stream_fallback_sync(
        self,
        msg: A2AMessage,
        config: RunnableConfig,
        sanitized_config: RunnableConfig,
        requested: list[StreamMode],
        req_single: bool,
        parent_stream: StreamProtocol | None,
        version: str,
    ) -> Iterator[dict[str, Any] | Any]:
        """Fallback: message/send -> poll tasks/get -> return final result.

        Used when the agent does not support SSE streaming.
        Emits a single "values" event with the complete result.
        """
        metadata = sanitized_config.get("metadata")
        raw = self.sync_client.send_message(msg, metadata=metadata)
        task = self._parse_send_response(raw)
        task = self._poll_until_done(task)
        output = self._task_to_output(task, config)

        if parent_stream and "values" in parent_stream.modes:
            parent_stream(((), "values", output))
        if "values" in requested:
            if version == "v2":
                yield {"type": "values", "ns": (), "data": output, "interrupts": ()}
            elif req_single:
                yield output
            else:
                yield StreamPart(event="values", data=output)

    # -- astream ------------------------------------------------------------ #

    @overload
    def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v2"],
        **kwargs: Any,
    ) -> AsyncIterator[StreamPart]: ...

    @overload
    def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v1"] = ...,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any] | Any]: ...

    async def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        subgraphs: bool = False,
        version: Literal["v1", "v2"] = "v1",
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        """Async version of ``stream``.

        Uses ``message/stream`` (SSE) via the async A2A client.
        Logic is symmetric with the sync version.
        """
        merged = merge_configs(self.config, config)
        sanitized = self._sanitize_config(merged)
        stream_modes, requested, req_single, parent_stream = (
            self._get_stream_modes(stream_mode, config)
        )

        msg = self._build_a2a_message(input, merged)
        supports_streaming = (
            self.agent_card.capabilities.streaming if self.agent_card else True
        )

        if supports_streaming:
            async for chunk in self._astream_sse(
                msg, merged, sanitized, requested, req_single, parent_stream, version
            ):
                yield chunk
        else:
            async for chunk in self._astream_fallback(
                msg, merged, sanitized, requested, req_single, parent_stream, version
            ):
                yield chunk

    async def _astream_sse(
        self,
        msg: A2AMessage,
        config: RunnableConfig,
        sanitized_config: RunnableConfig,
        requested: list[StreamMode],
        req_single: bool,
        parent_stream: StreamProtocol | None,
        version: str,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        """Async SSE streaming handler; logic mirrors _stream_sync."""
        metadata = sanitized_config.get("metadata")
        last_task: A2ATask | None = None

        async for event in self.client.send_streaming_message(
            msg, metadata=metadata
        ):
            event_data = event.get("result", event)

            # Full Task update
            if "id" in event_data and "status" in event_data:
                last_task = A2ATask.from_dict(event_data)
                output = {"messages": _task_to_ai_messages(last_task)}
                if parent_stream and "values" in parent_stream.modes:
                    parent_stream(((), "values", output))
                if "values" in requested:
                    if version == "v2":
                        yield {"type": "values", "ns": (), "data": output, "interrupts": ()}
                    elif req_single:
                        yield output
                    else:
                        yield StreamPart(event="values", data=output)

            # Status update
            elif "status" in event_data and "task_id" in event_data:
                status = TaskStatus.from_dict(event_data["status"])
                update = {self.name: {"status": status.state.value}}
                if parent_stream and "updates" in parent_stream.modes:
                    parent_stream(((), "updates", update))
                if "updates" in requested:
                    if version == "v2":
                        yield {"type": "updates", "ns": (), "data": update, "interrupts": ()}
                    elif req_single:
                        yield update
                    else:
                        yield StreamPart(event="updates", data=update)

            # Artifact event
            elif "artifact" in event_data and "task_id" in event_data:
                artifact = Artifact.from_dict(event_data["artifact"])
                texts = [p.text for p in artifact.parts if p.text]
                if texts:
                    ai_msg = {
                        "role": "assistant",
                        "type": "ai",
                        "content": "\n".join(texts),
                    }
                    update = {self.name: {"messages": [ai_msg]}}
                    if parent_stream and "updates" in parent_stream.modes:
                        parent_stream(((), "updates", update))
                    if "updates" in requested:
                        if version == "v2":
                            yield {"type": "updates", "ns": (), "data": update, "interrupts": ()}
                        elif req_single:
                            yield update
                        else:
                            yield StreamPart(event="updates", data=update)

        # Post-stream interrupt check
        if last_task and last_task.status.state.is_interrupted:
            caller_ns = config.get(CONF, {}).get(CONFIG_KEY_CHECKPOINT_NS)
            if caller_ns:
                raise GraphInterrupt(
                    [Interrupt(value=last_task.status.state.value)]
                )

    async def _astream_fallback(
        self,
        msg: A2AMessage,
        config: RunnableConfig,
        sanitized_config: RunnableConfig,
        requested: list[StreamMode],
        req_single: bool,
        parent_stream: StreamProtocol | None,
        version: str,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        """Async fallback; logic mirrors _stream_fallback_sync."""
        metadata = sanitized_config.get("metadata")
        raw = await self.client.send_message(msg, metadata=metadata)
        task = self._parse_send_response(raw)
        task = await self._apoll_until_done(task)
        output = self._task_to_output(task, config)

        if parent_stream and "values" in parent_stream.modes:
            parent_stream(((), "values", output))
        if "values" in requested:
            if version == "v2":
                yield {"type": "values", "ns": (), "data": output, "interrupts": ()}
            elif req_single:
                yield output
            else:
                yield StreamPart(event="values", data=output)

    # -- invoke ------------------------------------------------------------- #

    @overload
    def invoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v2"],
        **kwargs: Any,
    ) -> GraphOutput[dict[str, Any]]: ...

    @overload
    def invoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v1"] = ...,
        **kwargs: Any,
    ) -> dict[str, Any] | Any: ...

    def invoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v1", "v2"] = "v1",
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        """Send a message and wait for the A2A task to complete.

        Built on top of stream: consumes all events, returns the last chunk.
        Same implementation pattern as RemoteGraph.invoke.
        """
        # Consume the entire stream, keeping only the last value
        for chunk in self.stream(  # type: ignore[call-overload]
            input,
            config=config,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            stream_mode="values",
            version=version,
            **kwargs,
        ):
            pass
        try:
            if version == "v2":
                return GraphOutput(
                    value=chunk["data"],
                    interrupts=tuple(chunk.get("interrupts", ())),
                )
            return chunk
        except UnboundLocalError:
            # Stream produced no events (chunk is undefined)
            logger.warning("No events received from A2A agent")
            return None

    @overload
    async def ainvoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v2"],
        **kwargs: Any,
    ) -> GraphOutput[dict[str, Any]]: ...

    @overload
    async def ainvoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v1"] = ...,
        **kwargs: Any,
    ) -> dict[str, Any] | Any: ...

    async def ainvoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        version: Literal["v1", "v2"] = "v1",
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        """Async version of ``invoke``."""
        async for chunk in self.astream(  # type: ignore[call-overload]
            input,
            config=config,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            stream_mode="values",
            version=version,
            **kwargs,
        ):
            pass
        try:
            if version == "v2":
                return GraphOutput(
                    value=chunk["data"],
                    interrupts=tuple(chunk.get("interrupts", ())),
                )
            return chunk
        except UnboundLocalError:
            logger.warning("No events received from A2A agent")
            return None
