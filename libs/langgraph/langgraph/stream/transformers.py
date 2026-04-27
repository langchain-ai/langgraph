from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.language_models._compat_bridge import message_to_events
from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    ChatModelStream,
)
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_protocol.protocol import (
    CheckpointRef,
    LifecycleCause,
    LifecycleData,
    MessagesData,
)

from langgraph.errors import GraphInterrupt
from langgraph.stream._event_log import EventLog
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.run_stream import BaseRunStream

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.stream._mux import StreamMux


logger = logging.getLogger(__name__)


SubgraphStatus = Literal["started", "running", "completed", "failed", "interrupted"]
_TERMINAL_STATUSES: frozenset[SubgraphStatus] = frozenset(
    {"completed", "failed", "interrupted"}
)


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _to_chat_model_stream_event(event: MessagesData) -> MessagesData:
    """Convert wire-shaped message fields to ChatModelStream's internal shape."""
    event_type = event.get("event")
    converted: dict[str, Any] = dict(event)
    if (
        event_type == "message-start"
        and "message_id" not in converted
        and isinstance(converted.get("id"), str)
    ):
        converted["message_id"] = converted["id"]
    if (
        event_type in ("content-block-start", "content-block-delta", "content-block-finish")
        and "content_block" not in converted
        and isinstance(converted.get("content"), dict)
    ):
        converted["content_block"] = converted["content"]
    return cast("MessagesData", converted)


def _message_event_id(event: MessagesData) -> str | None:
    raw_id = event.get("id") or event.get("message_id")
    return str(raw_id) if raw_id is not None else None


def _content_block_start_skeleton(content: Any) -> dict[str, Any] | None:
    """Return a minimal content-block-start payload for a delta/finish block."""
    if not _is_record(content) or not isinstance(content.get("type"), str):
        return None

    block_type = content["type"]
    skeleton: dict[str, Any] = {"type": block_type}
    if block_type == "text":
        skeleton["text"] = ""
    elif block_type == "reasoning":
        skeleton["reasoning"] = ""
    elif block_type in ("tool_call", "tool_call_chunk"):
        skeleton["type"] = "tool_call_chunk"
        if isinstance(content.get("id"), str):
            skeleton["id"] = content["id"]
        if isinstance(content.get("name"), str):
            skeleton["name"] = content["name"]
        skeleton["args"] = ""
    elif block_type in ("server_tool_call", "server_tool_call_chunk"):
        skeleton["type"] = "server_tool_call_chunk"
        if isinstance(content.get("id"), str):
            skeleton["id"] = content["id"]
        if isinstance(content.get("name"), str):
            skeleton["name"] = content["name"]
        skeleton["args"] = ""
    return skeleton


def _copy_event(
    source: ProtocolEvent,
    *,
    method: str,
    namespace: list[str],
    data: Any,
) -> ProtocolEvent:
    params = {**source["params"], "namespace": namespace, "data": data}
    return {"type": "event", "method": method, "params": params}


def _message_repair_key(event: ProtocolEvent, run_id: str) -> str:
    namespace_key = "\x1f".join(event["params"]["namespace"])
    return f"{namespace_key}\x1e{run_id}"


def _extract_tool_calls_from_values(data: Any) -> dict[str, dict[str, Any]]:
    if not _is_record(data):
        return {}
    messages = data.get("messages")
    if not isinstance(messages, list):
        return {}
    known: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not _is_record(message):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not _is_record(tool_call):
                continue
            tool_call_id = tool_call.get("id")
            if not isinstance(tool_call_id, str):
                continue
            name = tool_call.get("name")
            args = tool_call.get("args")
            known[tool_call_id] = {
                "tool_name": name if isinstance(name, str) else "",
                "input": args if _is_record(args) else {},
            }
    return known


class ValuesTransformer(StreamTransformer):
    """Capture values events as a drainable stream of state snapshots.

    Keeps `_latest` / `_interrupted` / `_interrupts` as scalar state
    regardless of whether the log has a subscriber — so `run.output()`
    and `run.interrupted` work without forcing the caller to iterate
    `run.values`. Log pushes are silent no-ops when unsubscribed.

    Native transformer — projection keys are exposed as direct
    attributes on the run stream (e.g. `run.values`).

    `scope` (inherited from `StreamTransformer`) is the namespace the
    transformer captures values for. `()` matches the root graph;
    subgraph mini-muxes pass their subgraph's namespace, so each
    instance sees only its own level.
    """

    _native = True
    required_stream_modes = ("values",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: EventLog[dict[str, Any]] = EventLog()
        self._latest: dict[str, Any] | None = None
        self._interrupted = False
        self._interrupts: list[Any] = []

    def init(self) -> dict[str, Any]:
        return {"values": self._log}

    @property
    def error(self) -> BaseException | None:
        """The error that ended the run, or `None` if it succeeded.

        Set by the mux when it auto-fails the projection log.
        """
        return self._log._error

    def process(self, event: ProtocolEvent) -> bool:
        # Namespace filtering is handled by the mux via `scope_exact`.
        if event["method"] != "values":
            return True
        params = event["params"]
        self._latest = params["data"]
        interrupts = params.get("interrupts", ())
        if interrupts:
            self._interrupted = True
            self._interrupts.extend(interrupts)
        self._log.push(params["data"])
        return True


class ToolLifecycleTransformer(StreamTransformer):
    """Repair tool-start events needed for deterministic subagent discovery.

    Some subagent frameworks expose a tool-caused subgraph lifecycle before
    a LangChain tool callback has emitted the matching `tool-started`
    frame. Core can infer the missing start from the latest values snapshot
    (`messages[*].tool_calls`) and emit it before the lifecycle event leaves
    the mux, keeping remote clients from guessing from values snapshots.
    """

    scope_exact = False
    required_stream_modes = ("values", "tools", "lifecycle")

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._known_tool_calls: dict[str, dict[str, Any]] = {}
        self._emitted_tool_starts: set[str] = set()
        self._mux: StreamMux | None = None

    def init(self) -> dict[str, Any]:
        return {}

    def _on_register(self, mux: StreamMux) -> None:
        self._mux = mux

    def process(self, event: ProtocolEvent) -> bool:
        method = event["method"]
        data = event["params"]["data"]
        if method == "values":
            self._known_tool_calls.update(_extract_tool_calls_from_values(data))
            return True
        if method == "tools" and _is_record(data):
            if (
                data.get("event") == "tool-started"
                and isinstance(data.get("tool_call_id"), str)
            ):
                tool_call_id = cast("str", data["tool_call_id"])
                if tool_call_id in self._emitted_tool_starts:
                    return False
                self._emitted_tool_starts.add(tool_call_id)
            return True
        if method == "lifecycle":
            self._emit_missing_tool_started(event)
        return True

    def _emit_missing_tool_started(self, event: ProtocolEvent) -> None:
        if self._mux is None:
            return
        data = event["params"]["data"]
        if not _is_record(data) or data.get("event") != "started":
            return
        cause = data.get("cause")
        if not _is_record(cause) or cause.get("type") != "toolCall":
            return
        tool_call_id = cause.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            return
        if tool_call_id in self._emitted_tool_starts:
            return
        known = self._known_tool_calls.get(tool_call_id)
        if known is None:
            return

        self._emitted_tool_starts.add(tool_call_id)
        namespace = event["params"]["namespace"]
        self._mux.emit(
            _copy_event(
                event,
                method="tools",
                namespace=namespace[:-1],
                data={
                    "event": "tool-started",
                    "tool_call_id": tool_call_id,
                    "tool_name": known["tool_name"],
                    "input": known["input"],
                },
            )
        )


class MessagesTransformer(StreamTransformer):
    """Capture messages events as ChatModelStream objects.

    The messages projection yields one `ChatModelStream` (or
    `AsyncChatModelStream`) per LLM call. Consumers iterate
    `run.messages` to get stream handles, then use each handle's typed
    projections (`.text`, `.reasoning`, `.tool_calls`, `.usage`,
    `.output`) for per-message content.

    Two input shapes are handled (via `params["data"] = (payload,
    metadata)` from `StreamMessagesHandler`):

    1. Protocol event (dict with `"event"` key) — emitted by
       `stream_v2()` / `astream_v2()` via the `on_stream_event`
       callback. Routed to an existing `ChatModelStream` by
       `metadata["run_id"]`. A `message-start` event creates a new
       stream; `message-finish` closes it.
    2. Whole `AIMessage` — emitted from `on_chain_end` when a node
       returns a finalized message. Replayed as a synthetic protocol
       event lifecycle via `message_to_events`, then the
       already-complete stream is pushed to the log.

    V1 `AIMessageChunk` tuples (from `on_llm_new_token`) are not
    streamed into this projection: chat models that want to populate
    `run.messages` with content-block streaming must use
    `stream_v2()` / `astream_v2()`. Models called via the legacy
    `stream()` method still surface their final `AIMessage` via
    `on_chain_end` when a node returns it as state.

    `scope` (inherited from `StreamTransformer`) is the namespace the
    transformer captures messages for. `()` matches the root graph;
    subgraph mini-muxes pass their subgraph's namespace, so each
    instance sees only its own level.

    Native transformer — the `messages` projection is exposed as a
    direct attribute on the run stream.

    `scope_exact = False`: matches events at the transformer's own
    namespace **or** exactly one segment deeper (the chat-model /
    node's own task ns). Mirrors JS's root-feed filter
    (`namespaces=[[]], depth=1`) — root accepts depth-0 events plus
    its own nodes' depth-1 tokens; subgraph mini-muxes accept their
    own scope plus their internal nodes' tokens. Events deeper than
    scope + 1 are dropped (the enclosing `SubgraphTransformer` has
    already forwarded them to the matching child mini-mux).
    """

    _native = True
    scope_exact = False
    required_stream_modes = ("messages",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: EventLog[ChatModelStream] = EventLog()
        # Correlate protocol events back to a ChatModelStream by run_id
        # (attached to the event's metadata by StreamMessagesHandler).
        self._by_run: dict[str, ChatModelStream] = {}
        self._started_blocks: dict[str, set[int]] = {}
        self._mux: StreamMux | None = None
        self._pump_fn: Callable[[], bool] | None = None
        self._apump_fn: Callable[[], Awaitable[bool]] | None = None

    def init(self) -> dict[str, Any]:
        return {"messages": self._log}

    def _on_register(self, mux: StreamMux) -> None:
        self._mux = mux

    def _bind_pump(self, fn: Callable[[], bool]) -> None:
        """Wire the sync pull callback. Called by GraphRunStream._wire_request_more."""
        self._pump_fn = fn

    def _bind_apump(self, fn: Callable[[], Awaitable[bool]]) -> None:
        """Wire the async pull callback.

        Called by `AsyncGraphRunStream._wire_arequest_more` so each
        `AsyncChatModelStream` this transformer creates can drive the
        shared graph pump from its projection cursors.
        """
        self._apump_fn = fn

    def _make_stream(
        self,
        *,
        namespace: list[str],
        node: str | None,
        message_id: str | None,
    ) -> ChatModelStream:
        """Create a ChatModelStream (sync) or AsyncChatModelStream (async).

        Wires whichever pump is bound. Prefers the async pump so nested
        iteration under `AsyncGraphRunStream` drives the graph forward
        without a background task. The unwired fallback (no pump bound)
        is used by unit tests that dispatch events manually.
        """
        if self._apump_fn is not None:
            astream = AsyncChatModelStream(
                namespace=namespace,
                node=node,
                message_id=message_id,
            )
            astream.set_arequest_more(self._apump_fn)
            return astream
        if self._pump_fn is not None:
            stream: ChatModelStream = ChatModelStream(
                namespace=namespace,
                node=node,
                message_id=message_id,
            )
            stream.set_request_more(self._pump_fn)
            return stream
        return AsyncChatModelStream(
            namespace=namespace,
            node=node,
            message_id=message_id,
        )

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "messages":
            return True
        params = event["params"]
        # Accept events at our scope or exactly one segment deeper
        # (the chat-model / node's own task ns). Deeper events belong
        # to a subgraph and are routed by `SubgraphTransformer`.
        ns = tuple(params["namespace"])
        depth = len(self.scope)
        if ns[:depth] != self.scope:
            return True

        raw_data = params["data"]
        metadata: dict[str, Any] = {}
        if isinstance(raw_data, tuple) and len(raw_data) == 2:
            payload, raw_metadata = raw_data
            metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        else:
            payload = raw_data
        node = params.get("node")
        if not isinstance(node, str):
            node = metadata.get("langgraph_node")
            if not isinstance(node, str):
                node = None
        raw_run_id = params.get("run_id", metadata.get("run_id"))
        run_id = str(raw_run_id) if raw_run_id is not None else ""

        if isinstance(payload, dict) and "event" in payload:
            self._repair_content_block_lifecycle(
                event, cast("MessagesData", payload), run_id=run_id
            )
            if len(ns) > depth + 1:
                return True
            self._route_protocol_event(
                cast("MessagesData", payload), run_id=run_id, node=node
            )
        elif isinstance(payload, BaseMessage) and not isinstance(
            payload, AIMessageChunk
        ):
            self._route_whole_message(payload, node=node)
        # Legacy AIMessageChunk tuples (from on_llm_new_token) are ignored;
        # v1 streaming callers must switch to stream_v2() to populate this
        # projection.

        return True

    def _route_protocol_event(
        self,
        event: MessagesData,
        *,
        run_id: str,
        node: str | None,
    ) -> None:
        stream_event = _to_chat_model_stream_event(event)
        event_type = event.get("event")
        if event_type == "message-start":
            message_id = _message_event_id(event)
            stream = self._make_stream(
                namespace=list(self.scope),
                node=node,
                message_id=message_id,
            )
            self._by_run[run_id or message_id or ""] = stream
            self._log.push(stream)
            stream.dispatch(stream_event)
        elif run_id in self._by_run:
            stream = self._by_run[run_id]
            stream.dispatch(stream_event)
            if event_type == "message-finish":
                del self._by_run[run_id]

    def _repair_content_block_lifecycle(
        self,
        source: ProtocolEvent,
        event: MessagesData,
        *,
        run_id: str,
    ) -> None:
        if self._mux is None:
            return
        event_type = event.get("event")
        key = _message_repair_key(source, run_id)
        if event_type == "message-start":
            self._started_blocks[key] = set()
            return
        if event_type == "content-block-start":
            index = event.get("index")
            if isinstance(index, int):
                self._started_blocks.setdefault(key, set()).add(index)
            return
        if event_type in ("content-block-delta", "content-block-finish"):
            index = event.get("index")
            if not isinstance(index, int):
                return
            started = self._started_blocks.setdefault(key, set())
            if index in started:
                return
            skeleton = _content_block_start_skeleton(event.get("content"))
            if skeleton is None:
                return
            started.add(index)
            self._mux.emit(
                _copy_event(
                    source,
                    method="messages",
                    namespace=list(source["params"]["namespace"]),
                    data={
                        "event": "content-block-start",
                        "index": index,
                        "content": skeleton,
                    },
                )
            )
        elif event_type == "message-finish":
            self._started_blocks.pop(key, None)

    def _route_whole_message(self, message: BaseMessage, *, node: str | None) -> None:
        stream = self._make_stream(
            namespace=list(self.scope),
            node=node,
            message_id=message.id,
        )
        for evt in message_to_events(message, message_id=message.id):
            stream.dispatch(evt)
        self._log.push(stream)

    def finalize(self) -> None:
        """Clear any routing state — streams close themselves via `message-finish`."""
        self._by_run.clear()
        self._started_blocks.clear()

    def fail(self, err: BaseException) -> None:
        """Propagate run error to any streams still open when the graph fails."""
        for stream in list(self._by_run.values()):
            stream.fail(err)
        self._by_run.clear()
        self._started_blocks.clear()


class SubgraphRunStream(BaseRunStream):
    """Scoped view of a single nested subgraph execution.

    Yielded on `run.subgraphs` (or `parent.subgraphs` for grandchildren)
    when a nested `Pregel` spawns. Wraps a mini-`StreamMux` built with
    the same transformer factories as the root mux, so `.values`,
    `.messages`, `.subgraphs` are populated by the standard
    transformers scoped to this handle's namespace — no duplicated
    routing logic. The mini-mux borrows the root's pump via
    `make_child`'s pump inheritance, so any cursor on a subagent
    projection drives the whole run forward.

    Lifecycle fields update in place as events arrive:

    - `path`: the namespace tuple — stable for the life of the handle.
    - `graph_name` / `cause`: set once from the `started` payload.
      `cause` is populated by product-specific stream transformers
      (see `LifecycleCause` in the protocol definition); pregel itself
      emits no `cause`, so it may be `None` for subgraphs not covered
      by a product transformer.
    - `status`: advances `started` → `running` → `completed` /
      `failed` / `interrupted`.
    - `error` / `checkpoint`: set on the terminal event when present.

    `.output` is a snapshot of the latest values seen at this
    namespace — it doesn't drive the pump (unlike root's
    `GraphRunStream.output`), because advancing a subgraph to
    completion is only meaningful as part of advancing the whole run.
    """

    def __init__(
        self,
        path: tuple[str, ...],
        mux: StreamMux,
        *,
        graph_name: str | None = None,
        cause: LifecycleCause | None = None,
    ) -> None:
        super().__init__(mux)
        self.path: tuple[str, ...] = path
        self.graph_name: str | None = graph_name
        self.cause: LifecycleCause | None = cause
        self.status: SubgraphStatus = "started"
        self.error: str | None = None
        self.checkpoint: CheckpointRef | None = None

    @property
    def output(self) -> dict[str, Any] | None:
        """Latest values snapshot at this namespace, or `None`.

        Snapshot-only — iterating other projections or the root's
        `.output` is what drives the pump.
        """
        values_t = self._mux.transformer_by_key("values")
        if isinstance(values_t, ValuesTransformer):
            return values_t._latest
        return None


class SubgraphTransformer(StreamTransformer):
    """Discover subgraphs and route events into per-subgraph mini-muxes.

    Thin state-machine + dispatcher. At its own `scope` (inherited
    from `StreamTransformer`, determined by the enclosing mux), it
    watches for `lifecycle` events at exactly one level deeper to
    discover direct children. Each discovered child gets its own
    `SubgraphRunStream` backed by a mini-`StreamMux` — built via
    `parent_mux.make_child(path)`, so the same factory list produces
    fresh transformer instances at the child's scope.

    Every incoming event that falls under one of the direct children
    (ns starts with a child's `path`) is forwarded into that child's
    mini-mux via `push`. The standard transformers in that mini-mux
    (`ValuesTransformer`, `MessagesTransformer`, and another
    `SubgraphTransformer` for grandchildren) handle the rest. No
    duplicated routing or assembly logic.

    Lifecycle state for each handle (running / completed / failed /
    interrupted) is updated in place as events fire. On terminal
    events, the handle's mini-mux is closed so any subscribed cursors
    unblock. `finalize` / `fail` handle dangling handles left mid-run.

    Native transformer — `subgraphs` exposes the direct-children log.

    `scope_exact = False`: this transformer sees events at any
    namespace, because it forwards out-of-scope events to the matching
    direct-child mini-mux.
    """

    _native = True
    scope_exact = False
    required_stream_modes = ("lifecycle",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._root_log: EventLog[SubgraphRunStream] = EventLog()
        # Direct children only (namespace = scope + one segment).
        self._by_ns: dict[tuple[str, ...], SubgraphRunStream] = {}
        self._mux: StreamMux | None = None

    def init(self) -> dict[str, Any]:
        return {"subgraphs": self._root_log}

    def _on_register(self, mux: StreamMux) -> None:
        """Capture the enclosing mux so we can build child mini-muxes."""
        self._mux = mux

    def process(self, event: ProtocolEvent) -> bool:
        ns = tuple(event["params"]["namespace"])
        method = event["method"]
        depth = len(self.scope)

        # 1. On `started` for a direct child (ns depth = mine + 1 and
        #    ns prefix matches mine), register the handle.
        if method == "lifecycle" and len(ns) == depth + 1 and ns[:-1] == self.scope:
            data = cast(LifecycleData, event["params"]["data"])
            if data.get("event") == "started":
                self._on_started(ns, data)

        # 2. Forward the event to the matching direct-child mini-mux
        #    before the status-change step below so that terminal events
        #    reach the child's log and grandchild transformers *before*
        #    the child's mini-mux is closed. Prefix-match: ns must start
        #    with some child's path.
        direct_child_ns = ns[: depth + 1] if len(ns) > depth else None
        if direct_child_ns is not None and direct_child_ns in self._by_ns:
            self._by_ns[direct_child_ns]._mux.push(event)

        # 3. Status change for a direct child (ns = child's path, method
        #    = lifecycle). Update handle fields, close mini-mux on
        #    terminal.
        if (
            method == "lifecycle"
            and ns in self._by_ns
            and len(ns) == depth + 1
            and ns[:-1] == self.scope
        ):
            data = cast(LifecycleData, event["params"]["data"])
            event_type = data.get("event")
            if event_type in ("running", "completed", "failed", "interrupted"):
                self._on_status_change(ns, event_type, data)

        return True

    def _on_started(self, ns: tuple[str, ...], data: LifecycleData) -> None:
        if ns in self._by_ns:
            # Duplicate started — ignore.
            return
        # `_on_register` is called by the mux during registration, which
        # happens before any event can be dispatched — so this should
        # always be set by the time we process an event.
        assert self._mux is not None, (
            "SubgraphTransformer processed an event before _on_register; "
            "transformer registration ordering is broken."
        )
        child_mux = self._mux.make_child(ns)
        handle = SubgraphRunStream(
            path=ns,
            mux=child_mux,
            graph_name=data.get("graph_name"),
            cause=data.get("cause"),
        )
        self._by_ns[ns] = handle
        self._root_log.push(handle)

    def _on_status_change(
        self,
        ns: tuple[str, ...],
        event_type: SubgraphStatus,
        data: LifecycleData,
    ) -> None:
        handle = self._by_ns[ns]
        handle.status = event_type
        err = data.get("error")
        if err is not None:
            handle.error = err
        checkpoint = data.get("checkpoint")
        if checkpoint is not None:
            handle.checkpoint = checkpoint
        if event_type in _TERMINAL_STATUSES:
            self._close_handle_mux(handle)

    @staticmethod
    def _close_handle_mux(handle: SubgraphRunStream) -> None:
        # Idempotent close — mux.close() runs finalize on its transformers
        # (which cascades through grandchildren) and closes projection logs.
        if not handle._mux._events._closed:
            try:
                handle._mux.close()
            except Exception:
                logger.warning(
                    "Error closing subgraph mini-mux at %s; subscribers "
                    "may not see a clean close.",
                    handle.path,
                    exc_info=True,
                )

    def finalize(self) -> None:
        """Transition any still-open direct children to `completed`."""
        for handle in self._by_ns.values():
            if handle.status not in _TERMINAL_STATUSES:
                handle.status = "completed"
            self._close_handle_mux(handle)

    def fail(self, err: BaseException) -> None:
        """Transition any still-open direct children to `failed` / `interrupted`."""
        is_interrupt = isinstance(err, GraphInterrupt)
        terminal: SubgraphStatus = "interrupted" if is_interrupt else "failed"
        error_str = None if is_interrupt else str(err)
        for handle in self._by_ns.values():
            if handle.status not in _TERMINAL_STATUSES:
                handle.status = terminal
                if error_str is not None and handle.error is None:
                    handle.error = error_str
            if not handle._mux._events._closed:
                try:
                    handle._mux.fail(err)
                except Exception:
                    logger.warning(
                        "Error failing subgraph mini-mux at %s; subscribers "
                        "may not see the terminal error.",
                        handle.path,
                        exc_info=True,
                    )
