from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.language_models._compat_bridge import message_to_events
from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    ChatModelStream,
)
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_protocol.protocol import CheckpointRef, MessagesData
from typing_extensions import TypedDict

from langgraph.errors import GraphInterrupt
from langgraph.stream._event_log import EventLog
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.run_stream import BaseRunStream
from langgraph.stream.stream_channel import StreamChannel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.stream._mux import StreamMux


logger = logging.getLogger(__name__)


SubgraphStatus = Literal["started", "completed", "failed", "interrupted"]
_TERMINAL_STATUSES: frozenset[SubgraphStatus] = frozenset(
    {"completed", "failed", "interrupted"}
)


def _is_new_direct_child(
    ns: tuple[str, ...],
    scope: tuple[str, ...],
    seen: set[tuple[str, ...]] | dict[tuple[str, ...], Any],
) -> bool:
    """Return True iff `ns` is a direct child of `scope` not yet seen.

    Shared by `SubgraphTransformer` (in-process handle discovery) and
    `LifecycleTransformer` (wire event emission) so the two can't
    disagree on what counts as a new subgraph.
    """
    return len(ns) == len(scope) + 1 and ns[:-1] == scope and ns not in seen


def _parse_ns_segment(segment: str) -> tuple[str, str | None]:
    """Split `node_name:task_id` into (node_name, task_id).

    Task ids are present when Pregel spawned the subgraph as a task;
    absent on synthesized namespaces (tests, hand-crafted events).
    """
    node_name, sep, task_id = segment.partition(":")
    if not sep:
        return segment, None
    return node_name, task_id or None


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
    """

    _native = True

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: EventLog[ChatModelStream] = EventLog()
        # Correlate protocol events back to a ChatModelStream by run_id
        # (attached to the event's metadata by StreamMessagesHandler).
        self._by_run: dict[str, ChatModelStream] = {}
        self._pump_fn: Callable[[], bool] | None = None
        self._apump_fn: Callable[[], Awaitable[bool]] | None = None

    def init(self) -> dict[str, Any]:
        return {"messages": self._log}

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
        # Namespace filtering is handled by the mux via `scope_exact`.
        if event["method"] != "messages":
            return True
        params = event["params"]

        payload, metadata = params["data"]
        node: str | None = metadata.get("langgraph_node")
        run_id = str(metadata.get("run_id", "")) if metadata else ""

        if isinstance(payload, dict) and "event" in payload:
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
        event_type = event.get("event")
        if event_type == "message-start":
            message_id = event.get("message_id")
            stream = self._make_stream(
                namespace=list(self.scope),
                node=node,
                message_id=str(message_id) if message_id is not None else None,
            )
            self._by_run[run_id] = stream
            self._log.push(stream)
            stream.dispatch(event)
        elif run_id in self._by_run:
            stream = self._by_run[run_id]
            stream.dispatch(event)
            if event_type == "message-finish":
                del self._by_run[run_id]

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

    def fail(self, err: BaseException) -> None:
        """Propagate run error to any streams still open when the graph fails."""
        for stream in list(self._by_run.values()):
            stream.fail(err)
        self._by_run.clear()


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

    Handle fields:

    - `path`: the namespace tuple — stable for the life of the handle.
    - `graph_name` / `trigger_call_id`: parsed from the namespace
      segment at discovery (`node_name:task_id`).
    - `status`: `started` on discovery; advances to `completed` when
      the parent mux closes, or `failed` / `interrupted` when it
      errors.
    - `error`: set on terminal error.
    - `checkpoint`: unused by the current discovery path — kept for
      compatibility with consumers that inspect it.

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
        trigger_call_id: str | None = None,
    ) -> None:
        super().__init__(mux)
        self.path: tuple[str, ...] = path
        self.graph_name: str | None = graph_name
        self.trigger_call_id: str | None = trigger_call_id
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

    Thin dispatcher. At its own `scope` (inherited from
    `StreamTransformer`, determined by the enclosing mux), it watches
    for the first event at exactly one namespace level deeper to
    discover a direct child. Each discovered child gets its own
    `SubgraphRunStream` backed by a mini-`StreamMux` — built via
    `parent_mux.make_child(path)`, so the same factory list produces
    fresh transformer instances at the child's scope.

    Every incoming event that falls under one of the direct children
    (ns starts with a child's `path`) is forwarded into that child's
    mini-mux via `push`. The standard transformers in that mini-mux
    (`ValuesTransformer`, `MessagesTransformer`, and another
    `SubgraphTransformer` for grandchildren) handle the rest. No
    duplicated routing or assembly logic.

    Discovery is method-agnostic: the first event of any mode whose
    namespace places it directly below `scope` spawns the handle.
    `graph_name` and `trigger_call_id` are parsed from the namespace
    segment, which encodes `node_name:task_id`.

    Terminal status for each handle is set by the parent mux's
    `close` / `fail` path. `finalize` transitions still-open handles
    to `completed`; `fail` transitions them to `failed` or
    `interrupted` depending on the error.

    Native transformer — `subgraphs` exposes the direct-children log.

    `scope_exact = False`: this transformer sees events at any
    namespace, because it forwards out-of-scope events to the matching
    direct-child mini-mux.
    """

    _native = True
    scope_exact = False

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
        depth = len(self.scope)

        # 1. Discover: first-seen direct-child namespace registers a
        #    handle. Any event method triggers discovery — no dedicated
        #    channel.
        if _is_new_direct_child(ns, self.scope, self._by_ns):
            self._on_started(ns)

        # 2. Forward the event to the matching direct-child mini-mux.
        #    Prefix-match: ns must start with some child's path.
        direct_child_ns = ns[: depth + 1] if len(ns) > depth else None
        if direct_child_ns is not None and direct_child_ns in self._by_ns:
            self._by_ns[direct_child_ns]._mux.push(event)

        return True

    def _on_started(self, ns: tuple[str, ...]) -> None:
        # `_on_register` is called by the mux during registration, which
        # happens before any event can be dispatched — so this should
        # always be set by the time we process an event.
        assert self._mux is not None, (
            "SubgraphTransformer processed an event before _on_register; "
            "transformer registration ordering is broken."
        )
        graph_name, trigger_call_id = _parse_ns_segment(ns[-1])
        child_mux = self._mux.make_child(ns)
        handle = SubgraphRunStream(
            path=ns,
            mux=child_mux,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
        )
        self._by_ns[ns] = handle
        self._root_log.push(handle)

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
        """Transition any still-open direct children to `completed`.

        Subgraph interrupts surface as a values event with a populated
        `interrupts` field rather than as an exception at the graph
        boundary — the parent pump exhausts normally and `finalize`
        runs the close path. Inspect each child's `ValuesTransformer`
        to distinguish "completed cleanly" from "interrupted".
        """
        for handle in self._by_ns.values():
            if handle.status not in _TERMINAL_STATUSES:
                values_t = handle._mux.transformer_by_key("values")
                if isinstance(values_t, ValuesTransformer) and values_t._interrupted:
                    handle.status = "interrupted"
                else:
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


class LifecyclePayload(TypedDict, total=False):
    """Payload of a lifecycle event emitted by `LifecycleTransformer`."""

    event: SubgraphStatus
    namespace: list[str]
    graph_name: str | None
    trigger_call_id: str | None
    error: str | None


class LifecycleTransformer(StreamTransformer):
    """Synthesize subgraph lifecycle events from observed namespaces.

    Observes the same namespace signal `SubgraphTransformer` uses for
    in-process discovery and emits `started` / `completed` / `failed`
    / `interrupted` payloads onto its `lifecycle` channel. Consumers
    subscribed to that channel see the events in-process; wire
    consumers receive them as protocol events with `method:
    "lifecycle"` (unprefixed because this transformer is `_native`).

    No `running` event: the ns-discovery signal only fires once a
    subgraph has emitted output, so `started` already implies
    execution. Consumers needing finer-grained task-start visibility
    should read the `tasks` stream mode alongside.

    No root `started`: the run object itself signals run start.

    Terminal events are synthesized — `finalize` emits `completed` for
    still-open handles; `fail` emits `failed` or `interrupted`
    depending on whether the error is a `GraphInterrupt`.

    `scope_exact = False` so the transformer sees events at any
    namespace (needed for discovery of direct children).
    """

    _native = True
    scope_exact = False

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        # retain=True: lifecycle events are low-volume and consumers
        # commonly inspect them after draining `values`; without
        # retention those pushes would be dropped.
        self._channel: StreamChannel[LifecyclePayload] = StreamChannel(
            "lifecycle", retain=True
        )
        self._seen: set[tuple[str, ...]] = set()
        self._open: set[tuple[str, ...]] = set()

    def init(self) -> dict[str, Any]:
        return {"lifecycle": self._channel}

    def process(self, event: ProtocolEvent) -> bool:
        ns = tuple(event["params"]["namespace"])
        if _is_new_direct_child(ns, self.scope, self._seen):
            self._emit_started(ns)
        return True

    def _emit_started(self, ns: tuple[str, ...]) -> None:
        graph_name, trigger_call_id = _parse_ns_segment(ns[-1])
        self._seen.add(ns)
        self._open.add(ns)
        payload: LifecyclePayload = {
            "event": "started",
            "namespace": list(ns),
        }
        if graph_name:
            payload["graph_name"] = graph_name
        if trigger_call_id is not None:
            payload["trigger_call_id"] = trigger_call_id
        self._channel.push(payload)

    def finalize(self) -> None:
        """Emit `completed` for every still-open direct child."""
        for ns in list(self._open):
            self._channel.push({"event": "completed", "namespace": list(ns)})
        self._open.clear()

    def fail(self, err: BaseException) -> None:
        """Emit `failed` / `interrupted` for every still-open direct child.

        Closes the channel after emitting rather than letting the mux
        auto-fail it — the "failed" payload is the signal to
        consumers, so they should be able to iterate it. A failed
        channel would raise on iteration and hide the events that just
        got pushed.
        """
        is_interrupt = isinstance(err, GraphInterrupt)
        event_type: SubgraphStatus = "interrupted" if is_interrupt else "failed"
        error_str = None if is_interrupt else str(err)
        for ns in list(self._open):
            payload: LifecyclePayload = {
                "event": event_type,
                "namespace": list(ns),
            }
            if error_str is not None:
                payload["error"] = error_str
            self._channel.push(payload)
        self._open.clear()
        self._channel._close()
