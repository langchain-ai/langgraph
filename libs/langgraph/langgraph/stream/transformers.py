from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.language_models._compat_bridge import message_to_events
from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    ChatModelStream,
)
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_protocol.protocol import MessagesData
from typing_extensions import NotRequired, TypedDict

from langgraph.errors import GraphInterrupt
from langgraph.stream._event_log import EventLog
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.stream_channel import StreamChannel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class ValuesTransformer(StreamTransformer):
    """Capture values events as a drainable stream of state snapshots.

    Keeps `_latest` / `_interrupted` / `_interrupts` as scalar state
    regardless of whether the log has a subscriber — so `run.output()`
    and `run.interrupted` work without forcing the caller to iterate
    `run.values`. Log pushes are silent no-ops when unsubscribed.

    Native transformer — projection keys are exposed as direct
    attributes on the run stream (e.g. `run.values`).

    Only values events at the run's own level are captured; snapshots
    from deeper subgraphs are left in the main event log but excluded
    from the projection. "Own level" is defined by `parent_ns`, which
    `stream_v2` / `astream_v2` populate from the caller's checkpoint
    namespace so that a nested `stream_v2` call still sees its own
    root snapshots.
    """

    _native = True

    def __init__(self, *, parent_ns: tuple[str, ...] = ()) -> None:
        self._log: EventLog[dict[str, Any]] = EventLog()
        self._latest: dict[str, Any] | None = None
        self._interrupted = False
        self._interrupts: list[Any] = []
        self._parent_ns: list[str] = list(parent_ns)

    def init(self) -> dict[str, Any]:
        return {"values": self._log}

    @property
    def error(self) -> BaseException | None:
        """The error that ended the run, or `None` if it succeeded.

        Set by the mux when it auto-fails the projection log.
        """
        return self._log._error

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "values":
            return True
        params = event["params"]
        if params["namespace"] != self._parent_ns:
            return True
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

    Only events at the run's own level are projected; tokens from
    deeper subgraphs are left in the main event log but excluded from
    `.messages`. "Own level" is defined by `parent_ns`, which
    `stream_v2` / `astream_v2` populate from the caller's checkpoint
    namespace so that a `stream_v2` call inside a node still sees its
    own root chat model streams on `.messages`. Consumers that need
    subgraph tokens should iterate the raw event stream or register a
    custom transformer.

    Native transformer — the `messages` projection is exposed as a
    direct attribute on the run stream.
    """

    _native = True

    def __init__(self, *, parent_ns: tuple[str, ...] = ()) -> None:
        self._log: EventLog[ChatModelStream] = EventLog()
        # Correlate protocol events back to a ChatModelStream by run_id
        # (attached to the event's metadata by StreamMessagesHandler).
        self._by_run: dict[str, ChatModelStream] = {}
        self._pump_fn: Callable[[], bool] | None = None
        self._apump_fn: Callable[[], Awaitable[bool]] | None = None
        # Root scope for this projection. Only chat model streams whose
        # emitted namespace matches `parent_ns` are surfaced on
        # `.messages`; events from deeper subgraphs stay in the main
        # event log for other consumers but are not projected here.
        self._parent_ns: list[str] = list(parent_ns)

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
        if event["method"] != "messages":
            return True
        params = event["params"]
        if params["namespace"] != self._parent_ns:
            return True

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
                namespace=[],
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
        stream = self._make_stream(namespace=[], node=node, message_id=message.id)
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


SubgraphStatus = Literal["started", "completed", "failed", "interrupted"]


def _parse_ns_segment(segment: str) -> tuple[str, str | None]:
    """Split a namespace segment into `(graph_name, trigger_call_id)`.

    Segments are formatted `node_name:task_id` by `prepare_next_tasks`.
    Returns `(segment, None)` if no `:` is present.
    """
    name, sep, task_id = segment.partition(":")
    return name, task_id if sep else None


class LifecyclePayload(TypedDict, total=False):
    """Payload of a lifecycle event surfaced on the `lifecycle` channel.

    Auto-forwarded as `lifecycle` protocol events (no `custom:` prefix
    because `LifecycleTransformer` is a native transformer) so remote
    SDK clients receive the same data in-process consumers see via
    `run.lifecycle`.
    """

    event: SubgraphStatus
    namespace: list[str]
    graph_name: NotRequired[str]
    trigger_call_id: NotRequired[str]
    error: NotRequired[str]


class LifecycleTransformer(StreamTransformer):
    """Surface subgraph lifecycle as `lifecycle` protocol events.

    Subscribes to `tasks` events and emits `LifecyclePayload` to a
    `StreamChannel` named `lifecycle`. The channel is auto-forwarded
    by the mux so payloads land in the main event log under
    `method = "lifecycle"` (native transformer — no `custom:` prefix)
    — visible to remote SDK clients over the wire and to in-process
    consumers via `run.lifecycle`.

    Discovery: a `tasks` event tagged with a namespace one segment
    deeper than this transformer's scope, seen for the first time,
    triggers a `started` payload. Discovery uses `tasks` rather than
    arbitrary protocol events so the inference is silent-mode-safe
    when the consumer subscribes to `lifecycle` alone.

    Terminal state: a `TaskResultPayload` at the parent's namespace
    whose `id` matches an open child's encoded task id triggers the
    matching terminal payload (`completed` / `failed` / `interrupted`).
    `finalize` and `fail` are safety nets for subgraphs that didn't
    receive a parent-result event before the run ended.

    Native transformer — projection key `lifecycle` is exposed as
    `run.lifecycle`.
    """

    _native = True
    required_stream_modes = ("tasks",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._channel: StreamChannel[LifecyclePayload] = StreamChannel("lifecycle")
        self._seen: set[tuple[str, ...]] = set()
        # Maps direct-child namespace -> task_id of the parent task that
        # owns the child loop (encoded in the last segment of the child ns).
        self._open: dict[tuple[str, ...], str] = {}

    def init(self) -> dict[str, Any]:
        return {"lifecycle": self._channel}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "tasks":
            return True
        ns = tuple(event["params"]["namespace"])
        data = event["params"]["data"]

        if "result" in data:
            self._handle_task_result(ns, data)
        else:
            self._handle_task_start(ns)
        return True

    def _is_direct_child_ns(self, ns: tuple[str, ...]) -> bool:
        """True iff `ns` is exactly one segment deeper than this transformer's scope."""
        depth = len(self.scope)
        return len(ns) == depth + 1 and ns[:depth] == self.scope

    @staticmethod
    def _terminal_from_result(
        payload: dict[str, Any],
    ) -> tuple[SubgraphStatus, str | None]:
        """Map a `TaskResultPayload` to a `(status, error)` pair.

        Order matters: a result with both `error` and `interrupts`
        prefers the interrupt classification, since `GraphInterrupt`
        manifests as a populated `interrupts` list, not as `error`.
        """
        if payload.get("interrupts"):
            return "interrupted", None
        error = payload.get("error")
        if error:
            return "failed", str(error)
        return "completed", None

    def _handle_task_start(self, ns: tuple[str, ...]) -> None:
        if not self._is_direct_child_ns(ns):
            return
        if ns in self._seen:
            return
        self._seen.add(ns)
        graph_name, trigger_call_id = _parse_ns_segment(ns[-1])
        if trigger_call_id is None:
            # No task_id encoded — can't correlate a parent-result event
            # back to this namespace. Treat as a started-only entry; rely
            # on finalize/fail to close it.
            return
        self._open[ns] = trigger_call_id
        payload: LifecyclePayload = {"event": "started", "namespace": list(ns)}
        if graph_name:
            payload["graph_name"] = graph_name
        payload["trigger_call_id"] = trigger_call_id
        self._channel.push(payload)

    def _handle_task_result(self, ns: tuple[str, ...], data: dict[str, Any]) -> None:
        # A parent's task result closes any direct-child subgraph whose
        # owning task id matches this result's id.
        result_id = data.get("id")
        if not result_id:
            return
        for child_ns, parent_task_id in list(self._open.items()):
            if child_ns[:-1] != ns or parent_task_id != result_id:
                continue
            status, error = self._terminal_from_result(data)
            self._emit_terminal(child_ns, status, error)
            del self._open[child_ns]

    def _emit_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        payload: LifecyclePayload = {"event": status, "namespace": list(ns)}
        if error is not None:
            payload["error"] = error
        self._channel.push(payload)

    def finalize(self) -> None:
        """Emit `completed` for any subgraph still open at run end."""
        for ns in list(self._open):
            self._emit_terminal(ns, "completed", None)
        self._open.clear()

    def fail(self, err: BaseException) -> None:
        """Emit `failed` / `interrupted` for any subgraph still open at run failure."""
        is_interrupt = isinstance(err, GraphInterrupt)
        status: SubgraphStatus = "interrupted" if is_interrupt else "failed"
        error_str = None if is_interrupt else str(err)
        for ns in list(self._open):
            self._emit_terminal(ns, status, error_str)
        self._open.clear()
