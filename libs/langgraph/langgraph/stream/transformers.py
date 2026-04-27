from __future__ import annotations

import logging
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

    from langgraph.stream._mux import StreamMux
    from langgraph.stream.run_stream import (
        AsyncSubgraphRunStream,
        SubgraphRunStream,
    )

_logger = logging.getLogger(__name__)


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
    from the projection. "Own level" is defined by `scope`, which
    `stream_v2` / `astream_v2` populate from the caller's checkpoint
    namespace so that a nested `stream_v2` call still sees its own
    root snapshots.
    """

    _native = True
    required_stream_modes = ("values",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: EventLog[dict[str, Any]] = EventLog()
        self._latest: dict[str, Any] | None = None
        self._interrupted = False
        self._interrupts: list[Any] = []
        # Cached as a list once for cheap equality with the protocol
        # event's `namespace` field, which is `list[str]`.
        self._scope_list: list[str] = list(scope)

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
        if params["namespace"] != self._scope_list:
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
    `.messages`. "Own level" is defined by `scope`, which
    `stream_v2` / `astream_v2` populate from the caller's checkpoint
    namespace so that a `stream_v2` call inside a node still sees its
    own root chat model streams on `.messages`. Consumers that need
    subgraph tokens should iterate the raw event stream or register a
    custom transformer.

    Native transformer — the `messages` projection is exposed as a
    direct attribute on the run stream.
    """

    _native = True
    required_stream_modes = ("messages",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: EventLog[ChatModelStream] = EventLog()
        # Correlate protocol events back to a ChatModelStream by run_id
        # (attached to the event's metadata by StreamMessagesHandler).
        self._by_run: dict[str, ChatModelStream] = {}
        self._pump_fn: Callable[[], bool] | None = None
        self._apump_fn: Callable[[], Awaitable[bool]] | None = None
        # Cached as a list once for cheap equality with the protocol
        # event's `namespace` field, which is `list[str]`.
        self._scope_list: list[str] = list(scope)

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
        if params["namespace"] != self._scope_list:
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


class _TasksLifecycleBase(StreamTransformer):
    """Shared bookkeeping for `tasks`-event-driven lifecycle inference.

    Both `LifecycleTransformer` (wire-serializable channel) and
    `SubgraphTransformer` (in-process navigation handles) discover
    subgraphs by watching the same `tasks` stream — `started` on the
    first event at a tracked namespace, terminal status when the
    parent's `TaskResultPayload` arrives. Centralizing the dispatch
    + open-set bookkeeping here keeps the inference rules from
    drifting between the two surfaces.

    Subclasses provide three template-method hooks:

    - `_should_track(ns)` — scope filter (e.g. multi-depth vs
      direct-children-only).
    - `_on_started(ns, graph_name, trigger_call_id)` — first sighting
      action (push payload / build handle / etc.). Called once per
      discovered namespace.
    - `_on_terminal(ns, status, error)` — terminal action (push
      terminal payload / mark handle status). Called once per
      tracked namespace at result time, or via `finalize` / `fail`
      sweeps if no parent result arrived.

    Tasks events are suppressed from the main event log (`process`
    returns False) — they're folded into whichever projection the
    subclass populates; consumers iterating the raw protocol stream
    see the higher-level view.
    """

    required_stream_modes = ("tasks",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._seen: set[tuple[str, ...]] = set()
        # Maps tracked namespace -> task_id of the parent task whose
        # `TaskResultPayload` will close it.
        self._open: dict[tuple[str, ...], str] = {}

    # --- Template-method hooks (subclass overrides) ---

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        """Scope filter — return True iff `ns` is in this transformer's region."""
        raise NotImplementedError

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,
        trigger_call_id: str | None,
    ) -> None:
        """Fired once per discovered namespace (first observed task event)."""
        raise NotImplementedError

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        """Fired once per tracked namespace when its parent's result arrives,
        or via finalize/fail safety-net sweeps.
        """
        raise NotImplementedError

    # --- Dispatch + bookkeeping (shared) ---

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "tasks":
            return True
        ns = tuple(event["params"]["namespace"])
        data = event["params"]["data"]
        if "result" in data:
            self._handle_task_result(ns, data)
        else:
            self._handle_task_start(ns)
        # Tasks events are folded into the synthesized projections;
        # suppress from the main event log so iterators don't double-see
        # the same information in two shapes.
        return False

    def _handle_task_start(self, ns: tuple[str, ...]) -> None:
        if not self._should_track(ns) or ns in self._seen:
            return
        self._seen.add(ns)
        graph_name, trigger_call_id = _parse_ns_segment(ns[-1])
        self._on_started(ns, graph_name or None, trigger_call_id)
        if trigger_call_id is not None:
            self._open[ns] = trigger_call_id

    def _handle_task_result(self, ns: tuple[str, ...], data: dict[str, Any]) -> None:
        result_id = data.get("id")
        if not result_id:
            return
        for child_ns, parent_task_id in list(self._open.items()):
            if child_ns[:-1] != ns or parent_task_id != result_id:
                continue
            status, error = _terminal_from_result(data)
            self._on_terminal(child_ns, status, error)
            del self._open[child_ns]

    def finalize(self) -> None:
        """Emit `completed` for any tracked namespace still open at run end."""
        for ns in list(self._open):
            self._on_terminal(ns, "completed", None)
        self._open.clear()

    def fail(self, err: BaseException) -> None:
        """Emit `failed` / `interrupted` for any tracked namespace still open."""
        is_interrupt = isinstance(err, GraphInterrupt)
        status: SubgraphStatus = "interrupted" if is_interrupt else "failed"
        error_str = None if is_interrupt else str(err)
        for ns in list(self._open):
            self._on_terminal(ns, status, error_str)
        self._open.clear()


def _terminal_from_result(
    payload: dict[str, Any],
) -> tuple[SubgraphStatus, str | None]:
    """Map a `TaskResultPayload` to a `(status, error)` pair.

    Order matters: a result with both `error` and `interrupts` prefers
    the interrupt classification, since `GraphInterrupt` manifests as
    a populated `interrupts` list, not as `error`.
    """
    if payload.get("interrupts"):
        return "interrupted", None
    error = payload.get("error")
    if error:
        return "failed", str(error)
    return "completed", None


class LifecycleTransformer(_TasksLifecycleBase):
    """Surface subgraph lifecycle as `lifecycle` protocol events.

    Pushes `LifecyclePayload` to a `StreamChannel` named `lifecycle`.
    The channel is auto-forwarded by the mux so payloads land in the
    main event log under `method = "lifecycle"` (native transformer —
    no `custom:` prefix) — visible to remote SDK clients over the
    wire and to in-process consumers via `run.lifecycle`.

    Tracks subgraphs at every depth strictly below the transformer's
    scope, so a graph → subgraph → subgraph chain produces lifecycle
    events for both nested levels in a flat stream.

    Native transformer — projection key `lifecycle` is exposed as
    `run.lifecycle`.
    """

    _native = True

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._channel: StreamChannel[LifecyclePayload] = StreamChannel("lifecycle")

    def init(self) -> dict[str, Any]:
        return {"lifecycle": self._channel}

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        depth = len(self.scope)
        return len(ns) > depth and ns[:depth] == self.scope

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,
        trigger_call_id: str | None,
    ) -> None:
        if trigger_call_id is None:
            # Without a task id we can't correlate a parent-result
            # event back to this namespace — skip the started payload
            # and rely on finalize/fail to close.
            return
        payload: LifecyclePayload = {"event": "started", "namespace": list(ns)}
        if graph_name:
            payload["graph_name"] = graph_name
        payload["trigger_call_id"] = trigger_call_id
        self._channel.push(payload)

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        payload: LifecyclePayload = {"event": status, "namespace": list(ns)}
        if error is not None:
            payload["error"] = error
        self._channel.push(payload)


class SubgraphTransformer(_TasksLifecycleBase):
    """Discover subgraph invocations as in-process navigation handles.

    Per discovered direct-child subgraph, builds a `SubgraphRunStream`
    (or `AsyncSubgraphRunStream`) wrapping a child mini-mux scoped to
    the subgraph's namespace. Consumers iterate `run.subgraphs` to
    receive handles, then drill into `handle.values` / `handle.messages`
    / `handle.subgraphs` (recursive grandchildren) / `handle.lifecycle`.

    Each mini-mux owns its own scope and uses its own
    `SubgraphTransformer` to discover its direct children, so
    grandchildren live on the child handle — never on the root's
    `subgraphs` log. Forwarding non-tasks events into the matching
    child mini-mux is what keeps the child's projections populated.

    Native transformer — `subgraphs` is exposed as `run.subgraphs`.
    """

    _native = True
    supports_sync = True

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: EventLog[SubgraphRunStream | AsyncSubgraphRunStream] = EventLog()
        self._handles: dict[
            tuple[str, ...], SubgraphRunStream | AsyncSubgraphRunStream
        ] = {}
        self._mux: StreamMux | None = None

    def init(self) -> dict[str, Any]:
        return {"subgraphs": self._log}

    def _on_register(self, mux: Any) -> None:
        self._mux = mux

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        # Direct children only — grandchildren are picked up by the
        # child mini-mux's own SubgraphTransformer.
        depth = len(self.scope)
        return len(ns) == depth + 1 and ns[:depth] == self.scope

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,
        trigger_call_id: str | None,
    ) -> None:
        if self._mux is None:
            return
        try:
            child_mux = self._mux.make_child(ns)
        except RuntimeError:
            # Mux wasn't built from factories — no mini-mux navigation
            # available. Skip; LifecycleTransformer still tracks the
            # subgraph via the flat event stream.
            return
        # Late import dodges the run_stream → transformers cycle.
        from langgraph.stream.run_stream import (
            AsyncSubgraphRunStream,
            SubgraphRunStream,
        )

        values_t = child_mux.transformer_by_key("values")
        if not isinstance(values_t, ValuesTransformer):
            return
        handle_cls = AsyncSubgraphRunStream if child_mux.is_async else SubgraphRunStream
        handle = handle_cls(
            mux=child_mux,
            values_transformer=values_t,
            path=ns,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
        )
        self._handles[ns] = handle
        self._log.push(handle)

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        handle = self._handles.get(ns)
        if handle is None or handle._seen_terminal:
            return
        handle.status = status
        if error is not None and handle.error is None:
            handle.error = error
        handle._seen_terminal = True
        self._close_handle_mux(handle)

    def process(self, event: ProtocolEvent) -> bool:
        # Discover / update terminal status before forwarding so a
        # `started` handle exists by the time the child mini-mux sees
        # its own first event.
        keep = super().process(event)
        # Forward every event (tasks included) into the matching
        # direct-child mini-mux so the child's own transformers
        # populate its projections — including grandchild discovery.
        self._forward_to_children(event)
        return keep

    async def aprocess(self, event: ProtocolEvent) -> bool:
        # Async counterpart to `process`: repeat the tasks bookkeeping
        # here instead of delegating to `process`, so child mini-muxes
        # receive events through their async lane.
        if event["method"] == "tasks":
            ns = tuple(event["params"]["namespace"])
            data = event["params"]["data"]
            if "result" in data:
                await self._ahandle_task_result(ns, data)
            else:
                self._handle_task_start(ns)
            keep = False
        else:
            keep = True
        await self._aforward_to_children(event)
        return keep

    async def _ahandle_task_result(
        self, ns: tuple[str, ...], data: dict[str, Any]
    ) -> None:
        result_id = data.get("id")
        if not result_id:
            return
        for child_ns, parent_task_id in list(self._open.items()):
            if child_ns[:-1] != ns or parent_task_id != result_id:
                continue
            status, error = _terminal_from_result(data)
            await self._aon_terminal(child_ns, status, error)
            del self._open[child_ns]

    async def _aon_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        handle = self._handles.get(ns)
        if handle is None or handle._seen_terminal:
            return
        handle.status = status
        if error is not None and handle.error is None:
            handle.error = error
        handle._seen_terminal = True
        await self._aclose_handle_mux(handle)

    def _forward_to_children(self, event: ProtocolEvent) -> None:
        """Push the event into the matching direct-child mini-mux, if any.

        Events at a child's exact level (length `depth + 1`) and below
        (length `> depth + 1`) are routed into the child whose path
        matches the event's first `depth + 1` segments.
        """
        ns = tuple(event["params"]["namespace"])
        depth = len(self.scope)
        if len(ns) < depth + 1:
            return
        candidate_path = ns[: depth + 1]
        handle = self._handles.get(candidate_path)
        if handle is None or handle._mux is None:
            return
        try:
            handle._mux.push(event)
        except Exception:
            _logger.warning(
                "Error forwarding event to subgraph mini-mux at %s; "
                "subscribers may miss events.",
                handle.path,
                exc_info=True,
            )

    async def _aforward_to_children(self, event: ProtocolEvent) -> None:
        """Async counterpart to `_forward_to_children`."""
        ns = tuple(event["params"]["namespace"])
        depth = len(self.scope)
        if len(ns) < depth + 1:
            return
        candidate_path = ns[: depth + 1]
        handle = self._handles.get(candidate_path)
        if handle is None or handle._mux is None:
            return
        try:
            await handle._mux.apush(event)
        except Exception:
            _logger.warning(
                "Error forwarding event to subgraph mini-mux at %s; "
                "subscribers may miss events.",
                handle.path,
                exc_info=True,
            )

    def _close_handle_mux(
        self, handle: SubgraphRunStream | AsyncSubgraphRunStream
    ) -> None:
        if handle._mux is None or handle._mux._events._closed:
            return
        try:
            handle._mux.close()
        except Exception:
            _logger.warning(
                "Error closing subgraph mini-mux at %s; subscribers "
                "may not see a clean close.",
                handle.path,
                exc_info=True,
            )

    async def _aclose_handle_mux(
        self, handle: SubgraphRunStream | AsyncSubgraphRunStream
    ) -> None:
        if handle._mux is None or handle._mux._events._closed:
            return
        try:
            await handle._mux.aclose()
        except Exception:
            _logger.warning(
                "Error closing subgraph mini-mux at %s; subscribers "
                "may not see a clean close.",
                handle.path,
                exc_info=True,
            )

    def finalize(self) -> None:
        super().finalize()
        # Close any handles whose started fired without a trigger_call_id
        # (so `_open` never tracked them) — they finish implicitly at
        # run end with status `completed`.
        for handle in self._handles.values():
            if not handle._seen_terminal:
                handle.status = "completed"
                handle._seen_terminal = True
                self._close_handle_mux(handle)

    async def afinalize(self) -> None:
        for ns in list(self._open):
            await self._aon_terminal(ns, "completed", None)
        self._open.clear()
        # Close any handles whose started fired without a trigger_call_id
        # (so `_open` never tracked them) — they finish implicitly at
        # run end with status `completed`.
        for handle in self._handles.values():
            if not handle._seen_terminal:
                handle.status = "completed"
                handle._seen_terminal = True
                await self._aclose_handle_mux(handle)

    def fail(self, err: BaseException) -> None:
        is_interrupt = isinstance(err, GraphInterrupt)
        status: SubgraphStatus = "interrupted" if is_interrupt else "failed"
        error_str = None if is_interrupt else str(err)
        self._open.clear()
        for handle in self._handles.values():
            if not handle._seen_terminal:
                handle.status = status
                if error_str is not None and handle.error is None:
                    handle.error = error_str
                handle._seen_terminal = True
            if handle._mux is not None and not handle._mux._events._closed:
                try:
                    handle._mux.fail(err)
                except Exception:
                    _logger.warning(
                        "Error failing subgraph mini-mux at %s; "
                        "subscribers may not see the terminal error.",
                        handle.path,
                        exc_info=True,
                    )

    async def afail(self, err: BaseException) -> None:
        is_interrupt = isinstance(err, GraphInterrupt)
        status: SubgraphStatus = "interrupted" if is_interrupt else "failed"
        error_str = None if is_interrupt else str(err)
        self._open.clear()
        for handle in self._handles.values():
            if not handle._seen_terminal:
                handle.status = status
                if error_str is not None and handle.error is None:
                    handle.error = error_str
                handle._seen_terminal = True
            if handle._mux is not None and not handle._mux._events._closed:
                try:
                    await handle._mux.afail(err)
                except Exception:
                    _logger.warning(
                        "Error failing subgraph mini-mux at %s; "
                        "subscribers may not see the terminal error.",
                        handle.path,
                        exc_info=True,
                    )
