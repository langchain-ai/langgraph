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

from langgraph.errors import GraphDrained, GraphInterrupt
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.run_stream import AsyncSubgraphRunStream, SubgraphRunStream
from langgraph.stream.stream_channel import StreamChannel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.stream._mux import StreamMux

_logger = logging.getLogger(__name__)


class ValuesTransformer(StreamTransformer):
    """Capture values events as a drainable stream of state snapshots.

    Provides the `run.values` projection. `run.output`,
    `run.interrupted` and `run.interrupts` are tracked directly
    by the run stream and do not depend on this transformer.

    Native transformer — projection keys are exposed as direct
    attributes on the run stream (e.g. `run.values`).

    Only values events at the run's own level are captured; snapshots
    from deeper subgraphs are left in the main event log but excluded
    from the projection. "Own level" is defined by `scope`, which
    `stream_events(version="v3")` / `astream_events(version="v3")` populate from the caller's
    checkpoint namespace so that a nested `stream_events(version="v3")` call still
    sees its own root snapshots.
    """

    _native = True
    required_stream_modes = ("values",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict[str, Any]] = StreamChannel()
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


class CustomTransformer(StreamTransformer):
    """Capture custom events as a drainable stream of arbitrary payloads.

    Nodes emit custom data via `get_stream_writer()`. This transformer
    surfaces those events on `run.custom` as a `StreamChannel[Any]`,
    preserving payloads in arrival order.

    Only events at the run's own scope are captured; custom data from
    deeper subgraphs is available on the respective subgraph handle's
    `.custom` projection.

    Native transformer — `run.custom` is a direct attribute.
    """

    _native = True
    required_stream_modes = ("custom",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[Any] = StreamChannel()
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        return {"custom": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "custom":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
        self._log.push(params["data"])
        return True


class UpdatesTransformer(StreamTransformer):
    """Capture updates events as a drainable stream of node outputs.

    Surfaces `stream_mode="updates"` data on `run.updates` as a
    `StreamChannel[dict[str, Any]]`. Each item is a dict mapping a node
    (or task) name to the update it returned after a step.

    Only events at the run's own scope are captured; updates from deeper
    subgraphs are available on the respective subgraph handle's
    `.updates` projection.

    Native transformer — `run.updates` is a direct attribute.
    """

    _native = True
    required_stream_modes = ("updates",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict[str, Any]] = StreamChannel()
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        return {"updates": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "updates":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
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
       `stream_events(version="v3")` / `astream_events(version="v3")` via the `on_stream_event`
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
    `stream_events(version="v3")` / `astream_events(version="v3")`. Models called via the legacy
    `stream()` method still surface their final `AIMessage` via
    `on_chain_end` when a node returns it as state.

    Only events at the run's own level are projected; tokens from
    deeper subgraphs are left in the main event log but excluded from
    `.messages`. "Own level" is defined by `scope`, which
    `stream_events(version="v3")` / `astream_events(version="v3")` populate from the caller's checkpoint
    namespace so that a `stream_events(version="v3")` call inside a node still sees its
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
        self._log: StreamChannel[ChatModelStream] = StreamChannel()
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
        # v1 streaming callers must switch to stream_events(version="v3") to populate this
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


LifecycleEvent = Literal["started", "completed", "failed", "interrupted", "drained"]
"""State transition surfaced on the `lifecycle` channel for a tracked subgraph.

Each value:

- `started` — first `tasks` event observed at the tracked namespace. Carries
  `graph_name` / `parent_task_id` / optional `metadata` describing what spawned
  the subgraph.
- `completed` — the dispatching task's `TaskResultPayload` arrived with neither
  error nor interrupts. Also emitted by `finalize` for any tracked namespace
  still open at run end.
- `failed` — the dispatching task's result carried an `error`, OR the run
  failed at top level with a non-interrupt / non-drain exception. Carries
  `error` (string).
- `interrupted` — the dispatching task's result carried `interrupts` (takes
  precedence over `error` if both present), OR the run failed with
  `GraphInterrupt`.
- `drained` — the run was cooperatively stopped at a superstep boundary via
  `RunControl.request_drain()` (e.g. SIGTERM). The checkpoint is saved and
  the run is resumable.
"""


def _parse_ns_segment(segment: str) -> tuple[str, str | None]:
    """Split a namespace segment into `(graph_name, parent_task_id)`.

    Segments are formatted `node_name:task_id` by `prepare_next_tasks`.
    Returns `(segment, None)` if no `:` is present.
    """
    name, sep, task_id = segment.partition(":")
    return name, task_id if sep else None


def _extract_dispatching_tool_call_id(payload: Any) -> str | None:
    """Return the model-side `tool_call_id` from a per-call dispatched task's
    `input`, or `None` if the payload doesn't match a recognised shape.

    Two shapes are recognised; both are duck-typed so any tool runner
    that mimics the layout participates without naming any specific
    dispatcher's types:

    1. Single-element list of tool-call dicts:
       `[{"id": ..., "name": ..., "args": {...}}]`. The current public
       shape — `langchain.agents.create_agent` Send-fans this out per
       pending tool call.
    2. Dict envelope wrapping a tool call:
       `{"tool_call": {"id": ..., "args": {...}, ...}, ...}`. Older
       prebuilt agent paths Send-fan-out this shape.
    """
    if isinstance(payload, dict):
        tool_call = payload.get("tool_call")
        if not isinstance(tool_call, dict):
            return None
    elif (
        isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict)
    ):
        tool_call = payload[0]
    else:
        return None
    raw_id = tool_call.get("id")
    return raw_id if isinstance(raw_id, str) else None


class LifecyclePayload(TypedDict, total=False):
    """Payload of a lifecycle event surfaced on the `lifecycle` channel.

    Auto-forwarded as `lifecycle` protocol events (no `custom:` prefix
    because `LifecycleTransformer` is a native transformer) so remote
    SDK clients receive the same data in-process consumers see via
    `run.lifecycle`.
    """

    event: LifecycleEvent
    """State transition. See `LifecycleEvent` for per-value semantics."""
    namespace: list[str]
    """Checkpoint namespace of the subgraph the event is about. Always present.

    A list of `node_name:task_id` segments, one per nesting level (root has
    `[]`, a direct child of root has `["agent:abc123"]`, a grandchild has
    `["agent:abc123", "tool:def456"]`, etc.). Stable identity across the
    `started → terminal` pair for the same subgraph instance.
    """
    graph_name: NotRequired[str]
    """Name of the parent-scope node that dispatched this subgraph
    (`add_node` name, surrounding tool's name for in-tool invokes,
    `Send` target name, etc.) — parsed from the namespace tail
    segment. Absent when the segment has no `:` separator.
    """
    parent_task_id: str
    """Pregel task id of the dispatching task — the task whose execution
    spawned this subgraph.

    Always present on every event for the same subgraph instance. This is
    the join key for correlating `started` ↔ terminal events and for
    matching a `started` back to its `tasks` parent. Each Send produces
    its own pregel task with its own id, so the join is 1:1 even when a
    model dispatches multiple parallel tool calls in one turn.
    """
    metadata: NotRequired[dict[str, Any]]
    """Optional generic descriptor of *what triggered* this subgraph.
    Forwarded by protocol layers as the wire `lifecycle.started.metadata` field.

    Shape:

    - `{"type": "tool_call", "tool_call_id": "<id>"}` — set when the subgraph
      was triggered by a per-call tool dispatch (a model tool call routed
      through whatever tool node the agent uses). `tool_call_id` is the
      model-side id of the originating tool call, exposed so UI consumers
      can anchor the lifecycle event back to the AI message that dispatched
      it. The langgraph layer deliberately doesn't mine `args` — those live
      on the AIMessage's `tool_calls[i].args` already, and consumers that
      want descriptive intent (subagent type, prompt text, etc.) look it
      up there to keep one source of truth.

    Absent for structurally-triggered subgraphs (parallel branches via
    `Send` with non-tool-call payloads, nested `graph.invoke()`, etc.) and
    for tool dispatches whose envelope carried no `id`.
    """
    error: NotRequired[str]
    """Error string. Set on `failed` events; absent otherwise."""


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
    - `_on_started(ns, graph_name, parent_task_id, tool_call_id)` —
      first sighting action (push payload / build handle / etc.).
      Called once per discovered namespace.
    - `_on_terminal(ns, status, error, parent_task_id)` — terminal
      action (push terminal payload / mark handle status). Called
      once per tracked namespace at result time, or via `finalize` /
      `fail` sweeps if no parent result arrived.

    Tasks events are suppressed from the main event log (`process`
    returns False) — they're folded into whichever projection the
    subclass populates; consumers iterating the raw protocol stream
    see the higher-level view.
    """

    required_stream_modes = ("tasks",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._seen: set[tuple[str, ...]] = set()
        # Maps tracked namespace -> task_id of the dispatching task whose
        # `TaskResultPayload` will close it.
        self._open: dict[tuple[str, ...], str] = {}
        # Maps task_id -> model-side `tool_call_id` for tasks whose `input`
        # matched a recognized per-call tool-dispatch shape. The lifecycle
        # hook joins on this when a child subgraph fires its first task
        # event so it can anchor the lifecycle.started to the originating
        # AI message tool call.
        self._dispatching_tool_call_id: dict[str, str] = {}

    # --- Template-method hooks (subclass overrides) ---

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        """Scope filter — return True iff `ns` is in this transformer's region."""
        raise NotImplementedError

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,
        parent_task_id: str | None,
        tool_call_id: str | None = None,
    ) -> None:
        """Fired once per discovered namespace (first observed task event).

        `tool_call_id` is the model-side id of the originating tool call
        (from the per-call dispatched task's `input`). `None` for
        structurally-triggered subgraphs or per-call envelopes that omitted
        an `id`. Consumers join on `parent_task_id` (the pregel task id)
        for identity; `tool_call_id` is purely an anchor back to the AI
        message that dispatched the subgraph.
        """
        raise NotImplementedError

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: LifecycleEvent,
        error: str | None,
        parent_task_id: str,
    ) -> None:
        """Fired once per tracked namespace when its dispatching task's
        result arrives, or via finalize/fail safety-net sweeps.

        `parent_task_id` is the same id paired with the namespace at
        `_on_started` time, so subscribers can correlate the terminal
        event back to its `started`.
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
            self._handle_task_start(ns, data)
        # Tasks events are folded into the synthesized projections;
        # suppress from the main event log so iterators don't double-see
        # the same information in two shapes.
        return False

    def _handle_task_start(self, ns: tuple[str, ...], data: dict[str, Any]) -> None:
        # Mine input shape on every tasks event (not just tracked ones)
        # so we capture dispatching tasks that themselves live outside the
        # tracked region but whose `id` will appear as `parent_task_id`
        # for a child subgraph.
        self._record_dispatching_tool_call_id(data)
        if not self._should_track(ns) or ns in self._seen:
            return
        self._seen.add(ns)
        graph_name, parent_task_id = _parse_ns_segment(ns[-1])
        tool_call_id = (
            self._dispatching_tool_call_id.pop(parent_task_id, None)
            if parent_task_id is not None
            else None
        )
        self._on_started(
            ns,
            graph_name or None,
            parent_task_id,
            tool_call_id,
        )
        if parent_task_id is not None:
            self._open[ns] = parent_task_id

    def _record_dispatching_tool_call_id(self, data: dict[str, Any]) -> None:
        """Remember `task_id -> tool_call_id` if the task input matches
        a recognized per-call tool-dispatch shape.

        Shape detection and id extraction both live in
        `_extract_dispatching_tool_call_id`; this method just records the
        mapping under the dispatching task's own `id` so the lifecycle hook
        can anchor a child subgraph back to the originating AI message
        tool call when that subgraph's first task event arrives.
        """
        task_id = data.get("id")
        if not isinstance(task_id, str):
            return
        tool_call_id = _extract_dispatching_tool_call_id(data.get("input"))
        if tool_call_id is None:
            return
        self._dispatching_tool_call_id[task_id] = tool_call_id

    def _pop_terminal_transitions(
        self, ns: tuple[str, ...], data: dict[str, Any]
    ) -> list[tuple[tuple[str, ...], LifecycleEvent, str | None, str]]:
        """Return and remove tracked children closed by this task result.

        Each tuple is `(child_ns, status, error, parent_task_id)`.
        `parent_task_id` is the dispatching task's id — the same id
        we'd already paired with the namespace at `_on_started`.
        """
        result_id = data.get("id")
        if not result_id:
            return []
        transitions: list[tuple[tuple[str, ...], LifecycleEvent, str | None, str]] = []
        for child_ns, dispatching_task_id in list(self._open.items()):
            if child_ns[:-1] != ns or dispatching_task_id != result_id:
                continue
            status, error = _terminal_from_result(data)
            transitions.append((child_ns, status, error, dispatching_task_id))
            del self._open[child_ns]
        return transitions

    def _handle_task_result(self, ns: tuple[str, ...], data: dict[str, Any]) -> None:
        for child_ns, status, error, parent_task_id in self._pop_terminal_transitions(
            ns, data
        ):
            self._on_terminal(child_ns, status, error, parent_task_id)

    def finalize(self) -> None:
        """Emit `completed` for any tracked namespace still open at run end."""
        for ns, parent_task_id in list(self._open.items()):
            self._on_terminal(ns, "completed", None, parent_task_id)
        self._open.clear()

    def fail(self, err: BaseException) -> None:
        """Emit terminal status for any tracked namespace still open."""
        status, error_str = _status_from_exception(err)
        for ns, parent_task_id in list(self._open.items()):
            self._on_terminal(ns, status, error_str, parent_task_id)
        self._open.clear()


def _status_from_exception(err: BaseException) -> tuple[LifecycleEvent, str | None]:
    """Map a run exception to a subgraph terminal status and error string."""
    if isinstance(err, GraphDrained):
        return "drained", None
    if isinstance(err, GraphInterrupt):
        return "interrupted", None
    return "failed", str(err)


def _terminal_from_result(
    payload: dict[str, Any],
) -> tuple[LifecycleEvent, str | None]:
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
        parent_task_id: str | None,
        tool_call_id: str | None = None,
    ) -> None:
        if parent_task_id is None:
            # Without a task id we can't correlate a dispatching-task-result
            # event back to this namespace — skip the started payload
            # and rely on finalize/fail to close.
            return
        payload: LifecyclePayload = {
            "event": "started",
            "namespace": list(ns),
            "parent_task_id": parent_task_id,
        }
        if graph_name:
            payload["graph_name"] = graph_name
        if tool_call_id is not None:
            payload["metadata"] = {"type": "tool_call", "tool_call_id": tool_call_id}
        self._channel.push(payload)

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: LifecycleEvent,
        error: str | None,
        parent_task_id: str,
    ) -> None:
        payload: LifecyclePayload = {
            "event": status,
            "namespace": list(ns),
            "parent_task_id": parent_task_id,
        }
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
    `subgraphs` log. Forwarding events into the matching child mini-mux
    is what keeps the child's projections populated.

    Native transformer — `subgraphs` is exposed as `run.subgraphs`.
    """

    _native = True
    supports_sync = True

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[SubgraphRunStream | AsyncSubgraphRunStream] = (
            StreamChannel()
        )
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
        parent_task_id: str | None,
        tool_call_id: str | None = None,  # noqa: ARG002
    ) -> None:
        if self._mux is None:
            return
        try:
            child_mux = self._mux._make_child(ns)
        except RuntimeError:
            return
        handle_cls = AsyncSubgraphRunStream if child_mux.is_async else SubgraphRunStream
        handle = handle_cls(
            mux=child_mux,
            path=ns,
            graph_name=graph_name,
            parent_task_id=parent_task_id,
        )
        self._handles[ns] = handle
        self._log.push(handle)

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: LifecycleEvent,
        error: str | None,
        parent_task_id: str,  # noqa: ARG002
    ) -> None:
        handle = self._handles.get(ns)
        if handle is None or not self._mark_terminal(handle, status, error):
            return
        self._close_or_fail_handle(handle, status, error)

    async def _aon_terminal(
        self,
        ns: tuple[str, ...],
        status: LifecycleEvent,
        error: str | None,
        parent_task_id: str,  # noqa: ARG002
    ) -> None:
        handle = self._handles.get(ns)
        if handle is None or not self._mark_terminal(handle, status, error):
            return
        await self._aclose_or_fail_handle(handle, status, error)

    def _mark_terminal(
        self,
        handle: SubgraphRunStream | AsyncSubgraphRunStream,
        status: LifecycleEvent,
        error: str | None,
    ) -> bool:
        """Mark a handle terminal once. Returns True on first transition."""
        if handle._seen_terminal:
            return False
        handle.status = status
        if error is not None and handle.error is None:
            handle.error = error
        handle._seen_terminal = True
        return True

    def _close_or_fail_handle(
        self,
        handle: SubgraphRunStream | AsyncSubgraphRunStream,
        status: LifecycleEvent,
        error: str | None,
    ) -> None:
        if handle._mux is None or handle._mux._events._closed:
            return
        if status == "failed":
            handle._mux.fail(RuntimeError(error or "Subgraph failed"))
        else:
            handle._mux.close()

    async def _aclose_or_fail_handle(
        self,
        handle: SubgraphRunStream | AsyncSubgraphRunStream,
        status: LifecycleEvent,
        error: str | None,
    ) -> None:
        if handle._mux is None or handle._mux._events._closed:
            return
        if status == "failed":
            await handle._mux.afail(RuntimeError(error or "Subgraph failed"))
        else:
            await handle._mux.aclose()

    def _handle_for_event(
        self, event: ProtocolEvent
    ) -> SubgraphRunStream | AsyncSubgraphRunStream | None:
        ns = tuple(event["params"]["namespace"])
        depth = len(self.scope)
        if len(ns) < depth + 1:
            return None
        handle = self._handles.get(ns[: depth + 1])
        if handle is None or handle._mux is None or handle._mux._events._closed:
            return None
        return handle

    def process(self, event: ProtocolEvent) -> bool:
        # Run tasks bookkeeping first so a `started` handle exists
        # by the time we forward the event to the child mini-mux.
        keep = super().process(event)
        handle = self._handle_for_event(event)
        if handle is not None:
            handle._observe_event(event)
            handle._mux.push(event)
        return keep

    async def aprocess(self, event: ProtocolEvent) -> bool:
        # Async counterpart: repeats the tasks bookkeeping here so
        # child mini-muxes receive events through their async lane.
        if event["method"] == "tasks":
            ns = tuple(event["params"]["namespace"])
            data = event["params"]["data"]
            if "result" in data:
                for (
                    child_ns,
                    status,
                    error,
                    parent_task_id,
                ) in self._pop_terminal_transitions(ns, data):
                    await self._aon_terminal(child_ns, status, error, parent_task_id)
            else:
                self._handle_task_start(ns, data)
            keep = False
        else:
            keep = True
        handle = self._handle_for_event(event)
        if handle is not None:
            handle._observe_event(event)
            await handle._mux.apush(event)
        return keep

    def _complete_open_handles(self) -> BaseException | None:
        first_error: BaseException | None = None
        for ns, parent_task_id in list(self._open.items()):
            try:
                self._on_terminal(ns, "completed", None, parent_task_id)
            except BaseException as e:
                if first_error is None:
                    first_error = e
        self._open.clear()
        for handle in self._handles.values():
            if self._mark_terminal(handle, "completed", None):
                try:
                    self._close_or_fail_handle(handle, "completed", None)
                except BaseException as e:
                    if first_error is None:
                        first_error = e
        return first_error

    async def _acomplete_open_handles(self) -> BaseException | None:
        first_error: BaseException | None = None
        for ns, parent_task_id in list(self._open.items()):
            try:
                await self._aon_terminal(ns, "completed", None, parent_task_id)
            except BaseException as e:
                if first_error is None:
                    first_error = e
        self._open.clear()
        for handle in self._handles.values():
            if self._mark_terminal(handle, "completed", None):
                try:
                    await self._aclose_or_fail_handle(handle, "completed", None)
                except BaseException as e:
                    if first_error is None:
                        first_error = e
        return first_error

    def finalize(self) -> None:
        first_error = self._complete_open_handles()
        if first_error is not None:
            raise first_error

    async def afinalize(self) -> None:
        first_error = await self._acomplete_open_handles()
        if first_error is not None:
            raise first_error

    def fail(self, err: BaseException) -> None:
        status, error_str = _status_from_exception(err)
        self._open.clear()
        for handle in self._handles.values():
            self._mark_terminal(handle, status, error_str)
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
        status, error_str = _status_from_exception(err)
        self._open.clear()
        for handle in self._handles.values():
            self._mark_terminal(handle, status, error_str)
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


class CheckpointsTransformer(StreamTransformer):
    """Capture checkpoint events as a drainable stream.

    Surfaces `stream_mode="checkpoints"` data on `run.checkpoints` as
    a `StreamChannel[dict[str, Any]]`. Each item is in the same format
    as returned by `get_state()`.

    Checkpoint events are only emitted when a checkpointer is configured
    on the graph. When no checkpointer is present, the projection exists
    but receives no events.

    Only events at the run's own scope are captured; checkpoint data from
    deeper subgraphs is available on the respective subgraph handle's
    `.checkpoints` projection.

    Native transformer — `run.checkpoints` is a direct attribute.
    """

    _native = True
    required_stream_modes = ("checkpoints",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict[str, Any]] = StreamChannel()
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        return {"checkpoints": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "checkpoints":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
        self._log.push(params["data"])
        return True


class DebugTransformer(StreamTransformer):
    """Capture debug events as a drainable stream.

    Surfaces `stream_mode="debug"` data on `run.debug` as a
    `StreamChannel[dict[str, Any]]`. Each item is a debug event with
    step-level detail (checkpoint snapshots, task payloads, and
    task results wrapped with step number and timestamp).

    Only events at the run's own scope are captured; debug data from
    deeper subgraphs is available on the respective subgraph handle's
    `.debug` projection.

    Native transformer — `run.debug` is a direct attribute.
    """

    _native = True
    required_stream_modes = ("debug",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict[str, Any]] = StreamChannel()
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        return {"debug": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "debug":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
        self._log.push(params["data"])
        return True


class TasksTransformer(StreamTransformer):
    """Capture raw task events as a drainable stream.

    Surfaces `stream_mode="tasks"` data on `run.tasks` as a
    `StreamChannel[dict[str, Any]]`. Each item is a task payload
    (start or result).

    `LifecycleTransformer` and `SubgraphTransformer` also consume
    `tasks` events for subgraph discovery and lifecycle tracking.
    This transformer captures the raw payloads independently for
    consumers who need task-level detail.

    Only events at the run's own scope are captured; task data from
    deeper subgraphs is available on the respective subgraph handle's
    `.tasks` projection.

    Native transformer — `run.tasks` is a direct attribute.
    """

    _native = True
    required_stream_modes = ("tasks",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict[str, Any]] = StreamChannel()
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        return {"tasks": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "tasks":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
        self._log.push(params["data"])
        return True
