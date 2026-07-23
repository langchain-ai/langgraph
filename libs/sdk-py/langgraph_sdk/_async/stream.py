"""Async thread-centric streaming surface for the v3 protocol.

`AsyncThreadStream` is an async context manager that owns a
`ProtocolSseTransport` for one thread, dispatches commands (`run.start`,
`run.respond`), exposes typed subscriptions over a single shared SSE
(`subscribe`, `events`), surfaces lifecycle state (`interrupted`,
`interrupts`) via an always-on lifecycle watcher SSE, and provides typed
projections (`thread.values`, `thread.messages`, `thread.tool_calls`,
`thread.extensions`).

Direct port of `libs/sdk/src/client/stream/index.ts`.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, cast

from langchain_core.language_models.chat_model_stream import AsyncChatModelStream
from langchain_protocol import Event, SubscribeParams

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk.schema import QueryParamTypes
from langgraph_sdk.stream.controller import _SeenEventIds
from langgraph_sdk.stream.decoders import (
    DataDecoder,
    Decoder,
    ExtensionsDecoder,
    MessagesDecoder,
    SubgraphsDecoder,
    ToolCallsDecoder,
    validate_interleave_channels,
)
from langgraph_sdk.stream.subscription import compute_union_filter, infer_channel
from langgraph_sdk.stream.transport import (
    AsyncProtocolTransport,
    EventStreamHandle,
    ProtocolSseTransport,
    ProtocolWebSocketTransport,
)


class InterruptPayload(TypedDict):
    """Payload surfaced when the server requests human input for a thread."""

    interrupt_id: str
    value: Any
    namespace: list[str]


@dataclass
class _RunTerminal:
    """Terminal state record resolved into `_run_done` on lifecycle completion."""

    status: Literal["completed", "errored"]
    error: BaseException | None = None


@dataclass
class _Subscription:
    """Internal record for one active subscription on an `AsyncThreadStream`."""

    id: int
    params: SubscribeParams
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # Why: asyncio.Queue[Event | None] as a subscript in the field annotation
    # causes a type error with ty; bare asyncio.Queue is accepted.


# All public protocol channels used by the raw `events`/`subscribe` surface.
# Typed projections open narrower channel filters on the shared SSE.
_ALL_CHANNELS: list[str] = [
    "values",
    "updates",
    "messages",
    "tools",
    "lifecycle",
    "input",
    "checkpoints",
    "tasks",
    "custom",
]


def _exact_namespace_params(
    channels: list[str],
    namespace: list[str],
) -> SubscribeParams:
    return {
        "channels": channels,
        "namespaces": [list(namespace)],
        "depth": 0,
    }


def _event_namespace(params_field: Any) -> list[str]:
    if not isinstance(params_field, dict):
        return []
    namespace = params_field.get("namespace") or []
    return list(namespace) if isinstance(namespace, list) else []


_ROOT_TERMINAL_LIFECYCLE_EVENTS = frozenset({"completed", "failed"})


def _is_root_terminal_lifecycle(event: Any) -> bool:
    """Return True for a root-namespace lifecycle event marking run end.

    Matches the wire shape ``{method: "lifecycle", params: {namespace: [],
    data: {event: "completed" | "failed"}}}``. Subgraph lifecycle events
    (non-empty namespace) do not terminate the parent run.
    """
    if not isinstance(event, dict):
        return False
    if event.get("method") != "lifecycle":
        return False
    params = event.get("params") or {}
    if not isinstance(params, dict):
        return False
    if params.get("namespace") or []:
        return False
    data = params.get("data") or {}
    if not isinstance(data, dict):
        return False
    return data.get("event") in _ROOT_TERMINAL_LIFECYCLE_EVENTS


class _AgentModule:
    """Assistant graph helpers scoped to one thread stream."""

    def __init__(self, owner: AsyncThreadStream) -> None:
        self._owner = owner

    async def get_tree(
        self,
        *,
        xray: int | bool = False,
        headers: Mapping[str, str] | None = None,
        params: QueryParamTypes | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        if self._owner._closed:
            raise RuntimeError("AsyncThreadStream is closed.")
        query_params: dict[str, Any] = {}
        if xray:
            query_params["xray"] = xray
        if params:
            query_params.update(dict(params))
        request_headers = {**self._owner._headers, **dict(headers or {})}
        return await self._owner._http.get(
            f"/assistants/{self._owner.assistant_id}/graph",
            params=query_params,
            headers=request_headers or None,
        )


class RunModule:
    """Command dispatcher for `run.start`.

    Bound to one `AsyncThreadStream`; accesses its transport and id allocator.
    """

    def __init__(self, owner: AsyncThreadStream) -> None:
        self._owner = owner

    async def start(
        self,
        *,
        input: Any = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send `run.start` to the server. Returns the result (`{"run_id": ...}`)."""
        params: dict[str, Any] = {"assistant_id": self._owner.assistant_id}
        if input is not None:
            params["input"] = input
        if config is not None:
            params["config"] = config
        if metadata is not None:
            params["metadata"] = metadata
        loop = asyncio.get_running_loop()
        gate: asyncio.Future[None] = loop.create_future()
        self._owner._run_start_ready = gate
        try:
            result = await self._owner._send_command("run.start", params)
            if not gate.done():
                gate.set_result(None)
            self._owner._run_seen = True
            return result
        except BaseException as err:
            # Why: gate MUST reject on any exit type, including CancelledError,
            # so awaiters see the failure rather than hanging indefinitely.
            if not gate.done():
                gate.set_exception(err)
            raise
        finally:
            # Why: concurrent run.start calls (multitask_strategy="enqueue")
            # can replace _run_start_ready before our finally fires.
            # Identity-check before clearing so the later call's gate isn't stomped.
            if self._owner._run_start_ready is gate:
                self._owner._run_start_ready = None
            # Why: if the gate stored an exception that no awaiter consumed,
            # retrieve it here to suppress asyncio's GC warning. The exception
            # is already propagated to our caller via the `raise` above.
            if gate.done() and not gate.cancelled():
                gate.exception()

    async def respond(
        self,
        response: Any,
        *,
        interrupt_id: str | None = None,
    ) -> dict[str, Any]:
        """Reply to a server-side interrupt and resume the run.

        Args:
            response: the response value forwarded as `params.response` on the
                wire (protocol field name).
            interrupt_id: optional explicit id. When omitted, requires exactly
                one outstanding interrupt and uses its id.

        Raises:
            RuntimeError: no outstanding interrupts; `interrupt_id` is None but
                multiple interrupts are outstanding; or the explicit
                `interrupt_id` doesn't match any outstanding interrupt.
        """
        # Why: take the `interrupts` snapshot AND dispatch the command under
        # `_interrupts_lock`, so the lifecycle watcher's terminal-clear path
        # cannot wipe `interrupts` between the snapshot and the wire send.
        async with self._owner._interrupts_lock:
            outstanding = list(self._owner.interrupts)
            if interrupt_id is None:
                if len(outstanding) == 0:
                    raise RuntimeError(
                        "thread.run.respond: no outstanding interrupt. Provide "
                        "an explicit `interrupt_id` or wait for "
                        "`thread.interrupted`."
                    )
                if len(outstanding) > 1:
                    ids = [p["interrupt_id"] for p in outstanding]
                    raise RuntimeError(
                        f"thread.run.respond: ambiguous — {len(outstanding)} "
                        f"outstanding interrupts ({ids!r}). Provide an explicit "
                        "`interrupt_id`."
                    )
                match = outstanding[0]
            else:
                match = next(
                    (p for p in outstanding if p["interrupt_id"] == interrupt_id),
                    None,
                )
                if match is None:
                    raise RuntimeError(
                        f"thread.run.respond: interrupt_id {interrupt_id!r} does "
                        "not match any outstanding interrupt in "
                        "`thread.interrupts`."
                    )
            params = {
                "interrupt_id": match["interrupt_id"],
                "namespace": match["namespace"],
                "response": response,
            }
            return await self._owner._send_command("input.respond", params)


async def _close_after(handle: EventStreamHandle, *, delay: float = 0.0) -> None:
    """Close a handle, optionally after a brief delay. Used to detach
    closing the old stream from the synchronous rotation step so the new
    stream can absorb server-side replayed events first.
    """
    if delay:
        await asyncio.sleep(delay)
    await handle.close()


class _OutputAwaitable:
    """Awaitable that waits for lifecycle completion then fetches durable thread state.

    Multiple awaiters share one underlying task (idempotent task caching).
    Call `with_timeout(seconds)` to bound the wait on the lifecycle terminal.
    """

    def __init__(self, thread: AsyncThreadStream) -> None:
        self._thread = thread
        self._task: asyncio.Task[Any] | None = None
        self._timeout: float | None = None

    def __await__(self):  # type: ignore[override]
        return self._get_task().__await__()

    def with_timeout(self, timeout: float) -> _OutputAwaitable:
        """Return a new awaitable that raises `asyncio.TimeoutError` after `timeout` seconds.

        Bounds the wait for the lifecycle terminal (and only that wait); the
        subsequent REST GET for terminal state is not bounded. Returns a
        fresh `_OutputAwaitable` so the original `thread.output` is unaffected.
        """
        bounded = _OutputAwaitable(self._thread)
        bounded._timeout = timeout
        return bounded

    def _get_task(self) -> asyncio.Task[Any]:
        """Return the shared fetch task, creating it on first call.

        A cancelled task is intentionally NOT respawned: subsequent awaiters
        receive `asyncio.CancelledError` from the shared task instead of
        triggering a fresh REST GET. This preserves "one fetch per
        `thread.output`" semantics even when callers wrap awaits with
        `asyncio.wait_for` (which cancels the underlying task on timeout).
        """
        if self._task is None:
            self._task = asyncio.create_task(self._fetch())
        return self._task

    async def _fetch(self) -> Any:
        """Fetch terminal thread state, waiting for the lifecycle if needed."""
        # Fast path: explicit thread_id with no run in flight — state may already
        # be terminal so we can skip the lifecycle wait entirely.
        if self._thread._can_return_existing_state_immediately():
            state = await self._thread._fetch_state()
            if self._thread._state_is_terminal(state):
                return state["values"]

        # Normal path: wait for the lifecycle terminal signal.
        if self._timeout is not None:
            terminal = await asyncio.wait_for(
                self._thread._wait_for_run_done(), timeout=self._timeout
            )
        else:
            terminal = await self._thread._wait_for_run_done()
        if terminal.error is not None:
            raise terminal.error
        state = await self._thread._fetch_state()
        return state["values"]


class _ValuesProjection:
    """Typed projection for `thread.values` — yields state snapshots as they arrive.

    Supports both `async for` (live stream of state snapshots) and `await`
    (delegates to `thread.output` for the terminal state value).
    """

    def __init__(self, thread: AsyncThreadStream) -> None:
        self._thread = thread

    def __await__(self) -> Generator[Any, None, Any]:
        return self._thread.output.__await__()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._values_iter()

    async def _values_iter(self) -> AsyncGenerator[Any, None]:
        if self._thread._transport is None:
            raise RuntimeError("AsyncThreadStream not entered — use `async with`.")
        params: SubscribeParams = {"channels": ["values"]}
        sub = self._thread._register_subscription(params)
        decoder = DataDecoder("values")
        try:
            await self._thread._reconcile_stream(params)
            self._thread._ensure_fanout_running()
            state = await self._thread._fetch_state()
            yield state["values"]
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                for out in decoder.feed(item):
                    yield out
        finally:
            self._thread._unregister_subscription(sub.id)


class _MessagesProjection:
    """Typed projection for root-scope `thread.messages`.

    Iterating yields one `AsyncChatModelStream` per message-start event.
    Each iterator owns its own `messages` subscription and routes events
    from the root namespace only.
    """

    def __init__(
        self, thread: AsyncThreadStream, namespace: list[str] | None = None
    ) -> None:
        self._thread = thread
        self._namespace = list(namespace or [])

    def __aiter__(self) -> AsyncIterator[AsyncChatModelStream]:
        return self._messages_iter()

    async def _messages_iter(self) -> AsyncGenerator[AsyncChatModelStream, None]:
        if self._thread._transport is None:
            raise RuntimeError("AsyncThreadStream not entered — use `async with`.")
        # If the subgraphs projection already ran (and consumed messages events
        # from the shared SSE), drain its root inbox rather than opening a new
        # subscription. Dedup prevents the SSE from replaying those events.
        root_inbox = self._thread._root_messages_inbox if not self._namespace else None
        if root_inbox is not None:
            async for stream in self._drain_inbox(root_inbox):
                yield stream
            return
        params = _exact_namespace_params(["messages"], self._namespace)
        sub = self._thread._register_subscription(params)
        decoder = MessagesDecoder(
            namespace=self._namespace,
            stream_factory=lambda *, namespace, node, message_id: AsyncChatModelStream(
                namespace=namespace, node=node, message_id=message_id
            ),
        )
        registered: list[AsyncChatModelStream] = []
        try:
            await self._thread._reconcile_stream(params)
            self._thread._ensure_fanout_running()
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                for stream in decoder.feed(item):
                    self._thread._register_active_message_stream(stream)
                    registered.append(stream)
                    yield stream
        finally:
            for stream in registered:
                self._thread._unregister_active_message_stream(stream)
            self._thread._unregister_subscription(sub.id)

    async def _drain_inbox(
        self, inbox: asyncio.Queue[Event | None]
    ) -> AsyncGenerator[AsyncChatModelStream, None]:
        """Drain a pre-filled inbox of messages events, yielding one stream per message."""
        from langchain_core.language_models.chat_model_stream import (
            AsyncChatModelStream,
        )

        active: dict[str, AsyncChatModelStream] = {}
        try:
            while True:
                item = await inbox.get()
                if item is None:
                    return
                params_field = item.get("params") or {}
                data = (
                    params_field.get("data") if isinstance(params_field, dict) else None
                )
                if not isinstance(data, dict):
                    continue
                event_type = data.get("event")
                if event_type == "message-start":
                    message_id = _message_event_id(data)
                    key = _message_route_key(data, fallback=message_id)
                    metadata = (
                        data.get("metadata")
                        if isinstance(data.get("metadata"), dict)
                        else {}
                    )
                    stream = AsyncChatModelStream(
                        namespace=list(self._namespace),
                        node=metadata.get("langgraph_node") if metadata else None,
                        message_id=message_id,
                    )
                    active[key] = stream
                    self._thread._register_active_message_stream(stream)
                    stream.dispatch(data)
                    yield stream
                else:
                    key = _message_route_key(data)
                    stream = active.get(key)
                    if stream is None and len(active) == 1:
                        stream = next(iter(active.values()))
                    if stream is None:
                        continue
                    stream.dispatch(data)
                    if event_type in ("message-finish", "error"):
                        self._thread._unregister_active_message_stream(stream)
                        for route_key, candidate in list(active.items()):
                            if candidate is stream:
                                del active[route_key]
        finally:
            for stream in active.values():
                self._thread._unregister_active_message_stream(stream)


def _message_event_id(data: dict[str, Any]) -> str | None:
    message_id = data.get("id") or data.get("message_id")
    return str(message_id) if message_id is not None else None


def _message_route_key(data: dict[str, Any], fallback: str | None = None) -> str:
    """Return the routing key for a message-channel event in `active`.

    Keys on `message_id` when available so concurrent messages that share the
    same `run_id` (two AI turns in one agent step) route to independent streams
    rather than colliding on a shared `run:<run_id>` slot.
    """
    message_id = _message_event_id(data)
    if message_id is not None:
        return f"message:{message_id}"
    if fallback is not None:
        return f"message:{fallback}"
    return "__single__"


SubgraphStatus = Literal["started", "completed", "failed", "interrupted"]


def _parse_namespace_segment(segment: str) -> tuple[str, str | None]:
    name, sep, task_id = segment.partition(":")
    return name, task_id if sep else None


def _terminal_from_tasks_result(
    data: dict[str, Any],
) -> tuple[SubgraphStatus, str | None]:
    if data.get("interrupts"):
        return "interrupted", None
    error = data.get("error")
    if error:
        return "failed", str(error)
    return "completed", None


def _is_direct_child(namespace: list[str], scope: tuple[str, ...]) -> bool:
    return len(namespace) == len(scope) + 1 and tuple(namespace[: len(scope)]) == scope


def _subgraph_subscription_params(scope: tuple[str, ...]) -> SubscribeParams:
    # Subscribe to tasks + messages + tools + lifecycle without a depth limit
    # so all descendant-namespace events are captured in one SSE and buffered
    # into each child handle's inbox. ``lifecycle`` is included so child-
    # namespace ``started`` events (the canonical signal for
    # ``create_deep_agent``-style subagent discovery, matching JS behavior)
    # reach ``_subgraphs_iter``; servers that surface child invocations via
    # ``tasks`` events instead are also handled via the existing ``method ==
    # "tasks"`` branch.
    return {
        "channels": ["messages", "tasks", "tools", "lifecycle"],
        "namespaces": [list(scope)],
    }


class ScopedStreamHandle:
    """Scoped streaming handle for one discovered child invocation."""

    def __init__(
        self,
        *,
        thread: AsyncThreadStream,
        path: tuple[str, ...],
        graph_name: str | None,
        trigger_call_id: str | None,
        max_queue_size: int = 0,
    ) -> None:
        self._thread = thread
        self.path = path
        self.namespace = list(path)
        self.graph_name = graph_name
        self.trigger_call_id = trigger_call_id
        self.status: SubgraphStatus = "started"
        self.error: str | None = None
        self._max_queue_size = max_queue_size
        # Per-channel inboxes: events captured by the parent _SubgraphsProjection
        # while the SSE was alive. Child projections drain these after the parent
        # finishes so sequential consumption works without a second SSE open.
        self._messages_inbox: asyncio.Queue[Event | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._tools_inbox: asyncio.Queue[Event | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._tasks_inbox: asyncio.Queue[Event | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        # Descendant handles registered by _HandleSubgraphsProjection when a
        # grandchild is discovered. _push_event fans out to each matching
        # descendant at dispatch time so events arrive in arrival order without
        # any drain-and-replay.
        self._descendant_handles: dict[tuple[str, ...], ScopedStreamHandle] = {}
        # Track which inboxes have a consumer so _close_inboxes only sends a
        # sentinel where it is needed. Inboxes with no consumer would otherwise
        # accumulate a leaked None sentinel that is never drained.
        self._iterated_inboxes: set[str] = set()
        self.messages = _HandleMessagesProjection(self)
        self.tool_calls = _HandleToolCallsProjection(self)
        self.subgraphs = _HandleSubgraphsProjection(self)
        self.subagents = self.subgraphs
        self.extensions = _ExtensionsProjection(thread, namespace=list(path))

    def _push_event(self, event: Event) -> None:
        """Route a descendant event into the appropriate channel inbox.

        Also fans out to any registered descendant handles whose path is a
        prefix of the event namespace, so grandchild events are delivered at
        push time rather than via a post-hoc drain-and-replay.
        """
        method = event.get("method")
        if method == "messages":
            self._messages_inbox.put_nowait(event)
        elif method == "tools":
            self._tools_inbox.put_nowait(event)
        elif method == "tasks":
            self._tasks_inbox.put_nowait(event)
        # Fan out to descendant handles whose namespace is a prefix of the
        # event namespace so they receive the event at push time.
        if method in ("messages", "tools", "tasks"):
            ns_tuple = tuple(_event_namespace(event.get("params") or {}))
            for desc_path, desc_handle in self._descendant_handles.items():
                desc_len = len(desc_path)
                if len(ns_tuple) >= desc_len and ns_tuple[:desc_len] == desc_path:
                    desc_handle._push_event(event)

    def _register_descendant(self, handle: ScopedStreamHandle) -> None:
        """Register a newly-discovered grandchild so future events are fanned out.

        Also drains any events already buffered in this handle's inboxes whose
        namespace matches the grandchild, so events that arrived before the
        grandchild was discovered are forwarded in arrival order.
        """
        self._descendant_handles[handle.path] = handle
        desc_len = len(handle.path)
        for inbox_attr in (
            "_messages_inbox",
            "_tools_inbox",
            "_tasks_inbox",
        ):
            inbox: asyncio.Queue[Event | None] = getattr(self, inbox_attr)
            staging: list[Event | None] = []
            while not inbox.empty():
                staging.append(inbox.get_nowait())
            for event in staging:
                inbox.put_nowait(event)
                if event is None:
                    continue
                ns_tuple = tuple(_event_namespace(event.get("params") or {}))
                if len(ns_tuple) >= desc_len and ns_tuple[:desc_len] == handle.path:
                    getattr(handle, inbox_attr).put_nowait(event)

    def _unregister_descendant(self, path: tuple[str, ...]) -> None:
        """Remove a grandchild after it reaches a terminal state."""
        self._descendant_handles.pop(path, None)

    def _mark_iterated(self, kind: str) -> None:
        """Record that an inbox has an active consumer.

        If the handle is already closed (status != 'started'), immediately
        enqueue a sentinel so the consumer's `await get()` terminates. This
        handles sequential consumption (iterate after the handle is finished).

        Must be called by each projection at the start of iteration.
        """
        self._iterated_inboxes.add(kind)
        if self.status != "started":
            # Handle already closed before this consumer started; send the
            # sentinel now so the projection iterator can terminate.
            getattr(self, f"_{kind}_inbox").put_nowait(None)

    def _close_inboxes(self) -> None:
        """Signal EOF only on channel inboxes that have an active consumer.

        Inboxes without a consumer would accumulate a leaked None sentinel
        that is never drained, so we skip them. For inboxes whose consumer
        starts after this call, `_mark_iterated` sends the sentinel lazily.
        """
        for kind in ("messages", "tools", "tasks"):
            if kind in self._iterated_inboxes:
                getattr(self, f"_{kind}_inbox").put_nowait(None)

    def _finish(self, status: SubgraphStatus, error: str | None = None) -> None:
        if self.status != "started":
            return
        self.status = status
        self.error = error
        self._close_inboxes()


class _HandleMessagesProjection:
    """Messages projection that drains a `ScopedStreamHandle`'s messages inbox."""

    def __init__(self, handle: ScopedStreamHandle) -> None:
        self._handle = handle

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._messages_iter()

    async def _messages_iter(self) -> AsyncGenerator[Any, None]:
        from langchain_core.language_models.chat_model_stream import (
            AsyncChatModelStream,
        )

        self._handle._mark_iterated("messages")
        active: dict[str, AsyncChatModelStream] = {}
        while True:
            item = await self._handle._messages_inbox.get()
            if item is None:
                return
            params_field = item.get("params") or {}
            ns = _event_namespace(params_field)
            if ns != self._handle.namespace:
                continue
            data = params_field.get("data") if isinstance(params_field, dict) else None
            if not isinstance(data, dict):
                continue
            event_type = data.get("event")
            if event_type == "message-start":
                message_id = _message_event_id(data)
                key = _message_route_key(data, fallback=message_id)
                metadata = (
                    data.get("metadata")
                    if isinstance(data.get("metadata"), dict)
                    else {}
                )
                stream = AsyncChatModelStream(
                    namespace=list(self._handle.namespace),
                    node=metadata.get("langgraph_node") if metadata else None,
                    message_id=message_id,
                )
                active[key] = stream
                stream.dispatch(data)
                yield stream
            else:
                key = _message_route_key(data)
                stream = active.get(key)
                if stream is None and len(active) == 1:
                    stream = next(iter(active.values()))
                if stream is None:
                    continue
                stream.dispatch(data)
                if event_type in ("message-finish", "error"):
                    for route_key, candidate in list(active.items()):
                        if candidate is stream:
                            del active[route_key]


class _HandleToolCallsProjection:
    """Tool calls projection that drains a `ScopedStreamHandle`'s tools inbox."""

    def __init__(self, handle: ScopedStreamHandle) -> None:
        self._handle = handle

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._tool_calls_iter()

    async def _tool_calls_iter(self) -> AsyncGenerator[Any, None]:
        self._handle._mark_iterated("tools")
        active: dict[str, ToolCallHandle] = {}
        while True:
            item = await self._handle._tools_inbox.get()
            if item is None:
                err = RuntimeError(
                    "Tool call stream closed before terminal tool event."
                )
                for handle in active.values():
                    handle._fail(err)
                return
            params_field = item.get("params") or {}
            ns = _event_namespace(params_field)
            if ns != self._handle.namespace:
                continue
            data = params_field.get("data") if isinstance(params_field, dict) else None
            if not isinstance(data, dict):
                continue
            event_type = data.get("event")
            tool_call_id = data.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                continue
            if event_type == "tool-started":
                tool_name = data.get("tool_name")
                if not isinstance(tool_name, str):
                    tool_name = ""
                handle = ToolCallHandle(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    input=data.get("input"),
                    namespace=list(self._handle.namespace),
                )
                active[tool_call_id] = handle
                yield handle
            elif event_type == "tool-output-delta":
                handle = active.get(tool_call_id)
                delta = data.get("delta")
                if handle is not None and isinstance(delta, str):
                    handle._push_delta(delta)
            elif event_type == "tool-finished":
                handle = active.pop(tool_call_id, None)
                if handle is not None:
                    handle._finish(data.get("output"))
            elif event_type == "tool-error":
                handle = active.pop(tool_call_id, None)
                if handle is not None:
                    message = data.get("message")
                    handle._fail(
                        RuntimeError(str(message) if message else "Tool call errored")
                    )


class _HandleSubgraphsProjection:
    """Subgraphs projection that drains a `ScopedStreamHandle`'s tasks inbox."""

    def __init__(self, handle: ScopedStreamHandle) -> None:
        self._handle = handle

    def __aiter__(self) -> AsyncIterator[ScopedStreamHandle]:
        return self._subgraphs_iter()

    def _route_sibling_inboxes_to_grandchildren(
        self,
        active: dict[tuple[str, ...], ScopedStreamHandle],
    ) -> None:
        """Drain non-blocking events from parent's messages/tools inboxes to grandchildren.

        Called after each tasks event so grandchild handles receive events that
        were enqueued in the parent handle's inboxes before (or just after) the
        grandchild was discovered.
        """
        parent = self._handle
        for inbox_attr, grandchild_attr in (
            ("_messages_inbox", "_messages_inbox"),
            ("_tools_inbox", "_tools_inbox"),
        ):
            inbox: asyncio.Queue[Event | None] = getattr(parent, inbox_attr)
            staging: list[Event | None] = []
            # Drain without blocking.
            while not inbox.empty():
                staging.append(inbox.get_nowait())
            for event in staging:
                if event is None:
                    # Re-queue the EOF sentinel — it belongs to the parent inbox consumer.
                    inbox.put_nowait(None)
                    continue
                event_params = event.get("params") or {}
                ns_tuple = tuple(_event_namespace(event_params))
                routed = False
                for _child_path, grandchild in active.items():
                    grandchild_len = len(grandchild.path)
                    if (
                        len(ns_tuple) >= grandchild_len
                        and ns_tuple[:grandchild_len] == grandchild.path
                    ):
                        gc_inbox: asyncio.Queue[Event | None] = getattr(
                            grandchild, grandchild_attr
                        )
                        gc_inbox.put_nowait(event)
                        routed = True
                        break
                if not routed:
                    # Not a grandchild event — put it back for the handle projection.
                    inbox.put_nowait(event)

    async def _subgraphs_iter(self) -> AsyncGenerator[ScopedStreamHandle, None]:
        self._handle._mark_iterated("tasks")
        seen: set[tuple[str, ...]] = set()
        active: dict[tuple[str, ...], ScopedStreamHandle] = {}
        scope = self._handle.path
        while True:
            item = await self._handle._tasks_inbox.get()
            if item is None:
                for child in active.values():
                    if child.status == "started":
                        child._finish("completed")
                return
            params_field = item.get("params") or {}
            namespace = _event_namespace(params_field)
            data = params_field.get("data") if isinstance(params_field, dict) else None
            if not isinstance(data, dict):
                continue
            if "result" in data:
                result_id = data.get("id")
                if not result_id:
                    continue
                parent_path = tuple(namespace)
                for child_path, child_handle in list(active.items()):
                    if child_path[:-1] != parent_path:
                        continue
                    if child_handle.trigger_call_id != result_id:
                        continue
                    status, error = _terminal_from_tasks_result(data)
                    child_handle._finish(status, error)
                    del active[child_path]
                    self._handle._unregister_descendant(child_path)
                continue
            if not _is_direct_child(namespace, scope):
                continue
            path = tuple(namespace)
            if path in seen:
                continue
            seen.add(path)
            graph_name, trigger_call_id = _parse_namespace_segment(path[-1])
            child_handle = ScopedStreamHandle(
                thread=self._handle._thread,
                path=path,
                graph_name=graph_name or None,
                trigger_call_id=trigger_call_id,
                max_queue_size=self._handle._max_queue_size,
            )
            active[path] = child_handle
            # Register so future _push_event calls on this handle fan out to the
            # grandchild at push time, preserving arrival order without drain-and-replay.
            self._handle._register_descendant(child_handle)
            yield child_handle


class _SubgraphsProjection:
    """Discover direct child invocations for a namespace scope."""

    def __init__(self, thread: AsyncThreadStream, scope: tuple[str, ...] = ()) -> None:
        self._thread = thread
        self._scope = scope

    def __aiter__(self) -> AsyncIterator[ScopedStreamHandle]:
        return self._subgraphs_iter()

    async def _subgraphs_iter(self) -> AsyncGenerator[ScopedStreamHandle, None]:
        if self._thread._transport is None:
            raise RuntimeError("AsyncThreadStream not entered - use `async with`.")
        params = _subgraph_subscription_params(self._scope)
        sub = self._thread._register_subscription(params)
        decoder = SubgraphsDecoder(
            scope=self._scope,
            handle_factory=lambda *, path, graph_name, trigger_call_id: (
                ScopedStreamHandle(
                    thread=self._thread,
                    path=path,
                    graph_name=graph_name,
                    trigger_call_id=trigger_call_id,
                )
            ),
        )
        root_inbox: asyncio.Queue[Event | None] | None = (
            self._thread._activate_root_messages_inbox() if not self._scope else None
        )
        try:
            await self._thread._reconcile_stream(params)
            self._thread._ensure_fanout_running()
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                params_field = item.get("params") or {}
                if (
                    root_inbox is not None
                    and item.get("method") == "messages"
                    and tuple(_event_namespace(params_field)) == self._scope
                ):
                    root_inbox.put_nowait(item)
                for handle in decoder.feed(item):
                    yield handle
        finally:
            # Determine terminal status from the parent run's lifecycle result.
            # If _run_done resolved as errored, force-complete remaining children
            # as errored so callers see the correct terminal state.
            terminal_status: SubgraphStatus = "completed"
            run_done = self._thread._run_done
            if run_done is not None and run_done.done() and not run_done.cancelled():
                result = run_done.result()
                if isinstance(result, _RunTerminal) and result.status == "errored":
                    terminal_status = "failed"
            for handle in decoder._active.values():
                if handle.status == "started":
                    handle._finish(terminal_status)
            self._thread._unregister_subscription(sub.id)
            if root_inbox is not None:
                root_inbox.put_nowait(None)


class ToolCallHandle:
    """Async handle for one root-scope tool call."""

    def __init__(
        self,
        *,
        tool_call_id: str,
        name: str,
        input: Any = None,
        namespace: list[str] | None = None,
        max_queue_size: int = 1024,
    ) -> None:
        self.tool_call_id = tool_call_id
        self.name = name
        self.input = input
        self.namespace = list(namespace or [])
        self.done = False
        self.error: BaseException | None = None
        loop = asyncio.get_running_loop()
        self.output: asyncio.Future[Any] = loop.create_future()
        self._deltas: asyncio.Queue[str | None] = asyncio.Queue(maxsize=max_queue_size)
        self._deltas_consumed = False

    @property
    def deltas(self) -> AsyncIterator[str]:
        """Stream tool output deltas emitted before the terminal event.

        Raises:
            RuntimeError: if called more than once — the underlying queue is
                single-consumer and cannot be fanned out safely.
        """
        if self._deltas_consumed:
            raise RuntimeError(
                "ToolCallHandle.deltas can only be iterated by a single consumer."
            )
        self._deltas_consumed = True
        return self._delta_iter()

    async def _delta_iter(self) -> AsyncGenerator[str, None]:
        while True:
            item = await self._deltas.get()
            if item is None:
                return  # errors surface via output, not deltas
            yield item

    def _push_delta(self, delta: str) -> None:
        if self.done:
            return
        self._deltas.put_nowait(delta)

    def _finish(self, output: Any) -> None:
        if self.done:
            return
        self.done = True
        if not self.output.done():
            self.output.set_result(output)
        self._deltas.put_nowait(None)

    def _fail(self, err: BaseException) -> None:
        if self.done:
            return
        self.done = True
        self.error = err
        if not self.output.done():
            self.output.set_exception(err)
        self._deltas.put_nowait(None)


class _ToolCallsProjection:
    """Typed projection for root-scope `thread.tool_calls`."""

    def __init__(
        self, thread: AsyncThreadStream, namespace: list[str] | None = None
    ) -> None:
        self._thread = thread
        self._namespace = list(namespace or [])

    def __aiter__(self) -> AsyncIterator[ToolCallHandle]:
        return self._tool_calls_iter()

    async def _tool_calls_iter(self) -> AsyncGenerator[ToolCallHandle, None]:
        if self._thread._transport is None:
            raise RuntimeError("AsyncThreadStream not entered - use `async with`.")
        params = _exact_namespace_params(["tools"], self._namespace)
        sub = self._thread._register_subscription(params)
        decoder = ToolCallsDecoder(
            namespace=self._namespace,
            handle_factory=lambda *, tool_call_id, name, input, namespace: (
                ToolCallHandle(
                    tool_call_id=tool_call_id,
                    name=name,
                    input=input,
                    namespace=namespace,
                )
            ),
        )
        registered: list[ToolCallHandle] = []
        try:
            await self._thread._reconcile_stream(params)
            self._thread._ensure_fanout_running()
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                for handle in decoder.feed(item):
                    self._thread._register_active_tool_call(handle)
                    registered.append(handle)
                    yield handle
        finally:
            # Read terminal error from _run_done if it is already resolved.
            # We do NOT block here: callers who need a terminal observation
            # should await `thread.output` directly. Blocking in iterator
            # teardown would stall every early break or exception exit for
            # up to the full shield-wait timeout (previously 1 s).
            run_done = self._thread._run_done
            terminal_err: BaseException | None = None
            if run_done is not None and run_done.done() and not run_done.cancelled():
                terminal = run_done.result()
                terminal_err = terminal.error
            err: BaseException = (
                terminal_err
                if terminal_err is not None
                else RuntimeError("Tool call stream closed before terminal tool event.")
            )
            for handle in list(decoder._active.values()):
                handle._fail(err)
            for handle in registered:
                self._thread._unregister_active_tool_call(handle)
            self._thread._unregister_subscription(sub.id)


class _ExtensionsProjection:
    """Mapping from extension name to custom event payload stream.

    Repeated access for the same `name` returns the cached projection so that
    callers receive the same subscription handle across multiple references to
    `thread.extensions["foo"]` within one session.
    """

    def __init__(self, thread: AsyncThreadStream, namespace: list[str]) -> None:
        self._thread = thread
        self._namespace = namespace
        self._cache: dict[str, _ExtensionProjection] = {}

    def __getitem__(self, name: str) -> _ExtensionProjection:
        if not name:
            raise ValueError("extension name must be non-empty.")
        if name not in self._cache:
            self._cache[name] = _ExtensionProjection(
                self._thread, name=name, namespace=self._namespace
            )
        return self._cache[name]


class _ExtensionProjection:
    def __init__(
        self,
        thread: AsyncThreadStream,
        *,
        name: str,
        namespace: list[str],
    ) -> None:
        self._thread = thread
        self._name = name
        self._namespace = namespace

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iter()

    async def _iter(self) -> AsyncGenerator[dict[str, Any], None]:
        params: SubscribeParams = {"channels": [f"custom:{self._name}"]}
        if self._namespace:
            params["namespaces"] = [self._namespace]
        sub = self._thread._register_subscription(params)
        decoder = ExtensionsDecoder(name=self._name)
        try:
            if self._thread._closed:
                return
            await self._thread._reconcile_stream(params)
            self._thread._ensure_fanout_running()
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                for out in decoder.feed(item):
                    yield out
        finally:
            self._thread._unregister_subscription(sub.id)


class AsyncThreadStream:
    """Async context manager for one thread's v3 streaming session.

    Construct via `client.threads.stream(thread_id=None, *, assistant_id, ...)`
    rather than instantiating directly.
    """

    def __init__(
        self,
        *,
        http: HttpClient,
        thread_id: str,
        assistant_id: str,
        headers: Mapping[str, str] | None = None,
        max_queue_size: int = 1024,
        run_start_timeout: float | None = None,
        explicit_thread_id: bool = False,
        transport_kind: Literal["sse", "websocket"] = "sse",
    ) -> None:
        self._http = http
        self._headers = dict(headers or {})
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self._max_queue_size = max_queue_size
        self._run_start_timeout = run_start_timeout
        self._explicit_thread_id = explicit_thread_id
        self._transport_kind = transport_kind
        self._closed = False
        self._transport: AsyncProtocolTransport | None = None
        self._open_handles: list[EventStreamHandle] = []
        self._next_command_id = 1
        self._next_subscription_id = 1
        self._subscriptions: dict[int, _Subscription] = {}
        self._seen_event_ids = _SeenEventIds()
        self._shared_stream: EventStreamHandle | None = None
        self._shared_stream_filter: dict[str, Any] | None = None
        self._fanout_task: asyncio.Task[None] | None = None
        self.interrupted: bool = False
        self.interrupts: list[InterruptPayload] = []
        # Why: serialize the `interrupts` snapshot in `run.respond` with the
        # terminal-clear path in `_apply_lifecycle_event`, so a respond() in
        # flight cannot send a stale `interrupt_id` after the lifecycle watcher
        # observes a `completed`/`errored` event.
        self._interrupts_lock = asyncio.Lock()
        self._lifecycle_watcher_task: asyncio.Task[None] | None = None
        self._lifecycle_watcher_handle: EventStreamHandle | None = None
        self._lifecycle_cursor: int | None = None
        self._lifecycle_max_reconnect_attempts = 5
        # Shared-stream reconnect knobs: applied by `_fanout` after a post-ready
        # transport drop so subscribers (messages/tools/tasks/values projections,
        # subgraph child handles) survive a brief SSE disconnect without losing
        # buffered events. Cursor (`_cursor`) is replayed as `since` so the
        # server resumes from where the prior stream left off; per-event
        # `event_id` dedup in `_dedup_iter` drops any overlap on the new stream.
        self._shared_max_reconnect_attempts = 5
        self._shared_reconnect_backoff_base = 0.1
        self._shared_reconnect_backoff_cap = 2.0
        self._run_start_ready: asyncio.Future[None] | None = None
        self._run_seen: bool = False
        self._run_done: asyncio.Future[_RunTerminal] | None = None
        self._cursor: int | None = None
        self._active_message_streams: set[AsyncChatModelStream] = set()
        self._active_tool_calls: set[ToolCallHandle] = set()
        # Root-scope inbox: populated by `_SubgraphsProjection` when it consumes
        # messages events at namespace `[]` so that `thread.messages` can drain
        # them even after the shared SSE has ended (dedup prevents replay).
        self._root_messages_inbox: asyncio.Queue[Event | None] | None = None
        self.run = RunModule(self)
        self.agent = _AgentModule(self)
        self.output = _OutputAwaitable(self)
        self.values = _ValuesProjection(self)
        self.messages = _MessagesProjection(self, namespace=[])
        self.tool_calls = _ToolCallsProjection(self, namespace=[])
        self.subgraphs = _SubgraphsProjection(self, scope=())
        self.subagents = self.subgraphs
        self.extensions = _ExtensionsProjection(self, namespace=[])

    @property
    def _controller(self) -> AsyncThreadStream:
        """Return self as the subscription controller (duck-type compatible with StreamController).

        Exposes `_subscriptions` so tests can verify subscription counts via
        `thread._controller._subscriptions` without requiring a separate controller object.
        """
        return self

    async def __aenter__(self) -> AsyncThreadStream:
        if self._closed:
            raise RuntimeError("AsyncThreadStream is closed and cannot be re-entered.")
        transport_cls = (
            ProtocolWebSocketTransport
            if self._transport_kind == "websocket"
            else ProtocolSseTransport
        )
        self._transport = transport_cls(
            client=self._http.client,
            thread_id=self.thread_id,
            headers=self._headers,
            max_queue_size=self._max_queue_size,
        )
        # Create the run-done future here (async context guarantees a running loop).
        self._run_done = asyncio.get_running_loop().create_future()
        # Start the lifecycle watcher immediately so reattach and thread.output
        # work without a preceding run.start call.
        self._ensure_lifecycle_watcher_running()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            await self.close()
        except BaseException as close_err:
            if exc is None:
                raise
            # Original exception takes precedence; chain close error as context.
            close_err.__context__ = exc

    @property
    def events(self) -> AsyncIterator[Event]:
        """Return a fresh subscription to ALL channels.

        Each property access opens a new subscription; callers iterating twice
        will see two independent streams (both filtered by the same channel union).
        Terminates when the stream closes (server hangup, `__aexit__`, or
        transport-level close).
        """
        if self._transport is None:
            raise RuntimeError("AsyncThreadStream not entered — use `async with`.")
        handle = self._transport.open_event_stream({"channels": _ALL_CHANNELS})
        self._open_handles.append(handle)
        return handle.events

    async def close(self) -> None:
        """Tear down the thread stream. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for handle in self._open_handles:
            await handle.close()
        # Cancel _run_done so thread.output doesn't wait forever on close.
        run_done = self._run_done
        if run_done is not None and not run_done.done():
            run_done.cancel()
        if self._lifecycle_watcher_task is not None:
            self._lifecycle_watcher_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._lifecycle_watcher_task
        if self._lifecycle_watcher_handle is not None:
            await self._lifecycle_watcher_handle.close()
        self._fail_active_message_streams(asyncio.CancelledError())
        self._fail_active_tool_calls(asyncio.CancelledError())
        if self._fanout_task is not None:
            self._fanout_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._fanout_task
        if self._shared_stream is not None:
            await self._shared_stream.close()
        if self._transport is not None:
            await self._transport.close()

    def _register_subscription(self, params: SubscribeParams) -> _Subscription:
        """Allocate a subscription id and add it to the registry."""
        sub = _Subscription(
            id=self._next_subscription_id,
            params=params,
            queue=asyncio.Queue(maxsize=self._max_queue_size),
        )
        self._next_subscription_id += 1
        self._subscriptions[sub.id] = sub
        return sub

    def _unregister_subscription(self, subscription_id: int) -> None:
        """Remove a subscription from the registry. No-op if already absent."""
        self._subscriptions.pop(subscription_id, None)

    def _activate_root_messages_inbox(self) -> asyncio.Queue[Event | None]:
        """Create the root-scope messages inbox if not already active and return it.

        Called by `_SubgraphsProjection` at scope `()` to capture messages events
        that arrive at namespace `[]` before `thread.messages` has subscribed.
        """
        if self._root_messages_inbox is None:
            self._root_messages_inbox = asyncio.Queue()
        return self._root_messages_inbox

    def _register_active_message_stream(self, stream: AsyncChatModelStream) -> None:
        self._active_message_streams.add(stream)

    def _unregister_active_message_stream(self, stream: AsyncChatModelStream) -> None:
        self._active_message_streams.discard(stream)

    def _fail_active_message_streams(self, err: BaseException) -> None:
        for stream in list(self._active_message_streams):
            stream.fail(err)
        self._active_message_streams.clear()

    def _register_active_tool_call(self, handle: ToolCallHandle) -> None:
        self._active_tool_calls.add(handle)

    def _unregister_active_tool_call(self, handle: ToolCallHandle) -> None:
        self._active_tool_calls.discard(handle)

    def _fail_active_tool_calls(self, err: BaseException) -> None:
        for handle in list(self._active_tool_calls):
            handle._fail(err)
        self._active_tool_calls.clear()

    def _signal_paused(self) -> None:
        """Wake every active projection iterator on interrupt / run end.

        Pushes the terminal sentinel (`None`) into every subscription
        queue. Iterators see `None` and return; the shared SSE keeps
        running so re-iteration after `run.respond(...)` registers a
        fresh subscription and resumes.

        `root_messages_inbox` is intentionally NOT signaled here: the
        subgraphs projection that populates it is responsible for
        pushing the terminal `None` in its own `finally` block, so any
        message events it redirected to the inbox land before the
        sentinel. Signaling root_inbox here would race the redirection
        and could drop messages.
        """
        # On a saturated queue the consumer is already behind; the iterator
        # will still terminate when it drains to this point.
        for sub in list(self._subscriptions.values()):
            with contextlib.suppress(asyncio.QueueFull):
                sub.queue.put_nowait(None)

    def observe_applied_through_seq(self, seq: Any) -> None:
        """Advance the reconnect cursor from a command response meta sequence."""
        if isinstance(seq, int) and (self._cursor is None or seq > self._cursor):
            self._cursor = seq

    def _observe_event(self, event: Event) -> None:
        seq = event.get("seq")
        if isinstance(seq, int) and (self._cursor is None or seq > self._cursor):
            self._cursor = seq

    def subscribe(
        self,
        channels: list[str],
        *,
        namespaces: list[list[str]] | None = None,
        depth: int | None = None,
    ) -> AsyncIterator[Event]:
        """Open a typed subscription against the shared SSE.

        Returns an async iterator that yields raw `Event` dicts matching the
        given filter. Multiple concurrent subscribes share one HTTP connection
        whose union expands or rotates as subscriptions come and go.
        """
        if self._transport is None:
            raise RuntimeError("AsyncThreadStream not entered — use `async with`.")
        params: SubscribeParams = {"channels": list(channels)}
        if namespaces is not None:
            params["namespaces"] = namespaces
        if depth is not None:
            params["depth"] = depth
        return self._subscription_iter(params)

    async def interleave_projections(
        self, channels: list[str]
    ) -> AsyncIterator[tuple[str, Any]]:
        """Yield `(channel_name, item)` tuples across multiple projections.

        One shared subscription drives all per-channel decoders; items arrive
        in server-emit order (the SDK analog of `GraphRunStream.interleave`).

        Args:
            channels: Flat list of `"values"`, `"messages"`, `"tool_calls"`,
                `"subgraphs"`, and/or extension names. Built-ins yield their
                typed item (snapshot dict / `AsyncChatModelStream` /
                `ToolCallHandle` / `ScopedStreamHandle`); an extension yields
                its payload dict, keyed by the bare extension name.

        Note:
            Handles and streams are yielded eagerly (before their sub-stream
            completes), so items arrive interleaved in real time. To receive a
            fully-resolved handle (output already populated), use the dedicated
            `thread.tool_calls` / `thread.messages` projections instead.
        """
        validate_interleave_channels(channels)
        if self._transport is None:
            raise RuntimeError("AsyncThreadStream not entered — use `async with`.")
        decoders: dict[str, Decoder] = {}
        sub_params: list[SubscribeParams] = []
        for ch in channels:
            if ch == "values":
                decoders[ch] = DataDecoder("values")
                sub_params.append({"channels": ["values"]})
            elif ch in ("updates", "checkpoints", "tasks"):
                # Plain payload channels (local Updates/Checkpoints/Tasks
                # analog). Root-scope filter is load-bearing: a co-requested
                # unscoped `values` widens the merged subscription to all
                # namespaces, so the decoder itself keeps subgraph payloads out.
                decoders[ch] = DataDecoder(ch, namespace=[])
                sub_params.append(_exact_namespace_params([ch], []))
            elif ch == "messages":
                decoders[ch] = MessagesDecoder(
                    namespace=[],
                    stream_factory=lambda *, namespace, node, message_id: (
                        AsyncChatModelStream(
                            namespace=namespace, node=node, message_id=message_id
                        )
                    ),
                )
                sub_params.append(_exact_namespace_params(["messages"], []))
            elif ch == "tool_calls":
                decoders[ch] = ToolCallsDecoder(
                    namespace=[],
                    handle_factory=lambda *, tool_call_id, name, input, namespace: (
                        ToolCallHandle(
                            tool_call_id=tool_call_id,
                            name=name,
                            input=input,
                            namespace=namespace,
                        )
                    ),
                )
                sub_params.append(_exact_namespace_params(["tools"], []))
            elif ch == "subgraphs":
                decoders[ch] = SubgraphsDecoder(
                    scope=(),
                    handle_factory=lambda *, path, graph_name, trigger_call_id: (
                        ScopedStreamHandle(
                            thread=self,
                            path=path,
                            graph_name=graph_name,
                            trigger_call_id=trigger_call_id,
                        )
                    ),
                )
                sub_params.append(_subgraph_subscription_params(()))
            else:
                decoders[ch] = ExtensionsDecoder(name=ch)
                sub_params.append({"channels": [f"custom:{ch}"]})
        if not sub_params:
            return
        merged = cast(
            SubscribeParams,
            compute_union_filter(cast(list[dict[str, Any]], sub_params)),
        )
        subgraphs = decoders.get("subgraphs")
        # Track decoder-created handles so teardown can finalize anything still
        # in flight; otherwise an awaiting `handle.output` / `handle.messages`
        # would hang after an early break or run termination.
        registered_tool_calls: list[ToolCallHandle] = []
        registered_message_streams: list[AsyncChatModelStream] = []
        try:
            async for event in self._subscription_iter(merged):
                if subgraphs is not None:
                    for item in subgraphs.feed(event):
                        yield ("subgraphs", item)
                wire = infer_channel(event)
                public = self._interleave_public_name(wire)
                # subgraphs is driven separately above (it consumes all events); never dispatch it here.
                if public is not None and public != "subgraphs":
                    decoder = decoders.get(public)
                    if decoder is not None:
                        for item in decoder.feed(event):
                            if public == "tool_calls":
                                self._register_active_tool_call(item)
                                registered_tool_calls.append(item)
                            elif public == "messages":
                                self._register_active_message_stream(item)
                                registered_message_streams.append(item)
                            yield (public, item)
        finally:
            self._finalize_interleave_decoders(
                decoders.get("tool_calls"),
                subgraphs,
                registered_tool_calls,
                registered_message_streams,
            )

    @staticmethod
    def _interleave_public_name(wire: str | None) -> str | None:
        """Map a wire channel name to the public channel name used in interleave tuples."""
        if wire is None:
            return None
        if wire == "tools":
            return "tool_calls"
        if wire.startswith("custom:"):
            return wire[len("custom:") :]
        return wire  # values, messages (tasks/lifecycle pass through with no decoder match)

    def _finalize_interleave_decoders(
        self,
        tool_calls: Decoder | None,
        subgraphs: Decoder | None,
        registered_tool_calls: list[ToolCallHandle],
        registered_message_streams: list[AsyncChatModelStream],
    ) -> None:
        """Finalize in-flight handles when `interleave_projections` tears down.

        Mirrors the terminal handling of the dedicated `_ToolCallsProjection` /
        `_SubgraphsProjection`: in-flight tool calls are failed (so awaiting
        `handle.output` can't hang) and discovered subgraph children are
        force-completed with the run's terminal status.
        """
        run_done = self._run_done
        resolved = (
            run_done.result()
            if run_done is not None and run_done.done() and not run_done.cancelled()
            else None
        )
        if isinstance(tool_calls, ToolCallsDecoder):
            err: BaseException = (
                resolved.error
                if resolved is not None and resolved.error is not None
                else RuntimeError("Tool call stream closed before terminal tool event.")
            )
            for handle in list(tool_calls._active.values()):
                handle._fail(err)
        for handle in registered_tool_calls:
            self._unregister_active_tool_call(handle)
        for stream in registered_message_streams:
            self._unregister_active_message_stream(stream)
        if isinstance(subgraphs, SubgraphsDecoder):
            terminal_status: SubgraphStatus = (
                "failed"
                if isinstance(resolved, _RunTerminal) and resolved.status == "errored"
                else "completed"
            )
            for child in subgraphs._active.values():
                if child.status == "started":
                    child._finish(terminal_status)

    async def _subscription_iter(
        self, params: SubscribeParams
    ) -> AsyncGenerator[Event, None]:
        sub = self._register_subscription(params)
        try:
            if self._closed:
                return
            await self._reconcile_stream(params)
            self._ensure_fanout_running()
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                yield item
        finally:
            self._unregister_subscription(sub.id)

    def _ensure_fanout_running(self) -> None:
        if self._fanout_task is None or self._fanout_task.done():
            self._fanout_task = asyncio.create_task(self._fanout())

    async def _fanout(self) -> None:
        """Single consumer of the shared SSE; routes events to subscriptions.

        Why: rotation in `_reconcile_stream` replaces `_shared_stream` mid-loop.
        Re-read `self._shared_stream` on each outer iteration so we always
        consume from the current handle. The old handle's iterator exhausts
        naturally after `_close_after` closes it.

        On a post-ready transport drop (non-cancelled error in `shared.done`),
        attempts to reconnect up to `_shared_max_reconnect_attempts` times so
        scoped projections (subgraph child handles, message streams) survive
        without losing buffered events. The reconnect replays `since=<cursor>`
        and `_dedup_iter` drops any overlap.
        """
        from langgraph_sdk.stream.subscription import matches_subscription

        while not self._closed:
            shared = self._shared_stream
            if shared is None:
                return
            try:
                async for event in self._dedup_iter(shared.events):
                    if self._closed:
                        break
                    self._observe_event(event)
                    for sub in list(self._subscriptions.values()):
                        if matches_subscription(event, sub.params):
                            sub.queue.put_nowait(event)
                    # On root-terminal lifecycle, push the `None` sentinel
                    # into all subscription queues so projection iterators
                    # exit when the run ends naturally. Runs on the shared
                    # SSE so the terminal is processed in seq order with
                    # the projection events -- any in-flight values /
                    # tools / messages events for this run are already
                    # queued before None.
                    if _is_root_terminal_lifecycle(event):
                        self._signal_paused()
            except Exception:
                # Pump errored — fall through to error-handling/reconnect.
                pass
            if self._shared_stream is shared:
                # No rotation happened; the stream genuinely ended. Check
                # `shared.done` for a post-ready drop and, if so, attempt to
                # reconnect with `since=<cursor>` so subscribers don't lose
                # buffered events on a transient transport failure.
                err = await shared.done
                if (
                    err is not None
                    and not isinstance(err, asyncio.CancelledError)
                    and not self._closed
                ):
                    with contextlib.suppress(Exception):
                        await shared.close()
                    if await self._reconnect_shared_stream():
                        continue
                break
            # Rotation: loop again to pick up the new _shared_stream.

        # Terminate consumers cleanly on shutdown / stream-end.
        for sub in self._subscriptions.values():
            sub.queue.put_nowait(None)

    async def _reconnect_sleep(self, attempt: int) -> None:
        """Sleep with exponential backoff and jitter for reconnect attempt `attempt`."""
        base = self._shared_reconnect_backoff_base
        cap = self._shared_reconnect_backoff_cap
        delay = min(cap, base * (2**attempt))
        jitter = random.uniform(0, delay * 0.25)
        await asyncio.sleep(delay + jitter)

    async def _reconnect_shared_stream(self) -> bool:
        """Attempt to reopen the shared stream after a post-ready transport drop.

        Returns:
            `True` if a new stream was opened (caller should resume fanout),
            `False` if all reconnect attempts were exhausted or the controller
            was closed in the meantime.
        """
        if self._transport is None:
            return False
        # Use the current shared-stream filter (latest computed union); if
        # subscriptions changed during the drop, this picks up the new shape.
        base_filter = self._shared_stream_filter
        if base_filter is None:
            return False
        for attempt in range(self._shared_max_reconnect_attempts):
            if self._closed:
                return False
            stream_params: dict[str, Any] = dict(base_filter)
            if self._cursor is not None:
                stream_params["since"] = self._cursor
            try:
                new_stream = self._transport.open_event_stream(stream_params)
                await new_stream.ready
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._reconnect_sleep(attempt)
                continue
            self._shared_stream = new_stream
            return True
        return False

    async def _reconcile_stream(self, candidate_filter: SubscribeParams) -> None:
        """Ensure the shared SSE covers `candidate_filter`. Rotate if not.

        Open-new-before-close-old: any events buffered server-side between
        the two opens are replayed on the new SSE, and the per-thread
        `_seen_event_ids` set dedupes the overlap. Awaits `new_stream.ready`
        so the HTTP connection is established before returning, guaranteeing
        that both old and new streams are simultaneously connected during
        rotation (enabling correct peak-count tracking and dedup correctness).
        """
        await self._await_run_start_gate(timeout=self._run_start_timeout)
        from langgraph_sdk.stream.subscription import filter_covers

        if self._transport is None:
            raise RuntimeError("AsyncThreadStream not entered — use `async with`.")

        if (
            self._shared_stream is not None
            and self._shared_stream_filter is not None
            and filter_covers(self._shared_stream_filter, dict(candidate_filter))
        ):
            return  # Existing stream is sufficient.

        new_filter = self._compute_current_union(extra=candidate_filter)
        stream_params: dict[str, Any] = dict(new_filter)
        if self._cursor is not None:
            stream_params["since"] = self._cursor
        new_stream = self._transport.open_event_stream(stream_params)
        old_stream = self._shared_stream
        self._shared_stream = new_stream
        self._shared_stream_filter = new_filter
        # Await the new stream's ready future so the HTTP connection is
        # established before we schedule the old stream's close. This ensures
        # old and new are simultaneously open during the rotation window.
        await new_stream.ready
        if old_stream is not None:
            # Schedule the old stream's close as a separate task so the
            # caller doesn't pay close() latency in the rotation hot path.
            asyncio.create_task(_close_after(old_stream))  # noqa: RUF006

    def _compute_current_union(
        self, extra: SubscribeParams | None = None
    ) -> dict[str, Any]:
        from langgraph_sdk.stream.subscription import compute_union_filter

        filters: list[dict[str, Any]] = [
            dict(sub.params) for sub in self._subscriptions.values()
        ]
        if extra is not None:
            filters.append(dict(extra))
        # Always include lifecycle in the shared SSE so the fanout consumer
        # sees root-terminal events in seq order with the projection events.
        # See `_is_root_terminal_lifecycle` -- the fanout uses it to push
        # the `None` sentinel into sub queues when the run ends naturally,
        # which is what makes projection iterators exit on a long-lived
        # SSE that doesn't EOF after the run. Per-subscription filtering
        # (`matches_subscription`) drops lifecycle events for any
        # subscription that didn't ask for them, so user-visible queues
        # don't see leaked events.
        filters.append({"channels": ["lifecycle"]})
        return compute_union_filter(filters)

    async def _dedup_iter(self, source: AsyncIterator[Event]) -> AsyncIterator[Event]:
        async for event in source:
            event_id = event.get("event_id")
            if event_id is not None:
                if event_id in self._seen_event_ids:
                    continue
                self._seen_event_ids.add(event_id)
            yield event

    async def _send_command(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a protocol command and return the `result` payload.

        Returns `{}` for 202/204 responses (no body). Raises `RuntimeError`
        with the protocol code/message when the server returns an error
        envelope (`{"type": "error", ...}`).
        """
        if self._transport is None:
            raise RuntimeError("AsyncThreadStream not entered — use `async with`.")
        command_id = self._next_command_id
        self._next_command_id += 1
        response = await self._transport.send_command(
            {"id": command_id, "method": method, "params": params}
        )
        if response is None:
            # 202/204 — no body. Caller gets an empty result.
            return {}
        if response.get("type") == "error":
            code = response.get("error", "unknown")
            message = response.get("message", "")
            raise RuntimeError(f"Protocol error [{code}]: {message}")
        meta = response.get("meta")
        if isinstance(meta, dict):
            applied_through_seq = meta.get("applied_through_seq")
            if self._controller is not None:
                self._controller.observe_applied_through_seq(applied_through_seq)
        return response.get("result", {})

    async def _await_run_start_gate(self, *, timeout: float | None = None) -> None:
        """Wait for the current run.start to commit the thread server-side.

        No-op when no run.start is in flight. Re-raises if run.start failed.
        Raises `asyncio.TimeoutError` if `timeout` is set and the gate does
        not resolve in time; the gate itself is left intact for later callers.
        """
        gate = self._run_start_ready
        if gate is None or gate.done():
            return
        if timeout is None:
            await gate
        else:
            await asyncio.wait_for(asyncio.shield(gate), timeout=timeout)

    def _ensure_lifecycle_watcher_running(self) -> None:
        if self._lifecycle_watcher_task is not None:
            return
        self._lifecycle_watcher_task = asyncio.create_task(
            self._run_lifecycle_watcher()
        )

    def _observe_lifecycle_event(self, event: Event) -> None:
        seq = event.get("seq")
        if isinstance(seq, int) and (
            self._lifecycle_cursor is None or seq > self._lifecycle_cursor
        ):
            self._lifecycle_cursor = seq

    def _lifecycle_stream_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"channels": ["lifecycle", "input"]}
        if self._lifecycle_cursor is not None:
            params["since"] = self._lifecycle_cursor
        return params

    async def _run_lifecycle_watcher(self) -> None:
        """Always-on SSE consuming lifecycle + input channels."""
        if self._transport is None:
            return
        reconnect_attempts = 0
        while not self._closed:
            try:
                handle = self._transport.open_event_stream(
                    self._lifecycle_stream_params()
                )
                self._lifecycle_watcher_handle = handle
                await asyncio.wait_for(handle.ready, timeout=5.0)
                async for event in handle.events:
                    if self._closed:
                        return
                    self._observe_lifecycle_event(event)
                    await self._apply_lifecycle_event(event)
                err = await handle.done
                if err is None or isinstance(err, asyncio.CancelledError):
                    # Clean EOF: stream ended without a terminal lifecycle
                    # event. Resolve `_run_done` as errored so awaiters of
                    # `thread.output` don't hang.
                    if err is None:
                        run_done = self._run_done
                        if run_done is not None and not run_done.done():
                            run_done.set_result(
                                _RunTerminal(
                                    status="errored",
                                    error=RuntimeError(
                                        "lifecycle stream ended before terminal event"
                                    ),
                                )
                            )
                    return
                reconnect_attempts += 1
                if reconnect_attempts > self._lifecycle_max_reconnect_attempts:
                    raise err
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reconnect_attempts += 1
                if reconnect_attempts <= self._lifecycle_max_reconnect_attempts:
                    await asyncio.sleep(0.05)
                    continue
                run_done = self._run_done
                if run_done is not None and not run_done.done():
                    run_done.set_result(
                        _RunTerminal(
                            status="errored",
                            error=RuntimeError(f"Lifecycle transport failed: {exc}"),
                        )
                    )
                return

    async def _fetch_state(self) -> dict[str, Any]:
        """Fetch the current thread state from the REST endpoint."""
        return await self._http.get(
            f"/threads/{self.thread_id}/state",
            headers=self._headers or None,
        )

    def _state_is_terminal(self, state: dict[str, Any]) -> bool:
        """Return `True` if the thread state has no pending tasks or next nodes."""
        return not state.get("next") and not state.get("tasks")

    def _can_return_existing_state_immediately(self) -> bool:
        """Return `True` if we can try the REST state before waiting on the lifecycle.

        True only when the caller passed an explicit `thread_id` (not a minted
        UUID) and no run has been seen yet, indicating a potential reattach to
        an already-terminal thread.
        """
        return self._explicit_thread_id and not self._run_seen

    async def _wait_for_run_done(self) -> _RunTerminal:
        """Await `_run_done`, raising if the stream was never entered or no run exists.

        Raises:
            RuntimeError: stream not entered, or no run started and no explicit
                thread_id was provided.
        """
        if self._run_done is None:
            raise RuntimeError("AsyncThreadStream not entered — use async with")
        if not self._run_seen and not self._explicit_thread_id:
            raise RuntimeError(
                "thread.output: no run has been started and no explicit thread_id "
                "was provided. Call thread.run.start() first."
            )
        return await self._run_done

    async def _apply_lifecycle_event(self, event: Event) -> None:
        """Update `interrupted` / `interrupts` / `_run_done` from a lifecycle or input event."""
        method = event.get("method")
        if method == "input.requested":
            params = event.get("params") or {}
            data = params.get("data") if isinstance(params, dict) else None
            interrupt_id = data.get("interrupt_id") if isinstance(data, dict) else None
            if isinstance(interrupt_id, str):
                payload: InterruptPayload = {
                    "interrupt_id": interrupt_id,
                    "value": data.get("value") if isinstance(data, dict) else None,
                    "namespace": params.get("namespace") or []
                    if isinstance(params, dict)
                    else [],
                }
                async with self._interrupts_lock:
                    was_interrupted = self.interrupted
                    self.interrupts.append(payload)
                    self.interrupted = True
                # On the rising edge of `interrupted`, push the terminal
                # sentinel into every active projection subscription so their
                # iterators exit cleanly. The run is paused — not done — so
                # the shared SSE and fanout keep running; a subsequent
                # `async for snap in thread.values:` (or any other
                # projection) registers a fresh subscription and resumes
                # iteration once the consumer calls `run.respond(...)`.
                if not was_interrupted:
                    self._signal_paused()
        elif method == "lifecycle":
            params = event.get("params") or {}
            data = params.get("data") if isinstance(params, dict) else None
            phase = data.get("event") if isinstance(data, dict) else None
            if phase in ("started", "running"):
                # Mark that we have observed an active run so thread.output
                # knows a run exists (handles reattach without run.start).
                self._run_seen = True
            elif phase in ("completed", "failed"):
                # Why: interrupts describe current-run state. Clear on terminal
                # lifecycle so a subsequent run.respond() can't fire against a
                # stale prior-run interrupt_id. Acquire `_interrupts_lock` so
                # any in-flight `run.respond` either completes against the
                # pre-clear snapshot or sees the cleared state — never both.
                async with self._interrupts_lock:
                    self.interrupted = False
                    self.interrupts = []
                run_done = self._run_done
                if run_done is not None and not run_done.done():
                    if phase == "failed":
                        error_msg = (
                            data.get("error") if isinstance(data, dict) else None
                        )
                        error = RuntimeError(
                            f"Run errored: {error_msg}" if error_msg else "Run errored"
                        )
                        self._fail_active_message_streams(error)
                        self._fail_active_tool_calls(error)
                        run_done.set_result(_RunTerminal(status="errored", error=error))
                    else:
                        run_done.set_result(_RunTerminal(status="completed"))
