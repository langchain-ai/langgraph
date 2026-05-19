"""Async thread-centric streaming surface for the v3 protocol.

`AsyncThreadStream` is an async context manager that owns a
`ProtocolSseTransport` for one thread, dispatches commands (`run.start`,
`run.respond`), exposes typed subscriptions over a single shared SSE
(`subscribe`, `events`), and surfaces lifecycle state (`interrupted`,
`interrupts`) via an always-on lifecycle watcher SSE. Typed projections
(`thread.values`, `thread.messages`, etc.) mirror the v3 protocol surface.

Direct port of `libs/sdk/src/client/stream/index.ts`.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langchain_core.language_models.chat_model_stream import AsyncChatModelStream
from langchain_protocol import Event, SubscribeParams

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk.stream.transport import EventStreamHandle, ProtocolSseTransport


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


# All public protocol channels used by the raw `events` surface.
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
        try:
            await self._thread._reconcile_stream(params)
            self._thread._ensure_fanout_running()
            state = await self._thread._fetch_state()
            yield state["values"]
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                params_field = item.get("params") or {}
                data = (
                    params_field.get("data") if isinstance(params_field, dict) else None
                )
                if data is not None:
                    yield data
        finally:
            self._thread._unregister_subscription(sub.id)


class _MessagesProjection:
    """Typed projection for root-scope `thread.messages`.

    Iterating yields one `AsyncChatModelStream` per message-start event.
    Each iterator owns its own `messages` subscription and routes events
    from the root namespace only.
    """

    def __init__(self, thread: AsyncThreadStream) -> None:
        self._thread = thread

    def __aiter__(self) -> AsyncIterator[AsyncChatModelStream]:
        return self._messages_iter()

    async def _messages_iter(self) -> AsyncGenerator[AsyncChatModelStream, None]:
        if self._thread._transport is None:
            raise RuntimeError("AsyncThreadStream not entered — use `async with`.")
        params: SubscribeParams = {
            "channels": ["messages"],
            "namespaces": [[]],
            "depth": 0,
        }
        sub = self._thread._register_subscription(params)
        active: dict[str, AsyncChatModelStream] = {}
        try:
            await self._thread._reconcile_stream(params)
            self._thread._ensure_fanout_running()
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                params_field = item.get("params") or {}
                if not isinstance(params_field, dict):
                    continue
                if params_field.get("namespace") not in (None, []):
                    continue
                data = params_field.get("data")
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
                        namespace=[],
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
            self._thread._unregister_subscription(sub.id)


def _message_event_id(data: dict[str, Any]) -> str | None:
    message_id = data.get("id") or data.get("message_id")
    return str(message_id) if message_id is not None else None


def _message_route_key(data: dict[str, Any], fallback: str | None = None) -> str:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    run_id = metadata.get("run_id") if metadata else None
    message_id = _message_event_id(data)
    if run_id is not None:
        return f"run:{run_id}"
    if message_id is not None:
        return f"message:{message_id}"
    if fallback is not None:
        return f"message:{fallback}"
    return "__single__"


class ToolCallHandle:
    """Async handle for one root-scope tool call."""

    def __init__(
        self,
        *,
        tool_call_id: str,
        name: str,
        input: Any = None,
        namespace: list[str] | None = None,
    ) -> None:
        self.tool_call_id = tool_call_id
        self.name = name
        self.input = input
        self.namespace = list(namespace or [])
        self.done = False
        self.error: BaseException | None = None
        loop = asyncio.get_running_loop()
        self.output: asyncio.Future[Any] = loop.create_future()
        self._deltas: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def deltas(self) -> AsyncIterator[str]:
        """Stream tool output deltas emitted before the terminal event."""
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

    def __init__(self, thread: AsyncThreadStream) -> None:
        self._thread = thread

    def __aiter__(self) -> AsyncIterator[ToolCallHandle]:
        return self._tool_calls_iter()

    async def _tool_calls_iter(self) -> AsyncGenerator[ToolCallHandle, None]:
        if self._thread._transport is None:
            raise RuntimeError("AsyncThreadStream not entered - use `async with`.")
        params: SubscribeParams = {
            "channels": ["tools"],
            "namespaces": [[]],
            "depth": 0,
        }
        sub = self._thread._register_subscription(params)
        active: dict[str, ToolCallHandle] = {}
        try:
            await self._thread._reconcile_stream(params)
            self._thread._ensure_fanout_running()
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                params_field = item.get("params") or {}
                if not isinstance(params_field, dict):
                    continue
                namespace = params_field.get("namespace") or []
                if namespace != []:
                    continue
                data = params_field.get("data")
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
                        namespace=namespace,
                    )
                    active[tool_call_id] = handle
                    self._thread._register_active_tool_call(handle)
                    yield handle
                elif event_type == "tool-output-delta":
                    handle = active.get(tool_call_id)
                    delta = data.get("delta")
                    if handle is not None and isinstance(delta, str):
                        handle._push_delta(delta)
                elif event_type == "tool-finished":
                    handle = active.pop(tool_call_id, None)
                    if handle is not None:
                        self._thread._unregister_active_tool_call(handle)
                        handle._finish(data.get("output"))
                elif event_type == "tool-error":
                    handle = active.pop(tool_call_id, None)
                    if handle is not None:
                        self._thread._unregister_active_tool_call(handle)
                        message = data.get("message")
                        handle._fail(
                            RuntimeError(
                                str(message) if message else "Tool call errored"
                            )
                        )
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
            err = (
                terminal_err
                if terminal_err is not None
                else RuntimeError("Tool call stream closed before terminal tool event.")
            )
            for handle in active.values():
                self._thread._unregister_active_tool_call(handle)
                handle._fail(err)
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
    ) -> None:
        self._http = http
        self._headers = dict(headers or {})
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self._max_queue_size = max_queue_size
        self._run_start_timeout = run_start_timeout
        self._explicit_thread_id = explicit_thread_id
        self._closed = False
        self._transport: ProtocolSseTransport | None = None
        self._open_handles: list[EventStreamHandle] = []
        self._next_command_id = 1
        self._next_subscription_id = 1
        self._subscriptions: dict[int, _Subscription] = {}
        self._seen_event_ids: set[str] = set()
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
        self._run_start_ready: asyncio.Future[None] | None = None
        self._run_seen: bool = False
        self._run_done: asyncio.Future[_RunTerminal] | None = None
        self._active_message_streams: set[AsyncChatModelStream] = set()
        self._active_tool_calls: set[ToolCallHandle] = set()
        self.run = RunModule(self)
        self.output = _OutputAwaitable(self)
        self.values = _ValuesProjection(self)
        self.messages = _MessagesProjection(self)
        self.tool_calls = _ToolCallsProjection(self)

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
        self._transport = ProtocolSseTransport(
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
                    for sub in list(self._subscriptions.values()):
                        if matches_subscription(event, sub.params):
                            sub.queue.put_nowait(event)
            except Exception:
                # Pump errored — close all subscription queues so consumers
                # don't hang.
                for sub in self._subscriptions.values():
                    sub.queue.put_nowait(None)
                raise
            if self._shared_stream is shared:
                # No rotation happened; stream genuinely ended.
                break
            # Rotation: loop again to pick up the new _shared_stream.

        # Terminate consumers cleanly on shutdown / stream-end.
        for sub in self._subscriptions.values():
            sub.queue.put_nowait(None)

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
        new_stream = self._transport.open_event_stream(new_filter)
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
        # Why: this watcher is intentionally one-shot. If it crashes, it stays
        # dead until the AsyncThreadStream is closed.
        if self._lifecycle_watcher_task is not None:
            return
        self._lifecycle_watcher_task = asyncio.create_task(
            self._run_lifecycle_watcher()
        )

    async def _run_lifecycle_watcher(self) -> None:
        """Always-on SSE consuming lifecycle + input channels.

        Independent of the union-filter shared stream so that interrupts
        surface even when no other subscription is active. Starts immediately
        on session entry (before any run.start) so reattach and thread.output
        work for existing runs.
        """
        if self._transport is None:
            return
        try:
            handle = self._transport.open_event_stream(
                {"channels": ["lifecycle", "input"]}
            )
            self._lifecycle_watcher_handle = handle
            await asyncio.wait_for(handle.ready, timeout=5.0)
            async for event in handle.events:
                if self._closed:
                    return
                await self._apply_lifecycle_event(event)
            # Why: iterator exhausted without `_run_done` being resolved by a
            # terminal lifecycle event. Surface any transport error captured
            # on `handle.done`, otherwise treat the clean EOF as errored so
            # awaiters of `_run_done` (e.g. `thread.output`) don't hang.
            err = await handle.done
            run_done = self._run_done
            if run_done is not None and not run_done.done():
                if err is not None:
                    run_done.set_result(
                        _RunTerminal(
                            status="errored",
                            error=RuntimeError(f"Lifecycle transport failed: {err}"),
                        )
                    )
                else:
                    run_done.set_result(
                        _RunTerminal(
                            status="errored",
                            error=RuntimeError(
                                "lifecycle stream ended before terminal event"
                            ),
                        )
                    )
            return
        except (Exception, asyncio.CancelledError) as exc:
            # Why: advisory-only watcher. Any error (HTTP failure, malformed
            # event in `_apply_lifecycle_event`, cancellation on close) must
            # not crash the caller; the watcher is one-shot best-effort.
            # Resolve _run_done with an error so thread.output doesn't wait
            # forever when the lifecycle transport fails.
            run_done = self._run_done
            if run_done is not None and not run_done.done():
                if not isinstance(exc, asyncio.CancelledError):
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
                    self.interrupts.append(payload)
                    self.interrupted = True
        elif method == "lifecycle":
            params = event.get("params") or {}
            data = params.get("data") if isinstance(params, dict) else None
            phase = data.get("phase") if isinstance(data, dict) else None
            if phase in ("started", "running"):
                # Mark that we have observed an active run so thread.output
                # knows a run exists (handles reattach without run.start).
                self._run_seen = True
            elif phase in ("completed", "errored"):
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
                    if phase == "errored":
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
