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
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_protocol import Event, SubscribeParams

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk.stream.transport import EventStreamHandle, ProtocolSseTransport


class InterruptPayload(TypedDict):
    """Payload surfaced when the server requests human input for a thread."""

    interrupt_id: str
    value: Any
    namespace: list[str]


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
            self._owner._ensure_lifecycle_watcher_running()
            result = await self._owner._send_command("run.start", params)
            if not gate.done():
                gate.set_result(None)
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
        outstanding = self._owner.interrupts
        if interrupt_id is None:
            if len(outstanding) == 0:
                raise RuntimeError(
                    "thread.run.respond: no outstanding interrupt. Provide an "
                    "explicit `interrupt_id` or wait for `thread.interrupted`."
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
                    f"thread.run.respond: interrupt_id {interrupt_id!r} does not "
                    "match any outstanding interrupt in `thread.interrupts`."
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
    ) -> None:
        self._http = http
        self._headers = dict(headers or {})
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self._max_queue_size = max_queue_size
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
        self._lifecycle_watcher_task: asyncio.Task[None] | None = None
        self._lifecycle_watcher_handle: EventStreamHandle | None = None
        self._run_start_ready: asyncio.Future[None] | None = None
        self.run = RunModule(self)

    async def __aenter__(self) -> AsyncThreadStream:
        if self._closed:
            raise RuntimeError("AsyncThreadStream is closed and cannot be re-entered.")
        self._transport = ProtocolSseTransport(
            client=self._http.client,
            thread_id=self.thread_id,
            headers=self._headers,
            max_queue_size=self._max_queue_size,
        )
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
        await self._await_run_start_gate()
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

    async def _await_run_start_gate(self) -> None:
        """Wait for the current run.start to commit the thread server-side.

        No-op when no run.start is in flight. Re-raises if run.start failed.
        """
        gate = self._run_start_ready
        if gate is None:
            return
        await gate

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
        surface even when no other subscription is active.

        The watcher waits for the run-start gate before opening so it does not
        race server-side thread creation.
        """
        if self._transport is None:
            return
        try:
            await self._await_run_start_gate()
            handle = self._transport.open_event_stream(
                {"channels": ["lifecycle", "input"]}
            )
            self._lifecycle_watcher_handle = handle
            await asyncio.wait_for(handle.ready, timeout=5.0)
            async for event in handle.events:
                if self._closed:
                    return
                self._apply_lifecycle_event(event)
        except (Exception, asyncio.CancelledError):
            # Why: advisory-only watcher. Any error (HTTP failure, malformed
            # event in `_apply_lifecycle_event`, cancellation on close) must
            # not crash the caller; the watcher is one-shot best-effort.
            return

    def _apply_lifecycle_event(self, event: Event) -> None:
        """Update `interrupted` / `interrupts` state from a lifecycle or input event."""
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
                self.interrupts.append(payload)
                self.interrupted = True
        elif method == "lifecycle":
            params = event.get("params") or {}
            data = params.get("data") if isinstance(params, dict) else None
            phase = data.get("phase") if isinstance(data, dict) else None
            if phase in ("completed", "errored"):
                # Why: interrupts describe current-run state. Clear on terminal
                # lifecycle so a subsequent run.respond() can't fire against a
                # stale prior-run interrupt_id.
                self.interrupted = False
                self.interrupts = []
