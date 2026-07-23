"""Stream controller: subscription registry and fan-out for AsyncThreadStream.

`StreamController` manages the set of active subscriptions against one shared
SSE connection, routing events from the shared stream to per-subscription
queues.  It is the centralised place for:

- subscription registration / teardown
- shared-stream lifecycle (open, rotate, close)
- dedup of replayed events across rotations
- fan-out from the shared stream to subscriber queues
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_protocol import Event, SubscribeParams

from langgraph_sdk.stream.transport import AsyncProtocolTransport, EventStreamHandle

# ---------------------------------------------------------------------------
# Bounded LRU set for event-id dedup
# ---------------------------------------------------------------------------


class _SeenEventIds:
    """LRU set of event ids with bounded memory."""

    __slots__ = ("_data", "_maxsize")

    def __init__(self, maxsize: int = 10_000) -> None:
        self._data: OrderedDict[str, None] = OrderedDict()
        self._maxsize = maxsize

    def add(self, event_id: str) -> None:
        if event_id in self._data:
            self._data.move_to_end(event_id)
            return
        self._data[event_id] = None
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __contains__(self, event_id: object) -> bool:
        return event_id in self._data

    def __iter__(self):
        return iter(self._data)


# ---------------------------------------------------------------------------
# Per-subscription record
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)


@dataclass
class _Subscription:
    """Internal record for one active subscription on a `StreamController`."""

    id: int
    params: SubscribeParams
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # Why: asyncio.Queue[Event | None] as a subscript in the field annotation
    # causes a type error with ty; bare asyncio.Queue is accepted.


# ---------------------------------------------------------------------------
# Rotation close helper
# ---------------------------------------------------------------------------


async def _close_after(handle: EventStreamHandle, *, delay: float = 0.0) -> None:
    """Close a handle, optionally after a brief delay.

    Used to detach closing the old stream from the synchronous rotation step
    so the new stream can absorb server-side replayed events first.
    """
    if delay:
        await asyncio.sleep(delay)
    await handle.close()


# ---------------------------------------------------------------------------
# StreamController
# ---------------------------------------------------------------------------


class StreamController:
    """Manages subscriptions and fan-out against one shared SSE connection.

    Responsibilities:
      - subscription registry (register / unregister)
      - shared-stream lifecycle (open on first subscribe, rotate on filter widen)
      - dedup of replayed events via a bounded LRU `_SeenEventIds`
      - fan-out from the shared stream to per-subscription queues

    Args:
        transport: the `AsyncProtocolTransport` bound to this thread session.
        run_start_gate: zero-argument async callable that resolves once the
            current `run.start` has committed server-side (no-op when no
            run is in flight).
        max_queue_size: per-subscription queue bound (default 1024).
        seen_event_ids_max: LRU cap for the dedup set (default 10_000).
    """

    def __init__(
        self,
        *,
        transport: AsyncProtocolTransport,
        run_start_gate: Callable[[], Awaitable[None]] | None = None,  # noqa: ARG002
        max_queue_size: int = 1024,
        seen_event_ids_max: int = 10_000,
        max_reconnect_attempts: int = 5,
        reconnect_backoff_base: float = 0.1,
        reconnect_backoff_cap: float = 2.0,
    ) -> None:
        self._transport = transport
        self._max_queue_size = max_queue_size
        self._seen_event_ids = _SeenEventIds(maxsize=seen_event_ids_max)
        self._next_subscription_id = 1
        self._subscriptions: dict[int, _Subscription] = {}
        self._shared_stream: EventStreamHandle | None = None
        self._shared_stream_filter: dict[str, Any] | None = None
        self._fanout_task: asyncio.Task[None] | None = None
        self._rotation_close_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._cursor: int | None = None
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_backoff_base = reconnect_backoff_base
        self._reconnect_backoff_cap = reconnect_backoff_cap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        params: SubscribeParams = {"channels": list(channels)}
        if namespaces is not None:
            params["namespaces"] = namespaces
        if depth is not None:
            params["depth"] = depth
        return self._subscription_iter(params)

    async def close(self) -> None:
        """Tear down the controller, awaiting any pending rotation closes."""
        if self._closed:
            return
        self._closed = True
        if self._fanout_task is not None:
            self._fanout_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._fanout_task
        if self._shared_stream is not None:
            await self._shared_stream.close()
        if self._rotation_close_tasks:
            await asyncio.gather(*self._rotation_close_tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Subscription internals
    # ------------------------------------------------------------------

    def _register_subscription(self, params: SubscribeParams) -> _Subscription:
        """Allocate a subscription id, create a bounded queue, add to registry."""
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

    # Public aliases used by tests and external callers.
    register_subscription = _register_subscription
    unregister_subscription = _unregister_subscription

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

    # ------------------------------------------------------------------
    # Fan-out
    # ------------------------------------------------------------------

    def _ensure_fanout_running(self) -> None:
        if self._fanout_task is None or self._fanout_task.done():
            self._fanout_task = asyncio.create_task(self._fanout())

    # Public alias.
    ensure_fanout_running = _ensure_fanout_running

    async def _fanout(self) -> None:
        """Single consumer of the shared SSE; routes events to subscriptions.

        Why: rotation in `_reconcile_stream` replaces `_shared_stream` mid-loop.
        Re-read `self._shared_stream` on each outer iteration so we always
        consume from the current handle. The old handle's iterator exhausts
        naturally after `_close_after` closes it.

        On a post-ready transport drop (non-cancelled error in `shared.done`),
        attempts to reconnect up to `_max_reconnect_attempts` times before
        giving up and closing subscriber queues.
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
            except Exception as drop_err:
                _logger.debug("transport drop in fanout: %r", drop_err)

            if self._shared_stream is shared:
                err = await shared.done
                if (
                    err is not None
                    and not isinstance(err, asyncio.CancelledError)
                    and not self._closed
                ):
                    with contextlib.suppress(Exception):
                        await self._shared_stream.close()
                    reconnected = await self._reconnect_shared_stream()
                    if reconnected:
                        continue
                break
            # Rotation: loop again to pick up the new _shared_stream.

        # Terminate consumers cleanly on shutdown / stream-end.
        for sub in self._subscriptions.values():
            sub.queue.put_nowait(None)

    async def _reconnect_sleep(self, attempt: int) -> None:
        """Sleep with exponential backoff and jitter for reconnect attempt *attempt*."""
        base = self._reconnect_backoff_base
        cap = self._reconnect_backoff_cap
        delay = min(cap, base * (2**attempt))
        jitter = random.uniform(0, delay * 0.25)
        await asyncio.sleep(delay + jitter)

    async def _reconnect_shared_stream(self) -> bool:
        """Attempt to reopen the shared stream after a transport drop.

        Returns True if a new stream was successfully opened, False if all
        reconnect attempts were exhausted or the controller was closed.
        """
        # We intentionally use the *current* shared_stream_filter (the latest
        # computed union of all live subscriptions), not the filter that was
        # active when this stream was originally opened. If subscriptions were
        # added or removed during the drop window, the reconnect picks up the
        # new shape.
        base_filter = self._shared_stream_filter
        if base_filter is None:
            return False
        for attempt in range(self._max_reconnect_attempts):
            if self._closed:
                return False
            try:
                new_stream = self._transport.open_event_stream(
                    self._filter_with_since(base_filter)
                )
                await new_stream.ready
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._reconnect_sleep(attempt)
                continue
            self._shared_stream = new_stream
            return True
        return False

    # ------------------------------------------------------------------
    # Stream rotation
    # ------------------------------------------------------------------

    async def _reconcile_stream(self, candidate_filter: SubscribeParams) -> None:
        """Ensure the shared SSE covers `candidate_filter`. Rotate if not.

        Open-new-before-close-old: any events buffered server-side between
        the two opens are replayed on the new SSE, and `_seen_event_ids`
        dedupes the overlap. Awaits `new_stream.ready` so the HTTP connection
        is established before returning.
        """
        from langgraph_sdk.stream.subscription import filter_covers

        if (
            self._shared_stream is not None
            and self._shared_stream_filter is not None
            and filter_covers(self._shared_stream_filter, dict(candidate_filter))
        ):
            return  # Existing stream is sufficient.

        new_filter = self._compute_current_union(extra=candidate_filter)
        new_stream = self._transport.open_event_stream(
            self._filter_with_since(new_filter)
        )
        old_stream = self._shared_stream
        self._shared_stream = new_stream
        self._shared_stream_filter = new_filter
        await new_stream.ready
        if old_stream is not None:
            task = asyncio.create_task(_close_after(old_stream))
            self._rotation_close_tasks.add(task)
            task.add_done_callback(self._rotation_close_tasks.discard)

    async def reconcile_stream(self, candidate_filter: SubscribeParams) -> None:
        """Public alias for `_reconcile_stream`."""
        return await self._reconcile_stream(candidate_filter)

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

    # ------------------------------------------------------------------
    # Cursor tracking
    # ------------------------------------------------------------------

    def observe_applied_through_seq(self, seq: Any) -> None:
        """Advance the reconnect cursor from a command response meta sequence."""
        self._observe_seq(seq)

    def _observe_event(self, event: Event) -> None:
        self._observe_seq(event.get("seq"))

    def _observe_seq(self, seq: Any) -> None:
        if isinstance(seq, int) and (self._cursor is None or seq > self._cursor):
            self._cursor = seq

    def _filter_with_since(self, params: dict[str, Any]) -> dict[str, Any]:
        out = dict(params)
        if self._cursor is not None:
            out["since"] = self._cursor
        return out

    # ------------------------------------------------------------------
    # Dedup iterator
    # ------------------------------------------------------------------

    async def _dedup_iter(self, source: AsyncIterator[Event]) -> AsyncIterator[Event]:
        async for event in source:
            event_id = event.get("event_id")
            if event_id is not None:
                if event_id in self._seen_event_ids:
                    continue
                self._seen_event_ids.add(event_id)
            self._observe_event(event)
            yield event
