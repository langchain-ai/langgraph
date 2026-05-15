"""Synchronous shared-stream fan-out controller for v3 thread streaming."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from queue import Queue as _Queue
from typing import Any

from langchain_protocol import Event, SubscribeParams

from langgraph_sdk.stream.subscription import compute_union_filter, filter_covers
from langgraph_sdk.stream.transport.sync_http import (
    SyncEventStreamHandle,
    SyncProtocolSseTransport,
)


@dataclass
class _SyncSubscription:
    id: int
    params: SubscribeParams
    queue: _Queue[Event | None] = field(default_factory=_Queue)
    # Why: using `queue.Queue` in the annotation causes ty to resolve `queue`
    # as the field being defined (name shadowing), not the stdlib module.


class SyncStreamController:
    """Owns the sync shared SSE handle, subscription registry, and fan-out thread."""

    def __init__(self, transport: SyncProtocolSseTransport) -> None:
        self._transport = transport
        self._next_subscription_id = 1
        self._subscriptions: dict[int, _SyncSubscription] = {}
        self._seen_event_ids: set[str] = set()
        self._shared_stream: SyncEventStreamHandle | None = None
        self._shared_stream_filter: dict[str, Any] | None = None
        self._fanout_thread: threading.Thread | None = None
        self._closed = False
        self._lock = threading.RLock()
        self._cursor: int | None = None

    def register_subscription(self, params: SubscribeParams) -> _SyncSubscription:
        with self._lock:
            sub = _SyncSubscription(id=self._next_subscription_id, params=params)
            self._next_subscription_id += 1
            self._subscriptions[sub.id] = sub
            return sub

    def unregister_subscription(self, subscription_id: int) -> None:
        with self._lock:
            self._subscriptions.pop(subscription_id, None)

    def reconcile_stream(self, candidate_filter: SubscribeParams) -> None:
        with self._lock:
            if (
                self._shared_stream is not None
                and self._shared_stream_filter is not None
                and filter_covers(self._shared_stream_filter, dict(candidate_filter))
            ):
                return
            new_filter = self._compute_current_union(extra=candidate_filter)
            old_stream = self._shared_stream
            self._shared_stream = self._transport.open_event_stream(
                self._filter_with_since(new_filter)
            )
            self._shared_stream_filter = new_filter
            if old_stream is not None:
                old_stream.close()

    def ensure_fanout_running(self) -> None:
        with self._lock:
            if self._fanout_thread is not None and self._fanout_thread.is_alive():
                return
            self._fanout_thread = threading.Thread(
                target=self._fanout,
                name="langgraph-sdk-sync-stream-fanout",
                daemon=True,
            )
            self._fanout_thread.start()

    def _fanout(self) -> None:
        from langgraph_sdk.stream.subscription import matches_subscription

        while True:
            with self._lock:
                if self._closed:
                    return
                shared = self._shared_stream
            if shared is None:
                return
            try:
                for event in self._dedup_iter(shared.events):
                    with self._lock:
                        if self._closed:
                            break
                        subscriptions = list(self._subscriptions.values())
                    for sub in subscriptions:
                        if matches_subscription(event, sub.params):
                            sub.queue.put(event)
            except Exception:
                with self._lock:
                    for sub in self._subscriptions.values():
                        sub.queue.put(None)
                raise
            with self._lock:
                if self._shared_stream is shared:
                    break

        with self._lock:
            for sub in self._subscriptions.values():
                sub.queue.put(None)

    def _compute_current_union(
        self, extra: SubscribeParams | None = None
    ) -> dict[str, Any]:
        filters = [dict(sub.params) for sub in self._subscriptions.values()]
        if extra is not None:
            filters.append(dict(extra))
        return compute_union_filter(filters)

    def observe_applied_through_seq(self, seq: Any) -> None:
        """Advance the reconnect cursor from a command response meta sequence."""
        with self._lock:
            self._observe_seq(seq)

    def _observe_event(self, event: Event) -> None:
        with self._lock:
            self._observe_seq(event.get("seq"))

    def _observe_seq(self, seq: Any) -> None:
        if isinstance(seq, int) and (self._cursor is None or seq > self._cursor):
            self._cursor = seq

    def _filter_with_since(self, params: dict[str, Any]) -> dict[str, Any]:
        out = dict(params)
        if self._cursor is not None:
            out["since"] = self._cursor
        return out

    def _dedup_iter(self, source: Any) -> Any:
        for event in source:
            event_id = event.get("event_id")
            if event_id is not None:
                if event_id in self._seen_event_ids:
                    continue
                self._seen_event_ids.add(event_id)
            self._observe_event(event)
            yield event

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            shared = self._shared_stream
            for sub in self._subscriptions.values():
                sub.queue.put(None)
        if shared is not None:
            shared.close()
        thread = self._fanout_thread
        if thread is not None and thread.is_alive():
            with contextlib.suppress(RuntimeError):
                thread.join(timeout=1.0)
