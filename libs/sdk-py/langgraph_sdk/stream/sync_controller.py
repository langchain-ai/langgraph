"""Synchronous shared-stream fan-out controller for v3 thread streaming."""

from __future__ import annotations

import contextlib
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from queue import Queue as _Queue
from typing import Any

from langchain_protocol import Event, SubscribeParams

from langgraph_sdk.stream.subscription import compute_union_filter, filter_covers
from langgraph_sdk.stream.transport import (
    SyncEventStreamHandle,
    SyncProtocolTransport,
)

_logger = logging.getLogger(__name__)


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


@dataclass
class _SyncSubscription:
    id: int
    params: SubscribeParams
    queue: _Queue[Event | None] = field(default_factory=_Queue)
    # Why: using `queue.Queue` in the annotation causes ty to resolve `queue`
    # as the field being defined (name shadowing), not the stdlib module.


_DEFAULT_RUN_START_TIMEOUT: float = 30.0


class SyncStreamController:
    """Owns the sync shared SSE handle, subscription registry, and fan-out thread."""

    def __init__(
        self,
        transport: SyncProtocolTransport,
        *,
        run_start_gate: threading.Event | None = None,
        run_start_timeout: float = _DEFAULT_RUN_START_TIMEOUT,
        max_reconnect_attempts: int = 5,
        reconnect_backoff_base: float = 0.1,
        reconnect_backoff_cap: float = 10.0,
    ) -> None:
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
        # When None, no gate is applied and reconcile_stream proceeds immediately.
        # SyncThreadStream passes an un-set Event so subscriptions wait until
        # run.start completes.
        self._run_start_gate = run_start_gate
        self._run_start_timeout = run_start_timeout
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_backoff_base = reconnect_backoff_base
        self._reconnect_backoff_cap = reconnect_backoff_cap
        self._drain_threads: set[threading.Thread] = set()

    def register_subscription(self, params: SubscribeParams) -> _SyncSubscription:
        with self._lock:
            sub = _SyncSubscription(id=self._next_subscription_id, params=params)
            self._next_subscription_id += 1
            self._subscriptions[sub.id] = sub
            return sub

    def unregister_subscription(self, subscription_id: int) -> None:
        with self._lock:
            self._subscriptions.pop(subscription_id, None)

    def signal_paused(self) -> None:
        """Wake every active subscription iterator on interrupt (run pause).

        Pushes the terminal sentinel (`None`) into every subscription queue.
        Iterators see `None` and return; the shared SSE keeps running so
        re-iteration after `run.respond(...)` registers a fresh subscription
        and resumes.
        """
        with self._lock:
            subs = list(self._subscriptions.values())
        for sub in subs:
            sub.queue.put(None)

    def reconcile_stream(self, candidate_filter: SubscribeParams) -> None:
        if self._run_start_gate is not None and not self._run_start_gate.wait(
            timeout=self._run_start_timeout
        ):
            raise TimeoutError("Sync run.start gate timeout.")
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
                drain_thread = threading.Thread(
                    target=self._drain_and_close,
                    args=(old_stream,),
                    daemon=True,
                    name="langgraph-sdk-sync-rotation-drain",
                )
                self._drain_threads.add(drain_thread)
                drain_thread.start()

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
                    # Root-terminal lifecycle: push `None` into all sub
                    # queues so projection iterators exit when the run
                    # ends naturally. Terminal is processed in seq order
                    # on the shared SSE, so in-flight values/tools/
                    # messages events for this run are already queued
                    # before None.
                    if _is_root_terminal_lifecycle(event):
                        self.signal_paused()
            except Exception:
                pass  # transport drop — attempt reconnect below

            with self._lock:
                if self._shared_stream is not shared:
                    continue  # rotation happened; pick up new stream
                # No rotation — check if this was a transport drop
            err = shared.error()
            if err is not None and not self._closed:
                if self._reconnect_shared_stream():
                    continue
            break

        with self._lock:
            for sub in self._subscriptions.values():
                sub.queue.put(None)

    def _reconnect_shared_stream(self) -> bool:
        with self._lock:
            base_filter = self._shared_stream_filter
        if base_filter is None:
            return False
        for _ in range(self._max_reconnect_attempts):
            with self._lock:
                if self._closed:
                    return False
                params = self._filter_with_since(base_filter)
            try:
                new_stream = self._transport.open_event_stream(params)
            except Exception:
                continue
            with self._lock:
                self._shared_stream = new_stream
            return True
        return False

    def _compute_current_union(
        self, extra: SubscribeParams | None = None
    ) -> dict[str, Any]:
        filters = [dict(sub.params) for sub in self._subscriptions.values()]
        if extra is not None:
            filters.append(dict(extra))
        # Always include lifecycle in the shared SSE filter so `_fanout`
        # sees root-terminal events in seq order with the projection
        # events. See `_is_root_terminal_lifecycle`.
        filters.append({"channels": ["lifecycle"]})
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

    def _drain_and_close(self, handle: SyncEventStreamHandle) -> None:
        """Drain remaining events from an old handle before closing it.

        Runs in a background thread spawned by `reconcile_stream` on rotation
        so buffered events are not lost when the shared stream is replaced.
        Events are dispatched to subscribers regardless of `_closed` so that
        already-buffered events reach consumers before the handle is closed.
        """
        from langgraph_sdk.stream.subscription import matches_subscription

        try:
            for event in self._dedup_iter(handle.events):
                with self._lock:
                    subscriptions = list(self._subscriptions.values())
                for sub in subscriptions:
                    if matches_subscription(event, sub.params):
                        sub.queue.put(event)
        except Exception as err:
            _logger.debug("rotation drain exception: %r", err)
        finally:
            with contextlib.suppress(Exception):
                handle.close()
            with self._lock:
                self._drain_threads.discard(threading.current_thread())

    def _reconnect_sleep(self, attempt: int) -> None:
        """Sleep with exponential backoff + jitter before a reconnect attempt."""
        base = self._reconnect_backoff_base
        cap = self._reconnect_backoff_cap
        delay = min(cap, base * (2**attempt))
        jitter = random.uniform(0, delay * 0.25)
        time.sleep(delay + jitter)

    def _reconnect_shared_stream(self) -> bool:
        """Attempt to reopen the shared stream after a transport drop.

        Returns True if a new stream was successfully opened, False if all
        reconnect attempts were exhausted or the controller was closed.
        """
        base_filter = self._shared_stream_filter
        if base_filter is None:
            return False
        for attempt in range(self._max_reconnect_attempts):
            if self._closed:
                return False
            if attempt > 0:
                self._reconnect_sleep(attempt - 1)
            try:
                new_handle = self._transport.open_event_stream(
                    self._filter_with_since(base_filter)
                )
                old = self._shared_stream
                self._shared_stream = new_handle
                if old is not None:
                    with contextlib.suppress(Exception):
                        old.close()
                return True
            except Exception as err:
                _logger.debug("sync reconnect attempt %d failed: %r", attempt, err)
        return False

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
        with self._lock:
            drain_threads = set(self._drain_threads)
        for drain in drain_threads:
            if drain.is_alive():
                with contextlib.suppress(RuntimeError):
                    drain.join(timeout=1.0)
