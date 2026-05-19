"""Tests for SyncThreadStream — Tasks 9.1 through 9.6."""

from __future__ import annotations

import threading
import time

import httpx

from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.threads import SyncThreadsClient
from streaming._sync_fake_server import SyncFakeServer, SyncStreamScript

# ---------------------------------------------------------------------------
# Task 9.1 — run_start_gate
# ---------------------------------------------------------------------------


def test_sync_subscribe_before_run_start_waits_on_gate():
    """A subscribe issued before run.start completes must block until the
    gate is set, mirroring async behavior."""
    fake = SyncFakeServer()
    # Lifecycle + fanout streams: empty so threads terminate cleanly.
    fake.script_sequence(
        [
            SyncStreamScript(events=[]),  # lifecycle watcher
            SyncStreamScript(events=[]),  # first subscribe
        ]
    )

    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            controller = thread._controller
            assert controller is not None

            started = threading.Event()

            def slow_subscriber() -> None:
                started.set()
                list(thread.subscribe(["values"]))

            t = threading.Thread(target=slow_subscriber)
            t.start()
            started.wait(timeout=0.5)

            # Set the gate manually (simulating run.start completing)
            time.sleep(0.05)
            assert controller._run_start_gate is not None
            controller._run_start_gate.set()

            t.join(timeout=2.0)

    # The subscriber should have unblocked and terminated cleanly.
    assert not t.is_alive(), "subscriber thread should have terminated"


# ---------------------------------------------------------------------------
# Task 9.2 — reconnect backoff + ready check
# ---------------------------------------------------------------------------


def test_sync_reconnect_uses_backoff_between_attempts(monkeypatch):
    """_reconnect_shared_stream sleeps between retry attempts with exp+jitter
    backoff, mirroring the async reconnect behavior."""
    import langgraph_sdk.stream.sync_controller as _ctrl_mod

    sleeps: list[float] = []
    monkeypatch.setattr(_ctrl_mod.time, "sleep", lambda d: sleeps.append(d))

    from langgraph_sdk.stream.sync_controller import SyncStreamController
    from langgraph_sdk.stream.transport.sync_http import (
        SyncEventStreamHandle,
        SyncProtocolSseTransport,
    )

    class _FailingTransport(SyncProtocolSseTransport):
        """Transport that always raises on open_event_stream."""

        def open_event_stream(self, params: dict) -> SyncEventStreamHandle:  # noqa: ARG002
            raise RuntimeError("scripted transport failure")

    with httpx.Client(base_url="http://test") as raw:
        transport = _FailingTransport(client=raw, thread_id="t-1")
        controller = SyncStreamController(transport, max_reconnect_attempts=5)
        controller._shared_stream_filter = {"channels": ["values"]}
        result = controller._reconnect_shared_stream()

    assert result is False, "all attempts should have failed"
    # Attempts 0..4 → sleeps before attempts 1..4 → 4 sleeps
    assert len(sleeps) == 4, f"Expected 4 sleeps, got {sleeps}"
    # Backoff should grow (each delay is larger than previous, ignoring jitter)
    delays_without_jitter = [0.1 * (2**i) for i in range(4)]
    for i, (sleep, expected_base) in enumerate(
        zip(sleeps, delays_without_jitter, strict=False)
    ):
        assert sleep >= expected_base, (
            f"sleep[{i}]={sleep} < expected base {expected_base}"
        )


# ---------------------------------------------------------------------------
# Task 9.3 — rotation drains buffered events from old stream
# ---------------------------------------------------------------------------


def test_sync_rotation_does_not_lose_buffered_events():
    """When the shared stream rotates, old-stream events already in the queue
    are not dropped.  _drain_and_close dispatches remaining events from the
    old handle to subscribers before closing it."""
    import queue
    from collections.abc import Iterator
    from typing import Any

    from langgraph_sdk.stream.sync_controller import SyncStreamController
    from langgraph_sdk.stream.transport.sync_http import (
        SyncEventStreamHandle,
        SyncProtocolSseTransport,
    )
    from streaming._events import values_event

    event_a = values_event(seq=1, counter=1)

    class _ScriptedTransport(SyncProtocolSseTransport):
        """First call produces event_a; second call produces an empty stream."""

        def open_event_stream(self, params: dict) -> SyncEventStreamHandle:  # noqa: ARG002
            def _gen_a() -> Iterator[Any]:
                yield event_a

            def _gen_empty() -> Iterator[Any]:
                return
                yield  # pragma: no cover

            # Alternate: first call → a, second → empty.
            if not hasattr(self, "_call_count"):
                self._call_count = 0
            self._call_count += 1
            events_gen: Iterator[Any] = (
                _gen_a() if self._call_count == 1 else _gen_empty()
            )
            return SyncEventStreamHandle(
                events=events_gen,
                error=lambda: None,
                close=lambda: None,
            )

    with httpx.Client(base_url="http://test") as raw:
        transport = _ScriptedTransport(client=raw, thread_id="t-1")
        controller = SyncStreamController(transport)
        sub = controller.register_subscription({"channels": ["values"]})

        # First reconcile — opens old stream (event_a available immediately).
        controller.reconcile_stream({"channels": ["values"]})
        # Do NOT start fanout; let reconcile_stream cause a rotation directly.

        # Second reconcile: rotates to empty stream; drain thread handles old.
        controller.reconcile_stream({"channels": ["values", "updates"]})

        # Start fanout AFTER rotation (picks up the new empty stream).
        controller.ensure_fanout_running()

        # Allow drain thread to finish before collecting results.
        controller.close()

        received = []
        while True:
            try:
                item = sub.queue.get_nowait()
                if item is None:
                    continue
                received.append(item)
            except queue.Empty:
                break

    seqs = [e.get("seq") for e in received]
    assert 1 in seqs, f"event_a (seq=1) not received via drain; got seqs={seqs}"
