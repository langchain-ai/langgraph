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
