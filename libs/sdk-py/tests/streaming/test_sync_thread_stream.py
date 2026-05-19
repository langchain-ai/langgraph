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
