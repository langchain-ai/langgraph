"""Tests for the lifecycle watcher: `interrupted` / `interrupts` state."""

from __future__ import annotations

import asyncio

import httpx

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from streaming._events import input_requested_event, lifecycle_event
from streaming._fake_server import FakeServer


async def test_interrupted_starts_false():
    async with httpx.AsyncClient(base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            assert thread.interrupted is False
            assert thread.interrupts == []


async def test_interrupts_populated_from_input_requested_event():
    fake = FakeServer()
    fake.script([input_requested_event(seq=0)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            # Lifecycle watcher consumes asynchronously — poll briefly.
            for _ in range(20):
                if thread.interrupted:
                    break
                await asyncio.sleep(0.05)
    assert thread.interrupted is True
    assert len(thread.interrupts) == 1
    assert thread.interrupts[0]["interrupt_id"] == "i-1"


async def test_aenter_starts_lifecycle_watcher():
    """Entering AsyncThreadStream opens lifecycle/input SSE before run.start."""
    fake = FakeServer()
    fake.script([lifecycle_event(seq=0, phase="started")])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            # The lifecycle watcher task must be created on __aenter__, no run.start needed.
            assert thread._lifecycle_watcher_task is not None
            # Poll until the watcher has consumed the started event.
            for _ in range(20):
                if thread._run_seen:
                    break
                await asyncio.sleep(0.05)
    assert thread._run_seen is True
    # No run.start was ever called — but the server still received a stream request.
    assert len(fake.stream_request_bodies) >= 1


async def test_reattach_observes_terminal_state():
    """Reattach (no run.start) consumes lifecycle replay and observes terminal state."""
    fake = FakeServer()
    fake.script(
        [
            lifecycle_event(seq=0, phase="running"),
            lifecycle_event(seq=1, phase="completed"),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="existing", assistant_id="agent") as thread:
            # Never call run.start — this is a reattach scenario.
            # Poll until _run_done is resolved.
            for _ in range(20):
                run_done = thread._run_done
                if run_done is not None and run_done.done():
                    break
                await asyncio.sleep(0.05)
            assert thread._run_done is not None
            assert thread._run_done.done()
            terminal = thread._run_done.result()
    assert terminal.status == "completed"
    assert terminal.error is None


async def test_terminal_lifecycle_clears_interrupts():
    """Terminal lifecycle event clears interrupted/interrupts (Phase 3 behavior preserved)."""
    fake = FakeServer()
    fake.script([lifecycle_event(seq=0, phase="completed")])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            # Manually set interrupted state to simulate a prior interrupt.
            thread.interrupted = True
            thread.interrupts = [
                {"interrupt_id": "i-1", "value": None, "namespace": []}
            ]
            # Poll until the lifecycle watcher processes the completed event.
            for _ in range(20):
                if not thread.interrupted:
                    break
                await asyncio.sleep(0.05)
    assert thread.interrupted is False
    assert thread.interrupts == []


async def test_lifecycle_error_captured_for_output():
    """Lifecycle error terminal state is captured in _run_done with error set."""
    fake = FakeServer()
    fake.script([lifecycle_event(seq=0, phase="errored", error="something went wrong")])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            # Poll until _run_done is resolved.
            for _ in range(20):
                run_done = thread._run_done
                if run_done is not None and run_done.done():
                    break
                await asyncio.sleep(0.05)
            assert thread._run_done is not None
            assert thread._run_done.done()
            terminal = thread._run_done.result()
    assert terminal.status == "errored"
    assert terminal.error is not None
    assert "something went wrong" in str(terminal.error)


async def test_run_start_sets_run_seen():
    """run.start() sets _run_seen to True (even without lifecycle event)."""
    fake = FakeServer()
    fake.script([])  # No events; the command response is sufficient.
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            assert thread._run_seen is False
            await thread.run.start(input={})
            # _run_seen is set synchronously in run.start, before awaiting the result.
            assert thread._run_seen is True
