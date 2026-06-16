"""Tests for the lifecycle watcher: `interrupted` / `interrupts` state."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import httpx

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from streaming._events import (
    input_requested_event,
    lifecycle_completed_event,
    lifecycle_event,
)
from streaming._fake_server import FakeServer, _StreamScript


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
    """Terminal lifecycle event clears interrupted/interrupts."""
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


async def test_lifecycle_clean_eof_resolves_run_done_with_errored():
    """If the lifecycle SSE stream ends cleanly (server closes without a
    terminal `completed` or `errored` event), `_run_done` must resolve with
    an errored terminal so awaiters don't hang."""
    import pytest

    fake = FakeServer()
    # Emit a non-terminal lifecycle event, then close cleanly without
    # `completed` or `errored`.
    fake.script([lifecycle_event(seq=0, phase="started")])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            run_done = thread._run_done
            assert run_done is not None
            terminal = await asyncio.wait_for(run_done, timeout=2.0)
    assert terminal.status == "errored"
    assert terminal.error is not None
    assert "ended before terminal" in str(terminal.error)
    # Quiet unused-import warning under strict configs.
    _ = pytest


async def test_lifecycle_mid_iteration_error_resolves_run_done_with_error(
    monkeypatch: Any,
) -> None:
    """If the transport reports an error via `handle.done` after iteration
    exits without a terminal lifecycle event, `_run_done` propagates the
    transport error rather than the generic clean-EOF message."""
    from langgraph_sdk.stream.transport import EventStreamHandle, ProtocolSseTransport

    def synthetic_handle() -> EventStreamHandle:
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()
        ready.set_result(None)
        done: asyncio.Future[BaseException | None] = loop.create_future()
        done.set_result(RuntimeError("simulated transport error"))

        async def empty_events() -> Any:
            if False:
                yield  # pragma: no cover  # make this an async generator
            return

        async def noop_close() -> None:
            return

        return EventStreamHandle(
            events=empty_events(),
            ready=ready,
            done=done,
            close=noop_close,
        )

    def patched_open(_self: ProtocolSseTransport, _params: Any) -> EventStreamHandle:
        return synthetic_handle()

    monkeypatch.setattr(ProtocolSseTransport, "open_event_stream", patched_open)

    fake = FakeServer()
    fake.script([])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            run_done = thread._run_done
            assert run_done is not None
            terminal = await asyncio.wait_for(run_done, timeout=2.0)
    assert terminal.status == "errored"
    assert terminal.error is not None
    assert "simulated transport error" in str(terminal.error)
    # Quiet unused-import warnings under strict configs.
    _ = contextlib


async def test_lifecycle_watcher_reconnects_with_since_after_transport_drop():
    fake = FakeServer()
    fake.set_state({"ok": True})
    fake.script_sequence(
        [
            _StreamScript(
                events=[lifecycle_event(seq=1, phase="running")],
                fail_after=1,
            ),
            _StreamScript(events=[lifecycle_completed_event(seq=2)]),
        ]
    )
    async with httpx.AsyncClient(
        transport=fake.transport, base_url="http://test"
    ) as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="existing", assistant_id="agent") as thread:
            for _ in range(20):
                run_done = thread._run_done
                if run_done is not None and run_done.done():
                    break
                await asyncio.sleep(0.05)
            assert thread._run_done is not None
            terminal = thread._run_done.result()

    assert terminal.status == "completed"
    assert terminal.error is None
    assert fake.stream_request_bodies[0]["channels"] == ["lifecycle", "input"]
    assert fake.stream_request_bodies[1]["channels"] == ["lifecycle", "input"]
    assert fake.stream_request_bodies[1]["since"] == 1
