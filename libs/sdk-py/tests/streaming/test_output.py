"""Tests for `thread.output` — REST-backed awaitable for terminal thread state."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from streaming._events import (
    lifecycle_completed_event,
    lifecycle_errored_event,
    lifecycle_started_event,
)
from streaming._fake_server import FakeServer


async def test_output_waits_for_lifecycle_then_fetches_state():
    """run.start + lifecycle completion → await thread.output returns state values."""
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            lifecycle_completed_event(seq=1),
        ]
    )
    fake.set_state({"messages": ["hello"]})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            result = await thread.output
    assert result == {"messages": ["hello"]}
    assert fake.state_request_count == 1


async def test_output_with_lifecycle_replay():
    """Lifecycle completion event already in stream when output is awaited → returns immediately."""
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=0)])
    fake.set_state({"counter": 42})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            # Yield to the event loop until the lifecycle watcher has processed
            # the completed event and resolved _run_done.
            for _ in range(50):
                run_done = thread._run_done
                if run_done is not None and run_done.done():
                    break
                await asyncio.sleep(0)
            result = await thread.output
    assert result == {"counter": 42}
    assert fake.state_request_count == 1


async def test_output_completed_before_attach_returns_rest_state():
    """Explicit thread_id, no run.start, state is terminal → returns REST state without hanging."""
    fake = FakeServer()
    fake.script([])  # No lifecycle events — nothing in flight.
    fake.set_state({"done": True})  # Terminal: next=[], tasks=[]
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        # Pass explicit thread_id — this is the reattach scenario.
        async with threads.stream(
            thread_id="existing-1", assistant_id="agent"
        ) as thread:
            result = await thread.output
    assert result == {"done": True}
    assert fake.state_request_count == 1


async def test_output_explicit_thread_id_non_terminal_falls_through_to_lifecycle():
    """Explicit thread_id, non-terminal state → falls through fast path and waits for lifecycle."""
    fake = FakeServer()
    # Non-terminal state: next has a pending node so the fast-path check fails.
    # The same state is returned on the second fetch (after lifecycle fires).
    fake.set_state(values={"result": "done"}, next=["still_running"])
    # Script a lifecycle completion event — the watcher fires this to resolve _run_done.
    fake.script([lifecycle_completed_event(seq=1)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        # Explicit thread_id triggers the fast-path check (no run.start called).
        async with threads.stream(
            thread_id="existing-2", assistant_id="agent"
        ) as thread:
            # _run_seen is False (no run.start), explicit_thread_id is True:
            # _can_return_existing_state_immediately() returns True.
            # First fetch returns non-terminal state → falls through to _wait_for_run_done.
            # Lifecycle completed event resolves _run_done.
            # Second fetch returns same state; values are returned.
            result = await thread.output
    assert result == {"result": "done"}
    # Two fetches: one for the fast-path terminal check, one after lifecycle fires.
    assert fake.state_request_count == 2


async def test_output_does_not_open_values_stream():
    """Awaiting thread.output must NOT open a values SSE channel."""
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=0)])
    fake.set_state({"x": 1})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            await thread.output

    # No stream request body should contain a "values" channel.
    for body in fake.stream_request_bodies:
        channels = body.get("channels", [])
        assert "values" not in channels, (
            f"Expected no 'values' channel, but found one in: {body}"
        )


async def test_output_multiple_awaiters_share_one_state_request():
    """Awaiting thread.output twice shares a single underlying task and REST call."""
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=0)])
    fake.set_state({"shared": True})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            result1, result2 = await asyncio.gather(thread.output, thread.output)
    assert result1 == {"shared": True}
    assert result2 == {"shared": True}
    assert fake.state_request_count == 1


async def test_output_terminal_error_raises():
    """Lifecycle errored event → awaiting thread.output raises RuntimeError."""
    fake = FakeServer()
    fake.script([lifecycle_errored_event(seq=0, error="something exploded")])
    fake.set_state({})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            with pytest.raises(RuntimeError, match="something exploded"):
                await thread.output


async def test_output_no_run_no_lifecycle_raises():
    """Minted thread_id, no run.start, no lifecycle events → raises usage error."""
    fake = FakeServer()
    fake.script([])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        # thread_id=None → minted UUID, explicit_thread_id=False
        async with threads.stream(thread_id=None, assistant_id="agent") as thread:
            with pytest.raises(RuntimeError, match="no run has been started"):
                await thread.output


async def test_output_headers_propagate_to_state_request():
    """Custom headers on the stream session propagate to the GET /state REST call."""
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=0)])
    fake.set_state({"ok": True})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(
            thread_id="t-1",
            assistant_id="agent",
            headers={"X-Custom-Header": "test-value"},
        ) as thread:
            await thread.run.start(input={})
            await thread.output

    assert fake.state_request_count == 1
    assert fake.state_request_headers[0].get("x-custom-header") == "test-value"
