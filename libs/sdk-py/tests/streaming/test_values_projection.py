"""Tests for `thread.values` — state-backed values projection."""

from __future__ import annotations

import asyncio

import httpx

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from streaming._events import (
    lifecycle_completed_event,
    lifecycle_started_event,
    values_event,
)
from streaming._fake_server import FakeServer


async def test_values_subscribes_before_rest_fetch():
    """Values subscription is opened (values channel present in stream body)."""
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    fake.set_state({"x": 1})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            results = []
            async for snapshot in thread.values:
                results.append(snapshot)
                break  # One snapshot is enough to verify subscription was opened.

    # At least one stream request should contain "values" in its channels.
    values_channel_seen = any(
        "values" in body.get("channels", []) for body in fake.stream_request_bodies
    )
    assert values_channel_seen, (
        f"Expected a stream request with 'values' channel, got: "
        f"{fake.stream_request_bodies}"
    )


async def test_values_first_yield_is_rest_state():
    """First item from `async for snapshot in thread.values` equals REST state values."""
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    fake.set_state({"foo": "bar", "count": 42})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            first = None
            async for snapshot in thread.values:
                first = snapshot
                break

    assert first == {"foo": "bar", "count": 42}
    assert fake.state_request_count >= 1


async def test_values_subsequent_yields_from_stream_events():
    """Items after the first come from live values stream events."""
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            values_event(seq=1, counter=1),
            values_event(seq=2, counter=2),
            lifecycle_completed_event(seq=3),
        ]
    )
    fake.set_state({"counter": 0})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            snapshots = []
            async for snapshot in thread.values:
                snapshots.append(snapshot)

    # First snapshot is REST state values.
    assert snapshots[0] == {"counter": 0}
    # Subsequent snapshots are params.data from values events (full data dict).
    assert {"counter": 1} in snapshots
    assert {"counter": 2} in snapshots


async def test_values_completed_run_terminates():
    """Lifecycle completed causes `async for thread.values` to end cleanly."""
    fake = FakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            lifecycle_completed_event(seq=1),
        ]
    )
    fake.set_state({"done": True})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            snapshots = []
            async for snapshot in thread.values:
                snapshots.append(snapshot)

    # Should have terminated without hanging; at least the REST snapshot.
    assert len(snapshots) >= 1
    assert snapshots[0] == {"done": True}


async def test_values_multiple_iterators_allowed():
    """Two concurrent `async for` loops on `thread.values` both yield independently."""
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    fake.set_state({"shared": True})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})

            async def collect_first():
                async for snapshot in thread.values:
                    return snapshot

            result1, result2 = await asyncio.gather(collect_first(), collect_first())

    assert result1 == {"shared": True}
    assert result2 == {"shared": True}


async def test_values_await_delegates_to_output():
    """`await thread.values` returns the same result as `await thread.output`."""
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    fake.set_state({"result": "terminal"})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            values_result = await thread.values

    assert values_result == {"result": "terminal"}


async def test_values_no_historical_retention():
    """First snapshot is current REST state, not a cached/historical value."""
    fake = FakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    # Set REST state to a specific value that can be verified as the source.
    fake.set_state({"current": "state-v1"})
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            first = None
            async for snapshot in thread.values:
                first = snapshot
                break

    # First snapshot must be the REST state, not some cached prior value.
    assert first == {"current": "state-v1"}
    assert fake.state_request_count >= 1
