from __future__ import annotations

import httpx

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from streaming._events import lifecycle_event, values_event
from streaming._fake_server import FakeServer


async def test_shared_stream_serves_single_subscription():
    fake = FakeServer()
    fake.script([lifecycle_event(seq=0), values_event(seq=1)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            await thread._reconcile_stream({"channels": ["lifecycle", "values"]})
            assert thread._shared_stream is not None
            received = [
                e async for e in thread._dedup_iter(thread._shared_stream.events)
            ]
    methods = [e["method"] for e in received]
    assert methods == ["lifecycle", "values"]
    assert fake.peak_open_event_streams == 1


async def test_seen_event_ids_dedupes_replayed_events():
    fake = FakeServer()
    fake.script(
        [
            lifecycle_event(seq=0),
            lifecycle_event(seq=0),  # duplicate event_id
            values_event(seq=1),
        ]
    )
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            await thread._reconcile_stream({"channels": ["lifecycle", "values"]})
            assert thread._shared_stream is not None
            received = [
                e async for e in thread._dedup_iter(thread._shared_stream.events)
            ]
    seqs = [e["seq"] for e in received]
    assert seqs == [0, 1]  # the duplicate seq=0 was deduped via event_id


async def test_rotation_when_new_subscription_widens_filter():
    fake = FakeServer()
    fake.script([lifecycle_event(seq=0), values_event(seq=1)])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            # First subscription: lifecycle only.
            await thread._reconcile_stream({"channels": ["lifecycle"]})
            # Second subscription widens to lifecycle + values.
            await thread._reconcile_stream({"channels": ["lifecycle", "values"]})
    # Rotation: two separate SSE requests were opened (old + new).
    assert len(fake.stream_request_bodies) >= 2


async def test_no_rotation_when_existing_filter_covers_new_subscription():
    fake = FakeServer()
    fake.script([])
    asgi = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            await thread._reconcile_stream({"channels": ["lifecycle", "values"]})
            # New subscription is a subset — existing filter covers it.
            await thread._reconcile_stream({"channels": ["values"]})
    # No rotation: only one SSE request was opened.
    assert len(fake.stream_request_bodies) == 1
