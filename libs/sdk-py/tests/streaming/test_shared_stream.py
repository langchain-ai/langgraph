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
            # Internal helper to drive the shared stream until the public
            # `subscribe()` API lands in Task 5.
            handle = thread._ensure_shared_stream({"channels": ["lifecycle", "values"]})
            received = [e async for e in handle]
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
            handle = thread._ensure_shared_stream({"channels": ["lifecycle", "values"]})
            received = [e async for e in handle]
    seqs = [e["seq"] for e in received]
    assert seqs == [0, 1]  # the duplicate seq=0 was deduped via event_id
