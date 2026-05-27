from __future__ import annotations

import httpx

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.threads import ThreadsClient
from streaming._events import custom_event, lifecycle_completed_event
from streaming._fake_server import FakeServer


async def test_extension_projection_yields_matching_custom_payloads():
    fake = FakeServer()
    fake.script(
        [
            custom_event(seq=1, name="progress", step=1),
            custom_event(seq=2, name="metrics", tokens=12),
            custom_event(seq=3, name="progress", step=2),
            lifecycle_completed_event(seq=4),
        ]
    )
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            payloads = [payload async for payload in thread.extensions["progress"]]

    assert payloads == [
        {"name": "progress", "step": 1},
        {"name": "progress", "step": 2},
    ]
    assert any(
        "custom:progress" in body.get("channels", [])
        for body in fake.stream_request_bodies
    )


async def test_extension_projection_supports_namespace_scope_on_subgraph_handle():
    fake = FakeServer()
    fake.script(
        [
            custom_event(seq=1, name="progress", namespace=["worker:abc"], step=1),
            custom_event(seq=2, name="progress", namespace=[], step=0),
            lifecycle_completed_event(seq=3),
        ]
    )
    transport = httpx.ASGITransport(app=fake.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw:
        from langgraph_sdk._async.stream import ScopedStreamHandle

        threads = ThreadsClient(HttpClient(raw))
        async with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            await thread.run.start(input={})
            handle = ScopedStreamHandle(
                thread=thread,
                path=("worker:abc",),
                graph_name="worker",
                trigger_call_id=None,
            )
            payloads = [payload async for payload in handle.extensions["progress"]]

    assert payloads == [{"name": "progress", "step": 1}]
