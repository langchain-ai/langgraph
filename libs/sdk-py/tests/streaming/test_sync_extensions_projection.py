from __future__ import annotations

import httpx

from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.threads import SyncThreadsClient
from streaming._events import custom_event, lifecycle_completed_event
from streaming._sync_fake_server import SyncFakeServer


def test_sync_extension_projection_yields_matching_custom_payloads():
    fake = SyncFakeServer()
    fake.script(
        [
            custom_event(seq=1, name="progress", step=1),
            custom_event(seq=2, name="metrics", tokens=12),
            custom_event(seq=3, name="progress", step=2),
            lifecycle_completed_event(seq=4),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            payloads = list(thread.extensions["progress"])

    assert payloads == [
        {"name": "progress", "step": 1},
        {"name": "progress", "step": 2},
    ]
    assert "custom:progress" in fake.stream_request_bodies[-1]["channels"]


def test_sync_extension_projection_supports_namespace_scope_on_subgraph_handle():
    fake = SyncFakeServer()
    fake.script(
        [
            custom_event(seq=1, name="progress", namespace=["worker:abc"], step=1),
            custom_event(seq=2, name="progress", namespace=[], step=0),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        from langgraph_sdk._sync.stream import SyncScopedStreamHandle

        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            handle = SyncScopedStreamHandle(
                thread=thread,
                path=("worker:abc",),
                graph_name="worker",
                trigger_call_id=None,
            )
            payloads = list(handle.extensions["progress"])

    assert payloads == [{"name": "progress", "step": 1}]
