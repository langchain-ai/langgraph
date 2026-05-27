"""Sync v3 streaming projection tests."""

from __future__ import annotations

import httpx

from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.threads import SyncThreadsClient
from streaming._events import lifecycle_completed_event
from streaming._sync_fake_server import SyncFakeServer


def test_sync_values_first_yield_is_rest_state_and_output_returns_final_state():
    fake = SyncFakeServer()
    fake.script([lifecycle_completed_event(seq=1)])
    fake.set_state({"answer": 42})
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            first = next(iter(thread.values))
            output = thread.output

    assert first == {"answer": 42}
    assert output == {"answer": 42}
