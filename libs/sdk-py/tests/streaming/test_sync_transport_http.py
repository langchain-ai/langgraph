"""Sync HTTP/SSE transport tests."""

from __future__ import annotations

import httpx
import pytest

from langgraph_sdk.stream.transport.sync_http import SyncProtocolSseTransport
from streaming._events import values_event
from streaming._sync_fake_server import SyncFakeServer


def test_sync_transport_sends_command():
    fake = SyncFakeServer()
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        transport = SyncProtocolSseTransport(client=raw, thread_id="t-1")
        result = transport.send_command(
            {"id": 1, "method": "run.start", "params": {"assistant_id": "agent"}}
        )

    assert result == {"type": "success", "id": 1, "result": {"run_id": "run-1"}}
    assert fake.received_commands[0]["method"] == "run.start"


def test_sync_transport_streams_events():
    fake = SyncFakeServer()
    fake.script([values_event(seq=1, counter=1)])
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        transport = SyncProtocolSseTransport(client=raw, thread_id="t-1")
        handle = transport.open_event_stream({"channels": ["values"]})
        events = list(handle.events)

    assert events == [values_event(seq=1, counter=1)]
    assert fake.stream_request_bodies == [{"channels": ["values"]}]


def test_sync_open_event_stream_records_post_ready_error():
    fake = SyncFakeServer()
    fake.script([values_event(seq=1)], fail_after=1)
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        sse = SyncProtocolSseTransport(client=raw, thread_id="t-1")
        handle = sse.open_event_stream({"channels": ["values"]})
        with pytest.raises(httpx.ReadError, match="scripted sync stream failure"):
            list(handle.events)
        err = handle.error()
        handle.close()

    assert isinstance(err, httpx.ReadError)
