"""Sync shared stream controller tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.threads import SyncThreadsClient
from langgraph_sdk.stream.sync_controller import SyncStreamController
from langgraph_sdk.stream.transport.sync_http import (
    SyncEventStreamHandle,
    SyncProtocolSseTransport,
)
from streaming._events import thread_stream_failed_frame, values_event
from streaming._sync_fake_server import SyncFakeServer, SyncStreamScript


def test_sync_controller_fans_out_to_subscription():
    fake = SyncFakeServer()
    fake.script([values_event(seq=1, counter=1)])
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        transport = SyncProtocolSseTransport(client=raw, thread_id="t-1")
        controller = SyncStreamController(transport)
        sub = controller.register_subscription({"channels": ["values"]})
        controller.reconcile_stream({"channels": ["values"]})
        controller.ensure_fanout_running()

        assert sub.queue.get(timeout=1) == values_event(seq=1, counter=1)
        assert sub.queue.get(timeout=1) is None
        controller.close()


@pytest.mark.parametrize("raise_on_close", [False, True])
def test_sync_shared_stream_terminal_error_prevents_1011_reconnect(raise_on_close):
    transport = MagicMock(spec=SyncProtocolSseTransport)
    transport.thread_id = "t-1"
    controller = SyncStreamController(transport)
    controller._shared_stream_filter = {"channels": ["values"]}
    sub = controller.register_subscription({"channels": ["values"]})
    error = ConnectionClosedError(Close(1011, "internal error"), None)

    # A later attempt on the same controller must still retry a standalone 1011.
    for seq, terminal in enumerate([True, False], start=1):
        event = values_event(seq=seq)

        def events(terminal=terminal, event=event):
            if terminal:
                yield thread_stream_failed_frame()
            yield event
            if raise_on_close:
                raise error

        controller._shared_stream = SyncEventStreamHandle(
            events=events(), error=lambda: error, close=lambda: None
        )
        transport.open_event_stream.reset_mock()
        transport.open_event_stream.return_value = SyncEventStreamHandle(
            events=iter([]), error=lambda: None, close=lambda: None
        )

        controller._fanout()

        assert sub.queue.get_nowait() == event
        assert sub.queue.get_nowait() is None
        assert sub.queue.empty()
        assert transport.open_event_stream.call_count == (0 if terminal else 1)


def test_sync_send_command_applied_through_seq_seeds_shared_stream_since():
    fake = SyncFakeServer()
    fake.script_sequence([SyncStreamScript(events=[]), SyncStreamScript(events=[])])
    fake.script_command_response(
        {
            "type": "success",
            "id": None,
            "result": {"run_id": "run-1"},
            "meta": {"applied_through_seq": 17},
        }
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            assert list(thread.subscribe(["values"])) == []

    # The shared SSE filter is a union of subscription params plus
    # ``lifecycle`` (added by ``_compute_current_union`` for projection-
    # iterator termination on root-terminal). Match any request whose
    # channels include ``values``.
    values_requests = [
        b for b in fake.stream_request_bodies if "values" in (b.get("channels") or [])
    ]
    assert len(values_requests) == 1
    assert values_requests[0]["since"] == 17


def test_sync_shared_stream_reconnects_with_since_after_transport_drop():
    fake = SyncFakeServer()
    fake.script_sequence(
        [
            SyncStreamScript(
                events=[values_event(seq=1, values={"counter": 1})],
                fail_after=1,
            ),
            SyncStreamScript(events=[values_event(seq=2, values={"counter": 2})]),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        transport = SyncProtocolSseTransport(client=raw, thread_id="t-1")
        controller = SyncStreamController(transport)
        sub = controller.register_subscription({"channels": ["values"]})
        controller.reconcile_stream({"channels": ["values"]})
        controller.ensure_fanout_running()

        first = sub.queue.get(timeout=1.0)
        second = sub.queue.get(timeout=1.0)
        end = sub.queue.get(timeout=1.0)
        controller.close()
        transport.close()

    assert first is not None
    assert second is not None
    assert first["seq"] == 1
    assert second["seq"] == 2
    assert end is None
    assert fake.stream_request_bodies[1]["since"] == 1
