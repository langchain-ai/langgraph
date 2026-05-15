"""Sync v3 streaming projection tests."""

from __future__ import annotations

import httpx
from langchain_core.language_models.chat_model_stream import ChatModelStream

from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.threads import SyncThreadsClient
from streaming._events import (
    lifecycle_completed_event,
    lifecycle_started_event,
    message_finish_event,
    message_start_event,
    message_text_delta_event,
    message_text_finish_event,
    tool_finished_event,
    tool_output_delta_event,
    tool_started_event,
)
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


def test_sync_messages_yield_chat_model_stream():
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-1", run_id="run-1"),
            message_text_delta_event(seq=2, text="hi"),
            message_text_finish_event(seq=3, text="hi"),
            message_finish_event(seq=4),
            lifecycle_completed_event(seq=5),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            messages = list(thread.messages)

    assert len(messages) == 1
    assert isinstance(messages[0], ChatModelStream)
    assert messages[0].message_id == "msg-1"
    assert str(messages[0].text) == "hi"
    assert messages[0].output.id == "msg-1"


def test_sync_tool_calls_yield_handle_deltas_and_output():
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1", tool_name="search"),
            tool_output_delta_event(seq=2, tool_call_id="call-1", delta="a"),
            tool_output_delta_event(seq=3, tool_call_id="call-1", delta="b"),
            tool_finished_event(seq=4, tool_call_id="call-1", output={"ok": True}),
            lifecycle_completed_event(seq=5),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            calls = list(thread.tool_calls)

    assert [call.tool_call_id for call in calls] == ["call-1"]
    assert list(calls[0].deltas) == ["a", "b"]
    assert calls[0].output == {"ok": True}
