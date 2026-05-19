"""Sync v3 streaming projection tests."""

from __future__ import annotations

import httpx
import pytest
from langchain_core.language_models.chat_model_stream import ChatModelStream

from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.threads import SyncThreadsClient
from streaming._events import (
    lifecycle_completed_event,
    lifecycle_errored_event,
    lifecycle_started_event,
    message_error_event,
    message_finish_event,
    message_start_event,
    message_text_delta_event,
    message_text_finish_event,
    tool_error_event,
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


# ---------------------------------------------------------------------------
# Task 10.6 — messages pre-dispatch inner loop must filter by namespace
# ---------------------------------------------------------------------------


def test_sync_messages_ignores_nested_namespace_for_root_projection():
    """Root projection must not yield messages from child namespaces."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, namespace=["child:1"], message_id="nested"),
            message_text_delta_event(seq=2, namespace=["child:1"], text="nested"),
            message_finish_event(seq=3, namespace=["child:1"]),
            lifecycle_completed_event(seq=4),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            messages = list(thread.messages)

    assert messages == []


# ---------------------------------------------------------------------------
# Task 10.7 — comprehensive sync messages projection tests
# ---------------------------------------------------------------------------


def test_sync_messages_multiple_messages_are_distinct_streams():
    """Two sequential messages produce two separate ChatModelStream objects."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-1", run_id="run-1"),
            message_text_delta_event(seq=2, text="one"),
            message_text_finish_event(seq=3, text="one"),
            message_finish_event(seq=4),
            message_start_event(seq=5, message_id="msg-2", run_id="run-2"),
            message_text_delta_event(seq=6, text="two"),
            message_text_finish_event(seq=7, text="two"),
            message_finish_event(seq=8),
            lifecycle_completed_event(seq=9),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            streams = list(thread.messages)

    assert [s.message_id for s in streams] == ["msg-1", "msg-2"]
    assert [str(s.text) for s in streams] == ["one", "two"]


def test_sync_messages_error_event_fails_active_stream():
    """A messages `error` event marks the stream as failed."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            message_start_event(seq=1, message_id="msg-1"),
            message_error_event(seq=2, message="model failed"),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            streams = list(thread.messages)

    assert len(streams) == 1
    with pytest.raises(RuntimeError, match="model failed"):
        _ = streams[0].output


# ---------------------------------------------------------------------------
# Task 10.7 — comprehensive sync tool_calls projection tests
# ---------------------------------------------------------------------------


def test_sync_tool_calls_multiple_concurrent_calls_route_by_id():
    """Two interleaved tool calls each resolve to the correct output."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-a", tool_name="alpha"),
            tool_output_delta_event(seq=2, tool_call_id="call-a", delta="a1"),
            tool_finished_event(seq=3, tool_call_id="call-a", output="A"),
            tool_started_event(seq=4, tool_call_id="call-b", tool_name="beta"),
            tool_output_delta_event(seq=5, tool_call_id="call-b", delta="b1"),
            tool_finished_event(seq=6, tool_call_id="call-b", output="B"),
            lifecycle_completed_event(seq=7),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            calls = list(thread.tool_calls)

    by_id = {call.tool_call_id: call for call in calls}
    assert set(by_id) == {"call-a", "call-b"}
    assert list(by_id["call-a"].deltas) == ["a1"]
    assert list(by_id["call-b"].deltas) == ["b1"]
    assert by_id["call-a"].output == "A"
    assert by_id["call-b"].output == "B"


def test_sync_tool_calls_ignores_nested_namespace_for_root_projection():
    """Root tool_calls projection must not yield handles from child namespaces."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, namespace=["child:1"], tool_call_id="nested"),
            tool_finished_event(seq=2, namespace=["child:1"], tool_call_id="nested"),
            lifecycle_completed_event(seq=3),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            calls = list(thread.tool_calls)

    assert calls == []


def test_sync_tool_calls_error_event_fails_output():
    """A `tool-error` event fails the handle's output property."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1"),
            tool_output_delta_event(seq=2, tool_call_id="call-1", delta="before"),
            tool_error_event(seq=3, tool_call_id="call-1", message="boom"),
            lifecycle_completed_event(seq=4),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            calls = list(thread.tool_calls)

    assert len(calls) == 1
    assert list(calls[0].deltas) == ["before"]
    with pytest.raises(RuntimeError, match="boom"):
        _ = calls[0].output


# ---------------------------------------------------------------------------
# Task 10.10 — run-error propagates to active tool-call handle
# ---------------------------------------------------------------------------


def test_sync_tool_calls_run_error_fails_active_handle():
    """A lifecycle errored event propagates its message to the active handle."""
    fake = SyncFakeServer()
    fake.script(
        [
            lifecycle_started_event(seq=0),
            tool_started_event(seq=1, tool_call_id="call-1"),
            tool_output_delta_event(seq=2, tool_call_id="call-1", delta="partial"),
            lifecycle_errored_event(seq=3, error="run failed"),
        ]
    )
    with httpx.Client(transport=fake.transport, base_url="http://test") as raw:
        threads = SyncThreadsClient(SyncHttpClient(raw))
        with threads.stream(thread_id="t-1", assistant_id="agent") as thread:
            thread.run.start(input={})
            calls = list(thread.tool_calls)

    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="Run errored: run failed"):
        _ = calls[0].output
