"""Unit tests for the per-channel Decoders.

Each test drives a single decoder with synthetic events from `_events` and
asserts the items the decoder yields.
"""

from __future__ import annotations

from typing import Any

from langgraph_sdk.stream.decoders import (
    MessagesDecoder,
    ToolCallsDecoder,
    ValuesDecoder,
)
from streaming._events import (
    lifecycle_completed_event,
    message_error_event,
    message_finish_event,
    message_start_event,
    message_text_delta_event,
    tool_error_event,
    tool_finished_event,
    tool_output_delta_event,
    tool_started_event,
    values_event,
)


def test_values_decoder_yields_params_data():
    decoder = ValuesDecoder()
    assert list(decoder.feed(values_event(seq=1, x=1))) == [{"x": 1}]
    assert list(decoder.feed(values_event(seq=2, x=2, y=3))) == [{"x": 2, "y": 3}]


def test_values_decoder_ignores_non_values_events():
    decoder = ValuesDecoder()
    assert list(decoder.feed(lifecycle_completed_event(seq=1))) == []


class _FakeStream:
    """Stand-in for AsyncChatModelStream/ChatModelStream in decoder tests."""

    def __init__(self, *, namespace, node, message_id):
        self.namespace = namespace
        self.node = node
        self.message_id = message_id
        self.dispatched: list[dict] = []

    def dispatch(self, data):
        self.dispatched.append(data)


def _factory(*, namespace, node, message_id):
    return _FakeStream(namespace=namespace, node=node, message_id=message_id)


def test_messages_decoder_yields_stream_on_message_start():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    streams = list(
        decoder.feed(message_start_event(seq=1, message_id="m-1", node="agent"))
    )
    assert len(streams) == 1
    assert streams[0].message_id == "m-1"
    assert streams[0].node == "agent"
    # The start event is dispatched into the stream too (matches stream.py:432).
    assert (
        streams[0].dispatched and streams[0].dispatched[0]["event"] == "message-start"
    )


def test_messages_decoder_dispatches_delta_to_active_stream():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [stream] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    delta = message_text_delta_event(seq=2, message_id="m-1", text="hi")
    assert list(decoder.feed(delta)) == []
    assert stream.dispatched[-1]["event"] == "content-block-delta"


def test_messages_decoder_finish_retires_stream():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [stream] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    list(decoder.feed(message_finish_event(seq=2, message_id="m-1")))
    assert stream.dispatched[-1]["event"] == "message-finish"
    assert all(s is not stream for s in decoder._active.values())
    [again] = list(decoder.feed(message_start_event(seq=3, message_id="m-1")))
    assert again is not stream


def test_messages_decoder_error_retires_stream():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [stream] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    list(decoder.feed(message_error_event(seq=2, message_id="m-1", message="boom")))
    assert stream.dispatched[-1]["event"] == "error"
    assert all(s is not stream for s in decoder._active.values())


def test_messages_decoder_single_fallback_routes_idless_events():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [stream] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    list(decoder.feed(message_text_delta_event(seq=2, text="x")))  # no message_id
    assert stream.dispatched[-1]["event"] == "content-block-delta"


def test_messages_decoder_drops_idful_events_for_unknown_stream():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [stream] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    list(decoder.feed(message_text_delta_event(seq=2, message_id="ghost", text="x")))
    assert all(d["event"] != "content-block-delta" for d in stream.dispatched)


def test_messages_decoder_ignores_other_namespaces():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    assert (
        list(
            decoder.feed(
                message_start_event(seq=1, namespace=["child"], message_id="m-1")
            )
        )
        == []
    )


def test_messages_decoder_drops_idless_delta_when_multiple_active():
    decoder = MessagesDecoder(namespace=[], stream_factory=_factory)
    [a] = list(decoder.feed(message_start_event(seq=1, message_id="m-1")))
    [b] = list(decoder.feed(message_start_event(seq=2, message_id="m-2")))
    # id-less delta is ambiguous with two active streams -> dropped, routed to neither
    list(decoder.feed(message_text_delta_event(seq=3, text="x")))
    assert all(d["event"] != "content-block-delta" for d in a.dispatched)
    assert all(d["event"] != "content-block-delta" for d in b.dispatched)


class _FakeToolHandle:
    def __init__(self, *, tool_call_id, name, input, namespace):
        self.tool_call_id = tool_call_id
        self.name = name
        self.input = input
        self.namespace = namespace
        self.deltas: list[str] = []
        self.finished_output: Any = None
        self.finished = False
        self.error: BaseException | None = None

    def _push_delta(self, delta):
        self.deltas.append(delta)

    def _finish(self, output):
        self.finished = True
        self.finished_output = output

    def _fail(self, exc):
        self.error = exc


def _tool_factory(*, tool_call_id, name, input, namespace):
    return _FakeToolHandle(
        tool_call_id=tool_call_id, name=name, input=input, namespace=namespace
    )


def test_tool_calls_decoder_yields_handle_on_start():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [handle] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    assert handle.tool_call_id == "tc-1"
    assert handle.name == "search"


def test_tool_calls_decoder_routes_delta_finish_and_error():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    list(decoder.feed(tool_output_delta_event(seq=2, tool_call_id="tc-1", delta="x")))
    assert h.deltas == ["x"]
    list(
        decoder.feed(
            tool_finished_event(seq=3, tool_call_id="tc-1", output={"ok": True})
        )
    )
    assert h.finished and h.finished_output == {"ok": True}

    decoder2 = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h2] = list(
        decoder2.feed(
            tool_started_event(seq=1, tool_call_id="tc-2", tool_name="search")
        )
    )
    list(decoder2.feed(tool_error_event(seq=2, tool_call_id="tc-2", message="boom")))
    assert isinstance(h2.error, RuntimeError) and "boom" in str(h2.error)


def test_tool_calls_decoder_drops_events_for_unknown_id():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    assert (
        list(
            decoder.feed(
                tool_output_delta_event(seq=1, tool_call_id="ghost", delta="x")
            )
        )
        == []
    )


def test_tool_calls_decoder_finish_and_error_retire_handle():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    list(decoder.feed(tool_finished_event(seq=2, tool_call_id="tc-1")))
    # retired: a late delta for the same id is now dropped
    list(
        decoder.feed(tool_output_delta_event(seq=3, tool_call_id="tc-1", delta="late"))
    )
    assert h.deltas == []


def test_tool_calls_decoder_ignores_other_namespaces():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    assert (
        list(
            decoder.feed(
                tool_started_event(seq=1, namespace=["child"], tool_call_id="tc-1")
            )
        )
        == []
    )


def test_tool_calls_decoder_error_retires_handle():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    list(decoder.feed(tool_error_event(seq=2, tool_call_id="tc-1", message="boom")))
    # retired: a late delta for the same id is now dropped
    list(
        decoder.feed(tool_output_delta_event(seq=3, tool_call_id="tc-1", delta="late"))
    )
    assert h.deltas == []


def test_tool_calls_decoder_skips_non_str_tool_call_id():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    # Build a tools event whose data.tool_call_id is not a string.
    bad = tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search")
    bad["params"]["data"]["tool_call_id"] = 123
    assert list(decoder.feed(bad)) == []


def test_tool_calls_decoder_skips_non_str_delta():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    evt = tool_output_delta_event(seq=2, tool_call_id="tc-1", delta="x")
    evt["params"]["data"]["delta"] = 123  # non-str delta must be ignored
    list(decoder.feed(evt))
    assert h.deltas == []


def test_tool_calls_decoder_defaults_missing_tool_name_to_empty():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    evt = tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search")
    del evt["params"]["data"]["tool_name"]  # absent tool_name -> handle.name == ""
    [h] = list(decoder.feed(evt))
    assert h.name == ""


def test_tool_calls_decoder_error_message_defaults_when_blank():
    decoder = ToolCallsDecoder(namespace=[], handle_factory=_tool_factory)
    [h] = list(
        decoder.feed(tool_started_event(seq=1, tool_call_id="tc-1", tool_name="search"))
    )
    evt = tool_error_event(seq=2, tool_call_id="tc-1")
    evt["params"]["data"]["message"] = ""  # blank -> default message
    list(decoder.feed(evt))
    assert str(h.error) == "Tool call errored"
