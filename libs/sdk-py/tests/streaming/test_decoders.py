"""Unit tests for the per-channel Decoders.

Each test drives a single decoder with synthetic events from `_events` and
asserts the items the decoder yields.
"""

from __future__ import annotations

from langgraph_sdk.stream.decoders import MessagesDecoder, ValuesDecoder
from streaming._events import (
    lifecycle_completed_event,
    message_error_event,
    message_finish_event,
    message_start_event,
    message_text_delta_event,
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
