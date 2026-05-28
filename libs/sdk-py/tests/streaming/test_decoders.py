"""Unit tests for the per-channel Decoders.

Each test drives a single decoder with synthetic events from `_events` and
asserts the items the decoder yields.
"""

from __future__ import annotations

from langgraph_sdk.stream.decoders import ValuesDecoder
from streaming._events import lifecycle_completed_event, values_event


def test_values_decoder_yields_params_data():
    decoder = ValuesDecoder()
    assert list(decoder.feed(values_event(seq=1, x=1))) == [{"x": 1}]
    assert list(decoder.feed(values_event(seq=2, x=2, y=3))) == [{"x": 2, "y": 3}]


def test_values_decoder_ignores_non_values_events():
    decoder = ValuesDecoder()
    assert list(decoder.feed(lifecycle_completed_event(seq=1))) == []
