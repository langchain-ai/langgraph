from typing import Any

import pytest

from langgraph.stream._convert import convert_to_protocol_event
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.stream_channel import StreamChannel


def _event(mode: str, data: Any, ns: list[str] | None = None) -> ProtocolEvent:
    ev = convert_to_protocol_event(tuple(ns or []), mode, data)
    assert ev is not None
    return ev


class _MockTransformer:
    def __init__(self, *, suppress: bool = False):
        self.calls: list[ProtocolEvent] = []
        self._suppress = suppress

    def init(self) -> Any:
        return None

    def process(self, event: ProtocolEvent) -> bool:
        self.calls.append(event)
        return not self._suppress

    def finalize(self) -> None:
        pass

    def fail(self, err: BaseException) -> None:
        pass


@pytest.mark.anyio
async def test_events_through_reducer_pipeline():
    reducer = _MockTransformer()
    mux = StreamMux(transformers=[reducer])
    event = _event("values", {"key": "val"})
    mux.push(event)
    assert len(reducer.calls) == 1
    assert reducer.calls[0] is event


@pytest.mark.anyio
async def test_reducer_suppresses_event():
    reducer = _MockTransformer(suppress=True)
    mux = StreamMux(transformers=[reducer])
    mux.push(_event("values", {"x": 1}))
    mux.close()
    assert len(reducer.calls) == 1
    assert len(mux.event_log) == 0


@pytest.mark.anyio
async def test_namespace_discovery():
    mux = StreamMux()
    mux.push(_event("values", {"a": 1}, ns=["child:0"]))
    assert "child:0" in mux._discovered_ns


@pytest.mark.anyio
async def test_top_level_ns_only():
    mux = StreamMux()
    mux.push(_event("values", {"a": 1}, ns=["agent:0", "tools:1"]))
    assert "agent:0" in mux._discovered_ns
    assert "tools:1" not in mux._discovered_ns


@pytest.mark.anyio
async def test_subscribe_events_filter():
    mux = StreamMux()
    mux.push(_event("values", {"a": 1}, ns=["child:0"]))
    mux.push(_event("values", {"b": 2}, ns=["other:1"]))
    mux.push(_event("values", {"c": 3}, ns=["child:0"]))
    mux.close()

    collected = []
    async for ev in mux.subscribe_events(["child:0"]):
        collected.append(ev)
    assert len(collected) == 2
    assert collected[0]["params"]["data"] == {"a": 1}
    assert collected[1]["params"]["data"] == {"c": 3}


@pytest.mark.anyio
async def test_close_resolves_output():
    mux = StreamMux()
    fut = mux.get_output_future()
    mux.push(_event("values", {"v": 1}))
    mux.push(_event("values", {"v": 2}))
    mux.close()
    result = await fut
    assert result == {"v": 2}


@pytest.mark.anyio
async def test_fail_rejects_output():
    mux = StreamMux()
    fut = mux.get_output_future()
    mux.fail(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        await fut


@pytest.mark.anyio
async def test_latest_values_tracked():
    mux = StreamMux()
    mux.push(_event("values", {"v": 1}, ns=["child:0"]))
    mux.push(_event("values", {"v": 2}, ns=["child:0"]))
    assert mux.get_latest_values(["child:0"]) == {"v": 2}


@pytest.mark.anyio
async def test_interrupt_tracking():
    """StreamMux should track __interrupt__ payloads in values events."""

    class _FakeInterrupt:
        def __init__(self, id: str, payload: Any):
            self.id = id
            self.payload = payload

    mux = StreamMux()
    interrupt_obj = _FakeInterrupt("int-1", "what do you want?")
    mux.push(
        _event(
            "values",
            {"__interrupt__": [interrupt_obj]},
        )
    )
    assert mux.interrupted is True
    assert len(mux.interrupts) == 1
    assert mux.interrupts[0]["interrupt_id"] == "int-1"
    assert mux.interrupts[0]["payload"] is interrupt_obj


@pytest.mark.anyio
async def test_no_interrupt_by_default():
    mux = StreamMux()
    mux.push(_event("values", {"x": 1}))
    mux.close()
    assert mux.interrupted is False
    assert mux.interrupts == []


@pytest.mark.anyio
async def test_push_after_close_ignored():
    mux = StreamMux()
    mux.push(_event("values", {"a": 1}))
    mux.close()
    mux.push(_event("values", {"b": 2}))
    assert len(mux.event_log) == 1


@pytest.mark.anyio
async def test_fail_rejects_all_futures():
    mux = StreamMux()
    fut1 = mux.get_output_future([])
    fut2 = mux.get_output_future(["child:0"])
    mux.fail(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        await fut1
    with pytest.raises(ValueError, match="boom"):
        await fut2


@pytest.mark.anyio
async def test_channel_events_bypass_transformer_pipeline():
    """Events emitted via ``StreamChannel.push()`` are appended directly
    to the event log, bypassing the transformer pipeline.  This matches
    the JS implementation and avoids re-entrancy bugs.
    """
    mock = _MockTransformer()
    mux = StreamMux(transformers=[mock])

    channel: StreamChannel[str] = StreamChannel("my_channel")
    mux.wire_channels({"ch": channel})

    # Regular push — transformer sees it
    mux.push(_event("values", {"a": 1}))
    assert len(mock.calls) == 1

    # Channel push — bypasses transformers, goes straight to event log
    channel.push("hello from channel")

    assert len(mock.calls) == 1, (
        f"Transformer saw {len(mock.calls)} events (expected 1).  "
        "Channel events should bypass the transformer pipeline."
    )

    # But the event IS in the log
    mux.close()
    events = []
    async for ev in mux.subscribe_events():
        events.append(ev)
    assert len(events) == 2
    assert events[1]["method"] == "my_channel"
    assert events[1]["params"]["data"] == "hello from channel"


@pytest.mark.anyio
async def test_event_log_has_monotonic_seq_numbers():
    """All events in the event log should have strictly monotonically
    increasing seq numbers so consumers can reason about ordering.

    Events from ``mux.push()`` carry seq numbers assigned by the pump
    while channel-emitted events use a separate counter
    (``_next_emit_seq``).  When interleaved, seq numbers can duplicate.
    """
    mux = StreamMux()
    channel: StreamChannel[str] = StreamChannel("test_ch")
    mux.wire_channels({"ch": channel})

    mux.push(_event("values", {"a": 1}))  # log seq: 0
    channel.push("from_channel")  # log seq: 0 (from _next_emit_seq)
    mux.push(_event("values", {"b": 2}))  # log seq: 1
    mux.close()

    seqs: list[int] = []
    async for event in mux.subscribe_events():
        seqs.append(event["seq"])

    assert len(seqs) == 3, f"Expected 3 events but got {len(seqs)}"

    for i in range(1, len(seqs)):
        assert seqs[i] > seqs[i - 1], (
            f"Seq numbers not strictly monotonic: {seqs}.  "
            f"seq[{i}]={seqs[i]} <= seq[{i - 1}]={seqs[i - 1]}.  "
            "Channel events use a separate counter from push() events."
        )


@pytest.mark.anyio
async def test_channel_push_during_process_preserves_namespace():
    """When two transformers both call channel.push() during the same
    outer mux.push(), the second transformer's channel event should
    still carry the original event's namespace.

    Bug: the first channel.push() re-enters mux.push(), which resets
    ``_current_namespace`` to ``[]`` on exit.  The second transformer's
    channel.push() then reads the clobbered value and its event gets
    ``namespace: []`` instead of the original.
    """

    class _ChannelTransformer:
        """Pushes to its channel whenever it sees a ``values`` event."""

        def __init__(self, name: str) -> None:
            self.name = name
            self.channel: StreamChannel[str] = StreamChannel(name)

        def init(self) -> Any:
            return {self.name: self.channel}

        def process(self, event: ProtocolEvent) -> bool:
            if event["method"] == "values":
                self.channel.push(f"from_{self.name}")
            return True

        def finalize(self) -> None:
            pass

        def fail(self, err: BaseException) -> None:
            pass

    t1 = _ChannelTransformer("first")
    t2 = _ChannelTransformer("second")
    mux = StreamMux(transformers=[t1, t2])
    mux.wire_channels({"first": t1.channel})
    mux.wire_channels({"second": t2.channel})

    # Push a values event with a non-root namespace
    mux.push(_event("values", {"x": 1}, ns=["agent:0"]))
    mux.close()

    # Collect channel events emitted by each transformer
    channel_events: list[ProtocolEvent] = []
    async for ev in mux.subscribe_events():
        if ev["method"] in ("first", "second"):
            channel_events.append(ev)

    assert len(channel_events) == 2, (
        f"Expected 2 channel events but got {len(channel_events)}"
    )

    for ev in channel_events:
        assert ev["params"]["namespace"] == ["agent:0"], (
            f"Channel event for method={ev['method']!r} has "
            f"namespace={ev['params']['namespace']!r}, expected ['agent:0'].  "
            "The nested mux.push() from the first channel.push() clobbered "
            "_current_namespace before the second transformer ran."
        )
