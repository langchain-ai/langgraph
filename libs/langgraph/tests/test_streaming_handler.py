"""Tests for the StreamingHandler and its supporting infrastructure."""

from __future__ import annotations

import asyncio
import operator
import sys
import time
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.stream import (
    AsyncEventLog,
    EventLog,
    StreamChannel,
    StreamingHandler,
    StreamTransformer,
)
from langgraph.stream._convert import convert_to_protocol_event
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.transformers import MessagesTransformer, ValuesTransformer
from langgraph.types import StreamWriter, interrupt

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)

TS = int(time.time() * 1000)


def _event(
    method: str,
    data: Any = None,
    *,
    namespace: list[str] | None = None,
    interrupts: tuple[Any, ...] | None = None,
) -> ProtocolEvent:
    """Build a test ProtocolEvent with sensible defaults."""
    params: dict[str, Any] = {
        "namespace": namespace or [],
        "timestamp": TS,
        "data": data if data is not None else {},
    }
    if interrupts is not None:
        params["interrupts"] = interrupts
    return {"type": "event", "method": method, "params": params}


# ---------------------------------------------------------------------------
# Shared state and graph builders
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def _build_simple_graph():
    """Two-node graph: node_a appends 'a', node_b appends 'b'."""

    def node_a(state: SimpleState) -> dict:
        return {"value": state["value"] + "A", "items": ["a"]}

    def node_b(state: SimpleState) -> dict:
        return {"value": state["value"] + "B", "items": ["b"]}

    builder = StateGraph(SimpleState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    return builder.compile()


def _build_interrupt_graph():
    """Graph that interrupts before node_b."""

    def node_a(state: SimpleState) -> dict:
        return {"value": state["value"] + "A", "items": ["a"]}

    def node_b(state: SimpleState) -> dict:
        interrupt("need approval")
        return {"value": state["value"] + "B", "items": ["b"]}

    builder = StateGraph(SimpleState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    return builder.compile(checkpointer=InMemorySaver())


def _build_error_graph():
    """Graph where node_b raises."""

    def node_a(state: SimpleState) -> dict:
        return {"value": state["value"] + "A", "items": ["a"]}

    def node_b(state: SimpleState) -> dict:
        raise ValueError("boom")

    builder = StateGraph(SimpleState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    return builder.compile()


def _build_custom_stream_graph():
    """Graph that emits custom stream events."""

    def node_a(state: SimpleState, *, writer: StreamWriter) -> dict:
        writer({"step": "start"})
        writer({"step": "end"})
        return {"value": state["value"] + "A", "items": ["a"]}

    builder = StateGraph(SimpleState)
    builder.add_node("node_a", node_a)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# EventLog unit tests
# ---------------------------------------------------------------------------


class TestEventLog:
    def test_sync_iteration(self) -> None:
        log: EventLog[int] = EventLog()
        log.push(1)
        log.push(2)
        log.push(3)
        log.close()
        assert list(log) == [1, 2, 3]

    def test_multi_cursor(self) -> None:
        log: EventLog[str] = EventLog()
        log.push("a")
        log.push("b")
        log.close()
        # Two independent cursors see all items.
        assert list(log) == ["a", "b"]
        assert list(log) == ["a", "b"]

    def test_fail_propagation(self) -> None:
        log: EventLog[int] = EventLog()
        log.push(1)
        log.fail(ValueError("test error"))
        with pytest.raises(ValueError, match="test error"):
            list(log)

    @pytest.mark.anyio
    async def test_async_iteration(self) -> None:
        log: AsyncEventLog[int] = AsyncEventLog()

        async def producer():
            for i in range(3):
                log.push(i)
            log.close()

        asyncio.get_event_loop().call_soon(lambda: asyncio.ensure_future(producer()))
        items = [item async for item in log]
        assert items == [0, 1, 2]

    @pytest.mark.anyio
    async def test_async_multi_cursor(self) -> None:
        log: AsyncEventLog[str] = AsyncEventLog()
        log.push("x")
        log.push("y")
        log.close()
        items1 = [item async for item in log]
        items2 = [item async for item in log]
        assert items1 == ["x", "y"]
        assert items2 == ["x", "y"]

    @pytest.mark.anyio
    async def test_async_fail(self) -> None:
        log: AsyncEventLog[int] = AsyncEventLog()
        log.push(1)
        log.fail(RuntimeError("async error"))
        with pytest.raises(RuntimeError, match="async error"):
            async for _ in log:
                pass

    def test_sync_cursor_yields_items_before_error(self) -> None:
        """Sync cursor should yield all buffered items before raising."""
        log: EventLog[int] = EventLog()
        log.push(1)
        log.push(2)
        log.push(3)
        log.fail(ValueError("late error"))
        items: list[int] = []
        with pytest.raises(ValueError, match="late error"):
            for item in log:
                items.append(item)
        assert items == [1, 2, 3]

    @pytest.mark.anyio
    async def test_async_cursor_yields_items_before_error(self) -> None:
        """Async cursor should yield all buffered items before raising."""
        log: AsyncEventLog[int] = AsyncEventLog()
        log.push(1)
        log.push(2)
        log.push(3)
        log.fail(ValueError("late error"))
        items: list[int] = []
        with pytest.raises(ValueError, match="late error"):
            async for item in log:
                items.append(item)
        assert items == [1, 2, 3]

    def test_push_after_close_raises(self) -> None:
        """Push after close should raise RuntimeError."""
        log: EventLog[int] = EventLog()
        log.push(1)
        log.close()
        with pytest.raises(RuntimeError, match="Cannot push to a closed EventLog"):
            log.push(2)

    def test_push_after_fail_raises(self) -> None:
        """Fail closes the log, so push after fail should also raise."""
        log: EventLog[int] = EventLog()
        log.fail(ValueError("err"))
        with pytest.raises(RuntimeError, match="Cannot push to a closed EventLog"):
            log.push(1)

    def test_empty_log_sync(self) -> None:
        """Iterating a closed empty log should yield nothing."""
        log: EventLog[int] = EventLog()
        log.close()
        assert list(log) == []

    @pytest.mark.anyio
    async def test_empty_log_async(self) -> None:
        """Async-iterating a closed empty log should yield nothing."""
        log: AsyncEventLog[int] = AsyncEventLog()
        log.close()
        assert [item async for item in log] == []

    def test_empty_log_fail_sync(self) -> None:
        """Failing an empty log should raise immediately with no items."""
        log: EventLog[int] = EventLog()
        log.fail(ValueError("empty fail"))
        with pytest.raises(ValueError, match="empty fail"):
            list(log)

    @pytest.mark.anyio
    async def test_empty_log_fail_async(self) -> None:
        """Failing an empty log should raise immediately with no items (async)."""
        log: AsyncEventLog[int] = AsyncEventLog()
        log.fail(ValueError("empty fail"))
        with pytest.raises(ValueError, match="empty fail"):
            async for _ in log:
                pass

    def test_sync_has_no_aiter(self) -> None:
        """EventLog (sync) should not support async iteration."""
        log: EventLog[int] = EventLog()
        log.close()
        assert not hasattr(log, "__aiter__")

    @pytest.mark.anyio
    async def test_async_has_no_iter(self) -> None:
        """AsyncEventLog should not support sync iteration."""
        log: AsyncEventLog[int] = AsyncEventLog()
        log.close()
        assert not hasattr(log, "__iter__")


# ---------------------------------------------------------------------------
# StreamChannel unit tests
# ---------------------------------------------------------------------------


class TestStreamChannel:
    def test_push_and_iterate(self) -> None:
        ch: StreamChannel[str] = StreamChannel("test")
        ch.push("a")
        ch.push("b")
        ch._close()
        assert list(ch) == ["a", "b"]

    def test_wire_callback(self) -> None:
        forwarded: list[str] = []
        ch: StreamChannel[str] = StreamChannel("test")
        ch._wire(lambda item: forwarded.append(item))
        ch.push("x")
        ch.push("y")
        ch._close()
        assert forwarded == ["x", "y"]
        assert list(ch) == ["x", "y"]

    def test_fail_propagation(self) -> None:
        """_fail() should propagate the error through the underlying log."""
        ch: StreamChannel[str] = StreamChannel("test")
        ch.push("a")
        ch._fail(ValueError("channel error"))
        items: list[str] = []
        with pytest.raises(ValueError, match="channel error"):
            for item in ch:
                items.append(item)
        assert items == ["a"]

    @pytest.mark.anyio
    async def test_async_iteration(self) -> None:
        """Async iteration should delegate to the inner AsyncEventLog."""
        ch: StreamChannel[str] = StreamChannel("test", is_async=True)
        ch.push("x")
        ch.push("y")
        ch._close()
        items = [item async for item in ch]
        assert items == ["x", "y"]

    def test_push_without_wire(self) -> None:
        """Push without a wire callback should still append to the log."""
        ch: StreamChannel[int] = StreamChannel("test")
        assert ch._wire_fn is None
        ch.push(42)
        ch._close()
        assert list(ch) == [42]


# ---------------------------------------------------------------------------
# StreamingHandler sync tests
# ---------------------------------------------------------------------------


class TestStreamingHandlerSync:
    def test_values_projection(self) -> None:
        graph = _build_simple_graph()
        handler = StreamingHandler(graph)
        run = handler.stream({"value": "x", "items": []})
        snapshots = list(run.values)
        # Should have at least the initial + per-node snapshots.
        assert len(snapshots) >= 1
        # Last snapshot should have both nodes' effects.
        last = snapshots[-1]
        assert "A" in last["value"]
        assert "B" in last["value"]

    def test_output(self) -> None:
        graph = _build_simple_graph()
        handler = StreamingHandler(graph)
        run = handler.stream({"value": "x", "items": []})
        output = run.output
        assert output is not None
        assert output["value"] == "xAB"
        assert output["items"] == ["a", "b"]

    def test_raw_event_iteration(self) -> None:
        graph = _build_simple_graph()
        handler = StreamingHandler(graph)
        run = handler.stream({"value": "x", "items": []})
        events = list(run)
        assert len(events) > 0
        for event in events:
            assert event["type"] == "event"
            assert "method" in event
            assert "seq" in event
            assert isinstance(event["params"]["timestamp"], int)

    def test_extensions_has_native_keys(self) -> None:
        graph = _build_simple_graph()
        handler = StreamingHandler(graph)
        run = handler.stream({"value": "x", "items": []})
        # Drain events so the run completes.
        _ = run.output
        assert "values" in run.extensions
        assert "messages" in run.extensions
        # Native keys should also be direct attributes.
        assert run.values is run.extensions["values"]
        assert run.messages is run.extensions["messages"]

    def test_custom_stream_events(self) -> None:
        graph = _build_custom_stream_graph()
        handler = StreamingHandler(graph)
        run = handler.stream({"value": "x", "items": []})
        custom_events = [e for e in run if e["method"] == "custom"]
        assert len(custom_events) == 2
        assert custom_events[0]["params"]["data"] == {"step": "start"}
        assert custom_events[1]["params"]["data"] == {"step": "end"}


class TestStreamingHandlerSyncErrors:
    def test_error_propagation_output(self) -> None:
        graph = _build_error_graph()
        handler = StreamingHandler(graph)
        run = handler.stream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            _ = run.output

    def test_error_propagation_values(self) -> None:
        graph = _build_error_graph()
        handler = StreamingHandler(graph)
        run = handler.stream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            list(run.values)

    def test_error_propagation_raw_events(self) -> None:
        graph = _build_error_graph()
        handler = StreamingHandler(graph)
        run = handler.stream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            list(run)


class TestStreamingHandlerSyncInterrupt:
    def test_interrupted(self) -> None:
        graph = _build_interrupt_graph()
        handler = StreamingHandler(graph)
        run = handler.stream(
            {"value": "x", "items": []},
            {"configurable": {"thread_id": "t1"}},
        )
        _ = run.output
        assert run.interrupted is True
        assert len(run.interrupts) > 0


# ---------------------------------------------------------------------------
# StreamingHandler async tests
# ---------------------------------------------------------------------------


class TestStreamingHandlerAsync:
    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_values_projection(self) -> None:
        graph = _build_simple_graph()
        handler = StreamingHandler(graph)
        run = await handler.astream({"value": "x", "items": []})
        snapshots = [s async for s in run.values]
        assert len(snapshots) >= 1
        last = snapshots[-1]
        assert "A" in last["value"]
        assert "B" in last["value"]

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_output(self) -> None:
        graph = _build_simple_graph()
        handler = StreamingHandler(graph)
        run = await handler.astream({"value": "x", "items": []})
        output = await run.output
        assert output is not None
        assert output["value"] == "xAB"
        assert output["items"] == ["a", "b"]

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_raw_event_iteration(self) -> None:
        graph = _build_simple_graph()
        handler = StreamingHandler(graph)
        run = await handler.astream({"value": "x", "items": []})
        events = [e async for e in run]
        assert len(events) > 0
        for event in events:
            assert event["type"] == "event"

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_extensions_has_native_keys(self) -> None:
        graph = _build_simple_graph()
        handler = StreamingHandler(graph)
        run = await handler.astream({"value": "x", "items": []})
        _ = await run.output
        assert "values" in run.extensions
        assert "messages" in run.extensions
        assert run.values is run.extensions["values"]
        assert run.messages is run.extensions["messages"]


class TestStreamingHandlerAsyncErrors:
    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_error_propagation_output(self) -> None:
        graph = _build_error_graph()
        handler = StreamingHandler(graph)
        run = await handler.astream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            await run.output

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_error_propagation_values(self) -> None:
        graph = _build_error_graph()
        handler = StreamingHandler(graph)
        run = await handler.astream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            async for _ in run.values:
                pass

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_error_propagation_raw_events(self) -> None:
        graph = _build_error_graph()
        handler = StreamingHandler(graph)
        run = await handler.astream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            async for _ in run:
                pass


class TestStreamingHandlerAsyncInterrupt:
    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_interrupted(self) -> None:
        graph = _build_interrupt_graph()
        handler = StreamingHandler(graph)
        run = await handler.astream(
            {"value": "x", "items": []},
            {"configurable": {"thread_id": "t2"}},
        )
        _ = await run.output
        assert run.interrupted is True
        assert len(run.interrupts) > 0


class TestStreamingHandlerAsyncCustom:
    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_custom_stream_events(self) -> None:
        graph = _build_custom_stream_graph()
        handler = StreamingHandler(graph)
        run = await handler.astream({"value": "x", "items": []})
        events = [e async for e in run]
        custom_events = [e for e in events if e["method"] == "custom"]
        assert len(custom_events) == 2
        assert custom_events[0]["params"]["data"] == {"step": "start"}
        assert custom_events[1]["params"]["data"] == {"step": "end"}


# ---------------------------------------------------------------------------
# Custom transformer tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# convert_to_protocol_event unit tests
# ---------------------------------------------------------------------------


class TestConvertToProtocolEvent:
    def test_basic_conversion(self) -> None:
        before = int(time.time() * 1000)
        part = {"type": "values", "ns": ("sub", "graph"), "data": {"key": "val"}}
        event = convert_to_protocol_event(part)
        after = int(time.time() * 1000)
        assert event["type"] == "event"
        assert event["method"] == "values"
        assert event["params"]["namespace"] == ["sub", "graph"]
        assert event["params"]["data"] == {"key": "val"}
        assert "interrupts" not in event["params"]
        assert before <= event["params"]["timestamp"] <= after

    def test_conversion_with_interrupts(self) -> None:
        part = {
            "type": "values",
            "ns": (),
            "data": {"k": 1},
            "interrupts": ({"value": "pause"},),
        }
        event = convert_to_protocol_event(part)
        assert event["params"]["interrupts"] == ({"value": "pause"},)
        assert isinstance(event["params"]["timestamp"], int)

    def test_namespace_tuple_becomes_list(self) -> None:
        """ns tuple should be converted to a list."""
        part = {"type": "updates", "ns": ("a", "b", "c"), "data": {}}
        event = convert_to_protocol_event(part)
        assert isinstance(event["params"]["namespace"], list)
        assert event["params"]["namespace"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# StreamMux unit tests
# ---------------------------------------------------------------------------


class TestStreamMux:
    def test_register_non_dict_raises(self) -> None:
        """init() returning a non-dict should raise TypeError."""

        class BadTransformer(StreamTransformer):
            def init(self) -> Any:
                return ["not", "a", "dict"]

            def process(self, event: ProtocolEvent) -> bool:
                return True

        mux = StreamMux()
        with pytest.raises(TypeError, match="must return a dict"):
            mux.register(BadTransformer())

    def test_event_suppression(self) -> None:
        """When process() returns False, the event should not appear in the main log."""

        class FilterTransformer(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                # Suppress "updates" events
                return event["method"] != "updates"

        mux = StreamMux()
        mux.register(FilterTransformer())

        mux.push(_event("values", {"a": 1}))
        mux.push(_event("updates", {"b": 2}))
        mux.push(_event("custom", {"c": 3}))
        mux.close()

        events = list(mux._events)
        methods = [e["method"] for e in events]
        assert "updates" not in methods
        assert methods == ["values", "custom"]

    def test_suppression_partial_transformers(self) -> None:
        """If any transformer returns False, the event is suppressed,
        but all transformers still see it."""

        seen_by_second: list[str] = []

        class PassTransformer(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                return True

        class RejectTransformer(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                seen_by_second.append(event["method"])
                return False

        mux = StreamMux()
        mux.register(PassTransformer())
        mux.register(RejectTransformer())

        mux.push(_event("values"))
        mux.close()

        # RejectTransformer saw the event even though it rejected it
        assert seen_by_second == ["values"]
        # But nothing in the main log
        assert list(mux._events) == []

    def test_empty_mux(self) -> None:
        """Push/close/fail on a mux with no transformers should work."""
        mux = StreamMux()
        mux.push(_event("values", {"x": 1}))
        mux.close()
        events = list(mux._events)
        assert len(events) == 1
        assert events[0]["method"] == "values"

    def test_empty_mux_fail(self) -> None:
        """Fail on an empty mux should propagate to the event log."""
        mux = StreamMux()
        mux.fail(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            list(mux._events)


# ---------------------------------------------------------------------------
# ValuesTransformer / MessagesTransformer unit tests
# ---------------------------------------------------------------------------


class TestValuesTransformer:
    def test_ignores_non_root_namespace(self) -> None:
        """Values events from subgraphs (non-empty namespace) should be ignored."""
        t = ValuesTransformer()
        t.init()

        t.process(_event("values", {"val": "root"}))
        t.process(_event("values", {"val": "sub"}, namespace=["sub"]))

        t.finalize()
        items = list(t._log)
        assert len(items) == 1
        assert items[0]["val"] == "root"

    def test_ignores_non_values_methods(self) -> None:
        """Non-values events should be passed through but not captured."""
        t = ValuesTransformer()
        t.init()

        result = t.process(_event("updates", {"x": 1}))
        assert result is True  # passed through
        t.finalize()
        assert list(t._log) == []  # but not captured

    def test_tracks_interrupts(self) -> None:
        """Interrupts should be accumulated across events."""
        t = ValuesTransformer()
        t.init()

        t.process(
            _event(
                "values",
                {"v": 1},
                interrupts=({"value": "pause1"}, {"value": "pause2"}),
            )
        )
        assert t._interrupted is True
        assert len(t._interrupts) == 2


class TestMessagesTransformer:
    def test_captures_root_messages(self) -> None:
        t = MessagesTransformer()
        t.init()

        t.process(_event("messages", ("chunk", {"meta": True})))
        t.finalize()
        items = list(t._log)
        assert len(items) == 1
        assert items[0] == ("chunk", {"meta": True})

    def test_ignores_non_root_namespace(self) -> None:
        t = MessagesTransformer()
        t.init()

        t.process(_event("messages", ("chunk", {}), namespace=["sub"]))
        t.finalize()
        assert list(t._log) == []

    def test_ignores_non_messages_methods(self) -> None:
        t = MessagesTransformer()
        t.init()

        result = t.process(_event("values", {"v": 1}))
        assert result is True
        t.finalize()
        assert list(t._log) == []

    def test_fail_propagates(self) -> None:
        t = MessagesTransformer()
        t.init()
        t.fail(ValueError("msg error"))
        with pytest.raises(ValueError, match="msg error"):
            list(t._log)


# ---------------------------------------------------------------------------
# StreamMux resilience tests
# ---------------------------------------------------------------------------


class TestStreamMuxResilience:
    """StreamMux.close() and fail() must complete cleanup even if a transformer raises."""

    def test_close_continues_after_finalize_error(self) -> None:
        """If a transformer's finalize() raises, the main event log and
        remaining transformers should still be closed/finalized."""

        class BrokenFinalizer(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                return True

            def finalize(self) -> None:
                raise RuntimeError("finalize broke")

        class GoodTransformer(StreamTransformer):
            def __init__(self) -> None:
                self.finalized = False

            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                return True

            def finalize(self) -> None:
                self.finalized = True

        mux = StreamMux()
        mux.register(BrokenFinalizer())
        good = GoodTransformer()
        mux.register(good)

        mux.push(_event("values"))

        with pytest.raises(RuntimeError, match="finalize broke"):
            mux.close()

        assert good.finalized
        assert mux._events._closed

    def test_fail_continues_after_transformer_error(self) -> None:
        """If a transformer's fail() raises, the main event log and
        remaining transformers should still be failed."""

        class BrokenFailer(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                return True

            def fail(self, err: BaseException) -> None:
                raise RuntimeError("fail handler broke")

        class GoodTransformer(StreamTransformer):
            def __init__(self) -> None:
                self.failed_with: BaseException | None = None

            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                return True

            def fail(self, err: BaseException) -> None:
                self.failed_with = err

        mux = StreamMux()
        mux.register(BrokenFailer())
        good = GoodTransformer()
        mux.register(good)

        original_error = ValueError("original")
        mux.fail(original_error)

        assert good.failed_with is original_error
        assert mux._events._error is original_error

    def test_close_still_closes_channels_after_finalize_error(self) -> None:
        """Channels should be closed even if a transformer's finalize raises."""

        class BrokenWithChannel(StreamTransformer):
            def __init__(self) -> None:
                self._channel: StreamChannel[str] = StreamChannel("ch")

            def init(self) -> dict[str, Any]:
                return {"ch": self._channel}

            def process(self, event: ProtocolEvent) -> bool:
                return True

            def finalize(self) -> None:
                raise RuntimeError("finalize broke")

        t = BrokenWithChannel()
        mux = StreamMux()
        mux.register(t)

        with pytest.raises(RuntimeError, match="finalize broke"):
            mux.close()

        assert t._channel._log._closed


class TestCustomTransformer:
    def test_extension_transformer_with_stream_channel(self) -> None:
        """User transformer with StreamChannel appears in extensions."""

        class CounterTransformer(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._channel: StreamChannel[int] = StreamChannel("counter")
                self._count = 0

            def init(self) -> dict[str, Any]:
                return {"counter": self._channel}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":
                    self._count += 1
                    self._channel.push(self._count)
                return True

        graph = _build_simple_graph()
        handler = StreamingHandler(graph)
        counter_t = CounterTransformer()
        run = handler.stream({"value": "x", "items": []}, transformers=[counter_t])
        _ = run.output
        assert "counter" in run.extensions
        # Counter channel should have been pushed to.
        counts = list(run.extensions["counter"])
        assert len(counts) > 0
        # Non-native transformer should not set direct attributes.
        assert not hasattr(run, "counter")

    def test_native_transformer_gets_direct_attr(self) -> None:
        """A transformer with _native=True gets its keys as run attributes."""

        class FooTransformer(StreamTransformer):
            _native = True

            def __init__(self) -> None:
                super().__init__()
                self._log: EventLog[str] = EventLog()

            def init(self) -> dict[str, Any]:
                return {"foo": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":
                    self._log.push("saw_values")
                return True

            def finalize(self) -> None:
                self._log.close()

            def fail(self, err: BaseException) -> None:
                self._log.fail(err)

        graph = _build_simple_graph()
        handler = StreamingHandler(graph)
        foo_t = FooTransformer()
        run = handler.stream({"value": "x", "items": []}, transformers=[foo_t])
        _ = run.output
        # foo should be both in extensions and as a direct attribute.
        assert "foo" in run.extensions
        assert hasattr(run, "foo")
        assert run.foo is run.extensions["foo"]
        items = list(run.foo)
        assert "saw_values" in items

    def test_stream_channel_auto_forward(self) -> None:
        """StreamChannel pushes inject ProtocolEvents into main log."""

        class EmitterTransformer(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._channel: StreamChannel[str] = StreamChannel("emitter")

            def init(self) -> dict[str, Any]:
                return {"emitter": self._channel}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":
                    self._channel.push("emitted")
                return True

        graph = _build_simple_graph()
        handler = StreamingHandler(graph)
        run = handler.stream(
            {"value": "x", "items": []}, transformers=[EmitterTransformer()]
        )
        events = list(run)
        custom_events = [e for e in events if e["method"] == "custom:emitter"]
        assert len(custom_events) > 0
        assert custom_events[0]["params"]["data"] == "emitted"

    def test_stream_channel_seq_ordering(self) -> None:
        """Seq numbers in the main event log must be monotonically increasing.

        When a transformer pushes to a StreamChannel during process(), the
        auto-forwarded event enters the main log before the original event.
        The seq numbers must still be in order.
        """

        class ChannelPusher(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._channel: StreamChannel[str] = StreamChannel("ch")

            def init(self) -> dict[str, Any]:
                return {"ch": self._channel}

            def process(self, event: ProtocolEvent) -> bool:
                # Push to channel during process — this triggers auto-forward
                # which injects an event into the main log mid-pipeline.
                self._channel.push(f"saw:{event['method']}")
                return True

        mux = StreamMux()
        mux.register(ChannelPusher())

        mux.push(_event("values"))
        mux.push(_event("updates"))
        mux.close()

        events = list(mux._events)
        seqs = [e["seq"] for e in events]
        # Seq numbers must be strictly increasing.
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], f"Seq out of order at index {i}: {seqs}"
