"""Tests for the GraphStreamer and its supporting infrastructure."""

from __future__ import annotations

import asyncio
import operator
import sys
import time
from collections.abc import AsyncIterator, Iterator
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.stream import (
    AsyncGraphRunStream,
    EventLog,
    GraphRunStream,
    GraphStreamer,
    StreamChannel,
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


class _CustomPassthroughTransformer(StreamTransformer):
    """Opts a run into the `custom` stream mode without building a projection.

    `GraphStreamer` requests only the modes that registered
    transformers declare via `required_stream_modes`. Custom events are
    raw user emissions from `StreamWriter`, so tests that want them
    visible on the main event log register this pass-through transformer.
    """

    required_stream_modes = ("custom",)

    def init(self) -> dict[str, Any]:
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        return True


# ---------------------------------------------------------------------------
# EventLog unit tests
# ---------------------------------------------------------------------------


class TestEventLog:
    def test_sync_iteration(self) -> None:
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        it = iter(log)  # subscribe before pushing
        log.push(1)
        log.push(2)
        log.push(3)
        log.close()
        assert list(it) == [1, 2, 3]

    def test_drain_on_consume(self) -> None:
        """Items are popped as consumed — no retention across iterations."""
        log: EventLog[str] = EventLog()
        log._bind(is_async=False)
        it = iter(log)
        log.push("a")
        log.push("b")
        log.close()
        assert list(it) == ["a", "b"]
        # Buffer is drained.
        assert list(log._items) == []

    def test_second_subscribe_raises(self) -> None:
        """Only one subscriber allowed; tee() is the fan-out escape hatch."""
        log: EventLog[str] = EventLog()
        log._bind(is_async=False)
        log.close()
        _ = iter(log)
        with pytest.raises(RuntimeError, match="already has a subscriber"):
            iter(log)

    def test_pre_subscription_push_is_noop(self) -> None:
        """Lazy-subscribe: pushes before subscription are dropped silently."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        log.push(1)
        log.push(2)
        it = iter(log)
        log.push(3)
        log.close()
        # Only the post-subscribe push survives.
        assert list(it) == [3]

    def test_fail_propagation(self) -> None:
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        it = iter(log)
        log.push(1)
        log.fail(ValueError("test error"))
        with pytest.raises(ValueError, match="test error"):
            list(it)

    @pytest.mark.anyio
    async def test_async_iteration(self) -> None:
        log: EventLog[int] = EventLog()
        log._bind(is_async=True)
        cursor = aiter(log)
        for i in range(3):
            log.push(i)
        log.close()
        items = [item async for item in cursor]
        assert items == [0, 1, 2]

    @pytest.mark.anyio
    async def test_async_second_subscribe_raises(self) -> None:
        log: EventLog[str] = EventLog()
        log._bind(is_async=True)
        log.close()
        _ = log.__aiter__()
        with pytest.raises(RuntimeError, match="already has a subscriber"):
            log.__aiter__()

    @pytest.mark.anyio
    async def test_async_fail(self) -> None:
        log: EventLog[int] = EventLog()
        log._bind(is_async=True)
        cursor = aiter(log)
        log.push(1)
        log.fail(RuntimeError("async error"))
        with pytest.raises(RuntimeError, match="async error"):
            async for _ in cursor:
                pass

    def test_sync_cursor_yields_items_before_error(self) -> None:
        """Sync cursor should yield all buffered items before raising."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        it = iter(log)
        log.push(1)
        log.push(2)
        log.push(3)
        log.fail(ValueError("late error"))
        items: list[int] = []
        with pytest.raises(ValueError, match="late error"):
            for item in it:
                items.append(item)
        assert items == [1, 2, 3]

    @pytest.mark.anyio
    async def test_async_cursor_yields_items_before_error(self) -> None:
        """Async cursor should yield all buffered items before raising."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=True)
        cursor = aiter(log)
        log.push(1)
        log.push(2)
        log.push(3)
        log.fail(ValueError("late error"))
        items: list[int] = []
        with pytest.raises(ValueError, match="late error"):
            async for item in cursor:
                items.append(item)
        assert items == [1, 2, 3]

    def test_push_after_close_raises(self) -> None:
        """Push after close should raise RuntimeError (when subscribed)."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        it = iter(log)
        log.push(1)
        log.close()
        with pytest.raises(RuntimeError, match="Cannot push to a closed EventLog"):
            log.push(2)
        _ = list(it)

    def test_push_after_fail_raises(self) -> None:
        """Fail closes the log, so push after fail should also raise (when subscribed)."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        it = iter(log)
        log.fail(ValueError("err"))
        with pytest.raises(RuntimeError, match="Cannot push to a closed EventLog"):
            log.push(1)
        with pytest.raises(ValueError, match="err"):
            list(it)

    def test_empty_log_sync(self) -> None:
        """Iterating a closed empty log should yield nothing."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        log.close()
        assert list(log) == []

    @pytest.mark.anyio
    async def test_empty_log_async(self) -> None:
        """Async-iterating a closed empty log should yield nothing."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=True)
        log.close()
        assert [item async for item in log] == []

    def test_empty_log_fail_sync(self) -> None:
        """Failing an empty log should raise immediately with no items."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        log.fail(ValueError("empty fail"))
        with pytest.raises(ValueError, match="empty fail"):
            list(log)

    @pytest.mark.anyio
    async def test_empty_log_fail_async(self) -> None:
        """Failing an empty log should raise immediately with no items (async)."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=True)
        log.fail(ValueError("empty fail"))
        with pytest.raises(ValueError, match="empty fail"):
            async for _ in log:
                pass

    def test_unbound_iter_raises(self) -> None:
        """Iterating an unbound EventLog should raise TypeError."""
        log: EventLog[int] = EventLog()
        log.close()
        with pytest.raises(TypeError, match="has not been bound"):
            list(log)

    def test_sync_bound_aiter_raises(self) -> None:
        """Sync-bound EventLog should reject async iteration."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        log.close()
        with pytest.raises(TypeError, match="bound to sync mode"):
            log.__aiter__()

    @pytest.mark.anyio
    async def test_async_bound_iter_raises(self) -> None:
        """Async-bound EventLog should reject sync iteration."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=True)
        log.close()
        with pytest.raises(TypeError, match="bound to async mode"):
            iter(log)

    def test_double_bind_raises(self) -> None:
        """Binding an already-bound EventLog should raise."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        with pytest.raises(RuntimeError, match="already bound"):
            log._bind(is_async=True)


# ---------------------------------------------------------------------------
# StreamChannel unit tests
# ---------------------------------------------------------------------------


class TestStreamChannel:
    def test_push_and_iterate(self) -> None:
        ch: StreamChannel[str] = StreamChannel("test")
        ch._bind(is_async=False)
        it = iter(ch)
        ch.push("a")
        ch.push("b")
        ch._close()
        assert list(it) == ["a", "b"]

    def test_wire_callback(self) -> None:
        forwarded: list[str] = []
        ch: StreamChannel[str] = StreamChannel("test")
        ch._bind(is_async=False)
        ch._wire(lambda item: forwarded.append(item))
        it = iter(ch)
        ch.push("x")
        ch.push("y")
        ch._close()
        # Wire callback fires on every push, regardless of subscription.
        assert forwarded == ["x", "y"]
        assert list(it) == ["x", "y"]

    def test_fail_propagation(self) -> None:
        """_fail() should propagate the error through the underlying log."""
        ch: StreamChannel[str] = StreamChannel("test")
        ch._bind(is_async=False)
        it = iter(ch)
        ch.push("a")
        ch._fail(ValueError("channel error"))
        items: list[str] = []
        with pytest.raises(ValueError, match="channel error"):
            for item in it:
                items.append(item)
        assert items == ["a"]

    @pytest.mark.anyio
    async def test_async_iteration(self) -> None:
        """Async iteration should delegate to the inner event log."""
        ch: StreamChannel[str] = StreamChannel("test")
        ch._bind(is_async=True)
        cursor = ch.__aiter__()
        ch._log.push("x")
        ch._log.push("y")
        ch._close()
        items = [item async for item in cursor]
        assert items == ["x", "y"]

    def test_push_without_wire(self) -> None:
        """Push without a wire callback should still append to the log."""
        ch: StreamChannel[int] = StreamChannel("test")
        ch._bind(is_async=False)
        assert ch._wire_fn is None
        it = iter(ch)
        ch.push(42)
        ch._close()
        assert list(it) == [42]


# ---------------------------------------------------------------------------
# GraphStreamer sync tests
# ---------------------------------------------------------------------------


class TestGraphStreamerSync:
    def test_values_projection(self) -> None:
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
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
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        output = run.output
        assert output is not None
        assert output["value"] == "xAB"
        assert output["items"] == ["a", "b"]

    def test_raw_event_iteration(self) -> None:
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
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
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        # Drain events so the run completes.
        _ = run.output
        assert "values" in run.extensions
        assert "messages" in run.extensions
        # Native keys should also be direct attributes.
        assert run.values is run.extensions["values"]
        assert run.messages is run.extensions["messages"]

    def test_extensions_is_read_only(self) -> None:
        """`run.extensions` must reject mutations so users can't corrupt mux state."""
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        with pytest.raises(TypeError):
            run.extensions["new_key"] = object()  # type: ignore[index]
        with pytest.raises(TypeError):
            del run.extensions["values"]  # type: ignore[attr-defined]

    def test_custom_stream_events(self) -> None:
        graph = _build_custom_stream_graph()
        handler = GraphStreamer(graph)
        run = handler.stream(
            {"value": "x", "items": []},
            transformers=[_CustomPassthroughTransformer],
        )
        custom_events = [e for e in run if e["method"] == "custom"]
        assert len(custom_events) == 2
        assert custom_events[0]["params"]["data"] == {"step": "start"}
        assert custom_events[1]["params"]["data"] == {"step": "end"}

    def test_custom_events_suppressed_without_transformer(self) -> None:
        """Without a transformer declaring `"custom"`, no custom events flow.

        `GraphStreamer` asks the graph only for the modes that
        registered transformers require. Built-ins cover
        `values` / `messages` / `lifecycle`; consumers that want raw
        custom events surface them by registering a transformer whose
        `required_stream_modes` includes `"custom"`.
        """
        graph = _build_custom_stream_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        custom_events = [e for e in run if e["method"] == "custom"]
        assert custom_events == []

    def test_interleave_values_and_messages(self) -> None:
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})

        tagged = list(run.interleave("values", "messages"))
        names = [name for name, _ in tagged]
        assert set(names).issubset({"values", "messages"})
        # Values projection must have fired at least once.
        assert names.count("values") >= 1
        # Values have been drained by the interleave cursor — re-subscribing raises.
        with pytest.raises(RuntimeError, match="already has a subscriber"):
            list(run.values)

    def test_abort_marks_exhausted_and_closes_mux(self) -> None:
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        values_iter = iter(run.values)
        # Consume one item so the pump advances.
        _ = next(values_iter)
        run.abort()
        # Remaining iteration yields whatever was buffered, then stops.
        list(values_iter)
        assert run._exhausted is True
        # Second abort is idempotent.
        run.abort()

    def test_context_manager_calls_abort_on_exit(self) -> None:
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        with handler.stream({"value": "x", "items": []}) as run:
            values_iter = iter(run.values)
            _ = next(values_iter)
        assert run._exhausted is True

    def test_interleave_unknown_projection(self) -> None:
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        with pytest.raises(KeyError):
            list(run.interleave("values", "does_not_exist"))


class TestGraphStreamerSyncErrors:
    def test_error_propagation_output(self) -> None:
        graph = _build_error_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            _ = run.output

    def test_error_propagation_values(self) -> None:
        graph = _build_error_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            list(run.values)

    def test_error_propagation_raw_events(self) -> None:
        graph = _build_error_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            list(run)

    def test_error_propagation_interrupted(self) -> None:
        """`run.interrupted` should raise on a failed run, not silently return False."""
        graph = _build_error_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            _ = run.interrupted

    def test_error_propagation_interrupts(self) -> None:
        """`run.interrupts` should raise on a failed run."""
        graph = _build_error_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            _ = run.interrupts


class TestGraphStreamerSyncInterrupt:
    def test_interrupted(self) -> None:
        graph = _build_interrupt_graph()
        handler = GraphStreamer(graph)
        run = handler.stream(
            {"value": "x", "items": []},
            {"configurable": {"thread_id": "t1"}},
        )
        _ = run.output
        assert run.interrupted is True
        assert len(run.interrupts) > 0


# ---------------------------------------------------------------------------
# GraphStreamer async tests
# ---------------------------------------------------------------------------


class TestGraphStreamerAsync:
    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_values_projection(self) -> None:
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
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
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        output = await run.output()
        assert output is not None
        assert output["value"] == "xAB"
        assert output["items"] == ["a", "b"]

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_raw_event_iteration(self) -> None:
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        events = [e async for e in run]
        assert len(events) > 0
        for event in events:
            assert event["type"] == "event"

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_abort_marks_exhausted_and_closes_mux(self) -> None:
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        values_iter = aiter(run.values)
        _ = await anext(values_iter)
        await run.abort()
        # Drain the rest; should terminate promptly now that mux is closed.
        async for _item in values_iter:
            pass
        assert run._exhausted is True
        await run.abort()  # idempotent

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_async_context_manager_calls_abort_on_exit(self) -> None:
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        async with run:
            values_iter = aiter(run.values)
            _ = await anext(values_iter)
        assert run._exhausted is True

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_extensions_has_native_keys(self) -> None:
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        _ = await run.output()
        assert "values" in run.extensions
        assert "messages" in run.extensions
        assert run.values is run.extensions["values"]
        assert run.messages is run.extensions["messages"]


class TestGraphStreamerAsyncErrors:
    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_error_propagation_output(self) -> None:
        graph = _build_error_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            await run.output()

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_error_propagation_values(self) -> None:
        graph = _build_error_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            async for _ in run.values:
                pass

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_error_propagation_raw_events(self) -> None:
        graph = _build_error_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            async for _ in run:
                pass

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_error_propagation_interrupted(self) -> None:
        """`await run.interrupted()` should raise on a failed async run."""
        graph = _build_error_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            await run.interrupted()

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_error_propagation_interrupts(self) -> None:
        """`await run.interrupts()` should raise on a failed async run."""
        graph = _build_error_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        with pytest.raises(ValueError, match="boom"):
            await run.interrupts()


class TestGraphStreamerAsyncInterrupt:
    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_interrupted(self) -> None:
        graph = _build_interrupt_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream(
            {"value": "x", "items": []},
            {"configurable": {"thread_id": "t2"}},
        )
        _ = await run.output()
        assert await run.interrupted() is True
        assert len(await run.interrupts()) > 0


class TestGraphStreamerAsyncCustom:
    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_custom_stream_events(self) -> None:
        graph = _build_custom_stream_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream(
            {"value": "x", "items": []},
            transformers=[_CustomPassthroughTransformer],
        )
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
        """init() returning a non-dict should raise TypeError at construction."""

        class BadTransformer(StreamTransformer):
            def init(self) -> Any:
                return ["not", "a", "dict"]

            def process(self, event: ProtocolEvent) -> bool:
                return True

        with pytest.raises(TypeError, match="must return a dict"):
            StreamMux([BadTransformer()])

    def test_event_suppression(self) -> None:
        """When process() returns False, the event should not appear in the main log."""

        class FilterTransformer(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                # Suppress "updates" events
                return event["method"] != "updates"

        mux = StreamMux([FilterTransformer()])
        it = iter(mux._events)

        mux.push(_event("values", {"a": 1}))
        mux.push(_event("updates", {"b": 2}))
        mux.push(_event("custom", {"c": 3}))
        mux.close()

        events = list(it)
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

        mux = StreamMux([PassTransformer(), RejectTransformer()])

        mux.push(_event("values"))
        mux.close()

        # RejectTransformer saw the event even though it rejected it
        assert seen_by_second == ["values"]
        # But nothing in the main log
        assert list(mux._events) == []

    def test_empty_mux(self) -> None:
        """Push/close/fail on a mux with no transformers should work."""
        mux = StreamMux()
        it = iter(mux._events)
        mux.push(_event("values", {"x": 1}))
        mux.close()
        events = list(it)
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
        """The root mux only dispatches root-ns values events to its ValuesTransformer.

        Namespace filtering is enforced by the mux via `scope_exact`
        — the transformer itself no longer filters.
        """
        mux = StreamMux([ValuesTransformer()], is_async=False)
        t = mux.transformer_by_key("values")
        assert isinstance(t, ValuesTransformer)
        it = iter(t._log)

        mux.push(_event("values", {"val": "root"}))
        mux.push(_event("values", {"val": "sub"}, namespace=["sub"]))

        t._log.close()
        items = list(it)
        assert len(items) == 1
        assert items[0]["val"] == "root"

    def test_ignores_non_values_methods(self) -> None:
        """Non-values events should be passed through but not captured."""
        t = ValuesTransformer()
        t.init()
        t._log._bind(is_async=False)
        it = iter(t._log)

        result = t.process(_event("updates", {"x": 1}))
        assert result is True  # passed through
        t._log.close()
        assert list(it) == []  # but not captured

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
        """Protocol-event lifecycle produces a ChatModelStream in the log."""
        t = MessagesTransformer()
        t.init()
        t._log._bind(is_async=False)
        t._bind_pump(lambda: False)
        it = iter(t._log)

        meta = {"langgraph_node": "llm", "run_id": "run-1"}
        for evt in (
            {"event": "message-start", "role": "ai", "message_id": "run-1"},
            {"event": "message-finish", "reason": "stop"},
        ):
            t.process(_event("messages", (evt, meta)))
        t._log.close()
        items = list(it)
        assert len(items) == 1
        # Items in the messages log are ChatModelStream objects, not raw
        # tuples — the content-block-centric projection.
        assert hasattr(items[0], "dispatch")
        assert items[0].message_id == "run-1"

    def test_ignores_non_root_namespace(self) -> None:
        """Namespace filtering is enforced by the mux via `scope_exact`."""
        mux = StreamMux([MessagesTransformer()], is_async=False)
        t = mux.transformer_by_key("messages")
        assert isinstance(t, MessagesTransformer)
        t._bind_pump(lambda: False)
        it = iter(t._log)

        meta = {"langgraph_node": "llm", "run_id": "run-1"}
        mux.push(
            _event(
                "messages",
                ({"event": "message-start", "message_id": "run-1"}, meta),
                namespace=["sub"],
            )
        )
        t._log.close()
        assert list(it) == []

    def test_ignores_non_messages_methods(self) -> None:
        t = MessagesTransformer()
        t.init()
        t._log._bind(is_async=False)
        it = iter(t._log)

        result = t.process(_event("values", {"v": 1}))
        assert result is True
        t._log.close()
        assert list(it) == []

    def test_fail_propagates(self) -> None:
        t = MessagesTransformer()
        t.init()
        t._log._bind(is_async=False)
        it = iter(t._log)
        t._log.fail(ValueError("msg error"))
        with pytest.raises(ValueError, match="msg error"):
            list(it)


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

        good = GoodTransformer()
        mux = StreamMux([BrokenFinalizer(), good])

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

        good = GoodTransformer()
        mux = StreamMux([BrokenFailer(), good])

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
        mux = StreamMux([t])

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
        handler = GraphStreamer(graph)
        counter_t = CounterTransformer()
        run = handler.stream(
            {"value": "x", "items": []},
            transformers=[lambda _scope: counter_t],
        )
        assert "counter" in run.extensions
        # Subscribe before driving the run so channel pushes are retained.
        counter_iter = iter(run.extensions["counter"])
        _ = run.output
        counts = list(counter_iter)
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

        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        foo_t = FooTransformer()
        run = handler.stream(
            {"value": "x", "items": []},
            transformers=[lambda _scope: foo_t],
        )
        # Subscribe before driving the run.
        foo_iter = iter(run.foo)
        _ = run.output
        # foo should be both in extensions and as a direct attribute.
        assert "foo" in run.extensions
        assert hasattr(run, "foo")
        assert run.foo is run.extensions["foo"]
        items = list(foo_iter)
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
        handler = GraphStreamer(graph)
        run = handler.stream(
            {"value": "x", "items": []},
            transformers=[lambda _scope: EmitterTransformer()],
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

        mux = StreamMux([ChannelPusher()])
        it = iter(mux._events)

        mux.push(_event("values"))
        mux.push(_event("updates"))
        mux.close()

        events = list(it)
        seqs = [e["seq"] for e in events]
        # Seq numbers must be strictly increasing.
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], f"Seq out of order at index {i}: {seqs}"

    def test_projection_key_conflict_raises(self) -> None:
        """User transformer that collides with a built-in key should raise."""

        class ConflictTransformer(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._log: EventLog[str] = EventLog()

            def init(self) -> dict[str, Any]:
                return {"values": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                return True

        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        with pytest.raises(
            ValueError,
            match=r"conflict.*'values'.*ValuesTransformer",
        ):
            handler.stream(
                {"value": "x", "items": []},
                transformers=[lambda _scope: ConflictTransformer()],
            )


class TestEventLogAutoLifecycle:
    def test_mux_auto_closes_event_logs(self) -> None:
        """EventLogs in projections should be auto-closed by mux.close()."""

        class SimpleTransformer(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._log: EventLog[str] = EventLog()

            def init(self) -> dict[str, Any]:
                return {"items": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                self._log.push("saw_event")
                return True

        mux = StreamMux([SimpleTransformer()])
        it = iter(mux._events)

        mux.push(_event("values"))
        mux.close()

        # The log should have been auto-closed — iteration should work.
        items = list(it)
        assert len(items) == 1

    def test_mux_auto_fails_event_logs(self) -> None:
        """EventLogs in projections should be auto-failed by mux.fail()."""

        class SimpleTransformer(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._log: EventLog[str] = EventLog()

            def init(self) -> dict[str, Any]:
                return {"items": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                self._log.push("saw_event")
                return True

        t = SimpleTransformer()
        mux = StreamMux([t])
        it = iter(t._log)

        mux.push(_event("values"))
        mux.fail(ValueError("boom"))

        # The transformer's log should have been auto-failed.
        with pytest.raises(ValueError, match="boom"):
            list(it)

    def test_no_double_close_if_transformer_closes_own_log(self) -> None:
        """If a transformer closes its log in finalize(), mux should not error."""

        class ManualCloseTransformer(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._log: EventLog[str] = EventLog()

            def init(self) -> dict[str, Any]:
                return {"items": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                return True

            def finalize(self) -> None:
                self._log.close()

        mux = StreamMux([ManualCloseTransformer()])
        # Should not raise even though the log is closed by both
        # the transformer and the mux.
        mux.close()

    def test_transformer_without_finalize_works(self) -> None:
        """Transformer with only init+process should work end-to-end."""

        class MinimalTransformer(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._log: EventLog[str] = EventLog()

            def init(self) -> dict[str, Any]:
                return {"minimal": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":
                    self._log.push("got_it")
                return True

        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        t = MinimalTransformer()
        run = handler.stream(
            {"value": "x", "items": []},
            transformers=[lambda _scope: t],
        )
        minimal_iter = iter(run.extensions["minimal"])
        _ = run.output
        items = list(minimal_iter)
        assert len(items) > 0


# ---------------------------------------------------------------------------
# Async transformer lane — aprocess / afinalize / afail / schedule()
# ---------------------------------------------------------------------------


class TestAsyncTransformerLane:
    @pytest.mark.anyio
    async def test_aprocess_is_awaited_before_next_transformer(self) -> None:
        """aprocess must complete before the next transformer sees the event.

        This is the load-bearing guarantee for mutating transformers
        like PII redaction: the downstream transformer reads the mutated
        event synchronously.
        """
        order: list[str] = []

        class RedactTransformer(StreamTransformer):
            requires_async = True

            def init(self) -> dict[str, Any]:
                return {}

            async def aprocess(self, event: ProtocolEvent) -> bool:
                await asyncio.sleep(0.01)
                order.append("redact")
                event["params"]["data"]["redacted"] = True
                return True

        class ObserverTransformer(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":
                    order.append(f"observe:{event['params']['data'].get('redacted')}")
                return True

        mux = StreamMux([RedactTransformer(), ObserverTransformer()], is_async=True)

        await mux.apush(_event("values", {"secret": "x"}))
        await mux.aclose()

        assert order == ["redact", "observe:True"]

    @pytest.mark.anyio
    async def test_schedule_joins_tasks_before_afinalize(self) -> None:
        """Every scheduled task must complete before afinalize runs."""
        phase: list[str] = []

        class SchedTransformer(StreamTransformer):
            requires_async = True

            def __init__(self) -> None:
                self._log: EventLog[str] = EventLog()

            def init(self) -> dict[str, Any]:
                return {"out": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":

                    async def work() -> None:
                        await asyncio.sleep(0.01)
                        phase.append("task")
                        self._log.push("done")

                    self.schedule(work())
                return True

            async def afinalize(self) -> None:
                phase.append("afinalize")
                self._log.close()

        t = SchedTransformer()
        mux = StreamMux([t], is_async=True)

        await mux.apush(_event("values", {}))
        await mux.apush(_event("values", {}))
        await mux.aclose()

        # Both tasks ran before afinalize; the log holds both pushes.
        assert phase.count("task") == 2
        assert phase[-1] == "afinalize"

    @pytest.mark.anyio
    async def test_sync_stream_rejects_async_transformer(self) -> None:
        """Registering a requires_async transformer on a sync mux raises."""

        class NeedsAsync(StreamTransformer):
            requires_async = True

            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                return True

        with pytest.raises(RuntimeError, match="requires an async run"):
            StreamMux([NeedsAsync()], is_async=False)

    @pytest.mark.anyio
    async def test_sync_stream_rejects_aprocess_override(self) -> None:
        """Overriding aprocess also marks the transformer as async-required."""

        class HasAprocess(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            async def aprocess(self, event: ProtocolEvent) -> bool:
                return True

        with pytest.raises(RuntimeError, match="requires an async run"):
            StreamMux([HasAprocess()], is_async=False)

    def test_schedule_without_running_loop_raises(self) -> None:
        """schedule() called outside an event loop fails with a clear message."""

        class Sched(StreamTransformer):
            requires_async = True

            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                return True

        t = Sched()

        async def noop() -> None:
            pass

        coro = noop()
        try:
            with pytest.raises(RuntimeError, match="requires a running event loop"):
                t.schedule(coro)
        finally:
            coro.close()

    @pytest.mark.anyio
    async def test_schedule_on_error_log_swallows_exceptions(self) -> None:
        """on_error="log" (default) keeps the run alive when a task fails."""

        class Bad(StreamTransformer):
            requires_async = True

            def __init__(self) -> None:
                self._log: EventLog[str] = EventLog()

            def init(self) -> dict[str, Any]:
                return {"out": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                async def work() -> None:
                    raise ValueError("boom")

                self.schedule(work())  # default on_error="log"
                return True

            async def afinalize(self) -> None:
                self._log.close()

        mux = StreamMux([Bad()], is_async=True)

        await mux.apush(_event("values", {}))
        # Should not raise; the scheduled task's exception is logged.
        await mux.aclose()

    @pytest.mark.anyio
    async def test_schedule_on_error_raise_fails_the_run(self) -> None:
        """on_error="raise" propagates the exception through aclose."""

        class Strict(StreamTransformer):
            requires_async = True

            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                async def work() -> None:
                    raise ValueError("strict boom")

                self.schedule(work(), on_error="raise")
                return True

        mux = StreamMux([Strict()], is_async=True)

        await mux.apush(_event("values", {}))
        with pytest.raises(ValueError, match="strict boom"):
            await mux.aclose()

    @pytest.mark.anyio
    async def test_afail_cancels_pending_scheduled_tasks(self) -> None:
        """When the run fails, outstanding scheduled tasks are cancelled."""
        cancelled = asyncio.Event()

        class Sched(StreamTransformer):
            requires_async = True

            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                async def work() -> None:
                    try:
                        await asyncio.sleep(5)
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise

                self.schedule(work())
                return True

        mux = StreamMux([Sched()], is_async=True)

        await mux.apush(_event("values", {}))
        # Yield so the scheduled task actually starts before we cancel it;
        # otherwise it's cancelled before its first step and the `except`
        # inside work() never runs.
        await asyncio.sleep(0)
        await mux.afail(RuntimeError("run died"))

        assert cancelled.is_set()

    @pytest.mark.anyio
    async def test_mixed_sync_and_async_transformers(self) -> None:
        """Sync and async transformers coexist under astream."""
        seen_sync: list[str] = []

        class SyncOne(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                seen_sync.append(event["method"])
                return True

        class AsyncOne(StreamTransformer):
            requires_async = True

            def __init__(self) -> None:
                self._log: EventLog[str] = EventLog()

            def init(self) -> dict[str, Any]:
                return {"seen": self._log}

            async def aprocess(self, event: ProtocolEvent) -> bool:
                await asyncio.sleep(0)
                self._log.push(event["method"])
                return True

            async def afinalize(self) -> None:
                self._log.close()

        async_t = AsyncOne()
        mux = StreamMux([SyncOne(), async_t], is_async=True)
        seen_cursor = aiter(async_t._log)

        await mux.apush(_event("values", {}))
        await mux.apush(_event("updates", {}))
        await mux.aclose()

        assert seen_sync == ["values", "updates"]
        items = [x async for x in seen_cursor]
        assert items == ["values", "updates"]

    @pytest.mark.anyio
    async def test_handler_astream_with_scheduled_work(self) -> None:
        """End-to-end: transformer schedules work during an astream run."""

        class Scorer(StreamTransformer):
            requires_async = True

            def __init__(self) -> None:
                self._log: EventLog[int] = EventLog()

            def init(self) -> dict[str, Any]:
                return {"scores": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":

                    async def work() -> None:
                        await asyncio.sleep(0.01)
                        self._log.push(42)

                    self.schedule(work())
                return True

            async def afinalize(self) -> None:
                self._log.close()

        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        scorer = Scorer()
        run = await handler.astream(
            {"value": "x", "items": []},
            transformers=[lambda _scope: scorer],
        )
        # Subscribe before driving the run so scheduled pushes are retained.
        scores_cursor = aiter(run.extensions["scores"])
        _ = await run.output()
        scores = [x async for x in scores_cursor]
        assert scores and all(s == 42 for s in scores)


# ---------------------------------------------------------------------------
# Drain-on-consume semantics — bounded backpressure, single subscriber
# ---------------------------------------------------------------------------


class TestMemoryBounds:
    """Drain-on-consume guarantees memory stays bounded for the common
    access patterns. These tests lock in the property — if a change
    re-introduces retention, they should fail."""

    def test_sync_subscribed_buffer_stays_at_most_one_between_yields(self) -> None:
        """With a single sync consumer, the pump produces exactly one event
        per cursor advance, so the buffer never holds more than one."""
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        events_iter = iter(run)
        max_buffered = 0
        count = 0
        for _ in events_iter:
            max_buffered = max(max_buffered, len(run._mux._events._items))
            count += 1
        assert count > 0
        assert max_buffered == 0, (
            f"Subscribed buffer should hold 0 items after each yield "
            f"(drain-on-consume), observed max {max_buffered}"
        )

    def test_unsubscribed_projections_never_accumulate(self) -> None:
        """Projections without a subscriber drop pushes silently —
        their buffers stay empty regardless of run length."""
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        # Subscribe to main events only; leave values and messages unsubscribed.
        list(run)
        values_log = run.extensions["values"]
        messages_log = run.extensions["messages"]
        assert len(values_log._items) == 0
        assert len(messages_log._items) == 0
        assert values_log._subscribed is False
        assert messages_log._subscribed is False

    def test_output_path_does_not_retain_values(self) -> None:
        """`run.output` is a scalar accessor — it updates `_latest` from
        process() without populating the log, so the values log buffer
        stays empty even across a full run."""
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        _ = run.output
        values_log = run.extensions["values"]
        assert len(values_log._items) == 0
        assert values_log._subscribed is False

    def test_drained_subscriber_buffer_returns_to_empty(self) -> None:
        """After fully draining a subscribed log, the internal deque is empty."""
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = handler.stream({"value": "x", "items": []})
        values_log = run.extensions["values"]
        list(run.values)
        assert len(values_log._items) == 0

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_async_single_consumer_buffer_stays_at_most_one(self) -> None:
        """Same drain-on-consume guarantee for the async lane."""
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        max_buffered = 0
        count = 0
        async for _ in run:
            max_buffered = max(max_buffered, len(run._mux._events._items))
            count += 1
        assert count > 0
        assert max_buffered == 0

    @pytest.mark.anyio
    @NEEDS_CONTEXTVARS
    async def test_async_unsubscribed_projections_never_accumulate(self) -> None:
        """Projections with no async subscriber stay empty under astream."""
        graph = _build_simple_graph()
        handler = GraphStreamer(graph)
        run = await handler.astream({"value": "x", "items": []})
        _ = await run.output()
        values_log = run.extensions["values"]
        messages_log = run.extensions["messages"]
        assert len(values_log._items) == 0
        assert len(messages_log._items) == 0
        assert values_log._subscribed is False
        assert messages_log._subscribed is False


class TestDrainOnConsume:
    def test_invalid_maxlen_raises(self) -> None:
        with pytest.raises(ValueError, match="positive int or None"):
            EventLog(maxlen=0)
        with pytest.raises(ValueError, match="positive int or None"):
            EventLog(maxlen=-3)

    def test_push_unbounded_by_design(self) -> None:
        """Push is non-blocking and doesn't enforce capacity — the
        caller-driven pump bounds memory via iteration pace."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        it = iter(log)
        for i in range(100):
            log.push(i)
        log.close()
        assert list(it) == list(range(100))

    @pytest.mark.anyio
    async def test_atee_fans_out(self) -> None:
        """atee provides the documented fan-out for concurrent consumers."""
        log: EventLog[int] = EventLog()
        log._bind(is_async=True)
        a, b = log.atee(2)
        for i in range(3):
            log.push(i)
        log.close()
        items_a = [x async for x in a]
        items_b = [x async for x in b]
        assert items_a == [0, 1, 2]
        assert items_b == [0, 1, 2]

    def test_tee_fans_out_sync(self) -> None:
        log: EventLog[int] = EventLog()
        log._bind(is_async=False)
        a, b = log.tee(2)
        for i in range(3):
            log.push(i)
        log.close()
        assert list(a) == [0, 1, 2]
        assert list(b) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Subclassing hooks
# ---------------------------------------------------------------------------


class _CountingTransformer(StreamTransformer):
    """Record how many events the transformer saw — nothing more."""

    required_stream_modes = ("values",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: EventLog[int] = EventLog()
        self.count = 0

    def init(self) -> dict[str, Any]:
        return {"counts": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "values":
            self.count += 1
        return True


class _CustomRunStream(GraphRunStream):
    """Trivial subclass used to prove the `_make_run_stream` hook fires."""

    marker = "custom-sync"


class _CustomAsyncRunStream(AsyncGraphRunStream):
    marker = "custom-async"


class _CustomStreamer(GraphStreamer):
    """`GraphStreamer` subclass that injects a transformer and narrows run types."""

    builtin_factories = (
        *GraphStreamer.builtin_factories,
        _CountingTransformer,
    )

    def _make_run_stream(
        self,
        graph_iter: Iterator[Any],
        mux: StreamMux,
    ) -> _CustomRunStream:
        return _CustomRunStream(graph_iter, mux)

    def _make_async_run_stream(
        self,
        graph_aiter: AsyncIterator[Any],
        mux: StreamMux,
    ) -> _CustomAsyncRunStream:
        return _CustomAsyncRunStream(graph_aiter, mux)


class TestGraphStreamerSubclassing:
    def test_subclass_injects_transformer_without_user_opt_in(self) -> None:
        graph = _build_simple_graph()
        streamer = _CustomStreamer(graph)
        run = streamer.stream({"value": "", "items": []})

        # Subclass-injected transformer is present in every run's mux
        # without the caller passing `transformers=`.
        assert isinstance(run, _CustomRunStream)
        assert run.marker == "custom-sync"
        counting = run._mux.transformer_by_key("counts")
        assert isinstance(counting, _CountingTransformer)
        run.output  # drive to completion
        assert counting.count >= 1

    @pytest.mark.anyio
    async def test_async_subclass_returns_custom_async_stream(self) -> None:
        graph = _build_simple_graph()
        streamer = _CustomStreamer(graph)
        run = await streamer.astream({"value": "", "items": []})

        assert isinstance(run, _CustomAsyncRunStream)
        assert run.marker == "custom-async"
        counting = run._mux.transformer_by_key("counts")
        assert isinstance(counting, _CountingTransformer)
        await run.output()
        assert counting.count >= 1

    def test_user_transformers_appended_after_builtin_factories(self) -> None:
        """User transformers run after `builtin_factories` in registration order."""

        class _UserTransformer(StreamTransformer):
            required_stream_modes = ()

            def __init__(self, scope: tuple[str, ...] = ()) -> None:
                super().__init__(scope)

            def init(self) -> dict[str, Any]:
                return {"user_flag": EventLog()}

            def process(self, event: ProtocolEvent) -> bool:
                return True

        graph = _build_simple_graph()
        streamer = _CustomStreamer(graph)
        run = streamer.stream(
            {"value": "", "items": []}, transformers=[_UserTransformer]
        )

        # Both the subclass-injected and user-supplied projections are
        # available.
        assert "counts" in run.extensions
        assert "user_flag" in run.extensions
        run.output  # drain
