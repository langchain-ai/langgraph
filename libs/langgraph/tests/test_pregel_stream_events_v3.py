"""Tests for Pregel.stream_events(version="v3") / astream_events(version="v3") and the transformer pipeline."""

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
    StreamChannel,
    StreamTransformer,
)
from langgraph.stream._convert import convert_to_protocol_event
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.run_stream import AsyncGraphRunStream, GraphRunStream
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
    params: dict[str, Any] = {
        "namespace": namespace or [],
        "timestamp": TS,
        "data": data if data is not None else {},
    }
    if interrupts is not None:
        params["interrupts"] = interrupts
    return {"type": "event", "method": method, "params": params}


# ---------------------------------------------------------------------------
# Shared graph builders
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def _build_simple_graph():
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

    `stream_events(version="v3")` requests only the modes that registered transformers
    declare via `required_stream_modes`. Custom events are raw user
    emissions from `StreamWriter`, so tests that want them visible on
    the main event log register this pass-through transformer.
    """

    required_stream_modes = ("custom",)

    def init(self) -> dict[str, Any]:
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        return True


# ---------------------------------------------------------------------------
# StreamChannel (local, unnamed) unit tests
# ---------------------------------------------------------------------------


class TestStreamChannelLocal:
    def test_sync_iteration(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=False)
        it = iter(log)
        log.push(1)
        log.push(2)
        log.push(3)
        log.close()
        assert list(it) == [1, 2, 3]

    def test_drain_on_consume(self) -> None:
        log: StreamChannel[str] = StreamChannel()
        log._bind(is_async=False)
        it = iter(log)
        log.push("a")
        log.push("b")
        log.close()
        assert list(it) == ["a", "b"]
        assert list(log._items) == []

    def test_second_subscribe_raises(self) -> None:
        log: StreamChannel[str] = StreamChannel()
        log._bind(is_async=False)
        log.close()
        _ = iter(log)
        with pytest.raises(RuntimeError, match="already has a subscriber"):
            iter(log)

    def test_pre_subscription_push_is_noop(self) -> None:
        # Lazy-subscribe: pushes before subscription are dropped silently.
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=False)
        log.push(1)
        log.push(2)
        it = iter(log)
        log.push(3)
        log.close()
        assert list(it) == [3]

    def test_fail_propagation(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=False)
        it = iter(log)
        log.push(1)
        log.fail(ValueError("test error"))
        with pytest.raises(ValueError, match="test error"):
            list(it)

    def test_sync_cursor_yields_items_before_error(self) -> None:
        log: StreamChannel[int] = StreamChannel()
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

    def test_push_after_close_raises(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=False)
        it = iter(log)
        log.push(1)
        log.close()
        with pytest.raises(RuntimeError, match="Cannot push to a closed StreamChannel"):
            log.push(2)
        _ = list(it)

    def test_push_after_fail_raises(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=False)
        it = iter(log)
        log.fail(ValueError("err"))
        with pytest.raises(RuntimeError, match="Cannot push to a closed StreamChannel"):
            log.push(1)
        with pytest.raises(ValueError, match="err"):
            list(it)

    def test_empty_log_sync(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=False)
        log.close()
        assert list(log) == []

    def test_empty_log_fail_sync(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=False)
        log.fail(ValueError("empty fail"))
        with pytest.raises(ValueError, match="empty fail"):
            list(log)

    def test_unbound_iter_raises(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log.close()
        with pytest.raises(TypeError, match="has not been bound"):
            list(log)

    def test_sync_bound_aiter_raises(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=False)
        log.close()
        with pytest.raises(TypeError, match="bound to sync mode"):
            log.__aiter__()

    def test_double_bind_raises(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=False)
        with pytest.raises(RuntimeError, match="already bound"):
            log._bind(is_async=True)

    @pytest.mark.anyio
    async def test_async_iteration(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=True)
        cursor = aiter(log)
        for i in range(3):
            log.push(i)
        log.close()
        assert [item async for item in cursor] == [0, 1, 2]

    @pytest.mark.anyio
    async def test_async_second_subscribe_raises(self) -> None:
        log: StreamChannel[str] = StreamChannel()
        log._bind(is_async=True)
        log.close()
        _ = log.__aiter__()
        with pytest.raises(RuntimeError, match="already has a subscriber"):
            log.__aiter__()

    @pytest.mark.anyio
    async def test_async_fail(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=True)
        cursor = aiter(log)
        log.push(1)
        log.fail(RuntimeError("async error"))
        with pytest.raises(RuntimeError, match="async error"):
            async for _ in cursor:
                pass

    @pytest.mark.anyio
    async def test_async_cursor_yields_items_before_error(self) -> None:
        log: StreamChannel[int] = StreamChannel()
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

    @pytest.mark.anyio
    async def test_empty_log_async(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=True)
        log.close()
        assert [item async for item in log] == []

    @pytest.mark.anyio
    async def test_empty_log_fail_async(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=True)
        log.fail(ValueError("empty fail"))
        with pytest.raises(ValueError, match="empty fail"):
            async for _ in log:
                pass

    @pytest.mark.anyio
    async def test_async_bound_iter_raises(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=True)
        log.close()
        with pytest.raises(TypeError, match="bound to async mode"):
            iter(log)


# ---------------------------------------------------------------------------
# StreamChannel (named, wired) unit tests
# ---------------------------------------------------------------------------


class TestStreamChannelNamed:
    def test_push_and_iterate(self) -> None:
        ch: StreamChannel[str] = StreamChannel("test")
        ch._bind(is_async=False)
        it = iter(ch)
        ch.push("a")
        ch.push("b")
        ch.close()
        assert list(it) == ["a", "b"]

    def test_wire_callback(self) -> None:
        forwarded: list[str] = []
        ch: StreamChannel[str] = StreamChannel("test")
        ch._bind(is_async=False)
        ch._wire(lambda item: forwarded.append(item))
        it = iter(ch)
        ch.push("x")
        ch.push("y")
        ch.close()
        assert forwarded == ["x", "y"]
        assert list(it) == ["x", "y"]

    def test_fail_propagation(self) -> None:
        ch: StreamChannel[str] = StreamChannel("test")
        ch._bind(is_async=False)
        it = iter(ch)
        ch.push("a")
        ch.fail(ValueError("channel error"))
        items: list[str] = []
        with pytest.raises(ValueError, match="channel error"):
            for item in it:
                items.append(item)
        assert items == ["a"]

    def test_push_without_wire(self) -> None:
        ch: StreamChannel[int] = StreamChannel("test")
        ch._bind(is_async=False)
        assert ch._wire_fn is None
        it = iter(ch)
        ch.push(42)
        ch.close()
        assert list(it) == [42]

    @pytest.mark.anyio
    async def test_async_iteration(self) -> None:
        ch: StreamChannel[str] = StreamChannel("test")
        ch._bind(is_async=True)
        cursor = ch.__aiter__()
        ch.push("x")
        ch.push("y")
        ch.close()
        assert [item async for item in cursor] == ["x", "y"]


# ---------------------------------------------------------------------------
# stream_events(version="v3") sync tests
# ---------------------------------------------------------------------------


class TestStreamV2Sync:
    def test_values_projection(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        snapshots = list(run.values)
        assert len(snapshots) >= 1
        last = snapshots[-1]
        assert "A" in last["value"] and "B" in last["value"]

    def test_output(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        output = run.output
        assert output == {"value": "xAB", "items": ["a", "b"]}

    def test_raw_event_iteration(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        events = list(run)
        assert len(events) > 0
        for event in events:
            assert event["type"] == "event"
            assert "method" in event
            assert "seq" in event
            assert isinstance(event["params"]["timestamp"], int)

    def test_extensions_has_native_keys(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        _ = run.output
        assert "values" in run.extensions and "messages" in run.extensions
        assert run.values is run.extensions["values"]
        assert run.messages is run.extensions["messages"]

    def test_extensions_is_read_only(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(TypeError):
            run.extensions["new_key"] = object()  # type: ignore[index]
        with pytest.raises(TypeError):
            del run.extensions["values"]  # type: ignore[attr-defined]

    def test_custom_stream_events(self) -> None:
        run = _build_custom_stream_graph().stream_events({"value": "x", "items": []}, version="v3", transformers=[_CustomPassthroughTransformer],
        )
        custom_events = [e for e in run if e["method"] == "custom"]
        assert len(custom_events) == 2
        assert custom_events[0]["params"]["data"] == {"step": "start"}
        assert custom_events[1]["params"]["data"] == {"step": "end"}

    def test_custom_events_suppressed_without_transformer(self) -> None:
        """Without a transformer declaring `"custom"`, no custom events flow.

        `stream_events(version="v3")` asks the graph only for the modes that registered
        transformers require. Built-ins cover `values` / `messages`;
        consumers that want raw custom events surface them by
        registering a transformer whose `required_stream_modes`
        includes `"custom"`.
        """
        run = _build_custom_stream_graph().stream_events({"value": "x", "items": []}, version="v3")
        custom_events = [e for e in run if e["method"] == "custom"]
        assert custom_events == []

    def test_interleave_values_and_messages(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        tagged = list(run.interleave("values", "messages"))
        names = [name for name, _ in tagged]
        assert set(names).issubset({"values", "messages"})
        assert names.count("values") >= 1
        # interleave releases its subscription on completion.
        assert run.extensions["values"]._subscribed is False
        assert run.extensions["messages"]._subscribed is False

    def test_abort_marks_exhausted_and_closes_mux(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        values_iter = iter(run.values)
        _ = next(values_iter)
        run.abort()
        list(values_iter)
        assert run._exhausted is True
        run.abort()  # idempotent

    def test_context_manager_calls_abort_on_exit(self) -> None:
        with _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3") as run:
            _ = next(iter(run.values))
        assert run._exhausted is True

    def test_interleave_unknown_projection(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(KeyError):
            list(run.interleave("values", "does_not_exist"))


class TestStreamV2SyncErrors:
    def test_error_propagation_output(self) -> None:
        run = _build_error_graph().stream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(ValueError, match="boom"):
            _ = run.output

    def test_error_propagation_values(self) -> None:
        run = _build_error_graph().stream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(ValueError, match="boom"):
            list(run.values)

    def test_error_propagation_raw_events(self) -> None:
        run = _build_error_graph().stream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(ValueError, match="boom"):
            list(run)

    def test_error_propagation_interrupted(self) -> None:
        run = _build_error_graph().stream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(ValueError, match="boom"):
            _ = run.interrupted

    def test_error_propagation_interrupts(self) -> None:
        run = _build_error_graph().stream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(ValueError, match="boom"):
            _ = run.interrupts


class TestStreamV2SyncInterrupt:
    def test_interrupted(self) -> None:
        run = _build_interrupt_graph().stream_events({"value": "x", "items": []}, {"configurable": {"thread_id": "t1"}}, version="v3")
        _ = run.output
        assert run.interrupted is True
        assert len(run.interrupts) > 0


# ---------------------------------------------------------------------------
# astream_events(version="v3") async tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@NEEDS_CONTEXTVARS
class TestStreamV2Async:
    async def test_values_projection(self) -> None:
        run = await _build_simple_graph().astream_events({"value": "x", "items": []}, version="v3")
        snapshots = [s async for s in run.values]
        assert len(snapshots) >= 1
        last = snapshots[-1]
        assert "A" in last["value"] and "B" in last["value"]

    async def test_output(self) -> None:
        run = await _build_simple_graph().astream_events({"value": "x", "items": []}, version="v3")
        output = await run.output()
        assert output == {"value": "xAB", "items": ["a", "b"]}

    async def test_raw_event_iteration(self) -> None:
        run = await _build_simple_graph().astream_events({"value": "x", "items": []}, version="v3")
        events = [e async for e in run]
        assert len(events) > 0
        for event in events:
            assert event["type"] == "event"

    async def test_abort_marks_exhausted_and_closes_mux(self) -> None:
        run = await _build_simple_graph().astream_events({"value": "x", "items": []}, version="v3")
        values_iter = aiter(run.values)
        _ = await anext(values_iter)
        await run.abort()
        async for _item in values_iter:
            pass
        assert run._exhausted is True
        await run.abort()  # idempotent

    async def test_context_manager_calls_abort_on_exit(self) -> None:
        run = await _build_simple_graph().astream_events({"value": "x", "items": []}, version="v3")
        async with run:
            _ = await anext(aiter(run.values))
        assert run._exhausted is True

    async def test_extensions_has_native_keys(self) -> None:
        run = await _build_simple_graph().astream_events({"value": "x", "items": []}, version="v3")
        _ = await run.output()
        assert "values" in run.extensions and "messages" in run.extensions
        assert run.values is run.extensions["values"]
        assert run.messages is run.extensions["messages"]

    async def test_custom_stream_events(self) -> None:
        run = await _build_custom_stream_graph().astream_events({"value": "x", "items": []}, version="v3", transformers=[_CustomPassthroughTransformer],
        )
        events = [e async for e in run]
        custom_events = [e for e in events if e["method"] == "custom"]
        assert len(custom_events) == 2
        assert custom_events[0]["params"]["data"] == {"step": "start"}
        assert custom_events[1]["params"]["data"] == {"step": "end"}


@pytest.mark.anyio
@NEEDS_CONTEXTVARS
class TestStreamV2AsyncErrors:
    async def test_error_propagation_output(self) -> None:
        run = await _build_error_graph().astream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(ValueError, match="boom"):
            await run.output()

    async def test_error_propagation_values(self) -> None:
        run = await _build_error_graph().astream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(ValueError, match="boom"):
            async for _ in run.values:
                pass

    async def test_error_propagation_raw_events(self) -> None:
        run = await _build_error_graph().astream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(ValueError, match="boom"):
            async for _ in run:
                pass

    async def test_error_propagation_interrupted(self) -> None:
        run = await _build_error_graph().astream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(ValueError, match="boom"):
            await run.interrupted()

    async def test_error_propagation_interrupts(self) -> None:
        run = await _build_error_graph().astream_events({"value": "x", "items": []}, version="v3")
        with pytest.raises(ValueError, match="boom"):
            await run.interrupts()


@pytest.mark.anyio
@NEEDS_CONTEXTVARS
class TestStreamV2AsyncInterrupt:
    async def test_interrupted(self) -> None:
        run = await _build_interrupt_graph().astream_events({"value": "x", "items": []}, {"configurable": {"thread_id": "t2"}}, version="v3")
        _ = await run.output()
        assert await run.interrupted() is True
        assert len(await run.interrupts()) > 0


# ---------------------------------------------------------------------------
# convert_to_protocol_event unit tests
# ---------------------------------------------------------------------------


class TestConvertToProtocolEvent:
    def test_basic_conversion(self) -> None:
        before = int(time.time() * 1000)
        event = convert_to_protocol_event(
            {"type": "values", "ns": ("sub", "graph"), "data": {"key": "val"}}
        )
        after = int(time.time() * 1000)
        assert event["type"] == "event"
        assert event["method"] == "values"
        assert event["params"]["namespace"] == ["sub", "graph"]
        assert event["params"]["data"] == {"key": "val"}
        assert "interrupts" not in event["params"]
        assert before <= event["params"]["timestamp"] <= after

    def test_conversion_with_interrupts(self) -> None:
        event = convert_to_protocol_event(
            {
                "type": "values",
                "ns": (),
                "data": {"k": 1},
                "interrupts": ({"value": "pause"},),
            }
        )
        assert event["params"]["interrupts"] == ({"value": "pause"},)

    def test_namespace_tuple_becomes_list(self) -> None:
        event = convert_to_protocol_event(
            {"type": "updates", "ns": ("a", "b", "c"), "data": {}}
        )
        assert event["params"]["namespace"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# StreamMux unit tests
# ---------------------------------------------------------------------------


class TestStreamMux:
    def test_register_non_dict_raises(self) -> None:
        class BadTransformer(StreamTransformer):
            def init(self) -> Any:
                return ["not", "a", "dict"]

            def process(self, event: ProtocolEvent) -> bool:
                return True

        with pytest.raises(TypeError, match="must return a dict"):
            StreamMux([BadTransformer()])

    def test_event_suppression(self) -> None:
        class FilterTransformer(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                return event["method"] != "updates"

        mux = StreamMux([FilterTransformer()])
        it = iter(mux._events)
        mux.push(_event("values", {"a": 1}))
        mux.push(_event("updates", {"b": 2}))
        mux.push(_event("custom", {"c": 3}))
        mux.close()
        assert [e["method"] for e in it] == ["values", "custom"]

    def test_suppression_all_transformers_still_see_event(self) -> None:
        """If any transformer returns False, the event is suppressed from the main
        log, but all transformers still receive it."""
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
        assert seen_by_second == ["values"]
        assert list(mux._events) == []

    def test_empty_mux(self) -> None:
        mux = StreamMux()
        it = iter(mux._events)
        mux.push(_event("values", {"x": 1}))
        mux.close()
        events = list(it)
        assert len(events) == 1
        assert events[0]["method"] == "values"

    def test_empty_mux_fail(self) -> None:
        mux = StreamMux()
        mux.fail(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            list(mux._events)


# ---------------------------------------------------------------------------
# ValuesTransformer / MessagesTransformer unit tests
# ---------------------------------------------------------------------------


class TestValuesTransformer:
    def test_ignores_non_root_namespace(self) -> None:
        t = ValuesTransformer()
        t.init()
        t._log._bind(is_async=False)
        it = iter(t._log)
        t.process(_event("values", {"val": "root"}))
        t.process(_event("values", {"val": "sub"}, namespace=["sub"]))
        t._log.close()
        items = list(it)
        assert len(items) == 1
        assert items[0]["val"] == "root"

    def test_ignores_non_values_methods(self) -> None:
        t = ValuesTransformer()
        t.init()
        t._log._bind(is_async=False)
        it = iter(t._log)
        assert t.process(_event("updates", {"x": 1})) is True
        t._log.close()
        assert list(it) == []

    def test_tracks_interrupts(self) -> None:
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


class TestOutputWithoutValuesTransformer:
    """run.output / run.interrupted / run.interrupts must work even when
    ValuesTransformer is not registered."""

    def _stream_part(
        self, method: str, data: Any, namespace: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        return {"type": method, "ns": namespace, "data": data}

    def test_output_without_values_transformer(self) -> None:
        mux = StreamMux(factories=[MessagesTransformer], is_async=False)
        run = GraphRunStream(
            iter([self._stream_part("values", {"v": "final"})]),
            mux,
        )
        assert "values" not in run.extensions
        assert run.output == {"v": "final"}

    def test_interrupts_without_values_transformer(self) -> None:
        part = self._stream_part("values", {"v": 1})
        part["interrupts"] = ({"value": "pause"},)
        mux = StreamMux(factories=[MessagesTransformer], is_async=False)
        run = GraphRunStream(iter([part]), mux)
        assert run.interrupted is True
        assert len(run.interrupts) == 1

    @pytest.mark.anyio
    async def test_async_output_without_values_transformer(self) -> None:
        async def _parts() -> Any:
            yield {"type": "values", "ns": (), "data": {"v": "async_final"}}

        mux = StreamMux(factories=[MessagesTransformer], is_async=True)
        run = AsyncGraphRunStream(_parts(), mux)
        assert "values" not in run.extensions
        assert await run.output() == {"v": "async_final"}


class TestMessagesTransformer:
    def test_captures_root_messages(self) -> None:
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
        assert hasattr(items[0], "dispatch")
        assert items[0].message_id == "run-1"

    def test_ignores_non_root_namespace(self) -> None:
        t = MessagesTransformer()
        t.init()
        t._log._bind(is_async=False)
        t._bind_pump(lambda: False)
        it = iter(t._log)
        meta = {"langgraph_node": "llm", "run_id": "run-1"}
        t.process(
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
        assert t.process(_event("values", {"v": 1})) is True
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
# StreamMux resilience: close/fail continue cleanup on transformer errors
# ---------------------------------------------------------------------------


class TestStreamMuxResilience:
    def test_close_continues_after_finalize_error(self) -> None:
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

    def test_channels_closed_after_finalize_error(self) -> None:
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
        assert t._channel._closed


# ---------------------------------------------------------------------------
# Custom transformer tests
# ---------------------------------------------------------------------------


class TestCustomTransformer:
    def test_extension_transformer_with_stream_channel(self) -> None:
        class CounterTransformer(StreamTransformer):
            def __init__(self, scope: tuple[str, ...] = ()) -> None:
                super().__init__(scope)
                self._channel: StreamChannel[int] = StreamChannel("counter")
                self._count = 0

            def init(self) -> dict[str, Any]:
                return {"counter": self._channel}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":
                    self._count += 1
                    self._channel.push(self._count)
                return True

        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3", transformers=[CounterTransformer]
        )
        assert "counter" in run.extensions
        counter_iter = iter(run.extensions["counter"])
        _ = run.output
        counts = list(counter_iter)
        assert len(counts) > 0
        assert not hasattr(run, "counter")  # non-native: no direct attribute

    def test_native_transformer_gets_direct_attr(self) -> None:
        class FooTransformer(StreamTransformer):
            _native = True

            def __init__(self, scope: tuple[str, ...] = ()) -> None:
                super().__init__(scope)
                self._log: StreamChannel[str] = StreamChannel()

            def init(self) -> dict[str, Any]:
                return {"foo": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":
                    self._log.push("saw_values")
                return True

        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3", transformers=[FooTransformer]
        )
        foo_iter = iter(run.foo)
        _ = run.output
        assert "foo" in run.extensions and run.foo is run.extensions["foo"]
        assert "saw_values" in list(foo_iter)

    def test_stream_events_v3_rejects_transformer_instances(self) -> None:
        class InstanceTransformer(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                return True

        with pytest.raises(TypeError, match="pre-built instance"):
            _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3", transformers=[InstanceTransformer()]
            )

    def test_stream_channel_auto_forward(self) -> None:
        """StreamChannel pushes inject ProtocolEvents into the main log."""

        class EmitterTransformer(StreamTransformer):
            def __init__(self, scope: tuple[str, ...] = ()) -> None:
                super().__init__(scope)
                self._channel: StreamChannel[str] = StreamChannel("emitter")

            def init(self) -> dict[str, Any]:
                return {"emitter": self._channel}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":
                    self._channel.push("emitted")
                return True

        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3", transformers=[EmitterTransformer]
        )
        custom_events = [e for e in run if e["method"] == "custom:emitter"]
        assert len(custom_events) > 0
        assert custom_events[0]["params"]["data"] == "emitted"

    def test_stream_channel_seq_ordering(self) -> None:
        """Seq numbers must be monotonically increasing even when a channel push
        auto-forwards an event mid-pipeline."""

        class ChannelPusher(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._channel: StreamChannel[str] = StreamChannel("ch")

            def init(self) -> dict[str, Any]:
                return {"ch": self._channel}

            def process(self, event: ProtocolEvent) -> bool:
                self._channel.push(f"saw:{event['method']}")
                return True

        mux = StreamMux([ChannelPusher()])
        it = iter(mux._events)
        mux.push(_event("values"))
        mux.push(_event("updates"))
        mux.close()
        seqs = [e["seq"] for e in it]
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], f"Seq out of order at index {i}: {seqs}"

    def test_projection_key_conflict_raises(self) -> None:
        class ConflictTransformer(StreamTransformer):
            def __init__(self, scope: tuple[str, ...] = ()) -> None:
                super().__init__(scope)
                self._log: StreamChannel[str] = StreamChannel()

            def init(self) -> dict[str, Any]:
                return {"values": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                return True

        with pytest.raises(ValueError, match=r"conflict.*'values'.*ValuesTransformer"):
            _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3", transformers=[ConflictTransformer]
            )


# ---------------------------------------------------------------------------
# StreamChannel auto-lifecycle via StreamMux
# ---------------------------------------------------------------------------


class TestStreamChannelAutoLifecycle:
    def test_mux_auto_closes_channels(self) -> None:
        class SimpleTransformer(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._log: StreamChannel[str] = StreamChannel()

            def init(self) -> dict[str, Any]:
                return {"items": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                self._log.push("saw_event")
                return True

        mux = StreamMux([SimpleTransformer()])
        it = iter(mux._events)
        mux.push(_event("values"))
        mux.close()
        assert len(list(it)) == 1

    def test_mux_auto_fails_channels(self) -> None:
        class SimpleTransformer(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._log: StreamChannel[str] = StreamChannel()

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
        with pytest.raises(ValueError, match="boom"):
            list(it)

    def test_no_double_close_if_transformer_closes_own_log(self) -> None:
        class ManualCloseTransformer(StreamTransformer):
            def __init__(self) -> None:
                super().__init__()
                self._log: StreamChannel[str] = StreamChannel()

            def init(self) -> dict[str, Any]:
                return {"items": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                return True

            def finalize(self) -> None:
                self._log.close()

        mux = StreamMux([ManualCloseTransformer()])
        mux.close()  # should not raise even with double-close

    def test_transformer_without_finalize_works(self) -> None:
        class MinimalTransformer(StreamTransformer):
            def __init__(self, scope: tuple[str, ...] = ()) -> None:
                super().__init__(scope)
                self._log: StreamChannel[str] = StreamChannel()

            def init(self) -> dict[str, Any]:
                return {"minimal": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":
                    self._log.push("got_it")
                return True

        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3", transformers=[MinimalTransformer]
        )
        minimal_iter = iter(run.extensions["minimal"])
        _ = run.output
        assert len(list(minimal_iter)) > 0


class TestStreamTransformerSchedule:
    def test_schedule_without_running_loop_raises(self) -> None:
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


# ---------------------------------------------------------------------------
# Async transformer lane
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestAsyncTransformerLane:
    async def test_aprocess_is_awaited_before_next_transformer(self) -> None:
        """aprocess must complete before the next transformer sees the event —
        load-bearing guarantee for mutating transformers like PII redaction."""
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

    async def test_schedule_joins_tasks_before_afinalize(self) -> None:
        """Every scheduled task must complete before afinalize runs."""
        phase: list[str] = []

        class SchedTransformer(StreamTransformer):
            requires_async = True

            def __init__(self) -> None:
                self._log: StreamChannel[str] = StreamChannel()

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
        assert phase.count("task") == 2
        assert phase[-1] == "afinalize"

    async def test_sync_stream_rejects_async_transformer(self) -> None:
        class NeedsAsync(StreamTransformer):
            requires_async = True

            def init(self) -> dict[str, Any]:
                return {}

            def process(self, event: ProtocolEvent) -> bool:
                return True

        with pytest.raises(RuntimeError, match="requires an async run"):
            StreamMux([NeedsAsync()], is_async=False)

    async def test_sync_stream_rejects_aprocess_override(self) -> None:
        class HasAprocess(StreamTransformer):
            def init(self) -> dict[str, Any]:
                return {}

            async def aprocess(self, event: ProtocolEvent) -> bool:
                return True

        with pytest.raises(RuntimeError, match="requires an async run"):
            StreamMux([HasAprocess()], is_async=False)

    async def test_schedule_on_error_log_swallows_exceptions(self) -> None:
        class Bad(StreamTransformer):
            requires_async = True

            def __init__(self) -> None:
                self._log: StreamChannel[str] = StreamChannel()

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
        await mux.aclose()  # should not raise; exception is logged

    async def test_schedule_on_error_raise_fails_the_run(self) -> None:
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

    async def test_afail_cancels_pending_scheduled_tasks(self) -> None:
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
        # Yield so the task actually starts before we cancel it.
        await asyncio.sleep(0)
        await mux.afail(RuntimeError("run died"))
        assert cancelled.is_set()

    async def test_mixed_sync_and_async_transformers(self) -> None:
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
                self._log: StreamChannel[str] = StreamChannel()

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
        assert [x async for x in seen_cursor] == ["values", "updates"]

    async def test_handler_astream_with_scheduled_work(self) -> None:
        class Scorer(StreamTransformer):
            requires_async = True

            def __init__(self, scope: tuple[str, ...] = ()) -> None:
                super().__init__(scope)
                self._log: StreamChannel[int] = StreamChannel()

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

        run = await _build_simple_graph().astream_events({"value": "x", "items": []}, version="v3", transformers=[Scorer]
        )
        scores_cursor = aiter(run.extensions["scores"])
        _ = await run.output()
        scores = [x async for x in scores_cursor]
        assert scores and all(s == 42 for s in scores)


# ---------------------------------------------------------------------------
# Memory bounds: drain-on-consume semantics
# ---------------------------------------------------------------------------


@NEEDS_CONTEXTVARS
class TestMemoryBounds:
    def test_sync_subscribed_buffer_stays_at_most_one_between_yields(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        events_iter = iter(run)
        max_buffered = 0
        count = 0
        for _ in events_iter:
            max_buffered = max(max_buffered, len(run._mux._events._items))
            count += 1
        assert count > 0
        assert max_buffered == 0, (
            f"drain-on-consume violated, observed max {max_buffered}"
        )

    def test_unsubscribed_projections_never_accumulate(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        list(run)
        values_log = run.extensions["values"]
        messages_log = run.extensions["messages"]
        assert len(values_log._items) == 0 and not values_log._subscribed
        assert len(messages_log._items) == 0 and not messages_log._subscribed

    def test_output_path_does_not_retain_values(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        _ = run.output
        values_log = run.extensions["values"]
        assert len(values_log._items) == 0 and not values_log._subscribed

    def test_drained_subscriber_buffer_returns_to_empty(self) -> None:
        run = _build_simple_graph().stream_events({"value": "x", "items": []}, version="v3")
        list(run.values)
        assert len(run.extensions["values"]._items) == 0

    @pytest.mark.anyio
    async def test_async_single_consumer_buffer_stays_at_most_one(self) -> None:
        run = await _build_simple_graph().astream_events({"value": "x", "items": []}, version="v3")
        max_buffered = 0
        count = 0
        async for _ in run:
            max_buffered = max(max_buffered, len(run._mux._events._items))
            count += 1
        assert count > 0
        assert max_buffered == 0

    @pytest.mark.anyio
    async def test_async_unsubscribed_projections_never_accumulate(self) -> None:
        run = await _build_simple_graph().astream_events({"value": "x", "items": []}, version="v3")
        _ = await run.output()
        values_log = run.extensions["values"]
        messages_log = run.extensions["messages"]
        assert len(values_log._items) == 0 and not values_log._subscribed
        assert len(messages_log._items) == 0 and not messages_log._subscribed


# ---------------------------------------------------------------------------
# DrainOnConsume: StreamChannel capacity semantics
# ---------------------------------------------------------------------------


class TestDrainOnConsume:
    def test_invalid_maxlen_raises(self) -> None:
        with pytest.raises(ValueError, match="positive int or None"):
            StreamChannel(maxlen=0)
        with pytest.raises(ValueError, match="positive int or None"):
            StreamChannel(maxlen=-3)

    def test_push_unbounded_by_design(self) -> None:
        """Push is non-blocking; the caller-driven pump bounds memory via iteration pace."""
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=False)
        it = iter(log)
        for i in range(100):
            log.push(i)
        log.close()
        assert list(it) == list(range(100))

    def test_tee_fans_out_sync(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=False)
        a, b = log.tee(2)
        for i in range(3):
            log.push(i)
        log.close()
        assert list(a) == [0, 1, 2]
        assert list(b) == [0, 1, 2]

    @pytest.mark.anyio
    async def test_atee_fans_out(self) -> None:
        log: StreamChannel[int] = StreamChannel()
        log._bind(is_async=True)
        a, b = log.atee(2)
        for i in range(3):
            log.push(i)
        log.close()
        assert [x async for x in a] == [0, 1, 2]
        assert [x async for x in b] == [0, 1, 2]
