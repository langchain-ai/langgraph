"""Tests for SubgraphTransformer namespace-based discovery."""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.errors import GraphInterrupt
from langgraph.graph import StateGraph
from langgraph.stream._event_log import EventLog
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.transformers import (
    MessagesTransformer,
    SubgraphRunStream,
    SubgraphTransformer,
    ValuesTransformer,
)
from langgraph.types import interrupt

TS = int(time.time() * 1000)


def _values(payload: dict[str, Any], *, namespace: list[str]) -> ProtocolEvent:
    return {
        "type": "event",
        "method": "values",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": payload,
        },
    }


def _subscribe(log: EventLog) -> None:
    """Flip `_subscribed = True` so pushes retain items for test inspection."""
    log._subscribed = True


# ---------------------------------------------------------------------------
# Unit tests: feed events directly into the transformer
# ---------------------------------------------------------------------------


_FACTORIES = [ValuesTransformer, MessagesTransformer, SubgraphTransformer]


def _handle_values_items(handle: SubgraphRunStream) -> list:
    return list(handle._mux.extensions["values"]._items)  # type: ignore[attr-defined]


def _handle_subgraphs_items(handle: SubgraphRunStream) -> list:
    return list(handle._mux.extensions["subgraphs"]._items)  # type: ignore[attr-defined]


def _pre_subscribe_handle(handle: SubgraphRunStream) -> None:
    """Flip `_subscribed` on every EventLog inside the handle's mini-mux.

    The mini-mux is built via `make_child` with the full factory list,
    so values / messages / subgraphs logs all exist as projections.
    Tests that feed events directly need them subscribed so pushes
    retain items in the deque for `_items` inspection.
    """
    for value in handle._mux.extensions.values():
        if isinstance(value, EventLog):
            _subscribe(value)


class TestSubgraphTransformerUnit:
    def _mux(self) -> tuple[StreamMux, SubgraphTransformer]:
        mux = StreamMux(factories=_FACTORIES, is_async=False)
        transformer = mux.transformer_by_key("subgraphs")
        assert isinstance(transformer, SubgraphTransformer)
        _subscribe(transformer._root_log)
        return mux, transformer

    def _handle(self, transformer: SubgraphTransformer) -> SubgraphRunStream:
        (handle,) = list(transformer._root_log._items)
        return handle

    def test_root_event_does_not_create_handle(self) -> None:
        mux, transformer = self._mux()
        mux.push(_values({"value": 1}, namespace=[]))
        assert list(transformer._root_log._items) == []
        assert transformer._by_ns == {}

    def test_first_event_at_child_depth_yields_handle(self) -> None:
        mux, transformer = self._mux()
        mux.push(_values({"value": 1}, namespace=["child:task_a"]))

        handle = self._handle(transformer)
        assert handle.path == ("child:task_a",)
        assert handle.graph_name == "child"
        assert handle.trigger_call_id == "task_a"
        assert handle.status == "started"

    def test_handle_without_task_id_suffix(self) -> None:
        mux, transformer = self._mux()
        mux.push(_values({"value": 1}, namespace=["child"]))

        handle = self._handle(transformer)
        assert handle.path == ("child",)
        assert handle.graph_name == "child"
        assert handle.trigger_call_id is None

    def test_discovery_is_method_agnostic(self) -> None:
        mux, transformer = self._mux()
        # Method-agnostic means "any event method triggers discovery".
        # Use `updates` — neither ValuesTransformer nor
        # MessagesTransformer care about it, so the test only exercises
        # SubgraphTransformer's discovery path.
        mux.push(
            {
                "type": "event",
                "method": "updates",
                "params": {"namespace": ["c:t"], "timestamp": TS, "data": "x"},
            }
        )
        handle = self._handle(transformer)
        assert handle.path == ("c:t",)

    def test_grandchild_surfaces_under_child(self) -> None:
        mux, transformer = self._mux()
        mux.push(_values({"value": 1}, namespace=["child:t"]))

        child = self._handle(transformer)
        _pre_subscribe_handle(child)

        mux.push(_values({"value": 2}, namespace=["child:t", "grand:u"]))

        (grand,) = _handle_subgraphs_items(child)
        assert grand.path == ("child:t", "grand:u")
        assert grand.graph_name == "grand"
        assert grand.trigger_call_id == "u"

    def test_values_routed_into_handle(self) -> None:
        mux, transformer = self._mux()
        mux.push(_values({"value": 1}, namespace=["c:t"]))

        handle = self._handle(transformer)
        _pre_subscribe_handle(handle)

        mux.push(_values({"value": 2}, namespace=["c:t"]))

        # Both the discovery event and subsequent values land in the child.
        assert _handle_values_items(handle) == [{"value": 1}, {"value": 2}]
        assert handle.output == {"value": 2}

    def test_root_values_not_routed(self) -> None:
        mux, transformer = self._mux()
        mux.push(_values({"value": 1}, namespace=["c:t"]))
        handle = self._handle(transformer)
        _pre_subscribe_handle(handle)

        # Values event at root namespace — must not leak into child handle.
        mux.push(_values({"value": "root"}, namespace=[]))
        assert _handle_values_items(handle) == [{"value": 1}]

    def test_finalize_closes_dangling(self) -> None:
        mux, transformer = self._mux()
        mux.push(_values({"value": 1}, namespace=["c:t"]))
        handle = self._handle(transformer)

        mux.close()
        assert handle.status == "completed"
        assert handle._mux.extensions["values"]._closed
        assert handle._mux.extensions["subgraphs"]._closed

    def test_fail_with_graph_interrupt_marks_interrupted(self) -> None:
        mux, transformer = self._mux()
        mux.push(_values({"value": 1}, namespace=["c:t"]))
        handle = self._handle(transformer)

        mux.fail(GraphInterrupt())
        assert handle.status == "interrupted"

    def test_fail_with_generic_error_marks_failed(self) -> None:
        mux, transformer = self._mux()
        mux.push(_values({"value": 1}, namespace=["c:t"]))
        handle = self._handle(transformer)

        mux.fail(RuntimeError("explode"))
        assert handle.status == "failed"
        assert handle.error == "explode"

    def test_repeated_events_same_ns_single_handle(self) -> None:
        mux, transformer = self._mux()
        mux.push(_values({"value": 1}, namespace=["c:t"]))
        mux.push(_values({"value": 2}, namespace=["c:t"]))
        mux.push(_values({"value": 3}, namespace=["c:t"]))

        handles = list(transformer._root_log._items)
        assert len(handles) == 1
        assert handles[0].path == ("c:t",)


# ---------------------------------------------------------------------------
# End-to-end tests via stream_v2 on real graphs
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def _build_nested_graph():
    """Parent graph with a compiled subgraph node."""

    def inner_node(state: SimpleState) -> dict:
        return {"value": state["value"] + "X", "items": ["x"]}

    inner_builder = StateGraph(SimpleState)
    inner_builder.add_node("inner_node", inner_node)
    inner_builder.add_edge(START, "inner_node")
    inner_builder.add_edge("inner_node", END)
    inner = inner_builder.compile()

    def outer_node(state: SimpleState) -> dict:
        return {"value": state["value"] + "Y", "items": ["y"]}

    outer_builder = StateGraph(SimpleState)
    outer_builder.add_node("outer_node", outer_node)
    outer_builder.add_node("sub", inner)
    outer_builder.add_edge(START, "outer_node")
    outer_builder.add_edge("outer_node", "sub")
    outer_builder.add_edge("sub", END)
    return outer_builder.compile()


class TestSubgraphTransformerEndToEnd:
    def test_flat_graph_yields_no_subgraphs(self) -> None:
        builder = StateGraph(SimpleState)
        builder.add_node("n", lambda s: {"value": s["value"] + "!", "items": ["!"]})
        builder.add_edge(START, "n")
        builder.add_edge("n", END)
        graph = builder.compile()

        run = graph.stream_v2({"value": "", "items": []})

        collected: list[SubgraphRunStream] = []
        for sub in run.subgraphs:
            collected.append(sub)
        assert collected == []
        # Output still resolves.
        assert run.output is not None

    def test_nested_graph_yields_one_child(self) -> None:
        graph = _build_nested_graph()
        run = graph.stream_v2({"value": "", "items": []})

        collected: list[SubgraphRunStream] = []
        for sub in run.subgraphs:
            collected.append(sub)

        assert len(collected) == 1
        child = collected[0]
        assert len(child.path) == 1
        assert child.path[0].startswith("sub:")
        assert child.status == "completed"

    def test_error_in_subgraph_fails_child(self) -> None:
        def boom(state: SimpleState) -> dict:
            raise RuntimeError("subgraph_failed")

        inner_builder = StateGraph(SimpleState)
        inner_builder.add_node("inner", boom)
        inner_builder.add_edge(START, "inner")
        inner_builder.add_edge("inner", END)
        inner = inner_builder.compile()

        outer_builder = StateGraph(SimpleState)
        outer_builder.add_node("sub", inner)
        outer_builder.add_edge(START, "sub")
        outer_builder.add_edge("sub", END)
        graph = outer_builder.compile()

        run = graph.stream_v2({"value": "", "items": []})

        collected: list[SubgraphRunStream] = []
        with pytest.raises(RuntimeError):
            for sub in run.subgraphs:
                collected.append(sub)

        assert len(collected) == 1
        assert collected[0].status == "failed"


class TestSubgraphTransformerAsyncEndToEnd:
    @pytest.mark.anyio
    async def test_nested_graph_yields_one_child(self) -> None:
        async def inner(state: SimpleState) -> dict:
            return {"value": state["value"] + "X", "items": ["x"]}

        inner_builder = StateGraph(SimpleState)
        inner_builder.add_node("inner", inner)
        inner_builder.add_edge(START, "inner")
        inner_builder.add_edge("inner", END)
        inner_graph = inner_builder.compile()

        outer_builder = StateGraph(SimpleState)
        outer_builder.add_node("sub", inner_graph)
        outer_builder.add_edge(START, "sub")
        outer_builder.add_edge("sub", END)
        graph = outer_builder.compile()

        run = await graph.astream_v2({"value": "", "items": []})

        collected: list[SubgraphRunStream] = []
        async for sub in run.subgraphs:
            collected.append(sub)

        assert len(collected) == 1
        child = collected[0]
        assert child.status == "completed"


class TestSubgraphTriggerCallId:
    """Confirm `trigger_call_id` flows from real pregel metadata."""

    def test_trigger_call_id_populated_end_to_end(self) -> None:
        graph = _build_nested_graph()
        run = graph.stream_v2({"value": "", "items": []})

        collected: list[SubgraphRunStream] = list(run.subgraphs)
        assert len(collected) == 1
        child = collected[0]

        # The child's single-segment path encodes `node_name:task_id`.
        # Both the parsed task_id (`trigger_call_id`) and the segment
        # should match the same task_id suffix.
        assert ":" in child.path[0]
        node_name, _, task_id = child.path[0].partition(":")
        assert node_name == "sub"
        assert task_id  # non-empty
        assert child.trigger_call_id == task_id


class TestSubgraphInterrupt:
    """Interrupts raised inside a subgraph surface as status=interrupted."""

    def _build_interrupt_subgraph(self):
        def inner_node(state: SimpleState) -> dict:
            interrupt("need approval")
            return {"value": state["value"] + "X", "items": ["x"]}

        inner_builder = StateGraph(SimpleState)
        inner_builder.add_node("inner_node", inner_node)
        inner_builder.add_edge(START, "inner_node")
        inner_builder.add_edge("inner_node", END)
        inner = inner_builder.compile()

        outer_builder = StateGraph(SimpleState)
        outer_builder.add_node("sub", inner)
        outer_builder.add_edge(START, "sub")
        outer_builder.add_edge("sub", END)
        return outer_builder.compile(checkpointer=InMemorySaver())

    def test_interrupt_in_subgraph_marks_handle_interrupted(self) -> None:
        graph = self._build_interrupt_subgraph()
        run = graph.stream_v2(
            {"value": "", "items": []},
            config={"configurable": {"thread_id": "t1"}},
        )

        collected: list[SubgraphRunStream] = list(run.subgraphs)

        assert run.interrupted is True
        assert len(collected) == 1
        assert collected[0].status == "interrupted"


class TestSubgraphNameCollision:
    """The subgraph's compiled `name` equaling its node name is detected.

    Primary detector `name != langgraph_node` fails here; the
    parent_run_id fallback in `_is_nested_pregel_start` is what keeps
    the subgraph visible.
    """

    def test_name_equals_node_name_still_detected(self) -> None:
        def inner_node(state: SimpleState) -> dict:
            return {"value": state["value"] + "X", "items": ["x"]}

        inner_builder = StateGraph(SimpleState)
        inner_builder.add_node("inner_node", inner_node)
        inner_builder.add_edge(START, "inner_node")
        inner_builder.add_edge("inner_node", END)
        # Compile with the same name as the node it will be registered as.
        inner = inner_builder.compile(name="sub")

        outer_builder = StateGraph(SimpleState)
        outer_builder.add_node("sub", inner)
        outer_builder.add_edge(START, "sub")
        outer_builder.add_edge("sub", END)
        graph = outer_builder.compile()

        run = graph.stream_v2({"value": "", "items": []})

        collected: list[SubgraphRunStream] = list(run.subgraphs)
        assert len(collected) == 1
        child = collected[0]
        assert child.graph_name == "sub"
        assert child.status == "completed"
