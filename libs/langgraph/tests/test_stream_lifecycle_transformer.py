"""Tests for LifecycleTransformer — derives subgraph lifecycle from ns discovery."""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any

import pytest
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.errors import GraphInterrupt
from langgraph.graph import StateGraph
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.transformers import (
    LifecycleTransformer,
    ValuesTransformer,
)

TS = int(time.time() * 1000)


def _event(method: str, data: Any, *, namespace: list[str]) -> ProtocolEvent:
    return {
        "type": "event",
        "method": method,
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": data,
        },
    }


def _drain_channel(mux: StreamMux) -> list[dict[str, Any]]:
    ch = mux.extensions["lifecycle"]
    return list(ch._log._items)  # type: ignore[attr-defined]


def _drain_main_events(mux: StreamMux) -> list[ProtocolEvent]:
    return list(mux._events._items)  # type: ignore[attr-defined]


def _subscribe(mux: StreamMux) -> None:
    ch = mux.extensions["lifecycle"]
    ch._log._subscribed = True  # type: ignore[attr-defined]
    mux._events._subscribed = True  # type: ignore[attr-defined]


class TestLifecycleTransformerUnit:
    def _mux(self) -> StreamMux:
        mux = StreamMux(
            factories=[ValuesTransformer, LifecycleTransformer], is_async=False
        )
        _subscribe(mux)
        return mux

    def test_first_event_at_child_ns_emits_started(self) -> None:
        mux = self._mux()
        mux.push(_event("values", {"v": 1}, namespace=["child:task_a"]))

        events = _drain_channel(mux)
        assert events == [
            {
                "event": "started",
                "namespace": ["child:task_a"],
                "graph_name": "child",
                "trigger_call_id": "task_a",
            }
        ]

    def test_no_started_at_root_ns(self) -> None:
        mux = self._mux()
        mux.push(_event("values", {"v": 1}, namespace=[]))
        assert _drain_channel(mux) == []

    def test_method_agnostic_discovery(self) -> None:
        mux = self._mux()
        mux.push(_event("messages", "x", namespace=["c:t"]))

        (started,) = _drain_channel(mux)
        assert started["event"] == "started"
        assert started["namespace"] == ["c:t"]

    def test_repeated_events_single_started(self) -> None:
        mux = self._mux()
        mux.push(_event("values", {"v": 1}, namespace=["c:t"]))
        mux.push(_event("values", {"v": 2}, namespace=["c:t"]))
        mux.push(_event("updates", {"n": "x"}, namespace=["c:t"]))

        events = _drain_channel(mux)
        assert len(events) == 1
        assert events[0]["event"] == "started"

    def test_ns_without_task_id(self) -> None:
        mux = self._mux()
        mux.push(_event("values", {"v": 1}, namespace=["child"]))

        (started,) = _drain_channel(mux)
        assert started["graph_name"] == "child"
        assert "trigger_call_id" not in started

    def test_finalize_emits_completed(self) -> None:
        mux = self._mux()
        mux.push(_event("values", {"v": 1}, namespace=["c:t"]))
        mux.close()

        events = _drain_channel(mux)
        assert events == [
            {
                "event": "started",
                "namespace": ["c:t"],
                "graph_name": "c",
                "trigger_call_id": "t",
            },
            {"event": "completed", "namespace": ["c:t"]},
        ]

    def test_fail_with_graph_interrupt(self) -> None:
        mux = self._mux()
        mux.push(_event("values", {"v": 1}, namespace=["c:t"]))
        mux.fail(GraphInterrupt())

        events = _drain_channel(mux)
        assert events[-1] == {"event": "interrupted", "namespace": ["c:t"]}

    def test_fail_with_generic_error(self) -> None:
        mux = self._mux()
        mux.push(_event("values", {"v": 1}, namespace=["c:t"]))
        mux.fail(RuntimeError("boom"))

        events = _drain_channel(mux)
        assert events[-1] == {
            "event": "failed",
            "namespace": ["c:t"],
            "error": "boom",
        }


class TestLifecycleWireFormat:
    """Native transformer: method on the wire is `"lifecycle"`, no `custom:` prefix."""

    def test_wire_method_is_lifecycle_unprefixed(self) -> None:
        mux = StreamMux(factories=[LifecycleTransformer], is_async=False)
        _subscribe(mux)
        mux.push(_event("values", {"v": 1}, namespace=["c:t"]))

        # Find the lifecycle event in the main log.
        lifecycle_events = [
            ev for ev in _drain_main_events(mux) if ev["method"] == "lifecycle"
        ]
        assert len(lifecycle_events) == 1
        assert lifecycle_events[0]["params"]["data"]["event"] == "started"
        # Also verify no `custom:lifecycle` leaks through.
        assert not any(
            ev["method"].startswith("custom:") for ev in _drain_main_events(mux)
        )

    def test_started_precedes_originating_event_on_wire(self) -> None:
        """Seq ordering: synthesized lifecycle event lands before the event that triggered it."""
        mux = StreamMux(
            factories=[ValuesTransformer, LifecycleTransformer], is_async=False
        )
        _subscribe(mux)
        mux.push(_event("values", {"v": 1}, namespace=["c:t"]))

        wire = _drain_main_events(mux)
        methods = [ev["method"] for ev in wire]
        # Lifecycle's synthetic event is forwarded during process() and
        # gets an earlier seq than the originating values event.
        assert methods.index("lifecycle") < methods.index("values")


# ---------------------------------------------------------------------------
# End-to-end via stream_v2
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def _build_nested_graph():
    def inner_node(state: SimpleState) -> dict:
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
    return outer_builder.compile()


class TestLifecycleEndToEnd:
    def test_real_run_emits_started_and_completed(self) -> None:
        graph = _build_nested_graph()
        run = graph.stream_v2(
            {"value": "", "items": []}, transformers=[LifecycleTransformer]
        )

        # Drain the run so finalize fires.
        list(run.values)

        lifecycle = list(run.lifecycle)  # type: ignore[attr-defined]
        events = [e["event"] for e in lifecycle]
        assert "started" in events
        assert "completed" in events

    def test_real_run_error_emits_failed(self) -> None:
        def boom(state: SimpleState) -> dict:
            raise RuntimeError("kaboom")

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

        run = graph.stream_v2(
            {"value": "", "items": []}, transformers=[LifecycleTransformer]
        )
        with pytest.raises(RuntimeError):
            list(run.values)

        lifecycle = list(run.lifecycle)  # type: ignore[attr-defined]
        events = [e["event"] for e in lifecycle]
        assert "failed" in events

    def test_lifecycle_and_subgraphs_agree(self) -> None:
        """SubgraphTransformer and LifecycleTransformer share the discovery predicate."""
        graph = _build_nested_graph()
        run = graph.stream_v2(
            {"value": "", "items": []}, transformers=[LifecycleTransformer]
        )

        subs = list(run.subgraphs)
        lifecycle = list(run.lifecycle)  # type: ignore[attr-defined]

        sub_paths = {tuple(s.path) for s in subs}
        started_paths = {
            tuple(e["namespace"]) for e in lifecycle if e["event"] == "started"
        }
        assert sub_paths == started_paths
