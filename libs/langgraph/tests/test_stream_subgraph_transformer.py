"""Tests for SubgraphTransformer.

Subscribes to `tasks` events and produces in-process `SubgraphRunStream`
handles backed by mini-muxes (built via `StreamMux.make_child`). The
synthetic-event tests isolate the inference / mini-mux wiring; the
real-graph tests exercise the end-to-end navigation path through
`stream_v2`.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any

from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.errors import GraphInterrupt
from langgraph.graph import StateGraph
from langgraph.stream._mux import StreamMux
from langgraph.stream.run_stream import SubgraphRunStream
from langgraph.stream.transformers import (
    LifecycleTransformer,
    MessagesTransformer,
    SubgraphTransformer,
    ValuesTransformer,
)

TS = int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tasks_start(
    namespace: list[str],
    *,
    task_id: str,
    name: str,
) -> dict[str, Any]:
    return {
        "type": "event",
        "method": "tasks",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": {
                "id": task_id,
                "name": name,
                "input": None,
                "triggers": [],
            },
        },
    }


def _tasks_result(
    namespace: list[str],
    *,
    task_id: str,
    name: str,
    error: str | None = None,
    interrupts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "type": "event",
        "method": "tasks",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": {
                "id": task_id,
                "name": name,
                "error": error,
                "interrupts": interrupts or [],
                "result": {},
            },
        },
    }


def _native_factories() -> list[Any]:
    """Mirror the factory list `Pregel.stream_v2` registers."""
    return [
        ValuesTransformer,
        MessagesTransformer,
        LifecycleTransformer,
        SubgraphTransformer,
    ]


def _arm(mux: StreamMux) -> None:
    """Pre-subscribe every projection in the mux so synthetic pushes accumulate.

    Real consumer code subscribes by iterating the projection; tests
    inspect `_items` directly, so the lazy-subscribe gate has to be
    flipped manually before any synthetic events are pushed.
    """
    mux._events._subscribed = True
    for value in mux.extensions.values():
        if hasattr(value, "_subscribed"):  # EventLog
            value._subscribed = True
        elif hasattr(value, "_log"):  # StreamChannel
            value._log._subscribed = True


def _arm_recursive(mux: StreamMux) -> None:
    """Arm `mux` and every mini-mux currently held by SubgraphTransformer handles.

    Mini-muxes are created during `mux.push(...)` when a new direct
    child is discovered. Tests must call this after each push that
    might have created a new mini-mux so subsequent pushes' projection
    side effects accumulate (rather than dropping silently against an
    unsubscribed log).
    """
    _arm(mux)
    sub_t = mux.transformer_by_key("subgraphs")
    if isinstance(sub_t, SubgraphTransformer):
        for handle in sub_t._handles.values():
            if handle._mux is not None:
                _arm_recursive(handle._mux)


def _build_root_mux(*, scope: tuple[str, ...] = ()) -> StreamMux:
    mux = StreamMux(
        factories=_native_factories(),
        scope=scope,
        is_async=False,
    )
    _arm(mux)
    return mux


def _drain_subgraphs(mux: StreamMux) -> list[SubgraphRunStream]:
    transformer = mux.transformer_by_key("subgraphs")
    assert isinstance(transformer, SubgraphTransformer)
    return list(transformer._log._items)


# ---------------------------------------------------------------------------
# Synthetic-event tests
# ---------------------------------------------------------------------------


def test_handle_created_on_first_direct_child_task() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    [handle] = _drain_subgraphs(mux)
    assert handle.path == ("agent:abc",)
    assert handle.graph_name == "agent"
    assert handle.trigger_call_id == "abc"
    assert handle.status == "started"
    assert handle._mux is not None  # mini-mux backed


def test_handle_status_completes_on_parent_result() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent"))

    [handle] = _drain_subgraphs(mux)
    assert handle.status == "completed"
    assert handle.error is None


def test_handle_status_failed_with_error() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent", error="boom"))

    [handle] = _drain_subgraphs(mux)
    assert handle.status == "failed"
    assert handle.error == "boom"


def test_handle_status_interrupted() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(
        _tasks_result(
            [],
            task_id="abc",
            name="agent",
            interrupts=[{"value": "pause"}],
        )
    )

    [handle] = _drain_subgraphs(mux)
    assert handle.status == "interrupted"


def test_grandchild_discovered_via_child_mini_mux() -> None:
    """Each mini-mux owns its own scope; grandchildren live on the child handle."""
    mux = _build_root_mux()
    # Direct child started — creates the mini-mux.
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    # Pre-subscribe the freshly-created mini-mux so subsequent
    # forwarded events land on its projections (consumer would
    # subscribe naturally by iterating handle.subgraphs, but the
    # test inspects `_items` directly).
    _arm_recursive(mux)
    # Grandchild's first task event flows down into the child mini-mux.
    mux.push(_tasks_start(["agent:abc", "tool:def"], task_id="t2", name="deep"))

    [child_handle] = _drain_subgraphs(mux)
    assert child_handle.path == ("agent:abc",)
    # The grandchild appears on the CHILD'S subgraphs projection.
    grandchildren = list(child_handle.subgraphs._items)
    assert len(grandchildren) == 1
    assert grandchildren[0].path == ("agent:abc", "tool:def")


def test_finalize_completes_open_handles() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.close()
    [handle] = _drain_subgraphs(mux)
    assert handle.status == "completed"


def test_fail_marks_open_handles_interrupted_for_graph_interrupt() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.fail(GraphInterrupt())
    [handle] = _drain_subgraphs(mux)
    assert handle.status == "interrupted"


def test_fail_marks_open_handles_failed_for_other_errors() -> None:
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.fail(RuntimeError("boom"))
    [handle] = _drain_subgraphs(mux)
    assert handle.status == "failed"
    assert handle.error == "boom"


def test_make_child_raises_without_factories() -> None:
    """A mux constructed only from `transformers=` can't clone factories."""
    transformer = SubgraphTransformer()
    mux = StreamMux(transformers=[transformer], is_async=False)
    try:
        mux.make_child(("anything",))
    except RuntimeError as exc:
        assert "factories" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")


def test_subgraph_and_lifecycle_agree_on_terminal_status() -> None:
    """Both transformers consume the same tasks signal — no drift."""
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent", error="boom"))

    [handle] = _drain_subgraphs(mux)
    lifecycle_t = mux.transformer_by_key("lifecycle")
    assert isinstance(lifecycle_t, LifecycleTransformer)
    payloads = list(lifecycle_t._channel._log._items)
    assert handle.status == "failed"
    assert payloads[-1]["event"] == "failed"
    assert handle.error == payloads[-1]["error"]


def test_required_stream_modes_declared() -> None:
    assert SubgraphTransformer.required_stream_modes == ("tasks",)


def test_tasks_events_suppressed_from_main_log() -> None:
    """Tasks events are folded into discovery and don't appear on the main log."""
    mux = _build_root_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent"))

    methods = [evt["method"] for evt in mux._events._items]
    assert "tasks" not in methods


# ---------------------------------------------------------------------------
# End-to-end real-graph tests
# ---------------------------------------------------------------------------


class _State(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def _passthrough(state: _State) -> dict[str, Any]:
    return {"value": state["value"] + "!", "items": ["x"]}


def _make_two_level_nested() -> Any:
    """outer → middle → inner. Three Pregel instances, two nesting levels."""
    inner_b: StateGraph = StateGraph(_State, input_schema=_State)
    inner_b.add_node("inner_node", _passthrough)
    inner_b.add_edge(START, "inner_node")
    inner_b.add_edge("inner_node", END)
    inner = inner_b.compile()

    middle_b: StateGraph = StateGraph(_State, input_schema=_State)
    middle_b.add_node("inner", inner)
    middle_b.add_edge(START, "inner")
    middle_b.add_edge("inner", END)
    middle = middle_b.compile()

    outer_b: StateGraph = StateGraph(_State, input_schema=_State)
    outer_b.add_node("middle", middle)
    outer_b.add_edge(START, "middle")
    outer_b.add_edge("middle", END)
    return outer_b.compile()


def test_stream_v2_real_graph_yields_subgraph_handles() -> None:
    """Iterating `run.subgraphs` yields handles for direct-child subgraphs."""
    graph = _make_two_level_nested()
    run = graph.stream_v2({"value": "x", "items": []})

    handle_paths: list[tuple[str, ...]] = []
    final_status: dict[tuple[str, ...], str] = {}
    for handle in run.subgraphs:
        # Drill into the handle's projections inside the loop body so
        # the mini-mux is subscribed before the next pump cycle.
        list(handle.values)
        handle_paths.append(handle.path)
        final_status[handle.path] = handle.status

    assert len(handle_paths) == 1
    assert handle_paths[0][0].startswith("middle:")
    assert final_status[handle_paths[0]] == "completed"


def test_stream_v2_grandchild_visible_on_child_handle() -> None:
    """Drilling into `handle.subgraphs` surfaces nested grandchildren."""
    graph = _make_two_level_nested()
    run = graph.stream_v2({"value": "x", "items": []})

    grandchild_paths: list[tuple[str, ...]] = []
    middle_path: tuple[str, ...] | None = None
    for middle_handle in run.subgraphs:
        # Subscribe to grandchildren before the next pump cycle.
        for inner_handle in middle_handle.subgraphs:
            # Subscribe to inner.values so its mini-mux drains.
            list(inner_handle.values)
            grandchild_paths.append(inner_handle.path)
        middle_path = middle_handle.path

    assert middle_path is not None
    assert len(grandchild_paths) == 1
    inner_path = grandchild_paths[0]
    assert inner_path[1].startswith("inner:")
    assert inner_path[: len(middle_path)] == middle_path
