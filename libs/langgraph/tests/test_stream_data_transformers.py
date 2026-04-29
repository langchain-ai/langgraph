"""Tests for CustomTransformer, UpdatesTransformer, CheckpointsTransformer, DebugTransformer, TasksTransformer.

These transformers capture raw protocol events for their respective stream
modes and expose them as native projections on the run stream (run.custom,
run.updates, run.checkpoints, run.debug, run.tasks). Tests dispatch synthetic
protocol events through a StreamMux to isolate transformer logic; the final
group exercises real graphs through stream_v2.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any

from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.stream._mux import StreamMux
from langgraph.stream.stream_channel import StreamChannel
from langgraph.stream.transformers import (
    CheckpointsTransformer,
    CustomTransformer,
    DebugTransformer,
    LifecycleTransformer,
    TasksTransformer,
    UpdatesTransformer,
)

TS = int(time.time() * 1000)


def _custom_event(namespace: list[str], data: Any) -> dict[str, Any]:
    return {
        "type": "event",
        "method": "custom",
        "params": {"namespace": namespace, "timestamp": TS, "data": data},
    }


def _checkpoints_event(namespace: list[str], data: Any) -> dict[str, Any]:
    return {
        "type": "event",
        "method": "checkpoints",
        "params": {"namespace": namespace, "timestamp": TS, "data": data},
    }


def _debug_event(namespace: list[str], data: Any) -> dict[str, Any]:
    return {
        "type": "event",
        "method": "debug",
        "params": {"namespace": namespace, "timestamp": TS, "data": data},
    }


def _tasks_event(namespace: list[str], data: Any) -> dict[str, Any]:
    return {
        "type": "event",
        "method": "tasks",
        "params": {"namespace": namespace, "timestamp": TS, "data": data},
    }


def _updates_event(namespace: list[str], data: Any) -> dict[str, Any]:
    return {
        "type": "event",
        "method": "updates",
        "params": {"namespace": namespace, "timestamp": TS, "data": data},
    }


def _arm(mux: StreamMux, transformer: Any) -> None:
    """Force projection logs to accept pushes (skip lazy-subscribe gate)."""
    mux._events._subscribed = True
    transformer._log._subscribed = True


def _unstamped(items):
    """Strip push stamps from a StreamChannel's internal buffer."""
    return [item for _stamp, item in items]


def _drain(transformer: Any) -> list[Any]:
    return _unstamped(transformer._log._items)


# ---------------------------------------------------------------------------
# CustomTransformer
# ---------------------------------------------------------------------------


def test_custom_captures_root_scope_events() -> None:
    t = CustomTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_custom_event([], {"status": "processing"}))
    mux.push(_custom_event([], {"status": "done"}))

    items = _drain(t)
    assert items == [{"status": "processing"}, {"status": "done"}]


def test_custom_ignores_subgraph_scope_events() -> None:
    t = CustomTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_custom_event(["subgraph:abc"], {"from": "child"}))

    assert _drain(t) == []


def test_custom_scoped_transformer_captures_own_scope() -> None:
    t = CustomTransformer(scope=("agent:abc",))
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_custom_event([], {"from": "root"}))
    mux.push(_custom_event(["agent:abc"], {"from": "self"}))
    mux.push(_custom_event(["agent:abc", "deep:def"], {"from": "child"}))

    items = _drain(t)
    assert items == [{"from": "self"}]


def test_custom_preserves_any_payload_type() -> None:
    t = CustomTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_custom_event([], "string_payload"))
    mux.push(_custom_event([], 42))
    mux.push(_custom_event([], [1, 2, 3]))

    assert _drain(t) == ["string_payload", 42, [1, 2, 3]]


def test_custom_does_not_suppress_from_main_log() -> None:
    t = CustomTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_custom_event([], "data"))

    methods = [evt["method"] for evt in _unstamped(mux._events._items)]
    assert "custom" in methods


def test_custom_ignores_other_methods() -> None:
    t = CustomTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(
        {
            "type": "event",
            "method": "values",
            "params": {"namespace": [], "timestamp": TS, "data": {}},
        }
    )
    assert _drain(t) == []


def test_custom_required_stream_modes() -> None:
    assert CustomTransformer.required_stream_modes == ("custom",)


def test_custom_is_native() -> None:
    assert getattr(CustomTransformer, "_native", False) is True


def test_custom_init_returns_correct_key() -> None:
    t = CustomTransformer()
    projection = t.init()
    assert "custom" in projection
    assert isinstance(projection["custom"], StreamChannel)


# ---------------------------------------------------------------------------
# CheckpointsTransformer
# ---------------------------------------------------------------------------


def test_checkpoints_captures_root_scope_events() -> None:
    t = CheckpointsTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    checkpoint_data = {"values": {"x": 1}, "next": ["node_b"]}
    mux.push(_checkpoints_event([], checkpoint_data))

    items = _drain(t)
    assert items == [checkpoint_data]


def test_checkpoints_ignores_subgraph_events() -> None:
    t = CheckpointsTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_checkpoints_event(["child:abc"], {"values": {"x": 1}}))

    assert _drain(t) == []


def test_checkpoints_scoped_transformer() -> None:
    t = CheckpointsTransformer(scope=("sub:abc",))
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_checkpoints_event([], {"from": "root"}))
    mux.push(_checkpoints_event(["sub:abc"], {"from": "self"}))

    assert _drain(t) == [{"from": "self"}]


def test_checkpoints_does_not_suppress_from_main_log() -> None:
    t = CheckpointsTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_checkpoints_event([], {"values": {}}))

    methods = [evt["method"] for evt in _unstamped(mux._events._items)]
    assert "checkpoints" in methods


def test_checkpoints_required_stream_modes() -> None:
    assert CheckpointsTransformer.required_stream_modes == ("checkpoints",)


def test_checkpoints_is_native() -> None:
    assert getattr(CheckpointsTransformer, "_native", False) is True


# ---------------------------------------------------------------------------
# DebugTransformer
# ---------------------------------------------------------------------------


def test_debug_captures_root_scope_events() -> None:
    t = DebugTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    debug_data = {
        "step": 0,
        "type": "checkpoint",
        "timestamp": "2026-01-01T00:00:00Z",
        "payload": {"values": {"x": 1}},
    }
    mux.push(_debug_event([], debug_data))

    items = _drain(t)
    assert items == [debug_data]


def test_debug_ignores_subgraph_events() -> None:
    t = DebugTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_debug_event(["child:abc"], {"step": 0, "type": "task"}))

    assert _drain(t) == []


def test_debug_captures_multiple_event_types() -> None:
    t = DebugTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_debug_event([], {"step": 0, "type": "checkpoint", "payload": {}}))
    mux.push(_debug_event([], {"step": 1, "type": "task", "payload": {}}))
    mux.push(_debug_event([], {"step": 1, "type": "task_result", "payload": {}}))

    items = _drain(t)
    assert len(items) == 3
    assert [d["type"] for d in items] == ["checkpoint", "task", "task_result"]


def test_debug_does_not_suppress_from_main_log() -> None:
    t = DebugTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_debug_event([], {"step": 0}))

    methods = [evt["method"] for evt in _unstamped(mux._events._items)]
    assert "debug" in methods


def test_debug_required_stream_modes() -> None:
    assert DebugTransformer.required_stream_modes == ("debug",)


def test_debug_is_native() -> None:
    assert getattr(DebugTransformer, "_native", False) is True


# ---------------------------------------------------------------------------
# TasksTransformer
# ---------------------------------------------------------------------------


def test_tasks_captures_root_scope_events() -> None:
    t = TasksTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    task_start = {"id": "t1", "name": "my_node", "input": None, "triggers": []}
    mux.push(_tasks_event([], task_start))

    items = _drain(t)
    assert items == [task_start]


def test_tasks_captures_start_and_result() -> None:
    t = TasksTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    start = {"id": "t1", "name": "a", "input": None, "triggers": []}
    result = {"id": "t1", "name": "a", "result": {"output": 42}, "error": None}
    mux.push(_tasks_event([], start))
    mux.push(_tasks_event([], result))

    items = _drain(t)
    assert items == [start, result]


def test_tasks_ignores_subgraph_events() -> None:
    t = TasksTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_tasks_event(["child:abc"], {"id": "t1", "name": "x"}))

    assert _drain(t) == []


def test_tasks_scoped_transformer() -> None:
    t = TasksTransformer(scope=("agent:abc",))
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_tasks_event([], {"id": "t1"}))
    mux.push(_tasks_event(["agent:abc"], {"id": "t2"}))
    mux.push(_tasks_event(["agent:abc", "deep:def"], {"id": "t3"}))

    assert _drain(t) == [{"id": "t2"}]


def test_tasks_does_not_suppress_from_main_log() -> None:
    """TasksTransformer returns True — it doesn't suppress tasks events.

    (LifecycleTransformer suppresses them, but that's independent.)
    """
    t = TasksTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_tasks_event([], {"id": "t1"}))

    methods = [evt["method"] for evt in _unstamped(mux._events._items)]
    assert "tasks" in methods


def test_tasks_required_stream_modes() -> None:
    assert TasksTransformer.required_stream_modes == ("tasks",)


def test_tasks_is_native() -> None:
    assert getattr(TasksTransformer, "_native", False) is True


# ---------------------------------------------------------------------------
# UpdatesTransformer
# ---------------------------------------------------------------------------


def test_updates_captures_root_scope_events() -> None:
    t = UpdatesTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    update = {"my_node": {"value": "hello!"}}
    mux.push(_updates_event([], update))

    items = _drain(t)
    assert items == [update]


def test_updates_captures_multiple_steps() -> None:
    t = UpdatesTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_updates_event([], {"node_a": {"x": 1}}))
    mux.push(_updates_event([], {"node_b": {"x": 2}}))

    items = _drain(t)
    assert items == [{"node_a": {"x": 1}}, {"node_b": {"x": 2}}]


def test_updates_ignores_subgraph_events() -> None:
    t = UpdatesTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_updates_event(["child:abc"], {"inner_node": {"v": 1}}))

    assert _drain(t) == []


def test_updates_scoped_transformer() -> None:
    t = UpdatesTransformer(scope=("agent:abc",))
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_updates_event([], {"from": "root"}))
    mux.push(_updates_event(["agent:abc"], {"from": "self"}))

    assert _drain(t) == [{"from": "self"}]


def test_updates_does_not_suppress_from_main_log() -> None:
    t = UpdatesTransformer()
    mux = StreamMux([t], is_async=False)
    _arm(mux, t)

    mux.push(_updates_event([], {"n": {}}))

    methods = [evt["method"] for evt in _unstamped(mux._events._items)]
    assert "updates" in methods


def test_updates_required_stream_modes() -> None:
    assert UpdatesTransformer.required_stream_modes == ("updates",)


def test_updates_is_native() -> None:
    assert getattr(UpdatesTransformer, "_native", False) is True


# ---------------------------------------------------------------------------
# Cross-transformer: unrelated events pass through
# ---------------------------------------------------------------------------


def test_unrelated_events_ignored_by_all() -> None:
    """Non-matching method events don't land in any transformer's log."""
    transformers = [
        CustomTransformer(),
        UpdatesTransformer(),
        CheckpointsTransformer(),
        DebugTransformer(),
        TasksTransformer(),
    ]
    mux = StreamMux(transformers, is_async=False)
    mux._events._subscribed = True
    for t in transformers:
        t._log._subscribed = True

    mux.push(
        {
            "type": "event",
            "method": "values",
            "params": {"namespace": [], "timestamp": TS, "data": {"x": 1}},
        }
    )

    for t in transformers:
        assert _unstamped(t._log._items) == []


# ---------------------------------------------------------------------------
# End-to-end: real graphs through stream_v2
# ---------------------------------------------------------------------------


class _State(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def _my_node(state: _State) -> dict[str, Any]:
    from langgraph.config import get_stream_writer

    writer = get_stream_writer()
    writer({"status": "working", "node": "my_node"})
    return {"value": state["value"] + "!", "items": ["done"]}


def _make_simple_graph() -> Any:
    builder = StateGraph(_State, input_schema=_State)
    builder.add_node("my_node", _my_node)
    builder.add_edge(START, "my_node")
    builder.add_edge("my_node", END)
    return builder.compile()


def test_stream_v2_custom_projection_opt_in() -> None:
    """run.custom surfaces get_stream_writer() payloads when opted in."""
    graph = _make_simple_graph()
    run = graph.stream_v2(
        {"value": "hello", "items": []}, transformers=[CustomTransformer]
    )

    custom_events = list(run.custom)
    assert len(custom_events) >= 1
    assert any(e.get("status") == "working" for e in custom_events)


def test_stream_v2_custom_and_values_coexist() -> None:
    """Both run.custom and run.values work in the same run."""
    graph = _make_simple_graph()
    run = graph.stream_v2(
        {"value": "hello", "items": []}, transformers=[CustomTransformer]
    )

    custom_events = list(run.custom)
    assert run.output is not None
    assert run.output["value"] == "hello!"
    assert len(custom_events) >= 1


def test_stream_v2_tasks_projection_opt_in() -> None:
    """run.tasks surfaces raw task events when opted in via transformers=."""
    graph = _make_simple_graph()
    run = graph.stream_v2({"value": "x", "items": []}, transformers=[TasksTransformer])

    tasks_events = list(run.tasks)
    assert len(tasks_events) >= 1
    names = [t.get("name") for t in tasks_events if "name" in t]
    assert "my_node" in names


def test_stream_v2_debug_projection_opt_in() -> None:
    """run.debug surfaces debug events when opted in via transformers=."""
    graph = _make_simple_graph()
    run = graph.stream_v2({"value": "x", "items": []}, transformers=[DebugTransformer])

    debug_events = list(run.debug)
    assert len(debug_events) >= 1
    types = {d.get("type") for d in debug_events}
    assert types & {"checkpoint", "task", "task_result"}


def test_stream_v2_updates_projection_opt_in() -> None:
    """run.updates surfaces node output dicts when opted in via transformers=."""
    graph = _make_simple_graph()
    run = graph.stream_v2(
        {"value": "x", "items": []}, transformers=[UpdatesTransformer]
    )

    updates = list(run.updates)
    assert len(updates) >= 1
    node_names = {k for u in updates for k in u if k != "__interrupt__"}
    assert "my_node" in node_names


def test_stream_v2_all_transformers_interleaved() -> None:
    """All five transformers registered together, consumed via interleave."""
    graph = _make_simple_graph()
    run = graph.stream_v2(
        {"value": "x", "items": []},
        transformers=[
            CustomTransformer,
            UpdatesTransformer,
            CheckpointsTransformer,
            DebugTransformer,
            TasksTransformer,
        ],
    )

    collected: dict[str, list[Any]] = {
        "custom": [],
        "updates": [],
        "debug": [],
        "tasks": [],
    }
    for name, item in run.interleave("custom", "updates", "debug", "tasks"):
        collected[name].append(item)

    assert len(collected["custom"]) >= 1
    assert len(collected["updates"]) >= 1
    assert len(collected["tasks"]) >= 1
    assert len(collected["debug"]) >= 1
    types = {d.get("type") for d in collected["debug"]}
    assert types & {"checkpoint", "task", "task_result"}
    node_names = {k for u in collected["updates"] for k in u if k != "__interrupt__"}
    assert "my_node" in node_names

    assert run.output is not None
    assert run.output["value"] == "x!"


def test_stream_v2_all_transformers_with_checkpointer() -> None:
    """All transformers with a checkpointer — run.checkpoints populated."""
    from langgraph.checkpoint.memory import InMemorySaver

    builder = StateGraph(_State, input_schema=_State)
    builder.add_node("my_node", _my_node)
    builder.add_edge(START, "my_node")
    builder.add_edge("my_node", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    run = graph.stream_v2(
        {"value": "x", "items": []},
        config={"configurable": {"thread_id": "test-all"}},
        transformers=[
            CustomTransformer,
            UpdatesTransformer,
            CheckpointsTransformer,
            DebugTransformer,
            TasksTransformer,
        ],
    )

    collected: dict[str, list[Any]] = {
        "custom": [],
        "updates": [],
        "checkpoints": [],
        "debug": [],
        "tasks": [],
    }
    for name, item in run.interleave(
        "custom", "updates", "checkpoints", "debug", "tasks"
    ):
        collected[name].append(item)

    assert len(collected["checkpoints"]) >= 1
    assert len(collected["custom"]) >= 1


def test_stream_v2_checkpoints_projection_opt_in() -> None:
    """run.checkpoints surfaces checkpoint data when opted in with a checkpointer."""
    from langgraph.checkpoint.memory import InMemorySaver

    builder = StateGraph(_State, input_schema=_State)
    builder.add_node("my_node", _my_node)
    builder.add_edge(START, "my_node")
    builder.add_edge("my_node", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    run = graph.stream_v2(
        {"value": "x", "items": []},
        config={"configurable": {"thread_id": "test-ckpt-standalone"}},
        transformers=[CheckpointsTransformer],
    )

    checkpoints = list(run.checkpoints)
    assert len(checkpoints) >= 1


# ---------------------------------------------------------------------------
# TasksTransformer + LifecycleTransformer co-registration
# ---------------------------------------------------------------------------


def test_tasks_and_lifecycle_coregistration() -> None:
    """When both are in the same StreamMux, LifecycleTransformer suppresses
    tasks events from the main log (returns False) while TasksTransformer
    still captures them into its own log.
    """
    lifecycle = LifecycleTransformer()
    tasks = TasksTransformer()
    mux = StreamMux([lifecycle, tasks], is_async=False)
    mux._events._subscribed = True
    tasks._log._subscribed = True
    lifecycle._channel._subscribed = True

    task_data = {"id": "t1", "name": "my_node", "input": None, "triggers": []}
    mux.push(_tasks_event([], task_data))

    assert _drain(tasks) == [task_data]

    methods = [evt["method"] for evt in _unstamped(mux._events._items)]
    assert "tasks" not in methods


def test_tasks_and_lifecycle_coregistration_e2e() -> None:
    """E2e: TasksTransformer captures task events even when LifecycleTransformer
    is present and suppressing them from the main log.
    """
    graph = _make_simple_graph()
    run = graph.stream_v2(
        {"value": "x", "items": []},
        transformers=[TasksTransformer],
    )

    tasks_events = list(run.tasks)
    assert len(tasks_events) >= 1
    names = [t.get("name") for t in tasks_events if "name" in t]
    assert "my_node" in names
