"""Tests for LifecycleTransformer.

Consumes the `tasks` stream mode and emits subgraph lifecycle payloads
on the `lifecycle` channel for both in-process iteration via
`run.lifecycle` and wire delivery via `custom:lifecycle` protocol
events. Most tests dispatch synthetic protocol events through a
`StreamMux` to keep the inference logic isolated; the end-of-file
group exercises the path through real graphs (multi-depth
discovery, nested `stream_events(version="v3")` calls with non-empty `parent_ns`).
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any

from typing_extensions import TypedDict

from langgraph._internal._constants import CONF, CONFIG_KEY_CHECKPOINT_NS
from langgraph.constants import END, START
from langgraph.errors import GraphInterrupt
from langgraph.graph import StateGraph
from langgraph.stream._mux import StreamMux
from langgraph.stream.transformers import (
    LifecyclePayload,
    LifecycleTransformer,
)

TS = int(time.time() * 1000)


def _tasks_start(
    namespace: list[str],
    *,
    task_id: str,
    name: str,
    input: Any = None,
) -> dict[str, Any]:
    """Build a `tasks` ProtocolEvent carrying a TaskPayload (start).

    Pass `input={"tool_call": {"args": {...}}}` (or any envelope with
    that shape) to exercise the lifecycle transformer's input mining of
    spawn-intent metadata (`subagent_type`, `description`) — this
    mirrors the `ToolCallWithContext` payload `langgraph.prebuilt.ToolNode`
    Send-fans out per tool call.
    """
    return {
        "type": "event",
        "method": "tasks",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": {
                "id": task_id,
                "name": name,
                "input": input,
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
    """Build a `tasks` ProtocolEvent carrying a TaskResultPayload (finish)."""
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


def _arm(mux: StreamMux) -> None:
    """Force projection channels to accept pushes (skip lazy-subscribe gate).

    `StreamChannel.push` only appends to the local buffer when a
    subscriber is attached. Tests that inspect `_items` directly need
    the gate flipped before any event is dispatched.
    """
    mux._events._subscribed = True
    for transformer in mux._transformers:
        if isinstance(transformer, LifecycleTransformer):
            transformer._channel._subscribed = True


def _unstamped(items):
    """Strip push stamps from a StreamChannel's internal buffer."""
    return [item for _stamp, item in items]


def _drain_lifecycle(mux: StreamMux) -> list[LifecyclePayload]:
    """Snapshot the lifecycle channel's buffer."""
    transformer = mux.transformer_by_key("lifecycle")
    assert isinstance(transformer, LifecycleTransformer)
    return _unstamped(transformer._channel._items)


def _build_lifecycle_mux(*, scope: tuple[str, ...] = ()) -> StreamMux:
    mux = StreamMux([LifecycleTransformer(scope=scope)], is_async=False)
    _arm(mux)
    return mux


# ---------------------------------------------------------------------------
# LifecycleTransformer
# ---------------------------------------------------------------------------


def test_started_emitted_on_first_direct_child_task() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc123"], task_id="t1", name="tool"))

    [payload] = _drain_lifecycle(mux)
    assert payload["event"] == "started"
    assert payload["namespace"] == ["agent:abc123"]
    assert payload["graph_name"] == "agent"
    assert payload["trigger_call_id"] == "abc123"


def test_started_carries_cause_when_parent_input_has_spawn_metadata() -> None:
    """When a parent task's `input` is a `ToolCallWithContext`-shaped
    envelope (`{"tool_call": {"args": {...}}, ...}`, the layout
    `langgraph.prebuilt.ToolNode` Send-fans out per call), the
    transformer mines `subagent_type` and `description` from
    `tool_call.args` and remembers them keyed by `parent_task_id`.
    When that parent task spawns a subgraph (the child's namespace
    ends in `name:<parent_task_id>`), the `lifecycle.started` payload
    carries `cause = {"type": "tool_call", "subagent_type": ..., "description": ...}`.
    Consumers join on `trigger_call_id` (the pregel task id) for
    identity; this dict is purely descriptive."""
    mux = _build_lifecycle_mux()
    # Parent task at root ns whose input matches the Send envelope.
    mux.push(
        _tasks_start(
            [],
            task_id="abc123",
            name="tools",
            input={
                "tool_call": {
                    "id": "call_xyz",
                    "name": "task",
                    "args": {
                        "subagent_type": "researcher",
                        "description": "look up weather",
                    },
                }
            },
        )
    )
    # Child subgraph's first task — trigger_call_id parsed from segment.
    mux.push(_tasks_start(["agent:abc123"], task_id="t1", name="model"))

    [payload] = _drain_lifecycle(mux)
    assert payload["event"] == "started"
    assert payload["trigger_call_id"] == "abc123"
    assert payload["cause"] == {
        "type": "tool_call",
        "subagent_type": "researcher",
        "description": "look up weather",
    }
    # tool_call_id is intentionally not in cause — consumers join on
    # trigger_call_id (the pregel task id) instead.
    assert "tool_call_id" not in payload["cause"]


def test_started_cause_with_description_but_no_subagent_type() -> None:
    """Partial spawn metadata (only `description`, or only `subagent_type`)
    still produces a cause — both fields are optional within the dict."""
    mux = _build_lifecycle_mux()
    mux.push(
        _tasks_start(
            [],
            task_id="abc123",
            name="tools",
            input={
                "tool_call": {
                    "id": "call_xyz",
                    "name": "task",
                    "args": {"description": "do a thing"},
                }
            },
        )
    )
    mux.push(_tasks_start(["agent:abc123"], task_id="t1", name="model"))

    [payload] = _drain_lifecycle(mux)
    assert payload["cause"] == {
        "type": "tool_call",
        "description": "do a thing",
    }


def test_started_carries_cause_for_list_shape_per_call_input() -> None:
    """langchain v1's `create_agent` Send-fans out a per-call task whose
    `input` is a single-element list of tool-call dicts:
    `[{"id": ..., "name": ..., "args": {...}}]`. The transformer mines
    `subagent_type` and `description` from `args` exactly as for the
    `ToolCallWithContext` dict envelope, so `lifecycle.started.cause`
    fires regardless of which agent factory drove the dispatch."""
    mux = _build_lifecycle_mux()
    mux.push(
        _tasks_start(
            [],
            task_id="abc123",
            name="tools",
            input=[
                {
                    "id": "tc-1",
                    "name": "task",
                    "args": {
                        "subagent_type": "researcher",
                        "description": "Do X",
                    },
                }
            ],
        )
    )
    mux.push(_tasks_start(["agent:abc123"], task_id="t1", name="model"))

    [payload] = _drain_lifecycle(mux)
    assert payload["event"] == "started"
    assert payload["trigger_call_id"] == "abc123"
    assert payload["cause"] == {
        "type": "tool_call",
        "subagent_type": "researcher",
        "description": "Do X",
    }


def test_list_shape_ignored_when_not_single_element() -> None:
    """Only single-element lists are recognized as the per-call shape;
    a 0- or 2+-element list is some other batched/multi-call payload
    and must not be mined."""
    # Two-element list — not the per-call shape.
    mux = _build_lifecycle_mux()
    mux.push(
        _tasks_start(
            [],
            task_id="abc123",
            name="tools",
            input=[
                {
                    "id": "tc-1",
                    "name": "task",
                    "args": {"subagent_type": "researcher"},
                },
                {
                    "id": "tc-2",
                    "name": "task",
                    "args": {"subagent_type": "writer"},
                },
            ],
        )
    )
    mux.push(_tasks_start(["agent:abc123"], task_id="t1", name="model"))

    [payload] = _drain_lifecycle(mux)
    assert "cause" not in payload

    # Empty list.
    mux2 = _build_lifecycle_mux()
    mux2.push(_tasks_start([], task_id="def456", name="tools", input=[]))
    mux2.push(_tasks_start(["agent:def456"], task_id="t1", name="model"))
    [payload2] = _drain_lifecycle(mux2)
    assert "cause" not in payload2


def test_list_shape_robust_to_non_dict_or_missing_args() -> None:
    """Duck-typing safety: a single-element list whose element isn't a
    dict, or whose dict has no/non-dict `args`, or whose `args` lacks
    both fields, must not raise — it just no-ops."""
    # Element is not a dict.
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start([], task_id="t-a", name="tools", input=["not-a-dict"]))
    mux.push(_tasks_start(["agent:t-a"], task_id="t1", name="model"))
    [payload] = _drain_lifecycle(mux)
    assert "cause" not in payload

    # Args is not a dict.
    mux2 = _build_lifecycle_mux()
    mux2.push(
        _tasks_start(
            [],
            task_id="t-b",
            name="tools",
            input=[{"id": "tc", "name": "task", "args": "nope"}],
        )
    )
    mux2.push(_tasks_start(["agent:t-b"], task_id="t1", name="model"))
    [payload2] = _drain_lifecycle(mux2)
    assert "cause" not in payload2

    # Args dict lacks both subagent_type and description.
    mux3 = _build_lifecycle_mux()
    mux3.push(
        _tasks_start(
            [],
            task_id="t-c",
            name="tools",
            input=[{"id": "tc", "name": "task", "args": {"other": "field"}}],
        )
    )
    mux3.push(_tasks_start(["agent:t-c"], task_id="t1", name="model"))
    [payload3] = _drain_lifecycle(mux3)
    assert "cause" not in payload3


def test_started_omits_cause_for_structurally_spawned_subgraph() -> None:
    """Subgraphs spawned without a recognizable tool-call envelope on
    the parent's input (Send with custom payloads, plain nested
    `graph.invoke`, etc.) don't get a `cause` field on
    `lifecycle.started`."""
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc123"], task_id="t1", name="tool"))

    [payload] = _drain_lifecycle(mux)
    assert "cause" not in payload


def test_started_dedup_on_repeat_namespace() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="a"))
    mux.push(_tasks_start(["agent:abc"], task_id="t2", name="b"))

    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started"]


def test_grandchild_namespace_discovered() -> None:
    """Subgraphs at any depth below scope are tracked, not just direct children."""
    mux = _build_lifecycle_mux()
    # First-seen task at length-2 ns means a 2nd-level subgraph started.
    mux.push(_tasks_start(["agent:abc", "tool:def"], task_id="t1", name="x"))

    [payload] = _drain_lifecycle(mux)
    assert payload["event"] == "started"
    assert payload["namespace"] == ["agent:abc", "tool:def"]


def test_nested_chain_emits_started_at_each_depth() -> None:
    """A graph → subgraph → subgraph chain produces a started event per level."""
    mux = _build_lifecycle_mux()
    # Subgraph1 starts emitting tasks (events tagged with its own ns).
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    # Subgraph1 invokes subgraph2; subgraph2's first task event arrives.
    mux.push(_tasks_start(["agent:abc", "tool:def"], task_id="t2", name="deep"))

    payloads = _drain_lifecycle(mux)
    assert [p["namespace"] for p in payloads] == [
        ["agent:abc"],
        ["agent:abc", "tool:def"],
    ]
    assert all(p["event"] == "started" for p in payloads)


def test_nested_chain_emits_completed_at_each_depth() -> None:
    """Each subgraph in a nested chain closes when its parent task result arrives."""
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_start(["agent:abc", "tool:def"], task_id="t2", name="deep"))

    # Subgraph2's owning task (id=def, inside subgraph1) finishes.
    mux.push(_tasks_result(["agent:abc"], task_id="def", name="tool"))
    # Subgraph1's owning task (id=abc, at root) finishes.
    mux.push(_tasks_result([], task_id="abc", name="agent"))

    payloads = _drain_lifecycle(mux)
    events = [(p["event"], p["namespace"]) for p in payloads]
    assert events == [
        ("started", ["agent:abc"]),
        ("started", ["agent:abc", "tool:def"]),
        ("completed", ["agent:abc", "tool:def"]),
        ("completed", ["agent:abc"]),
    ]


def test_completed_on_parent_task_result() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent"))

    events = [p["event"] for p in _drain_lifecycle(mux)]
    assert events == ["started", "completed"]


def test_failed_on_parent_task_result_with_error() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent", error="boom"))

    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started", "failed"]
    assert payloads[1]["error"] == "boom"


def test_interrupted_on_parent_task_result_with_interrupts() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(
        _tasks_result(
            [],
            task_id="abc",
            name="agent",
            interrupts=[{"value": "pause"}],
        )
    )

    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started", "interrupted"]


def test_interrupt_takes_precedence_over_error() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(
        _tasks_result(
            [],
            task_id="abc",
            name="agent",
            error="should-be-suppressed",
            interrupts=[{"value": "pause"}],
        )
    )

    last = _drain_lifecycle(mux)[-1]
    assert last["event"] == "interrupted"
    assert "error" not in last


def test_finalize_completes_open_subgraphs() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.close()
    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started", "completed"]


def test_fail_emits_interrupted_for_graph_interrupt() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.fail(GraphInterrupt())
    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started", "interrupted"]
    assert "error" not in payloads[1]


def test_fail_emits_failed_for_other_exceptions() -> None:
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    mux.fail(RuntimeError("boom"))
    payloads = _drain_lifecycle(mux)
    assert [p["event"] for p in payloads] == ["started", "failed"]
    assert payloads[1]["error"] == "boom"


def test_unrelated_methods_pass_through() -> None:
    """Non-`tasks` events are not consumed and don't emit lifecycle."""
    mux = _build_lifecycle_mux()
    mux.push(
        {
            "type": "event",
            "method": "values",
            "params": {"namespace": ["agent:abc"], "timestamp": TS, "data": {}},
        }
    )
    assert _drain_lifecycle(mux) == []


def test_scoped_transformer_filters_outside_scope_but_tracks_all_depths() -> None:
    """Scope filters the prefix; subgraphs at any depth below scope are tracked."""
    mux = _build_lifecycle_mux(scope=("agent:abc",))
    # Root-level task — out of scope (no shared prefix).
    mux.push(_tasks_start(["other:1"], task_id="t1", name="other"))
    # Direct child of agent:abc — in scope.
    mux.push(_tasks_start(["agent:abc", "tool:def"], task_id="t2", name="tool"))
    # Grandchild of agent:abc — also in scope, tracked at its own depth.
    mux.push(
        _tasks_start(["agent:abc", "tool:def", "deep:ghi"], task_id="t3", name="deep")
    )

    payloads = _drain_lifecycle(mux)
    assert [p["namespace"] for p in payloads] == [
        ["agent:abc", "tool:def"],
        ["agent:abc", "tool:def", "deep:ghi"],
    ]


def test_required_stream_modes_declared() -> None:
    assert LifecycleTransformer.required_stream_modes == ("tasks",)


def test_protocol_event_method_is_native() -> None:
    """Native transformer — auto-forwarded events use `lifecycle`, not `custom:lifecycle`."""
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    methods = {evt["method"] for evt in _unstamped(mux._events._items)}
    assert "lifecycle" in methods
    assert "custom:lifecycle" not in methods


def test_tasks_events_suppressed_from_main_log() -> None:
    """Tasks events are folded into lifecycle and don't appear on the main log."""
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))
    mux.push(_tasks_result([], task_id="abc", name="agent"))

    methods = [evt["method"] for evt in _unstamped(mux._events._items)]
    assert "tasks" not in methods
    # Lifecycle events did make it through, though.
    assert "lifecycle" in methods


# ---------------------------------------------------------------------------
# End-to-end: real graphs through stream_events(version="v3")
# ---------------------------------------------------------------------------


class _State(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def _passthrough(state: _State) -> dict[str, Any]:
    return {"value": state["value"] + "!", "items": ["x"]}


def _make_two_level_nested() -> Any:
    """Build outer → middle → inner. Three Pregel instances, two nesting levels."""
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


def test_stream_events_v3_real_graph_emits_lifecycle_at_each_depth() -> None:
    """Outer graph with two nested subgraphs surfaces lifecycle for both."""
    graph = _make_two_level_nested()
    run = graph.stream_events({"value": "x", "items": []}, version="v3")

    # Iterating the projection drives the pump and drains synthesized
    # lifecycle events at the same time.
    payloads = list(run.lifecycle)
    # Each subgraph instance produces a started + a terminal event. Two
    # nested instances, so four payloads total in some interleaving.
    by_event = {p["event"] for p in payloads}
    assert "started" in by_event
    assert "completed" in by_event
    # Two distinct namespaces — direct child of root, and grandchild.
    namespaces = {tuple(p["namespace"]) for p in payloads}
    direct_children = {ns for ns in namespaces if len(ns) == 1}
    grandchildren = {ns for ns in namespaces if len(ns) == 2}
    assert direct_children, f"expected a level-1 lifecycle namespace, got {namespaces}"
    assert grandchildren, f"expected a level-2 lifecycle namespace, got {namespaces}"
    # Every direct-child namespace has a matching grandchild whose path extends it.
    for parent in direct_children:
        assert any(gc[: len(parent)] == parent for gc in grandchildren), (
            f"grandchild does not extend parent {parent}: {grandchildren}"
        )


def test_stream_events_v3_with_nested_parent_ns_scopes_lifecycle() -> None:
    """When `stream_events(version="v3")` is called with a non-empty checkpoint_ns in config,
    `_resolve_parent_ns` returns that namespace and the registered
    `LifecycleTransformer` is constructed with `scope=parent_ns`. This
    exercises the path that exists today purely for nested-stream_events(version="v3")
    callers; the test simulates such a caller by injecting a
    checkpoint_ns into the config.
    """
    graph = _make_two_level_nested()
    config = {CONF: {CONFIG_KEY_CHECKPOINT_NS: "outer:abc"}}
    run = graph.stream_events({"value": "x", "items": []}, config=config, version="v3")

    payloads = list(run.lifecycle)
    # Every emitted lifecycle namespace must extend the caller's scope —
    # nothing at root-level, nothing under a sibling prefix.
    for p in payloads:
        ns = tuple(p["namespace"])
        assert ns[:1] == ("outer:abc",), (
            f"namespace {ns} not within scoped prefix ('outer:abc',)"
        )
