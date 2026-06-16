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
    _TasksLifecycleBase,
)

TS = int(time.time() * 1000)


def _tasks_start(
    namespace: list[str],
    *,
    task_id: str,
    name: str,
    metadata: dict[str, Any] | None = None,
    input: Any = None,
) -> dict[str, Any]:
    """Build a `tasks` ProtocolEvent carrying a TaskPayload (start)."""
    data: dict[str, Any] = {
        "id": task_id,
        "name": name,
        "input": input,
        "triggers": [],
    }
    if metadata is not None:
        data["metadata"] = metadata
    return {
        "type": "event",
        "method": "tasks",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": data,
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


# ---------------------------------------------------------------------------
# Parsed-segment fallback (no subagent boundary)
# ---------------------------------------------------------------------------


def test_no_metadata_falls_through_to_existing_behavior() -> None:
    """Tasks events without metadata produce the same output as before T4."""
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool"))

    [payload] = _drain_lifecycle(mux)
    assert payload["graph_name"] == "agent"
    assert payload["trigger_call_id"] == "abc"
    assert "cause" not in payload


def test_empty_metadata_dict_falls_through() -> None:
    """An explicit empty metadata dict is treated the same as no metadata."""
    mux = _build_lifecycle_mux()
    mux.push(_tasks_start(["agent:abc"], task_id="t1", name="tool", metadata={}))

    [payload] = _drain_lifecycle(mux)
    assert payload["graph_name"] == "agent"
    assert "cause" not in payload


# ---------------------------------------------------------------------------
# Subagent discrimination via lc_agent_name transition
# ---------------------------------------------------------------------------
#
# A nested task is a subagent iff its metadata["lc_agent_name"] is present and
# differs from its PARENT namespace's lc_agent_name. These tests replicate the
# empirically-verified `create_agent` stream shape synthetically:
#
#   - A supervisor created via `create_agent(name="supervisor")` emits its own
#     node tasks (model, tools) at ns=(), each with
#     metadata["lc_agent_name"] == "supervisor".
#   - The parent `tools` task at ns=() carries a `tool_call_with_context`
#     dict as its `input` (with the LLM tool_call_id at
#     input["tool_call"]["id"]) and a task `id`. (A legacy / batched shape
#     passes a list of tool-call dicts instead; both are exercised below.)
#   - When a tool body invokes an inner `create_agent(name="weather_agent")`,
#     the inner agent's node tasks stream at ns=("tools:<taskid>",) with
#     metadata["lc_agent_name"] == "weather_agent", sharing the SAME <taskid>
#     as the parent `tools` task.
#   - A plain StateGraph (no name) inherits the parent's lc_agent_name, so its
#     child lc == parent lc -> NOT a subagent.


def test_lifecycle_uses_lc_agent_name_for_subagent() -> None:
    """A nested run whose lc_agent_name differs from its parent's is a subagent.

    graph_name becomes the child's lc_agent_name; cause is recovered by joining
    the child segment's task-id to the parent push task's tool call. This uses
    the production `tool_call_with_context` dict input shape current langgraph
    emits (tool_call_id at input["tool_call"]["id"]).
    """
    mux = _build_lifecycle_mux()
    # Supervisor's `tools` push task at scope ns: carries its own lc_agent_name
    # and a `tool_call_with_context` dict as `input`. Each tool call is its own
    # push task, and the task id seeds the child segment.
    mux.push(
        _tasks_start(
            [],
            task_id="tools_task_1",
            name="tools",
            metadata={"lc_agent_name": "supervisor"},
            input={
                "__type": "tool_call_with_context",
                "tool_call": {
                    "name": "call_weather",
                    "args": {"city": "Boston"},
                    "id": "call_w",
                    "type": "tool_call",
                },
                "state": {},
            },
        )
    )
    # Inner weather_agent's first node task streams under the parent `tools`
    # task's namespace segment (shared task id) with its own lc_agent_name.
    mux.push(
        _tasks_start(
            ["tools:tools_task_1"],
            task_id="inner_model_1",
            name="model",
            metadata={"lc_agent_name": "weather_agent"},
        )
    )

    payloads = _drain_lifecycle(mux)
    started = [p for p in payloads if p["event"] == "started"]
    [subagent] = [p for p in started if p["namespace"] == ["tools:tools_task_1"]]
    assert subagent["graph_name"] == "weather_agent", (
        "graph_name should be the child's lc_agent_name, not the parsed segment"
    )
    assert subagent["cause"] == {"type": "toolCall", "tool_call_id": "call_w"}, (
        "cause should recover the triggering tool_call_id from the parent push "
        "task's tool_call_with_context input via the shared task id"
    )


def test_lifecycle_subagent_cause_from_legacy_list_input() -> None:
    """cause recovery also handles the legacy / batched list input shape.

    A parent task whose `input` is a list of tool-call dicts (rather than a
    `tool_call_with_context` dict) still seeds the tool_call_id join.
    """
    mux = _build_lifecycle_mux()
    mux.push(
        _tasks_start(
            [],
            task_id="tools_task_1",
            name="tools",
            metadata={"lc_agent_name": "supervisor"},
            input=[{"name": "call_weather", "args": {"city": "SF"}, "id": "call_w"}],
        )
    )
    mux.push(
        _tasks_start(
            ["tools:tools_task_1"],
            task_id="inner_model_1",
            name="model",
            metadata={"lc_agent_name": "weather_agent"},
        )
    )

    payloads = _drain_lifecycle(mux)
    started = [p for p in payloads if p["event"] == "started"]
    [subagent] = [p for p in started if p["namespace"] == ["tools:tools_task_1"]]
    assert subagent["graph_name"] == "weather_agent"
    assert subagent["cause"] == {"type": "toolCall", "tool_call_id": "call_w"}


def test_lifecycle_same_name_nested_run_is_surfaced() -> None:
    """A nested run whose lc_agent_name matches the parent's is still surfaced.

    A subagent that invokes itself re-asserts its own lc_agent_name, so child
    lc == parent lc. The discriminator surfaces any nested run carrying an
    lc_agent_name, so the recursive call is reported (named after the agent,
    with the triggering tool call as cause).

    Trade-off: a non-agent subgraph that merely inherited the parent's
    lc_agent_name would also surface here. That is accepted; a caller can null
    lc_agent_name in the config it invokes such a graph with to exclude it.
    """
    mux = _build_lifecycle_mux()
    mux.push(
        _tasks_start(
            [],
            task_id="tools_task_1",
            name="tools",
            metadata={"lc_agent_name": "weather_agent"},
            input={
                "__type": "tool_call_with_context",
                "tool_call": {
                    "name": "recurse",
                    "args": {},
                    "id": "call_x",
                    "type": "tool_call",
                },
                "state": {},
            },
        )
    )
    # The agent invokes itself: the nested run re-asserts the SAME lc_agent_name.
    mux.push(
        _tasks_start(
            ["tools:tools_task_1"],
            task_id="inner_node_1",
            name="model",
            metadata={"lc_agent_name": "weather_agent"},
        )
    )

    payloads = _drain_lifecycle(mux)
    started = [p for p in payloads if p["event"] == "started"]
    [nested] = [p for p in started if p["namespace"] == ["tools:tools_task_1"]]
    assert nested["graph_name"] == "weather_agent", (
        "a same-named nested run (e.g. self-recursion) must still be surfaced"
    )
    assert nested["cause"] == {"type": "toolCall", "tool_call_id": "call_x"}


def test_lifecycle_unnamed_nested_agent_is_not_subagent() -> None:
    """A nested run with lc_agent_name None is excluded (not a subagent)."""
    mux = _build_lifecycle_mux()
    mux.push(
        _tasks_start(
            [],
            task_id="tools_task_1",
            name="tools",
            metadata={"lc_agent_name": "supervisor"},
            input={
                "__type": "tool_call_with_context",
                "tool_call": {
                    "name": "lookup",
                    "args": {},
                    "id": "call_x",
                    "type": "tool_call",
                },
                "state": {},
            },
        )
    )
    mux.push(
        _tasks_start(
            ["plain:tools_task_1"],
            task_id="inner_node_1",
            name="inner_node",
            metadata={"lc_agent_name": None},
        )
    )

    payloads = _drain_lifecycle(mux)
    started = [p for p in payloads if p["event"] == "started"]
    [nested] = [p for p in started if p["namespace"] == ["plain:tools_task_1"]]
    assert nested["graph_name"] == "plain"
    assert "cause" not in nested


def test_lifecycle_subagent_terminal_roundtrip() -> None:
    """A detected subagent closes with `completed` when its parent task results.

    Pushes the subagent's `started` (via the `tool_call_with_context` parent
    plus the child task event) and then the parent push task's terminal
    result, asserting the namespace is closed and the `started` payload's
    projected graph_name / cause survive the roundtrip.
    """
    mux = _build_lifecycle_mux()
    mux.push(
        _tasks_start(
            [],
            task_id="tools_task_1",
            name="tools",
            metadata={"lc_agent_name": "supervisor"},
            input={
                "__type": "tool_call_with_context",
                "tool_call": {
                    "name": "call_weather",
                    "args": {"city": "Boston"},
                    "id": "call_w",
                    "type": "tool_call",
                },
                "state": {},
            },
        )
    )
    mux.push(
        _tasks_start(
            ["tools:tools_task_1"],
            task_id="inner_model_1",
            name="model",
            metadata={"lc_agent_name": "weather_agent"},
        )
    )
    # The parent push task (id=tools_task_1, at scope ns) finishes, closing
    # the subagent subgraph that streamed under `tools:tools_task_1`.
    mux.push(_tasks_result([], task_id="tools_task_1", name="tools"))

    payloads = _drain_lifecycle(mux)
    ns = ["tools:tools_task_1"]
    subagent = [p for p in payloads if p["namespace"] == ns]
    assert [p["event"] for p in subagent] == ["started", "completed"]
    started, _completed = subagent
    assert started["graph_name"] == "weather_agent"
    assert started["cause"] == {"type": "toolCall", "tool_call_id": "call_w"}


def test_on_started_override_without_cause_is_backward_compatible() -> None:
    """An `_on_started` override with the original 3-arg signature must work.

    `cause` is delivered via `self._pending_cause`, not the call signature, so
    older/third-party subclasses (e.g. deepagents' `SubagentTransformer`) that
    override `_on_started(self, ns, graph_name, trigger_call_id)` keep working —
    no `TypeError: _on_started() got an unexpected keyword argument 'cause'`.
    """
    seen: list[tuple] = []

    class _LegacyTransformer(_TasksLifecycleBase):
        def init(self) -> dict[str, Any]:
            return {}

        def _should_track(self, ns: tuple[str, ...]) -> bool:
            return len(ns) == 1  # direct child of the root scope

        # Deliberately omits `cause` — mirrors a pre-`cause` override.
        def _on_started(self, ns, graph_name, trigger_call_id) -> None:  # type: ignore[override]
            seen.append((ns, graph_name, trigger_call_id))

        def _on_terminal(self, ns, status, error) -> None:
            pass

    transformer = _LegacyTransformer(scope=())
    # Must not raise: the base calls `_on_started` without a `cause` kwarg.
    transformer.process(_tasks_start(["agent:abc123"], task_id="abc123", name="agent"))

    assert seen == [(("agent:abc123",), "agent", "abc123")]
