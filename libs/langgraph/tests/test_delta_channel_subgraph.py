"""Tests for reading and updating a subgraph's `DeltaChannel` state.

Regression suite for #8470.

A subgraph is compiled without a checkpointer of its own. The parent lends it
one through `CONFIG_KEY_CHECKPOINTER` at read time. `DeltaChannel` stores no
value in `channel_values`, so hydrating it requires that saver to walk ancestors
and replay their writes. Reading with `self.checkpointer` instead of the
caller-resolved saver leaves a subgraph with none, and the channel falls through
to `from_checkpoint(MISSING)`: an empty value, indistinguishable from a channel
that was never written and raising nothing.

Coverage:

* `get_state` / `aget_state` and `get_state_history` / `aget_state_history` on a
  subgraph namespace, at one and two levels of nesting
* `get_state(subgraphs=True)` task states, the surface a human-in-the-loop client
  reads while a subgraph sits interrupted
* `update_state` / `aupdate_state`, which persist the hydrated value as a
  snapshot blob and so turn an empty hydration into permanent data loss
* root-graph controls, which own their checkpointer and were never affected
"""

from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

pytestmark = pytest.mark.anyio


def _extend(state: list | None, writes: list[Any]) -> list:
    """Batching-invariant list reducer: flattens each write batch onto the state."""
    out = list(state or [])
    for write in writes:
        out.extend(write if isinstance(write, list) else [write])
    return out


def _child_builder(*, snapshot_frequency: int) -> StateGraph:
    """Two-node graph writing `["a1"]` then `["b1", "b2"]` to a `DeltaChannel`."""

    class State(TypedDict, total=False):
        msgs: Annotated[
            list, DeltaChannel(_extend, snapshot_frequency=snapshot_frequency)
        ]

    builder = StateGraph(State)
    builder.add_node("a", lambda state: {"msgs": ["a1"]})
    builder.add_node("b", lambda state: {"msgs": ["b1", "b2"]})
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    return builder


def _wrap(inner: StateGraph, *, node_name: str) -> StateGraph:
    """Nest `inner` as a single node, sharing its state schema."""
    builder = StateGraph(inner.state_schema)
    builder.add_node(node_name, inner.compile())
    builder.add_edge(START, node_name)
    builder.add_edge(node_name, END)
    return builder


def _build_nested_graph(
    checkpointer: InMemorySaver,
    *,
    snapshot_frequency: int = 1000,
    depth: int = 1,
) -> Any:
    """Compile a graph whose `child` node is a subgraph nested `depth` levels deep."""
    graph = _child_builder(snapshot_frequency=snapshot_frequency)
    for _ in range(depth):
        graph = _wrap(graph, node_name="child")
    return graph.compile(checkpointer=checkpointer)


def _namespaced(config: dict, namespace: str) -> dict:
    return {"configurable": {**config["configurable"], "checkpoint_ns": namespace}}


def _subgraph_namespace(app: Any, config: dict, *, depth: int = 1) -> str:
    """Resolve the `child` namespace `depth` levels down, as a client would.

    Each snapshot's tasks carry `{name, state: {"configurable": {"checkpoint_ns"}}}`,
    and passing that namespace back is the documented way to read a subgraph's own
    supersteps.
    """
    namespace = ""
    for level in range(depth):
        scoped = _namespaced(config, namespace) if namespace else config
        namespace = next(
            (
                task.state["configurable"]["checkpoint_ns"]
                for snapshot in app.get_state_history(scoped)
                for task in snapshot.tasks
                if task.name == "child" and isinstance(task.state, dict)
            ),
            "",
        )
        assert namespace, f"no `child` subgraph task at nesting level {level}"
    return namespace


# ---------------------------------------------------------------------------
# Reading a subgraph's state
# ---------------------------------------------------------------------------


def test_subgraph_get_state_replays_delta_channel() -> None:
    app = _build_nested_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    namespace = _subgraph_namespace(app, config)

    assert app.get_state(config).values == {"msgs": ["a1", "b1", "b2"]}
    assert app.get_state(_namespaced(config, namespace)).values == {
        "msgs": ["a1", "b1", "b2"]
    }


async def test_subgraph_aget_state_replays_delta_channel() -> None:
    app = _build_nested_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({}, config)

    namespace = _subgraph_namespace(app, config)

    assert (await app.aget_state(_namespaced(config, namespace))).values == {
        "msgs": ["a1", "b1", "b2"]
    }


def test_subgraph_get_state_history_replays_delta_channel() -> None:
    app = _build_nested_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    namespace = _subgraph_namespace(app, config)
    values = [
        snapshot.values
        for snapshot in app.get_state_history(_namespaced(config, namespace))
    ]

    # newest first: after `b`, after `a`, then the two supersteps preceding any write
    assert values == [
        {"msgs": ["a1", "b1", "b2"]},
        {"msgs": ["a1"]},
        {"msgs": []},
        {"msgs": []},
    ]


async def test_subgraph_aget_state_history_replays_delta_channel() -> None:
    app = _build_nested_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({}, config)

    namespace = _subgraph_namespace(app, config)
    values = [
        snapshot.values
        async for snapshot in app.aget_state_history(_namespaced(config, namespace))
    ]

    assert values == [
        {"msgs": ["a1", "b1", "b2"]},
        {"msgs": ["a1"]},
        {"msgs": []},
        {"msgs": []},
    ]


def test_doubly_nested_subgraph_get_state_replays_delta_channel() -> None:
    app = _build_nested_graph(InMemorySaver(), depth=2)
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    namespace = _subgraph_namespace(app, config, depth=2)

    assert app.get_state(_namespaced(config, namespace)).values == {
        "msgs": ["a1", "b1", "b2"]
    }


def test_interrupted_subgraph_task_state_replays_delta_channel() -> None:
    """`get_state(subgraphs=True)` resolves task states through a separate
    recursion; it reads the paused subgraph a human-in-the-loop client is shown."""

    class State(TypedDict, total=False):
        msgs: Annotated[list, DeltaChannel(_extend)]

    def pause(state: State) -> dict:
        interrupt("pause")
        return {"msgs": ["b1"]}

    child = StateGraph(State)
    child.add_node("a", lambda state: {"msgs": ["a1", "a2"]})
    child.add_node("b", pause)
    child.add_edge(START, "a")
    child.add_edge("a", "b")
    child.add_edge("b", END)

    app = _wrap(child, node_name="child").compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    task_values = [
        task.state.values
        for task in app.get_state(config, subgraphs=True).tasks
        if task.name == "child"
    ]

    assert task_values == [{"msgs": ["a1", "a2"]}]


# ---------------------------------------------------------------------------
# Updating a subgraph's state
# ---------------------------------------------------------------------------


def test_subgraph_update_state_preserves_delta_channel_history() -> None:
    """`update_state` persists the hydrated channel, so an empty hydration is
    written to disk rather than merely returned.

    `snapshot_frequency=2` is what makes that observable: this update reaches the
    cadence and forces a `_DeltaSnapshot` blob built from the hydrated value. At
    the default frequency nothing is written and the loss stays latent, so the
    assertion below would hold either way.
    """
    app = _build_nested_graph(InMemorySaver(), snapshot_frequency=2)
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    namespace = _subgraph_namespace(app, config)
    child_config = _namespaced(config, namespace)
    app.update_state(child_config, {"msgs": ["manual"]})

    assert app.get_state(child_config).values == {"msgs": ["a1", "b1", "b2", "manual"]}


async def test_subgraph_aupdate_state_preserves_delta_channel_history() -> None:
    app = _build_nested_graph(InMemorySaver(), snapshot_frequency=2)
    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({}, config)

    namespace = _subgraph_namespace(app, config)
    child_config = _namespaced(config, namespace)
    await app.aupdate_state(child_config, {"msgs": ["manual"]})

    assert (await app.aget_state(child_config)).values == {
        "msgs": ["a1", "b1", "b2", "manual"]
    }


# ---------------------------------------------------------------------------
# Controls: a graph owning its checkpointer resolves the same saver as before
# ---------------------------------------------------------------------------


def test_root_graph_get_state_replays_delta_channel() -> None:
    app = _child_builder(snapshot_frequency=1000).compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    assert app.get_state(config).values == {"msgs": ["a1", "b1", "b2"]}
    assert [snapshot.values for snapshot in app.get_state_history(config)] == [
        {"msgs": ["a1", "b1", "b2"]},
        {"msgs": ["a1"]},
        {"msgs": []},
        {"msgs": []},
    ]


def test_root_graph_update_state_preserves_delta_channel_history() -> None:
    app = _child_builder(snapshot_frequency=2).compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    app.update_state(config, {"msgs": ["manual"]})

    assert app.get_state(config).values == {"msgs": ["a1", "b1", "b2", "manual"]}


def test_stateless_subgraph_persists_nothing() -> None:
    """A `checkpointer=False` subgraph opts out of persistence entirely.

    Its state is unavailable by design, and passing the caller's saver must not
    start surfacing state for a subgraph that asked not to be checkpointed.
    """
    child = _child_builder(snapshot_frequency=1000).compile(checkpointer=False)
    builder = StateGraph(child.builder.state_schema)
    builder.add_node("child", child)
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    app = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    subgraph_namespaces = [
        task.state["configurable"]["checkpoint_ns"]
        for snapshot in app.get_state_history(config)
        for task in snapshot.tasks
        if task.name == "child" and isinstance(task.state, dict)
    ]

    assert subgraph_namespaces == []
    assert app.get_state(config).values == {"msgs": ["a1", "b1", "b2"]}


def test_completed_subgraph_exposes_no_task_state() -> None:
    """`subgraphs=True` surfaces task state only while a task is pending.

    Once the subgraph has finished there is no task to attach state to, for every
    channel type alike. This is subgraph behaviour rather than delta replay, and
    the fix leaves it untouched.
    """
    app = _build_nested_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    assert app.get_state(config, subgraphs=True).tasks == ()
