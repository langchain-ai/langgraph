"""Tests for reading/updating a nested subgraph's `DeltaChannel` state.

A subgraph is compiled without a checkpointer of its own — the parent supplies
one via `CONFIG_KEY_CHECKPOINTER` at read time. The public readers resolve it
from the config, so they must hand it to `channels_from_checkpoint`: without a
saver a `DeltaChannel` that needs an ancestor walk hydrates empty (and silently,
since `from_checkpoint(MISSING)` is a valid empty channel).

Coverage:

* nested-subgraph `get_state` / `aget_state` and `get_state_history` /
  `aget_state_history` hydrate delta channels
* root-graph control (owns its checkpointer) still hydrates
* nested-subgraph `update_state` / `aupdate_state` accumulate onto the replayed
  state rather than onto an empty one
"""

from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import START, StateGraph
from langgraph.graph.message import _messages_delta_reducer

pytestmark = pytest.mark.anyio


def _build_graph(
    checkpointer: InMemorySaver,
    *,
    freq: int = 1000,
) -> Any:
    """Parent graph with a single `child` subgraph writing two messages."""
    channel = DeltaChannel(_messages_delta_reducer, snapshot_frequency=freq)
    # Functional TypedDict form: class form can't reference `channel` (a
    # local variable) inside Annotated due to forward-ref evaluation rules.
    State = TypedDict("State", {"messages": Annotated[list, channel]})  # type: ignore[call-overload]  # noqa: UP013

    def one(state: dict) -> dict:
        return {"messages": [AIMessage(content="one", id="ai1")]}

    def two(state: dict) -> dict:
        return {"messages": [AIMessage(content="two", id="ai2")]}

    child = StateGraph(State)
    child.add_node("one", one)
    child.add_node("two", two)
    child.add_edge(START, "one")
    child.add_edge("one", "two")
    child.set_finish_point("two")

    parent = StateGraph(State)
    parent.add_node("child", child.compile())
    parent.add_edge(START, "child")
    parent.set_finish_point("child")
    return parent.compile(checkpointer=checkpointer)


def _child_config(graph: Any, config: dict) -> dict:
    """Config addressing the `child` task's own checkpoint namespace."""
    ns = next(
        task.state["configurable"]["checkpoint_ns"]
        for snapshot in graph.get_state_history(config)
        for task in snapshot.tasks
        if task.name == "child" and isinstance(task.state, dict)
    )
    return {
        "configurable": {
            "thread_id": config["configurable"]["thread_id"],
            "checkpoint_ns": ns,
        }
    }


def _contents(values: dict) -> list[str]:
    return [m.content for m in values["messages"]]


def test_subgraph_get_state_delta_channel() -> None:
    graph = _build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "sub-get-state"}}
    graph.invoke({"messages": []}, config)

    assert _contents(graph.get_state(config).values) == ["one", "two"]
    child_state = graph.get_state(_child_config(graph, config))
    assert _contents(child_state.values) == ["one", "two"]


def test_subgraph_get_state_history_delta_channel() -> None:
    graph = _build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "sub-history"}}
    graph.invoke({"messages": []}, config)

    history = list(graph.get_state_history(_child_config(graph, config)))
    # newest first: after `two`, after `one`, before either ran
    assert [_contents(s.values) for s in history[:2]] == [["one", "two"], ["one"]]


async def test_subgraph_aget_state_delta_channel() -> None:
    graph = _build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "sub-aget-state"}}
    await graph.ainvoke({"messages": []}, config)

    child_state = await graph.aget_state(_child_config(graph, config))
    assert _contents(child_state.values) == ["one", "two"]


async def test_subgraph_aget_state_history_delta_channel() -> None:
    graph = _build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "sub-ahistory"}}
    await graph.ainvoke({"messages": []}, config)

    history = [s async for s in graph.aget_state_history(_child_config(graph, config))]
    assert [_contents(s.values) for s in history[:2]] == [["one", "two"], ["one"]]


def test_root_graph_get_state_delta_channel() -> None:
    """Control: a graph that owns its checkpointer was never affected."""
    graph = _build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "root"}}
    graph.invoke({"messages": []}, config)

    assert _contents(graph.get_state(config).values) == ["one", "two"]
    history = list(graph.get_state_history(config))
    assert _contents(history[0].values) == ["one", "two"]


def test_subgraph_update_state_delta_channel() -> None:
    """A forced snapshot must be built from the replayed state, not an empty one."""
    graph = _build_graph(InMemorySaver(), freq=3)
    config = {"configurable": {"thread_id": "sub-update"}}
    graph.invoke({"messages": []}, config)
    child_config = _child_config(graph, config)

    expected = ["one", "two"]
    for i in range(3):
        graph.update_state(
            child_config, {"messages": [AIMessage(content=f"u{i}", id=f"u{i}")]}
        )
        expected.append(f"u{i}")
        assert _contents(graph.get_state(child_config).values) == expected


async def test_subgraph_aupdate_state_delta_channel() -> None:
    graph = _build_graph(InMemorySaver(), freq=3)
    config = {"configurable": {"thread_id": "sub-aupdate"}}
    await graph.ainvoke({"messages": []}, config)
    child_config = _child_config(graph, config)

    expected = ["one", "two"]
    for i in range(3):
        await graph.aupdate_state(
            child_config, {"messages": [AIMessage(content=f"u{i}", id=f"u{i}")]}
        )
        expected.append(f"u{i}")
        assert _contents((await graph.aget_state(child_config)).values) == expected
