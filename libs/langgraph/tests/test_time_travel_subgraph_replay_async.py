"""Async regression tests for time travel into a subgraph checkpoint.

Async counterpart of ``test_time_travel_subgraph_replay.py`` for #8458:
replaying from a checkpoint *inside* a subgraph must resume at the selected
node, not silently re-run the whole subgraph from __start__.
"""

import operator
from typing import Annotated

import pytest
from typing_extensions import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

pytestmark = pytest.mark.anyio


class State(TypedDict):
    value: Annotated[list[str], operator.add]


async def test_subgraph_replay_from_mid_subgraph_checkpoint_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from a checkpoint *inside* a subgraph (next node "c") must
    resume at "c" and NOT re-run "a"/"b" (async variant).

    Regression test for #8458.
    """

    ran: list[str] = []

    def mk(name: str):
        def fn(state: State) -> State:
            ran.append(name)
            return {"value": [name]}

        fn.__name__ = name
        return fn

    async def review(state: State) -> State:
        ran.append("review")
        answer = interrupt("approve?")
        return {"value": [f"review:{answer}"]}

    subgraph = (
        StateGraph(State)
        .add_node("a", mk("a"))
        .add_node("b", mk("b"))
        .add_node("c", mk("c"))
        .add_node("review", review)
        .add_edge(START, "a")
        .add_edge("a", "b")
        .add_edge("b", "c")
        .add_edge("c", "review")
        .add_edge("review", END)
        .compile(checkpointer=True)
    )

    graph = (
        StateGraph(State)
        .add_node("pre", mk("pre"))
        .add_node("child", subgraph)
        .add_edge(START, "pre")
        .add_edge("pre", "child")
        .add_edge("child", END)
        .compile(checkpointer=async_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    result = await graph.ainvoke({"value": []}, config)
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value == "approve?"

    parent_state = await graph.aget_state(config, subgraphs=True)
    ns_before = parent_state.tasks[0].state.config["configurable"]["checkpoint_ns"]
    target = None
    async for h in graph.aget_state_history(
        {"configurable": {"thread_id": "1", "checkpoint_ns": ns_before}}
    ):
        if h.next == ("c",):
            target = h
            break
    assert target is not None, "expected a subgraph checkpoint with next == ('c',)"

    ran.clear()
    replay = await graph.ainvoke(None, target.config)
    assert "__interrupt__" in replay
    assert replay["__interrupt__"][0].value == "approve?"
    assert ran == ["c", "review"], f"expected ['c', 'review'], got {ran}"
