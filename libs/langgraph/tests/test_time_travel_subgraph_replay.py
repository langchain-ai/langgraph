"""Regression tests for time travel into a subgraph checkpoint.

When a client replays from a *subgraph* checkpoint (e.g. LangGraph Studio's
"Re-run from here" against a node inside a subgraph), the config carries the
subgraph's checkpoint id + namespace. The parent loop eagerly forks a new
checkpoint (#7498) which changes the subgraph task id, so the replayed subgraph
lands in a brand-new, empty namespace and the requested checkpoint is orphaned —
the whole subgraph silently re-executes from __start__.

See https://github.com/langchain-ai/langgraph/issues/8458
"""

import operator
from typing import Annotated

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict


class State(TypedDict):
    value: Annotated[list[str], operator.add]


def test_subgraph_replay_from_mid_subgraph_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Replay from a checkpoint *inside* a subgraph (next node "c", before the
    interrupt node "review") must resume at "c" and NOT re-run the nodes that
    already executed ("a", "b").

    Regression test for #8458: the eager parent fork checkpoint changes the
    subgraph task id, orphaning the requested subgraph checkpoint and silently
    re-running the whole subgraph from __start__.
    """

    ran: list[str] = []

    def mk(name: str):
        def fn(state: State) -> State:
            ran.append(name)
            return {"value": [name]}

        fn.__name__ = name
        return fn

    def review(state: State) -> State:
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
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"value": []}, config)
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value == "approve?"

    # Find the subgraph checkpoint whose next node is "c" (a mid-subgraph
    # checkpoint with no pending interrupt).
    parent_state = graph.get_state(config, subgraphs=True)
    ns_before = parent_state.tasks[0].state.config["configurable"]["checkpoint_ns"]
    target = next(
        h
        for h in graph.get_state_history(
            {"configurable": {"thread_id": "1", "checkpoint_ns": ns_before}}
        )
        if h.next == ("c",)
    )

    # "Re-run from here" — replay from the mid-subgraph checkpoint.
    ran.clear()
    replay = graph.invoke(None, target.config)
    assert "__interrupt__" in replay
    assert replay["__interrupt__"][0].value == "approve?"
    # The subgraph must resume at "c": "a" and "b" must NOT re-run, and the
    # parent's "pre" must not re-run either.
    assert ran == ["c", "review"], f"expected ['c', 'review'], got {ran}"
