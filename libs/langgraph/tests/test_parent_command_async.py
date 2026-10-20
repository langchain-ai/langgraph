from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

pytestmark = pytest.mark.anyio


async def test_parent_command_from_nested_subgraph() -> None:
    class ParentState(TypedDict):
        jump_from_idx: int

    class ChildState(TypedDict):
        jump: bool

    child_builder: StateGraph[ChildState] = StateGraph(ChildState)

    async def child_node(state: ChildState) -> Command | ChildState:
        if state["jump"]:
            return Command(graph=Command.PARENT, goto="parent_second")
        return state

    child_builder.add_node("node", child_node)
    child_builder.add_edge(START, "node")

    child_0 = child_builder.compile()
    child_1 = child_builder.compile()

    parent_builder: StateGraph[ParentState] = StateGraph(ParentState)

    async def parent_first(state: ParentState, config: RunnableConfig) -> ParentState:
        await child_0.ainvoke({"jump": state["jump_from_idx"] == 1}, config)
        if state["jump_from_idx"] == 1:
            raise AssertionError("Shouldn't be here")

        await child_1.ainvoke({"jump": state["jump_from_idx"] == 2}, config)
        if state["jump_from_idx"] == 2:
            raise AssertionError("Shouldn't be here")

        return state

    async def parent_second(state: ParentState) -> ParentState:
        return state

    parent_builder.add_node("parent_first", parent_first)
    parent_builder.add_node("parent_second", parent_second)
    parent_builder.add_edge(START, "parent_first")
    parent_builder.add_edge("parent_second", END)

    graph = parent_builder.compile().with_config(recursion_limit=10)

    assert await graph.ainvoke({"jump_from_idx": 1}) == {"jump_from_idx": 1}
    assert await graph.ainvoke({"jump_from_idx": 2}) == {"jump_from_idx": 2}


@pytest.mark.parametrize("child_checkpointer", [True, None])
async def test_parent_command_survives_a_retried_checkpointed_subgraph(
    child_checkpointer: bool | None,
) -> None:
    """A `Command.PARENT` from a retried `checkpointer=True` subgraph still reaches its parent.

    Retrying the enclosing node calls the subgraph again under a numeric namespace
    segment, and a `checkpointer=True` subgraph keeps that segment while dropping the
    task ids. The command is addressed to that form, so it has to be matched against it.
    """

    class State(TypedDict):
        x: int

    attempts: list[int] = []

    async def flaky(state: State) -> Command:
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("transient")
        return Command(update={"x": state["x"] + 1}, goto="done", graph=Command.PARENT)

    inner = StateGraph(State)
    inner.add_node("flaky", flaky)
    inner.add_edge(START, "flaky")
    inner = inner.compile(checkpointer=child_checkpointer)

    async def done(state: State) -> State:
        return state

    middle = StateGraph(State)
    middle.add_node("inner", inner, destinations=("done",))
    middle.add_node("done", done)
    middle.add_edge(START, "inner")
    middle.add_edge("done", END)
    middle = middle.compile(checkpointer=child_checkpointer)

    outer = StateGraph(State)
    outer.add_node(
        "middle",
        middle,
        retry_policy=RetryPolicy(
            max_attempts=2, initial_interval=0, jitter=False, retry_on=ValueError
        ),
    )
    outer.add_edge(START, "middle")
    outer.add_edge("middle", END)
    graph = outer.compile(checkpointer=InMemorySaver())

    assert await graph.ainvoke({"x": 0}, {"configurable": {"thread_id": "1"}}) == {
        "x": 1
    }
    assert len(attempts) == 2
