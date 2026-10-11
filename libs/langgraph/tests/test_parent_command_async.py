from __future__ import annotations

import operator
from typing import Annotated

import pytest
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

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


async def test_parent_command_from_subgraph_error_handler() -> None:
    class State(TypedDict):
        log: Annotated[list[str], operator.add]

    async def fail(state: State) -> State:
        raise RuntimeError("child failed")

    async def handle(state: State, error: NodeError) -> Command:
        # Route to the parent graph from the error handler.
        return Command(
            update={"log": [f"handled:{error.node}"]},
            goto="recover",
            graph=Command.PARENT,
        )

    async def recover(state: State) -> State:
        return {"log": ["recovered"]}

    child_builder: StateGraph[State] = StateGraph(State)
    child_builder.add_node("fail", fail, error_handler=handle)
    child_builder.add_edge(START, "fail")
    child = child_builder.compile()

    parent_builder: StateGraph[State] = StateGraph(State)

    async def child_entry(state: State, config: RunnableConfig) -> State:
        await child.ainvoke(state, config)
        return state

    parent_builder.add_node("child_entry", child_entry, destinations=("recover",))
    parent_builder.add_node("recover", recover)
    parent_builder.add_edge(START, "child_entry")
    parent_builder.add_edge("recover", END)

    graph = parent_builder.compile()

    assert await graph.ainvoke({"log": []}) == {"log": ["handled:fail", "recovered"]}
