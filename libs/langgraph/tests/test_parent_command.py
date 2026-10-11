from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


def test_parent_command_from_nested_subgraph() -> None:
    class ParentState(TypedDict):
        jump_from_idx: int

    class ChildState(TypedDict):
        jump: bool

    child_builder: StateGraph[ChildState] = StateGraph(ChildState)

    def child_node(state: ChildState) -> Command | ChildState:
        if state["jump"]:
            return Command(graph=Command.PARENT, goto="parent_second")
        return state

    child_builder.add_node("node", child_node)
    child_builder.add_edge(START, "node")

    child_0 = child_builder.compile()
    child_1 = child_builder.compile()

    parent_builder: StateGraph[ParentState] = StateGraph(ParentState)

    def parent_first(state: ParentState) -> ParentState:
        child_0.invoke({"jump": state["jump_from_idx"] == 1})
        if state["jump_from_idx"] == 1:
            raise AssertionError("Shouldn't be here")

        child_1.invoke({"jump": state["jump_from_idx"] == 2})
        if state["jump_from_idx"] == 2:
            raise AssertionError("Shouldn't be here")

        return state

    def parent_second(state: ParentState) -> ParentState:
        return state

    parent_builder.add_node("parent_first", parent_first)
    parent_builder.add_node("parent_second", parent_second)
    parent_builder.add_edge(START, "parent_first")
    parent_builder.add_edge("parent_second", END)

    graph = parent_builder.compile()

    assert graph.invoke({"jump_from_idx": 1}) == {"jump_from_idx": 1}
    assert graph.invoke({"jump_from_idx": 2}) == {"jump_from_idx": 2}


def test_parent_command_from_subgraph_error_handler() -> None:
    class State(TypedDict):
        log: Annotated[list[str], operator.add]

    def fail(state: State) -> State:
        raise RuntimeError("child failed")

    def handle(state: State, error: NodeError) -> Command:
        # Route to the parent graph from the error handler.
        return Command(
            update={"log": [f"handled:{error.node}"]},
            goto="recover",
            graph=Command.PARENT,
        )

    def recover(state: State) -> State:
        return {"log": ["recovered"]}

    child_builder: StateGraph[State] = StateGraph(State)
    child_builder.add_node("fail", fail, error_handler=handle)
    child_builder.add_edge(START, "fail")
    child = child_builder.compile()

    parent_builder: StateGraph[State] = StateGraph(State)
    parent_builder.add_node("child", child, destinations=("recover",))
    parent_builder.add_node("recover", recover)
    parent_builder.add_edge(START, "child")
    parent_builder.add_edge("recover", END)

    graph = parent_builder.compile()

    assert graph.invoke({"log": []}) == {"log": ["handled:fail", "recovered"]}
