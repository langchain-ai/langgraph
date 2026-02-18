from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class CustomState(TypedDict):
    x: list[str]


def _build_child_graph(*, label: str) -> object:
    builder: StateGraph[CustomState] = StateGraph(CustomState)

    def only_node(state: CustomState) -> Command | CustomState:
        if state["x"] and state["x"][-1] == "jump":
            return Command(graph=Command.PARENT, goto="second")
        return {"x": [*state["x"], f"{label}:stay"]}

    builder.add_node("only_node", only_node)
    builder.add_edge(START, "only_node")
    return builder.compile()


def _build_parent_graph() -> object:
    child_0 = _build_child_graph(label="child0")
    child_1 = _build_child_graph(label="child1")

    builder: StateGraph[CustomState] = StateGraph(CustomState)

    def first(state: CustomState) -> CustomState:
        child_0.invoke({"x": ["1"]})
        child_1.invoke({"x": ["jump"]})
        return {"x": [*state["x"], "first:done"]}

    def second(state: CustomState) -> CustomState:
        return {"x": [*state["x"], "second"]}

    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("second", END)

    return builder.compile().with_config(recursion_limit=10)


def test_parent_command_from_first_nested_invoke_jumps_to_second() -> None:
    builder: StateGraph[CustomState] = StateGraph(CustomState)
    child_0 = _build_child_graph(label="child0")

    def first(state: CustomState) -> CustomState:
        child_0.invoke({"x": ["jump"]})
        return {"x": [*state["x"], "first:done"]}

    def second(state: CustomState) -> CustomState:
        return {"x": [*state["x"], "second"]}

    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("second", END)

    graph = builder.compile().with_config(recursion_limit=10)
    assert graph.invoke({"x": ["init"]}) == {"x": ["init", "second"]}


def test_parent_command_from_second_nested_invoke_jumps_to_second() -> None:
    graph = _build_parent_graph()
    assert graph.invoke({"x": ["init"]}) == {"x": ["init", "second"]}
