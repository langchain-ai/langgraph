"""Regression tests for silent drop of undeclared node output keys (#8320)."""

from __future__ import annotations

import warnings
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


def test_undeclared_node_output_key_emits_user_warning() -> None:
    class State(TypedDict):
        x: int

    def node(state: State) -> dict:
        return {"x": 1, "undeclared_key": "this was dropped silently"}

    graph = StateGraph(State)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    app = graph.compile()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = app.invoke({"x": 0})

    assert result == {"x": 1}
    undeclared_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and "undeclared_key" in str(w.message)
        and "not declared" in str(w.message)
    ]
    assert undeclared_warnings, "expected UserWarning for dropped undeclared state keys"
    assert "Node 'n'" in str(undeclared_warnings[0].message)


def test_declared_keys_only_do_not_warn() -> None:
    class State(TypedDict):
        x: int

    def node(state: State) -> dict:
        return {"x": 2}

    graph = StateGraph(State)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    app = graph.compile()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = app.invoke({"x": 0})

    assert result == {"x": 2}
    assert not [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "not declared" in str(w.message)
    ]
