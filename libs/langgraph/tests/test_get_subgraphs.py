"""Tests for Pregel.get_subgraphs with common-prefix node names.

Regression test for https://github.com/langchain-ai/langgraph/issues/6924
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph


class State(TypedDict):
    value: str


def _node(state: State) -> State:
    return state


def _make_parent_graph():
    """Build a parent graph with two subgraph nodes sharing a common prefix."""
    grandchild_builder = StateGraph(State)
    grandchild_builder.add_node("grandchild", _node)
    grandchild_builder.set_entry_point("grandchild")
    grandchild_builder.add_edge("grandchild", END)
    grandchild_graph = grandchild_builder.compile()

    child_builder = StateGraph(State)
    child_builder.add_node("child_subgraph", grandchild_graph)
    child_builder.set_entry_point("child_subgraph")
    child_builder.add_edge("child_subgraph", END)
    child_graph = child_builder.compile()

    parent_builder = StateGraph(State)
    parent_builder.add_node("common_prefix", child_graph)
    parent_builder.add_node("common_prefix_2", child_graph)
    parent_builder.set_entry_point("common_prefix")
    parent_builder.add_edge("common_prefix", "common_prefix_2")
    parent_builder.add_edge("common_prefix_2", END)
    return parent_builder.compile()


def test_get_subgraphs_common_prefix_recursive_namespace():
    """get_subgraphs must not confuse nodes whose names share a common prefix."""
    graph = _make_parent_graph()
    result = list(
        graph.get_subgraphs(
            namespace="common_prefix_2|child_subgraph", recurse=True
        )
    )
    assert len(result) == 1
    assert result[0][0] == "common_prefix_2|child_subgraph"


def test_get_subgraphs_first_node_recursive():
    graph = _make_parent_graph()
    result = list(
        graph.get_subgraphs(
            namespace="common_prefix|child_subgraph", recurse=True
        )
    )
    assert len(result) == 1
    assert result[0][0] == "common_prefix|child_subgraph"


def test_get_subgraphs_no_namespace():
    graph = _make_parent_graph()
    result = list(graph.get_subgraphs())
    assert len(result) == 2


def test_get_subgraphs_exact_namespace():
    graph = _make_parent_graph()
    for name in ("common_prefix", "common_prefix_2"):
        result = list(graph.get_subgraphs(namespace=name))
        assert len(result) == 1
        assert result[0][0] == name


def test_get_subgraphs_recurse_no_namespace():
    graph = _make_parent_graph()
    result = list(graph.get_subgraphs(recurse=True))
    names = [n for n, _ in result]
    assert len(result) == 4
    assert "common_prefix" in names
    assert "common_prefix_2" in names
    assert "common_prefix|child_subgraph" in names
    assert "common_prefix_2|child_subgraph" in names
