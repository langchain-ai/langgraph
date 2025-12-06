"""Tests for node output validation (Issue #6491)"""

from typing import List

import pytest
from pydantic import BaseModel

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph


class State(BaseModel):
    items: List[str] = []


def test_node_output_validation_invalid():
    """Test that invalid node output is caught immediately"""

    def create_invalid_state(state: State) -> dict:
        """Node that returns invalid state"""
        return {"items": state.items + [None]}  # None is not a valid str!

    graph = StateGraph(State)
    graph.add_node("bad", create_invalid_state)
    graph.add_edge(START, "bad")
    graph.add_edge("bad", END)

    app = graph.compile()

    # Should raise InvalidUpdateError when node returns invalid output
    with pytest.raises(InvalidUpdateError) as exc_info:
        app.invoke(State(items=["hello"]))

    # Error message should mention the node name
    assert "bad" in str(exc_info.value).lower()
    assert "invalid state" in str(exc_info.value).lower()


def test_node_output_validation_valid():
    """Test that valid node output passes validation"""

    def create_valid_state(state: State) -> dict:
        """Node that returns valid state"""
        return {"items": state.items + ["world"]}

    graph = StateGraph(State)
    graph.add_node("good", create_valid_state)
    graph.add_edge(START, "good")
    graph.add_edge("good", END)

    app = graph.compile()

    # Should succeed
    result = app.invoke(State(items=["hello"]))
    assert result["items"] == ["hello", "world"]


def test_node_output_validation_with_multiple_nodes():
    """Test that validation works across multiple nodes"""

    def node1(state: State) -> dict:
        return {"items": state.items + ["valid1"]}

    def invalid_node(state: State) -> dict:
        return {"items": state.items + [123]}  # Invalid!

    def node3(state: State) -> dict:
        return {"items": state.items + ["valid3"]}

    graph = StateGraph(State)
    graph.add_node("node1", node1)
    graph.add_node("invalid", invalid_node)
    graph.add_node("node3", node3)
    graph.add_edge(START, "node1")
    graph.add_edge("node1", "invalid")
    graph.add_edge("invalid", "node3")
    graph.add_edge("node3", END)

    app = graph.compile()

    # Should fail at invalid_node
    with pytest.raises(InvalidUpdateError) as exc_info:
        app.invoke(State())

    # Error should mention the invalid node
    assert "invalid" in str(exc_info.value).lower()


def test_node_output_validation_pydantic_model_return():
    """Test that returning Pydantic model directly still validates"""

    def return_pydantic(state: State) -> State:
        """Node that returns Pydantic model"""
        state.items.append(None)  # Modify and make invalid!
        return state

    graph = StateGraph(State)
    graph.add_node("bad_pydantic", return_pydantic)
    graph.add_edge(START, "bad_pydantic")
    graph.add_edge("bad_pydantic", END)

    app = graph.compile()

    # Should catch the validation error
    with pytest.raises(Exception):  # Could be ValidationError or InvalidUpdateError
        app.invoke(State(items=["hello"]))
