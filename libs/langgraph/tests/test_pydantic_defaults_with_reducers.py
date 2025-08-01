"""Tests for Pydantic default values with reducer functions (Issue #5225)."""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from langgraph.graph import END, START, StateGraph


def test_pydantic_default_factory_with_reducer():
    """Test that default_factory works with Annotated reducer functions."""
    
    def extend_list(original: list, new: list):
        original.extend(new)
        return original

    class State(BaseModel):
        variable: Annotated[list[str], extend_list] = Field(default_factory=lambda: ["default"])

    def node(state: State) -> dict:
        return {"variable": ["new_item"]}

    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()

    # Test with empty input - should start with default value
    result = compiled.invoke({})
    # Should have both default and new item
    assert "default" in result["variable"]
    assert "new_item" in result["variable"]


def test_pydantic_default_value_with_reducer():
    """Test that default values work with Annotated reducer functions."""
    
    def string_concat(original: str, new: str):
        return original + new

    class State(BaseModel):
        text: Annotated[str, string_concat] = Field(default="start")

    def node(state: State) -> dict:
        return {"text": "_end"}

    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()

    # Test with empty input - should start with default value
    result = compiled.invoke({})
    assert result["text"] == "start_end"


def test_pydantic_operator_add_default_factory():
    """Test operator.add with default_factory."""
    
    class State(BaseModel):
        messages: Annotated[list[str], operator.add] = Field(default_factory=lambda: ["initial"])

    def node(state: State) -> dict:
        return {"messages": ["added"]}

    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()

    result = compiled.invoke({})
    assert result["messages"] == ["initial", "added"]


def test_pydantic_mixed_defaults_with_reducers():
    """Test mix of default values and reducers."""
    
    class State(BaseModel):
        items: Annotated[list[int], operator.add] = Field(default_factory=lambda: [1, 2])
        count: Annotated[int, operator.add] = Field(default=0)
        name: str = Field(default="test")

    def node(state: State) -> dict:
        return {
            "items": [3],
            "count": 5,
            "name": state.name + "_modified"
        }

    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()

    result = compiled.invoke({})
    assert result["items"] == [1, 2, 3]
    assert result["count"] == 5  # 0 + 5
    assert result["name"] == "test_modified"


def test_pydantic_no_defaults_provided():
    """Test behavior when no defaults are provided in input."""
    
    class State(BaseModel):
        values: Annotated[list[str], operator.add] = Field(default_factory=list)

    def node(state: State) -> dict:
        # Only add to existing list, don't provide initial value
        return {"values": ["item1"]}

    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()

    # Should work even with completely empty input
    result = compiled.invoke({})
    assert result["values"] == ["item1"]
