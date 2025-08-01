"""Tests for Pydantic default values with reducer functions (Issue #5225)."""

import operator
from typing import Annotated, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

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


# Backward Compatibility Tests
def test_backward_compatibility_existing_reducer_behavior():
    """Test that existing reducer behavior without defaults still works."""
    
    class State(BaseModel):
        items: Annotated[list[str], operator.add]
        count: Annotated[int, operator.add]

    def node1(state: State) -> dict:
        return {"items": ["first"], "count": 1}
    
    def node2(state: State) -> dict:
        return {"items": ["second"], "count": 2}

    graph = StateGraph(State)
    graph.add_node("node1", node1)
    graph.add_node("node2", node2)
    graph.add_edge(START, "node1")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", END)
    compiled = graph.compile()

    # Provide initial values as before - this should work without requiring defaults
    result = compiled.invoke({"items": [], "count": 0})
    assert result["items"] == ["first", "second"]
    assert result["count"] == 3


def test_backward_compatibility_without_pydantic_models():
    """Test that non-Pydantic state schemas still work."""
    
    class State(TypedDict):
        items: Annotated[list[str], operator.add]
        count: Annotated[int, operator.add]

    def node(state: State) -> dict:
        return {"items": ["item"], "count": 5}

    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()

    # Should work exactly as before
    result = compiled.invoke({"items": ["initial"], "count": 10})
    assert result["items"] == ["initial", "item"]
    assert result["count"] == 15


def test_backward_compatibility_regular_channels():
    """Test that regular channels (non-reducer) still work with Pydantic."""
    
    class State(BaseModel):
        name: str = Field(default="default_name")
        age: int = Field(default=25)
        tags: list[str] = Field(default_factory=lambda: ["default"])

    def node(state: State) -> dict:
        return {
            "name": "updated_name",
            "age": 30,
            "tags": ["updated"]  # This overwrites, doesn't reduce
        }

    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()

    result = compiled.invoke({})
    assert result["name"] == "updated_name"
    assert result["age"] == 30
    assert result["tags"] == ["updated"]  # Should be overwritten, not extended


def test_backward_compatibility_mixed_annotations():
    """Test mixing Annotated reducers with regular fields."""
    
    class State(BaseModel):
        # Regular fields with defaults
        config: str = Field(default="default_config")
        version: int = Field(default=1)
        
        # Reducer fields with defaults
        logs: Annotated[list[str], operator.add] = Field(default_factory=lambda: ["start"])
        total: Annotated[int, operator.add] = Field(default=0)
        
        # Regular fields without defaults
        status: str

    def node(state: State) -> dict:
        return {
            "config": "updated_config",  # Overwrites
            "version": 2,  # Overwrites
            "logs": ["new_log"],  # Adds to existing
            "total": 5,  # Adds to existing
            "status": "running"  # Sets value
        }

    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()

    result = compiled.invoke({"status": "pending"})
    assert result["config"] == "updated_config"
    assert result["version"] == 2
    assert result["logs"] == ["start", "new_log"]
    assert result["total"] == 5  # 0 + 5
    assert result["status"] == "running"


def test_backward_compatibility_custom_reducer_functions():
    """Test that custom reducer functions continue to work."""
    
    def merge_dicts(current: dict, update: dict) -> dict:
        """Custom reducer that merges dictionaries."""
        result = current.copy()
        result.update(update)
        return result

    class State(BaseModel):
        metadata: Annotated[dict, merge_dicts] = Field(default_factory=lambda: {"source": "default"})
        items: list[str] = Field(default_factory=list)

    def node(state: State) -> dict:
        return {
            "metadata": {"author": "user", "timestamp": "2024"},
            "items": ["new_item"]  # Regular field, overwrites
        }

    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()

    result = compiled.invoke({})
    assert result["metadata"] == {"source": "default", "author": "user", "timestamp": "2024"}
    assert result["items"] == ["new_item"]


def test_backward_compatibility_explicit_initial_values():
    """Test the interaction between defaults and explicitly provided initial values.
    
    In LangGraph, when both defaults and initial values are provided for reducer fields,
    the defaults are used as the starting point and initial values are reduced in,
    maintaining consistency with how all values flow through the reducer logic.
    """
    
    class State(BaseModel):
        messages: Annotated[list[str], operator.add] = Field(default_factory=lambda: ["default"])
        count: Annotated[int, operator.add] = Field(default=10)

    def node(state: State) -> dict:
        return {"messages": ["new"], "count": 5}

    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()

    # Test 1: With no initial values, defaults should be used
    result_empty = compiled.invoke({})
    assert result_empty["messages"] == ["default", "new"]  # default + new
    assert result_empty["count"] == 15  # 10 + 5

    # Test 2: With explicit initial values, they are combined with defaults using the reducer
    # This is the expected LangGraph behavior: initial values go through the same reducer logic
    # as node updates, so defaults are preserved and combined with provided values
    result_provided = compiled.invoke({"messages": ["provided"], "count": 20})
    # Behavior: defaults are used as the base, then initial values are reduced in
    # messages: ["default"] (default) + ["provided"] (initial) + ["new"] (node update)
    # count: 10 (default) + 20 (initial) + 5 (node update)
    
    assert result_provided["messages"] == ["default", "provided", "new"]
    assert result_provided["count"] == 35  # 10 + 20 + 5
    
    # This behavior ensures that defaults work consistently whether values come from
    # initial input or node updates - both go through the same reducer logic


def test_backward_compatibility_complex_types():
    """Test backward compatibility with complex types and nested structures."""
    
    
    class State(BaseModel):
        data: Annotated[list[dict[str, int]], operator.add] = Field(default_factory=lambda: [{"initial": 0}])
        optional_field: Optional[str] = Field(default=None)

    def node(state: State) -> dict:
        return {
            "data": [{"added": 1}],
            "optional_field": "set_value"
        }

    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()

    result = compiled.invoke({})
    assert result["data"] == [{"initial": 0}, {"added": 1}]
    assert result["optional_field"] == "set_value"
