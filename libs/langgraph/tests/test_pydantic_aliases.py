"""Tests for Pydantic field aliases support in LangGraph."""
import pytest
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END


def test_pydantic_field_aliases_basic():
    """Test basic Pydantic field alias support."""
    
    class State(BaseModel):
        foo: str = Field(alias='bar')
    
    def node(state: State) -> dict:
        return {"bar": state.foo + "_processed"}
    
    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()
    
    # Should work with alias
    result = compiled.invoke({"bar": "hello"})
    assert result["bar"] == "hello_processed"


def test_pydantic_field_aliases_multiple():
    """Test multiple field aliases."""
    
    class State(BaseModel):
        field1: str = Field(alias='alias1')
        field2: int = Field(alias='alias2')
        normal_field: str
    
    def node(state: State) -> dict:
        return {
            "alias1": state.field1 + "_mod",
            "alias2": state.field2 + 1,
            "normal_field": state.normal_field + "_norm"
        }
    
    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()
    
    result = compiled.invoke({
        "alias1": "test",
        "alias2": 42,
        "normal_field": "normal"
    })
    
    assert result["alias1"] == "test_mod"
    assert result["alias2"] == 43
    assert result["normal_field"] == "normal_norm"


def test_pydantic_field_aliases_backwards_compatibility():
    """Test that non-aliased fields still work."""
    
    class State(BaseModel):
        regular_field: str
    
    def node(state: State) -> dict:
        return {"regular_field": state.regular_field + "_processed"}
    
    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()
    
    result = compiled.invoke({"regular_field": "test"})
    assert result["regular_field"] == "test_processed"


def test_pydantic_field_aliases_mixed():
    """Test mix of aliased and non-aliased fields."""
    
    class State(BaseModel):
        aliased_field: str = Field(alias='my_alias')
        normal_field: str
    
    def node(state: State) -> dict:
        return {
            "my_alias": state.aliased_field + "_aliased",
            "normal_field": state.normal_field + "_normal"
        }
    
    graph = StateGraph(State)
    graph.add_node("process", node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    compiled = graph.compile()
    
    result = compiled.invoke({
        "my_alias": "alias_value",
        "normal_field": "normal_value"
    })
    
    assert result["my_alias"] == "alias_value_aliased"
    assert result["normal_field"] == "normal_value_normal"
