#!/usr/bin/env python3
"""Test script to verify reserved keyword injection works correctly."""

from typing import Any
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import AIMessage, ToolCall


# Test tool with reserved keyword 'state'
@tool
def tool_with_state(x: int, state) -> str:
    """Tool that uses reserved keyword 'state'."""
    return f"x={x}, state_keys={list(state.keys()) if isinstance(state, dict) else 'not_dict'}"


# Test tool with reserved keyword 'runtime'
@tool
def tool_with_runtime(x: int, runtime) -> str:
    """Tool that uses reserved keyword 'runtime'."""
    has_store = runtime.store is not None if hasattr(runtime, 'store') else False
    return f"x={x}, has_store={has_store}"


# Test tool with both reserved keywords
@tool
def tool_with_both(x: int, state, runtime) -> str:
    """Tool that uses both reserved keywords."""
    has_store = runtime.store is not None if hasattr(runtime, 'store') else False
    return f"x={x}, state_keys={list(state.keys()) if isinstance(state, dict) else 'not_dict'}, has_store={has_store}"


# Test regular tool without injection
@tool
def regular_tool(x: int, y: str) -> str:
    """Regular tool without injection."""
    return f"x={x}, y={y}"


def test_reserved_keywords():
    """Test that reserved keywords work correctly."""
    
    # Create ToolNode with all test tools
    tools = [tool_with_state, tool_with_runtime, tool_with_both, regular_tool]
    node = ToolNode(tools)
    
    # Check that reserved keywords are detected
    print("Tool to state args:", node.tool_to_state_args)
    print("Tool to runtime args:", node.tool_to_runtime_arg)
    
    # Check tool schemas - reserved keywords should be excluded
    for tool_name, tool_obj in node.tools_by_name.items():
        schema = tool_obj.get_input_schema()
        print(f"\n{tool_name} schema fields:", list(schema.__fields__.keys()))
        
        # Verify reserved keywords are not in the schema
        if tool_name == "tool_with_state":
            assert "state" not in schema.__fields__, f"'state' should be excluded from {tool_name} schema"
        elif tool_name == "tool_with_runtime":
            assert "runtime" not in schema.__fields__, f"'runtime' should be excluded from {tool_name} schema"
        elif tool_name == "tool_with_both":
            assert "state" not in schema.__fields__, f"'state' should be excluded from {tool_name} schema"
            assert "runtime" not in schema.__fields__, f"'runtime' should be excluded from {tool_name} schema"
    
    print("\nAll schema checks passed!")
    
    # Test actual injection
    store = InMemoryStore()
    state = {"messages": [], "foo": "bar"}
    
    # Create tool calls
    tool_call1: ToolCall = {
        "name": "tool_with_state",
        "args": {"x": 1},
        "id": "1",
        "type": "tool_call"
    }
    
    tool_call2: ToolCall = {
        "name": "tool_with_runtime", 
        "args": {"x": 2},
        "id": "2",
        "type": "tool_call"
    }
    
    tool_call3: ToolCall = {
        "name": "regular_tool",
        "args": {"x": 3, "y": "test"},
        "id": "3",
        "type": "tool_call"
    }
    
    # Test injection
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(configurable={"context": {"user_id": "test_user"}})
    
    injected1 = node.inject_tool_args(tool_call1, state, store, config)
    print(f"\nInjected args for tool_with_state: {injected1['args']}")
    assert "state" in injected1["args"], "State should be injected"
    
    injected2 = node.inject_tool_args(tool_call2, state, store, config)
    print(f"Injected args for tool_with_runtime: {injected2['args']}")
    assert "runtime" in injected2["args"], "Runtime should be injected"
    
    injected3 = node.inject_tool_args(tool_call3, state, store, config)
    print(f"Injected args for regular_tool: {injected3['args']}")
    assert "state" not in injected3["args"], "State should not be injected"
    assert "runtime" not in injected3["args"], "Runtime should not be injected"
    
    print("\nAll injection tests passed!")


if __name__ == "__main__":
    test_reserved_keywords()

