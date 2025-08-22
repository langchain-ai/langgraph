"""Test reserved keywords for tool injection."""
import pytest
from typing import Annotated, List
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.prebuilt import ToolNode, InjectedState, InjectedStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.types import MessagesState


def test_tool_node_inject_runtime_reserved_keyword() -> None:
    """Test that tools can use 'runtime' as a reserved keyword parameter."""
    from langgraph.runtime import Runtime
    
    def tool1(some_val: int, runtime) -> str:
        """Tool 1 with reserved keyword 'runtime'."""
        assert isinstance(runtime, Runtime), "runtime should be a Runtime instance"
        # Access store from runtime
        if runtime.store:
            store_val = runtime.store.get(("test",), "test_key")
            if store_val:
                return f"val: {some_val}, store: {store_val.value['foo']}"
        return f"val: {some_val}, no store"

    def tool2(some_val: int, runtime) -> str:
        """Tool 2 with reserved keyword 'runtime'."""
        assert isinstance(runtime, Runtime), "runtime should be a Runtime instance"
        # Access context from runtime
        if runtime.context:
            return f"val: {some_val}, context: {runtime.context.get('user_id', 'unknown')}"
        return f"val: {some_val}, no context"

    def tool3(x: int, y: str, runtime) -> str:
        """Tool 3 with reserved keyword 'runtime' and other params."""
        assert isinstance(runtime, Runtime), "runtime should be a Runtime instance"
        has_store = "yes" if runtime.store else "no"
        has_context = "yes" if runtime.context else "no"
        return f"x: {x}, y: {y}, store: {has_store}, context: {has_context}"

    store = InMemoryStore()
    store.put(("test",), "test_key", {"foo": "bar"})
    
    node = ToolNode([tool1, tool2, tool3])
    
    # Verify that 'runtime' is excluded from tool schemas
    for tool in [tool1, tool2, tool3]:
        schema = node.tools_by_name[tool.__name__].get_input_schema()
        if hasattr(schema, 'model_fields'):
            assert "runtime" not in schema.model_fields, f"'runtime' should be excluded from {tool.__name__} schema"
        else:
            assert "runtime" not in schema.__fields__, f"'runtime' should be excluded from {tool.__name__} schema"
    
    # Test with store
    tool_call = {
        "name": "tool1",
        "args": {"some_val": 1},
        "id": "some 0",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])
    result = node.invoke({"messages": [msg]}, store=store)
    tool_message = result["messages"][-1]
    assert tool_message.content == "val: 1, store: bar"
    
    # Test with context
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(configurable={"context": {"user_id": "test_user"}})
    
    tool_call = {
        "name": "tool2",
        "args": {"some_val": 2},
        "id": "some 1",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])
    result = node.invoke({"messages": [msg]}, config=config)
    tool_message = result["messages"][-1]
    assert tool_message.content == "val: 2, context: test_user"
    
    # Test with both store and context
    tool_call = {
        "name": "tool3",
        "args": {"x": 3, "y": "test"},
        "id": "some 2",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])
    result = node.invoke({"messages": [msg]}, store=store, config=config)
    tool_message = result["messages"][-1]
    assert tool_message.content == "x: 3, y: test, store: yes, context: yes"


def test_tool_node_mixed_injection_styles() -> None:
    """Test that tools can mix reserved keywords and annotations."""
    from langgraph.runtime import Runtime
    
    def tool1(some_val: int, state) -> str:
        """Tool with reserved keyword 'state'."""
        if isinstance(state, dict):
            return f"reserved state: {state['foo']}"
        else:
            return f"reserved state: {getattr(state, 'foo')}"
    
    def tool2(some_val: int, state: Annotated[dict, InjectedState]) -> str:
        """Tool with annotation-based state injection."""
        return f"annotated state: {state['foo']}"
    
    def tool3(some_val: int, runtime) -> str:
        """Tool with reserved keyword 'runtime'."""
        assert isinstance(runtime, Runtime)
        return f"reserved runtime: {runtime.context.get('user_id', 'none') if runtime.context else 'none'}"
    
    def tool4(some_val: int, store: Annotated[BaseStore, InjectedStore()]) -> str:
        """Tool with annotation-based store injection."""
        store_val = store.get(("test",), "test_key")
        return f"annotated store: {store_val.value['foo'] if store_val else 'none'}"
    
    def tool5(x: int, state, runtime) -> str:
        """Tool with both reserved keywords."""
        assert isinstance(runtime, Runtime)
        if isinstance(state, dict):
            return f"both: state={state['foo']}, runtime={runtime.context.get('user_id', 'none') if runtime.context else 'none'}"
        else:
            return f"both: state={getattr(state, 'foo')}, runtime={runtime.context.get('user_id', 'none') if runtime.context else 'none'}"
    
    store = InMemoryStore()
    store.put(("test",), "test_key", {"foo": "bar"})
    
    node = ToolNode([tool1, tool2, tool3, tool4, tool5])
    
    # Verify schemas exclude injected parameters
    for tool_name, expected_excluded in [
        ("tool1", ["state"]),
        ("tool2", ["state"]),
        ("tool3", ["runtime"]),
        ("tool4", ["store"]),
        ("tool5", ["state", "runtime"]),
    ]:
        schema = node.tools_by_name[tool_name].get_input_schema()
        if hasattr(schema, 'model_fields'):
            fields = schema.model_fields
        else:
            fields = schema.__fields__
        for param in expected_excluded:
            assert param not in fields, f"'{param}' should be excluded from {tool_name} schema"
    
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(configurable={"context": {"user_id": "test_user"}})
    
    # Test each tool
    test_cases = [
        ("tool1", {"some_val": 1}, "reserved state: baz"),
        ("tool2", {"some_val": 2}, "annotated state: baz"),
        ("tool3", {"some_val": 3}, "reserved runtime: test_user"),
        ("tool4", {"some_val": 4}, "annotated store: bar"),
        ("tool5", {"x": 5}, "both: state=baz, runtime=test_user"),
    ]
    
    for tool_name, args, expected in test_cases:
        tool_call = {
            "name": tool_name,
            "args": args,
            "id": f"id_{tool_name}",
            "type": "tool_call",
        }
        msg = AIMessage("test", tool_calls=[tool_call])
        result = node.invoke({"messages": [msg], "foo": "baz"}, store=store, config=config)
        tool_message = result["messages"][-1]
        assert tool_message.content == expected, f"Failed for {tool_name}: got {tool_message.content}, expected {expected}"


if __name__ == "__main__":
    print("Testing runtime reserved keyword...")
    test_tool_node_inject_runtime_reserved_keyword()
    print("✓ Runtime reserved keyword test passed!")
    
    print("\nTesting mixed injection styles...")
    test_tool_node_mixed_injection_styles()
    print("✓ Mixed injection styles test passed!")
    
    print("\nAll tests passed!")
