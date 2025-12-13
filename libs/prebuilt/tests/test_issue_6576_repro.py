"""
Reproduction and fix verification test for issue #6576:
Pydantic error when using custom tool node and having runtime in the tools.

This test demonstrates the issue and the recommended workaround:
- Tools with ToolRuntime should use ToolNode for proper injection
- For custom tool nodes, declare runtime with a default value: `runtime: ToolRuntime = None`
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool as dec_tool
from typing_extensions import TypedDict

from langgraph.prebuilt import ToolRuntime


class RouterTestState(TypedDict):
    router_history: list
    user_query: str


# WORKAROUND: Use a default value for runtime parameter
@dec_tool
def get_greeting_rules_with_default(
    runtime: ToolRuntime = None,
) -> str:
    """Get guidelines for responding to greetings.

    Call this when the user says hello, thanks you, asks about your capabilities,
    or engages in general conversation.

    NOTE: runtime parameter has a default value to allow direct invocation.
    """
    # Handle case where runtime is None (direct invocation without injection)
    if runtime is None or runtime.state is None:
        return "Greeting response (no runtime context)"
    user_query = runtime.state.get("user_query", "")
    return f"Greeting response for: {user_query}"


@dec_tool
def tool_with_query_and_default_runtime(
    query: str,
    runtime: ToolRuntime = None,
) -> str:
    """A tool that takes a query and uses runtime context."""
    if runtime is None or runtime.state is None:
        return f"Query: {query} (no runtime context)"
    user_query = runtime.state.get("user_query", "")
    return f"Query: {query}, Context: {user_query}"


ROUTING_TOOLS = [get_greeting_rules_with_default, tool_with_query_and_default_runtime]


def custom_tool_node(state: RouterTestState):
    """Custom tool node that directly invokes tools.

    This pattern requires tools to have default values for ToolRuntime parameters.
    """
    tools_by_name = {tool.name: tool for tool in ROUTING_TOOLS}

    result = []
    for tool_call in state["router_history"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        # Directly invoke with just args - works when tool has default for runtime
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"router_history": result}


def test_issue_6576_workaround_custom_tool_node_with_default_runtime():
    """Test that custom tool nodes work when tools have default runtime values.

    This is the recommended workaround for issue #6576: define tools with
    `runtime: ToolRuntime = None` to allow direct invocation outside of ToolNode.
    """
    tool_call = {
        "name": "get_greeting_rules_with_default",
        "args": {},
        "id": "test_call_1",
        "type": "tool_call",
    }

    state = RouterTestState(
        router_history=[AIMessage("", tool_calls=[tool_call])],
        user_query="Hello, how are you?",
    )

    # This works because the tool has a default value for runtime
    result = custom_tool_node(state)

    assert "router_history" in result
    assert len(result["router_history"]) == 1
    assert isinstance(result["router_history"][0], ToolMessage)
    assert "no runtime context" in result["router_history"][0].content


def test_issue_6576_workaround_tool_with_mixed_args():
    """Test that tools with both regular args and default runtime work correctly."""
    tool_call = {
        "name": "tool_with_query_and_default_runtime",
        "args": {"query": "test query"},
        "id": "test_call_2",
        "type": "tool_call",
    }

    state = RouterTestState(
        router_history=[AIMessage("", tool_calls=[tool_call])],
        user_query="Hello!",
    )

    result = custom_tool_node(state)

    assert "router_history" in result
    assert len(result["router_history"]) == 1
    assert isinstance(result["router_history"][0], ToolMessage)
    assert "test query" in result["router_history"][0].content


def test_issue_6576_tool_runtime_has_default_none():
    """Test that ToolRuntime can be instantiated with no arguments."""
    runtime = ToolRuntime()

    assert runtime.state is None
    assert runtime.context is None
    assert runtime.config is None
    assert runtime.stream_writer is None
    assert runtime.tool_call_id is None
    assert runtime.store is None


def test_issue_6576_tool_runtime_can_be_fully_populated():
    """Test that ToolRuntime still works when fully populated."""
    from unittest.mock import Mock

    mock_state = {"messages": [], "foo": "bar"}
    mock_config = {"run_id": "test123"}
    mock_writer = Mock()
    mock_store = Mock()

    runtime = ToolRuntime(
        state=mock_state,
        context={"user_id": "123"},
        config=mock_config,
        stream_writer=mock_writer,
        tool_call_id="call_abc",
        store=mock_store,
    )

    assert runtime.state == mock_state
    assert runtime.context == {"user_id": "123"}
    assert runtime.config == mock_config
    assert runtime.stream_writer == mock_writer
    assert runtime.tool_call_id == "call_abc"
    assert runtime.store == mock_store


def test_issue_6576_original_pattern_shows_pydantic_passes_but_call_fails():
    """Document the original issue: pydantic validation passes but function call fails.

    This test documents the behavior where:
    1. Pydantic validation passes (due to __get_pydantic_core_schema__ customization)
    2. But the actual function call fails because runtime param has no default

    The fix is to use `runtime: ToolRuntime = None` in the function signature.
    """

    # Original pattern without default value - this is what causes the issue
    @dec_tool
    def tool_without_default(runtime: ToolRuntime) -> str:
        """Tool without default runtime - will fail on direct invocation."""
        return "hello"

    # Pydantic validation now passes (schema accepts None), but actual call fails
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        tool_without_default.invoke({})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
