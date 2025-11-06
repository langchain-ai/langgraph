"""Test to reproduce the tool error fallback issue.

When using ToolNode(...).with_fallbacks(...) with an error handler that returns
ToolMessage objects, the messages are losing:
1. The `name` field (becomes `None` instead of the tool name)
2. The `status` field (becomes `success` instead of `error`)
"""

from unittest.mock import Mock

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode


def _create_mock_runtime():
    """Create a mock Runtime object for testing ToolNode outside of graph context."""
    mock_runtime = Mock()
    mock_runtime.store = None
    mock_runtime.context = None
    mock_runtime.stream_writer = lambda *args, **kwargs: None
    return mock_runtime


def _create_config_with_runtime() -> RunnableConfig:
    """Create a RunnableConfig with mock Runtime for testing ToolNode."""
    return {"configurable": {"__pregel_runtime": _create_mock_runtime()}}


@tool
def failing_tool(x: int) -> str:
    """A tool that always fails."""
    raise RuntimeError("This tool always fails!")


def handle_tool_error(state) -> dict:
    """Error handler that returns ToolMessages."""
    print(f"handle_tool_error called with state: {state}")
    print(f"State type: {type(state)}")
    print(f"State keys: {state.keys() if isinstance(state, dict) else 'N/A'}")
    error = state.get("error")
    print(f"Error: {error}")
    tool_calls = state["messages"][-1].tool_calls
    print(f"Tool calls: {tool_calls}")
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


fallback_runnable = RunnableLambda(handle_tool_error)


def test_tool_error_with_fallbacks():
    """Test that ToolMessages from fallback handlers preserve name and status."""
    # Create a ToolNode with a fallback
    tool_node = ToolNode([failing_tool], handle_tool_errors=True).with_fallbacks(
        [fallback_runnable],
    )

    # Create an AI message with a tool call
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "failing_tool",
                    "args": {"x": 1},
                    "id": "call_123",
                    "type": "tool_call",
                }
            ],
        )
    ]

    # Invoke the tool node
    result = tool_node.invoke({"messages": messages}, config=_create_config_with_runtime())

    print("Result:", result)
    print("Result type:", type(result))
    print("Result keys:", result.keys() if isinstance(result, dict) else "N/A")
    print("\nTool messages:")
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"  - content: {msg.content[:50]}...")
            print(f"    name: {msg.name}")
            print(f"    tool_call_id: {msg.tool_call_id}")
            print(f"    status: {msg.status}")
            print()

            # Check expectations
            assert msg.name == "failing_tool", f"Expected name='failing_tool', got {msg.name}"
            assert msg.status == "error", f"Expected status='error', got {msg.status}"
            print("✓ Test passed!")


if __name__ == "__main__":
    test_tool_error_with_fallbacks()
