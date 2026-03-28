"""
Tests for ToolNode timeout handling with MCP tools.

This test module reproduces GitHub issue #6412:
"ToolNode ainvoke freezes if sse_read_timeout"

The issue occurs when an MCP-based tool times out, but ToolNode doesn't properly
propagate the timeout exception, causing the async task to hang indefinitely.
"""

import asyncio
from unittest.mock import Mock
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool as dec_tool

from langgraph.prebuilt import ToolNode

pytestmark = pytest.mark.anyio


def _create_mock_runtime(store: Any | None = None) -> Mock:
    """Create a mock Runtime object for testing ToolNode outside of graph context."""
    mock_runtime = Mock()
    mock_runtime.store = store
    mock_runtime.context = None
    mock_runtime.stream_writer = lambda *args, **kwargs: None
    return mock_runtime


def _create_config_with_runtime(store: Any | None = None) -> RunnableConfig:
    """Create a RunnableConfig with mock Runtime for testing ToolNode."""
    return {"configurable": {"__pregel_runtime": _create_mock_runtime(store)}}


# ============================================================================
# Test Tools: simulating MCP timeout scenarios
# ============================================================================


async def tool_with_timeout() -> str:
    """Async tool that raises asyncio.TimeoutError (simulating MCP sse_read_timeout)."""
    # Simulate MCP stream timeout after receiving sse_read_timeout
    raise asyncio.TimeoutError("sse_read_timeout: MCP stream closed after 5 seconds")


async def tool_with_read_timeout_error() -> str:
    """Async tool that raises httpx.ReadTimeout (another MCP timeout variant)."""
    # Simulate httpx ReadTimeout
    try:
        import httpx
        raise httpx.ReadTimeout("Read timeout")
    except ImportError:
        # httpx not available, use OSError as fallback
        raise OSError("Read timed out") from None


async def tool_with_connection_error() -> str:
    """Async tool that simulates connection loss."""
    raise ConnectionError("MCP server connection lost")


async def slow_async_tool(delay_seconds: int) -> str:
    """Async tool that takes time to complete."""
    await asyncio.sleep(delay_seconds)
    return f"Completed after {delay_seconds}s"


# ============================================================================
# Tests for timeout handling
# ============================================================================


async def test_tool_node_handles_asyncio_timeout() -> None:
    """
    Test that ToolNode properly catches AsyncTimeoutError from MCP tools.
    
    This test reproduces the bug where ToolNode doesn't propagate timeout
    exceptions, causing ainvoke to hang instead of returning an error message.
    
    Expected (with fix): ToolMessage with error content
    Current (bug): Hangs indefinitely (caught by outer asyncio.wait_for)
    """
    tool_node = ToolNode([tool_with_timeout])
    
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "tool_with_timeout",
                        "args": {},
                    }
                ],
            )
        ]
    }
    
    # Wrap with timeout to prevent infinite hang during test
    # Without fix, this test times out after 10 seconds
    try:
        result = await asyncio.wait_for(
            tool_node.ainvoke(state, config=_create_config_with_runtime()),
            timeout=10,
        )
    except TimeoutError:
        # This indicates the bug: ToolNode.ainvoke hung instead of returning
        pytest.fail(
            "BUG REPRODUCED: ToolNode.ainvoke hanged despite MCP timeout. "
            "The timeout exception was not properly propagated from the tool."
        )
    
    # Verify result contains error message about timeout
    assert len(result["messages"]) >= 2
    tool_message: ToolMessage = result["messages"][-1]
    assert tool_message.type == "tool"
    # Should contain error information, not hang
    assert "timeout" in tool_message.content.lower() or "error" in tool_message.content.lower()


async def test_tool_node_handles_connection_error() -> None:
    """
    Test that ToolNode properly handles connection errors from MCP tools.
    
    Connection errors can occur when MCP server is unreachable after timeout.
    """
    tool_node = ToolNode([tool_with_connection_error])
    
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-2",
                        "name": "tool_with_connection_error",
                        "args": {},
                    }
                ],
            )
        ]
    }
    
    try:
        result = await asyncio.wait_for(
            tool_node.ainvoke(state, config=_create_config_with_runtime()),
            timeout=10,
        )
    except TimeoutError:
        pytest.fail(
            "ToolNode.ainvoke hanged handling connection error. "
            "Connection errors should be caught and returned as ToolMessage."
        )
    
    # Should return a ToolMessage with error content
    assert len(result["messages"]) >= 2
    tool_message: ToolMessage = result["messages"][-1]
    assert tool_message.type == "tool"


async def test_tool_node_multiple_tools_one_timeout() -> None:
    """
    Test that ToolNode handles timeout in one tool when running multiple tools.
    
    This scenario is common: LLM generates multiple tool calls, one times out.
    The other calls should still complete and the timeout should be reported.
    """
    tool_node = ToolNode([slow_async_tool, tool_with_timeout])
    
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "slow_async_tool",
                        "args": {"delay_seconds": 1},
                    },
                    {
                        "id": "call-2",
                        "name": "tool_with_timeout",
                        "args": {},
                    },
                ],
            )
        ]
    }
    
    try:
        result = await asyncio.wait_for(
            tool_node.ainvoke(state, config=_create_config_with_runtime()),
            timeout=10,
        )
    except TimeoutError:
        pytest.fail(
            "ToolNode.ainvoke hanged when one of multiple tools timed out. "
            "Should complete with mixed results (success + error)."
        )
    
    # Should have 3 messages: original + 2 tool results
    assert len(result["messages"]) >= 3
    
    # Verify we got both tool messages
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 2
    
    # One should succeed, one should have error
    contents = [msg.content for msg in tool_messages]
    has_success = any("Completed" in str(c) for c in contents)
    has_error = any("error" in str(c).lower() or "timeout" in str(c).lower() for c in contents)
    
    assert has_success, "Expected at least one tool to succeed"
    assert has_error, "Expected timeout error to be reported"


async def test_tool_node_timeout_propagates_to_caller() -> None:
    """
    Test that extreme timeouts in tools are properly caught and reported.
    
    The fix should prevent ToolNode from indefinitely waiting when a tool times out.
    """
    tool_node = ToolNode([tool_with_timeout])
    
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "tool_with_timeout",
                        "args": {},
                    }
                ],
            )
        ]
    }
    
    # Set a moderate timeout (5 seconds)
    # Without fix: This will timeout (bug is present)
    # With fix: Should return quickly with error ToolMessage
    try:
        result = await asyncio.wait_for(
            tool_node.ainvoke(state, config=_create_config_with_runtime()),
            timeout=5,
        )
        # If we get here without timeout, check the result contains error info
        assert len(result["messages"]) >= 2
        tool_message: ToolMessage = result["messages"][-1]
        assert tool_message.type == "tool"
        
    except TimeoutError:
        # This is the bug: ToolNode hung instead of returning error
        pytest.fail(
            "CRITICAL BUG: ToolNode.ainvoke did not complete within 5 seconds "
            "even though the tool's timeout error should have been caught. "
            "This indicates the timeout exception is not properly propagated."
        )


async def test_tool_node_sync_tool_not_affected() -> None:
    """
    Test that synchronous tools still work correctly.
    
    Ensure our timeout handling doesn't break sync tools.
    """
    def sync_tool(value: int) -> str:
        """A simple synchronous tool."""
        return f"sync result: {value}"
    
    tool_node = ToolNode([sync_tool])
    
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "sync_tool",
                        "args": {"value": 42},
                    }
                ],
            )
        ]
    }
    
    result = await tool_node.ainvoke(state, config=_create_config_with_runtime())
    
    assert len(result["messages"]) >= 2
    tool_message: ToolMessage = result["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.content == "sync result: 42"


async def test_tool_node_timeout_with_decorated_tool() -> None:
    """
    Test timeout handling with @tool decorated async function.
    
    Ensures timeout handling works with langchain's tool decorator.
    """
    @dec_tool
    async def decorated_timeout_tool() -> str:
        """A decorated tool that times out."""
        raise asyncio.TimeoutError("MCP timeout")
    
    tool_node = ToolNode([decorated_timeout_tool])
    
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "decorated_timeout_tool",
                        "args": {},
                    }
                ],
            )
        ]
    }
    
    try:
        result = await asyncio.wait_for(
            tool_node.ainvoke(state, config=_create_config_with_runtime()),
            timeout=10,
        )
    except TimeoutError:
        pytest.fail(
            "ToolNode.ainvoke hanged with @tool decorated async function that timed out."
        )
    
    # Verify error was caught
    assert len(result["messages"]) >= 2
    tool_message: ToolMessage = result["messages"][-1]
    assert tool_message.type == "tool"
