"""Test tool node interceptor handling of unregistered tools."""

from collections.abc import Awaitable, Callable
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool as dec_tool
from langgraph.store.base import BaseStore
from langgraph.types import Command

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest

pytestmark = pytest.mark.anyio


def _create_mock_runtime(store: BaseStore | None = None) -> Mock:
    """Create a mock Runtime object for testing ToolNode outside of graph context.

    This helper is needed because ToolNode._func expects a Runtime parameter
    which is injected by RunnableCallable from config["configurable"]["__pregel_runtime"].
    When testing ToolNode directly (outside a graph), we need to provide this manually.
    """
    mock_runtime = Mock()
    mock_runtime.store = store
    mock_runtime.context = None
    mock_runtime.stream_writer = lambda *args, **kwargs: None
    return mock_runtime


def _create_config_with_runtime(store: BaseStore | None = None) -> RunnableConfig:
    """Create a RunnableConfig with mock Runtime for testing ToolNode.

    Returns:
        RunnableConfig with __pregel_runtime in configurable dict.
    """
    return {"configurable": {"__pregel_runtime": _create_mock_runtime(store)}}


@dec_tool
def registered_tool(x: int) -> str:
    """A registered tool."""
    return f"Result: {x}"


def test_interceptor_can_handle_unregistered_tool_sync() -> None:
    """Test that interceptor can handle requests for unregistered tools (sync)."""

    def interceptor(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept and handle unregistered tools."""
        if request.tool_call["name"] == "unregistered_tool":
            # Short-circuit without calling execute for unregistered tool
            return ToolMessage(
                content="Handled by interceptor",
                tool_call_id=request.tool_call["id"],
                name="unregistered_tool",
            )
        # Pass through for registered tools
        return execute(request)

    node = ToolNode([registered_tool], wrap_tool_call=interceptor)

    # Test registered tool works normally
    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "registered_tool",
                        "args": {"x": 42},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )
    assert result[0].content == "Result: 42"
    assert result[0].tool_call_id == "1"

    # Test unregistered tool is intercepted and handled
    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "unregistered_tool",
                        "args": {"x": 99},
                        "id": "2",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )
    assert result[0].content == "Handled by interceptor"
    assert result[0].tool_call_id == "2"
    assert result[0].name == "unregistered_tool"


async def test_interceptor_can_handle_unregistered_tool_async() -> None:
    """Test that interceptor can handle requests for unregistered tools (async)."""

    async def async_interceptor(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Intercept and handle unregistered tools."""
        if request.tool_call["name"] == "unregistered_tool":
            # Short-circuit without calling execute for unregistered tool
            return ToolMessage(
                content="Handled by async interceptor",
                tool_call_id=request.tool_call["id"],
                name="unregistered_tool",
            )
        # Pass through for registered tools
        return await execute(request)

    node = ToolNode([registered_tool], awrap_tool_call=async_interceptor)

    # Test registered tool works normally
    result = await node.ainvoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "registered_tool",
                        "args": {"x": 42},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )
    assert result[0].content == "Result: 42"
    assert result[0].tool_call_id == "1"

    # Test unregistered tool is intercepted and handled
    result = await node.ainvoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "unregistered_tool",
                        "args": {"x": 99},
                        "id": "2",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )
    assert result[0].content == "Handled by async interceptor"
    assert result[0].tool_call_id == "2"
    assert result[0].name == "unregistered_tool"


def test_unregistered_tool_error_when_interceptor_calls_execute() -> None:
    """Test that unregistered tools error if interceptor tries to execute them."""

    def bad_interceptor(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Interceptor that tries to execute unregistered tool."""
        # This should fail validation when execute is called
        return execute(request)

    node = ToolNode([registered_tool], wrap_tool_call=bad_interceptor)

    # Registered tool should still work
    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "registered_tool",
                        "args": {"x": 42},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )
    assert result[0].content == "Result: 42"

    # Unregistered tool should error when interceptor calls execute
    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "unregistered_tool",
                        "args": {"x": 99},
                        "id": "2",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )
    # Should get validation error message
    assert result[0].status == "error"
    assert "is not a valid tool" in result[0].content
    assert result[0].tool_call_id == "2"


def test_interceptor_handles_mix_of_registered_and_unregistered() -> None:
    """Test interceptor handling mix of registered and unregistered tools."""

    def selective_interceptor(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Handle unregistered tools, pass through registered ones."""
        if request.tool_call["name"] == "magic_tool":
            return ToolMessage(
                content=f"Magic result: {request.tool_call['args'].get('value', 0) * 2}",
                tool_call_id=request.tool_call["id"],
                name="magic_tool",
            )
        return execute(request)

    node = ToolNode([registered_tool], wrap_tool_call=selective_interceptor)

    # Test multiple tool calls - mix of registered and unregistered
    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "registered_tool",
                        "args": {"x": 10},
                        "id": "1",
                        "type": "tool_call",
                    },
                    {
                        "name": "magic_tool",
                        "args": {"value": 5},
                        "id": "2",
                        "type": "tool_call",
                    },
                    {
                        "name": "registered_tool",
                        "args": {"x": 20},
                        "id": "3",
                        "type": "tool_call",
                    },
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    # All tools should execute successfully
    assert len(result) == 3
    assert result[0].content == "Result: 10"
    assert result[0].tool_call_id == "1"
    assert result[1].content == "Magic result: 10"
    assert result[1].tool_call_id == "2"
    assert result[2].content == "Result: 20"
    assert result[2].tool_call_id == "3"


def test_interceptor_command_for_unregistered_tool() -> None:
    """Test interceptor returning Command for unregistered tool."""

    def command_interceptor(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Return Command for unregistered tools."""
        if request.tool_call["name"] == "routing_tool":
            return Command(
                update=[
                    ToolMessage(
                        content="Routing to special handler",
                        tool_call_id=request.tool_call["id"],
                        name="routing_tool",
                    )
                ],
                goto="special_node",
            )
        return execute(request)

    node = ToolNode([registered_tool], wrap_tool_call=command_interceptor)

    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "routing_tool",
                        "args": {},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    # Should get Command back
    assert len(result) == 1
    assert isinstance(result[0], Command)
    assert result[0].goto == "special_node"
    assert result[0].update is not None
    assert len(result[0].update) == 1
    assert result[0].update[0].content == "Routing to special handler"


def test_interceptor_exception_with_unregistered_tool() -> None:
    """Test that interceptor exceptions are caught by error handling."""

    def failing_interceptor(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Interceptor that throws exception for unregistered tools."""
        if request.tool_call["name"] == "bad_tool":
            msg = "Interceptor failed"
            raise ValueError(msg)
        return execute(request)

    node = ToolNode(
        [registered_tool], wrap_tool_call=failing_interceptor, handle_tool_errors=True
    )

    # Interceptor exception should be caught and converted to error message
    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "bad_tool",
                        "args": {},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    assert len(result) == 1
    assert result[0].status == "error"
    assert "Interceptor failed" in result[0].content
    assert result[0].tool_call_id == "1"

    # Test that exception is raised when handle_tool_errors is False
    node_no_handling = ToolNode(
        [registered_tool], wrap_tool_call=failing_interceptor, handle_tool_errors=False
    )

    with pytest.raises(ValueError, match="Interceptor failed"):
        node_no_handling.invoke(
            [
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "bad_tool",
                            "args": {},
                            "id": "2",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            config=_create_config_with_runtime(),
        )


async def test_async_interceptor_exception_with_unregistered_tool() -> None:
    """Test that async interceptor exceptions are caught by error handling."""

    async def failing_async_interceptor(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async interceptor that throws exception for unregistered tools."""
        if request.tool_call["name"] == "bad_async_tool":
            msg = "Async interceptor failed"
            raise RuntimeError(msg)
        return await execute(request)

    node = ToolNode(
        [registered_tool],
        awrap_tool_call=failing_async_interceptor,
        handle_tool_errors=True,
    )

    # Interceptor exception should be caught and converted to error message
    result = await node.ainvoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "bad_async_tool",
                        "args": {},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    assert len(result) == 1
    assert result[0].status == "error"
    assert "Async interceptor failed" in result[0].content
    assert result[0].tool_call_id == "1"

    # Test that exception is raised when handle_tool_errors is False
    node_no_handling = ToolNode(
        [registered_tool],
        awrap_tool_call=failing_async_interceptor,
        handle_tool_errors=False,
    )

    with pytest.raises(RuntimeError, match="Async interceptor failed"):
        await node_no_handling.ainvoke(
            [
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "bad_async_tool",
                            "args": {},
                            "id": "2",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            config=_create_config_with_runtime(),
        )


def test_interceptor_with_dict_input_format() -> None:
    """Test that interceptor works with dict input format."""

    def interceptor(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept unregistered tools with dict input."""
        if request.tool_call["name"] == "dict_tool":
            return ToolMessage(
                content="Handled dict input",
                tool_call_id=request.tool_call["id"],
                name="dict_tool",
            )
        return execute(request)

    node = ToolNode([registered_tool], wrap_tool_call=interceptor)

    # Test with dict input format
    result = node.invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "dict_tool",
                            "args": {"value": 5},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    # Should return dict format output
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Handled dict input"
    assert result["messages"][0].tool_call_id == "1"


def test_interceptor_verifies_tool_is_none_for_unregistered() -> None:
    """Test that request.tool is None for unregistered tools."""

    captured_requests: list[ToolCallRequest] = []

    def capturing_interceptor(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Capture request to verify tool field."""
        captured_requests.append(request)
        if request.tool is None:
            # Tool is unregistered
            return ToolMessage(
                content=f"Unregistered: {request.tool_call['name']}",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
            )
        # Tool is registered
        return execute(request)

    node = ToolNode([registered_tool], wrap_tool_call=capturing_interceptor)

    # Test unregistered tool
    node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "unknown_tool",
                        "args": {},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    assert len(captured_requests) == 1
    assert captured_requests[0].tool is None
    assert captured_requests[0].tool_call["name"] == "unknown_tool"

    # Clear and test registered tool
    captured_requests.clear()
    node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "registered_tool",
                        "args": {"x": 10},
                        "id": "2",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    assert len(captured_requests) == 1
    assert captured_requests[0].tool is not None
    assert captured_requests[0].tool.name == "registered_tool"


def test_wrap_tool_call_override_unregistered_tool_with_custom_impl() -> None:
    """Test that wrap_tool_call can override an unregistered tool with custom implementation.

    This test verifies that a wrap_tool_call hook can provide a complete custom
    implementation for a tool that is NOT registered with the ToolNode. The hook
    receives the tool call request, executes custom logic, and returns the result
    without ever calling the execute() function.
    """
    # Track that our custom implementation was actually called
    execution_log: list[dict] = []

    def custom_calculator(a: int, b: int, operation: str) -> int:
        """Custom calculator implementation provided via hook."""
        execution_log.append({"a": a, "b": b, "operation": operation})
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        elif operation == "subtract":
            return a - b
        else:
            msg = f"Unknown operation: {operation}"
            raise ValueError(msg)

    def wrap_tool_call_hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Hook that provides custom implementation for unregistered 'calculator' tool."""
        if request.tool_call["name"] == "calculator":
            # Verify the tool is not registered (request.tool should be None)
            assert request.tool is None, (
                "Expected tool to be None for unregistered tool"
            )

            # Execute our custom calculator implementation
            args = request.tool_call["args"]
            result = custom_calculator(
                a=args["a"],
                b=args["b"],
                operation=args["operation"],
            )
            return ToolMessage(
                content=str(result),
                tool_call_id=request.tool_call["id"],
                name="calculator",
            )
        # Pass through for registered tools
        return execute(request)

    # Create ToolNode with only 'registered_tool' - no 'calculator' tool
    node = ToolNode([registered_tool], wrap_tool_call=wrap_tool_call_hook)

    # Invoke with the unregistered 'calculator' tool
    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"a": 10, "b": 5, "operation": "add"},
                        "id": "calc-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    # Verify the custom implementation was called
    assert len(execution_log) == 1
    assert execution_log[0] == {"a": 10, "b": 5, "operation": "add"}

    # Verify the result
    assert len(result) == 1
    assert result[0].content == "15"  # 10 + 5
    assert result[0].tool_call_id == "calc-1"
    assert result[0].name == "calculator"

    # Test multiply operation
    execution_log.clear()
    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"a": 7, "b": 3, "operation": "multiply"},
                        "id": "calc-2",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    assert len(execution_log) == 1
    assert execution_log[0] == {"a": 7, "b": 3, "operation": "multiply"}
    assert result[0].content == "21"  # 7 * 3


async def test_awrap_tool_call_override_unregistered_tool_with_custom_impl() -> None:
    """Test that awrap_tool_call can override an unregistered tool with custom async implementation."""
    execution_log: list[dict] = []

    async def custom_async_fetcher(url: str, timeout: int) -> str:
        """Custom async fetcher implementation provided via hook."""
        execution_log.append({"url": url, "timeout": timeout})
        # Simulate async operation result
        return f"Fetched content from {url} with timeout {timeout}s"

    async def awrap_tool_call_hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async hook that provides custom implementation for unregistered 'fetch_url' tool."""
        if request.tool_call["name"] == "fetch_url":
            # Verify the tool is not registered
            assert request.tool is None, (
                "Expected tool to be None for unregistered tool"
            )

            # Execute our custom async implementation
            args = request.tool_call["args"]
            result = await custom_async_fetcher(
                url=args["url"],
                timeout=args.get("timeout", 30),
            )
            return ToolMessage(
                content=result,
                tool_call_id=request.tool_call["id"],
                name="fetch_url",
            )
        # Pass through for registered tools
        return await execute(request)

    # Create ToolNode with only 'registered_tool' - no 'fetch_url' tool
    node = ToolNode([registered_tool], awrap_tool_call=awrap_tool_call_hook)

    # Invoke with the unregistered 'fetch_url' tool
    result = await node.ainvoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "fetch_url",
                        "args": {"url": "https://example.com", "timeout": 60},
                        "id": "fetch-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    # Verify the custom implementation was called
    assert len(execution_log) == 1
    assert execution_log[0] == {"url": "https://example.com", "timeout": 60}

    # Verify the result
    assert len(result) == 1
    assert (
        result[0].content == "Fetched content from https://example.com with timeout 60s"
    )
    assert result[0].tool_call_id == "fetch-1"
    assert result[0].name == "fetch_url"


def test_graceful_failure_when_hook_does_not_override_unregistered_tool_sync() -> None:
    """Test graceful failure with helpful message when hook doesn't override unregistered tool.

    When a wrap_tool_call hook receives a request for an unregistered tool but calls
    execute() instead of handling it, the ToolNode should:
    1. Return an error ToolMessage (not raise an exception) when handle_tool_errors=True
    2. Include a helpful message listing the available tools
    3. Set status="error" on the ToolMessage
    """

    def passthrough_hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Hook that doesn't handle unregistered tools - just passes through."""
        # This hook doesn't check request.tool, so it will try to execute
        # unregistered tools which should fail gracefully
        return execute(request)

    # Create ToolNode with multiple registered tools to verify error message
    @dec_tool
    def tool_alpha(x: int) -> str:
        """Tool alpha."""
        return f"alpha: {x}"

    @dec_tool
    def tool_beta(y: str) -> str:
        """Tool beta."""
        return f"beta: {y}"

    node = ToolNode(
        [registered_tool, tool_alpha, tool_beta],
        wrap_tool_call=passthrough_hook,
        handle_tool_errors=True,  # Enable graceful error handling
    )

    # Invoke with an unregistered tool
    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "nonexistent_tool",
                        "args": {"foo": "bar"},
                        "id": "test-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    # Should return error ToolMessage, not raise exception
    assert len(result) == 1
    assert result[0].status == "error"
    assert result[0].tool_call_id == "test-1"
    assert result[0].name == "nonexistent_tool"

    # Error message should be helpful - mention the invalid tool and list available tools
    error_content = result[0].content
    assert "nonexistent_tool" in error_content
    assert "is not a valid tool" in error_content
    # Should list available tools
    assert "registered_tool" in error_content
    assert "tool_alpha" in error_content
    assert "tool_beta" in error_content


def test_graceful_failure_even_when_handle_errors_disabled_sync() -> None:
    """Test that unregistered tool validation always returns error message, even with handle_tool_errors=False.

    The handle_tool_errors setting only affects errors during tool *execution*.
    Validation errors for unregistered tools always return a helpful ToolMessage
    rather than raising an exception, regardless of the handle_tool_errors setting.
    """

    def passthrough_hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Hook that doesn't handle unregistered tools."""
        return execute(request)

    node = ToolNode(
        [registered_tool],
        wrap_tool_call=passthrough_hook,
        handle_tool_errors=False,  # Even with this disabled, validation errors are handled gracefully
    )

    # Should still return error ToolMessage (not raise) for unregistered tools
    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "missing_tool",
                        "args": {},
                        "id": "test-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    # Validation errors always return ToolMessage, regardless of handle_tool_errors
    assert len(result) == 1
    assert result[0].status == "error"
    assert result[0].tool_call_id == "test-1"
    assert "missing_tool" in result[0].content
    assert "is not a valid tool" in result[0].content
    assert "registered_tool" in result[0].content  # Lists available tools


async def test_graceful_failure_when_hook_does_not_override_unregistered_tool_async() -> (
    None
):
    """Test graceful failure with helpful message for async hook not overriding unregistered tool."""

    async def passthrough_async_hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async hook that doesn't handle unregistered tools."""
        return await execute(request)

    @dec_tool
    def another_tool(value: int) -> str:
        """Another tool."""
        return f"value: {value}"

    node = ToolNode(
        [registered_tool, another_tool],
        awrap_tool_call=passthrough_async_hook,
        handle_tool_errors=True,
    )

    # Invoke with an unregistered tool
    result = await node.ainvoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "unknown_async_tool",
                        "args": {"data": 123},
                        "id": "async-test-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    # Should return error ToolMessage with helpful message
    assert len(result) == 1
    assert result[0].status == "error"
    assert result[0].tool_call_id == "async-test-1"
    assert result[0].name == "unknown_async_tool"

    # Error message should list available tools
    error_content = result[0].content
    assert "unknown_async_tool" in error_content
    assert "is not a valid tool" in error_content
    assert "registered_tool" in error_content
    assert "another_tool" in error_content


async def test_graceful_failure_even_when_handle_errors_disabled_async() -> None:
    """Test that async unregistered tool validation always returns error message.

    Same as sync version - validation errors for unregistered tools always return
    a helpful ToolMessage rather than raising, regardless of handle_tool_errors.
    """

    async def passthrough_async_hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async hook that doesn't handle unregistered tools."""
        return await execute(request)

    node = ToolNode(
        [registered_tool],
        awrap_tool_call=passthrough_async_hook,
        handle_tool_errors=False,
    )

    # Should still return error ToolMessage (not raise) for unregistered tools
    result = await node.ainvoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "missing_async_tool",
                        "args": {},
                        "id": "test-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    # Validation errors always return ToolMessage
    assert len(result) == 1
    assert result[0].status == "error"
    assert result[0].tool_call_id == "test-1"
    assert "missing_async_tool" in result[0].content
    assert "is not a valid tool" in result[0].content
    assert "registered_tool" in result[0].content
