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
    assert (
        result[0].content
        == "Error: unregistered_tool is not a valid tool, try one of [registered_tool]."
    )
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
    """Test that wrap_tool_call can provide custom implementation for unregistered tool."""
    called = False

    @dec_tool
    def custom_tool_impl() -> str:
        """Custom tool implementation."""
        nonlocal called
        called = True
        return "custom result"

    def hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if request.tool_call["name"] == "custom_tool":
            assert request.tool is None  # Unregistered tools have tool=None
            return execute(request.override(tool=custom_tool_impl))
        return execute(request)

    node = ToolNode([registered_tool], wrap_tool_call=hook)

    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"name": "custom_tool", "args": {}, "id": "1", "type": "tool_call"}
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    assert called
    assert result[0].content == "custom result"
    assert result[0].tool_call_id == "1"


async def test_awrap_tool_call_override_unregistered_tool_with_custom_impl() -> None:
    """Test that awrap_tool_call can provide custom implementation for unregistered tool."""
    called = False

    @dec_tool
    def custom_async_tool_impl() -> str:
        """Custom async tool implementation."""
        nonlocal called
        called = True
        return "async custom result"

    async def hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if request.tool_call["name"] == "custom_async_tool":
            assert request.tool is None  # Unregistered tools have tool=None
            return await execute(request.override(tool=custom_async_tool_impl))
        return await execute(request)

    node = ToolNode([registered_tool], awrap_tool_call=hook)

    result = await node.ainvoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "custom_async_tool",
                        "args": {},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    assert called
    assert result[0].content == "async custom result"
    assert result[0].tool_call_id == "1"


def test_graceful_failure_when_hook_does_not_override_unregistered_tool_sync() -> None:
    """Test graceful failure when hook doesn't override unregistered tool."""

    def passthrough_hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        return execute(request)

    node = ToolNode(
        [registered_tool],
        wrap_tool_call=passthrough_hook,
        handle_tool_errors=True,
    )

    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"name": "nonexistent", "args": {}, "id": "1", "type": "tool_call"}
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    assert result[0].status == "error"
    assert result[0].tool_call_id == "1"
    assert (
        result[0].content
        == "Error: nonexistent is not a valid tool, try one of [registered_tool]."
    )


def test_graceful_failure_even_when_handle_errors_disabled_sync() -> None:
    """Test that unregistered tool validation returns error even with handle_tool_errors=False."""

    def passthrough_hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        return execute(request)

    node = ToolNode(
        [registered_tool],
        wrap_tool_call=passthrough_hook,
        handle_tool_errors=False,
    )

    result = node.invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"name": "missing", "args": {}, "id": "1", "type": "tool_call"}
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    assert result[0].status == "error"
    assert (
        result[0].content
        == "Error: missing is not a valid tool, try one of [registered_tool]."
    )


async def test_graceful_failure_when_hook_does_not_override_unregistered_tool_async() -> (
    None
):
    """Test graceful failure when async hook doesn't override unregistered tool."""

    async def passthrough_hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        return await execute(request)

    node = ToolNode(
        [registered_tool],
        awrap_tool_call=passthrough_hook,
        handle_tool_errors=True,
    )

    result = await node.ainvoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"name": "unknown", "args": {}, "id": "1", "type": "tool_call"}
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    assert result[0].status == "error"
    assert result[0].tool_call_id == "1"
    assert (
        result[0].content
        == "Error: unknown is not a valid tool, try one of [registered_tool]."
    )


async def test_graceful_failure_even_when_handle_errors_disabled_async() -> None:
    """Test that async unregistered tool validation returns error even with handle_tool_errors=False."""

    async def passthrough_hook(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        return await execute(request)

    node = ToolNode(
        [registered_tool],
        awrap_tool_call=passthrough_hook,
        handle_tool_errors=False,
    )

    result = await node.ainvoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"name": "missing", "args": {}, "id": "1", "type": "tool_call"}
                ],
            )
        ],
        config=_create_config_with_runtime(),
    )

    assert result[0].status == "error"
    assert (
        result[0].content
        == "Error: missing is not a valid tool, try one of [registered_tool]."
    )
