"""Unit tests for ValidationError filtering in ToolNode.

This module tests that validation errors are filtered to only include arguments
that the LLM controls. Injected arguments (InjectedState, InjectedStore,
ToolRuntime) are automatically provided by the system and should not appear in
validation error messages. This ensures the LLM receives focused, actionable
feedback about the parameters it can actually control, improving error correction
and reducing confusion from irrelevant system implementation details.
"""

from typing import Annotated
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool as dec_tool
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from langgraph.prebuilt import InjectedState, InjectedStore, ToolNode, ToolRuntime
from langgraph.prebuilt.tool_node import ToolInvocationError

pytestmark = pytest.mark.anyio


def _create_mock_runtime(store: BaseStore | None = None) -> Mock:
    """Create a mock Runtime object for testing ToolNode outside of graph context."""
    mock_runtime = Mock()
    mock_runtime.store = store
    mock_runtime.context = None
    mock_runtime.stream_writer = lambda *args, **kwargs: None
    return mock_runtime


def _create_config_with_runtime(store: BaseStore | None = None) -> RunnableConfig:
    """Create a RunnableConfig with mock Runtime for testing ToolNode."""
    return {"configurable": {"__pregel_runtime": _create_mock_runtime(store)}}


async def test_filter_injected_state_validation_errors() -> None:
    """Test that validation errors for InjectedState arguments are filtered out.

    InjectedState parameters are not controlled by the LLM, so any validation
    errors related to them should not appear in error messages. This ensures
    the LLM receives only actionable feedback about its own tool call arguments.
    """

    @dec_tool
    def my_tool(
        value: int,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Tool that uses injected state.

        Args:
            value: An integer value.
            state: The graph state (injected).
        """
        return f"value={value}, messages={len(state.get('messages', []))}"

    tool_node = ToolNode([my_tool])

    # Call with invalid 'value' argument (should be int, not str)
    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {"value": "not_an_int"},  # Invalid type
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    # Should get a ToolMessage with error
    assert len(result["messages"]) == 1
    tool_message = result["messages"][0]
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "call_1"

    # Error should mention 'value' but NOT 'state' (which is injected)
    assert "value" in tool_message.content
    assert "state" not in tool_message.content.lower()


async def test_filter_injected_store_validation_errors() -> None:
    """Test that validation errors for InjectedStore arguments are filtered out.

    InjectedStore parameters are not controlled by the LLM, so any validation
    errors related to them should not appear in error messages. This keeps
    error feedback focused on LLM-controllable parameters.
    """

    @dec_tool
    def my_tool(
        key: str,
        store: Annotated[BaseStore, InjectedStore()],
    ) -> str:
        """Tool that uses injected store.

        Args:
            key: A key to look up.
            store: The persistent store (injected).
        """
        return f"key={key}"

    tool_node = ToolNode([my_tool])

    # Call with invalid 'key' argument (missing required argument)
    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {},  # Missing 'key'
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(store=InMemoryStore()),
    )

    # Should get a ToolMessage with error
    assert len(result["messages"]) == 1
    tool_message = result["messages"][0]
    assert tool_message.status == "error"

    # Error should mention 'key' is required
    assert "key" in tool_message.content.lower()
    # The error should be about 'key' field specifically (not about store field)
    # Note: 'store' might appear in input_value representation, but the validation
    # error itself should only be for 'key'
    assert (
        "field required" in tool_message.content.lower()
        or "missing" in tool_message.content.lower()
    )


async def test_filter_tool_runtime_validation_errors() -> None:
    """Test that validation errors for ToolRuntime arguments are filtered out.

    ToolRuntime parameters are not controlled by the LLM, so any validation
    errors related to them should not appear in error messages. This ensures
    the LLM only sees errors for parameters it can fix.
    """

    @dec_tool
    def my_tool(
        query: str,
        runtime: ToolRuntime,
    ) -> str:
        """Tool that uses ToolRuntime.

        Args:
            query: A query string.
            runtime: The tool runtime context (injected).
        """
        return f"query={query}"

    tool_node = ToolNode([my_tool])

    # Call with invalid 'query' argument (wrong type)
    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {"query": 123},  # Should be str, not int
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    # Should get a ToolMessage with error
    assert len(result["messages"]) == 1
    tool_message = result["messages"][0]
    assert tool_message.status == "error"

    # Error should mention 'query' but NOT 'runtime' (which is injected)
    assert "query" in tool_message.content.lower()
    assert "runtime" not in tool_message.content.lower()


async def test_filter_multiple_injected_args() -> None:
    """Test filtering when a tool has multiple injected arguments.

    When a tool uses multiple injected parameters (state, store, runtime), none of
    them should appear in validation error messages since they're all system-provided
    and not controlled by the LLM. Only LLM-controllable parameter errors should appear.
    """

    @dec_tool
    def my_tool(
        value: int,
        state: Annotated[dict, InjectedState],
        store: Annotated[BaseStore, InjectedStore()],
        runtime: ToolRuntime,
    ) -> str:
        """Tool with multiple injected arguments.

        Args:
            value: An integer value.
            state: The graph state (injected).
            store: The persistent store (injected).
            runtime: The tool runtime context (injected).
        """
        return f"value={value}"

    tool_node = ToolNode([my_tool])

    # Call with invalid 'value' - injected args should be filtered from error
    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {"value": "not_an_int"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(store=InMemoryStore()),
    )

    tool_message = result["messages"][0]
    assert tool_message.status == "error"

    # Only 'value' error should be reported
    assert "value" in tool_message.content
    # None of the injected args should appear in error
    assert "state" not in tool_message.content.lower()
    assert "store" not in tool_message.content.lower()
    assert "runtime" not in tool_message.content.lower()


async def test_no_filtering_when_all_errors_are_model_args() -> None:
    """Test that validation errors for LLM-controlled arguments are preserved.

    When validation fails for arguments the LLM controls, those errors should
    be fully reported to help the LLM correct its tool calls. This ensures
    the LLM receives complete feedback about all issues it can fix.
    """

    @dec_tool
    def my_tool(
        value1: int,
        value2: str,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Tool with both regular and injected arguments.

        Args:
            value1: First value.
            value2: Second value.
            state: The graph state (injected).
        """
        return f"value1={value1}, value2={value2}"

    tool_node = ToolNode([my_tool])

    # Call with invalid arguments for BOTH non-injected parameters
    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {
                                "value1": "not_an_int",  # Invalid
                                "value2": 456,  # Invalid (should be str)
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][0]
    assert tool_message.status == "error"

    # Both errors should be present
    assert "value1" in tool_message.content
    assert "value2" in tool_message.content
    # Injected state should not appear
    assert "state" not in tool_message.content.lower()


async def test_validation_error_with_no_injected_args() -> None:
    """Test that tools without injected arguments show all validation errors.

    For tools that only have LLM-controlled parameters, all validation errors
    should be reported since everything is under the LLM's control and can be
    corrected by the LLM in subsequent tool calls.
    """

    @dec_tool
    def my_tool(value1: int, value2: str) -> str:
        """Regular tool without injected arguments.

        Args:
            value1: First value.
            value2: Second value.
        """
        return f"{value1} {value2}"

    tool_node = ToolNode([my_tool])

    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {"value1": "invalid", "value2": 123},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][0]
    assert tool_message.status == "error"

    # Both errors should be present since there are no injected args to filter
    assert "value1" in tool_message.content
    assert "value2" in tool_message.content


async def test_tool_invocation_error_without_handle_errors() -> None:
    """Test that ToolInvocationError contains only LLM-controlled parameter errors.

    When handle_tool_errors is False, the raised ToolInvocationError should still
    filter out system-injected arguments from the error details, ensuring that
    error messages focus on what the LLM can control.
    """

    @dec_tool
    def my_tool(
        value: int,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Tool with injected state.

        Args:
            value: An integer value.
            state: The graph state (injected).
        """
        return f"value={value}"

    tool_node = ToolNode([my_tool], handle_tool_errors=False)

    # Should raise ToolInvocationError with filtered errors
    with pytest.raises(ToolInvocationError) as exc_info:
        await tool_node.ainvoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "my_tool",
                                "args": {"value": "not_an_int"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            config=_create_config_with_runtime(),
        )

    error = exc_info.value
    assert error.tool_name == "my_tool"
    assert error.filtered_errors is not None
    assert len(error.filtered_errors) > 0

    # Filtered errors should only contain 'value' error, not 'state'
    error_locs = [err["loc"] for err in error.filtered_errors]
    assert any("value" in str(loc) for loc in error_locs)
    assert not any("state" in str(loc) for loc in error_locs)


async def test_sync_tool_validation_error_filtering() -> None:
    """Test that error filtering works for sync tools.

    Error filtering should work identically for both sync and async tool execution,
    excluding injected arguments from validation error messages.
    """

    @dec_tool
    def my_tool(
        value: int,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Sync tool with injected state.

        Args:
            value: An integer value.
            state: The graph state (injected).
        """
        return f"value={value}"

    tool_node = ToolNode([my_tool])

    # Test sync invocation
    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {"value": "not_an_int"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][0]
    assert tool_message.status == "error"
    assert "value" in tool_message.content
    assert "state" not in tool_message.content.lower()
