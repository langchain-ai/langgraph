"""Tests for langgraph.errors module."""

import pytest

from langgraph.errors import (
    EmptyInputError,
    ErrorCode,
    GraphBubbleUp,
    GraphInterrupt,
    GraphRecursionError,
    InvalidUpdateError,
    ParentCommand,
    TaskNotFound,
    create_error_message,
)
from langgraph.types import Command, Interrupt

# --- ErrorCode tests ---


def test_error_code_values() -> None:
    """Test all ErrorCode enum members have expected string values."""
    assert ErrorCode.GRAPH_RECURSION_LIMIT.value == "GRAPH_RECURSION_LIMIT"
    assert (
        ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE.value
        == "INVALID_CONCURRENT_GRAPH_UPDATE"
    )
    assert (
        ErrorCode.INVALID_GRAPH_NODE_RETURN_VALUE.value
        == "INVALID_GRAPH_NODE_RETURN_VALUE"
    )
    assert ErrorCode.MULTIPLE_SUBGRAPHS.value == "MULTIPLE_SUBGRAPHS"
    assert ErrorCode.INVALID_CHAT_HISTORY.value == "INVALID_CHAT_HISTORY"


def test_error_code_member_count() -> None:
    """Test that ErrorCode has the expected number of members."""
    assert len(ErrorCode) == 5


# --- create_error_message tests ---


def test_create_error_message_includes_message() -> None:
    """Test create_error_message includes the user-provided message."""
    msg = create_error_message(
        message="Something went wrong",
        error_code=ErrorCode.GRAPH_RECURSION_LIMIT,
    )
    assert "Something went wrong" in msg


def test_create_error_message_includes_troubleshooting_url() -> None:
    """Test create_error_message includes a troubleshooting URL."""
    msg = create_error_message(
        message="test",
        error_code=ErrorCode.GRAPH_RECURSION_LIMIT,
    )
    assert "https://docs.langchain.com/oss/python/langgraph/errors/" in msg
    assert "GRAPH_RECURSION_LIMIT" in msg


def test_create_error_message_all_error_codes() -> None:
    """Test create_error_message works for every ErrorCode member."""
    for code in ErrorCode:
        msg = create_error_message(message="test error", error_code=code)
        assert code.value in msg
        assert "test error" in msg
        assert "troubleshooting" in msg.lower() or "docs.langchain.com" in msg


# --- Exception class hierarchy tests ---


def test_graph_recursion_error_is_recursion_error() -> None:
    """Test GraphRecursionError inherits from RecursionError."""
    assert issubclass(GraphRecursionError, RecursionError)

    with pytest.raises(RecursionError):
        raise GraphRecursionError("too many steps")


def test_invalid_update_error_is_exception() -> None:
    """Test InvalidUpdateError inherits from Exception."""
    assert issubclass(InvalidUpdateError, Exception)

    with pytest.raises(InvalidUpdateError):
        raise InvalidUpdateError("bad update")


def test_graph_bubble_up_is_exception() -> None:
    """Test GraphBubbleUp inherits from Exception."""
    assert issubclass(GraphBubbleUp, Exception)


def test_graph_interrupt_is_graph_bubble_up() -> None:
    """Test GraphInterrupt inherits from GraphBubbleUp."""
    assert issubclass(GraphInterrupt, GraphBubbleUp)


def test_graph_interrupt_with_empty_interrupts() -> None:
    """Test GraphInterrupt can be created with no interrupts."""
    exc = GraphInterrupt()
    assert exc.args == ((),)


def test_graph_interrupt_with_interrupts() -> None:
    """Test GraphInterrupt stores interrupt objects."""
    interrupts = (Interrupt(value="please confirm"),)
    exc = GraphInterrupt(interrupts)
    assert exc.args == (interrupts,)


def test_empty_input_error() -> None:
    """Test EmptyInputError can be raised and caught."""
    assert issubclass(EmptyInputError, Exception)

    with pytest.raises(EmptyInputError):
        raise EmptyInputError("no input provided")


def test_task_not_found() -> None:
    """Test TaskNotFound can be raised and caught."""
    assert issubclass(TaskNotFound, Exception)

    with pytest.raises(TaskNotFound, match="missing task"):
        raise TaskNotFound("missing task")


def test_parent_command_is_graph_bubble_up() -> None:
    """Test ParentCommand inherits from GraphBubbleUp."""
    assert issubclass(ParentCommand, GraphBubbleUp)


def test_parent_command_stores_command() -> None:
    """Test ParentCommand wraps a Command object."""
    cmd = Command(update={"key": "value"})
    exc = ParentCommand(cmd)
    assert exc.args == (cmd,)


# --- Error message formatting tests ---


def test_graph_recursion_error_message() -> None:
    """Test GraphRecursionError preserves its message."""
    msg = "Recursion limit of 25 reached"
    exc = GraphRecursionError(msg)
    assert str(exc) == msg


def test_invalid_update_error_message() -> None:
    """Test InvalidUpdateError preserves its message."""
    msg = "Cannot update channel 'foo' with multiple values"
    exc = InvalidUpdateError(msg)
    assert str(exc) == msg


def test_create_error_message_format() -> None:
    """Test the exact format of create_error_message output."""
    msg = create_error_message(
        message="Test message",
        error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
    )
    assert msg.startswith("Test message\n")
    assert msg.endswith("errors/INVALID_CONCURRENT_GRAPH_UPDATE")
