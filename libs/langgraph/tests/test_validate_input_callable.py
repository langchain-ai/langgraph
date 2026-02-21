"""Tests for validate_input with callable approach."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState, validate_messages_append_only
from langgraph.types import Command


def test_validate_messages_append_only_allows_new():
    """Test that validate_messages_append_only allows adding new messages."""

    def chatbot(state: MessagesState) -> MessagesState:
        return {"messages": [AIMessage(content="Response")]}

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.set_entry_point("chatbot")
    builder.set_finish_point("chatbot")

    graph = builder.compile(validate_input=validate_messages_append_only)

    result = graph.invoke({"messages": [HumanMessage(content="Hello", id="1")]})
    assert len(result["messages"]) == 2


def test_validate_messages_append_only_blocks_updates():
    """Test that validate_messages_append_only blocks message updates."""

    def chatbot(state: MessagesState) -> MessagesState:
        return {"messages": [AIMessage(content="Response")]}

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.set_entry_point("chatbot")
    builder.set_finish_point("chatbot")

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer, validate_input=validate_messages_append_only
    )

    config = {"configurable": {"thread_id": "test"}}
    graph.invoke({"messages": [HumanMessage(content="Hello", id="1")]}, config)

    # Try to update - should fail
    with pytest.raises(ValueError, match="Cannot update existing message"):
        graph.invoke({"messages": [HumanMessage(content="Modified", id="1")]}, config)


def test_validate_messages_append_only_blocks_removals():
    """Test that validate_messages_append_only blocks removals."""

    def chatbot(state: MessagesState) -> MessagesState:
        return {"messages": []}

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.set_entry_point("chatbot")
    builder.set_finish_point("chatbot")

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer, validate_input=validate_messages_append_only
    )

    config = {"configurable": {"thread_id": "test"}}
    graph.invoke({"messages": [HumanMessage(content="Hello", id="1")]}, config)

    # Try to remove - should fail
    with pytest.raises(ValueError, match="Cannot remove existing message"):
        graph.invoke({"messages": [RemoveMessage(id="1")]}, config)


def test_validate_with_command_update():
    """Test that Command.update is also validated."""

    def chatbot(state: MessagesState) -> MessagesState:
        return {"messages": [AIMessage(content="Response")]}

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.set_entry_point("chatbot")
    builder.set_finish_point("chatbot")

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer, validate_input=validate_messages_append_only
    )

    config = {"configurable": {"thread_id": "test"}}
    graph.invoke({"messages": [HumanMessage(content="Hello", id="1")]}, config)

    # Try to update via Command - should fail
    with pytest.raises(ValueError, match="Cannot update existing message"):
        graph.invoke(
            Command(update={"messages": [HumanMessage(content="Modified", id="1")]}),
            config,
        )


def test_custom_validator():
    """Test with custom validation function."""

    def my_validator(input: dict, current_state: dict) -> None:
        if "forbidden" in str(input):
            raise ValueError("Forbidden content")

    def node(state: MessagesState) -> MessagesState:
        return {"messages": [AIMessage(content="OK")]}

    builder = StateGraph(MessagesState)
    builder.add_node("node", node)
    builder.set_entry_point("node")
    builder.set_finish_point("node")

    graph = builder.compile(validate_input=my_validator)

    # Should work
    graph.invoke({"messages": [("user", "Hello")]})

    # Should fail
    with pytest.raises(ValueError, match="Forbidden content"):
        graph.invoke({"messages": [("user", "forbidden word")]})


def test_node_bypasses_validation():
    """Test that internal node operations bypass validation."""

    def node_that_modifies(state: MessagesState) -> MessagesState:
        # Node modifies existing message - should work even with strict validator
        messages = state["messages"]
        if messages:
            updated = messages[0].model_copy(update={"content": "Modified by node"})
            return {"messages": [updated]}
        return {"messages": []}

    builder = StateGraph(MessagesState)
    builder.add_node("modifier", node_that_modifies)
    builder.set_entry_point("modifier")
    builder.set_finish_point("modifier")

    graph = builder.compile(validate_input=validate_messages_append_only)

    # Node can modify - validation only applies to external input
    result = graph.invoke({"messages": [HumanMessage(content="Original", id="1")]})
    assert any("Modified by node" in m.content for m in result["messages"])


def test_without_validator_allows_everything():
    """Test backwards compatibility - without validator, everything is allowed."""

    def chatbot(state: MessagesState) -> MessagesState:
        return {"messages": []}

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.set_entry_point("chatbot")
    builder.set_finish_point("chatbot")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)  # No validator

    config = {"configurable": {"thread_id": "test"}}
    graph.invoke({"messages": [HumanMessage(content="Hello", id="1")]}, config)

    # Should allow update (old behavior)
    result = graph.invoke(
        {"messages": [HumanMessage(content="Modified", id="1")]}, config
    )
    assert result["messages"][0].content == "Modified"
