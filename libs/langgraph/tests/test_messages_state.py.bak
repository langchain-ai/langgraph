from typing import Annotated
from uuid import UUID

import langchain_core
import pytest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph import add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES, MessagesState, push_message
from langgraph.graph.state import StateGraph
from tests.messages import _AnyIdHumanMessage

_, CORE_MINOR, CORE_PATCH = (
    int("".join(c for c in v if c.isdigit()))
    for v in langchain_core.__version__.split(".")
)


def test_add_single_message():
    left = [HumanMessage(content="Hello", id="1")]
    right = AIMessage(content="Hi there!", id="2")
    result = add_messages(left, right)
    expected_result = [
        HumanMessage(content="Hello", id="1"),
        AIMessage(content="Hi there!", id="2"),
    ]
    assert result == expected_result


def test_add_multiple_messages():
    left = [HumanMessage(content="Hello", id="1")]
    right = [
        AIMessage(content="Hi there!", id="2"),
        SystemMessage(content="System message", id="3"),
    ]
    result = add_messages(left, right)
    expected_result = [
        HumanMessage(content="Hello", id="1"),
        AIMessage(content="Hi there!", id="2"),
        SystemMessage(content="System message", id="3"),
    ]
    assert result == expected_result


def test_update_existing_message():
    left = [HumanMessage(content="Hello", id="1")]
    right = HumanMessage(content="Hello again", id="1")
    result = add_messages(left, right)
    expected_result = [HumanMessage(content="Hello again", id="1")]
    assert result == expected_result


def test_missing_ids():
    left = [HumanMessage(content="Hello")]
    right = [AIMessage(content="Hi there!")]
    result = add_messages(left, right)
    assert len(result) == 2
    assert all(isinstance(m.id, str) and UUID(m.id, version=4) for m in result)


def test_duplicates_in_input():
    left = []
    right = [
        AIMessage(id="1", content="Hi there!"),
        AIMessage(id="1", content="Hi there again!"),
    ]
    result = add_messages(left, right)
    assert len(result) == 1
    assert result[0].id == "1"
    assert result[0].content == "Hi there again!"


def test_duplicates_in_input_with_remove():
    left = [AIMessage(id="1", content="Hello!")]
    right = [
        RemoveMessage(id="1"),
        AIMessage(id="1", content="Hi there!"),
        AIMessage(id="1", content="Hi there again!"),
    ]
    result = add_messages(left, right)
    assert len(result) == 1
    assert result[0].id == "1"
    assert result[0].content == "Hi there again!"


def test_remove_message():
    left = [
        HumanMessage(content="Hello", id="1"),
        AIMessage(content="Hi there!", id="2"),
    ]
    right = RemoveMessage(id="2")
    result = add_messages(left, right)
    expected_result = [HumanMessage(content="Hello", id="1")]
    assert result == expected_result


def test_duplicate_remove_message():
    left = [
        HumanMessage(content="Hello", id="1"),
        AIMessage(content="Hi there!", id="2"),
    ]
    right = [RemoveMessage(id="2"), RemoveMessage(id="2")]
    result = add_messages(left, right)
    expected_result = [HumanMessage(content="Hello", id="1")]
    assert result == expected_result


def test_remove_nonexistent_message():
    left = [HumanMessage(content="Hello", id="1")]
    right = RemoveMessage(id="2")
    with pytest.raises(
        ValueError, match="Attempting to delete a message with an ID that doesn't exist"
    ):
        add_messages(left, right)


def test_mixed_operations():
    left = [
        HumanMessage(content="Hello", id="1"),
        AIMessage(content="Hi there!", id="2"),
    ]
    right = [
        HumanMessage(content="Updated hello", id="1"),
        RemoveMessage(id="2"),
        SystemMessage(content="New message", id="3"),
    ]
    result = add_messages(left, right)
    expected_result = [
        HumanMessage(content="Updated hello", id="1"),
        SystemMessage(content="New message", id="3"),
    ]
    assert result == expected_result


def test_empty_inputs():
    assert add_messages([], []) == []
    assert add_messages([], [HumanMessage(content="Hello", id="1")]) == [
        HumanMessage(content="Hello", id="1")
    ]
    assert add_messages([HumanMessage(content="Hello", id="1")], []) == [
        HumanMessage(content="Hello", id="1")
    ]


def test_non_list_inputs():
    left = HumanMessage(content="Hello", id="1")
    right = AIMessage(content="Hi there!", id="2")
    result = add_messages(left, right)
    expected_result = [
        HumanMessage(content="Hello", id="1"),
        AIMessage(content="Hi there!", id="2"),
    ]
    assert result == expected_result


def test_delete_all():
    left = [
        HumanMessage(content="Hello", id="1"),
        AIMessage(content="Hi there!", id="2"),
    ]
    right = [
        RemoveMessage(id="1"),
        RemoveMessage(id="2"),
    ]
    result = add_messages(left, right)
    expected_result = []
    assert result == expected_result


class MessagesStatePydantic(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]


MESSAGES_STATE_SCHEMAS = [MessagesState, MessagesStatePydantic]


@pytest.mark.parametrize("state_schema", MESSAGES_STATE_SCHEMAS)
def test_messages_state(state_schema):
    def foo(state):
        return {"messages": [HumanMessage("foo")]}

    graph = StateGraph(state_schema)
    graph.add_edge(START, "foo")
    graph.add_edge("foo", END)
    graph.add_node(foo)

    app = graph.compile()

    assert app.invoke({"messages": [("user", "meow")]}) == {
        "messages": [
            _AnyIdHumanMessage(content="meow"),
            _AnyIdHumanMessage(content="foo"),
        ]
    }


@pytest.mark.skipif(
    condition=not ((CORE_MINOR == 3 and CORE_PATCH >= 11) or CORE_MINOR > 3),
    reason="Requires langchain_core>=0.3.11.",
)
def test_messages_state_format_openai():
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages(format="langchain-openai")]

    def foo(state):
        messages = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Here's an image:",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "1234",
                        },
                    },
                ]
            ),
            AIMessage(
                content=[
                    {
                        "type": "tool_use",
                        "name": "foo",
                        "input": {"bar": "baz"},
                        "id": "1",
                    }
                ]
            ),
            HumanMessage(
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "1",
                        "is_error": False,
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": "1234",
                                },
                            },
                        ],
                    }
                ]
            ),
        ]
        return {"messages": messages}

    expected = [
        HumanMessage(content="meow"),
        HumanMessage(
            content=[
                {"type": "text", "text": "Here's an image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,1234"},
                },
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "foo",
                    "type": "tool_calls",
                    "args": {"bar": "baz"},
                    "id": "1",
                }
            ],
        ),
        ToolMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,1234"},
                }
            ],
            tool_call_id="1",
        ),
    ]

    graph = StateGraph(State)
    graph.add_edge(START, "foo")
    graph.add_edge("foo", END)
    graph.add_node(foo)

    app = graph.compile()

    result = app.invoke({"messages": [("user", "meow")]})
    for m in result["messages"]:
        m.id = None
    assert result == {"messages": expected}


def test_remove_all_messages():
    # simple removal
    left = [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
    right = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
    result = add_messages(left, right)
    assert result == []

    # removal and update (i.e., overwriting)
    left = [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
    right = [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        HumanMessage(content="Updated hello"),
    ]
    result = add_messages(left, right)
    assert result == [_AnyIdHumanMessage(content="Updated hello")]

    # test removing preceding messages in the right list
    left = [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
    right = [
        HumanMessage(content="Updated hello"),
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        HumanMessage(content="Updated hi there"),
    ]
    result = add_messages(left, right)
    assert result == [
        _AnyIdHumanMessage(content="Updated hi there"),
    ]


def test_push_messages_in_graph():
    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def chat(_: MessagesState) -> MessagesState:
        with pytest.raises(ValueError, match="Message ID is required"):
            push_message(AIMessage(content="No ID"))

        push_message(AIMessage(content="First", id="1"))
        push_message(HumanMessage(content="Second", id="2"))
        push_message(AIMessage(content="Third", id="3"))

    builder = StateGraph(MessagesState)
    builder.add_node(chat)
    builder.add_edge(START, "chat")

    graph = builder.compile()

    messages, values = [], None
    for event, chunk in graph.stream(
        {"messages": []}, stream_mode=["messages", "values"]
    ):
        if event == "values":
            values = chunk
        elif event == "messages":
            message, _ = chunk
            messages.append(message)

    assert values["messages"] == messages


def test_append_only_mode_allows_new_messages():
    """Test that append_only mode allows adding new messages."""
    left = [HumanMessage(content="Hello", id="1")]
    right = [AIMessage(content="Hi there!", id="2")]
    result = add_messages(left, right, mode="append_only")
    expected = [
        HumanMessage(content="Hello", id="1"),
        AIMessage(content="Hi there!", id="2"),
    ]
    assert result == expected


def test_append_only_mode_prevents_updates():
    """Test that append_only mode raises an error when trying to update an existing message."""
    left = [HumanMessage(content="Hello", id="1")]
    right = [HumanMessage(content="Hello again", id="1")]
    with pytest.raises(
        ValueError,
        match="Cannot update existing message with ID '1' in append_only mode",
    ):
        add_messages(left, right, mode="append_only")


def test_append_only_mode_prevents_remove_messages():
    """Test that append_only mode prevents RemoveMessage operations."""
    left = [
        HumanMessage(content="Hello", id="1"),
        AIMessage(content="Hi there!", id="2"),
    ]
    right = [RemoveMessage(id="2")]
    with pytest.raises(
        ValueError,
        match="Cannot remove existing message with ID '2' in append_only mode",
    ):
        add_messages(left, right, mode="append_only")


def test_append_only_mode_prevents_remove_all_messages():
    """Test that append_only mode prevents REMOVE_ALL_MESSAGES operations."""
    left = [
        HumanMessage(content="Hello", id="1"),
        AIMessage(content="Hi there!", id="2"),
    ]
    right = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
    with pytest.raises(
        ValueError,
        match="Cannot remove all messages in append_only mode",
    ):
        add_messages(left, right, mode="append_only")


def test_append_only_mode_with_multiple_new_messages():
    """Test that append_only mode allows adding multiple new messages."""
    left = [HumanMessage(content="Hello", id="1")]
    right = [
        AIMessage(content="Hi there!", id="2"),
        SystemMessage(content="System message", id="3"),
        HumanMessage(content="Another message", id="4"),
    ]
    result = add_messages(left, right, mode="append_only")
    expected = [
        HumanMessage(content="Hello", id="1"),
        AIMessage(content="Hi there!", id="2"),
        SystemMessage(content="System message", id="3"),
        HumanMessage(content="Another message", id="4"),
    ]
    assert result == expected


def test_append_only_mode_with_mixed_operations():
    """Test that append_only mode fails when mixing new messages with updates."""
    left = [HumanMessage(content="Hello", id="1")]
    right = [
        AIMessage(content="Hi there!", id="2"),  # new message
        HumanMessage(content="Updated hello", id="1"),  # update attempt
    ]
    with pytest.raises(
        ValueError,
        match="Cannot update existing message with ID '1' in append_only mode",
    ):
        add_messages(left, right, mode="append_only")


def test_allow_everything_mode_default_behavior():
    """Test that allow_everything mode is the default and allows updates."""
    left = [HumanMessage(content="Hello", id="1")]
    right = [HumanMessage(content="Hello again", id="1")]
    # Test without specifying mode (should default to allow_everything)
    result = add_messages(left, right)
    expected = [HumanMessage(content="Hello again", id="1")]
    assert result == expected

    # Test with explicit mode
    result = add_messages(left, right, mode="allow_everything")
    assert result == expected


def test_append_only_mode_in_state_graph():
    """Test append_only mode works correctly in a StateGraph."""

    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages(mode="append_only")]

    def add_message(state: State):
        return {"messages": [AIMessage(content="Response", id="2")]}

    def try_update_message(state: State):
        # This should fail because message with id="1" already exists
        return {"messages": [HumanMessage(content="Updated", id="1")]}

    # Test successful case
    builder = StateGraph(State)
    builder.add_node("add_message", add_message)
    builder.add_edge(START, "add_message")
    builder.add_edge("add_message", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [HumanMessage(content="Hello", id="1")]})
    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "Hello"
    assert result["messages"][1].content == "Response"

    # Test failure case
    builder2 = StateGraph(State)
    builder2.add_node("try_update", try_update_message)
    builder2.add_edge(START, "try_update")
    builder2.add_edge("try_update", END)
    graph2 = builder2.compile()

    with pytest.raises(
        ValueError,
        match="Cannot update existing message with ID '1' in append_only mode",
    ):
        graph2.invoke({"messages": [HumanMessage(content="Hello", id="1")]})
