from typing import Annotated
from uuid import UUID

import langchain_core
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
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


def test_fast_path_preserves_format_openai():
    """Pure-append fast path must still apply the `langchain-openai` formatter."""
    left = [HumanMessage(content="prior", id="1")]
    right = [
        AIMessage(
            content=[
                {
                    "type": "tool_use",
                    "name": "foo",
                    "input": {"bar": "baz"},
                    "id": "t1",
                }
            ],
            id="2",
        )
    ]
    result = add_messages(left, right, format="langchain-openai")
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "prior"
    assert isinstance(result[1], AIMessage)
    # formatter collapses the tool_use content block into `tool_calls`
    assert result[1].content == ""
    assert len(result[1].tool_calls) == 1
    assert result[1].tool_calls[0]["name"] == "foo"
    assert result[1].tool_calls[0]["args"] == {"bar": "baz"}
    assert result[1].tool_calls[0]["id"] == "t1"


def test_fast_path_rejects_invalid_format():
    """Pure-append fast path must validate the `format` arg like the slow path."""
    left = [HumanMessage(content="prior", id="1")]
    right = [AIMessage(content="new", id="2")]
    with pytest.raises(ValueError, match="Unrecognized format="):
        add_messages(left, right, format="bogus")  # type: ignore[arg-type]


def test_left_starting_with_chunk_is_normalized():
    """Opt-1 guard: a `BaseMessageChunk` at left[0] must trigger full conversion."""
    chunk = AIMessageChunk(content="chunk", id="c1")
    result = add_messages([chunk], [HumanMessage(content="h", id="h1")])
    assert len(result) == 2
    # chunk must be converted to a non-chunk message
    assert type(result[0]).__name__ == "AIMessage"
    assert result[0].id == "c1"
    assert result[1].id == "h1"


def test_left_as_dicts_is_normalized():
    """Opt-1 guard: dicts at left[0] must trigger full conversion."""
    left = [{"role": "user", "content": "hi", "id": "d1"}]
    right = [AIMessage(content="reply", id="a1")]
    result = add_messages(left, right)
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert result[0].id == "d1"
    assert result[0].content == "hi"


def test_left_as_tuples_is_normalized():
    """Opt-1 guard: tuple-form messages must trigger full conversion."""
    left = [("user", "hi")]
    right = [AIMessage(content="reply", id="a1")]
    result = add_messages(left, right)
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    # id is auto-assigned
    assert isinstance(result[0].id, str) and UUID(result[0].id, version=4)


def test_left_first_msg_missing_id_is_normalized():
    """Opt-1 guard: a BaseMessage without an id at left[0] falls to the else branch."""
    left = [HumanMessage(content="hi")]  # no id
    right = [AIMessage(content="reply", id="a1")]
    result = add_messages(left, right)
    assert len(result) == 2
    # left's id must have been auto-assigned
    assert isinstance(result[0].id, str) and UUID(result[0].id, version=4)


def test_duplicate_ids_in_right_with_nonempty_left():
    """Opt-2 guard: intra-right duplicate ids must take slow path (dedup kept)."""
    left = [HumanMessage(content="prior", id="1")]
    right = [
        AIMessage(content="first", id="2"),
        AIMessage(content="second", id="2"),
    ]
    result = add_messages(left, right)
    assert len(result) == 2
    assert result[0].id == "1"
    assert result[1].id == "2"
    assert result[1].content == "second"


def test_right_with_none_ids_pure_append():
    """Fast path still correct when right entries start with id=None (fresh uuids assigned)."""
    left = [HumanMessage(content="prior", id="1")]
    right = [AIMessage(content="a"), AIMessage(content="b")]
    result = add_messages(left, right)
    assert len(result) == 3
    assert result[0].id == "1"
    for m in result[1:]:
        assert isinstance(m.id, str) and UUID(m.id, version=4)
    # fresh uuids must be distinct
    assert result[1].id != result[2].id


def test_fast_path_returns_fresh_list():
    """Fast path must return a new list object (not mutate or alias left)."""
    left = [HumanMessage(content="prior", id="1")]
    right = [AIMessage(content="new", id="2")]
    result = add_messages(left, right)
    assert result is not left
    # left must be untouched
    assert len(left) == 1
    assert left[0].id == "1"


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
