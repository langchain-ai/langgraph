from typing import Annotated
from uuid import UUID

import pytest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

from langgraph.graph import add_messages
from langgraph.graph.message import MessagesState
from langgraph.graph.state import END, START, StateGraph
from tests.conftest import IS_LANGCHAIN_CORE_030_OR_GREATER
from tests.messages import _AnyIdHumanMessage


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


class MessagesStatePydanticV1(BaseModelV1):
    messages: Annotated[list[AnyMessage], add_messages]


class MessagesStatePydantic(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]


MESSAGES_STATE_SCHEMAS = [MessagesState, MessagesStatePydanticV1]
if IS_LANGCHAIN_CORE_030_OR_GREATER:
    MESSAGES_STATE_SCHEMAS.append(MessagesStatePydantic)


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
