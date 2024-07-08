import pytest
from uuid import UUID
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.graph import add_messages

def test_add_single_message():
    left = [HumanMessage(content="Hello", id="1")]
    right = AIMessage(content="Hi there!", id="2")
    result = add_messages(left, right)
    expected_result = [HumanMessage(content="Hello", id="1"), AIMessage(content="Hi there!", id="2")]
    assert result == expected_result

def test_add_multiple_messages():
    left = [HumanMessage(content="Hello", id="1")]
    right = [AIMessage(content="Hi there!", id="2"), SystemMessage(content="System message", id="3")]
    result = add_messages(left, right)
    expected_result = [HumanMessage(content="Hello", id="1"), AIMessage(content="Hi there!", id="2"),  SystemMessage(content="System message", id="3")]
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
    left = [HumanMessage(content="Hello", id="1"), AIMessage(content="Hi there!", id="2")]
    right = RemoveMessage(id="2")
    result = add_messages(left, right)
    expected_result = [HumanMessage(content="Hello", id="1")]
    assert result == expected_result

def test_remove_nonexistent_message():
    left = [HumanMessage(content="Hello", id="1")]
    right = RemoveMessage(id="2")
    with pytest.raises(ValueError, match="Attempting to delete a message with an ID that doesn't exist"):
        add_messages(left, right)

def test_mixed_operations():
    left = [HumanMessage(content="Hello", id="1"), AIMessage(content="Hi there!", id="2")]
    right = [
        HumanMessage(content="Updated hello", id="1"),
        RemoveMessage(id="2"),
        SystemMessage(content="New message", id="3")
    ]
    result = add_messages(left, right)
    expected_result = [HumanMessage(content="Updated hello", id="1"),
        SystemMessage(content="New message", id="3")]
    assert result == expected_result

def test_empty_inputs():
    assert add_messages([], []) == []
    assert add_messages([], [HumanMessage(content="Hello", id="1")]) == [HumanMessage(content="Hello", id="1")]
    assert add_messages([HumanMessage(content="Hello", id="1")], []) == [HumanMessage(content="Hello", id="1")]

def test_non_list_inputs():
    left = HumanMessage(content="Hello", id="1")
    right = AIMessage(content="Hi there!", id="2")
    result = add_messages(left, right)
    expected_result = [HumanMessage(content="Hello", id="1"), AIMessage(content="Hi there!", id="2")]
    assert result == expected_result


def test_delete_all():
    left = [HumanMessage(content="Hello", id="1"), AIMessage(content="Hi there!", id="2")]
    right = [
        RemoveMessage(id="1"),
        RemoveMessage(id="2"),
    ]
    result = add_messages(left, right)
    expected_result = []
    assert result == expected_result
