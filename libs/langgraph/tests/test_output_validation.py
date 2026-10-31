"""Tests for node output validation (Issue #6491)"""

from typing import List

import pytest
from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph


class State(BaseModel):
    items: List[str] = []


def test_node_output_validation_invalid():
    """Test that invalid node output is caught immediately"""

    def create_invalid_state(state: State) -> dict:
        """Node that returns invalid state"""
        return {"items": state.items + [None]}  # None is not a valid str!

    graph = StateGraph(State)
    graph.add_node("bad", create_invalid_state)
    graph.add_edge(START, "bad")
    graph.add_edge("bad", END)

    app = graph.compile(checkpointer=MemorySaver())

    # Should raise InvalidUpdateError when node returns invalid output
    with pytest.raises(InvalidUpdateError) as exc_info:
        app.invoke(State(items=["hello"]), {"configurable": {"thread_id": "1"}})

    # Error message should mention the node name
    assert "bad" in str(exc_info.value).lower()
    assert "invalid output state" in str(exc_info.value).lower()


def test_node_output_validation_valid():
    """Test that valid node output passes validation"""

    def create_valid_state(state: State) -> dict:
        """Node that returns valid state"""
        return {"items": state.items + ["world"]}

    graph = StateGraph(State)
    graph.add_node("good", create_valid_state)
    graph.add_edge(START, "good")
    graph.add_edge("good", END)

    app = graph.compile(checkpointer=MemorySaver())

    # Should succeed
    result = app.invoke(State(items=["hello"]), {"configurable": {"thread_id": "1"}})
    assert result["items"] == ["hello", "world"]


def test_node_output_validation_with_multiple_nodes():
    """Test that validation works across multiple nodes"""

    def node1(state: State) -> dict:
        return {"items": state.items + ["valid1"]}

    def invalid_node(state: State) -> dict:
        return {"items": state.items + [123]}  # Invalid!

    def node3(state: State) -> dict:
        return {"items": state.items + ["valid3"]}

    graph = StateGraph(State)
    graph.add_node("node1", node1)
    graph.add_node("invalid", invalid_node)
    graph.add_node("node3", node3)
    graph.add_edge(START, "node1")
    graph.add_edge("node1", "invalid")
    graph.add_edge("invalid", "node3")
    graph.add_edge("node3", END)

    app = graph.compile(checkpointer=MemorySaver())

    # Should fail at invalid_node
    with pytest.raises(InvalidUpdateError) as exc_info:
        app.invoke(State(), {"configurable": {"thread_id": "1"}})

    # Error should mention the invalid node
    assert "invalid" in str(exc_info.value).lower()


def test_node_output_validation_pydantic_model_return():
    """Test that returning Pydantic model directly still validates"""

    def return_pydantic(state: State) -> State:
        """Node that returns Pydantic model"""
        state.items.append(None)  # Modify and make invalid!
        return state

    graph = StateGraph(State)
    graph.add_node("bad_pydantic", return_pydantic)
    graph.add_edge(START, "bad_pydantic")
    graph.add_edge("bad_pydantic", END)

    app = graph.compile(checkpointer=MemorySaver())

    # Should catch the validation error
    with pytest.raises(Exception):  # Could be ValidationError or InvalidUpdateError
        app.invoke(State(items=["hello"]), {"configurable": {"thread_id": "1"}})


def test_node_output_validation_with_reducer_single_object():
    """Test validation with reducer that accepts single object and appends to list.

    This is a common pattern like `add_messages` where:
    - State has `Annotated[list[Item], add_items]`
    - Node returns a single `Item` (not a list)
    - Reducer appends it to the list

    The validation should catch when the single object is invalid.
    """
    from typing import Annotated, Union

    class Message(BaseModel):
        content: str
        role: str

    def add_messages(
        left: List[Message], right: Union[Message, List[Message]]
    ) -> List[Message]:
        """Reducer that accepts single Message or list of Messages."""
        if isinstance(right, list):
            return left + right
        return left + [right]

    class ChatState(BaseModel):
        messages: Annotated[List[Message], add_messages] = []

    def valid_node(state: ChatState) -> dict:
        """Node that returns a single valid Message."""
        return {"messages": Message(content="hello", role="user")}

    def invalid_node(state: ChatState) -> dict:
        """Node that returns invalid data for Message field."""
        # Returning None where Message is expected
        return {"messages": None}

    # Test valid case
    graph = StateGraph(ChatState)
    graph.add_node("valid", valid_node)
    graph.add_edge(START, "valid")
    graph.add_edge("valid", END)
    app = graph.compile(checkpointer=MemorySaver())

    result = app.invoke(ChatState(), {"configurable": {"thread_id": "1"}})
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "hello"

    # Test invalid case - None instead of Message
    graph2 = StateGraph(ChatState)
    graph2.add_node("invalid", invalid_node)
    graph2.add_edge(START, "invalid")
    graph2.add_edge("invalid", END)
    app2 = graph2.compile(checkpointer=MemorySaver())

    with pytest.raises(InvalidUpdateError) as exc_info:
        app2.invoke(ChatState(), {"configurable": {"thread_id": "1"}})

    assert "invalid" in str(exc_info.value).lower()


def test_node_output_validation_with_reducer_wrong_type():
    """Test validation when reducer receives completely wrong type.

    Node returns a string where list[Message] is expected after reduction.
    """
    from typing import Annotated, Union

    class Item(BaseModel):
        value: int

    def add_items(left: List[Item], right: Union[Item, List[Item]]) -> List[Item]:
        if isinstance(right, list):
            return left + right
        return left + [right]

    class ItemState(BaseModel):
        items: Annotated[List[Item], add_items] = []

    def wrong_type_node(state: ItemState) -> dict:
        """Node that returns wrong type - string instead of Item."""
        return {"items": "not an item"}

    graph = StateGraph(ItemState)
    graph.add_node("wrong", wrong_type_node)
    graph.add_edge(START, "wrong")
    graph.add_edge("wrong", END)
    app = graph.compile(checkpointer=MemorySaver())

    with pytest.raises(InvalidUpdateError) as exc_info:
        app.invoke(ItemState(), {"configurable": {"thread_id": "1"}})

    assert "wrong" in str(exc_info.value).lower()


def test_pregel_node_mapper_output_single_channel():
    """Test that PregelNode.mapper_output is properly set for single channel state.

    Single channel states (e.g., List[str], str) have mapper=None by design,
    so mapper_output should also be None (no validation for single channel).
    This is consistent with the existing behavior where single channels skip
    validation.
    """
    # Single channel state with List[str]
    graph = StateGraph(List[str])
    graph.add_node("node1", lambda x: x)
    graph.add_edge(START, "node1")
    graph.add_edge("node1", END)

    # Access the compiled graph's nodes
    compiled = graph.compile()
    pregel_nodes = compiled.nodes

    # For single channel state, mapper_output should be None (same as mapper)
    node1 = pregel_nodes["node1"]
    assert node1.mapper is None
    assert node1.mapper_output is None


def test_pregel_node_mapper_output_multi_channel():
    """Test that PregelNode.mapper_output is properly set for multi-channel state.

    Multi-channel states (e.g., Pydantic models) have mapper set to validate input,
    so mapper_output should be the same to validate output.
    """

    class MultiState(BaseModel):
        items: List[str] = []
        count: int = 0

    graph = StateGraph(MultiState)
    graph.add_node("node1", lambda x: {"items": x.items})
    graph.add_edge(START, "node1")
    graph.add_edge("node1", END)

    compiled = graph.compile()
    pregel_nodes = compiled.nodes

    # For multi-channel state, mapper_output should be set (same as mapper)
    node1 = pregel_nodes["node1"]
    assert node1.mapper is not None
    assert node1.mapper_output is not None
    assert node1.mapper_output == node1.mapper
