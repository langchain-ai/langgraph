"""Tests for interrupt input schema validation feature."""

import uuid
from dataclasses import dataclass

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import Command, Interrupt, _validate_resume_value, interrupt


class PydanticUserInput(BaseModel):
    """Pydantic schema for user input."""

    name: str
    age: int


@dataclass
class DataclassUserInput:
    """Dataclass schema for user input."""

    name: str
    age: int


class GraphState(TypedDict):
    """State for test graph."""

    messages: list[str]
    user_data: dict | None


def test_validate_resume_value_pydantic():
    """Test validation with Pydantic BaseModel."""
    # Valid data
    valid_data = {"name": "John", "age": 25}
    assert _validate_resume_value(valid_data, PydanticUserInput) is True

    # Invalid data - wrong type
    invalid_data = {"name": "John", "age": "not_a_number"}
    assert _validate_resume_value(invalid_data, PydanticUserInput) is False

    # Missing required field
    incomplete_data = {"name": "John"}
    assert _validate_resume_value(incomplete_data, PydanticUserInput) is False


def test_validate_resume_value_dataclass():
    """Test validation with dataclass."""
    # Valid dict data
    valid_data = {"name": "John", "age": 25}
    assert _validate_resume_value(valid_data, DataclassUserInput) is True

    # Valid instance
    valid_instance = DataclassUserInput(name="John", age=25)
    assert _validate_resume_value(valid_instance, DataclassUserInput) is True

    # Missing required field
    incomplete_data = {"name": "John"}
    assert _validate_resume_value(incomplete_data, DataclassUserInput) is False


def test_validate_resume_value_dict():
    """Test validation with dict schema."""
    # Valid dict data
    valid_data = {"name": "John", "age": 25}
    assert _validate_resume_value(valid_data, {"name": str, "age": int}) is True

    # Invalid non-dict data
    invalid_data = "not_a_dict"
    assert _validate_resume_value(invalid_data, {"name": str, "age": int}) is False


def test_validate_resume_value_no_schema():
    """Test that validation passes when no schema is provided."""
    assert _validate_resume_value("any_value", None) is True
    assert _validate_resume_value({"key": "value"}, None) is True
    assert _validate_resume_value(123, None) is True


def test_interrupt_class_with_schema():
    """Test Interrupt class accepts input_schema parameter."""
    interrupt_obj = Interrupt(
        value="Please provide user info", input_schema=PydanticUserInput
    )
    assert interrupt_obj.value == "Please provide user info"
    assert interrupt_obj.input_schema == PydanticUserInput


def test_interrupt_from_ns_with_schema():
    """Test Interrupt.from_ns accepts input_schema parameter."""
    interrupt_obj = Interrupt.from_ns(
        value="test", ns="test_namespace", input_schema=DataclassUserInput
    )
    assert interrupt_obj.input_schema == DataclassUserInput


def test_interrupt_with_valid_schema():
    """Test interrupt with valid resume value matching schema."""

    def node(state: GraphState):
        user_input = interrupt(
            "Please provide user info", input_schema=PydanticUserInput
        )
        return {"user_data": user_input, "messages": ["Received user data"]}

    builder = StateGraph(GraphState)
    builder.add_node("node", node)
    builder.add_edge(START, "node")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # First invocation - should interrupt
    result = graph.invoke({"messages": [], "user_data": None}, config)
    assert "__interrupt__" in result

    # Resume with valid data
    valid_data = {"name": "John", "age": 25}
    result = graph.invoke(Command(resume=valid_data), config)
    assert result["user_data"] == valid_data
    assert "Received user data" in result["messages"]


def test_interrupt_with_invalid_schema():
    """Test interrupt re-raises when resume value doesn't match schema."""

    def node(state: GraphState):
        user_input = interrupt(
            "Please provide user info", input_schema=PydanticUserInput
        )
        return {"user_data": user_input, "messages": ["Received user data"]}

    builder = StateGraph(GraphState)
    builder.add_node("node", node)
    builder.add_edge(START, "node")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # First invocation - should interrupt
    result = graph.invoke({"messages": [], "user_data": None}, config)
    assert "__interrupt__" in result

    # Resume with invalid data (wrong type for age)
    invalid_data = {"name": "John", "age": "not_a_number"}
    result = graph.invoke(Command(resume=invalid_data), config)

    # Should still be interrupted because validation failed
    assert "__interrupt__" in result


def test_interrupt_with_dataclass_schema():
    """Test interrupt with dataclass schema."""

    def node(state: GraphState):
        user_input = interrupt("Provide info", input_schema=DataclassUserInput)
        return {"user_data": user_input, "messages": ["Got data"]}

    builder = StateGraph(GraphState)
    builder.add_node("node", node)
    builder.add_edge(START, "node")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # First invocation
    result = graph.invoke({"messages": [], "user_data": None}, config)
    assert "__interrupt__" in result

    # Resume with valid data
    valid_data = {"name": "Jane", "age": 30}
    result = graph.invoke(Command(resume=valid_data), config)
    assert result["user_data"] == valid_data


def test_interrupt_without_schema_backward_compatibility():
    """Test that interrupt works without schema (backward compatibility)."""

    def node(state: GraphState):
        # No schema provided - should accept any value
        user_input = interrupt("Provide any value")
        return {"user_data": user_input, "messages": ["Got input"]}

    builder = StateGraph(GraphState)
    builder.add_node("node", node)
    builder.add_edge(START, "node")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # First invocation
    result = graph.invoke({"messages": [], "user_data": None}, config)
    assert "__interrupt__" in result

    # Resume with any value - should work
    result = graph.invoke(Command(resume="any string value"), config)
    assert result["user_data"] == "any string value"
