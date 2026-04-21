"""Test InjectedState with NotRequired state fields.

This tests the fix for https://github.com/langchain-ai/langchain/issues/35585

When using InjectedState(<field>) on a tool parameter, and the referenced field is
declared as NotRequired in the custom state schema, the ToolNode should gracefully
handle missing fields without raising KeyError so the tool's default can apply.
"""

import sys
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from langgraph.prebuilt import InjectedState, ToolNode, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from .model import FakeToolCallingModel


class CustomAgentStateWithNotRequired(AgentState):
    """Custom state with a NotRequired field (TypedDict style)."""

    city: NotRequired[str]


class CustomAgentStatePydanticWithDefault(BaseModel):
    """Custom state with Optional field and default (Pydantic style)."""

    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: int = Field(default=10)
    city: str | None = Field(default=None)


@tool
def get_weather(city: Annotated[str | None, InjectedState("city")] = None) -> str:
    """Get weather for a given city."""
    if city is None:
        return "No city provided"
    return f"It's always sunny in {city}!"


@tool
def get_weather_with_default(
    city: Annotated[str, InjectedState("city")] = "Boston",
) -> str:
    """Get weather for a given city, defaulting when state omits the field."""
    return f"It's always sunny in {city}!"


def _create_mock_runtime(
    state: dict | None = None,
    store=None,
):
    """Create a mock Runtime for testing ToolNode directly."""
    from unittest.mock import Mock

    from langgraph.runtime import Runtime

    mock_runtime = Mock(spec=Runtime)
    mock_runtime.context = {}
    return mock_runtime


def _create_config_with_runtime(store=None, state=None):
    """Create a RunnableConfig with mocked runtime for direct ToolNode testing."""
    from langgraph.prebuilt.tool_node import ToolRuntime

    tool_runtime = ToolRuntime(
        state=state or {},
        config={},
        context={},
        store=store,
        stream_writer=None,
        tool_call_id="test_id",
    )
    return {
        "configurable": {
            "__pregel_runtime": _create_mock_runtime(),
            "__tool_runtime__": tool_runtime,
        }
    }


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="InjectedState field extraction from Optional[Annotated[...]] not supported on Python <3.11",
)
def test_injected_state_not_required_field_missing_injects_none():
    """Test that missing optional InjectedState leaves the tool default in place.

    This verifies the fix for https://github.com/langchain-ai/langchain/issues/35585
    """
    tool_node = ToolNode([get_weather])

    tool_call = {
        "name": "get_weather",
        "args": {},
        "id": "call_1",
        "type": "tool_call",
    }
    ai_msg = AIMessage("Let me check the weather", tool_calls=[tool_call])

    # State WITHOUT the "city" field - should inject None instead of raising KeyError
    state_without_city: CustomAgentStateWithNotRequired = {
        "messages": [HumanMessage("What's the weather?"), ai_msg],
    }

    result = tool_node.invoke(
        state_without_city,
        config=_create_config_with_runtime(state=state_without_city),
    )

    assert len(result["messages"]) == 1
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert "No city provided" in tool_msg.content


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="InjectedState field extraction from Optional[Annotated[...]] not supported on Python <3.11",
)
def test_injected_state_not_required_field_missing_preserves_tool_default():
    """Test that missing optional InjectedState preserves a non-None tool default."""
    tool_node = ToolNode([get_weather_with_default])

    tool_call = {
        "name": "get_weather_with_default",
        "args": {},
        "id": "call_1",
        "type": "tool_call",
    }
    ai_msg = AIMessage("Let me check the weather", tool_calls=[tool_call])

    state_without_city: CustomAgentStateWithNotRequired = {
        "messages": [HumanMessage("What's the weather?"), ai_msg],
    }

    result = tool_node.invoke(
        state_without_city,
        config=_create_config_with_runtime(state=state_without_city),
    )

    assert len(result["messages"]) == 1
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert "Boston" in tool_msg.content


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="InjectedState field extraction from Optional[Annotated[...]] not supported on Python <3.11",
)
def test_injected_state_not_required_field_present_works():
    """Test that InjectedState with NotRequired field works when field IS present."""
    tool_node = ToolNode([get_weather])

    tool_call = {
        "name": "get_weather",
        "args": {},
        "id": "call_1",
        "type": "tool_call",
    }
    ai_msg = AIMessage("Let me check the weather", tool_calls=[tool_call])

    # State WITH the "city" field - this should work
    state_with_city: CustomAgentStateWithNotRequired = {
        "messages": [HumanMessage("What's the weather?"), ai_msg],
        "city": "San Francisco",
    }

    result = tool_node.invoke(
        state_with_city,
        config=_create_config_with_runtime(state=state_with_city),
    )

    assert len(result["messages"]) == 1
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert "San Francisco" in tool_msg.content


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="InjectedState field extraction from Optional[Annotated[...]] not supported on Python <3.11",
)
def test_create_react_agent_injected_state_not_required_field_missing():
    """Test create_react_agent with InjectedState using NotRequired field that is missing.

    This verifies the fix for https://github.com/langchain-ai/langchain/issues/35585
    """
    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "get_weather", "args": {}, "id": "call_1"}],
            [],  # No more tool calls, agent should stop
        ]
    )

    agent = create_react_agent(
        model,
        tools=[get_weather],
        state_schema=CustomAgentStateWithNotRequired,
    )

    # Invoke WITHOUT the city field - should work, injecting None
    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather?")]},
    )

    # Check that the tool was called successfully with None injected
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "No city provided" in tool_messages[0].content


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="InjectedState field extraction from Optional[Annotated[...]] not supported on Python <3.11",
)
def test_create_react_agent_injected_state_not_required_field_present():
    """Test create_react_agent with InjectedState using NotRequired field that IS present."""
    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "get_weather", "args": {}, "id": "call_1"}],
            [],  # No more tool calls, agent should stop
        ]
    )

    agent = create_react_agent(
        model,
        tools=[get_weather],
        state_schema=CustomAgentStateWithNotRequired,
    )

    # Invoke WITH the city field
    result = agent.invoke(
        {
            "messages": [HumanMessage("What's the weather?")],
            "city": "San Francisco",
        },
    )

    # Check that the tool was called successfully
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "San Francisco" in tool_messages[0].content


@tool
def get_weather_optional(city: Annotated[str | None, InjectedState("city")]) -> str:
    """Get weather for a given city (accepts None)."""
    if city is None:
        return "Please provide a city!"
    return f"It's always sunny in {city}!"


def test_pydantic_state_with_default_field_missing_works():
    """Test that Pydantic state with Optional field and default=None works when field is missing.

    This is the workaround suggested in the issue comments - using Pydantic BaseModel
    with `city: Optional[str] = Field(default=None)` instead of TypedDict with NotRequired.
    """
    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "get_weather_optional", "args": {}, "id": "call_1"}],
            [],  # No more tool calls, agent should stop
        ]
    )

    agent = create_react_agent(
        model,
        tools=[get_weather_optional],
        state_schema=CustomAgentStatePydanticWithDefault,
    )

    # Invoke WITHOUT the city field - should work because Pydantic provides default
    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather?")]},
    )

    # Check that the tool was called successfully with None
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Please provide a city!" in tool_messages[0].content


def test_pydantic_state_with_default_field_present_works():
    """Test that Pydantic state with Optional field works when field IS present."""
    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "get_weather_optional", "args": {}, "id": "call_1"}],
            [],  # No more tool calls, agent should stop
        ]
    )

    agent = create_react_agent(
        model,
        tools=[get_weather_optional],
        state_schema=CustomAgentStatePydanticWithDefault,
    )

    # Invoke WITH the city field
    result = agent.invoke(
        {
            "messages": [HumanMessage("What's the weather?")],
            "city": "San Francisco",
        },
    )

    # Check that the tool was called successfully
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "San Francisco" in tool_messages[0].content
