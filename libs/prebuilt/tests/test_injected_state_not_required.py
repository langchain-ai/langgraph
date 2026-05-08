"""Test InjectedState with NotRequired state fields.

This tests the fix for https://github.com/langchain-ai/langchain/issues/35585

When using InjectedState(<field>) on a tool parameter, and the referenced field is
declared as NotRequired in the custom state schema, the ToolNode should gracefully
handle missing fields by injecting None instead of raising KeyError.
"""

import logging
import re
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approved tool allow list
# ---------------------------------------------------------------------------
APPROVED_TOOL_NAMES = {"get_weather", "get_weather_optional"}

# ---------------------------------------------------------------------------
# Approved LLM registry (fake/stub models accepted for testing)
# ---------------------------------------------------------------------------
APPROVED_LLM_CLASSES = {"FakeToolCallingModel"}

# ---------------------------------------------------------------------------
# Input sanitization helpers
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERN = re.compile(
    r"(base64\s*:|"
    r"(?:^|[\s;|&`])\s*(?:rm|curl|wget|bash|sh|python|exec|eval|os\.system|subprocess)\b|"
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|"
    r"(?:[A-Za-z0-9+/]{20,}={0,2}))",
    re.IGNORECASE | re.MULTILINE,
)


def _sanitize_text(text: str) -> str:
    """Raise ValueError if text contains suspicious content, else return it."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")
    if _DANGEROUS_PATTERN.search(text):
        raise ValueError(f"Input failed sanitization check: {text!r}")
    return text


def _sanitize_messages(messages: list) -> list:
    """Validate and sanitize a list of message objects."""
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            _sanitize_text(msg.content)
    return messages


def _sanitize_state(state: dict) -> dict:
    """Sanitize all string values in a state dict."""
    for key, value in state.items():
        if isinstance(value, str):
            _sanitize_text(value)
        elif isinstance(value, list):
            _sanitize_messages(value)
    return state


def _validate_tool_allow_list(tools: list) -> list:
    """Ensure every tool in the list is on the approved allow list."""
    for t in tools:
        tool_name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if tool_name not in APPROVED_TOOL_NAMES:
            logger.warning("Tool '%s' is not on the approved allow list.", tool_name)
            raise ValueError(
                f"Tool '{tool_name}' is not permitted. "
                f"Approved tools: {APPROVED_TOOL_NAMES}"
            )
    return tools


def _validate_llm(model) -> None:
    """Ensure the model class is on the approved LLM registry."""
    class_name = type(model).__name__
    if class_name not in APPROVED_LLM_CLASSES:
        raise ValueError(
            f"LLM class '{class_name}' is not in the approved registry. "
            f"Approved classes: {APPROVED_LLM_CLASSES}"
        )


# ---------------------------------------------------------------------------
# State schemas
# ---------------------------------------------------------------------------


class CustomAgentStateWithNotRequired(AgentState):
    """Custom state with a NotRequired field (TypedDict style)."""

    city: NotRequired[str]


class CustomAgentStatePydanticWithDefault(BaseModel):
    """Custom state with Optional field and default (Pydantic style)."""

    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: int = Field(default=10)
    city: str | None = Field(default=None)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_weather(city: Annotated[str | None, InjectedState("city")] = None) -> str:
    """Get weather for a given city."""
    if city is None:
        return "No city provided"
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
        tools=[],
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
    """Test that InjectedState with NotRequired field injects None when field is missing.

    This verifies the fix for https://github.com/langchain-ai/langchain/issues/35585
    """
    _validate_tool_allow_list([get_weather])
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

    _sanitize_state(state_without_city)
    logger.info(
        "tool_node.invoke called with state keys: %s", list(state_without_city.keys())
    )
    result = tool_node.invoke(
        state_without_city,
        config=_create_config_with_runtime(state=state_without_city),
    )
    logger.info("tool_node.invoke result message count: %d", len(result["messages"]))

    assert len(result["messages"]) == 1
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert "No city provided" in tool_msg.content


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="InjectedState field extraction from Optional[Annotated[...]] not supported on Python <3.11",
)
def test_injected_state_not_required_field_present_works():
    """Test that InjectedState with NotRequired field works when field IS present."""
    _validate_tool_allow_list([get_weather])
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

    _sanitize_state(state_with_city)
    logger.info(
        "tool_node.invoke called with state keys: %s", list(state_with_city.keys())
    )
    result = tool_node.invoke(
        state_with_city,
        config=_create_config_with_runtime(state=state_with_city),
    )
    logger.info("tool_node.invoke result message count: %d", len(result["messages"]))

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
    _validate_llm(model)
    _validate_tool_allow_list([get_weather])

    agent = create_react_agent(
        model,
        tools=[get_weather],
        state_schema=CustomAgentStateWithNotRequired,
    )

    input_state = {"messages": [HumanMessage("What's the weather?")]}
    _sanitize_state(input_state)

    # Invoke WITHOUT the city field - should work, injecting None
    logger.info("agent.invoke called with input keys: %s", list(input_state.keys()))
    result = agent.invoke(input_state)
    logger.info(
        "agent.invoke completed; total messages: %d", len(result.get("messages", []))
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
    _validate_llm(model)
    _validate_tool_allow_list([get_weather])

    agent = create_react_agent(
        model,
        tools=[get_weather],
        state_schema=CustomAgentStateWithNotRequired,
    )

    input_state = {
        "messages": [HumanMessage("What's the weather?")],
        "city": "San Francisco",
    }
    _sanitize_state(input_state)

    # Invoke WITH the city field
    logger.info("agent.invoke called with input keys: %s", list(input_state.keys()))
    result = agent.invoke(input_state)
    logger.info(
        "agent.invoke completed; total messages: %d", len(result.get("messages", []))
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
    _validate_llm(model)
    _validate_tool_allow_list([get_weather_optional])

    agent = create_react_agent(
        model,
        tools=[get_weather_optional],
        state_schema=CustomAgentStatePydanticWithDefault,
    )

    input_state = {"messages": [HumanMessage("What's the weather?")]}
    _sanitize_state(input_state)

    # Invoke WITHOUT the city field - should work because Pydantic provides default
    logger.info("agent.invoke called with input keys: %s", list(input_state.keys()))
    result = agent.invoke(input_state)
    logger.info(
        "agent.invoke completed; total messages: %d", len(result.get("messages", []))
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
    _validate_llm(model)
    _validate_tool_allow_list([get_weather_optional])

    agent = create_react_agent(
        model,
        tools=[get_weather_optional],
        state_schema=CustomAgentStatePydanticWithDefault,
    )

    input_state = {
        "messages": [HumanMessage("What's the weather?")],
        "city": "San Francisco",
    }
    _sanitize_state(input_state)

    # Invoke WITH the city field
    logger.info("agent.invoke called with input keys: %s", list(input_state.keys()))
    result = agent.invoke(input_state)
    logger.info(
        "agent.invoke completed; total messages: %d", len(result.get("messages", []))
    )

    # Check that the tool was called successfully
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "San Francisco" in tool_messages[0].content