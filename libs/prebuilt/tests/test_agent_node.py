"""Tests for create_agent_node and get_prompt_runnable composable primitives."""

from typing import Annotated

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import tool as dec_tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from typing_extensions import TypedDict

from langgraph.prebuilt import (
    ToolNode,
    create_agent_node,
    get_prompt_runnable,
    tools_condition,
)
from langgraph.prebuilt.chat_agent_executor import _get_prompt_runnable
from tests.model import FakeToolCallingModel

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# State schemas
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps


# ---------------------------------------------------------------------------
# get_prompt_runnable
# ---------------------------------------------------------------------------


def test_get_prompt_runnable_none():
    pr = get_prompt_runnable(None)
    result = pr.invoke({"messages": [HumanMessage("hi")]})
    assert result == [HumanMessage("hi")]


def test_get_prompt_runnable_string():
    pr = get_prompt_runnable("You are helpful.")
    result = pr.invoke({"messages": [HumanMessage("hi")]})
    assert result[0] == SystemMessage(content="You are helpful.")
    assert result[1] == HumanMessage("hi")


def test_get_prompt_runnable_system_message():
    sys_msg = SystemMessage(content="Be concise.")
    pr = get_prompt_runnable(sys_msg)
    result = pr.invoke({"messages": [HumanMessage("hi")]})
    assert result[0] == sys_msg
    assert result[1] == HumanMessage("hi")


def test_get_prompt_runnable_callable():
    def my_prompt(state):
        return [SystemMessage(content="dynamic")] + state["messages"]

    pr = get_prompt_runnable(my_prompt)
    result = pr.invoke({"messages": [HumanMessage("hi")]})
    assert result[0] == SystemMessage(content="dynamic")


def test_get_prompt_runnable_private_alias():
    """_get_prompt_runnable must still work for backwards compatibility."""
    pr = _get_prompt_runnable("sys")
    result = pr.invoke({"messages": [HumanMessage("x")]})
    assert result[0] == SystemMessage(content="sys")


# ---------------------------------------------------------------------------
# create_agent_node – basic invocation via a full graph
# ---------------------------------------------------------------------------


def test_create_agent_node_no_tools():
    model = FakeToolCallingModel()
    agent_node = create_agent_node(model, [])

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    graph = workflow.compile()

    result = graph.invoke({"messages": [HumanMessage("hello")]})
    assert result["messages"][-1].content == "hello"


def test_create_agent_node_with_string_prompt():
    model = FakeToolCallingModel()
    agent_node = create_agent_node(model, [], prompt="You are a bot.")

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    graph = workflow.compile()

    result = graph.invoke({"messages": [HumanMessage("hi")]})
    # FakeToolCallingModel echoes the joined message contents
    assert "You are a bot." in result["messages"][-1].content
    assert "hi" in result["messages"][-1].content


def test_create_agent_node_with_system_message_prompt():
    model = FakeToolCallingModel()
    sys_msg = SystemMessage(content="Speak like a pirate.")
    agent_node = create_agent_node(model, [], prompt=sys_msg)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    graph = workflow.compile()

    result = graph.invoke({"messages": [HumanMessage("ahoy")]})
    assert "Speak like a pirate." in result["messages"][-1].content


def test_create_agent_node_name_set_on_response():
    model = FakeToolCallingModel()
    agent_node = create_agent_node(model, [], name="my-agent")

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    graph = workflow.compile()

    result = graph.invoke({"messages": [HumanMessage("hi")]})
    assert result["messages"][-1].name == "my-agent"


# ---------------------------------------------------------------------------
# create_agent_node – ReAct loop (tools_condition + ToolNode)
# ---------------------------------------------------------------------------


def test_create_agent_node_react_loop():
    @dec_tool
    def get_weather(city: str) -> str:
        """Return weather for a city."""
        return f"Sunny in {city}"

    tool_calls_sequence = [
        [ToolCall(name="get_weather", args={"city": "Paris"}, id="tc1")],
        [],  # second LLM call ends the loop
    ]
    model = FakeToolCallingModel(tool_calls=tool_calls_sequence)
    tools = [get_weather]
    agent_node = create_agent_node(model, tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    graph = workflow.compile()

    result = graph.invoke({"messages": [HumanMessage("What is the weather in Paris?")]})
    messages = result["messages"]
    # Should have: HumanMessage, AIMessage(tool_call), ToolMessage, AIMessage(final)
    assert any(isinstance(m, ToolMessage) for m in messages)
    assert isinstance(messages[-1], AIMessage)
    assert not messages[-1].tool_calls


# ---------------------------------------------------------------------------
# create_agent_node – custom node insertion
# ---------------------------------------------------------------------------


def test_create_agent_node_custom_pre_node():
    """A custom node upstream of the agent node must be composable."""
    model = FakeToolCallingModel()
    agent_node = create_agent_node(model, [])

    audit_calls = []

    def audit_node(state: AgentState) -> dict:
        audit_calls.append(len(state["messages"]))
        return {}

    workflow = StateGraph(AgentState)
    workflow.add_node("audit", audit_node)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("audit")
    workflow.add_edge("audit", "agent")
    workflow.add_edge("agent", END)
    graph = workflow.compile()

    graph.invoke({"messages": [HumanMessage("hi")]})
    assert len(audit_calls) == 1


# ---------------------------------------------------------------------------
# create_agent_node – llm_input_messages pre-model-hook protocol
# ---------------------------------------------------------------------------


class StateWithLLMInput(TypedDict):
    messages: Annotated[list, add_messages]
    llm_input_messages: list
    remaining_steps: RemainingSteps


def test_create_agent_node_llm_input_messages():
    """When llm_input_messages is set in state, it overrides messages for LLM input."""
    model = FakeToolCallingModel()
    agent_node = create_agent_node(model, [])

    def pre_hook(state: StateWithLLMInput) -> dict:
        # Provide a trimmed view to the LLM without modifying stored messages
        return {"llm_input_messages": [SystemMessage(content="trimmed"), state["messages"][-1]]}

    workflow = StateGraph(StateWithLLMInput)
    workflow.add_node("pre_hook", pre_hook)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("pre_hook")
    workflow.add_edge("pre_hook", "agent")
    workflow.add_edge("agent", END)
    graph = workflow.compile()

    result = graph.invoke({"messages": [HumanMessage("hello")]})
    # LLM received "trimmed-hello"; stored messages unchanged except for AI response
    ai_response = result["messages"][-1]
    assert "trimmed" in ai_response.content
    assert "hello" in ai_response.content


# ---------------------------------------------------------------------------
# create_agent_node – async
# ---------------------------------------------------------------------------


async def test_create_agent_node_async():
    model = FakeToolCallingModel()
    agent_node = create_agent_node(model, [])

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    graph = workflow.compile()

    result = await graph.ainvoke({"messages": [HumanMessage("async hello")]})
    assert result["messages"][-1].content == "async hello"