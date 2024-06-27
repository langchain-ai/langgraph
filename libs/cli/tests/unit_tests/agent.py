import asyncio
import os
from typing import Annotated, Sequence, TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph, add_messages

# check that env var is present
os.environ["SOME_ENV_VAR"]


class AgentState(TypedDict):
    some_bytes: bytes
    some_byte_array: bytearray
    dict_with_bytes: dict[str, bytes]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sleep: int


async def call_model(state, config):
    if sleep := state.get("sleep"):
        await asyncio.sleep(sleep)

    messages = state["messages"]

    if len(messages) > 1:
        assert state["some_bytes"] == b"some_bytes"
        assert state["some_byte_array"] == bytearray(b"some_byte_array")
        assert state["dict_with_bytes"] == {"more_bytes": b"more_bytes"}

    # hacky way to reset model to the "first" response
    if isinstance(messages[-1], HumanMessage):
        model.i = 0

    response = await model.ainvoke(messages)
    return {
        "messages": [response],
        "some_bytes": b"some_bytes",
        "some_byte_array": bytearray(b"some_byte_array"),
        "dict_with_bytes": {"more_bytes": b"more_bytes"},
    }


def call_tool(state):
    last_message_content = state["messages"][-1].content
    return {
        "messages": [
            ToolMessage(
                f"tool_call__{last_message_content}", tool_call_id="tool_call_id"
            )
        ]
    }


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content == "end":
        return END
    else:
        return "tool"


# NOTE: the model cycles through responses infinitely here
model = FakeListChatModel(responses=["begin", "end"])
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tool", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tool", "agent")

graph = workflow.compile()
