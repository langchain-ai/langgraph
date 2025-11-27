
import asyncio
import pytest
from typing import Annotated, TypedDict, Any
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.config import get_stream_writer
from langgraph.graph.message import add_messages

@tool
async def tool_a(query: str):
    """Tool A."""
    writer = get_stream_writer()
    writer({"type": "tool_a", "status": "start"})
    await asyncio.sleep(0.1)
    writer({"type": "tool_a", "status": "end"})
    return "A"

@tool
async def tool_b(query: str):
    """Tool B."""
    writer = get_stream_writer()
    writer({"type": "tool_b", "status": "start"})
    await asyncio.sleep(0.1)
    writer({"type": "tool_b", "status": "end"})
    return "B"

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@pytest.mark.asyncio
async def test_async_tool_node_streaming():
    tools = [tool_a, tool_b]
    tool_node = ToolNode(tools)

    async def agent(state: State):
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "tool_a", "args": {"query": "a"}, "id": "1"},
            {"name": "tool_b", "args": {"query": "b"}, "id": "2"}
        ])]}

    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", END)

    app = workflow.compile()

    chunks = []
    async for chunk in app.astream(
        {"messages": []},
        stream_mode="custom",
    ):
        chunks.append(chunk)

    # Verify we got chunks from both tools
    tool_a_chunks = [c for c in chunks if c.get("type") == "tool_a"]
    tool_b_chunks = [c for c in chunks if c.get("type") == "tool_b"]

    assert len(tool_a_chunks) == 2
    assert len(tool_b_chunks) == 2
