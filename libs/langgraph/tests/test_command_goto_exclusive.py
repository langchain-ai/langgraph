import asyncio
from typing import TypedDict

import pytest

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


class State(TypedDict):
    foo: str


async def node_y(state: State) -> dict:
    await asyncio.sleep(0.1)
    return {"foo": "Y done"}


async def node_x(state: State) -> dict:
    await asyncio.sleep(0.1)
    return {"foo": "X done"}


@pytest.mark.asyncio
async def test_command_goto_executes_exclusively():
    """Test that Command(goto=...) executes ONLY the specified node, not racing with normal flow."""
    builder = StateGraph(State)
    builder.add_node("Y", node_y)
    builder.add_node("X", node_x)
    builder.add_edge(START, "Y")
    builder.add_edge("Y", END)
    builder.add_edge("X", END)
    graph = builder.compile()
    
    # Test that goto X executes ONLY X, never Y
    results = []
    async for chunk in graph.astream(Command(goto="X"), stream_mode="updates"):
        results.append(chunk)
    
    # Should only see X executing, never Y
    assert len(results) == 1
    assert "X" in results[0]
    assert results[0]["X"]["foo"] == "X done"
    assert "Y" not in results[0]  # Y should not execute at all


@pytest.mark.asyncio
async def test_normal_execution_still_works():
    """Test that normal execution without goto still works as expected."""
    builder = StateGraph(State)
    builder.add_node("Y", node_y)
    builder.add_node("X", node_x)
    builder.add_edge(START, "Y")
    builder.add_edge("Y", END)
    builder.add_edge("X", END)
    graph = builder.compile()
    
    # Test normal execution (should only execute Y due to START -> Y edge)
    results = []
    async for chunk in graph.astream({"foo": "start"}, stream_mode="updates"):
        results.append(chunk)
    
    # Should only see Y executing in normal flow
    assert len(results) == 1
    assert "Y" in results[0]
    assert results[0]["Y"]["foo"] == "Y done"
    assert "X" not in results[0]  # X should not execute without explicit trigger
