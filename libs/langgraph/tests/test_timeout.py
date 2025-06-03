"""Tests for timeout functionality in langgraph."""
import asyncio
import pytest
from typing import TypedDict, Any
from langgraph.graph import StateGraph
from langgraph.errors import ParentCommand
from langgraph.types import Command

class State(TypedDict):
    value: str

@pytest.mark.asyncio
async def test_basic_timeout():
    """Test basic timeout functionality with a slow node."""
    async def slow_node(state: State) -> State:
        await asyncio.sleep(2)
        return {"value": "done"}

    builder = StateGraph(State)
    builder.add_node("slow", slow_node)
    builder.set_entry_point("slow")
    graph = builder.compile()
    graph.step_timeout = 1

    with pytest.raises(asyncio.TimeoutError):
        await graph.ainvoke({"value": "start"})

@pytest.mark.asyncio
async def test_timeout_with_parent_command():
    """Test that parent commands are properly propagated during timeouts."""
    async def parent_command_node(state: State) -> State:
        await asyncio.sleep(0.5)  # Add some delay before raising
        raise ParentCommand(Command(graph=Command.PARENT, goto="test_cmd", update={"key": "value"}))

    builder = StateGraph(State)
    builder.add_node("parent_cmd", parent_command_node)
    builder.set_entry_point("parent_cmd")
    graph = builder.compile()
    graph.step_timeout = 1

    # Should propagate parent command, not timeout
    with pytest.raises(ParentCommand) as exc_info:
        await graph.ainvoke({"value": "start"})
    assert exc_info.value.args[0].goto == "test_cmd"
    assert exc_info.value.args[0].update == {"key": "value"}

@pytest.mark.asyncio
async def test_timeout_with_agent_handoff():
    """Test timeout behavior during agent handoffs."""
    async def first_agent(state: State) -> State:
        await asyncio.sleep(0.5)
        return {"value": "first_done"}

    async def second_agent(state: State) -> State:
        await asyncio.sleep(2)  # This will trigger timeout
        return {"value": "second_done"}

    builder = StateGraph(State)
    builder.add_node("first", first_agent)
    builder.add_node("second", second_agent)
    builder.add_edge("first", "second")
    builder.set_entry_point("first")
    graph = builder.compile()
    graph.step_timeout = 1

    with pytest.raises(asyncio.TimeoutError):
        await graph.ainvoke({"value": "start"})

@pytest.mark.asyncio
async def test_concurrent_timeouts():
    """Test timeout behavior with concurrent node execution."""
    async def slow_node_1(state: State) -> State:
        await asyncio.sleep(2)
        return {"value": "slow1_done"}

    async def slow_node_2(state: State) -> State:
        await asyncio.sleep(2)
        return {"value": "slow2_done"}

    builder = StateGraph(State)
    builder.add_node("slow1", slow_node_1)
    builder.add_node("slow2", slow_node_2)
    builder.set_conditional_entry_point(lambda _: ["slow1", "slow2"])
    graph = builder.compile()
    graph.step_timeout = 1

    with pytest.raises(asyncio.TimeoutError):
        await graph.ainvoke({"value": "start"})

@pytest.mark.asyncio
async def test_sequential_timeouts():
    """Test timeout behavior with sequential node execution."""
    async def fast_node(state: State) -> State:
        await asyncio.sleep(0.5)
        return {"value": "fast_done"}

    async def slow_node(state: State) -> State:
        await asyncio.sleep(2)
        return {"value": "slow_done"}

    builder = StateGraph(State)
    builder.add_node("fast", fast_node)
    builder.add_node("slow", slow_node)
    builder.add_edge("fast", "slow")
    builder.set_entry_point("fast")
    graph = builder.compile()
    graph.step_timeout = 1

    with pytest.raises(asyncio.TimeoutError):
        await graph.ainvoke({"value": "start"})

@pytest.mark.asyncio
async def test_timeout_cancellation():
    """Test that tasks are properly cancelled on timeout."""
    cancel_flag = False

    async def cancellable_node(state: State) -> State:
        try:
            await asyncio.sleep(2)
            return {"value": "done"}
        except asyncio.CancelledError:
            nonlocal cancel_flag
            cancel_flag = True
            raise

    builder = StateGraph(State)
    builder.add_node("cancellable", cancellable_node)
    builder.set_entry_point("cancellable")
    graph = builder.compile()
    graph.step_timeout = 1

    with pytest.raises(asyncio.TimeoutError):
        await graph.ainvoke({"value": "start"})
    
    assert cancel_flag, "Task was not properly cancelled"

@pytest.mark.asyncio
async def test_no_timeout():
    """Test that operations complete successfully when within timeout."""
    async def fast_node(state: State) -> State:
        await asyncio.sleep(0.5)
        return {"value": "done"}

    builder = StateGraph(State)
    builder.add_node("fast", fast_node)
    builder.set_entry_point("fast")
    graph = builder.compile()
    graph.step_timeout = 1

    result = await graph.ainvoke({"value": "start"})
    assert result["value"] == "done" 