"""Tests for subgraph persistence behavior (async).

Covers three checkpointer settings for subgraph state:
- checkpointer=False: no persistence, even when parent has a checkpointer
- checkpointer=None (default): "tool scope" — inherits parent checkpointer for
  interrupt support, but state resets each invocation
- checkpointer=True: "session scope" — state accumulates across invocations
"""

import sys

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from langgraph.graph import START, StateGraph
from langgraph.types import Command, Interrupt
from tests.any_str import AnyStr
from tests.subgraph_persistence_helpers import (
    ParentState,
    _contents,
    _echo_graph,
    _interrupt_echo_graph,
    _wrap_session_scope,
    _wrap_session_scope_interrupt,
)

pytestmark = pytest.mark.anyio

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)

# -- checkpointer=None (tool scope) --


@NEEDS_CONTEXTVARS
async def test_tool_scope_interrupt_resume_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=None supports interrupt/resume (async)."""
    inner = _interrupt_echo_graph()

    async def call_inner(state: ParentState):
        resp = await inner.ainvoke({"messages": [HumanMessage(content="apples")]})
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    result = await parent.ainvoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }

    result = await parent.ainvoke(Command(resume=True), config)
    assert result == {"result": "Done"}


@NEEDS_CONTEXTVARS
async def test_tool_scope_state_resets_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=None resets subgraph state on each invocation (async)."""
    inner = _echo_graph("Processing")
    subgraph_messages: list[list[str]] = []
    call_count = 0

    async def call_inner(state: ParentState):
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        resp = await inner.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    await parent.ainvoke({"result": ""}, config)
    await parent.ainvoke({"result": ""}, config)

    # Both invocations produce fresh history — no memory of prior call
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processing: tell me about apples",
    ]
    assert subgraph_messages[1] == [
        "tell me about bananas",
        "Processing: tell me about bananas",
    ]


@NEEDS_CONTEXTVARS
async def test_tool_scope_state_resets_with_interrupt_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=None resets subgraph state on each invocation (async, with interrupt)."""
    inner = _interrupt_echo_graph()
    subgraph_messages: list[list[str]] = []
    call_count = 0

    async def call_inner(state: ParentState):
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        resp = await inner.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    await parent.ainvoke({"result": ""}, config)
    await parent.ainvoke(Command(resume=True), config)
    await parent.ainvoke({"result": ""}, config)
    await parent.ainvoke(Command(resume=True), config)

    # Both invocations produce fresh history — no memory of prior call
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
    ]
    assert subgraph_messages[1] == [
        "tell me about bananas",
        "Processing: tell me about bananas",
        "Done",
    ]


# -- checkpointer=False --


@NEEDS_CONTEXTVARS
async def test_checkpointer_false_no_persistence_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=False prevents persistence even when parent has a checkpointer (async)."""
    inner = _echo_graph("Processed", checkpointer=False)
    subgraph_messages: list[list[str]] = []
    call_count = 0

    async def call_inner(state: ParentState):
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        resp = await inner.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    await parent.ainvoke({"result": ""}, config)
    await parent.ainvoke({"result": ""}, config)

    # Both start fresh — no history from first call
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processed: tell me about apples",
    ]
    assert subgraph_messages[1] == [
        "tell me about bananas",
        "Processed: tell me about bananas",
    ]


# -- checkpointer=True (session scope) --


@NEEDS_CONTEXTVARS
async def test_session_scope_state_accumulates_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=True retains subgraph state across invocations (async)."""
    wrapper = _wrap_session_scope(_echo_graph("Processing"), "agent")
    subgraph_messages: list[list[str]] = []
    topics = ["apples", "bananas"]

    async def call_inner(state: ParentState):
        topic = topics[len(subgraph_messages)]
        resp = await wrapper.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    await parent.ainvoke({"result": ""}, config)
    await parent.ainvoke({"result": ""}, config)

    # First call: fresh history
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processing: tell me about apples",
    ]
    # Second call: retains messages from first call
    assert subgraph_messages[1] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "tell me about bananas",
        "Processing: tell me about bananas",
    ]


@NEEDS_CONTEXTVARS
async def test_session_scope_state_accumulates_with_interrupt_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=True retains subgraph state across invocations (async, with interrupt)."""
    wrapper = _wrap_session_scope_interrupt("agent")
    subgraph_messages: list[list[str]] = []
    topics = ["apples", "bananas"]

    async def call_inner(state: ParentState):
        topic = topics[len(subgraph_messages)]
        resp = await wrapper.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    await parent.ainvoke({"result": ""}, config)
    await parent.ainvoke(Command(resume=True), config)
    await parent.ainvoke({"result": ""}, config)
    await parent.ainvoke(Command(resume=True), config)

    # First call: fresh history
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
    ]
    # Second call: retains messages from first call
    assert subgraph_messages[1] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
        "tell me about bananas",
        "Processing: tell me about bananas",
        "Done",
    ]


@NEEDS_CONTEXTVARS
async def test_session_scope_interrupt_resume_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=True supports interrupt/resume and accumulates state (async)."""
    wrapper = _wrap_session_scope_interrupt("agent")
    subgraph_messages: list[list[str]] = []
    topics = ["apples", "bananas"]

    async def call_inner(state: ParentState):
        topic = topics[len(subgraph_messages)]
        resp = await wrapper.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    # First invocation: hits interrupt
    result = await parent.ainvoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }

    # Resume: completes first call
    result = await parent.ainvoke(Command(resume=True), config)
    assert result == {"result": "Done"}
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
    ]

    # Second invocation: hits interrupt, state accumulated from first call
    result = await parent.ainvoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }

    # Resume: completes second call with accumulated state
    result = await parent.ainvoke(Command(resume=True), config)
    assert result == {"result": "Done"}
    assert subgraph_messages[1] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
        "tell me about bananas",
        "Processing: tell me about bananas",
        "Done",
    ]


@NEEDS_CONTEXTVARS
async def test_session_scope_namespace_isolation_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Different checkpointer=True subgraphs maintain independent state
    via unique wrapper node names (async)."""
    fruit = _wrap_session_scope(_echo_graph("Fruit"), "fruit_agent")
    veggie = _wrap_session_scope(_echo_graph("Veggie"), "veggie_agent")
    fruit_msgs: list[list[str]] = []
    veggie_msgs: list[list[str]] = []
    call_count = 0

    async def call_both(state: ParentState):
        nonlocal call_count
        call_count += 1
        suffix = "round 1" if call_count == 1 else "round 2"
        f = await fruit.ainvoke(
            {"messages": [HumanMessage(content=f"cherries {suffix}")]}
        )
        v = await veggie.ainvoke(
            {"messages": [HumanMessage(content=f"broccoli {suffix}")]}
        )
        fruit_msgs.append(_contents(f["messages"]))
        veggie_msgs.append(_contents(v["messages"]))
        return {"result": f["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_both", call_both)
        .add_edge(START, "call_both")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    await parent.ainvoke({"result": ""}, config)
    await parent.ainvoke({"result": ""}, config)

    # First call: each agent sees only its own history
    assert fruit_msgs[0] == ["cherries round 1", "Fruit: cherries round 1"]
    assert veggie_msgs[0] == ["broccoli round 1", "Veggie: broccoli round 1"]

    # Second call: each accumulated independently — no cross-contamination
    assert fruit_msgs[1] == [
        "cherries round 1",
        "Fruit: cherries round 1",
        "cherries round 2",
        "Fruit: cherries round 2",
    ]
    assert veggie_msgs[1] == [
        "broccoli round 1",
        "Veggie: broccoli round 1",
        "broccoli round 2",
        "Veggie: broccoli round 2",
    ]
