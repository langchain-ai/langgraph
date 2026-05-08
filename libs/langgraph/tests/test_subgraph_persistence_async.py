"""Tests for subgraph persistence behavior (async).

Covers three checkpointer settings for subgraph state:
- checkpointer=False: no persistence, even when parent has a checkpointer
- checkpointer=None (default): "stateless" — inherits parent checkpointer for
  interrupt support, but state resets each invocation. This is the common case
  when an agent is invoked from inside a tool used by another agent.
- checkpointer=True: "stateful" — state accumulates across invocations on the same thread id
"""

import sys
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command, Interrupt, interrupt
from tests.any_str import AnyStr

pytestmark = pytest.mark.anyio

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


class ParentState(TypedDict):
    result: str


# -- checkpointer=None (stateless) --


@NEEDS_CONTEXTVARS
async def test_stateless_interrupt_resume_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a subgraph compiled with checkpointer=None (the default) can
    still support interrupt/resume when invoked from inside a parent graph that
    has a checkpointer. This is the "stateless" pattern — the subgraph inherits
    the parent's checkpointer just enough to pause and resume, but does not
    retain any state across separate parent invocations. This pattern commonly
    appears when an agent is invoked from inside a tool used by another agent.
    """

    # Build a subgraph that interrupts before echoing.
    # Two nodes: "process" interrupts then echoes, "respond" returns "Done".
    def process(state: MessagesState) -> dict:
        interrupt("continue?")
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    def respond(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="Done")]}

    inner = (
        StateGraph(MessagesState)
        .add_node("process", process)
        .add_node("respond", respond)
        .add_edge(START, "process")
        .add_edge("process", "respond")
        .compile()
    )

    async def call_inner(state: ParentState) -> dict:
        resp = await inner.ainvoke({"messages": [HumanMessage(content="apples")]})
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    # First invoke hits the interrupt
    result = await parent.ainvoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }

    # Resume completes the subgraph
    result = await parent.ainvoke(Command(resume=True), config)
    assert result == {"result": "Done"}


@NEEDS_CONTEXTVARS
async def test_stateless_state_resets_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a subgraph compiled with checkpointer=None (the default) does
    not retain any message history between separate parent invocations. Each time
    the parent graph invokes the subgraph, it starts with a clean slate. This
    confirms the "stateless" behavior: even though the parent has a checkpointer,
    the subgraph state is not persisted across calls.
    """

    # Build a simple echo subgraph: echoes "Processing: <input>"
    def echo(state: MessagesState) -> dict:
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    inner = (
        StateGraph(MessagesState)
        .add_node("echo", echo)
        .add_edge(START, "echo")
        .compile()
    )

    subgraph_messages: list[list[str]] = []
    call_count = 0

    async def call_inner(state: ParentState) -> dict:
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        resp = await inner.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append([m.text for m in resp["messages"]])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    result1 = await parent.ainvoke({"result": ""}, config)
    assert result1 == {"result": "Processing: tell me about apples"}

    result2 = await parent.ainvoke({"result": ""}, config)
    assert result2 == {"result": "Processing: tell me about bananas"}

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
async def test_stateless_state_resets_with_interrupt_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a subgraph compiled with checkpointer=None resets its state
    between parent invocations even when interrupt/resume is used. The subgraph
    is invoked twice from the parent, each time with an interrupt that must be
    resumed. After both invoke+resume cycles, each subgraph run should only
    contain its own messages — no bleed-over from the previous run.
    """

    # Build a subgraph that interrupts before echoing, then responds "Done"
    def process(state: MessagesState) -> dict:
        interrupt("continue?")
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    def respond(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="Done")]}

    inner = (
        StateGraph(MessagesState)
        .add_node("process", process)
        .add_node("respond", respond)
        .add_edge(START, "process")
        .add_edge("process", "respond")
        .compile()
    )

    subgraph_messages: list[list[str]] = []
    call_count = 0

    async def call_inner(state: ParentState) -> dict:
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        resp = await inner.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append([m.text for m in resp["messages"]])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    # First invoke+resume cycle
    result = await parent.ainvoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }
    result = await parent.ainvoke(Command(resume=True), config)
    assert result == {"result": "Done"}

    # Second invoke+resume cycle
    result = await parent.ainvoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }
    result = await parent.ainvoke(Command(resume=True), config)
    assert result == {"result": "Done"}

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
    """Tests that a subgraph compiled with checkpointer=False gets no
    persistence at all, even when the parent graph has a checkpointer. Unlike
    the default (checkpointer=None) which inherits just enough from the parent
    to support interrupt/resume, checkpointer=False explicitly opts out of all
    checkpoint behavior. Each invocation starts completely fresh.
    """

    # Build a simple echo subgraph with checkpointer=False
    def echo(state: MessagesState) -> dict:
        return {
            "messages": [AIMessage(content=f"Processed: {state['messages'][-1].text}")]
        }

    inner = (
        StateGraph(MessagesState)
        .add_node("echo", echo)
        .add_edge(START, "echo")
        .compile(checkpointer=False)
    )

    subgraph_messages: list[list[str]] = []
    call_count = 0

    async def call_inner(state: ParentState) -> dict:
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        resp = await inner.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append([m.text for m in resp["messages"]])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    result1 = await parent.ainvoke({"result": ""}, config)
    assert result1 == {"result": "Processed: tell me about apples"}

    result2 = await parent.ainvoke({"result": ""}, config)
    assert result2 == {"result": "Processed: tell me about bananas"}

    # Both start fresh — no history from first call
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processed: tell me about apples",
    ]
    assert subgraph_messages[1] == [
        "tell me about bananas",
        "Processed: tell me about bananas",
    ]


# -- checkpointer=True (stateful) --


@NEEDS_CONTEXTVARS
async def test_stateful_state_accumulates_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a subgraph compiled with checkpointer=True ("stateful")
    retains its message history across separate parent invocations. To enable
    this, the subgraph is wrapped in an outer graph compiled with
    checkpointer=True — this wrapper gives the inner subgraph its own persistent
    checkpoint namespace. After two parent calls, the second subgraph invocation
    should see messages from both the first and second calls.
    """

    # Build a simple echo subgraph
    def echo(state: MessagesState) -> dict:
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    inner = (
        StateGraph(MessagesState)
        .add_node("echo", echo)
        .add_edge(START, "echo")
        .compile()
    )

    # Wrap the inner subgraph with checkpointer=True to enable stateful.
    # The wrapper graph gives the subgraph its own persistent checkpoint
    # namespace, keyed by the node name ("agent").
    wrapper = (
        StateGraph(MessagesState)
        .add_node("agent", inner)
        .add_edge(START, "agent")
        .compile(checkpointer=True)
    )

    subgraph_messages: list[list[str]] = []
    topics = ["apples", "bananas"]

    async def call_inner(state: ParentState) -> dict:
        topic = topics[len(subgraph_messages)]
        resp = await wrapper.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append([m.text for m in resp["messages"]])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    result1 = await parent.ainvoke({"result": ""}, config)
    assert result1 == {"result": "Processing: tell me about apples"}

    result2 = await parent.ainvoke({"result": ""}, config)
    assert result2 == {"result": "Processing: tell me about bananas"}

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
async def test_stateful_state_accumulates_with_interrupt_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a stateful subgraph (checkpointer=True) retains its
    message history across parent invocations even when interrupt/resume is
    involved. The subgraph interrupts before echoing, then responds "Done".
    After two invoke+resume cycles, the second run should contain the full
    accumulated history from both calls.
    """

    # Build a subgraph that interrupts before echoing, then responds "Done"
    def process(state: MessagesState) -> dict:
        interrupt("continue?")
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    def respond(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="Done")]}

    inner = (
        StateGraph(MessagesState)
        .add_node("process", process)
        .add_node("respond", respond)
        .add_edge(START, "process")
        .add_edge("process", "respond")
        .compile()
    )

    # Wrap with checkpointer=True for stateful
    wrapper = (
        StateGraph(MessagesState)
        .add_node("agent", inner)
        .add_edge(START, "agent")
        .compile(checkpointer=True)
    )

    subgraph_messages: list[list[str]] = []
    topics = ["apples", "bananas"]

    async def call_inner(state: ParentState) -> dict:
        topic = topics[len(subgraph_messages)]
        resp = await wrapper.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append([m.text for m in resp["messages"]])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    # First invoke+resume cycle
    result = await parent.ainvoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }
    result = await parent.ainvoke(Command(resume=True), config)
    assert result == {"result": "Done"}

    # Second invoke+resume cycle
    result = await parent.ainvoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }
    result = await parent.ainvoke(Command(resume=True), config)
    assert result == {"result": "Done"}

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
async def test_stateful_interrupt_resume_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that a stateful subgraph (checkpointer=True) correctly
    supports interrupt/resume while also accumulating state. Each invoke+resume
    pair triggers the subgraph, and after the second pair completes we verify
    both the per-step invoke outputs and the accumulated message history. This
    exercises the full lifecycle: interrupt, resume, state accumulation.
    """

    # Build a subgraph that interrupts before echoing, then responds "Done"
    def process(state: MessagesState) -> dict:
        interrupt("continue?")
        return {
            "messages": [AIMessage(content=f"Processing: {state['messages'][-1].text}")]
        }

    def respond(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="Done")]}

    inner = (
        StateGraph(MessagesState)
        .add_node("process", process)
        .add_node("respond", respond)
        .add_edge(START, "process")
        .add_edge("process", "respond")
        .compile()
    )

    # Wrap with checkpointer=True for stateful
    wrapper = (
        StateGraph(MessagesState)
        .add_node("agent", inner)
        .add_edge(START, "agent")
        .compile(checkpointer=True)
    )

    subgraph_messages: list[list[str]] = []
    topics = ["apples", "bananas"]

    async def call_inner(state: ParentState) -> dict:
        topic = topics[len(subgraph_messages)]
        resp = await wrapper.ainvoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append([m.text for m in resp["messages"]])
        return {"result": resp["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

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
async def test_stateful_namespace_isolation_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Tests that two different stateful subgraphs (checkpointer=True)
    maintain completely independent state when they use different wrapper node
    names. A "fruit_agent" and "veggie_agent" are each wrapped in their own
    stateful graph. After two parent invocations, each agent should only
    see its own accumulated history with no cross-contamination between them.
    """

    # Build two simple echo subgraphs with different prefixes
    def fruit_echo(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content=f"Fruit: {state['messages'][-1].text}")]}

    def veggie_echo(state: MessagesState) -> dict:
        return {
            "messages": [AIMessage(content=f"Veggie: {state['messages'][-1].text}")]
        }

    fruit_inner = (
        StateGraph(MessagesState)
        .add_node("echo", fruit_echo)
        .add_edge(START, "echo")
        .compile()
    )
    veggie_inner = (
        StateGraph(MessagesState)
        .add_node("echo", veggie_echo)
        .add_edge(START, "echo")
        .compile()
    )

    # Wrap each with checkpointer=True, using different node names to get
    # independent checkpoint namespaces
    fruit = (
        StateGraph(MessagesState)
        .add_node("fruit_agent", fruit_inner)
        .add_edge(START, "fruit_agent")
        .compile(checkpointer=True)
    )
    veggie = (
        StateGraph(MessagesState)
        .add_node("veggie_agent", veggie_inner)
        .add_edge(START, "veggie_agent")
        .compile(checkpointer=True)
    )

    fruit_msgs: list[list[str]] = []
    veggie_msgs: list[list[str]] = []
    call_count = 0

    async def call_both(state: ParentState) -> dict:
        nonlocal call_count
        call_count += 1
        suffix = "round 1" if call_count == 1 else "round 2"
        f = await fruit.ainvoke(
            {"messages": [HumanMessage(content=f"cherries {suffix}")]}
        )
        v = await veggie.ainvoke(
            {"messages": [HumanMessage(content=f"broccoli {suffix}")]}
        )
        fruit_msgs.append([m.text for m in f["messages"]])
        veggie_msgs.append([m.text for m in v["messages"]])
        return {"result": f["messages"][-1].text}

    parent = (
        StateGraph(ParentState)
        .add_node("call_both", call_both)
        .add_edge(START, "call_both")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    result1 = await parent.ainvoke({"result": ""}, config)
    assert result1 == {"result": "Fruit: cherries round 1"}

    result2 = await parent.ainvoke({"result": ""}, config)
    assert result2 == {"result": "Fruit: cherries round 2"}

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
