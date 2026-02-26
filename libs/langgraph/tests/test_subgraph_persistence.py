"""Tests for subgraph persistence behavior.

Covers three checkpointer settings for subgraph state:
- checkpointer=False: no persistence, even when parent has a checkpointer
- checkpointer=None (default): "tool scope" — inherits parent checkpointer for
  interrupt support, but state resets each invocation
- checkpointer=True: "session scope" — state accumulates across invocations
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command, Interrupt, interrupt
from tests.any_str import AnyStr

# -- Shared helpers --


class ParentState(TypedDict):
    result: str


def _contents(messages: list) -> list[str]:
    """Extract message content strings for readable assertions."""
    return [m.content for m in messages]


def _echo_graph(prefix: str = "Echo", *, checkpointer=None):
    """Single-node subgraph: echoes '<prefix>: <last message content>'."""

    def echo(state: MessagesState):
        content = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"{prefix}: {content}")]}

    return (
        StateGraph(MessagesState)
        .add_node("echo", echo)
        .add_edge(START, "echo")
        .compile(checkpointer=checkpointer)
    )


def _interrupt_echo_graph():
    """Two-node subgraph with interrupt: echoes 'Processing: <input>' then 'Done'."""

    def process(state: MessagesState):
        interrupt("continue?")
        content = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"Processing: {content}")]}

    def respond(state: MessagesState):
        return {"messages": [AIMessage(content="Done")]}

    return (
        StateGraph(MessagesState)
        .add_node("process", process)
        .add_node("respond", respond)
        .add_edge(START, "process")
        .add_edge("process", "respond")
        .compile()
    )


def _wrap_session_scope(inner, name: str):
    """Wrap a compiled subgraph for session scope (checkpointer=True)."""
    return (
        StateGraph(MessagesState)
        .add_node(name, inner)
        .add_edge(START, name)
        .compile(checkpointer=True)
    )


def _wrap_session_scope_interrupt(name: str):
    """Session-scope subgraph with interrupt: echoes 'Processing: <input>' then 'Done'."""
    return _wrap_session_scope(_interrupt_echo_graph(), name)


# -- checkpointer=None (tool scope) --


def test_tool_scope_interrupt_resume(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=None supports interrupt/resume."""
    inner = _interrupt_echo_graph()

    def call_inner(state: ParentState):
        resp = inner.invoke({"messages": [HumanMessage(content="apples")]})
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    result = parent.invoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }

    result = parent.invoke(Command(resume=True), config)
    assert result == {"result": "Done"}


def test_tool_scope_state_resets(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=None resets subgraph state on each invocation."""
    inner = _interrupt_echo_graph()
    subgraph_messages: list[list[str]] = []
    call_count = 0

    def call_inner(state: ParentState):
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        resp = inner.invoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    parent.invoke({"result": ""}, config)
    parent.invoke(Command(resume=True), config)
    parent.invoke({"result": ""}, config)
    parent.invoke(Command(resume=True), config)

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


def test_checkpointer_false_no_persistence(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=False prevents persistence even when parent has a checkpointer."""
    inner = _echo_graph("Processed", checkpointer=False)
    subgraph_messages: list[list[str]] = []
    call_count = 0

    def call_inner(state: ParentState):
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        resp = inner.invoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    parent.invoke({"result": ""}, config)
    parent.invoke({"result": ""}, config)

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


def test_session_scope_state_accumulates(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=True retains subgraph state across invocations."""
    wrapper = _wrap_session_scope(_echo_graph("Processing"), "agent")
    subgraph_messages: list[list[str]] = []
    topics = ["apples", "bananas"]

    def call_inner(state: ParentState):
        topic = topics[len(subgraph_messages)]
        resp = wrapper.invoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    parent.invoke({"result": ""}, config)
    parent.invoke({"result": ""}, config)

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


def test_session_scope_interrupt_resume(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=True supports interrupt/resume and accumulates state."""
    wrapper = _wrap_session_scope_interrupt("agent")
    subgraph_messages: list[list[str]] = []
    topics = ["apples", "bananas"]

    def call_inner(state: ParentState):
        topic = topics[len(subgraph_messages)]
        resp = wrapper.invoke(
            {"messages": [HumanMessage(content=f"tell me about {topic}")]}
        )
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    # First invocation: hits interrupt
    result = parent.invoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }

    # Resume: completes first call
    result = parent.invoke(Command(resume=True), config)
    assert result == {"result": "Done"}
    assert subgraph_messages[0] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
    ]

    # Second invocation: hits interrupt, state accumulated from first call
    result = parent.invoke({"result": ""}, config)
    assert result == {
        "result": "",
        "__interrupt__": [Interrupt(value="continue?", id=AnyStr())],
    }

    # Resume: completes second call with accumulated state
    result = parent.invoke(Command(resume=True), config)
    assert result == {"result": "Done"}
    assert subgraph_messages[1] == [
        "tell me about apples",
        "Processing: tell me about apples",
        "Done",
        "tell me about bananas",
        "Processing: tell me about bananas",
        "Done",
    ]


def test_session_scope_namespace_isolation(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Different checkpointer=True subgraphs maintain independent state
    via unique wrapper node names."""
    fruit = _wrap_session_scope(_echo_graph("Fruit"), "fruit_agent")
    veggie = _wrap_session_scope(_echo_graph("Veggie"), "veggie_agent")
    fruit_msgs: list[list[str]] = []
    veggie_msgs: list[list[str]] = []
    call_count = 0

    def call_both(state: ParentState):
        nonlocal call_count
        call_count += 1
        suffix = "round 1" if call_count == 1 else "round 2"
        f = fruit.invoke({"messages": [HumanMessage(content=f"cherries {suffix}")]})
        v = veggie.invoke({"messages": [HumanMessage(content=f"broccoli {suffix}")]})
        fruit_msgs.append(_contents(f["messages"]))
        veggie_msgs.append(_contents(v["messages"]))
        return {"result": f["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_both", call_both)
        .add_edge(START, "call_both")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    parent.invoke({"result": ""}, config)
    parent.invoke({"result": ""}, config)

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


def test_session_scope_shared_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Same checkpointer=True subgraph shares one checkpoint across all calls.

    Documents a known limitation: parallel tool calls to the same session-scope
    subgraph read/write the same checkpoint, causing unexpected state merges.
    """
    wrapper = _wrap_session_scope(_echo_graph(), "shared_agent")
    subgraph_messages: list[list[str]] = []
    call_count = 0

    def call_agent(state: ParentState):
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        resp = wrapper.invoke({"messages": [HumanMessage(content=topic)]})
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_agent", call_agent)
        .add_edge(START, "call_agent")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    parent.invoke({"result": ""}, config)
    parent.invoke({"result": ""}, config)

    # First call: fresh history
    assert subgraph_messages[0] == ["apples", "Echo: apples"]
    # Second call: shared checkpoint retained state from first call
    assert subgraph_messages[1] == [
        "apples",
        "Echo: apples",
        "bananas",
        "Echo: bananas",
    ]


# -- Async equivalents --


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


async def test_tool_scope_state_resets_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """checkpointer=None resets subgraph state on each invocation (async)."""
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


async def test_session_scope_shared_checkpoint_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Same checkpointer=True subgraph shares one checkpoint across all calls (async).

    Documents a known limitation: parallel tool calls to the same session-scope
    subgraph read/write the same checkpoint, causing unexpected state merges.
    """
    wrapper = _wrap_session_scope(_echo_graph(), "shared_agent")
    subgraph_messages: list[list[str]] = []
    call_count = 0

    async def call_agent(state: ParentState):
        nonlocal call_count
        call_count += 1
        topic = "apples" if call_count == 1 else "bananas"
        resp = await wrapper.ainvoke({"messages": [HumanMessage(content=topic)]})
        subgraph_messages.append(_contents(resp["messages"]))
        return {"result": resp["messages"][-1].content}

    parent = (
        StateGraph(ParentState)
        .add_node("call_agent", call_agent)
        .add_edge(START, "call_agent")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}

    await parent.ainvoke({"result": ""}, config)
    await parent.ainvoke({"result": ""}, config)

    # First call: fresh history
    assert subgraph_messages[0] == ["apples", "Echo: apples"]
    # Second call: shared checkpoint retained state from first call
    assert subgraph_messages[1] == [
        "apples",
        "Echo: apples",
        "bananas",
        "Echo: bananas",
    ]


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
