"""Tests for subgraph persistence behavior (async).

Covers three checkpointer settings for subgraph state:
- checkpointer=False: no persistence, even when parent has a checkpointer
- checkpointer=None (default): "stateless" — inherits parent checkpointer for
  interrupt support, but state resets each invocation. This is the common case
  when an agent is invoked from inside a tool used by another agent.
- checkpointer=True: "stateful" — state accumulates across invocations on the same thread id

Also covers reading a subgraph's namespace from the parent graph, including
one where no subgraph is declared, such as a subagent a tool invokes, and
finding the subgraph a namespace names.
"""

import sys
from typing import Annotated, Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState, _messages_delta_reducer
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


@NEEDS_CONTEXTVARS
async def test_child_with_own_thread_id_keeps_namespace_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """A child graph invoked from inside a parent node with its own thread_id
    must store and read its checkpoint under its own namespace, not inherit the
    parent task's checkpoint_ns.
    """

    class ChildState(TypedDict):
        count: int

    def child_node(state: ChildState) -> dict:
        return {"count": (state.get("count") or 0) + 1}

    child = (
        StateGraph(ChildState)
        .add_node("n", child_node)
        .add_edge(START, "n")
        .compile(checkpointer=async_checkpointer)
    )

    child_thread = str(uuid4())
    child_config = {"configurable": {"thread_id": child_thread}}

    async def parent_node(state: ParentState) -> dict:
        await child.ainvoke({}, config=child_config)
        return {"result": "ok"}

    parent = (
        StateGraph(ParentState)
        .add_node("p", parent_node)
        .add_edge(START, "p")
        .compile(checkpointer=async_checkpointer)
    )
    parent_config = {"configurable": {"thread_id": str(uuid4())}}

    await parent.ainvoke({"result": ""}, config=parent_config)
    state1 = await child.aget_state(child_config)
    assert state1.values.get("count") == 1
    assert state1.config["configurable"]["checkpoint_ns"] == ""

    await parent.ainvoke({"result": ""}, config=parent_config)
    state2 = await child.aget_state(child_config)
    assert state2.values.get("count") == 2


# -- reading a subgraph's namespace from the parent graph --


class NotesState(MessagesState):
    notes: str  # a key only the subgraphs have


class DeltaMessagesState(TypedDict):
    # Stores `messages` as deltas between snapshots, as deepagents' parent does.
    messages: Annotated[list[AnyMessage], DeltaChannel(_messages_delta_reducer)]


class DeltaNotesState(DeltaMessagesState):
    notes: str


def _notes_graph(label: str, state_schema: type = NotesState) -> Any:
    # Typed `dict` so it fits either schema (see `_run_tool_subgraph`).
    def work(state: dict) -> dict:
        return {"messages": [AIMessage(content=f"{label} done")], "notes": label}

    return (
        StateGraph(state_schema)
        .add_node("work", work)
        .add_edge(START, "work")
        .compile()
    )


async def _namespace_configs(parent: Any) -> dict[str, RunnableConfig]:
    """Run `parent` on a new thread. Returns a config for each subgraph
    namespace it wrote, keyed by the node the namespace starts with."""
    thread_id = str(uuid4())
    return {
        ns[0].split(":")[0]: {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ns[0]}
        }
        async for ns, _ in parent.astream(
            {"messages": [HumanMessage(content="go")]},
            {"configurable": {"thread_id": thread_id}},
            subgraphs=True,
        )
        if ns
    }


async def _run_tool_subgraph(
    checkpointer: BaseCheckpointSaver,
    *,
    parent_schema: type = MessagesState,
    subagent_schema: type = NotesState,
) -> tuple[Any, RunnableConfig]:
    """Run a parent whose `tools` node invokes a subgraph it doesn't declare,
    as deepagents runs subagents. Returns the parent and a config for the
    subgraph's `tools:<task_id>` namespace."""
    # Subgraph detection doesn't follow a subscript, so `tools` declares none.
    subagents = {"researcher": _notes_graph("researcher", subagent_schema)}

    # Typed `dict`: a schema here would add its `messages` channel to the
    # parent's, and a plain one clashes with a `DeltaChannel` one.
    async def tools(state: dict) -> dict:
        await subagents["researcher"].ainvoke(
            {"messages": [HumanMessage(content="research apples")]}
        )
        return {"messages": [AIMessage(content="tool done")]}

    parent = (
        StateGraph(parent_schema)
        .add_node("tools", tools)
        .add_edge(START, "tools")
        .compile(checkpointer=checkpointer)
    )
    return parent, (await _namespace_configs(parent))["tools"]


@NEEDS_CONTEXTVARS
async def test_undeclared_subgraph_get_state_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """A namespace with no declared subgraph is read with the parent graph."""
    parent, ns_config = await _run_tool_subgraph(async_checkpointer)

    state = await parent.aget_state(ns_config)

    assert [m.text for m in state.values["messages"]] == [
        "research apples",
        "researcher done",
    ]
    assert state.config["configurable"]["checkpoint_ns"] == AnyStr("tools:")


@NEEDS_CONTEXTVARS
async def test_undeclared_subgraph_get_state_history_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """History of a namespace with no declared subgraph is read with the parent
    graph, and only holds that namespace's checkpoints."""
    parent, ns_config = await _run_tool_subgraph(async_checkpointer)

    history = [s async for s in parent.aget_state_history(ns_config)]

    assert [[m.text for m in s.values.get("messages", [])] for s in history] == [
        ["research apples", "researcher done"],
        ["research apples"],
        [],
    ]
    assert {s.config["configurable"]["checkpoint_ns"] for s in history} == {
        ns_config["configurable"]["checkpoint_ns"]
    }


@NEEDS_CONTEXTVARS
async def test_undeclared_subgraph_update_state_raises_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Unlike a read, an update can't fall back to the parent graph: its nodes
    and channels would write the wrong state into the subgraph's namespace."""
    parent, ns_config = await _run_tool_subgraph(async_checkpointer)

    with pytest.raises(ValueError, match="Subgraph tools not found"):
        await parent.aupdate_state(ns_config, {"messages": [AIMessage(content="edit")]})

    assert len([s async for s in parent.aget_state_history(ns_config)]) == 3


@NEEDS_CONTEXTVARS
@pytest.mark.parametrize(
    ("parent_schema", "subagent_schema"),
    [(DeltaMessagesState, NotesState), (MessagesState, DeltaNotesState)],
    ids=["delta_parent", "delta_subagent"],
)
async def test_undeclared_subgraph_read_across_messages_channel_types_async(
    async_checkpointer: BaseCheckpointSaver,
    parent_schema: type,
    subagent_schema: type,
) -> None:
    """deepagents' parent stores `messages` in a `DeltaChannel`, and its
    subagents in a plain `add_messages` channel. Reading a subagent's namespace
    with the parent's channel restores its messages, either way round."""
    parent, ns_config = await _run_tool_subgraph(
        async_checkpointer,
        parent_schema=parent_schema,
        subagent_schema=subagent_schema,
    )

    state = await parent.aget_state(ns_config)
    history = [s async for s in parent.aget_state_history(ns_config)]

    assert [m.text for m in state.values["messages"]] == [
        "research apples",
        "researcher done",
    ]
    assert [[m.text for m in s.values.get("messages", [])] for s in history] == [
        ["research apples", "researcher done"],
        ["research apples"],
        [],
    ]


@NEEDS_CONTEXTVARS
@pytest.mark.parametrize("review_declared", [False, True])
async def test_prefix_named_namespaces_resolve_to_their_own_graph_async(
    async_checkpointer: BaseCheckpointSaver, review_declared: bool
) -> None:
    """`research_review` starts with `research`, whose subgraph has a `review`
    node with its own subgraph. Each namespace still reads with its own graph:
    a declared subgraph's channels, or the parent's when none is declared."""
    research = (
        StateGraph(NotesState)
        .add_node("review", _notes_graph("research"))
        .add_edge(START, "review")
        .compile()
    )

    class ReviewState(NotesState):
        verdict: str  # a key `research|review`'s graph doesn't have

    def review(state: ReviewState) -> dict:
        return {
            "messages": [AIMessage(content="review done")],
            "notes": "review",
            "verdict": "approved",
        }

    # Subgraph detection doesn't follow a subscript, so `call_review` declares
    # none.
    subgraphs = {
        "review": StateGraph(ReviewState)
        .add_node("work", review)
        .add_edge(START, "work")
        .compile()
    }

    async def call_review(state: MessagesState) -> dict:
        await subgraphs["review"].ainvoke(
            {"messages": [HumanMessage(content="check it")]}
        )
        return {}

    parent = (
        StateGraph(MessagesState)
        .add_node("research", research)
        .add_node(
            "research_review",
            subgraphs["review"] if review_declared else call_review,
        )
        .add_edge(START, "research")
        .add_edge("research", "research_review")
        .compile(checkpointer=async_checkpointer)
    )
    configs = await _namespace_configs(parent)

    async def values(name: str) -> dict:
        state = await parent.aget_state(configs[name])
        return {
            k: [m.text for m in v] if k == "messages" else v
            for k, v in state.values.items()
        }

    assert await values("research") == {
        "messages": ["go", "research done"],
        "notes": "research",
    }
    if review_declared:
        assert await values("research_review") == {
            "messages": ["go", "research done", "review done"],
            "notes": "review",
            "verdict": "approved",
        }
    else:
        # The parent's channels have no `notes` or `verdict`.
        assert await values("research_review") == {
            "messages": ["check it", "review done"]
        }


def _prefix_named_subgraphs() -> tuple[Any, dict[str, Any]]:
    """A parent whose subgraph namespaces start with one another's names:
    `research` has a `review` subgraph, and the parent declares
    `research_review` and `review` too. Returns the parent and each subgraph,
    keyed by its namespace."""
    graphs = {
        "research|review": _notes_graph("research's review"),
        "research_review": _notes_graph("research_review"),
        "review": _notes_graph("review"),
    }
    graphs["research"] = (
        StateGraph(NotesState)
        .add_node("review", graphs["research|review"])
        .add_edge(START, "review")
        .compile()
    )
    parent = (
        StateGraph(NotesState)
        .add_node("research", graphs["research"])
        .add_node("research_review", graphs["research_review"])
        .add_node("review", graphs["review"])
        .add_edge(START, "research")
        .add_edge("research", "research_review")
        .add_edge("research_review", "review")
        .compile()
    )
    return parent, graphs


@pytest.mark.parametrize(
    "namespace", ["research", "research|review", "research_review", "review"]
)
async def test_get_subgraphs_matches_whole_namespace_segments_async(
    namespace: str,
) -> None:
    """`aget_subgraphs(namespace=...)` yields only the subgraph at that
    namespace. `research_review` isn't `research` then `review`, and
    `research|review` doesn't also find the parent's own `review`."""
    parent, graphs = _prefix_named_subgraphs()

    assert [
        s async for s in parent.aget_subgraphs(namespace=namespace, recurse=True)
    ] == [(namespace, graphs[namespace])]


async def test_nested_namespace_update_raises_despite_top_level_namesake_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """`research|review` names a `review` inside `research`'s subgraph, which
    declares no subgraph there. An update to it raises rather than writing
    through the parent's own `review`."""
    research = (
        StateGraph(NotesState)
        .add_node("review", lambda state: {})
        .add_edge(START, "review")
        .compile()
    )
    parent = (
        StateGraph(NotesState)
        .add_node("research", research)
        .add_node("review", _notes_graph("review"))
        .add_edge(START, "research")
        .add_edge("research", "review")
        .compile(checkpointer=async_checkpointer)
    )
    config: RunnableConfig = {
        "configurable": {
            "thread_id": str(uuid4()),
            "checkpoint_ns": "research:a|review:b",
        }
    }

    with pytest.raises(ValueError, match=r"Subgraph research\|review not found"):
        await parent.aupdate_state(config, {"notes": "edit"})
