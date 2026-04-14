"""Prove V1 and StreamingHandler APIs expose identical information.

Each test runs the same graph through both APIs and asserts data
equivalence — same state snapshots, same messages, same custom events,
same interrupts.  Sync APIs are used where possible; async tests cover
features without sync equivalents (subgraphs projection, messages_from).

Run with:
    TEST=tests/test_streaming_comparison.py make test
"""

from __future__ import annotations

from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.stream import StreamingHandler
from langgraph.stream._convert import STREAM_V2_MODES
from langgraph.types import interrupt
from tests.fake_chat import FakeChatModel

# ---------------------------------------------------------------------------
# Graph factories
# ---------------------------------------------------------------------------


class State(TypedDict):
    value: str
    items: Annotated[list[str], lambda a, b: a + b]


def _linear_graph(n_nodes: int = 3):
    """Chain of *n_nodes* that concatenate strings."""
    g = StateGraph(State)
    names = [f"node_{i}" for i in range(n_nodes)]
    for name in names:

        def make_fn(n):
            def fn(state: State) -> dict:
                return {"value": state["value"] + f"_{n}", "items": [n]}

            return fn

        g.add_node(name, make_fn(name))

    g.add_edge(START, names[0])
    for i in range(len(names) - 1):
        g.add_edge(names[i], names[i + 1])
    g.add_edge(names[-1], END)
    return g.compile()


def _chat_graph():
    """Single agent node with a FakeChatModel."""
    model = FakeChatModel(messages=[AIMessage(content="Hello from agent")])

    def agent(state: dict) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    return g.compile()


def _multi_node_chat_graph():
    """Two LLM nodes: agent -> reviewer."""
    agent_model = FakeChatModel(messages=[AIMessage(content="Agent reply")])
    reviewer_model = FakeChatModel(messages=[AIMessage(content="Reviewer reply")])

    def agent(state: dict) -> dict:
        return {"messages": [agent_model.invoke(state["messages"])]}

    def reviewer(state: dict) -> dict:
        return {"messages": [reviewer_model.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("reviewer", reviewer)
    g.add_edge(START, "agent")
    g.add_edge("agent", "reviewer")
    g.add_edge("reviewer", END)
    return g.compile()


def _custom_events_graph():
    """Node that emits custom events via StreamWriter."""

    def worker(state: State) -> dict:
        writer = get_stream_writer()
        writer({"step": 1, "msg": "started"})
        writer({"step": 2, "msg": "processing"})
        writer({"step": 3, "msg": "done"})
        return {"value": state["value"] + "_done", "items": ["done"]}

    g = StateGraph(State)
    g.add_node("worker", worker)
    g.add_edge(START, "worker")
    g.add_edge("worker", END)
    return g.compile()


def _interrupt_graph():
    """Graph that interrupts for human input."""

    def ask_human(state: State) -> dict:
        answer = interrupt("What next?")
        return {"value": state["value"] + f"_{answer}", "items": [answer]}

    g = StateGraph(State)
    g.add_node("ask", ask_human)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    return g.compile(checkpointer=MemorySaver())


def _subgraph():
    """Parent with a compiled child subgraph."""

    class ChildState(TypedDict):
        value: str

    class ParentState(TypedDict):
        value: str

    def child_node(state: ChildState) -> dict:
        return {"value": state["value"] + "_child"}

    child = StateGraph(ChildState)
    child.add_node("inner", child_node)
    child.add_edge(START, "inner")
    child.add_edge("inner", END)
    child_compiled = child.compile()

    parent = StateGraph(ParentState)
    parent.add_node("child", child_compiled)
    parent.add_edge(START, "child")
    parent.add_edge("child", END)
    return parent.compile()


# ===================================================================
# 1. Final output
# ===================================================================


def test_output():
    """graph.invoke() produces the same result as StreamingHandler().stream().output."""
    graph = _linear_graph()
    inp = {"value": "x", "items": []}

    v1 = graph.invoke(inp)

    run = StreamingHandler(graph).stream(inp)
    v2 = run.output

    assert v1 == v2


# ===================================================================
# 2. Intermediate state snapshots (values mode)
# ===================================================================


def test_values():
    """stream(mode='values') snapshots == StreamingHandler().stream().values snapshots."""
    graph = _linear_graph()
    inp = {"value": "x", "items": []}

    v1 = list(graph.stream(inp, stream_mode="values"))

    run = StreamingHandler(graph).stream(inp)
    v2 = list(run.values)

    assert v1 == v2


# ===================================================================
# 3. Per-node updates (updates mode)
# ===================================================================


def test_updates():
    """stream(mode='updates') data == StreamingHandler raw events[method=updates]."""
    graph = _linear_graph()
    inp = {"value": "x", "items": []}

    v1 = list(graph.stream(inp, stream_mode="updates"))

    run = StreamingHandler(graph).stream(inp)
    v2 = [
        e["params"]["data"]
        for e in run
        if e["method"] == "updates" and not e["params"]["namespace"]
    ]

    assert v1 == v2


# ===================================================================
# 4. Message text and node attribution
# ===================================================================


def test_messages():
    """Reassembled V1 message text per node == V2 .messages text per node."""
    graph = _multi_node_chat_graph()
    inp = {"messages": [HumanMessage(content="hi")]}

    # V1: collect (chunk, metadata) pairs, group text by node
    v1_text_by_node: dict[str, list[str]] = {}
    for chunk, metadata in graph.stream(inp, stream_mode="messages"):
        node = metadata["langgraph_node"]
        v1_text_by_node.setdefault(node, []).append(chunk.content)
    v1_text = {k: "".join(v) for k, v in v1_text_by_node.items()}

    # V2: each ChatModelStream has .text and .node
    run = StreamingHandler(graph).stream(inp)
    v2_text: dict[str, str] = {}
    for msg in run.messages:
        assert msg.done is True
        v2_text[msg.node] = msg.text

    assert v1_text == v2_text


# ===================================================================
# 5. Custom events
# ===================================================================


def test_custom_events():
    """stream(mode='custom') payloads == StreamingHandler raw events[method=custom]."""
    graph = _custom_events_graph()
    inp = {"value": "x", "items": []}

    v1 = list(graph.stream(inp, stream_mode="custom"))

    run = StreamingHandler(graph).stream(inp)
    v2 = [
        e["params"]["data"]
        for e in run
        if e["method"] == "custom" and not e["params"]["namespace"]
    ]

    assert v1 == v2


# ===================================================================
# 6. Mode coverage
# ===================================================================


def test_mode_coverage():
    """V2 produces events for the same set of modes as V1."""
    graph = _chat_graph()
    inp = {"messages": [HumanMessage(content="hi")]}

    # V1: request all modes, collect which ones appear
    v1_modes: set[str] = set()
    for ns, mode, _ in graph.stream(
        inp, stream_mode=STREAM_V2_MODES, subgraphs=True, version="v1"
    ):
        if not ns:
            v1_modes.add(mode)

    # V2: iterate raw events, collect methods
    run = StreamingHandler(graph).stream(inp)
    v2_modes = {e["method"] for e in run if not e["params"]["namespace"]}

    assert v1_modes == v2_modes


# ===================================================================
# 7. Interrupt detection
# ===================================================================


def test_interrupts():
    """V1 __interrupt__ value == V2 .interrupted and .interrupts payload."""
    graph = _interrupt_graph()
    inp = {"value": "x", "items": []}

    # V1: detect __interrupt__ in values stream
    config1 = {"configurable": {"thread_id": "equiv-1"}}
    v1_interrupt_value = None
    for chunk in graph.stream(inp, config1, stream_mode="values"):
        if isinstance(chunk, dict) and "__interrupt__" in chunk:
            info = chunk["__interrupt__"]
            if info:
                v1_interrupt_value = info[0].value

    assert v1_interrupt_value is not None

    # V2: .interrupted and .interrupts (fresh thread)
    config2 = {"configurable": {"thread_id": "equiv-2"}}
    run = StreamingHandler(graph).stream(inp, config=config2)
    for _ in run:
        pass

    assert run.interrupted is True
    assert len(run.interrupts) > 0
    v2_interrupt_value = run.interrupts[0]["payload"].value

    assert v1_interrupt_value == v2_interrupt_value


# ===================================================================
# 8. Subgraph state snapshots
# ===================================================================


def test_subgraph_values():
    """V1 child namespace values == V2 child namespace values."""
    graph = _subgraph()
    inp = {"value": "x"}

    # V1: stream with subgraphs=True, collect child values
    v1_child_values = []
    for ns, data in graph.stream(inp, stream_mode="values", subgraphs=True):
        if ns:
            v1_child_values.append(data)

    # V2: filter raw events for child namespace + values mode
    run = StreamingHandler(graph).stream(inp)
    v2_child_values = [
        e["params"]["data"]
        for e in run
        if e["method"] == "values" and e["params"]["namespace"]
    ]

    assert v1_child_values == v2_child_values


# ===================================================================
# 9. Node filtering on messages
# ===================================================================


def test_messages_node_filtering():
    """V1 manual metadata filter == V2 .messages filtered by .node."""
    graph = _multi_node_chat_graph()
    inp = {"messages": [HumanMessage(content="hi")]}

    # V1: manual filter for "agent" node only
    v1_agent_text: list[str] = []
    for chunk, metadata in graph.stream(inp, stream_mode="messages"):
        if metadata.get("langgraph_node") == "agent":
            v1_agent_text.append(chunk.content)
    v1_text = "".join(v1_agent_text)

    # V2: filter .messages by .node
    run = StreamingHandler(graph).stream(inp)
    v2_agent_msgs = [msg for msg in run.messages if msg.node == "agent"]
    assert len(v2_agent_msgs) == 1
    v2_text = v2_agent_msgs[0].text

    assert v1_text == v2_text


# ===================================================================
# 10. Async: subgraphs projection
# ===================================================================


@pytest.mark.anyio
async def test_async_subgraph_projection():
    """V2 .subgraphs child output matches V1 child namespace output."""
    graph = _subgraph()
    inp = {"value": "x"}

    # V1
    v1_child_output = None
    async for ns, data in graph.astream(inp, stream_mode="values", subgraphs=True):
        if ns:
            v1_child_output = data

    # V2: .subgraphs yields typed child stream objects
    run = await StreamingHandler(graph).astream(inp)
    v2_child_output = None
    async for sub in run.subgraphs:
        v2_child_output = await sub.output

    assert v1_child_output == v2_child_output


# ===================================================================
# 11. Async: messages_from projection
# ===================================================================


@pytest.mark.anyio
async def test_async_messages_from():
    """V2 .messages_from('agent') text matches V1 filtered by metadata."""
    graph = _multi_node_chat_graph()
    inp = {"messages": [HumanMessage(content="hi")]}

    # V1: manual filter for agent node
    v1_agent_text: list[str] = []
    async for chunk, metadata in graph.astream(inp, stream_mode="messages"):
        if metadata.get("langgraph_node") == "agent":
            v1_agent_text.append(chunk.content)
    v1_text = "".join(v1_agent_text)

    # V2: declarative node filtering
    run = await StreamingHandler(graph).astream(inp)
    v2_texts: list[str] = []
    async for msg in run.messages_from("agent"):
        v2_texts.append(await msg.text)
    assert len(v2_texts) == 1
    v2_text = v2_texts[0]

    assert v1_text == v2_text
