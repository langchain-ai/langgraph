import asyncio
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.stream import AsyncChatModelStream, StreamingHandler
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from tests.fake_chat import FakeChatModel


class State(TypedDict):
    value: str
    items: Annotated[list[str], lambda a, b: a + b]


def make_simple_graph():
    def node_a(state):
        return {"value": state["value"] + "_a", "items": ["a"]}

    def node_b(state):
        return {"value": state["value"] + "_b", "items": ["b"]}

    graph = StateGraph(State)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    return graph.compile()


@pytest.mark.anyio
async def test_output():
    graph = make_simple_graph()
    run = await StreamingHandler(graph).astream({"value": "x", "items": []})
    await asyncio.sleep(0.1)
    output = await run.output
    assert output == {"value": "x_a_b", "items": ["a", "b"]}


@pytest.mark.anyio
async def test_values_iteration():
    graph = make_simple_graph()
    run = await StreamingHandler(graph).astream({"value": "x", "items": []})
    await asyncio.sleep(0.1)

    snapshots = []
    async for v in run.values:
        snapshots.append(v)

    assert len(snapshots) == 3
    assert snapshots[0]["value"] == "x"
    assert snapshots[1]["value"] == "x_a"
    assert snapshots[2]["value"] == "x_a_b"


@pytest.mark.anyio
async def test_updates_in_raw_events():
    graph = make_simple_graph()
    run = await StreamingHandler(graph).astream({"value": "x", "items": []})
    await asyncio.sleep(0.1)

    updates = []
    async for event in run:
        if event["method"] == "updates":
            updates.append(event["params"]["data"])

    assert len(updates) == 2
    assert "node_a" in updates[0]
    assert "node_b" in updates[1]


@pytest.mark.anyio
async def test_messages_with_chat_model():
    model = FakeChatModel(messages=[AIMessage(content="Hello world")])

    def agent(state):
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    run = await StreamingHandler(compiled).astream(
        {"messages": [HumanMessage(content="hi")]}
    )
    await asyncio.sleep(0.1)

    messages_seen = []
    async for msg in run.messages:
        messages_seen.append(msg)

    assert len(messages_seen) >= 1
    msg = messages_seen[0]
    assert isinstance(msg, AsyncChatModelStream)
    text = await msg.text
    assert text == "Hello world"


@pytest.mark.anyio
async def test_custom_events():
    def node(state):
        writer = get_stream_writer()
        writer("hello")
        writer(42)
        return {"value": state["value"] + "_a", "items": ["a"]}

    graph = StateGraph(State)
    graph.add_node("node_a", node)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", END)
    compiled = graph.compile()

    run = await StreamingHandler(compiled).astream({"value": "x", "items": []})
    await asyncio.sleep(0.1)

    custom_payloads = []
    async for event in run:
        if event["method"] == "custom":
            custom_payloads.append(event["params"]["data"])

    assert "hello" in custom_payloads
    assert 42 in custom_payloads


@pytest.mark.anyio
async def test_multiple_modes_present():
    graph = make_simple_graph()
    run = await StreamingHandler(graph).astream({"value": "x", "items": []})
    await asyncio.sleep(0.1)

    methods = set()
    async for event in run:
        methods.add(event["method"])

    assert {"values", "updates", "tasks", "debug"} <= methods


@pytest.mark.anyio
async def test_interrupted_false():
    graph = make_simple_graph()
    run = await StreamingHandler(graph).astream({"value": "x", "items": []})
    await asyncio.sleep(0.1)
    async for _ in run:
        pass
    assert run.interrupted is False


@pytest.mark.anyio
async def test_regression_v1_stream_unchanged():
    graph = make_simple_graph()
    chunks = []
    async for chunk in graph.astream(
        {"value": "x", "items": []}, stream_mode="values", version="v1"
    ):
        chunks.append(chunk)
    for chunk in chunks:
        assert isinstance(chunk, dict)


@pytest.mark.anyio
async def test_regression_v2_stream_unchanged():
    graph = make_simple_graph()
    chunks = []
    async for chunk in graph.astream(
        {"value": "x", "items": []}, stream_mode="values", version="v2"
    ):
        chunks.append(chunk)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert "type" in chunk
        assert chunk["type"] == "values"


@pytest.mark.anyio
async def test_regression_invoke_unchanged():
    graph = make_simple_graph()
    result = await graph.ainvoke({"value": "x", "items": []})
    assert result == {"value": "x_a_b", "items": ["a", "b"]}


def test_sync_stream_output():
    graph = make_simple_graph()
    run = StreamingHandler(graph).stream({"value": "x", "items": []})
    assert run.output == {"value": "x_a_b", "items": ["a", "b"]}


def test_sync_stream_values():
    graph = make_simple_graph()
    run = StreamingHandler(graph).stream({"value": "x", "items": []})
    snapshots = list(run.values)
    assert len(snapshots) == 3
    assert snapshots[0]["value"] == "x"
    assert snapshots[2]["value"] == "x_a_b"


def test_sync_stream_raw_events():
    graph = make_simple_graph()
    run = StreamingHandler(graph).stream({"value": "x", "items": []})
    methods = {e["method"] for e in run}
    assert {"values", "updates", "tasks", "debug"} <= methods


# ---------------------------------------------------------------------------
# Typed output (pydantic)
# ---------------------------------------------------------------------------


class ModelState(BaseModel):
    value: str
    items: Annotated[list[str], lambda a, b: a + b]


def _make_model_state_graph():
    def node_a(state):
        return {"value": state.value + "_a", "items": ["a"]}

    graph = StateGraph(ModelState)
    graph.add_node("node_a", node_a)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", END)
    return graph.compile()


@pytest.mark.anyio
async def test_pydantic_output():
    graph = _make_model_state_graph()
    run = await StreamingHandler(graph).astream(ModelState(value="x", items=[]))
    await asyncio.sleep(0.1)
    output = await run.output
    assert isinstance(output, ModelState)
    assert output.value == "x_a"


@pytest.mark.anyio
async def test_pydantic_values():
    graph = _make_model_state_graph()
    run = await StreamingHandler(graph).astream(ModelState(value="x", items=[]))
    await asyncio.sleep(0.1)
    snapshots = []
    async for v in run.values:
        snapshots.append(v)
    for v in snapshots:
        assert isinstance(v, ModelState)


def test_sync_pydantic_output():
    graph = _make_model_state_graph()
    run = StreamingHandler(graph).stream(ModelState(value="x", items=[]))
    assert isinstance(run.output, ModelState)
    assert run.output.value == "x_a"


# ---------------------------------------------------------------------------
# Interrupts
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_interrupts():
    from langgraph.checkpoint.memory import MemorySaver

    from langgraph.types import interrupt

    def ask_human(state: State):
        answer = interrupt("what do you want?")
        return {"value": state["value"] + f"_{answer}", "items": [answer]}

    graph = StateGraph(State)
    graph.add_node("ask", ask_human)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    compiled = graph.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "t1"}}
    run = await StreamingHandler(compiled).astream(
        {"value": "x", "items": []}, config=config
    )
    await asyncio.sleep(0.1)
    # Drain events
    async for _ in run:
        pass
    assert run.interrupted is True
    assert len(run.interrupts) > 0


# ---------------------------------------------------------------------------
# messages_from(node)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_messages_from_node():
    model = FakeChatModel(messages=[AIMessage(content="from agent")])

    def agent(state):
        return {"messages": [model.invoke(state["messages"])]}

    def postprocess(state):
        return {"messages": state["messages"]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("postprocess", postprocess)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "postprocess")
    graph.add_edge("postprocess", END)
    compiled = graph.compile()

    run = await StreamingHandler(compiled).astream(
        {"messages": [HumanMessage(content="hi")]}
    )
    await asyncio.sleep(0.1)

    # All messages
    all_msgs = []
    async for m in run.messages:
        all_msgs.append(m)
    assert len(all_msgs) >= 1
    # Node provenance should be set
    assert all_msgs[0].node == "agent"


# ---------------------------------------------------------------------------
# Subgraph child stream
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_subgraph_child_output():
    """AsyncSubgraphRunStream.output should contain the child graph's final state."""

    class ChildState(TypedDict):
        value: str

    class ParentState(TypedDict):
        value: str

    def child_node(state):
        return {"value": state["value"] + "_child"}

    child_graph = StateGraph(ChildState)
    child_graph.add_node("child_node", child_node)
    child_graph.add_edge(START, "child_node")
    child_graph.add_edge("child_node", END)
    # Add the compiled child as a node — this triggers LangGraph's
    # subgraph streaming mechanism and emits child namespace events.
    child_compiled = child_graph.compile()

    parent_graph = StateGraph(ParentState)
    parent_graph.add_node("child_node", child_compiled)
    parent_graph.add_edge(START, "child_node")
    parent_graph.add_edge("child_node", END)
    parent_compiled = parent_graph.compile()

    run = await StreamingHandler(parent_compiled).astream({"value": "x"})
    await asyncio.sleep(0.1)

    subgraph_streams = []
    async for sub in run.subgraphs:
        subgraph_streams.append(sub)

    assert len(subgraph_streams) >= 1
    child_output = await subgraph_streams[0].output
    assert child_output is not None
    assert child_output["value"] == "x_child"


# ---------------------------------------------------------------------------
# Custom reducers / .extensions
# ---------------------------------------------------------------------------


class _CountTransformer(StreamTransformer):
    """Counts events. Exposes count via .value for extensions."""

    name = "event_count"

    def __init__(self) -> None:
        self.value = 0

    def init(self) -> Any:
        return None

    def process(self, event: ProtocolEvent) -> bool:
        self.value += 1
        return True

    def finalize(self) -> None:
        pass

    def fail(self, err: BaseException) -> None:
        pass


@pytest.mark.anyio
async def test_custom_reducer_extensions():
    graph = make_simple_graph()
    counter = _CountTransformer()
    run = await StreamingHandler(graph).astream(
        {"value": "x", "items": []}, transformers=[counter]
    )
    await asyncio.sleep(0.1)
    async for _ in run:
        pass
    assert counter.value > 0
    assert run.extensions["event_count"] == counter.value


def test_sync_custom_reducer_extensions():
    graph = make_simple_graph()
    counter = _CountTransformer()
    run = StreamingHandler(graph).stream(
        {"value": "x", "items": []}, transformers=[counter]
    )
    for _ in run:
        pass
    assert counter.value > 0
    assert run.extensions["event_count"] == counter.value


# ---------------------------------------------------------------------------
# Double iteration over .values
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_values_double_iteration():
    """Iterating over run.values twice should yield the same snapshots both times."""
    graph = make_simple_graph()
    run = await StreamingHandler(graph).astream({"value": "x", "items": []})
    await asyncio.sleep(0.1)

    first = []
    async for v in run.values:
        first.append(v)

    second = []
    async for v in run.values:
        second.append(v)

    assert len(first) == 3
    assert first == second


def test_sync_values_double_iteration():
    """Iterating over run.values twice should yield the same snapshots both times."""
    graph = make_simple_graph()
    run = StreamingHandler(graph).stream({"value": "x", "items": []})

    first = list(run.values)
    second = list(run.values)

    assert len(first) == 3
    assert first == second


@pytest.mark.anyio
async def test_async_raw_events_double_iteration():
    """Iterating over the raw event stream twice should yield the same events."""
    graph = make_simple_graph()
    run = await StreamingHandler(graph).astream({"value": "x", "items": []})
    await asyncio.sleep(0.1)

    first = []
    async for event in run:
        first.append(event)

    second = []
    async for event in run:
        second.append(event)

    assert len(first) > 0
    assert first == second


def test_sync_raw_events_double_iteration():
    """Iterating over the raw event stream twice should yield the same events."""
    graph = make_simple_graph()
    run = StreamingHandler(graph).stream({"value": "x", "items": []})

    first = list(run)
    second = list(run)

    assert len(first) > 0
    assert first == second


# ---------------------------------------------------------------------------
# Tool transformer via extensions
# ---------------------------------------------------------------------------


class _ToolExecution:
    def __init__(self, tool_call_id: str, tool_name: str, input: Any, output: Any):
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.input = input
        self.output = output


class _ToolsTransformer(StreamTransformer):
    """Groups tool-started/tool-finished custom events into _ToolExecution objects."""

    name = "tools"

    def __init__(self) -> None:
        self._log: list[_ToolExecution] = []
        self._pending: dict[str, dict] = {}
        self.value = self._log

    def init(self) -> Any:
        return None

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "custom":
            return True
        data = event["params"]["data"]
        if not isinstance(data, dict) or "event" not in data:
            return True

        tool_call_id = data.get("tool_call_id")
        if tool_call_id is None:
            return True

        if data["event"] == "tool-started":
            self._pending[tool_call_id] = data
            return False

        if data["event"] == "tool-finished":
            started = self._pending.pop(tool_call_id, {})
            self._log.append(_ToolExecution(
                tool_call_id=tool_call_id,
                tool_name=started.get("tool_name", ""),
                input=started.get("input"),
                output=data["output"],
            ))
            return False

        return True

    def finalize(self) -> None:
        pass

    def fail(self, err: BaseException) -> None:
        pass


def _make_tool_graph():
    """Graph: agent emits a tool call, custom_tools executes it with writer events."""
    from langgraph.types import StreamWriter

    def agent(state):
        return {
            "value": "called",
            "items": ["agent"],
        }

    def custom_tools(state, *, writer: StreamWriter):
        writer({
            "event": "tool-started",
            "tool_call_id": "call_1",
            "tool_name": "get_weather",
            "input": {"city": "SF"},
        })
        writer({
            "event": "tool-finished",
            "tool_call_id": "call_1",
            "output": {"temp_f": 64},
        })
        return {"value": "done", "items": ["tools"]}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("custom_tools", custom_tools)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "custom_tools")
    graph.add_edge("custom_tools", END)
    return graph.compile()


def test_sync_tool_transformer_via_extensions():
    """Tool events flow through extensions and are iterable without draining raw events."""
    graph = _make_tool_graph()
    run = StreamingHandler(graph).stream(
        {"value": "", "items": []},
        transformers=[_ToolsTransformer()],
    )

    # Iterating extensions drives the pump — no need to drain raw events first
    executions = list(run.extensions["tools"])
    assert len(executions) == 1
    assert executions[0].tool_name == "get_weather"
    assert executions[0].input == {"city": "SF"}
    assert executions[0].output == {"temp_f": 64}


@pytest.mark.anyio
async def test_async_tool_transformer_via_extensions():
    """Tool events flow through extensions in async mode."""
    graph = _make_tool_graph()
    run = await StreamingHandler(graph).astream(
        {"value": "", "items": []},
        transformers=[_ToolsTransformer()],
    )
    await asyncio.sleep(0.1)

    # Drain main stream so transformer processes all events
    async for _ in run:
        pass

    tools_log = run.extensions["tools"]
    assert len(tools_log) == 1
    assert tools_log[0].tool_name == "get_weather"
    assert tools_log[0].output == {"temp_f": 64}
