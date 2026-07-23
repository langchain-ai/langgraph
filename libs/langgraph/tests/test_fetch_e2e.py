"""End-to-end tests for fetch() across checkpointer backends (sync + async).

Unlike test_fetch.py (which pins InMemorySaver), these drive the full graph runtime
over every parametrized `sync_checkpointer` / `async_checkpointer` backend — so the
FETCH / FETCH_RESULT writes and the FetchRequest / FetchResult payloads are proven to
persist and round-trip through each backend's serde (memory, sqlite, postgres). They
also cover multi-node graphs and streaming, not just single-node suspend/resume.
"""

import time

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.errors import FetchError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, fetch, fetch_all, interrupt

pytestmark = pytest.mark.anyio


class State(TypedDict):
    result: str


class MultiState(TypedDict):
    fetched: str
    result: str


# ── sync, parametrized over every sync checkpointer backend ─────────────────────


def test_e2e_suspend_and_fulfill(sync_checkpointer: BaseCheckpointSaver) -> None:
    builder = StateGraph(State)
    builder.add_node("n", lambda s: {"result": fetch({"resource": "txn"})})
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"result": ""}, config)
    # FetchRequest must have round-tripped through the backend serde
    pending = graph.pending_fetches(config)
    assert len(pending) == 1
    assert pending[0].value == {"resource": "txn"}

    out = graph.invoke(Command(fetch={pending[0].id: "DATA"}), config)
    assert out["result"] == "DATA"
    assert graph.pending_fetches(config) == []


def test_e2e_fetch_all_partial(sync_checkpointer: BaseCheckpointSaver) -> None:
    def node(state):
        a, b = fetch_all([{"r": "A"}, {"r": "B"}])
        return {"result": f"{a}:{b}"}

    builder = StateGraph(State)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"result": ""}, config)
    ids = {f.value["r"]: f.id for f in graph.pending_fetches(config)}
    graph.invoke(Command(fetch={ids["A"]: "AA"}), config)
    assert [f.value["r"] for f in graph.pending_fetches(config)] == ["B"]
    out = graph.invoke(Command(fetch={ids["B"]: "BB"}), config)
    assert out["result"] == "AA:BB"


def test_e2e_deadline_expiry(sync_checkpointer: BaseCheckpointSaver) -> None:
    def node(state):
        try:
            return {"result": fetch({"r": "x"}, deadline=0.001)}
        except FetchError as e:
            return {"result": f"exp:{e.status}"}

    builder = StateGraph(State)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"result": ""}, config)
    time.sleep(0.05)
    graph.expire_fetches(config)
    assert graph.get_state(config).values["result"] == "exp:expired"


def test_e2e_fail_fetch(sync_checkpointer: BaseCheckpointSaver) -> None:
    builder = StateGraph(State)
    builder.add_node("n", lambda s: {"result": fetch({"r": "x"})})
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"result": ""}, config)
    fid = graph.pending_fetches(config)[0].id
    with pytest.raises(FetchError) as exc:
        graph.fail_fetch(config, fid, "upstream 500")
    assert exc.value.status == "failed"


def test_e2e_fetch_and_interrupt_coexist(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    def node(state):
        data = fetch({"resource": "accounts"})
        answer = interrupt("what next?")
        return {"result": f"{data}:{answer}"}

    builder = StateGraph(State)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"result": ""}, config)
    task = graph.get_state(config).tasks[0]
    assert len(task.fetches) == 1 and len(task.interrupts) == 0

    graph.invoke(Command(fetch={task.fetches[0].id: "ACCT"}), config)
    task = graph.get_state(config).tasks[0]
    assert len(task.interrupts) == 1 and len(task.fetches) == 0

    out = graph.invoke(Command(resume="go"), config)
    assert out["result"] == "ACCT:go"


def test_e2e_multi_node_downstream_consumes(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Node A fetches; a downstream node B consumes the fetched value."""

    def a(state):
        return {"fetched": fetch({"resource": "profile"})}

    def b(state):
        return {"result": f"hello {state['fetched']}"}

    builder = StateGraph(MultiState)
    builder.add_node("a", a)
    builder.add_node("b", b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"fetched": "", "result": ""}, config)
    fid = graph.pending_fetches(config)[0].id
    out = graph.invoke(Command(fetch={fid: "Ada"}), config)
    assert out["fetched"] == "Ada"
    assert out["result"] == "hello Ada"


def test_e2e_streaming_emits_fetch_then_resumes(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    from langgraph._internal._constants import FETCH

    builder = StateGraph(State)
    builder.add_node("n", lambda s: {"result": fetch({"r": "x"})})
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    chunks = list(graph.stream({"result": ""}, config, stream_mode="updates"))
    assert any(isinstance(c, dict) and FETCH in c for c in chunks)

    fid = graph.pending_fetches(config)[0].id
    out = graph.invoke(Command(fetch={fid: "DATA"}), config)
    assert out["result"] == "DATA"


# ── async, parametrized over every async checkpointer backend ───────────────────


async def test_e2e_async_suspend_and_fulfill(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    async def node(state):
        return {"result": fetch({"resource": "txn"})}

    builder = StateGraph(State)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke({"result": ""}, config)
    pending = await graph.apending_fetches(config)
    assert len(pending) == 1
    out = await graph.afulfill(config, pending[0].id, "DATA")
    assert out["result"] == "DATA"


async def test_e2e_async_fetch_all_partial(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    async def node(state):
        a, b = fetch_all([{"r": "A"}, {"r": "B"}])
        return {"result": f"{a}:{b}"}

    builder = StateGraph(State)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke({"result": ""}, config)
    ids = {f.value["r"]: f.id for f in await graph.apending_fetches(config)}
    await graph.ainvoke(Command(fetch={ids["A"]: "AA"}), config)
    still = await graph.apending_fetches(config)
    assert [f.value["r"] for f in still] == ["B"]
    out = await graph.ainvoke(Command(fetch={ids["B"]: "BB"}), config)
    assert out["result"] == "AA:BB"


async def test_e2e_async_deadline_expiry(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    async def node(state):
        try:
            return {"result": fetch({"r": "x"}, deadline=0.001)}
        except FetchError as e:
            return {"result": f"exp:{e.status}"}

    builder = StateGraph(State)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke({"result": ""}, config)
    time.sleep(0.05)
    await graph.aexpire_fetches(config)
    state = await graph.aget_state(config)
    assert state.values["result"] == "exp:expired"


async def test_e2e_async_fail_fetch(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    async def node(state):
        return {"result": fetch({"r": "x"})}

    builder = StateGraph(State)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke({"result": ""}, config)
    pending = await graph.apending_fetches(config)
    with pytest.raises(FetchError) as exc:
        await graph.afail_fetch(config, pending[0].id, "boom")
    assert exc.value.status == "failed"


async def test_e2e_async_multi_node(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    async def a(state):
        return {"fetched": fetch({"resource": "profile"})}

    async def b(state):
        return {"result": f"hello {state['fetched']}"}

    builder = StateGraph(MultiState)
    builder.add_node("a", a)
    builder.add_node("b", b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke({"fetched": "", "result": ""}, config)
    pending = await graph.apending_fetches(config)
    out = await graph.ainvoke(Command(fetch={pending[0].id: "Ada"}), config)
    assert out["result"] == "hello Ada"


# ── subgraph nesting: a fetch() inside a subgraph must bubble up to the parent ───


def test_e2e_fetch_inside_subgraph(sync_checkpointer: BaseCheckpointSaver) -> None:
    """A fetch() declared in a subgraph node surfaces at the parent and is
    fulfilled by id through the parent — the content-id carries the subgraph ns."""

    def inner(state):
        return {"result": fetch({"resource": "inner-data"})}

    sub = StateGraph(State)
    sub.add_node("inner", inner)
    sub.add_edge(START, "inner")
    sub.add_edge("inner", END)
    subgraph = sub.compile()

    parent = StateGraph(State)
    parent.add_node("sub", subgraph)
    parent.add_edge(START, "sub")
    parent.add_edge("sub", END)
    graph = parent.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"result": ""}, config)
    pending = graph.pending_fetches(config)
    assert len(pending) == 1
    assert pending[0].value == {"resource": "inner-data"}

    out = graph.invoke(Command(fetch={pending[0].id: "SUBDATA"}), config)
    assert out["result"] == "SUBDATA"
    assert graph.pending_fetches(config) == []


async def test_e2e_async_fetch_inside_subgraph(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    async def inner(state):
        return {"result": fetch({"resource": "inner-data"})}

    sub = StateGraph(State)
    sub.add_node("inner", inner)
    sub.add_edge(START, "inner")
    sub.add_edge("inner", END)
    subgraph = sub.compile()

    parent = StateGraph(State)
    parent.add_node("sub", subgraph)
    parent.add_edge(START, "sub")
    parent.add_edge("sub", END)
    graph = parent.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke({"result": ""}, config)
    pending = await graph.apending_fetches(config)
    assert len(pending) == 1
    assert pending[0].value == {"resource": "inner-data"}
    out = await graph.afulfill(config, pending[0].id, "SUBDATA")
    assert out["result"] == "SUBDATA"


# ── ToolNode: fetch() inside an agent tool suspends and resumes (bubbles up) ─────


def test_e2e_fetch_inside_tool_node(sync_checkpointer: BaseCheckpointSaver) -> None:
    """A tool run by ToolNode may call fetch(); the GraphFetch bubbles up (ToolNode
    re-raises GraphBubbleUp), suspends the graph, and resumes with the fetched data."""
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    from langgraph.graph import MessagesState

    @tool
    def get_profile(user_id: str) -> str:
        """Fetch a user's profile from the profile service."""
        return fetch({"resource": "profile", "user_id": user_id})

    def agent(state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_profile",
                            "args": {"user_id": "u1"},
                            "id": "call_1",
                        }
                    ],
                )
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([get_profile]))
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", END)
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"messages": []}, config)
    pending = graph.pending_fetches(config)
    assert len(pending) == 1
    assert pending[0].value == {"resource": "profile", "user_id": "u1"}

    out = graph.invoke(Command(fetch={pending[0].id: "Ada Lovelace"}), config)
    last = out["messages"][-1]
    assert "Ada Lovelace" in last.content


async def test_e2e_async_fetch_inside_tool_node(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    from langgraph.graph import MessagesState

    @tool
    async def get_profile(user_id: str) -> str:
        """Fetch a user's profile from the profile service."""
        return fetch({"resource": "profile", "user_id": user_id})

    def agent(state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_profile",
                            "args": {"user_id": "u1"},
                            "id": "call_1",
                        }
                    ],
                )
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([get_profile]))
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", END)
    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke({"messages": []}, config)
    pending = await graph.apending_fetches(config)
    assert len(pending) == 1
    out = await graph.afulfill(config, pending[0].id, "Ada Lovelace")
    assert "Ada Lovelace" in out["messages"][-1].content


# ── provenance / audit: source, value_digest, resolved_at + resolved_fetches ─────


def test_e2e_resolved_fetches_records_provenance(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """After fulfillment, resolved_fetches() surfaces the audit record: digest,
    resolved_at, and the serving-layer-provided source."""
    builder = StateGraph(State)
    builder.add_node("n", lambda s: {"result": fetch({"clause": "penalty"})})
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"result": ""}, config)
    req = graph.pending_fetches(config)[0]
    # fulfill with provenance (where the value actually resolved from)
    graph.fulfill(config, req.id, "5% of contract value", source="cms://contract/v7")

    resolved = graph.resolved_fetches(config)
    assert len(resolved) == 1
    r = resolved[0]
    assert r.id == req.id
    assert r.status == "fulfilled"
    assert r.source == "cms://contract/v7"
    assert r.value_digest is not None  # content hash stamped
    assert r.resolved_at is not None  # point-in-time bound


def test_e2e_resolved_fetches_records_terminal_failure(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """A failed/expired dependency is also auditable via resolved_fetches()."""

    def node(state):
        try:
            return {"result": fetch({"r": "x"})}
        except FetchError as e:
            return {"result": f"handled:{e.status}"}

    builder = StateGraph(State)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"result": ""}, config)
    req = graph.pending_fetches(config)[0]
    graph.fail_fetch(config, req.id, "upstream 503")

    resolved = graph.resolved_fetches(config)
    assert len(resolved) == 1
    assert resolved[0].status == "failed"
    assert resolved[0].error == "upstream 503"
    assert resolved[0].resolved_at is not None


def test_e2e_value_digest_detects_drift(sync_checkpointer: BaseCheckpointSaver) -> None:
    """Reproducibility: two different resolved values yield different digests."""
    from langgraph.types import _value_digest

    assert _value_digest("5% of contract value") == _value_digest(
        "5% of contract value"
    )
    assert _value_digest("5% of contract value") != _value_digest(
        "7% of contract value"
    )


async def test_e2e_async_resolved_fetches_records_provenance(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    builder = StateGraph(State)
    builder.add_node("n", lambda s: {"result": fetch({"clause": "penalty"})})
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke({"result": ""}, config)
    req = (await graph.apending_fetches(config))[0]
    await graph.afulfill(config, req.id, "settled", source="ledger://tx/42")

    resolved = await graph.aresolved_fetches(config)
    assert len(resolved) == 1
    assert resolved[0].source == "ledger://tx/42"
    assert resolved[0].value_digest is not None
    assert resolved[0].resolved_at is not None
