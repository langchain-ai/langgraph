"""Tests for the fetch() / fetch_all() primitive and GraphFetch.

fetch() is a sibling of interrupt(): a service-to-service data dependency that
suspends the graph and is fulfilled by content-addressed id (never positionally).
"""

import time

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.errors import FetchError, GraphBubbleUp, GraphFetch, GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import (
    Command,
    FetchRequest,
    FetchResult,
    Interrupt,
    PregelTask,
    _fetch_id,
    fetch,
    fetch_all,
    interrupt,
)

pytestmark = pytest.mark.anyio


class State(TypedDict):
    result: str

    result: str


def _graph(node):
    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    return builder.compile(checkpointer=InMemorySaver())


# ── Exception hierarchy (GraphFetch is a sibling of GraphInterrupt) ──────────────


def test_graphfetch_is_bubbleup_not_interrupt():
    assert issubclass(GraphFetch, GraphBubbleUp)
    assert not issubclass(GraphFetch, GraphInterrupt)
    assert not issubclass(GraphInterrupt, GraphFetch)


def test_interrupt_has_no_kind_field():
    """interrupt() is untouched — no fetch-specific fields leak into Interrupt."""
    intr = Interrupt(value="please respond", id="abc")
    assert not hasattr(intr, "kind")


# ── content-addressed id ────────────────────────────────────────────────────────


def test_fetch_id_deterministic():
    assert _fetch_id("ns", {"a": 1}, None) == _fetch_id("ns", {"a": 1}, None)


def test_fetch_id_differs_by_request():
    assert _fetch_id("ns", {"a": 1}, None) != _fetch_id("ns", {"a": 2}, None)


def test_fetch_id_differs_by_ns():
    assert _fetch_id("ns1", {"a": 1}, None) != _fetch_id("ns2", {"a": 1}, None)


def test_fetch_id_explicit_key_overrides_payload():
    # same key -> same id even with different payloads (opt-in fan-out / dedup)
    assert _fetch_id("ns", {"a": 1}, "k") == _fetch_id("ns", {"a": 999}, "k")


# ── PregelTask.fetches ──────────────────────────────────────────────────────────


def test_pregeltask_fetches_field_default_empty():
    task = PregelTask(id="t1", name="node", path=())
    assert task.fetches == ()


def test_pregeltask_fetches_independent_of_interrupts():
    req = FetchRequest(id="f1", value={"resource": "txn"})
    task = PregelTask(
        id="t1",
        name="node",
        path=(),
        interrupts=(Interrupt(value="q?", id="h1"),),
        fetches=(req,),
    )
    assert task.fetches == (req,)
    assert len(task.interrupts) == 1


# ── fetch() suspend + fulfill ───────────────────────────────────────────────────


def test_fetch_suspends_with_request():
    graph = _graph(lambda s: {"result": fetch({"resource": "txn", "user_id": "1"})})
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"result": ""}, config)

    task = graph.get_state(config).tasks[0]
    assert len(task.fetches) == 1
    assert task.interrupts == ()
    req = task.fetches[0]
    assert isinstance(req, FetchRequest)
    assert req.value == {"resource": "txn", "user_id": "1"}


def test_fetch_fulfilled_by_id():
    graph = _graph(lambda s: {"result": fetch({"resource": "txn"})})
    config = {"configurable": {"thread_id": "t2"}}
    graph.invoke({"result": ""}, config)
    req = graph.get_state(config).tasks[0].fetches[0]

    out = graph.invoke(Command(fetch={req.id: "ACCOUNT_DATA"}), config)
    assert out["result"] == "ACCOUNT_DATA"


def test_fetch_requires_checkpointer_for_fulfillment():
    graph = _graph(lambda s: {"result": fetch({"r": "x"})})
    # no checkpointer would fail earlier; here just assert Command(fetch) validation
    with pytest.raises(ValueError):
        graph.invoke(
            Command(fetch={"not-a-valid-id": "v"}),
            {"configurable": {"thread_id": "tX"}},
        )


# ── fetch_all() batch + partial fulfillment ─────────────────────────────────────


def _fetch_all_node(state):
    a, b = fetch_all([{"r": "A"}, {"r": "B"}])
    return {"result": f"{a}:{b}"}


def test_fetch_all_suspends_with_all_requests():
    graph = _graph(_fetch_all_node)
    config = {"configurable": {"thread_id": "t3"}}
    graph.invoke({"result": ""}, config)
    fetches = graph.get_state(config).tasks[0].fetches
    assert len(fetches) == 2
    assert {f.value["r"] for f in fetches} == {"A", "B"}


def test_fetch_all_partial_fulfillment():
    graph = _graph(_fetch_all_node)
    config = {"configurable": {"thread_id": "t4"}}
    graph.invoke({"result": ""}, config)
    ids = {f.value["r"]: f.id for f in graph.get_state(config).tasks[0].fetches}

    # fulfill only A -> B still pending
    graph.invoke(Command(fetch={ids["A"]: "AA"}), config)
    still = graph.get_state(config).tasks[0].fetches
    assert len(still) == 1 and still[0].value["r"] == "B"

    # fulfill B -> completes, order preserved
    out = graph.invoke(Command(fetch={ids["B"]: "BB"}), config)
    assert out["result"] == "AA:BB"


# ── coexistence with interrupt() (no cross-resume) ──────────────────────────────


def _fetch_then_interrupt(state):
    data = fetch({"resource": "accounts"})
    answer = interrupt("what do you want to do?")
    return {"result": f"{data}:{answer}"}


def test_fetch_and_interrupt_coexist_no_cross_resume():
    graph = _graph(_fetch_then_interrupt)
    config = {"configurable": {"thread_id": "t5"}}

    # first invoke -> suspends on fetch (no human interrupt yet)
    graph.invoke({"result": ""}, config)
    task = graph.get_state(config).tasks[0]
    assert len(task.fetches) == 1
    assert len(task.interrupts) == 0

    # fulfilling the fetch does NOT satisfy the human interrupt: node advances
    # to interrupt(), and the stale fetch request is cleared
    graph.invoke(Command(fetch={task.fetches[0].id: "ACCT"}), config)
    task = graph.get_state(config).tasks[0]
    assert len(task.interrupts) == 1
    assert task.interrupts[0].value == "what do you want to do?"
    assert len(task.fetches) == 0

    # resuming the human interrupt completes the node
    out = graph.invoke(Command(resume="go"), config)
    assert out["result"] == "ACCT:go"


def test_resume_does_not_fulfill_fetch():
    """Command(resume=...) must not satisfy a data dependency — it re-suspends."""
    graph = _graph(lambda s: {"result": fetch({"r": "x"})})
    config = {"configurable": {"thread_id": "t6"}}
    graph.invoke({"result": ""}, config)
    req = graph.get_state(config).tasks[0].fetches[0]

    # a resume value is not a fetch fulfillment: the dependency stays pending
    out = graph.invoke(Command(resume="not-the-data"), config)
    assert out.get("result") in (None, "")
    still = graph.get_state(config).tasks[0].fetches
    assert len(still) == 1 and still[0].id == req.id

    # only fulfilling by id delivers the data
    out = graph.invoke(Command(fetch={req.id: "the-data"}), config)
    assert out["result"] == "the-data"


# ── interrupt() is unaffected ───────────────────────────────────────────────────


def test_interrupt_still_works():
    graph = _graph(lambda s: {"result": interrupt("please confirm")})
    config = {"configurable": {"thread_id": "t7"}}
    graph.invoke({"result": ""}, config)
    task = graph.get_state(config).tasks[0]
    assert len(task.interrupts) == 1
    assert task.interrupts[0].kind if hasattr(task.interrupts[0], "kind") else True
    assert task.fetches == ()

    out = graph.invoke(Command(resume="confirmed"), config)
    assert out["result"] == "confirmed"


def test_fetch_idempotent_across_reexecution():
    """A fulfilled dependency stays fulfilled if the node re-runs (re-suspends later)."""
    calls = {"n": 0}

    def node(state):
        calls["n"] += 1
        a = fetch({"step": 1})
        b = fetch({"step": 2})  # second, distinct dependency
        return {"result": f"{a}:{b}"}

    graph = _graph(node)
    config = {"configurable": {"thread_id": "t8"}}
    graph.invoke({"result": ""}, config)
    a_id = graph.get_state(config).tasks[0].fetches[0].id
    graph.invoke(Command(fetch={a_id: "first"}), config)
    # now suspended on the second fetch; first must remain fulfilled
    b_id = graph.get_state(config).tasks[0].fetches[0].id
    out = graph.invoke(Command(fetch={b_id: "second"}), config)
    assert out["result"] == "first:second"


# ── PR-2: deadline / SLA expiry, terminal failures (D5, D7, D10) ────────────────


def test_fetch_carries_deadline():
    graph = _graph(lambda s: {"result": fetch({"r": "x"}, deadline=1000)})
    config = {"configurable": {"thread_id": "d1"}}
    graph.invoke({"result": ""}, config)
    req = graph.get_state(config).tasks[0].fetches[0]
    assert req.deadline is not None
    assert req.created_at is not None


def test_fetch_no_deadline_never_expires():
    graph = _graph(lambda s: {"result": fetch({"r": "x"})})
    config = {"configurable": {"thread_id": "d2"}}
    graph.invoke({"result": ""}, config)
    req = graph.get_state(config).tasks[0].fetches[0]
    assert req.deadline is None
    # fulfilling much later still works (no SLA)
    out = graph.invoke(Command(fetch={req.id: "ok"}), config)
    assert out["result"] == "ok"


def test_deadline_fulfilled_in_time_succeeds():
    graph = _graph(lambda s: {"result": fetch({"r": "x"}, deadline=1000)})
    config = {"configurable": {"thread_id": "d3"}}
    graph.invoke({"result": ""}, config)
    req = graph.get_state(config).tasks[0].fetches[0]
    out = graph.invoke(Command(fetch={req.id: "in-time"}), config)
    assert out["result"] == "in-time"


def test_late_fulfillment_rejected_as_expired():
    """D5: past the deadline, a fulfillment is rejected and the node fails closed."""
    graph = _graph(lambda s: {"result": fetch({"r": "x"}, deadline=0.001)})
    config = {"configurable": {"thread_id": "d4"}}
    graph.invoke({"result": ""}, config)
    req = graph.get_state(config).tasks[0].fetches[0]
    time.sleep(0.05)  # let the deadline pass
    with pytest.raises(FetchError) as exc:
        graph.invoke(Command(fetch={req.id: "too-late"}), config)
    assert exc.value.status == "expired"
    assert exc.value.id == req.id


def test_lazy_expiry_on_resume_without_fulfillment():
    """D10: a resume past the deadline drives the fetch to a terminal expired failure."""
    graph = _graph(lambda s: {"result": fetch({"r": "x"}, deadline=0.001)})
    config = {"configurable": {"thread_id": "d5"}}
    graph.invoke({"result": ""}, config)
    time.sleep(0.05)
    # a bare resume (poke) re-runs the node, which self-expires
    with pytest.raises(FetchError) as exc:
        graph.invoke(Command(resume="anything"), config)
    assert exc.value.status == "expired"


def test_fail_fetch_raises_fetch_error():
    graph = _graph(lambda s: {"result": fetch({"r": "x"})})
    config = {"configurable": {"thread_id": "d6"}}
    graph.invoke({"result": ""}, config)
    req = graph.get_state(config).tasks[0].fetches[0]
    marker = FetchResult(id=req.id, status="failed", error="upstream 500")
    with pytest.raises(FetchError) as exc:
        graph.invoke(Command(fetch={req.id: marker}), config)
    assert exc.value.status == "failed"
    assert exc.value.error == "upstream 500"


def test_node_can_catch_fetch_error():
    """A node may handle a terminal fetch failure gracefully instead of crashing."""

    def node(state):
        try:
            data = fetch({"r": "x"}, deadline=0.001)
        except FetchError as e:
            return {"result": f"fallback:{e.status}"}
        return {"result": data}

    graph = _graph(node)
    config = {"configurable": {"thread_id": "d7"}}
    graph.invoke({"result": ""}, config)
    time.sleep(0.05)
    out = graph.invoke(Command(resume="poke"), config)
    assert out["result"] == "fallback:expired"


def test_expired_fetch_not_surfaced_as_pending():
    """Once expired (terminal), the request is no longer a pending fetch."""

    def node(state):
        try:
            return {"result": fetch({"r": "x"}, deadline=0.001)}
        except FetchError:
            return {"result": "done"}

    graph = _graph(node)
    config = {"configurable": {"thread_id": "d8"}}
    graph.invoke({"result": ""}, config)
    time.sleep(0.05)
    out = graph.invoke(Command(resume="poke"), config)
    assert out["result"] == "done"
    assert graph.get_state(config).tasks == () or all(
        t.fetches == () for t in graph.get_state(config).tasks
    )


# ── serving-layer API: pending_fetches / fulfill / fail_fetch / cancel / expire ──


def test_pending_fetches_and_fulfill_method():
    graph = _graph(lambda s: {"result": fetch({"r": "x"}, owner="billing")})
    config = {"configurable": {"thread_id": "m1"}}
    graph.invoke({"result": ""}, config)
    pending = graph.pending_fetches(config)
    assert len(pending) == 1
    assert pending[0].owner == "billing"
    out = graph.fulfill(config, pending[0].id, "DATA")
    assert out["result"] == "DATA"
    assert graph.pending_fetches(config) == []


def test_fail_fetch_method():
    graph = _graph(lambda s: {"result": fetch({"r": "x"})})
    config = {"configurable": {"thread_id": "m2"}}
    graph.invoke({"result": ""}, config)
    fid = graph.pending_fetches(config)[0].id
    with pytest.raises(FetchError) as exc:
        graph.fail_fetch(config, fid, "boom")
    assert exc.value.status == "failed" and exc.value.error == "boom"


def test_cancel_fetches_method():
    graph = _graph(lambda s: {"result": fetch({"r": "x"})})
    config = {"configurable": {"thread_id": "m3"}}
    graph.invoke({"result": ""}, config)
    with pytest.raises(FetchError) as exc:
        graph.cancel_fetches(config)
    assert exc.value.status == "cancelled"


def test_expire_fetches_method():
    def node(state):
        try:
            return {"result": fetch({"r": "x"}, deadline=0.001)}
        except FetchError as e:
            return {"result": f"exp:{e.status}"}

    graph = _graph(node)
    config = {"configurable": {"thread_id": "m4"}}
    graph.invoke({"result": ""}, config)
    time.sleep(0.05)
    due = graph.expire_fetches(config)
    assert len(due) == 1
    assert graph.get_state(config).values["result"] == "exp:expired"


def test_value_digest_recorded_on_fulfillment():
    """value_digest is stamped on the persisted FetchResult (readable via checkpointer)."""
    from langgraph._internal._constants import FETCH_RESULT

    saver = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node("node", lambda s: {"result": fetch({"r": "x"})})
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "m5"}}
    graph.invoke({"result": ""}, config)
    fid = graph.pending_fetches(config)[0].id
    graph.fulfill(config, fid, "DATA")
    # find the persisted FETCH_RESULT among the checkpoint writes
    results = {}
    for _, chan, val in saver.get_tuple(config).pending_writes or []:
        if chan == FETCH_RESULT and isinstance(val, dict):
            results.update(val)
    # after completion a new (clean) checkpoint may exist; walk history if needed
    if fid not in results:
        for snap in graph.get_state_history(config):
            tup = saver.get_tuple(snap.config)
            for _, chan, val in tup.pending_writes or []:
                if chan == FETCH_RESULT and isinstance(val, dict):
                    results.update(val)
    assert fid in results
    assert results[fid].status == "fulfilled"
    assert results[fid].value_digest is not None
    assert results[fid].resolved_at is not None


# ── background sweeper + async + subgraph nesting ───────────────────────────────


def test_fetch_sweeper_expires_idle_graph():
    """D10/D11: the daemon sweeper drives a never-resumed graph to terminal expired."""

    def node(state):
        try:
            return {"result": fetch({"r": "x"}, deadline=0.001)}
        except FetchError as e:
            return {"result": f"exp:{e.status}"}

    graph = _graph(node)
    config = {"configurable": {"thread_id": "sw1"}}
    graph.invoke({"result": ""}, config)
    graph.start_fetch_sweeper([config], interval_seconds=0.05)
    deadline = time.time() + 5
    try:
        while time.time() < deadline:
            if graph.get_state(config).values.get("result") == "exp:expired":
                break
            time.sleep(0.05)
    finally:
        graph.stop_fetch_sweeper()
    assert graph.get_state(config).values["result"] == "exp:expired"


async def test_async_fetch_serving_methods():
    graph = _graph(lambda s: {"result": fetch({"r": "x"})})
    config = {"configurable": {"thread_id": "as1"}}
    await graph.ainvoke({"result": ""}, config)
    pending = await graph.apending_fetches(config)
    assert len(pending) == 1
    out = await graph.afulfill(config, pending[0].id, "ADATA")
    assert out["result"] == "ADATA"


def test_fetch_in_subgraph():
    """A fetch declared inside a subgraph suspends and fulfills through the parent."""

    class Inner(TypedDict):
        result: str

    def inner_node(state):
        return {"result": fetch({"r": "sub"})}

    inner = StateGraph(Inner)
    inner.add_node("inner_node", inner_node)
    inner.add_edge(START, "inner_node")
    inner.add_edge("inner_node", END)

    parent = StateGraph(State)
    parent.add_node("sub", inner.compile())
    parent.add_edge(START, "sub")
    parent.add_edge("sub", END)
    graph = parent.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "sg1"}}

    graph.invoke({"result": ""}, config)
    pending = graph.pending_fetches(config)
    assert len(pending) == 1
    assert pending[0].value == {"r": "sub"}
    out = graph.invoke(Command(fetch={pending[0].id: "SUBDATA"}), config)
    assert out["result"] == "SUBDATA"
