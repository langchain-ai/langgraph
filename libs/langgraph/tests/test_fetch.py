"""Tests for the fetch() primitive and Interrupt.kind field."""
import pytest
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, fetch, interrupt

pytestmark = pytest.mark.anyio


class State(TypedDict):
    result: str


def _graph_with_fetch():
    def node(state: State):
        data = fetch({"resource": "transactions", "user_id": "123"})
        return {"result": data}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    return builder.compile(checkpointer=InMemorySaver())


def _graph_with_both():
    def node(state: State):
        data = fetch({"resource": "accounts"})
        answer = interrupt("what do you want to do?")
        return {"result": f"{data}:{answer}"}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    return builder.compile(checkpointer=InMemorySaver())


# ── Interrupt.kind ────────────────────────────────────────────────────────────


def test_interrupt_kind_default_is_human():
    intr = Interrupt(value="please respond", id="abc")
    assert intr.kind == "human"


def test_interrupt_kind_fetch():
    intr = Interrupt(value={"resource": "transactions"}, id="abc", kind="fetch")
    assert intr.kind == "fetch"


def test_interrupt_from_ns_default_kind():
    intr = Interrupt.from_ns(value="hi", ns="some-ns")
    assert intr.kind == "human"


def test_interrupt_from_ns_fetch_kind():
    intr = Interrupt.from_ns(value={"resource": "accounts"}, ns="some-ns", kind="fetch")
    assert intr.kind == "fetch"


# ── PregelTask.fetches ────────────────────────────────────────────────────────


def test_pregeltask_fetches_filters_by_kind():
    from langgraph.types import PregelTask

    human_intr = Interrupt(value="question?", id="h1", kind="human")
    fetch_intr = Interrupt(value={"resource": "txn"}, id="f1", kind="fetch")
    task = PregelTask(
        id="t1", name="node", path=(), interrupts=(human_intr, fetch_intr)
    )
    assert task.fetches == (fetch_intr,)
    assert len(task.interrupts) == 2  # all still visible in interrupts


def test_pregeltask_fetches_empty_when_no_fetch():
    from langgraph.types import PregelTask

    task = PregelTask(
        id="t1",
        name="node",
        path=(),
        interrupts=(Interrupt(value="hi", id="h1", kind="human"),),
    )
    assert task.fetches == ()


# ── fetch() function ──────────────────────────────────────────────────────────


def test_fetch_suspends_with_kind_fetch():
    graph = _graph_with_fetch()
    config = {"configurable": {"thread_id": "t1"}}

    graph.invoke({"result": ""}, config)

    snapshot = graph.get_state(config)
    assert len(snapshot.tasks) == 1
    task = snapshot.tasks[0]

    # task.fetches returns only fetch-kind interrupts
    assert len(task.fetches) == 1
    req = task.fetches[0]
    assert req.kind == "fetch"
    assert req.value == {"resource": "transactions", "user_id": "123"}

    # all interrupts also contains it
    assert req in task.interrupts


def test_fetch_resumes_with_data():
    graph = _graph_with_fetch()
    config = {"configurable": {"thread_id": "t2"}}

    graph.invoke({"result": ""}, config)
    result = graph.invoke(Command(resume="account_data"), config)
    assert result["result"] == "account_data"


def test_fetch_kind_does_not_appear_in_human_interrupts():
    """Serving layer can filter: task.interrupts where kind == human."""
    graph = _graph_with_fetch()
    config = {"configurable": {"thread_id": "t3"}}

    graph.invoke({"result": ""}, config)
    snapshot = graph.get_state(config)
    task = snapshot.tasks[0]

    human_interrupts = [i for i in task.interrupts if i.kind == "human"]
    assert human_interrupts == []


def test_interrupt_still_defaults_to_human_kind():
    """Existing interrupt() calls are unaffected — kind defaults to human."""

    def node(state: State):
        answer = interrupt("please confirm")
        return {"result": answer}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t4"}}

    graph.invoke({"result": ""}, config)
    snapshot = graph.get_state(config)
    task = snapshot.tasks[0]

    assert len(task.interrupts) == 1
    assert task.interrupts[0].kind == "human"
    assert task.fetches == ()


def test_fetch_and_interrupt_coexist_in_same_node():
    """A node can mix fetch() and interrupt() — task.fetches separates them."""
    graph = _graph_with_both()
    config = {"configurable": {"thread_id": "t5"}}

    # First invoke — hits fetch(), suspends
    graph.invoke({"result": ""}, config)
    snapshot = graph.get_state(config)
    task = snapshot.tasks[0]
    assert len(task.fetches) == 1
    assert task.fetches[0].value == {"resource": "accounts"}

    # Resume fetch — hits interrupt(), suspends
    graph.invoke(Command(resume="account_data"), config)
    snapshot = graph.get_state(config)
    task = snapshot.tasks[0]
    human_interrupts = [i for i in task.interrupts if i.kind == "human"]
    assert len(human_interrupts) == 1
    assert human_interrupts[0].value == "what do you want to do?"

    # Resume human interrupt — completes
    result = graph.invoke(Command(resume="nothing"), config)
    assert result["result"] == "account_data:nothing"
