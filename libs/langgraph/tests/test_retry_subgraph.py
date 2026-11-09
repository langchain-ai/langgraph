"""A retried node must reach the same subgraph checkpoints a later resume does.

A retry reruns a node from its first line, so each call to a subgraph inside it
has to land on the checkpoint namespace the previous attempt used. Otherwise the
retried attempt works on a namespace no later resume will look at, and what it
asked or computed there is lost.
"""

import sys
from typing import Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Durability, RetryPolicy, interrupt

pytestmark = pytest.mark.anyio
NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)

RETRY_ONCE = RetryPolicy(max_attempts=2, initial_interval=0, retry_on=RuntimeError)


class State(TypedDict, total=False):
    amount: int


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _ask_then_check(
    checkpointer: BaseCheckpointSaver,
) -> tuple[CompiledStateGraph, list[int]]:
    """A retried subgraph that asks a question and then calls a flaky service.

    The service fails on its first call only. Returns the graph and the list of
    amounts the service was called with.
    """
    calls: list[int] = []

    def ask(state: State) -> State:
        return {"amount": interrupt("How much?")}

    def check(state: State) -> State:
        calls.append(state["amount"])
        if len(calls) == 1:
            raise RuntimeError("service unavailable")
        return {"amount": state["amount"]}

    attempt = (
        StateGraph(State)
        .add_node("ask", ask)
        .add_node("check", check)
        .add_edge(START, "ask")
        .add_edge("ask", "check")
        .add_edge("check", END)
        .compile()
    )
    graph = (
        StateGraph(State)
        .add_node("attempt", attempt, retry_policy=RETRY_ONCE)
        .add_edge(START, "attempt")
        .add_edge("attempt", END)
        .compile(checkpointer=checkpointer)
    )
    return graph, calls


def _fail_in_second_node(
    checkpointer: BaseCheckpointSaver,
) -> tuple[CompiledStateGraph, list[str]]:
    """A retried subgraph whose second node fails once and needs no input."""
    ran: list[str] = []

    def first(state: State) -> State:
        ran.append("first")
        return {"amount": 1}

    def second(state: State) -> State:
        ran.append("second")
        if ran.count("second") == 1:
            raise RuntimeError("transient failure")
        return {"amount": 2}

    sub = (
        StateGraph(State)
        .add_node("first", first)
        .add_node("second", second)
        .add_edge(START, "first")
        .add_edge("first", "second")
        .add_edge("second", END)
        .compile()
    )
    graph = (
        StateGraph(State)
        .add_node("sub", sub, retry_policy=RETRY_ONCE)
        .add_edge(START, "sub")
        .add_edge("sub", END)
        .compile(checkpointer=checkpointer)
    )
    return graph, ran


def _two_subgraph_calls_then_fail(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """A retried node that calls one subgraph twice, then fails once.

    Each call asks its own question, so the two calls need separate checkpoint
    namespaces. The node fails after both calls have their answers.
    """
    failures = []

    def ask(state: State) -> State:
        return {"amount": interrupt("How much?")}

    inner = (
        StateGraph(State)
        .add_node("ask", ask)
        .add_edge(START, "ask")
        .add_edge("ask", END)
        .compile()
    )

    def both(state: State) -> State:
        tens = inner.invoke({})["amount"]
        units = inner.invoke({})["amount"]
        if not failures:
            failures.append(True)
            raise RuntimeError("fails after both answers")
        return {"amount": tens * 10 + units}

    return (
        StateGraph(State)
        .add_node("both", both, retry_policy=RETRY_ONCE)
        .add_edge(START, "both")
        .add_edge("both", END)
        .compile(checkpointer=checkpointer)
    )


def test_a_retried_subgraph_reuses_its_answer(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph, calls = _ask_then_check(sync_checkpointer)
    config = _config("reuse")

    first = graph.invoke({}, config, durability=durability)
    assert [i.value for i in first["__interrupt__"]] == ["How much?"]

    # The service fails once after the answer, the retry succeeds. The retried
    # subgraph must not ask the question again or drop what was answered.
    result = graph.invoke(Command(resume=5), config, durability=durability)

    assert "__interrupt__" not in result
    assert result == {"amount": 5}
    assert calls == [5, 5]


@NEEDS_CONTEXTVARS
async def test_a_retried_subgraph_reuses_its_answer_async(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph, calls = _ask_then_check(async_checkpointer)
    config = _config("reuse-async")

    first = await graph.ainvoke({}, config, durability=durability)
    assert [i.value for i in first["__interrupt__"]] == ["How much?"]

    result = await graph.ainvoke(Command(resume=5), config, durability=durability)

    assert "__interrupt__" not in result
    assert result == {"amount": 5}
    assert calls == [5, 5]


def test_a_retried_subgraph_resumes_from_the_node_that_failed(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph, ran = _fail_in_second_node(sync_checkpointer)

    result = graph.invoke({}, _config("resume-failed"), durability=durability)

    assert result == {"amount": 2}
    # `first` finished before `second` failed, so the retry does not repeat it.
    assert ran == ["first", "second", "second"]


@NEEDS_CONTEXTVARS
async def test_a_retried_subgraph_resumes_from_the_node_that_failed_async(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph, ran = _fail_in_second_node(async_checkpointer)

    result = await graph.ainvoke(
        {}, _config("resume-failed-async"), durability=durability
    )

    assert result == {"amount": 2}
    assert ran == ["first", "second", "second"]


# `durability="exit"` is left out here on purpose: it does not keep the first of
# two sequential subgraph calls that both interrupt, retry or no retry, so such
# a node never moves past its first question. That is a separate limitation.
@pytest.mark.parametrize("durability", ["sync", "async"])
def test_each_subgraph_call_in_a_retried_node_reuses_its_answer(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph = _two_subgraph_calls_then_fail(sync_checkpointer)
    config = _config("two-calls")

    result = graph.invoke({}, config, durability=durability)
    questions = 1
    for answer in (1, 2):
        assert [i.value for i in result["__interrupt__"]] == ["How much?"]
        result = graph.invoke(Command(resume=answer), config, durability=durability)
        if "__interrupt__" not in result:
            break
        questions += 1

    # One question per subgraph call. The retry after both answers must not
    # ask a third one.
    assert questions == 2
    assert result == {"amount": 12}


@NEEDS_CONTEXTVARS
@pytest.mark.parametrize("durability", ["sync", "async"])
async def test_each_subgraph_call_in_a_retried_node_reuses_its_answer_async(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph = _two_subgraph_calls_then_fail(async_checkpointer)
    config = _config("two-calls-async")

    result = await graph.ainvoke({}, config, durability=durability)
    questions = 1
    for answer in (1, 2):
        assert [i.value for i in result["__interrupt__"]] == ["How much?"]
        result = await graph.ainvoke(
            Command(resume=answer), config, durability=durability
        )
        if "__interrupt__" not in result:
            break
        questions += 1

    assert questions == 2
    assert result == {"amount": 12}
