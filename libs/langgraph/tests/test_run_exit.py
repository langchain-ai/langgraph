"""Tests for `Runtime.on_exit`: cleanup that runs however a graph run ends."""

from __future__ import annotations

import asyncio
import sys

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime, get_runtime
from langgraph.types import Command, interrupt

pytestmark = pytest.mark.anyio

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


class State(TypedDict, total=False):
    value: str


def _two_node_graph(first, second=None, *, checkpointer=None):
    builder = StateGraph(State)
    builder.add_node("first", first)
    builder.add_edge(START, "first")
    if second is None:
        builder.add_edge("first", END)
    else:
        builder.add_node("second", second)
        builder.add_edge("first", "second")
        builder.add_edge("second", END)
    return builder.compile(checkpointer=checkpointer)


# sync


def test_runs_on_success_in_lifo_order() -> None:
    events: list[str] = []

    def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(lambda: events.append("cleanup-a"))
        runtime.on_exit(lambda: events.append("cleanup-b"))
        events.append("first")
        return {"value": "x"}

    def second(state: State) -> State:
        events.append("second")
        return {}

    graph = _two_node_graph(first, second)
    assert graph.invoke({}) == {"value": "x"}
    assert events == ["first", "second", "cleanup-b", "cleanup-a"]


def test_runs_on_error_and_keeps_the_run_exception() -> None:
    events: list[str] = []

    def failing_cleanup() -> None:
        events.append("failing")
        raise KeyError("cleanup failed")

    def first(state: State) -> State:
        runtime = get_runtime()
        runtime.on_exit(lambda: events.append("cleanup"))
        runtime.on_exit(failing_cleanup)
        raise ValueError("node failed")

    graph = _two_node_graph(first)
    with pytest.raises(ValueError, match="node failed"):
        graph.invoke({})
    # the failing callback did not stop the earlier one from running
    assert events == ["failing", "cleanup"]


def test_callback_error_is_raised_when_the_run_succeeded() -> None:
    def failing_cleanup() -> None:
        raise KeyError("cleanup failed")

    def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(failing_cleanup)
        return {}

    graph = _two_node_graph(first)
    with pytest.raises(KeyError, match="cleanup failed"):
        graph.invoke({})


def test_runs_on_interrupt() -> None:
    events: list[str] = []

    def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(lambda: events.append("cleanup"))
        return {"value": interrupt("approve?")}

    graph = _two_node_graph(first, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({}, config)
    assert "__interrupt__" in result
    assert events == ["cleanup"]

    # each resumed run is its own invocation with its own callbacks
    graph.invoke(Command(resume="ok"), config)
    assert events == ["cleanup", "cleanup"]


def test_runs_when_the_stream_is_closed_early() -> None:
    events: list[str] = []

    def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(lambda: events.append("cleanup"))
        return {"value": "x"}

    def second(state: State) -> State:
        events.append("second")
        return {}

    graph = _two_node_graph(first, second)
    stream = graph.stream({}, stream_mode="updates")
    assert next(stream) == {"first": {"value": "x"}}
    assert events == []
    stream.close()
    assert events == ["cleanup"]


def test_runs_when_breaking_out_of_stream() -> None:
    events: list[str] = []

    def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(lambda: events.append("cleanup"))
        return {"value": "x"}

    graph = _two_node_graph(first, lambda s: {})
    for _ in graph.stream({}, stream_mode="updates"):
        break
    # the dropped generator is closed by refcounting on CPython
    assert events == ["cleanup"]


def test_subgraph_registers_on_the_root_run() -> None:
    events: list[str] = []

    def inner(state: State, runtime: Runtime) -> State:
        runtime.on_exit(lambda: events.append("cleanup"))
        events.append("inner")
        return {}

    sub = StateGraph(State)
    sub.add_node("inner", inner)
    sub.add_edge(START, "inner")
    subgraph = sub.compile()

    def call_subgraph(state: State) -> State:
        subgraph.invoke(state)
        events.append("subgraph returned")
        return {}

    def after(state: State) -> State:
        events.append("after")
        return {}

    builder = StateGraph(State)
    builder.add_node("as_node", subgraph)
    builder.add_node("call_subgraph", call_subgraph)
    builder.add_node("after", after)
    builder.add_edge(START, "as_node")
    builder.add_edge("as_node", "call_subgraph")
    builder.add_edge("call_subgraph", "after")
    graph = builder.compile()

    graph.invoke({})
    # nothing runs when either subgraph ends; both run when the root ends
    assert events == [
        "inner",
        "inner",
        "subgraph returned",
        "after",
        "cleanup",
        "cleanup",
    ]


def test_async_callback_rejected_in_sync_run() -> None:
    async def acleanup() -> None:
        pass

    def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(acleanup)
        return {}

    graph = _two_node_graph(first)
    with pytest.raises(TypeError, match="async callback in a synchronous"):
        graph.invoke({})


def test_outside_a_run_raises() -> None:
    with pytest.raises(RuntimeError, match="during a graph run"):
        Runtime().on_exit(lambda: None)


def test_each_invocation_gets_its_own_callbacks() -> None:
    calls: list[int] = []

    def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(lambda: calls.append(1))
        return {}

    graph = _two_node_graph(first)
    graph.invoke({})
    graph.invoke({})
    assert calls == [1, 1]


# async


async def test_async_runs_on_success_sync_and_async_callbacks() -> None:
    events: list[str] = []

    async def acleanup() -> None:
        await asyncio.sleep(0)
        events.append("async-cleanup")

    async def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(lambda: events.append("sync-cleanup"))
        runtime.on_exit(acleanup)
        events.append("first")
        return {"value": "x"}

    graph = _two_node_graph(first)
    assert await graph.ainvoke({}) == {"value": "x"}
    assert events == ["first", "async-cleanup", "sync-cleanup"]


async def test_async_runs_on_error_and_keeps_the_run_exception() -> None:
    events: list[str] = []

    async def failing_cleanup() -> None:
        raise KeyError("cleanup failed")

    async def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(lambda: events.append("cleanup"))
        runtime.on_exit(failing_cleanup)
        raise ValueError("node failed")

    graph = _two_node_graph(first)
    with pytest.raises(ValueError, match="node failed"):
        await graph.ainvoke({})
    assert events == ["cleanup"]


async def test_async_runs_on_interrupt() -> None:
    events: list[str] = []

    async def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(lambda: events.append("cleanup"))
        return {"value": interrupt("approve?")}

    graph = _two_node_graph(first, checkpointer=InMemorySaver())
    result = await graph.ainvoke({}, {"configurable": {"thread_id": "1"}})
    assert "__interrupt__" in result
    assert events == ["cleanup"]


async def test_async_runs_on_cancel_after_the_node_unwinds() -> None:
    events: list[str] = []
    started = asyncio.Event()

    async def acleanup() -> None:
        events.append("cleanup")

    async def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(acleanup)
        started.set()
        try:
            await asyncio.sleep(60)
        finally:
            events.append("node finally")
        return {}

    graph = _two_node_graph(first)
    task = asyncio.create_task(graph.ainvoke({}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # cleanup ran after the node had stopped
    assert events == ["node finally", "cleanup"]


async def test_async_runs_when_the_stream_is_closed_early() -> None:
    events: list[str] = []

    async def acleanup() -> None:
        events.append("cleanup")

    async def first(state: State, runtime: Runtime) -> State:
        runtime.on_exit(acleanup)
        return {"value": "x"}

    async def second(state: State) -> State:
        events.append("second")
        return {}

    graph = _two_node_graph(first, second)
    stream = graph.astream({}, stream_mode="updates")
    assert await stream.__anext__() == {"first": {"value": "x"}}
    assert events == []
    await stream.aclose()
    assert events == ["cleanup"]


@NEEDS_CONTEXTVARS
async def test_async_subgraph_registers_on_the_root_run() -> None:
    events: list[str] = []

    async def inner(state: State) -> State:
        get_runtime().on_exit(lambda: events.append("cleanup"))
        return {}

    sub = StateGraph(State)
    sub.add_node("inner", inner)
    sub.add_edge(START, "inner")
    subgraph = sub.compile()

    async def call_subgraph(state: State) -> State:
        await subgraph.ainvoke(state)
        events.append("subgraph returned")
        return {}

    graph = _two_node_graph(call_subgraph, lambda s: events.append("after") or {})
    await graph.ainvoke({})
    assert events == ["subgraph returned", "after", "cleanup"]
