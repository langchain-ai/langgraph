"""Regression tests for awaiting a `@task` inside a `StateGraph` node while streaming.

`arun_with_retry` consumes `astream` instead of `ainvoke` when the runner is in
streaming mode (as it is under `astream_events`). A task's result is written to the
`RETURN` channel rather than returned by `astream`, so awaiting the task future used
to yield `None` (#9284) even though `ainvoke` returned the real value.
"""

import asyncio

from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.func import task
from langgraph.graph import START, StateGraph


class State(TypedDict):
    out: int | None


@task
async def double(value: int) -> int:
    return value * 2


@task
async def nothing() -> None:
    return None


@task(timeout=30)
async def double_with_timeout(value: int) -> int:
    return value * 2


def _compile(node_fn):
    builder = StateGraph(State)
    builder.add_node("node", node_fn)
    builder.add_edge(START, "node")
    return builder.compile(checkpointer=InMemorySaver())


async def _astream_events_output(graph, thread_id: str):
    """Final `LangGraph` output reported through `astream_events`."""
    async for event in graph.astream_events(
        {"out": None}, {"configurable": {"thread_id": thread_id}}, version="v2"
    ):
        if event["event"] == "on_chain_end" and event["name"] == "LangGraph":
            return event["data"]["output"]
    return None


def test_task_awaited_in_node_under_astream_events() -> None:
    """The awaited task must resolve to its result, not to `None`."""

    async def node(state: State) -> dict:
        return {"out": await double(21)}

    graph = _compile(node)

    async def run() -> None:
        assert await graph.ainvoke(
            {"out": None}, {"configurable": {"thread_id": "invoke"}}
        ) == {"out": 42}
        assert await _astream_events_output(graph, "events") == {"out": 42}, (
            "task resolved to None under astream_events"
        )

    asyncio.run(run())


def test_task_returning_none_under_astream_events() -> None:
    """A task that really returns `None` keeps returning `None`."""

    async def node(state: State) -> dict:
        return {"out": await nothing()}

    graph = _compile(node)

    async def run() -> None:
        assert await _astream_events_output(graph, "none") == {"out": None}

    asyncio.run(run())


def test_task_with_timeout_under_astream_events() -> None:
    """Same, through the timeout-aware streaming path."""

    async def node(state: State) -> dict:
        return {"out": await double_with_timeout(21)}

    graph = _compile(node)

    async def run() -> None:
        assert await _astream_events_output(graph, "timeout") == {"out": 42}

    asyncio.run(run())
