import asyncio

import pytest
from typing_extensions import TypedDict

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class State(TypedDict, total=False):
    x: str
    y: str


def _build_graph(parallel: bool):
    handler_calls = {"count": 0}

    def boom(state: State) -> State:
        raise RuntimeError("test")

    def sibling(state: State) -> State:
        return {"y": "sibling"}

    def handler(state: State, error: NodeError) -> Command:
        handler_calls["count"] += 1
        return Command(update={"x": "handled"}, goto=END)

    builder = (
        StateGraph(State)
        .add_node("boom", boom, error_handler=handler)
        .add_edge(START, "boom")
    )
    if parallel:
        builder = builder.add_node("sibling", sibling).add_edge(START, "sibling")
    return builder.compile(), handler_calls


async def _collect(graph, **kwargs):
    return [event async for event in graph.astream({}, **kwargs)]


SYNC_CASES = [
    pytest.param(
        False,
        lambda g: list(g.stream({}, stream_mode="custom")),
        id="solo-custom",
    ),
    pytest.param(
        False,
        lambda g: list(g.stream({}, stream_mode="messages")),
        id="solo-messages",
    ),
    pytest.param(
        False,
        lambda g: list(g.stream({}, stream_mode="values", subgraphs=True)),
        id="solo-subgraphs",
    ),
    pytest.param(True, lambda g: g.invoke({}), id="parallel-invoke"),
    pytest.param(
        True,
        lambda g: list(g.stream({}, stream_mode="values")),
        id="parallel-values",
    ),
]

ASYNC_CASES = [
    pytest.param(
        False,
        lambda g: asyncio.run(_collect(g, stream_mode="custom")),
        id="solo-astream-custom",
    ),
    pytest.param(
        True,
        lambda g: asyncio.run(g.ainvoke({})),
        id="parallel-ainvoke",
    ),
]


@pytest.mark.parametrize("parallel,run", SYNC_CASES)
def test_error_handler_suppresses_node_error_sync(parallel, run) -> None:
    graph, handler_calls = _build_graph(parallel)
    # Must not raise: the error handler handled the node's exception.
    run(graph)
    assert handler_calls["count"] == 1


@pytest.mark.parametrize("parallel,run", ASYNC_CASES)
def test_error_handler_suppresses_node_error_async(parallel, run) -> None:
    graph, handler_calls = _build_graph(parallel)
    # Must not raise: the error handler handled the node's exception.
    run(graph)
    assert handler_calls["count"] == 1
