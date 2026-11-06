import asyncio
import logging
import operator
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime

pytestmark = pytest.mark.anyio


class State(TypedDict):
    log: Annotated[list[str], operator.add]


async def _call(state: State) -> dict[str, Any]:
    return {"log": ["call"]}


async def _slow_tool(state: State) -> dict[str, Any]:
    await asyncio.sleep(10)
    return {"log": ["result"]}


def _builder() -> StateGraph:
    return (
        StateGraph(State)
        .add_node("call", _call)
        .add_node("tool", _slow_tool)
        .add_edge(START, "call")
        .add_edge("call", "tool")
    )


async def _cancel_mid_tool(
    graph: Any, config: dict, value: list[str], durability: Any = None
) -> None:
    run = asyncio.create_task(
        graph.ainvoke({"log": value}, config, durability=durability)
    )
    await asyncio.sleep(0.1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run


@pytest.mark.parametrize("durability", ["sync", "async", "exit"])
async def test_cancel_handler_saves_update_after_cancel(durability: str) -> None:
    seen: list[Any] = []

    def on_cancel(state: State, runtime: Runtime) -> dict[str, Any]:
        seen.append((list(state["log"]), runtime.context))
        return {"log": ["cancelled"]}

    graph = (
        _builder().set_cancel_handler(on_cancel).compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "1"}}

    await _cancel_mid_tool(graph, config, ["h0"], durability)

    assert seen == [(["h0", "call"], None)]
    state = await graph.aget_state(config)
    assert state.values["log"] == ["h0", "call", "cancelled"]


async def test_async_cancel_handler_and_next_run() -> None:
    async def on_cancel(state: State, runtime: Runtime) -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"log": ["cancelled"]}

    graph = (
        _builder().set_cancel_handler(on_cancel).compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "1"}}

    await _cancel_mid_tool(graph, config, ["h0"])
    await _cancel_mid_tool(graph, config, ["h1"])

    state = await graph.aget_state(config)
    assert state.values["log"] == [
        "h0",
        "call",
        "cancelled",
        "h1",
        "call",
        "cancelled",
    ]


async def test_cancel_handler_returning_none_saves_nothing() -> None:
    graph = (
        _builder()
        .set_cancel_handler(lambda state, runtime: None)
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "1"}}

    await _cancel_mid_tool(graph, config, ["h0"])

    state = await graph.aget_state(config)
    assert state.values["log"] == ["h0", "call"]


async def test_cancel_handler_skipped_without_checkpointer() -> None:
    calls: list[State] = []
    graph = _builder().set_cancel_handler(lambda s, r: calls.append(s)).compile()

    await _cancel_mid_tool(graph, {}, ["h0"])

    assert calls == []


async def test_cancel_handler_timeout_is_logged_and_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def on_cancel(state: State, runtime: Runtime) -> dict[str, Any]:
        await asyncio.sleep(10)
        return {"log": ["cancelled"]}

    graph = (
        _builder()
        .set_cancel_handler(on_cancel, timeout=0.1)
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "1"}}

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(_cancel_mid_tool(graph, config, ["h0"]), timeout=2)

    assert "Cancel handler timed out" in caplog.text
    state = await graph.aget_state(config)
    assert state.values["log"] == ["h0", "call"]


async def test_cancel_handler_error_is_logged_and_cancel_propagates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def on_cancel(state: State, runtime: Runtime) -> dict[str, Any]:
        raise ValueError("boom")

    graph = (
        _builder().set_cancel_handler(on_cancel).compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "1"}}

    with caplog.at_level(logging.ERROR):
        await _cancel_mid_tool(graph, config, ["h0"])

    assert "Cancel handler failed" in caplog.text
    state = await graph.aget_state(config)
    assert state.values["log"] == ["h0", "call"]


async def test_cancel_handler_not_called_on_success_or_error() -> None:
    calls: list[State] = []

    async def fails(state: State) -> dict[str, Any]:
        raise ValueError("boom")

    graph = (
        StateGraph(State)
        .add_node("call", _call)
        .add_node("fails", fails)
        .add_edge(START, "call")
        .add_edge("call", "fails")
        .set_cancel_handler(lambda s, r: calls.append(s))
        .compile(checkpointer=InMemorySaver())
    )

    with pytest.raises(ValueError, match="boom"):
        await graph.ainvoke({"log": []}, {"configurable": {"thread_id": "1"}})

    assert calls == []


async def test_subgraph_cancel_handler_defers_to_parent() -> None:
    calls: list[str] = []

    subgraph = (
        _builder().set_cancel_handler(lambda s, r: calls.append("child")).compile()
    )
    graph = (
        StateGraph(State)
        .add_node("child", subgraph)
        .add_edge(START, "child")
        .set_cancel_handler(lambda s, r: calls.append("parent"))
        .compile(checkpointer=InMemorySaver())
    )

    await _cancel_mid_tool(graph, {"configurable": {"thread_id": "1"}}, ["h0"])

    assert calls == ["parent"]
