"""`DeltaChannel` replay must apply parallel writes in the order `invoke` did."""

from typing import Annotated, Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph

pytestmark = pytest.mark.anyio

# Sorted, because live execution applies PULL tasks in node-name order.
FAN_OUT_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]


def _append_reducer(current: list, updates: list) -> list:
    return [*current, *(x for u in updates for x in u)]


def _build_fan_out_graph(checkpointer: BaseCheckpointSaver) -> Any:
    class State(TypedDict):
        items: Annotated[
            list, DeltaChannel(_append_reducer, list, snapshot_frequency=10_000)
        ]

    def make_node(label: str) -> Any:
        return lambda state: {"items": [label]}

    builder = StateGraph(State)
    for name in FAN_OUT_NAMES:
        builder.add_node(name, make_node(name))
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    return builder.compile(checkpointer=checkpointer)


async def test_get_state_matches_live_invoke_order(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _build_fan_out_graph(async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    live = (await graph.ainvoke({"items": []}, config))["items"]
    replayed = (await graph.aget_state(config)).values["items"]

    assert live == FAN_OUT_NAMES
    assert replayed == live


async def test_continuing_thread_preserves_committed_prefix(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _build_fan_out_graph(async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    first = (await graph.ainvoke({"items": []}, config))["items"]
    second = (await graph.ainvoke({"items": []}, config))["items"]

    assert second == first + first
    assert (await graph.aget_state(config)).values["items"] == second


async def test_state_history_reports_live_order_at_every_step(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    runs = 3
    graph = _build_fan_out_graph(async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    for _ in range(runs):
        await graph.ainvoke({"items": []}, config)
    live = FAN_OUT_NAMES * runs

    seen = [
        s.values["items"]
        async for s in graph.aget_state_history(config)
        if "items" in s.values
    ]

    assert max(map(len, seen)) == len(live)
    for values in seen:
        assert values == live[: len(values)], f"{values} is not a prefix of {live}"
