import asyncio
from typing_extensions import TypedDict

import pytest

from langgraph.advanced_graph import AdvancedStateGraph
from langgraph.types import Command, Send

pytestmark = pytest.mark.anyio


class UpdateElisionState(TypedDict):
    x: int


async def test_noop_slow_update_does_not_override_fast_update() -> None:
    graph = AdvancedStateGraph(UpdateElisionState)

    async def start_node(state: UpdateElisionState) -> Command:
        return Command(goto=[Send("fast_node", None), Send("slow_node", None)])

    async def fast_node(state: UpdateElisionState) -> dict[str, int]:
        return {"x": 1}

    async def slow_node(state: UpdateElisionState) -> dict[str, int]:
        await asyncio.sleep(0.1)
        # Returns the same value as initial snapshot.
        return {"x": state["x"]}

    graph.add_entry_node(start_node)
    graph.add_node(fast_node)
    graph.add_finish_node(slow_node)

    result = await graph.compile().ainvoke({"x": 0})
    assert result["x"] == 1
