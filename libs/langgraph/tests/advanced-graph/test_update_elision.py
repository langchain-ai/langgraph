import asyncio
from dataclasses import dataclass
from pydantic import BaseModel
from typing_extensions import TypedDict

import pytest

from langgraph.advanced_graph import AdvancedStateGraph, CompiledGraphEngine
from langgraph.types import Command, Send

pytestmark = pytest.mark.anyio


@dataclass(frozen=True)
class DataClassPayload:
    value: int


class PydanticPayload(BaseModel):
    value: int


class InnerTypedDict(TypedDict):
    flag: bool
    n: int


class UpdateElisionState(TypedDict):
    x: int
    dc: DataClassPayload
    model: PydanticPayload
    td: InnerTypedDict
    obj: dict[str, int]
    items: list[int]


def _initial_state() -> UpdateElisionState:
    return {
        "x": 0,
        "dc": DataClassPayload(0),
        "model": PydanticPayload(value=0),
        "td": {"flag": False, "n": 0},
        "obj": {"n": 0},
        "items": [0],
    }



async def test_noop_slow_update_does_not_override_fast_update() -> None:
    graph: AdvancedStateGraph[UpdateElisionState] = AdvancedStateGraph(UpdateElisionState)

    async def start_node(state: UpdateElisionState) -> Command:
        return Command(goto=[Send("fast_node", None), Send("slow_node", None)])

    async def fast_node(state: UpdateElisionState) -> UpdateElisionState:
        state.x = 1
        state.dc.value = 1
        state.model.value = 1
        state.td["flag"] = True
        state.td["n"] = 1
        state.obj["n"] = 1
        state.items.append(1)
        return state

    async def slow_node(state: UpdateElisionState) -> UpdateElisionState:
        await asyncio.sleep(0.1)
        # Returns the same values as the initial snapshot.
        return state

    graph.add_entry_node(start_node)
    graph.add_node(fast_node)
    graph.add_finish_node(slow_node)

    compiled: CompiledGraphEngine[UpdateElisionState] = graph.compile()
    initial_state: UpdateElisionState = _initial_state()
    result: UpdateElisionState = await compiled.ainvoke(initial_state)
    assert result["x"] == 1
    assert result["dc"] == DataClassPayload(1)
    assert result["model"].value == 1
    assert result["td"] == {"flag": True, "n": 1}
    assert result["obj"] == {"n": 1}
    assert result["items"] == [1]


async def test_changed_slow_update_overrides_fast_update() -> None:
    graph: AdvancedStateGraph[UpdateElisionState] = AdvancedStateGraph(UpdateElisionState)

    async def start_node(state: UpdateElisionState) -> Command:
        return Command(goto=[Send("fast_node", None), Send("slow_node", None)])

    async def fast_node(state: UpdateElisionState) -> UpdateElisionState:
        state.x = 1
        state.dc.value = 1
        state.model.value = 1
        state.td["flag"] = True
        state.td["n"] = 1
        state.obj["n"] = 1
        state.items.append(1)
        return state

    async def slow_node(state: UpdateElisionState) -> UpdateElisionState:
        await asyncio.sleep(0.1)
        # Slow node makes real changes for all field types.
        state.x = 2
        state.dc.value = 2
        state.model.value = 2
        state.td["flag"] = False
        state.td["n"] = 2
        state.obj["n"] = 2
        state.items.append(2)
        return state

    graph.add_entry_node(start_node)
    graph.add_node(fast_node)
    graph.add_finish_node(slow_node)

    compiled: CompiledGraphEngine[UpdateElisionState] = graph.compile()
    initial_state: UpdateElisionState = _initial_state()
    result: UpdateElisionState = await compiled.ainvoke(initial_state)
    assert result["x"] == 2
    assert result["dc"] == DataClassPayload(2)
    assert result["model"].value == 2
    assert result["td"] == {"flag": False, "n": 2}
    assert result["obj"] == {"n": 2}
    assert result["items"] == [2]
