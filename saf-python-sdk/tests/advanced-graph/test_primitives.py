import pytest
from typing_extensions import TypedDict

from saf_python_sdk.advanced_graph import AdvancedStateGraph, Context, channel_condition
from saf_python_sdk.types import Command, Send

pytestmark = pytest.mark.anyio


class PrimitiveState(TypedDict):
    counter: int
    logs: list[str]
    done: str | None


async def test_input_and_state_primitives_are_compatible() -> None:
    graph = AdvancedStateGraph(PrimitiveState)

    async def start_node(state: PrimitiveState) -> Command:
        state["logs"].append(f"start:counter={state['counter']}")
        return Command(goto=Send("middle_node", "from_start"))

    async def middle_node(ctx: Context, tool_input: str, state: PrimitiveState) -> Command:
        state["logs"].append(f"middle:input={tool_input}")
        return Command(update=state, goto=Send("finish_node", "from_middle"))

    async def finish_node(payload: str, state: PrimitiveState) -> dict[str, object]:
        state["logs"].append(f"finish:input={payload}")
        return {
            "logs": state["logs"],
            "counter": state["counter"],
            "done": payload,
        }

    graph.add_entry_node(start_node)
    graph.add_node(middle_node)
    graph.add_finish_node(finish_node)

    result = await graph.compile().ainvoke({"counter": 7, "logs": [], "done": None})
    assert result["counter"] == 7
    assert result["done"] == "from_middle"
    assert result["logs"] == [
        "start:counter=7",
        "middle:input=from_start",
        "finish:input=from_middle",
    ]


async def test_run_ends_without_finish_node() -> None:
    graph = AdvancedStateGraph(PrimitiveState)

    async def start_node(state: PrimitiveState) -> Command:
        state["logs"].append("start")
        return Command(update=state, goto=Send("middle_node", "from_start"))

    async def middle_node(input: str, state: PrimitiveState) -> dict[str, object]:
        state["logs"].append(f"middle:{input}")
        return {"counter": state["counter"] + 1, "logs": state["logs"], "done": "stopped"}

    graph.add_entry_node(start_node)
    graph.add_node(middle_node)

    result = await graph.compile().ainvoke({"counter": 7, "logs": [], "done": None})
    assert result["counter"] == 8
    assert result["done"] == "stopped"
    assert result["logs"] == ["start", "middle:from_start"]


async def test_channel_wait_respects_max_m() -> None:
    graph = AdvancedStateGraph(PrimitiveState)
    graph.add_async_channel("events", list[str])

    async def start_node(ctx: Context, state: PrimitiveState) -> Command:
        ctx.publish_to_channel("events", "a")
        ctx.publish_to_channel("events", "b")
        ctx.publish_to_channel("events", "c")
        return Command(update=state, goto=Send("wait_node", None))

    async def wait_node(ctx: Context, _input: None, state: PrimitiveState) -> dict[str, object]:
        event = await ctx.wait_for(channel_condition("events", min=2, max=4))
        values = event["value"]
        assert isinstance(values, list)
        return {"counter": len(values), "logs": values, "done": "ok"}

    graph.add_entry_node(start_node)
    graph.add_finish_node(wait_node)

    result = await graph.compile().ainvoke({"counter": 0, "logs": [], "done": None})
    assert result["counter"] == 3
    assert result["logs"] == ["a", "b", "c"]
    assert result["done"] == "ok"

