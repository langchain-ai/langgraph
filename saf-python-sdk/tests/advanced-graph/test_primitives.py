import asyncio
import time
import pytest
from typing_extensions import TypedDict

from saf_python_sdk.advanced_graph import (
    AdvancedStateGraph,
    Context,
    all_of,
    any_of,
    channel_condition,
    timer_condition,
)
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
        result = await ctx.wait_for(channel_condition("events", min=2, max=4))
        values = result.conditions[0].values or []
        assert isinstance(values, list)
        return {"counter": len(values), "logs": values, "done": "ok"}

    graph.add_entry_node(start_node)
    graph.add_finish_node(wait_node)

    result = await graph.compile().ainvoke({"counter": 0, "logs": [], "done": None})
    assert result["counter"] == 3
    assert result["logs"] == ["a", "b", "c"]
    assert result["done"] == "ok"


async def test_any_of_consumes_all_ready_channels() -> None:
    graph = AdvancedStateGraph(PrimitiveState)
    graph.add_async_channel("alpha", str)
    graph.add_async_channel("beta", str)

    async def start_node(ctx: Context, state: PrimitiveState) -> Command:
        ctx.publish_to_channel("alpha", "a1")
        ctx.publish_to_channel("beta", "b1")
        return Command(update=state, goto=Send("wait_node", None))

    async def wait_node(ctx: Context, _input: None, state: PrimitiveState) -> Command:
        first = await ctx.wait_for(
            any_of(channel_condition("alpha"), channel_condition("beta"))
        )
        assert len(first.conditions) == 2
        assert first.conditions[0].met is True
        assert first.conditions[0].channel_name == "alpha"
        assert first.conditions[0].values == ["a1"]
        assert first.conditions[1].met is True
        assert first.conditions[1].channel_name == "beta"
        assert first.conditions[1].values == ["b1"]
        ctx.publish_to_channel("beta", "b2")
        return Command(
            update={"counter": 1, "logs": ["matched=2"], "done": None},
            goto=Send("verify_node", None),
        )

    async def verify_node(
        ctx: Context, _input: None, state: PrimitiveState
    ) -> dict[str, object]:
        second = await ctx.wait_for(channel_condition("beta"))
        values = second.conditions[0].values or []
        return {
            "counter": 2,
            "logs": [*state["logs"], f"beta={values[0]}"],
            "done": "ok",
        }

    graph.add_entry_node(start_node)
    graph.add_node(wait_node)
    graph.add_finish_node(verify_node)

    result = await graph.compile().ainvoke({"counter": 0, "logs": [], "done": None})
    assert result["counter"] == 2
    assert result["logs"] == [
        "matched=2",
        "beta=b2",
    ]
    assert result["done"] == "ok"


async def test_all_of_waits_until_all_channels_are_ready() -> None:
    graph = AdvancedStateGraph(PrimitiveState)
    graph.add_async_channel("alpha", str)
    graph.add_async_channel("beta", str)

    async def start_node(ctx: Context, state: PrimitiveState) -> Command:
        ctx.publish_to_channel("alpha", "a1")
        return Command(
            update=state,
            goto=[Send("wait_node", None), Send("publish_beta_node", None)],
        )

    async def publish_beta_node(ctx: Context, _input: None, state: PrimitiveState) -> Command:
        await ctx.wait_for(timer_condition(seconds=0.02))
        ctx.publish_to_channel("beta", "b1")
        return Command(update=state)

    async def wait_node(ctx: Context, _input: None, state: PrimitiveState) -> dict[str, object]:
        waited = await ctx.wait_for(
            all_of(channel_condition("alpha"), channel_condition("beta"))
        )
        assert len(waited.conditions) == 2
        assert waited.conditions[0].met is True
        assert waited.conditions[0].channel_name == "alpha"
        assert waited.conditions[0].values == ["a1"]
        assert waited.conditions[1].met is True
        assert waited.conditions[1].channel_name == "beta"
        assert waited.conditions[1].values == ["b1"]
        return {"counter": 1, "logs": ["all_of_channels"], "done": "ok"}

    graph.add_entry_node(start_node)
    graph.add_node(publish_beta_node)
    graph.add_finish_node(wait_node)

    result = await graph.compile().ainvoke({"counter": 0, "logs": [], "done": None})
    assert result["counter"] == 1
    assert result["logs"] == ["all_of_channels"]
    assert result["done"] == "ok"


async def test_all_of_channel_and_timer_marks_both_conditions() -> None:
    graph = AdvancedStateGraph(PrimitiveState)
    graph.add_async_channel("alpha", str)

    async def start_node(ctx: Context, state: PrimitiveState) -> Command:
        ctx.publish_to_channel("alpha", "a1")
        return Command(update=state, goto=Send("wait_node", None))

    async def wait_node(ctx: Context, _input: None, state: PrimitiveState) -> dict[str, object]:
        waited = await ctx.wait_for(
            all_of(channel_condition("alpha"), timer_condition(seconds=0.02))
        )
        assert len(waited.conditions) == 2
        assert waited.conditions[0].met is True
        assert waited.conditions[0].channel_name == "alpha"
        assert waited.conditions[0].values == ["a1"]
        assert waited.conditions[1].met is True
        return {"counter": 1, "logs": ["all_of_channel_timer"], "done": "ok"}

    graph.add_entry_node(start_node)
    graph.add_finish_node(wait_node)

    result = await graph.compile().ainvoke({"counter": 0, "logs": [], "done": None})
    assert result["counter"] == 1
    assert result["logs"] == ["all_of_channel_timer"]
    assert result["done"] == "ok"


async def test_is_resume_avoids_duplicate_side_effects() -> None:
    graph = AdvancedStateGraph(PrimitiveState)
    db_writes: list[str] = []

    async def start_node(state: PrimitiveState) -> Command:
        return Command(
            update={"counter": 0, "logs": [], "done": None},
            goto=Send("wait_node", None),
        )

    async def wait_node(ctx: Context, _input: None, state: PrimitiveState) -> Command:
        if not ctx.IsResume():
            # Simulate one-time side effect (e.g. database write).
            db_writes.append("write")
        await ctx.wait_for(timer_condition(seconds=0.02))
        state["logs"].append(f"resume={ctx.IsResume()}")
        return Command(update=state, goto=Send("finish_node", None))

    async def finish_node(_input: None, state: PrimitiveState) -> dict[str, object]:
        return {"counter": state["counter"], "logs": state["logs"], "done": "ok"}

    graph.add_entry_node(start_node)
    graph.add_node(wait_node)
    graph.add_finish_node(finish_node)

    result = await graph.compile().ainvoke({"counter": 0, "logs": [], "done": None})
    assert db_writes == ["write"]
    assert result["counter"] == 0
    assert result["logs"] == ["resume=True"]
    assert result["done"] == "ok"


async def test_state_field_locking_serializes_conflicting_nodes() -> None:
    graph = AdvancedStateGraph(PrimitiveState)
    graph.add_async_channel("done", str)
    intervals: dict[str, tuple[float, float]] = {}

    async def start_node(state: PrimitiveState) -> Command:
        return Command(
            update=state,
            goto=[
                Send("worker_a", None),
                Send("worker_b", None),
                Send("wait_node", None),
            ],
        )

    async def worker_a(ctx: Context, _input: None, state: PrimitiveState) -> Command:
        started = time.perf_counter()
        await asyncio.sleep(0.04)
        ended = time.perf_counter()
        intervals["a"] = (started, ended)
        ctx.publish_to_channel("done", "a")
        return Command(update=state)

    async def worker_b(ctx: Context, _input: None, state: PrimitiveState) -> Command:
        started = time.perf_counter()
        await asyncio.sleep(0.04)
        ended = time.perf_counter()
        intervals["b"] = (started, ended)
        ctx.publish_to_channel("done", "b")
        return Command(update=state)

    async def wait_node(ctx: Context, _input: None, state: PrimitiveState) -> Command:
        await ctx.wait_for(channel_condition("done", min=2))
        return Command(update=state, goto=Send("finish_node", None))

    async def finish_node(_input: None, state: PrimitiveState) -> dict[str, object]:
        return {"counter": state["counter"], "logs": state["logs"], "done": "ok"}

    graph.add_entry_node(start_node)
    graph.add_node(worker_a, state_option={"locked_fields": ["counter"]})
    graph.add_node(worker_b, state_option={"locked_fields": ["counter"]})
    graph.add_node(wait_node)
    graph.add_finish_node(finish_node)

    result = await graph.compile().ainvoke({"counter": 0, "logs": [], "done": None})
    assert result["done"] == "ok"
    assert "a" in intervals and "b" in intervals
    a_start, a_end = intervals["a"]
    b_start, b_end = intervals["b"]
    serialized = (a_end <= b_start) or (b_end <= a_start)
    assert serialized, f"expected serialized execution, got overlap: a={intervals['a']} b={intervals['b']}"

