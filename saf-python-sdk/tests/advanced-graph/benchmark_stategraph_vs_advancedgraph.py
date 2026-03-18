from __future__ import annotations

import asyncio
import os
import time
from typing import Any

# Configure advanced-graph runtime pools for this benchmark run.
os.environ["LANGGRAPH_RUN_POOL_SIZE"] = "10"
os.environ["LANGGRAPH_NODE_POOL_SIZE"] = "1000"

from saf_python_sdk.advanced_graph import (
    AdvancedStateGraph,
    channel_condition,
    timer_condition,
)
from saf_python_sdk.types import Command, Send
from langgraph.graph import END, START, StateGraph

RUNS = 100
MIDDLE_COUNT = 10
SLEEP_SECONDS = 2.0
BLOCKING_SECONDS = 0.1
STATE_BYTES = 10 * 1024


def make_initial_state() -> dict[str, Any]:
    return {"payload": "x" * STATE_BYTES, "done": False}


def build_advanced_parallel() -> Any:
    graph: AdvancedStateGraph[dict[str, Any]] = AdvancedStateGraph(dict)
    done_channel = "__bench_done_channel"
    graph.add_async_channel(done_channel, str)

    async def start_node(state: dict[str, Any]) -> Command:
        _ = state
        sends = [Send(f"middle_{i}", None) for i in range(MIDDLE_COUNT)]
        sends.append(Send("end_node", None))
        return Command(goto=sends)

    async def end_node(ctx: Any, state: dict[str, Any]) -> dict[str, Any]:
        await ctx.wait_for(channel_condition(done_channel, min=MIDDLE_COUNT))
        out = dict(state)
        out["done"] = True
        return out

    graph.add_entry_node(start_node)
    for i in range(MIDDLE_COUNT):

        async def middle_node(ctx: Any, state: dict[str, Any], idx: int = i) -> None:
            _ = idx
            _ = state
            await ctx.wait_for(timer_condition(seconds=SLEEP_SECONDS))
            ctx.publish_to_channel(done_channel, "done")

        graph.add_node(f"middle_{i}", middle_node)
    graph.add_finish_node(end_node)
    return graph.compile()


def build_advanced_sequential() -> Any:
    graph: AdvancedStateGraph[dict[str, Any]] = AdvancedStateGraph(dict)

    async def start_node(state: dict[str, Any]) -> Command:
        _ = state
        return Command(goto=Send("middle_0", None))

    async def end_node(state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        out["done"] = True
        return out

    graph.add_entry_node(start_node)

    def make_middle(target: str):
        async def middle_node(ctx: Any, state: dict[str, Any]) -> Command:
            _ = state
            await ctx.wait_for(timer_condition(seconds=SLEEP_SECONDS))
            return Command(goto=Send(target, None))

        return middle_node

    for i in range(MIDDLE_COUNT):
        next_name = "end_node" if i == MIDDLE_COUNT - 1 else f"middle_{i+1}"
        graph.add_node(f"middle_{i}", make_middle(next_name))
    graph.add_finish_node(end_node)
    return graph.compile()


def build_stategraph_parallel() -> Any:
    graph = StateGraph(dict)

    async def start_node(state: dict[str, Any]) -> None:
        _ = state

    async def end_node(state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        out["done"] = True
        return out

    graph.add_node("start_node", start_node)
    for i in range(MIDDLE_COUNT):

        async def middle_node(state: dict[str, Any], idx: int = i) -> None:
            _ = idx
            _ = state
            await asyncio.sleep(SLEEP_SECONDS)

        graph.add_node(f"middle_{i}", middle_node)
    graph.add_node("end_node", end_node)

    graph.add_edge(START, "start_node")
    for i in range(MIDDLE_COUNT):
        graph.add_edge("start_node", f"middle_{i}")
        graph.add_edge(f"middle_{i}", "end_node")
    graph.add_edge("end_node", END)
    return graph.compile()


def build_stategraph_sequential() -> Any:
    graph = StateGraph(dict)

    async def start_node(state: dict[str, Any]) -> None:
        _ = state

    async def end_node(state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        out["done"] = True
        return out

    graph.add_node("start_node", start_node)
    for i in range(MIDDLE_COUNT):

        async def middle_node(state: dict[str, Any], idx: int = i) -> None:
            _ = idx
            _ = state
            await asyncio.sleep(SLEEP_SECONDS)

        graph.add_node(f"middle_{i}", middle_node)
    graph.add_node("end_node", end_node)

    graph.add_edge(START, "start_node")
    graph.add_edge("start_node", "middle_0")
    for i in range(MIDDLE_COUNT - 1):
        graph.add_edge(f"middle_{i}", f"middle_{i+1}")
    graph.add_edge(f"middle_{MIDDLE_COUNT - 1}", "end_node")
    graph.add_edge("end_node", END)
    return graph.compile()


def build_advanced_parallel_blocking() -> Any:
    graph: AdvancedStateGraph[dict[str, Any]] = AdvancedStateGraph(dict)
    done_channel = "__bench_done_channel_blocking"
    graph.add_async_channel(done_channel, str)

    async def start_node(state: dict[str, Any]) -> Command:
        _ = state
        sends = [Send(f"middle_blocking_{i}", None) for i in range(MIDDLE_COUNT)]
        sends.append(Send("end_node_blocking", None))
        return Command(goto=sends)

    async def end_node_blocking(ctx: Any, state: dict[str, Any]) -> dict[str, Any]:
        await ctx.wait_for(channel_condition(done_channel, min=MIDDLE_COUNT))
        out = dict(state)
        out["done"] = True
        return out

    graph.add_entry_node(start_node)
    for i in range(MIDDLE_COUNT):

        async def middle_blocking(
            ctx: Any, state: dict[str, Any], idx: int = i
        ) -> None:
            _ = idx
            _ = state
            time.sleep(BLOCKING_SECONDS)
            ctx.publish_to_channel(done_channel, "done")

        graph.add_node(f"middle_blocking_{i}", middle_blocking)
    graph.add_finish_node(end_node_blocking)
    return graph.compile()


def build_advanced_sequential_blocking() -> Any:
    graph: AdvancedStateGraph[dict[str, Any]] = AdvancedStateGraph(dict)

    async def start_node(state: dict[str, Any]) -> Command:
        _ = state
        return Command(goto=Send("middle_blocking_seq_0", None))

    async def end_node_blocking_seq(state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        out["done"] = True
        return out

    graph.add_entry_node(start_node)

    def make_middle(target: str):
        async def middle_blocking_seq(state: dict[str, Any]) -> Command:
            _ = state
            time.sleep(BLOCKING_SECONDS)
            return Command(goto=Send(target, None))

        return middle_blocking_seq

    for i in range(MIDDLE_COUNT):
        next_name = (
            "end_node_blocking_seq"
            if i == MIDDLE_COUNT - 1
            else f"middle_blocking_seq_{i+1}"
        )
        graph.add_node(f"middle_blocking_seq_{i}", make_middle(next_name))
    graph.add_finish_node(end_node_blocking_seq)
    return graph.compile()


def build_stategraph_parallel_blocking() -> Any:
    graph = StateGraph(dict)

    async def start_node(state: dict[str, Any]) -> None:
        _ = state

    async def end_node(state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        out["done"] = True
        return out

    graph.add_node("start_node", start_node)
    for i in range(MIDDLE_COUNT):

        async def middle_blocking(state: dict[str, Any], idx: int = i) -> None:
            _ = idx
            _ = state
            time.sleep(BLOCKING_SECONDS)

        graph.add_node(f"middle_blocking_{i}", middle_blocking)
    graph.add_node("end_node", end_node)

    graph.add_edge(START, "start_node")
    for i in range(MIDDLE_COUNT):
        graph.add_edge("start_node", f"middle_blocking_{i}")
        graph.add_edge(f"middle_blocking_{i}", "end_node")
    graph.add_edge("end_node", END)
    return graph.compile()


def build_stategraph_sequential_blocking() -> Any:
    graph = StateGraph(dict)

    async def start_node(state: dict[str, Any]) -> None:
        _ = state

    async def end_node(state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state)
        out["done"] = True
        return out

    graph.add_node("start_node", start_node)
    for i in range(MIDDLE_COUNT):

        async def middle_blocking_seq(state: dict[str, Any], idx: int = i) -> None:
            _ = idx
            _ = state
            time.sleep(BLOCKING_SECONDS)

        graph.add_node(f"middle_blocking_seq_{i}", middle_blocking_seq)
    graph.add_node("end_node", end_node)

    graph.add_edge(START, "start_node")
    graph.add_edge("start_node", "middle_blocking_seq_0")
    for i in range(MIDDLE_COUNT - 1):
        graph.add_edge(
            f"middle_blocking_seq_{i}",
            f"middle_blocking_seq_{i+1}",
        )
    graph.add_edge(f"middle_blocking_seq_{MIDDLE_COUNT - 1}", "end_node")
    graph.add_edge("end_node", END)
    return graph.compile()


async def run_benchmark(name: str, compiled: Any) -> float:
    started = time.perf_counter()
    tasks = [asyncio.create_task(compiled.ainvoke(make_initial_state())) for _ in range(RUNS)]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    if not all(item.get("done") is True for item in results):
        raise RuntimeError(f"{name} produced unfinished runs")
    return elapsed


async def main() -> None:
    suites = [
        ("advanced-graph-parallel", build_advanced_parallel()),
        ("advanced-graph-sequential", build_advanced_sequential()),
        ("state-graph-parallel", build_stategraph_parallel()),
        ("state-graph-sequential", build_stategraph_sequential()),
        ("advanced-graph-parallel-blocking", build_advanced_parallel_blocking()),
        ("advanced-graph-sequential-blocking", build_advanced_sequential_blocking()),
        ("state-graph-parallel-blocking", build_stategraph_parallel_blocking()),
        ("state-graph-sequential-blocking", build_stategraph_sequential_blocking()),
    ]
    print(
        f"runs={RUNS}, middle_nodes={MIDDLE_COUNT}, sleep={SLEEP_SECONDS}s, "
        f"blocking_sleep={BLOCKING_SECONDS}s, state_bytes={STATE_BYTES}"
    )
    for name, compiled in suites:
        elapsed = await run_benchmark(name, compiled)
        print(f"{name}: {elapsed:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())

