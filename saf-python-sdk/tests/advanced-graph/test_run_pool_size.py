import os
import subprocess
import sys


def test_run_pool_size_one_still_allows_parallel_runs() -> None:
    script = r"""
import asyncio
import time
from typing_extensions import TypedDict
from saf_python_sdk.advanced_graph import AdvancedStateGraph, Context, timer_condition
from saf_python_sdk.types import Command, Send


class RunState(TypedDict):
    done: bool


async def wait_node(ctx: Context, _: object, state: RunState) -> Command:
    await ctx.wait_for(timer_condition(seconds=0.2))
    return Command(goto=Send("finish_node", None), update=state)


async def finish_node(_: object, state: RunState) -> dict[str, bool]:
    return {"done": True}


async def main() -> None:
    graph = AdvancedStateGraph(RunState)
    graph.add_entry_node(wait_node)
    graph.add_finish_node(finish_node)
    compiled = graph.compile()
    started = time.perf_counter()
    await asyncio.gather(
        compiled.ainvoke({"done": False}),
        compiled.ainvoke({"done": False}),
    )
    elapsed = time.perf_counter() - started
    print(f"{elapsed:.6f}")


asyncio.run(main())
"""
    env = os.environ.copy()
    env["LANGGRAPH_RUN_POOL_SIZE"] = "1"
    env.setdefault("LANGGRAPH_NODE_POOL_SIZE", "2")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed = float(completed.stdout.strip().splitlines()[-1])
    assert elapsed < 0.35, completed.stdout

