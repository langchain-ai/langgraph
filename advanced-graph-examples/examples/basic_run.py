from __future__ import annotations

import asyncio
from typing import TypedDict

from saf_python_sdk import Command, Send
from saf_python_sdk.advanced_graph import AdvancedStateGraph


class MyState(TypedDict):
    count: int
    logs: list[str]


async def start_node(state: MyState) -> Command:
    state["logs"].append("start")
    return Command(update=state, goto=Send("finish_node", "hello"))


async def finish_node(input: str, state: MyState) -> Command:
    state["logs"].append(f"finish:{input}")
    state["count"] += 1
    return Command(update=state)


async def main() -> None:
    graph = AdvancedStateGraph(MyState)
    graph.add_entry_node(start_node)
    graph.add_finish_node(finish_node)
    result = await graph.compile().ainvoke({"count": 0, "logs": []})
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

