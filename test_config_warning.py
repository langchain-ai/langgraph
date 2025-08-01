#!/usr/bin/env python3
"""Test script to verify the config warning implementation."""

from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
import asyncio
from typing_extensions import TypedDict

from langgraph.version import __version__ as version


class State(TypedDict):
    foo: str


async def call_model(state: State, config: dict) -> Dict[str, List[AIMessage]]:
    print("CONFIG", config)
    return {"foo": "bar"}


builder = StateGraph(State)
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
graph = builder.compile(name="ReAct Agent")


async def main():
    async for chunk in graph.astream(
        {
            "messages": [{"role": "user", "content": "Hi there!"}],
            "configurable": {"foo": "bar"},
        }
    ):
        pass


try:
    asyncio.run(main())
    print(f"Passed in langgraph version=={version}")
except Exception as e:
    print(f"Failed in langgraph version=={version}")
    print("ERROR", e)
