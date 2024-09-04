import asyncio
import operator
import random
from time import perf_counter_ns
from typing import Annotated, TypedDict

import uvloop
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START, Send
from langgraph.graph.state import StateGraph

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]


async def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


class JokeInput(TypedDict):
    subject: str


class JokeOutput(TypedDict):
    jokes: list[str]


async def bump(state: JokeOutput):
    return {"jokes": [state["jokes"][0] + " a"]}


async def generate(state: JokeInput):
    return {"jokes": [f"Joke about {state['subject']}"]}


async def edit(state: JokeInput):
    subject = state["subject"]
    return {"subject": f"{subject} - hohoho"}


async def bump_loop(state: JokeOutput):
    return END if state["jokes"][0].endswith(" a" * 10) else "bump"


# subgraph
subgraph = StateGraph(input=JokeInput, output=JokeOutput)
subgraph.add_node("edit", edit)
subgraph.add_node("generate", generate)
subgraph.add_node("bump", bump)
subgraph.set_entry_point("edit")
subgraph.add_edge("edit", "generate")
subgraph.add_edge("generate", "bump")
subgraph.add_conditional_edges("bump", bump_loop)
subgraph.set_finish_point("generate")
subgraphc = subgraph.compile()

# parent graph
builder = StateGraph(OverallState)
builder.add_node("generate_joke", subgraphc)
builder.add_conditional_edges(START, continue_to_jokes)
builder.add_edge("generate_joke", END)


async def main():
    graph = builder.compile(checkpointer=None)
    config = {"configurable": {"thread_id": "1"}}
    input = {
        "subjects": [random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(1000)]
    }

    # invoke and pause at nested interrupt
    s = perf_counter_ns()
    len([c async for c in graph.astream(input, config=config)])
    print("Time taken:", (perf_counter_ns() - s) / 1e9)


asyncio.run(main())
