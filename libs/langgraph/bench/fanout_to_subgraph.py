import operator
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.types import Send


def fanout_to_subgraph() -> StateGraph:
    class OverallState(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], operator.add]

    async def continue_to_jokes(state: OverallState):
        return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

    class JokeInput(TypedDict):
        subject: str

    class JokeOutput(TypedDict):
        jokes: list[str]

    class JokeState(JokeInput, JokeOutput): ...

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
    subgraph = StateGraph(JokeState, input_schema=JokeInput, output_schema=JokeOutput)
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

    return builder


def fanout_to_subgraph_sync() -> StateGraph:
    class OverallState(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], operator.add]

    def continue_to_jokes(state: OverallState):
        return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

    class JokeInput(TypedDict):
        subject: str

    class JokeOutput(TypedDict):
        jokes: list[str]

    class JokeState(JokeInput, JokeOutput): ...

    def bump(state: JokeOutput):
        return {"jokes": [state["jokes"][0] + " a"]}

    def generate(state: JokeInput):
        return {"jokes": [f"Joke about {state['subject']}"]}

    def edit(state: JokeInput):
        subject = state["subject"]
        return {"subject": f"{subject} - hohoho"}

    def bump_loop(state: JokeOutput):
        return END if state["jokes"][0].endswith(" a" * 10) else "bump"

    # subgraph
    subgraph = StateGraph(JokeState, input_schema=JokeInput, output_schema=JokeOutput)
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

    return builder


if __name__ == "__main__":
    import asyncio
    import random
    import time

    import uvloop
    from langgraph.checkpoint.memory import InMemorySaver

    graph = fanout_to_subgraph().compile(checkpointer=InMemorySaver())
    input = {
        "subjects": [
            random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(1000)
        ]
    }
    config = {"configurable": {"thread_id": "1"}}

    async def run():
        len([c async for c in graph.astream(input, config=config)])

    uvloop.install()
    start = time.time()
    asyncio.run(run())
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
