import random
from uuid import uuid4

from langchain_core.messages import HumanMessage
from pyperf._runner import Runner
from uvloop import new_event_loop

from bench.fanout_to_subgraph import fanout_to_subgraph, fanout_to_subgraph_sync
from bench.react_agent import react_agent
from bench.wide_state import wide_state
from langgraph.checkpoint.memory import MemorySaver
from langgraph.pregel import Pregel


async def arun(graph: Pregel, input: dict):
    len(
        [
            c
            async for c in graph.astream(
                input,
                {
                    "configurable": {"thread_id": str(uuid4())},
                    "recursion_limit": 1000000000,
                },
            )
        ]
    )


def run(graph: Pregel, input: dict):
    len(
        [
            c
            for c in graph.stream(
                input,
                {
                    "configurable": {"thread_id": str(uuid4())},
                    "recursion_limit": 1000000000,
                },
            )
        ]
    )


benchmarks = (
    (
        "fanout_to_subgraph_10x",
        fanout_to_subgraph().compile(checkpointer=None),
        fanout_to_subgraph_sync().compile(checkpointer=None),
        {
            "subjects": [
                random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(10)
            ]
        },
    ),
    (
        "fanout_to_subgraph_10x_checkpoint",
        fanout_to_subgraph().compile(checkpointer=MemorySaver()),
        fanout_to_subgraph_sync().compile(checkpointer=MemorySaver()),
        {
            "subjects": [
                random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(10)
            ]
        },
    ),
    (
        "fanout_to_subgraph_100x",
        fanout_to_subgraph().compile(checkpointer=None),
        fanout_to_subgraph_sync().compile(checkpointer=None),
        {
            "subjects": [
                random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(100)
            ]
        },
    ),
    (
        "fanout_to_subgraph_100x_checkpoint",
        fanout_to_subgraph().compile(checkpointer=MemorySaver()),
        fanout_to_subgraph_sync().compile(checkpointer=MemorySaver()),
        {
            "subjects": [
                random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(100)
            ]
        },
    ),
    (
        "react_agent_10x",
        react_agent(10, checkpointer=None),
        react_agent(10, checkpointer=None),
        {"messages": [HumanMessage("hi?")]},
    ),
    (
        "react_agent_10x_checkpoint",
        react_agent(10, checkpointer=MemorySaver()),
        react_agent(10, checkpointer=MemorySaver()),
        {"messages": [HumanMessage("hi?")]},
    ),
    (
        "react_agent_100x",
        react_agent(100, checkpointer=None),
        react_agent(100, checkpointer=None),
        {"messages": [HumanMessage("hi?")]},
    ),
    (
        "react_agent_100x_checkpoint",
        react_agent(100, checkpointer=MemorySaver()),
        react_agent(100, checkpointer=MemorySaver()),
        {"messages": [HumanMessage("hi?")]},
    ),
    (
        "wide_state_25x300",
        wide_state(300).compile(checkpointer=None),
        wide_state(300).compile(checkpointer=None),
        {
            "messages": [
                {
                    str(i) * 10: {
                        str(j) * 10: ["hi?" * 10, True, 1, 6327816386138, None] * 5
                        for j in range(5)
                    }
                    for i in range(5)
                }
            ]
        },
    ),
    (
        "wide_state_25x300_checkpoint",
        wide_state(300).compile(checkpointer=MemorySaver()),
        wide_state(300).compile(checkpointer=MemorySaver()),
        {
            "messages": [
                {
                    str(i) * 10: {
                        str(j) * 10: ["hi?" * 10, True, 1, 6327816386138, None] * 5
                        for j in range(5)
                    }
                    for i in range(5)
                }
            ]
        },
    ),
    (
        "wide_state_15x600",
        wide_state(600).compile(checkpointer=None),
        wide_state(600).compile(checkpointer=None),
        {
            "messages": [
                {
                    str(i) * 10: {
                        str(j) * 10: ["hi?" * 10, True, 1, 6327816386138, None] * 5
                        for j in range(5)
                    }
                    for i in range(3)
                }
            ]
        },
    ),
    (
        "wide_state_15x600_checkpoint",
        wide_state(600).compile(checkpointer=MemorySaver()),
        wide_state(600).compile(checkpointer=MemorySaver()),
        {
            "messages": [
                {
                    str(i) * 10: {
                        str(j) * 10: ["hi?" * 10, True, 1, 6327816386138, None] * 5
                        for j in range(5)
                    }
                    for i in range(3)
                }
            ]
        },
    ),
    (
        "wide_state_9x1200",
        wide_state(1200).compile(checkpointer=None),
        wide_state(1200).compile(checkpointer=None),
        {
            "messages": [
                {
                    str(i) * 10: {
                        str(j) * 10: ["hi?" * 10, True, 1, 6327816386138, None] * 5
                        for j in range(3)
                    }
                    for i in range(3)
                }
            ]
        },
    ),
    (
        "wide_state_9x1200_checkpoint",
        wide_state(1200).compile(checkpointer=MemorySaver()),
        wide_state(1200).compile(checkpointer=MemorySaver()),
        {
            "messages": [
                {
                    str(i) * 10: {
                        str(j) * 10: ["hi?" * 10, True, 1, 6327816386138, None] * 5
                        for j in range(3)
                    }
                    for i in range(3)
                }
            ]
        },
    ),
)


r = Runner()

for name, agraph, graph, input in benchmarks:
    r.bench_async_func(name, arun, agraph, input, loop_factory=new_event_loop)
    if graph is not None:
        r.bench_func(name + "_sync", run, graph, input)
