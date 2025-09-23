import random
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pyperf._runner import Runner
from uvloop import new_event_loop

from bench.fanout_to_subgraph import fanout_to_subgraph, fanout_to_subgraph_sync
from bench.pydantic_state import pydantic_state
from bench.react_agent import react_agent
from bench.sequential import create_sequential
from bench.wide_dict import wide_dict
from bench.wide_state import wide_state
from langgraph.graph import StateGraph
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
                durability="exit",
            )
        ]
    )


async def arun_first_event_latency(graph: Pregel, input: dict) -> None:
    """Latency for the first event.

    Run the graph until the first event is processed and then stop.
    """
    stream = graph.astream(
        input,
        {
            "configurable": {"thread_id": str(uuid4())},
            "recursion_limit": 1000000000,
        },
        durability="exit",
    )

    try:
        async for _ in stream:
            break
    finally:
        await stream.aclose()


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
                durability="exit",
            )
        ]
    )


def run_first_event_latency(graph: Pregel, input: dict) -> None:
    """Latency for the first event.

    Run the graph until the first event is processed and then stop.
    """
    stream = graph.stream(
        input,
        {
            "configurable": {"thread_id": str(uuid4())},
            "recursion_limit": 1000000000,
        },
        durability="exit",
    )

    try:
        for _ in stream:
            break
    finally:
        stream.close()


def compile_graph(graph: StateGraph) -> None:
    """Compile the graph."""
    graph.compile()


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
        fanout_to_subgraph().compile(checkpointer=InMemorySaver()),
        fanout_to_subgraph_sync().compile(checkpointer=InMemorySaver()),
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
        fanout_to_subgraph().compile(checkpointer=InMemorySaver()),
        fanout_to_subgraph_sync().compile(checkpointer=InMemorySaver()),
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
        react_agent(10, checkpointer=InMemorySaver()),
        react_agent(10, checkpointer=InMemorySaver()),
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
        react_agent(100, checkpointer=InMemorySaver()),
        react_agent(100, checkpointer=InMemorySaver()),
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
        wide_state(300).compile(checkpointer=InMemorySaver()),
        wide_state(300).compile(checkpointer=InMemorySaver()),
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
        wide_state(600).compile(checkpointer=InMemorySaver()),
        wide_state(600).compile(checkpointer=InMemorySaver()),
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
        wide_state(1200).compile(checkpointer=InMemorySaver()),
        wide_state(1200).compile(checkpointer=InMemorySaver()),
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
        "wide_dict_25x300",
        wide_dict(300).compile(checkpointer=None),
        wide_dict(300).compile(checkpointer=None),
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
        "wide_dict_25x300_checkpoint",
        wide_dict(300).compile(checkpointer=InMemorySaver()),
        wide_dict(300).compile(checkpointer=InMemorySaver()),
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
        "wide_dict_15x600",
        wide_dict(600).compile(checkpointer=None),
        wide_dict(600).compile(checkpointer=None),
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
        "wide_dict_15x600_checkpoint",
        wide_dict(600).compile(checkpointer=InMemorySaver()),
        wide_dict(600).compile(checkpointer=InMemorySaver()),
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
        "wide_dict_9x1200",
        wide_dict(1200).compile(checkpointer=None),
        wide_dict(1200).compile(checkpointer=None),
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
        "wide_dict_9x1200_checkpoint",
        wide_dict(1200).compile(checkpointer=InMemorySaver()),
        wide_dict(1200).compile(checkpointer=InMemorySaver()),
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
        "sequential_10",
        create_sequential(10).compile(),
        create_sequential(10).compile(),
        {"messages": []},  # Empty list of messages
    ),
    (
        "sequential_1000",
        create_sequential(1000).compile(),
        create_sequential(1000).compile(),
        {"messages": []},  # Empty list of messages
    ),
    (
        "pydantic_state_25x300",
        pydantic_state(300).compile(checkpointer=None),
        pydantic_state(300).compile(checkpointer=None),
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
        "pydantic_state_25x300_checkpoint",
        pydantic_state(300).compile(checkpointer=InMemorySaver()),
        pydantic_state(300).compile(checkpointer=InMemorySaver()),
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
        "pydantic_state_15x600",
        pydantic_state(600).compile(checkpointer=None),
        pydantic_state(600).compile(checkpointer=None),
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
        "pydantic_state_15x600_checkpoint",
        pydantic_state(600).compile(checkpointer=InMemorySaver()),
        pydantic_state(600).compile(checkpointer=InMemorySaver()),
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
        "pydantic_state_9x1200",
        pydantic_state(1200).compile(checkpointer=None),
        pydantic_state(1200).compile(checkpointer=None),
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
        "pydantic_state_9x1200_checkpoint",
        pydantic_state(1200).compile(checkpointer=InMemorySaver()),
        pydantic_state(1200).compile(checkpointer=InMemorySaver()),
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

# Full graph run time
for name, agraph, graph, input in benchmarks:
    r.bench_async_func(name, arun, agraph, input, loop_factory=new_event_loop)
    if graph is not None:
        r.bench_func(name + "_sync", run, graph, input)


# Pick a handful of graphs to measure the first event latency.
# At the moment, limiting just due to the size of the annotation on github.
GRAPHS_FOR_1st_EVENT_LATENCY = (
    "sequential_1000",
    "pydantic_state_25x300",
)

# First event latency
for name, agraph, graph, input in benchmarks:
    if graph not in GRAPHS_FOR_1st_EVENT_LATENCY:
        continue
    r.bench_async_func(
        name + "_first_event_latency",
        arun_first_event_latency,
        agraph,
        input,
        loop_factory=new_event_loop,
    )
    if graph is not None:
        r.bench_func(
            name + "_first_event_latency_sync", run_first_event_latency, graph, input
        )

# Graph compilation times
compilation_benchmarks = (
    (
        "sequential_1000",
        create_sequential(1_000),
    ),
    (
        "pydantic_state_25x300",
        pydantic_state(300),
    ),
    (
        "wide_state_15x600",
        wide_state(600),
    ),
)

for name, graph in compilation_benchmarks:
    r.bench_func(name + "_compilation", compile_graph, graph)
