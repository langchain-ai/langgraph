import random
from typing import Optional

from pyperf._runner import Runner
from uvloop import new_event_loop

from bench.fanout_to_subgraph import fanout_to_subgraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.pregel import Pregel


async def run(graph: Pregel, input: dict, config: Optional[dict]):
    len([c async for c in graph.astream(input, config=config)])


benchmarks = (
    (
        "fanout_to_subgraph_10x",
        fanout_to_subgraph().compile(checkpointer=None),
        {
            "subjects": [
                random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(10)
            ]
        },
        None,
    ),
    # (
    #     "fanout_to_subgraph_10x_checkpoint",
    #     fanout_to_subgraph().compile(checkpointer=MemorySaver()),
    #     {
    #         "subjects": [
    #             random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(10)
    #         ]
    #     },
    #     {"configurable": {"thread_id": "1"}},
    # ),
    # (
    #     "fanout_to_subgraph_100x",
    #     fanout_to_subgraph().compile(checkpointer=None),
    #     {
    #         "subjects": [
    #             random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(100)
    #         ]
    #     },
    #     None,
    # ),
    # (
    #     "fanout_to_subgraph_100x_checkpoint",
    #     fanout_to_subgraph().compile(checkpointer=MemorySaver()),
    #     {
    #         "subjects": [
    #             random.choices("abcdefghijklmnopqrstuvwxyz", k=1000) for _ in range(100)
    #         ]
    #     },
    #     {"configurable": {"thread_id": "1"}},
    # ),
)


r = Runner()

for name, graph, input, config in benchmarks:
    r.bench_async_func(name, run, graph, input, config, loop_factory=new_event_loop)
