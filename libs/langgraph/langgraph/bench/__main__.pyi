from _typeshed import Incomplete
from bench.fanout_to_subgraph import fanout_to_subgraph as fanout_to_subgraph, fanout_to_subgraph_sync as fanout_to_subgraph_sync
from bench.react_agent import react_agent as react_agent
from bench.wide_state import wide_state as wide_state
from langgraph.pregel import Pregel as Pregel

async def arun(graph: Pregel, input: dict): ...
def run(graph: Pregel, input: dict): ...

benchmarks: Incomplete
r: Incomplete
