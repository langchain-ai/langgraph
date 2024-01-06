from asyncio import iscoroutinefunction
from collections import defaultdict
from typing import Any, Callable, Dict, NamedTuple

from langchain_core.runnables import Runnable
from langchain_core.runnables.base import (
    RunnableLambda,
    RunnableLike,
    coerce_to_runnable,
)

from permchain.pregel import Channel, Pregel


class Edge(NamedTuple):
    start: str
    end: str


class Branch(NamedTuple):
    condition: Callable[..., str]
    ends: dict[str, str]

    def runnable(self, input: Any) -> Runnable:
        result = self.condition(input)
        return Channel.write_to(self.ends[result])


START = "__start__"
END = "__end__"


class Graph:
    def __init__(self):
        self.nodes: dict[str, Runnable] = {}
        self.edges = set[Edge]()
        self.branches: defaultdict[str, list[Branch]] = defaultdict(list)

    def add_node(self, key: str, action: RunnableLike) -> None:
        if key in self.nodes:
            raise ValueError(f"Node `{key}` already present.")

        self.nodes[key] = coerce_to_runnable(action)

    def add_edge(self, start_key: str, end_key: str) -> None:
        if start_key not in self.nodes:
            raise ValueError(f"Need to add_node `{start_key}` first")
        if end_key not in self.nodes:
            raise ValueError(f"Need to add_node `{end_key}` first")

        # TODO: support multiple message passing
        if start_key in set(start for start, _ in self.edges):
            raise ValueError(f"Already found path for {start_key}")

        self.edges.add((start_key, end_key))

    def add_conditional_edges(
        self,
        start_key: str,
        condition: Callable[..., str],
        conditional_edge_mapping: Dict[str, str],
    ):
        if start_key not in self.nodes:
            raise ValueError(f"Need to add_node `{start_key}` first")
        if iscoroutinefunction(condition):
            raise ValueError("Condition cannot be a coroutine function")

        self.branches[start_key].append(Branch(condition, conditional_edge_mapping))

    def set_entry_point(self, key: str):
        if key not in self.nodes:
            raise ValueError(f"Need to add_node `{key}` first")
        self.entry_point = key

    def set_finish_point(self, key: str):
        if key not in self.nodes:
            raise ValueError(f"Need to add_node `{key}` first")
        self.finish_point = key

    def compile(self):
        ################################################
        #       STEP 1: VALIDATE GRAPH STRUCTURE       #
        ################################################

        all_starts = (
            {start for start, _ in self.edges}
            | {start for start in self.branches}
            | ({self.finish_point} if hasattr(self, "finish_point") else set())
        )
        all_ends = (
            {end for _, end in self.edges}
            | {
                end
                for branch_list in self.branches.values()
                for branch in branch_list
                for end in branch.ends.values()
            }
            | {self.entry_point}
        )

        for node in self.nodes:
            if node not in all_ends:
                raise ValueError(f"Node `{node}` is not reachable")
            if node not in all_starts:
                raise ValueError(f"Node `{node}` is a dead-end")

        ################################################
        #             STEP 2: CREATE GRAPH             #
        ################################################

        outgoing_edges = defaultdict(list)
        for start, end in self.edges:
            outgoing_edges[start].append(end)
        if hasattr(self, "finish_point"):
            outgoing_edges[self.finish_point].append(END)

        nodes = {
            key: Channel.subscribe_to(key) | node for key, node in self.nodes.items()
        }

        for key, edges in outgoing_edges.items():
            if edges:
                nodes[key] |= Channel.write_to(*edges)

        for key, branches in self.branches.items():
            for branch in branches:
                nodes[key] |= RunnableLambda(branch.runnable, name=f"{key}_condition")

        return Pregel(
            nodes=nodes,
            input=self.entry_point,
            output=END,
        )
