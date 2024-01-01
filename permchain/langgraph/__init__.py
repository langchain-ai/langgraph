from langchain_core.runnables import Runnable, RunnableMap
from typing import Callable, Union, Optional
from permchain import Channel, Pregel

class Actor:

    def __init__(self, name: str, runnable: Runnable):
        self.name = name
        self.runnable = runnable


class DecisionPoint:

    def __init__(self, name: str, callable: Callable):
        self.name = name
        self.callable = callable

class End:
    name = "end"


def branch(data, condition, mapping):
    result = condition(data)
    return Channel.write_to(mapping[result])

class Graph:

    def __init__(self):
        self.nodes = {"end": End()}
        self.connections = {}
        self.branches = {}
        self.entry_point: Optional[str] = None
        self.finish_points = set()

    def register(self, node: Union[Actor, DecisionPoint]):
        if node.name in self.nodes:
            raise ValueError(f"Actor `{node.name}` already present.")
        self.nodes[node.name] = node

    def connect(self, start: Actor, end: Union[Actor, DecisionPoint]):
        if start.name not in self.nodes:
            raise ValueError(f"Need to register `{start.name}` first")
        if end.name not in self.nodes:
            raise ValueError(f"Need to register `{end.name}` first")
        if start.name in self.connections:
            raise ValueError(f"Already found path for {start.name}")
        self.connections[start.name] = end.name

    def branch(self, start: DecisionPoint, end: Optional[Actor], condition: str):
        end = end or End()
        if start.name not in self.nodes:
            raise ValueError(f"Need to register `{start.name}` first")
        if end.name not in self.nodes:
            raise ValueError(f"Need to register `{end.name}` first")
        if start.name not in self.branches:
            self.branches[start.name] = {}
        if condition in self.branches[start.name]:
            raise ValueError(f"Already found a condition for {start.name} and {condition}")
        self.branches[start.name][condition] = end.name

    def set_entry_point(self, node: Union[DecisionPoint, Actor]):
        if node.name not in self.nodes:
            raise ValueError(f"Need to register `{node.name}` first")
        self.entry_point = node.name

    def set_finish_point(self, node: Actor):
        if node.name not in self.nodes:
            raise ValueError(f"Need to register `{node.name}` first")
        self.finish_points |= node.name

    def compile(self):
        # Validate all nodes have an entry point
        all_nodes = set(self.nodes)
        all_entry_points = set(self.connections).union(self.branches).union(self.finish_points)
        branch_ends = set()
        for v in self.branches.values():
            branch_ends.update(v.values())
        all_finish_points = set(self.connections.values()).union(branch_ends).union({self.entry_point})
        # If a node is not a finish point, then it is missing an entry point
        missing_entry = all_nodes.difference(all_finish_points)
        if missing_entry:
            raise ValueError(f"Some nodes are missing entry points: {missing_entry}")
        # If a node is not an entry point, then it is missing a finish point
        missing_finish = all_nodes.difference(all_entry_points).difference({"end"})
        if missing_finish:
            raise ValueError(f"Some nodes are missing finish points: {missing_finish}")
        chains = {
            start: Channel.subscribe_to(start) | self.nodes[start].runnable | Channel.write_to(end)
            for start, end in self.connections.items()
        }

        decisions = {
            start: Channel.subscribe_to(start) | (lambda x: branch(x, self.nodes[start].callable, mapping))
            for start, mapping in self.branches.items()
        }
        endings = {
            end: Channel.subscribe_to(end) | self.nodes[end].runnable | Channel.write_to("end")
            for end in self.finish_points
        }
        app = Pregel(
            chains = {**chains, **decisions, **endings},
            input=self.entry_point,
            output="end"
        )
        return app










