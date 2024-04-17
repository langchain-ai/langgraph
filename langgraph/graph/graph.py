import logging
from collections import defaultdict
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    cast,
)

from langchain_core.runnables import Runnable
from langchain_core.runnables.base import RunnableLike, coerce_to_runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.graph import (
    Graph as RunnableGraph,
)
from langchain_core.runnables.graph import (
    Node as RunnableGraphNode,
)

from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.constants import TAG_HIDDEN
from langgraph.pregel import Channel, Pregel
from langgraph.pregel.read import PregelNode
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
from langgraph.utils import RunnableCallable

logger = logging.getLogger(__name__)

START = "__start__"
END = "__end__"


class Branch(NamedTuple):
    condition: Runnable[Any, Union[str, list[str]]]
    ends: Optional[dict[str, str]]

    def run(
        self,
        writer: Callable[[list[str]], Optional[Runnable]],
        reader: Optional[Callable[[RunnableConfig], Any]] = None,
    ) -> None:
        return ChannelWrite.register_writer(
            RunnableCallable(
                func=self._route,
                afunc=self._aroute,
                writer=writer,
                reader=reader,
                name=None,
                trace=False,
            )
        )

    def _route(
        self,
        input: Any,
        config: RunnableConfig,
        *,
        reader: Optional[Callable[[], Any]],
        writer: Callable[[list[str]], Optional[Runnable]],
    ) -> Runnable:
        result = self.condition.invoke(reader(config) if reader else input, config)
        if not isinstance(result, list):
            result = [result]
        if self.ends:
            destinations = [self.ends[r] for r in result]
        else:
            destinations = result
        return writer(destinations) or input

    async def _aroute(
        self,
        input: Any,
        config: RunnableConfig,
        *,
        reader: Optional[Callable[[], Any]],
        writer: Callable[[list[str]], Optional[Runnable]],
    ) -> Runnable:
        result = await self.condition.ainvoke(
            reader(config) if reader else input, config
        )
        if not isinstance(result, list):
            result = [result]
        if self.ends:
            destinations = [self.ends[r] for r in result]
        else:
            destinations = result
        return writer(destinations) or input


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Runnable] = {}
        self.edges = set[tuple[str, str]]()
        self.branches: defaultdict[str, dict[str, Branch]] = defaultdict(dict)
        self.support_multiple_edges = False
        self.compiled = False

    @property
    def _all_edges(self) -> set[tuple[str, str]]:
        return self.edges

    def add_node(self, key: str, action: RunnableLike) -> None:
        if self.compiled:
            logger.warning(
                "Adding a node to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        if key in self.nodes:
            raise ValueError(f"Node `{key}` already present.")
        if key == END:
            raise ValueError(f"Node `{key}` is reserved.")

        self.nodes[key] = coerce_to_runnable(action)

    def add_edge(self, start_key: str, end_key: str) -> None:
        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        if start_key == END:
            raise ValueError("END cannot be a start node")
        if end_key == START:
            raise ValueError("START cannot be an end node")
        if start_key not in self.nodes and start_key != START:
            raise ValueError(f"Need to add_node `{start_key}` first")
        if end_key not in self.nodes and end_key != END:
            raise ValueError(f"Need to add_node `{end_key}` first")

        if not self.support_multiple_edges and start_key in set(
            start for start, _ in self.edges
        ):
            raise ValueError(
                f"Already found path for node '{start_key}'.\n"
                "For multiple edges, use StateGraph with an annotated state key."
            )

        self.edges.add((start_key, end_key))

    def add_conditional_edges(
        self,
        start_key: str,
        condition: Union[
            Callable[..., Union[str, list[str]]],
            Callable[..., Awaitable[Union[str, list[str]]]],
            Runnable[Any, Union[str, list[str]]],
        ],
        conditional_edge_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        """Add a conditional edge from the starting node to any number of destination nodes.


        Args:
            start_key (str): The key of the starting node.
            condition (Union[Callable, Runnable]): The condition that determines the destination of the edge.
            conditional_edge_mapping (Optional[dict[str, str]]): A dictionary that maps the response of the condition to a name of
                the destination node(s). If the condition returns a list, the response will be matched against the keys of the
                dictionary. If the condition returns a string, the response will be matched against the values of the dictionary.
                If the condition returns a string and the dictionary contains a key with the value of `END` ("__end__"`),
                the graph will finish.

        Raises:
            ValueError: If the starting node is not found in the graph or if the conditional edge mapping contains missing nodes.

        Returns:
            None
        """  # noqa: E501
        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        # find a name for the condition
        condition = coerce_to_runnable(condition)
        name = condition.name or "condition"
        # validate the condition
        if start_key not in self.nodes and start_key != START:
            raise ValueError(f"Need to add_node `{start_key}` first")
        if conditional_edge_mapping and set(
            conditional_edge_mapping.values()
        ).difference([END]).difference(self.nodes):
            raise ValueError(
                f"Missing nodes which are in conditional edge mapping. Mapping "
                f"contains possible destinations: "
                f"{list(conditional_edge_mapping.values())}. Possible nodes are "
                f"{list(self.nodes.keys())}."
            )
        if name in self.branches[start_key]:
            raise ValueError(
                f"Branch with name `{condition.name}` already exists for node "
                f"`{start_key}`"
            )
        # save it
        self.branches[start_key][name] = Branch(condition, conditional_edge_mapping)

    def set_entry_point(self, key: str) -> None:
        """Specifies the first node to be called in the graph.

        Parameters:
            key (str): The key of the node to set as the entry point.

        Returns:
            None
        """
        return self.add_edge(START, key)

    def set_conditional_entry_point(
        self,
        condition: Union[
            Callable[..., str], Callable[..., Awaitable[str]], Runnable[Any, str]
        ],
        conditional_edge_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """Sets a conditional entry point in the graph.

        Args:
            condition: A callable object that takes any number of arguments and returns a string or an awaitable string.
            conditional_edge_mapping: A dictionary that maps condition names to edge names.

        Returns:
            None
        """
        return self.add_conditional_edges(START, condition, conditional_edge_mapping)

    def set_finish_point(self, key: str) -> None:
        """Marks a node as a finish point of the graph.

        If the graph reaches this node, it will cease execution.

        Parameters:
            key (str): The key of the node to set as the finish point.

        Returns:
            None
        """
        return self.add_edge(key, END)

    def validate(self, interrupt: Optional[Sequence[str]] = None) -> None:
        all_starts = {src for src, _ in self._all_edges} | {
            src for src in self.branches
        }
        for node in self.nodes:
            if node not in all_starts:
                raise ValueError(f"Node `{node}` is a dead-end")

        all_branches = [
            branch
            for branches in self.branches.values()
            for branch in branches.values()
        ]
        if all(branch.ends is not None for branch in all_branches):
            all_ends = {end for _, end in self._all_edges} | {
                end for branch in all_branches for end in branch.ends.values()
            }

            for node in self.nodes:
                if node not in all_ends:
                    raise ValueError(f"Node `{node}` is not reachable")

        if interrupt:
            for node in interrupt:
                if node not in self.nodes:
                    raise ValueError(f"Node `{node}` is not present")

        self.compiled = True

    def compile(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[Sequence[str]] = None,
        interrupt_after: Optional[Sequence[str]] = None,
        debug: bool = False,
    ) -> "CompiledGraph":
        # assign default values
        interrupt_before = interrupt_before or []
        interrupt_after = interrupt_after or []

        # validate the graph
        self.validate(interrupt=interrupt_before + interrupt_after)

        # create empty compiled graph
        compiled = CompiledGraph(
            graph=self,
            nodes={},
            channels={START: EphemeralValue(Any), END: EphemeralValue(Any)},
            input_channels=START,
            output_channels=END,
            stream_mode="values",
            stream_channels=[],
            checkpointer=checkpointer,
            interrupt_before_nodes=interrupt_before,
            interrupt_after_nodes=interrupt_after,
            auto_validate=False,
            debug=debug,
        )

        # attach nodes, edges, and branches
        for key, node in self.nodes.items():
            compiled.attach_node(key, node)

        for start, end in self.edges:
            compiled.attach_edge(start, end)

        for start, branches in self.branches.items():
            for name, branch in branches.items():
                compiled.attach_branch(start, name, branch)

        # validate the compiled graph
        return compiled.validate()


class CompiledGraph(Pregel):
    graph: Graph

    def attach_node(self, key: str, node: Runnable) -> None:
        self.channels[key] = EphemeralValue(Any)
        self.nodes[key] = (
            PregelNode(channels=[], triggers=[])
            | node
            | ChannelWrite([ChannelWriteEntry(key)], tags=[TAG_HIDDEN])
        )
        cast(list[str], self.stream_channels).append(key)

    def attach_edge(self, start: str, end: str) -> None:
        if end == END:
            # publish to end channel
            self.nodes[start].writers.append(
                ChannelWrite([ChannelWriteEntry(END)], tags=[TAG_HIDDEN])
            )
        else:
            # subscribe to start channel
            self.nodes[end].triggers.append(start)
            self.nodes[end].channels.append(start)

    def attach_branch(self, start: str, name: str, branch: Branch) -> None:
        def branch_writer(ends: list[str]) -> Optional[ChannelWrite]:
            channels = [
                f"branch:{start}:{name}:{end}" if end != END else END for end in ends
            ]
            return ChannelWrite(
                [ChannelWriteEntry(ch) for ch in channels], tags=[TAG_HIDDEN]
            )

        # add hidden start node
        if start == START and start not in self.nodes:
            self.nodes[start] = Channel.subscribe_to(START, tags=[TAG_HIDDEN])

        # attach branch writer
        self.nodes[start] |= branch.run(branch_writer)

        # attach branch readers
        ends = branch.ends.values() if branch.ends else [node for node in self.nodes]
        for end in ends:
            if end != END:
                channel_name = f"branch:{start}:{name}:{end}"
                self.channels[channel_name] = EphemeralValue(Any)
                self.nodes[end].triggers.append(channel_name)
                self.nodes[end].channels.append(channel_name)

    def get_graph(
        self, config: Optional[RunnableConfig] = None, *, xray: bool = False, add_condition_nodes: bool = True
    ) -> RunnableGraph:
        """Returns a drawable representation of the computation graph."""
        graph = RunnableGraph()
        start_nodes: dict[str, RunnableGraphNode] = {
            START: graph.add_node(self.get_input_schema(config), START)
        }
        end_nodes: dict[str, RunnableGraphNode] = {
            END: graph.add_node(self.get_output_schema(config), END)
        }

        for key, node in self.graph.nodes.items():
            if xray:
                subgraph = (
                    node.get_graph(config=config, xray=xray)
                    if isinstance(node, CompiledGraph)
                    else node.get_graph(config=config)
                )
                subgraph.trim_first_node()
                subgraph.trim_last_node()
                if len(subgraph.nodes) > 1:
                    graph.extend(subgraph)
                    start_nodes[key] = subgraph.last_node()
                    end_nodes[key] = subgraph.first_node()
                else:
                    n = graph.add_node(node, key)
                    start_nodes[key] = n
                    end_nodes[key] = n
            else:
                n = graph.add_node(node, key)
                start_nodes[key] = n
                end_nodes[key] = n
        for start, end in sorted(self.graph._all_edges):
            graph.add_edge(start_nodes[start], end_nodes[end])
        for start, branches in self.graph.branches.items():
            for name, branch in branches.items():
                ends = branch.ends or {
                    **{k: k for k in self.graph.nodes},
                    END: END,
                }

                if add_condition_nodes is True: 
                    cond = graph.add_node(branch.condition, name)
                    graph.add_edge(start_nodes[start], cond)
                    for label, end in ends.items():
                        graph.add_edge(cond, end_nodes[end], label)
                else:
                    for label, end in ends.items():
                        graph.add_edge(start_nodes[start], end_nodes[end], label)

        return graph
