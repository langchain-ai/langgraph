import asyncio
import logging
from collections import defaultdict
from typing import (
    Any,
    Awaitable,
    Callable,
    Hashable,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    cast,
    overload,
)

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.graph import Graph as DrawableGraph
from langchain_core.runnables.graph import Node as DrawableNode
from typing_extensions import Self

from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.constants import (
    EMPTY_SEQ,
    END,
    NS_END,
    NS_SEP,
    START,
    TAG_HIDDEN,
    Send,
)
from langgraph.graph.branch import Branch
from langgraph.pregel import Channel, Pregel
from langgraph.pregel.protocol import PregelProtocol
from langgraph.pregel.read import PregelNode
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
from langgraph.types import All, Checkpointer
from langgraph.utils.runnable import RunnableLike, coerce_to_runnable

logger = logging.getLogger(__name__)


class NodeSpec(NamedTuple):
    runnable: Runnable
    metadata: Optional[dict[str, Any]] = None
    ends: Optional[Union[tuple[str, ...], dict[str, str]]] = EMPTY_SEQ


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeSpec] = {}
        self.edges = set[tuple[str, str]]()
        self.branches: defaultdict[str, dict[str, Branch]] = defaultdict(dict)
        self.support_multiple_edges = False
        self.compiled = False

    @property
    def _all_edges(self) -> set[tuple[str, str]]:
        return self.edges

    @overload
    def add_node(
        self,
        node: RunnableLike,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Self: ...

    @overload
    def add_node(
        self,
        node: str,
        action: RunnableLike,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Self: ...

    def add_node(
        self,
        node: Union[str, RunnableLike],
        action: Optional[RunnableLike] = None,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Self:
        if isinstance(node, str):
            for character in (NS_SEP, NS_END):
                if character in node:
                    raise ValueError(
                        f"'{character}' is a reserved character and is not allowed in the node names."
                    )

        if self.compiled:
            logger.warning(
                "Adding a node to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        if not isinstance(node, str):
            action = node
            node = getattr(action, "name", getattr(action, "__name__"))
            if node is None:
                raise ValueError(
                    "Node name must be provided if action is not a function"
                )
        if action is None:
            raise RuntimeError(
                "Expected a function or Runnable action in add_node. Received None."
            )
        if node in self.nodes:
            raise ValueError(f"Node `{node}` already present.")
        if node == END or node == START:
            raise ValueError(f"Node `{node}` is reserved.")

        self.nodes[cast(str, node)] = NodeSpec(
            coerce_to_runnable(action, name=cast(str, node), trace=False), metadata
        )
        return self

    def add_edge(self, start_key: str, end_key: str) -> Self:
        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        if start_key == END:
            raise ValueError("END cannot be a start node")
        if end_key == START:
            raise ValueError("START cannot be an end node")

        # run this validation only for non-StateGraph graphs
        if not hasattr(self, "channels") and start_key in set(
            start for start, _ in self.edges
        ):
            raise ValueError(
                f"Already found path for node '{start_key}'.\n"
                "For multiple edges, use StateGraph with an Annotated state key."
            )

        self.edges.add((start_key, end_key))
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: Union[
            Callable[..., Union[Hashable, list[Hashable]]],
            Callable[..., Awaitable[Union[Hashable, list[Hashable]]]],
            Runnable[Any, Union[Hashable, list[Hashable]]],
        ],
        path_map: Optional[Union[dict[Hashable, str], list[str]]] = None,
        then: Optional[str] = None,
    ) -> Self:
        """Add a conditional edge from the starting node to any number of destination nodes.

        Args:
            source (str): The starting node. This conditional edge will run when
                exiting this node.
            path (Union[Callable, Runnable]): The callable that determines the next
                node or nodes. If not specifying `path_map` it should return one or
                more nodes. If it returns END, the graph will stop execution.
            path_map (Optional[dict[Hashable, str]]): Optional mapping of paths to node
                names. If omitted the paths returned by `path` should be node names.
            then (Optional[str]): The name of a node to execute after the nodes
                selected by `path`.

        Returns:
            Self: The instance of the graph, allowing for method chaining.

        Note: Without typehints on the `path` function's return value (e.g., `-> Literal["foo", "__end__"]:`)
            or a path_map, the graph visualization assumes the edge could transition to any node in the graph.

        """  # noqa: E501
        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )

        # find a name for the condition
        path = coerce_to_runnable(path, name=None, trace=True)
        name = path.name or "condition"
        # validate the condition
        if name in self.branches[source]:
            raise ValueError(
                f"Branch with name `{path.name}` already exists for node " f"`{source}`"
            )
        # save it
        self.branches[source][name] = Branch.from_path(path, path_map, then, False)
        return self

    def set_entry_point(self, key: str) -> Self:
        """Specifies the first node to be called in the graph.

        Equivalent to calling `add_edge(START, key)`.

        Parameters:
            key (str): The key of the node to set as the entry point.

        Returns:
            Self: The instance of the graph, allowing for method chaining.
        """
        return self.add_edge(START, key)

    def set_conditional_entry_point(
        self,
        path: Union[
            Callable[..., Union[Hashable, list[Hashable]]],
            Callable[..., Awaitable[Union[Hashable, list[Hashable]]]],
            Runnable[Any, Union[Hashable, list[Hashable]]],
        ],
        path_map: Optional[Union[dict[Hashable, str], list[str]]] = None,
        then: Optional[str] = None,
    ) -> Self:
        """Sets a conditional entry point in the graph.

        Args:
            path (Union[Callable, Runnable]): The callable that determines the next
                node or nodes. If not specifying `path_map` it should return one or
                more nodes. If it returns END, the graph will stop execution.
            path_map (Optional[dict[str, str]]): Optional mapping of paths to node
                names. If omitted the paths returned by `path` should be node names.
            then (Optional[str]): The name of a node to execute after the nodes
                selected by `path`.

        Returns:
            Self: The instance of the graph, allowing for method chaining.
        """
        return self.add_conditional_edges(START, path, path_map, then)

    def set_finish_point(self, key: str) -> Self:
        """Marks a node as a finish point of the graph.

        If the graph reaches this node, it will cease execution.

        Parameters:
            key (str): The key of the node to set as the finish point.

        Returns:
            Self: The instance of the graph, allowing for method chaining.
        """
        return self.add_edge(key, END)

    def validate(self, interrupt: Optional[Sequence[str]] = None) -> Self:
        # assemble sources
        all_sources = {src for src, _ in self._all_edges}
        for start, branches in self.branches.items():
            all_sources.add(start)
            for cond, branch in branches.items():
                if branch.then is not None:
                    if branch.ends is not None:
                        for end in branch.ends.values():
                            if end != END:
                                all_sources.add(end)
                    else:
                        for node in self.nodes:
                            if node != start and node != branch.then:
                                all_sources.add(node)
        for name, spec in self.nodes.items():
            if spec.ends:
                all_sources.add(name)
        # validate sources
        for source in all_sources:
            if source not in self.nodes and source != START:
                raise ValueError(f"Found edge starting at unknown node '{source}'")

        if START not in all_sources:
            raise ValueError(
                "Graph must have an entrypoint: add at least one edge from START to another node"
            )

        # assemble targets
        all_targets = {end for _, end in self._all_edges}
        for start, branches in self.branches.items():
            for cond, branch in branches.items():
                if branch.then is not None:
                    all_targets.add(branch.then)
                if branch.ends is not None:
                    for end in branch.ends.values():
                        if end not in self.nodes and end != END:
                            raise ValueError(
                                f"At '{start}' node, '{cond}' branch found unknown target '{end}'"
                            )
                        all_targets.add(end)
                else:
                    all_targets.add(END)
                    for node in self.nodes:
                        if node != start and node != branch.then:
                            all_targets.add(node)
        for name, spec in self.nodes.items():
            if spec.ends:
                all_targets.update(spec.ends)
        for target in all_targets:
            if target not in self.nodes and target != END:
                raise ValueError(f"Found edge ending at unknown node `{target}`")
        # validate interrupts
        if interrupt:
            for node in interrupt:
                if node not in self.nodes:
                    raise ValueError(f"Interrupt node `{node}` not found")

        self.compiled = True
        return self

    def compile(
        self,
        checkpointer: Checkpointer = None,
        interrupt_before: Optional[Union[All, list[str]]] = None,
        interrupt_after: Optional[Union[All, list[str]]] = None,
        debug: bool = False,
        name: Optional[str] = None,
    ) -> "CompiledGraph":
        # assign default values
        interrupt_before = interrupt_before or []
        interrupt_after = interrupt_after or []

        # validate the graph
        self.validate(
            interrupt=(
                (interrupt_before if interrupt_before != "*" else []) + interrupt_after
                if interrupt_after != "*"
                else []
            )
        )

        # create empty compiled graph
        compiled = CompiledGraph(
            builder=self,
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
            name=name or "LangGraph",
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
    builder: Graph

    def __init__(self, *, builder: Graph, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.builder = builder

    def attach_node(self, key: str, node: NodeSpec) -> None:
        self.channels[key] = EphemeralValue(Any)
        self.nodes[key] = (
            PregelNode(channels=[], triggers=[], metadata=node.metadata)
            | node.runnable
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
            cast(list[str], self.nodes[end].channels).append(start)

    def attach_branch(self, start: str, name: str, branch: Branch) -> None:
        def branch_writer(
            packets: Sequence[Union[str, Send]], config: RunnableConfig
        ) -> Optional[ChannelWrite]:
            writes = [
                (
                    ChannelWriteEntry(f"branch:{start}:{name}:{p}" if p != END else END)
                    if not isinstance(p, Send)
                    else p
                )
                for p in packets
            ]
            return ChannelWrite(
                cast(Sequence[Union[ChannelWriteEntry, Send]], writes),
                tags=[TAG_HIDDEN],
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
                cast(list[str], self.nodes[end].channels).append(channel_name)

    async def aget_graph(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        xray: Union[int, bool] = False,
    ) -> DrawableGraph:
        """Returns a drawable representation of the computation graph."""
        from langgraph.pregel.remote import RemoteGraph

        # gather subgraphs
        if xray:
            subpregels: dict[str, PregelProtocol] = {
                k: v
                async for k, v in self.aget_subgraphs()
                if isinstance(v, (CompiledGraph, RemoteGraph))
            }
            subgraphs = {
                k: v
                for k, v in zip(
                    subpregels,
                    await asyncio.gather(
                        *(
                            p.aget_graph(
                                config,
                                xray=xray
                                if isinstance(xray, bool) or xray <= 0
                                else xray - 1,
                            )
                            for p in subpregels.values()
                        )
                    ),
                )
            }
        else:
            subgraphs = {}

        # draw the graph
        return self._draw_graph(config, subgraphs=subgraphs)

    def get_graph(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        xray: Union[int, bool] = False,
    ) -> DrawableGraph:
        """Returns a drawable representation of the computation graph."""
        from langgraph.pregel.remote import RemoteGraph

        # gather subgraphs
        if xray:
            subgraphs = {
                k: v.get_graph(
                    config,
                    xray=xray if isinstance(xray, bool) or xray <= 0 else xray - 1,
                )
                for k, v in self.get_subgraphs()
                if isinstance(v, (CompiledGraph, RemoteGraph))
            }
        else:
            subgraphs = {}

        # draw the graph
        return self._draw_graph(config, subgraphs=subgraphs)

    def _draw_graph(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        subgraphs: dict[str, DrawableGraph] = {},
    ) -> DrawableGraph:
        # create the graph
        graph = DrawableGraph()
        start_nodes: dict[str, DrawableNode] = {
            START: graph.add_node(self.get_input_schema(config), START)
        }
        end_nodes: dict[str, DrawableNode] = {}

        def add_edge(
            start: str,
            end: str,
            label: Optional[Hashable] = None,
            conditional: bool = False,
        ) -> None:
            if end == END and END not in end_nodes:
                end_nodes[END] = graph.add_node(self.get_output_schema(config), END)
            return graph.add_edge(
                start_nodes[start],
                end_nodes[end],
                str(label) if label is not None else None,
                conditional,
            )

        for key, n in self.builder.nodes.items():
            node = n.runnable
            metadata = n.metadata or {}
            if key in self.interrupt_before_nodes and key in self.interrupt_after_nodes:
                metadata["__interrupt"] = "before,after"
            elif key in self.interrupt_before_nodes:
                metadata["__interrupt"] = "before"
            elif key in self.interrupt_after_nodes:
                metadata["__interrupt"] = "after"
            if key in subgraphs:
                subgraph = subgraphs[key]
                subgraph.trim_first_node()
                subgraph.trim_last_node()
                if len(subgraph.nodes) >= 1:
                    e, s = graph.extend(subgraph, prefix=key)
                    if e is None:
                        raise ValueError(
                            f"Could not extend subgraph '{key}' due to missing entrypoint"
                        )
                    if s is not None:
                        start_nodes[key] = s
                    end_nodes[key] = e
                else:
                    nn = graph.add_node(node, key, metadata=metadata or None)
                    start_nodes[key] = nn
                    end_nodes[key] = nn
            else:
                nn = graph.add_node(node, key, metadata=metadata or None)
                start_nodes[key] = nn
                end_nodes[key] = nn
        for start, end in sorted(self.builder._all_edges):
            add_edge(start, end)
        for start, branches in self.builder.branches.items():
            default_ends = {
                **{k: k for k in self.builder.nodes if k != start},
                END: END,
            }
            for _, branch in branches.items():
                if branch.ends is not None:
                    ends = branch.ends
                elif branch.then is not None:
                    ends = {k: k for k in default_ends if k not in (END, branch.then)}
                else:
                    ends = cast(dict[Hashable, str], default_ends)
                for label, end in ends.items():
                    add_edge(
                        start,
                        end,
                        label if label != end else None,
                        conditional=True,
                    )
                    if branch.then is not None:
                        add_edge(end, branch.then)
        for key, n in self.builder.nodes.items():
            if isinstance(n.ends, dict):
                for end, label in n.ends.items():
                    add_edge(key, end, label, conditional=True)
            elif isinstance(n.ends, tuple):
                for end in n.ends:
                    add_edge(key, end, conditional=True)

        return graph

    def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any]:
        """Mime bundle used by Jupyter to display the graph"""
        return {
            "text/plain": repr(self),
            "image/png": self.get_graph().draw_mermaid_png(),
        }
