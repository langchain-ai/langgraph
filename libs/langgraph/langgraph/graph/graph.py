import logging
from collections import defaultdict
from collections.abc import Awaitable, Hashable, Sequence
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Union,
    cast,
    overload,
)

from langchain_core.runnables import Runnable
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
        """Add a new node to the graph.

        Args:
            node: The function or runnable this node will run.
                If a string is provided, it will be used as the node name, and action will be used as the function or runnable.
            action: The action associated with the node. (default: None)
                Will be used as the node function or runnable if `node` is a string (node name).
            metadata: The metadata associated with the node. (default: None)
        """
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
        """Add a directed edge from the start node to the end node.

        Args:
            start_key: The key of the start node of the edge.
            end_key: The key of the end node of the edge.
        """
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
            source: The starting node. This conditional edge will run when
                exiting this node.
            path: The callable that determines the next
                node or nodes. If not specifying `path_map` it should return one or
                more nodes. If it returns END, the graph will stop execution.
            path_map: Optional mapping of paths to node
                names. If omitted the paths returned by `path` should be node names.
            then: The name of a node to execute after the nodes
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
                f"Branch with name `{path.name}` already exists for node `{source}`"
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
            path: The callable that determines the next
                node or nodes. If not specifying `path_map` it should return one or
                more nodes. If it returns END, the graph will stop execution.
            path_map: Optional mapping of paths to node
                names. If omitted the paths returned by `path` should be node names.
            then: The name of a node to execute after the nodes
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
        """Compiles the graph into a `CompiledGraph` object.

        The compiled graph implements the `Runnable` interface and can be invoked,
        streamed, batched, and run asynchronously.

        Args:
            checkpointer: A checkpoint saver object or flag.
                If provided, this Checkpointer serves as a fully versioned "short-term memory" for the graph,
                allowing it to be paused, resumed, and replayed from any point.
                If None, it may inherit the parent graph's checkpointer when used as a subgraph.
                If False, it will not use or inherit any checkpointer.
            interrupt_before: An optional list of node names to interrupt before.
            interrupt_after: An optional list of node names to interrupt after.
            debug: A flag indicating whether to enable debug mode.
            name: The name to use for the compiled graph.

        Returns:
            CompiledGraph: The compiled graph.
        """
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
            | ChannelWrite([ChannelWriteEntry(key)])
        )
        cast(list[str], self.stream_channels).append(key)

    def attach_edge(self, start: str, end: str) -> None:
        if end == END:
            # publish to end channel
            self.nodes[start].writers.append(ChannelWrite([ChannelWriteEntry(END)]))
        else:
            # subscribe to start channel
            self.nodes[end].triggers.append(start)
            cast(list[str], self.nodes[end].channels).append(start)

    def attach_branch(self, start: str, name: str, branch: Branch) -> None:
        def get_writes(
            packets: Sequence[Union[str, Send]], static: bool = False
        ) -> Sequence[Union[ChannelWriteEntry, Send]]:
            return [
                (
                    ChannelWriteEntry(f"branch:{start}:{name}:{p}" if p != END else END)
                    if not isinstance(p, Send)
                    else p
                )
                for p in packets
            ]

        # add hidden start node
        if start == START and start not in self.nodes:
            self.nodes[start] = Channel.subscribe_to(START, tags=[TAG_HIDDEN])

        # attach branch writer
        self.nodes[start] |= branch.run(get_writes)

        # attach branch readers
        ends = branch.ends.values() if branch.ends else [node for node in self.nodes]
        for end in ends:
            if end != END:
                channel_name = f"branch:{start}:{name}:{end}"
                self.channels[channel_name] = EphemeralValue(Any)
                self.nodes[end].triggers.append(channel_name)
                cast(list[str], self.nodes[end].channels).append(channel_name)
