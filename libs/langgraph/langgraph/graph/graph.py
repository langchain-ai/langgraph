import logging
from collections import defaultdict
from typing import (
    Any,
    Awaitable,
    Callable,
    Hashable,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from langchain_core.runnables import Runnable
from langchain_core.runnables.base import RunnableLike
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.graph import Graph as DrawableGraph
from langchain_core.runnables.graph import Node as DrawableNode

from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import (
    END,
    NS_END,
    NS_SEP,
    START,
    TAG_HIDDEN,
    Send,
)
from langgraph.errors import InvalidUpdateError
from langgraph.pregel import Channel, Pregel
from langgraph.pregel.read import PregelNode
from langgraph.pregel.types import All
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
from langgraph.utils.runnable import RunnableCallable, coerce_to_runnable

logger = logging.getLogger(__name__)


class NodeSpec(NamedTuple):
    runnable: Runnable
    metadata: Optional[dict[str, Any]] = None


class Branch(NamedTuple):
    path: Runnable[Any, Union[Hashable, list[Hashable]]]
    ends: Optional[dict[Hashable, str]]
    then: Optional[str] = None

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
        if reader:
            value = reader(config)
            # passthrough additional keys from node to branch
            # only doable when using dict states
            if isinstance(value, dict) and isinstance(input, dict):
                value = {**input, **value}
        else:
            value = input
        result = self.path.invoke(value, config)
        return self._finish(writer, input, result)

    async def _aroute(
        self,
        input: Any,
        config: RunnableConfig,
        *,
        reader: Optional[Callable[[], Any]],
        writer: Callable[[list[str]], Optional[Runnable]],
    ) -> Runnable:
        if reader:
            value = reader(config)
            # passthrough additional keys from node to branch
            # only doable when using dict states
            if isinstance(value, dict) and isinstance(input, dict):
                value = {**input, **value}
        else:
            value = input
        result = await self.path.ainvoke(value, config)
        return self._finish(writer, input, result)

    def _finish(
        self, writer: Callable[[list[str]], Optional[Runnable]], input: Any, result: Any
    ):
        if not isinstance(result, list):
            result = [result]
        if self.ends:
            destinations = [r if isinstance(r, Send) else self.ends[r] for r in result]
        else:
            destinations = result
        if any(dest is None or dest == START for dest in destinations):
            raise ValueError("Branch did not return a valid destination")
        if any(p.node == END for p in destinations if isinstance(p, Send)):
            raise InvalidUpdateError("Cannot send a packet to the END node")
        return writer(destinations) or input


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
    ) -> None: ...

    @overload
    def add_node(
        self,
        node: str,
        action: RunnableLike,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None: ...

    def add_node(
        self,
        node: Union[str, RunnableLike],
        action: Optional[RunnableLike] = None,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
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
            node = getattr(action, "name", action.__name__)
        if node in self.nodes:
            raise ValueError(f"Node `{node}` already present.")
        if node == END or node == START:
            raise ValueError(f"Node `{node}` is reserved.")

        self.nodes[node] = NodeSpec(
            coerce_to_runnable(action, name=node, trace=False), metadata
        )

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

        # run this validation only for non-StateGraph graphs
        if not hasattr(self, "channels") and start_key in set(
            start for start, _ in self.edges
        ):
            raise ValueError(
                f"Already found path for node '{start_key}'.\n"
                "For multiple edges, use StateGraph with an Annotated state key."
            )

        self.edges.add((start_key, end_key))

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
    ) -> None:
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
            None

        Note: Without typehints on the `path` function's return value (e.g., `-> Literal["foo", "__end__"]:`)
            or a path_map, the graph visualization assumes the edge could transition to any node in the graph.

        """  # noqa: E501
        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        # coerce path_map to a dictionary
        try:
            if isinstance(path_map, dict):
                path_map = path_map.copy()
            elif isinstance(path_map, list):
                path_map = {name: name for name in path_map}
            elif rtn_type := get_type_hints(path.__call__).get(
                "return"
            ) or get_type_hints(path).get("return"):
                if get_origin(rtn_type) is Literal:
                    path_map = {name: name for name in get_args(rtn_type)}
        except Exception:
            pass
        # find a name for the condition
        path = coerce_to_runnable(path, name=None, trace=True)
        name = path.name or "condition"
        # validate the condition
        if name in self.branches[source]:
            raise ValueError(
                f"Branch with name `{path.name}` already exists for node " f"`{source}`"
            )
        # save it
        self.branches[source][name] = Branch(path, path_map, then)

    def set_entry_point(self, key: str) -> None:
        """Specifies the first node to be called in the graph.

        Equivalent to calling `add_edge(START, key)`.

        Parameters:
            key (str): The key of the node to set as the entry point.

        Returns:
            None
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
    ) -> None:
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
            None
        """
        return self.add_conditional_edges(START, path, path_map, then)

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
        # validate sources
        for source in all_sources:
            if source not in self.nodes and source != START:
                raise ValueError(f"Found edge starting at unknown node '{source}'")

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
        # validate targets
        for node in self.nodes:
            if node not in all_targets:
                raise ValueError(f"Node `{node}` is not reachable")
        for target in all_targets:
            if target not in self.nodes and target != END:
                raise ValueError(f"Found edge ending at unknown node `{target}`")
        # validate interrupts
        if interrupt:
            for node in interrupt:
                if node not in self.nodes:
                    raise ValueError(f"Interrupt node `{node}` not found")

        self.compiled = True

    def compile(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: bool = False,
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

    def __init__(self, *, builder: Graph, **kwargs):
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
            self.nodes[end].channels.append(start)

    def attach_branch(self, start: str, name: str, branch: Branch) -> None:
        def branch_writer(packets: list[Union[str, Send]]) -> Optional[ChannelWrite]:
            writes = [
                (
                    ChannelWriteEntry(f"branch:{start}:{name}:{p}" if p != END else END)
                    if not isinstance(p, Send)
                    else p
                )
                for p in packets
            ]
            return ChannelWrite(writes, tags=[TAG_HIDDEN])

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
        self,
        config: Optional[RunnableConfig] = None,
        *,
        xray: Union[int, bool] = False,
    ) -> DrawableGraph:
        """Returns a drawable representation of the computation graph."""
        graph = DrawableGraph()
        start_nodes: dict[str, DrawableNode] = {
            START: graph.add_node(self.get_input_schema(config), START)
        }
        end_nodes: dict[str, DrawableNode] = {}
        if xray:
            subgraphs = dict(self.get_subgraphs())
        else:
            subgraphs = {}

        def add_edge(
            start: str, end: str, label: Optional[str] = None, conditional: bool = False
        ) -> None:
            if end == END and END not in end_nodes:
                end_nodes[END] = graph.add_node(self.get_output_schema(config), END)
            return graph.add_edge(
                start_nodes[start], end_nodes[end], label, conditional
            )

        for key, n in self.builder.nodes.items():
            node = n.runnable
            metadata = n.metadata or {}
            if key in self.interrupt_before_nodes:
                metadata["__interrupt"] = "before"
            elif key in self.interrupt_after_nodes:
                metadata["__interrupt"] = "after"
            if xray:
                subgraph = (
                    subgraphs[key].get_graph(
                        config=config,
                        xray=xray - 1 if isinstance(xray, int) and xray > 0 else xray,
                    )
                    if key in subgraphs
                    else node.get_graph(config=config)
                )
                subgraph.trim_first_node()
                subgraph.trim_last_node()
                if len(subgraph.nodes) > 1:
                    end_nodes[key], start_nodes[key] = graph.extend(
                        subgraph, prefix=key
                    )
                else:
                    n = graph.add_node(node, key, metadata=metadata or None)
                    start_nodes[key] = n
                    end_nodes[key] = n
            else:
                n = graph.add_node(node, key, metadata=metadata or None)
                start_nodes[key] = n
                end_nodes[key] = n
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
                    ends = default_ends
                for label, end in ends.items():
                    add_edge(
                        start,
                        end,
                        label if label != end else None,
                        conditional=True,
                    )
                    if branch.then is not None:
                        add_edge(end, branch.then)

        return graph
