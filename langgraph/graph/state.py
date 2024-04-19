import logging
from functools import partial
from inspect import signature
from typing import Any, Optional, Sequence, Type, Union, get_type_hints

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.base import RunnableLike

from langgraph.channels.base import BaseChannel, InvalidUpdateError
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.dynamic_barrier_value import DynamicBarrierValue, WaitForNames
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.channels.named_barrier_value import NamedBarrierValue
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.constants import TAG_HIDDEN
from langgraph.graph.graph import END, START, Branch, CompiledGraph, Graph
from langgraph.managed.base import ManagedValue, is_managed_value
from langgraph.pregel.read import ChannelRead, PregelNode
from langgraph.pregel.write import SKIP_WRITE, ChannelWrite, ChannelWriteEntry
from langgraph.utils import RunnableCallable

logger = logging.getLogger(__name__)


class StateGraph(Graph):
    """A graph whose nodes communicate by reading and writing to a shared state.
    The signature of each node is State -> Partial<State>.

    Each state key can optionally be annotated with a reducer function that
    will be used to aggregate the values of that key received from multiple nodes.
    The signature of a reducer function is (Value, Value) -> Value.
    """

    def __init__(self, schema: Type[Any]) -> None:
        super().__init__()
        self.schema = schema
        self.channels, self.managed = _get_channels(schema)
        if any(isinstance(c, BinaryOperatorAggregate) for c in self.channels.values()):
            self.support_multiple_edges = True
        self.waiting_edges: set[tuple[tuple[str, ...], str]] = set()

    @property
    def _all_edges(self) -> set[tuple[str, str]]:
        return self.edges | {
            (start, end) for starts, end in self.waiting_edges for start in starts
        }

    def add_node(self, key: str, action: RunnableLike) -> None:
        """Adds a new node to the state graph.

        Args:
            key (str): The key of the node.
            action (RunnableLike): The action associated with the node.

        Raises:
            ValueError: If the key is already being used as a state key.

        Returns:
            None
        """
        if key in self.channels:
            raise ValueError(f"'{key}' is already being used as a state key")
        return super().add_node(key, action)

    def add_edge(self, start_key: Union[str, list[str]], end_key: str) -> None:
        """Adds a directed edge from the start node to the end node.

        If the graph transitions to the start_key node, it will always transition to the end_key node next.

        Args:
            start_key (Union[str, list[str]]): The key(s) of the start node(s) of the edge.
            end_key (str): The key of the end node of the edge.

        Raises:
            ValueError: If the start key is 'END' or if the start key or end key is not present in the graph.

        Returns:
            None
        """
        if isinstance(start_key, str):
            return super().add_edge(start_key, end_key)

        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        for start in start_key:
            if start == END:
                raise ValueError("END cannot be a start node")
            if start not in self.nodes:
                raise ValueError(f"Need to add_node `{start}` first")
        if end_key == END:
            raise ValueError("END cannot be an end node")
        if end_key not in self.nodes:
            raise ValueError(f"Need to add_node `{end_key}` first")

        self.waiting_edges.add((tuple(start_key), end_key))

    def compile(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[Sequence[str]] = None,
        interrupt_after: Optional[Sequence[str]] = None,
        debug: bool = False,
    ) -> CompiledGraph:
        """Compiles the state graph into a `CompiledGraph` object.

        Args:
            checkpointer (Optional[BaseCheckpointSaver]): An optional checkpoint saver object.
            interrupt_before (Optional[Sequence[str]]): An optional list of node names to interrupt before.
            interrupt_after (Optional[Sequence[str]]): An optional list of node names to interrupt after.
            debug (bool): A flag indicating whether to enable debug mode.

        Returns:
            CompiledGraph: The compiled state graph.
        """
        # assign default values
        interrupt_before = interrupt_before or []
        interrupt_after = interrupt_after or []

        # validate the graph
        self.validate(interrupt=interrupt_before + interrupt_after)

        # prepare output channels
        state_keys = list(self.channels)
        output_channels = state_keys[0] if state_keys == ["__root__"] else state_keys

        compiled = CompiledStateGraph(
            graph=self,
            nodes={},
            channels={**self.channels, START: EphemeralValue(self.schema)},
            input_channels=START,
            stream_mode="updates",
            output_channels=output_channels,
            stream_channels=output_channels,
            checkpointer=checkpointer,
            interrupt_before_nodes=interrupt_before,
            interrupt_after_nodes=interrupt_after,
            auto_validate=False,
            debug=debug,
        )

        compiled.attach_node(START, None)
        for key, node in self.nodes.items():
            compiled.attach_node(key, node)

        for start, end in self.edges:
            compiled.attach_edge(start, end)

        for starts, end in self.waiting_edges:
            compiled.attach_edge(starts, end)

        for start, branches in self.branches.items():
            for name, branch in branches.items():
                compiled.attach_branch(start, name, branch)

        return compiled.validate()


class CompiledStateGraph(CompiledGraph):
    graph: StateGraph

    def attach_node(self, key: str, node: Optional[Runnable]) -> None:
        state_keys = list(self.graph.channels)

        def _get_state_key(input: dict, config: RunnableConfig, *, key: str) -> Any:
            if input is None:
                return SKIP_WRITE
            elif not isinstance(input, dict):
                raise InvalidUpdateError(f"Expected dict, got {input}")
            elif key not in state_keys:
                raise InvalidUpdateError(f"Expected one of {state_keys}, got {key}")
            else:
                return input.get(key, SKIP_WRITE)

        # state updaters
        state_write_entries = [
            (
                ChannelWriteEntry(key, skip_none=True)
                if key == "__root__"
                else ChannelWriteEntry(
                    key,
                    mapper=RunnableCallable(
                        _get_state_key, key=key, trace=False, recurse=False
                    ),
                )
            )
            for key in state_keys
        ]

        # add node and output channel
        if key == START:
            self.nodes[key] = PregelNode(
                tags=[TAG_HIDDEN],
                triggers=[START],
                channels=[START],
                writers=[
                    ChannelWrite(state_write_entries, tags=[TAG_HIDDEN]),
                ],
            )
        else:
            self.channels[key] = EphemeralValue(Any)
            self.nodes[key] = PregelNode(
                triggers=[],
                # read state keys and managed values
                channels=(
                    state_keys
                    if state_keys == ["__root__"]
                    else ({chan: chan for chan in state_keys} | self.graph.managed)
                ),
                # coerce state dict to schema class (eg. pydantic model)
                mapper=(
                    None
                    if state_keys == ["__root__"]
                    else partial(_coerce_state, self.graph.schema)
                ),
                writers=[
                    # publish to this channel and state keys
                    ChannelWrite(
                        [ChannelWriteEntry(key, key)] + state_write_entries,
                        tags=[TAG_HIDDEN],
                    ),
                ],
            ).pipe(node)

    def attach_edge(self, starts: Union[str, Sequence[str]], end: str) -> None:
        if isinstance(starts, str):
            if starts == START:
                channel_name = f"start:{end}"
                # register channel
                self.channels[channel_name] = EphemeralValue(Any)
                # subscribe to channel
                self.nodes[end].triggers.append(channel_name)
                # publish to channel
                self.nodes[START] |= ChannelWrite(
                    [ChannelWriteEntry(channel_name, START)], tags=[TAG_HIDDEN]
                )
            elif end != END:
                # subscribe to start channel
                self.nodes[end].triggers.append(starts)
        else:
            channel_name = f"join:{'+'.join(starts)}:{end}"
            # register channel
            self.channels[channel_name] = NamedBarrierValue(str, set(starts))
            # subscribe to channel
            self.nodes[end].triggers.append(channel_name)
            # publish to channel
            for start in starts:
                self.nodes[start] |= ChannelWrite(
                    [ChannelWriteEntry(channel_name, start)], tags=[TAG_HIDDEN]
                )

    def attach_branch(self, start: str, name: str, branch: Branch) -> None:
        def branch_writer(ends: list[str]) -> Optional[ChannelWrite]:
            if filtered_ends := [end for end in ends if end != END]:
                writes = [
                    ChannelWriteEntry(f"branch:{start}:{name}:{end}", start)
                    for end in filtered_ends
                ]
                if branch.then and branch.then != END:
                    writes.append(
                        ChannelWriteEntry(
                            f"branch:{start}:{name}:then",
                            WaitForNames(set(filtered_ends)),
                        )
                    )
                return ChannelWrite(writes, tags=[TAG_HIDDEN])

        # attach branch publisher
        self.nodes[start] |= branch.run(branch_writer, _get_state_reader(self.graph))

        # attach branch subscribers
        ends = (
            branch.ends.values()
            if branch.ends
            else [node for node in self.graph.nodes if node != branch.then]
        )
        for end in ends:
            if end != END:
                channel_name = f"branch:{start}:{name}:{end}"
                self.channels[channel_name] = EphemeralValue(Any)
                self.nodes[end].triggers.append(channel_name)

        # attach then subscriber
        if branch.then and branch.then != END:
            channel_name = f"branch:{start}:{name}:then"
            self.channels[channel_name] = DynamicBarrierValue(str)
            self.nodes[branch.then].triggers.append(channel_name)
            for end in ends:
                if end != END:
                    self.nodes[end] |= ChannelWrite(
                        [ChannelWriteEntry(channel_name, end)], tags=[TAG_HIDDEN]
                    )


def _get_state_reader(graph: StateGraph) -> ChannelRead:
    state_keys = list(graph.channels)
    return partial(
        ChannelRead.do_read,
        channel=state_keys[0] if state_keys == ["__root__"] else state_keys,
        fresh=True,
        # coerce state dict to schema class (eg. pydantic model)
        mapper=(
            None if state_keys == ["__root__"] else partial(_coerce_state, graph.schema)
        ),
    )


def _coerce_state(schema: Type[Any], input: dict[str, Any]) -> dict[str, Any]:
    return schema(**input)


def _get_channels(
    schema: Type[dict],
) -> tuple[dict[str, BaseChannel], dict[str, Type[ManagedValue]]]:
    if not hasattr(schema, "__annotations__"):
        return {"__root__": _get_channel(schema, allow_managed=False)}, {}

    all_keys = {
        name: _get_channel(typ)
        for name, typ in get_type_hints(schema, include_extras=True).items()
        if name != "__slots__"
    }
    return (
        {k: v for k, v in all_keys.items() if not is_managed_value(v)},
        {k: v for k, v in all_keys.items() if is_managed_value(v)},
    )


def _get_channel(
    annotation: Any, *, allow_managed: bool = True
) -> Union[BaseChannel, Type[ManagedValue]]:
    if manager := _is_field_managed_value(annotation):
        if allow_managed:
            return manager
        else:
            raise ValueError(f"This {annotation} not allowed in this position")
    elif channel := _is_field_binop(annotation):
        return channel
    return LastValue(annotation)


def _is_field_binop(typ: Type[Any]) -> Optional[BinaryOperatorAggregate]:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) == 1 and callable(meta[0]):
            sig = signature(meta[0])
            params = list(sig.parameters.values())
            if len(params) == 2 and len(
                [
                    p
                    for p in params
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
            ):
                return BinaryOperatorAggregate(typ, meta[0])
    return None


def _is_field_managed_value(typ: Type[Any]) -> Optional[Type[ManagedValue]]:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) == 1 and is_managed_value(meta[0]):
            return meta[0]

    return None
