import logging
from collections import defaultdict
from functools import partial
from inspect import signature
from typing import Any, Optional, Sequence, Type

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import RunnableLike

from langgraph.channels.any_value import AnyValue
from langgraph.channels.base import BaseChannel, InvalidUpdateError
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.channels.named_barrier_value import NamedBarrierValue
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph.graph import END, START, CompiledGraph, Graph
from langgraph.pregel import Channel
from langgraph.pregel.read import ChannelInvoke
from langgraph.pregel.write import SKIP_WRITE, ChannelWrite, ChannelWriteEntry

logger = logging.getLogger(__name__)


class StateGraph(Graph):
    def __init__(self, schema: Type[Any]) -> None:
        super().__init__()
        self.schema = schema
        self.channels = _get_channels(schema)
        if any(isinstance(c, BinaryOperatorAggregate) for c in self.channels.values()):
            self.support_multiple_edges = True
        self.w_edges: set[tuple[tuple[str, ...], str]] = set()

    def add_node(self, key: str, action: RunnableLike) -> None:
        if key in self.channels:
            raise ValueError(
                f"'{key}' is already being used as a state attribute "
                "(a.k.a. a channel), cannot also be used as a node name."
            )
        return super().add_node(key, action)

    def add_waiting_edge(self, starts: Sequence[str], end: str) -> None:
        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        for start in starts:
            if start == END:
                raise ValueError("END cannot be a start node")
            if start not in self.nodes:
                raise ValueError(f"Need to add_node `{start}` first")
        if end == END:
            raise ValueError("END cannot be an end node")
        if end not in self.nodes:
            raise ValueError(f"Need to add_node `{end}` first")

        self.w_edges.add((tuple(starts), end))

    def compile(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[Sequence[str]] = None,
        interrupt_after: Optional[Sequence[str]] = None,
        debug: bool = False,
    ) -> CompiledGraph:
        interrupt_before = interrupt_before or []
        interrupt_after = interrupt_after or []
        self.validate(
            interrupt=interrupt_before + interrupt_after,
            additional_edges={
                (start, end) for starts, end in self.w_edges for start in starts
            },
        )

        state_keys = list(self.channels)
        state_keys_read = state_keys[0] if state_keys == ["__root__"] else state_keys
        state_channels = (
            {chan: chan for chan in state_keys}
            if isinstance(state_keys_read, list)
            else {None: state_keys_read}
        )
        update_channels = (
            [ChannelWriteEntry("__root__", None, True)]
            if not isinstance(state_keys_read, list)
            else [
                ChannelWriteEntry(
                    key, RunnableLambda(partial(_dict_getter, state_keys, key)), False
                )
                for key in state_keys_read
            ]
        )
        coerce_state = (
            partial(_coerce_state, self.schema)
            if isinstance(state_keys_read, list)
            else None
        )

        waiting_edges = {
            (f"{starts}:{end}", starts, end) for starts, end in self.w_edges
        }
        waiting_edge_channels = {
            key: NamedBarrierValue(str, set(starts)) for key, starts, _ in waiting_edges
        }

        outgoing_edges = defaultdict(list)
        for start, end in self.edges:
            outgoing_edges[start].append(f"{end}:inbox" if end != END else END)
        for key, starts, end in waiting_edges:
            for start in starts:
                outgoing_edges[start].append(key)

        nodes = {
            key: (
                ChannelInvoke(
                    triggers=[
                        f"{key}:inbox",
                        *[chan for chan, _, end in waiting_edges if end == key],
                    ],
                    channels=state_channels,
                    mapper=coerce_state,
                )
                | node
                | ChannelWrite(
                    channels=[ChannelWriteEntry(key, None, False)] + update_channels
                )
            )
            for key, node in self.nodes.items()
        }
        node_inboxes = {
            # we take any value written to channel because all writers
            # write the entire state as of that step, which is equal for all
            f"{key}:inbox": AnyValue(self.schema)
            for key in list(self.nodes) + [START]
        }
        node_outboxes = {
            # we clear outbox channels after each step
            key: EphemeralValue(Any)
            for key in list(self.nodes) + [START]
        }

        for key in self.nodes:
            outgoing = outgoing_edges[key]
            edges_key = f"{key}:edges"
            if outgoing or key in self.branches:
                nodes[edges_key] = ChannelInvoke(
                    triggers=[key], tags=["langsmith:hidden"], channels=state_channels
                )
            if outgoing:
                nodes[edges_key] |= ChannelWrite(
                    channels=[
                        ChannelWriteEntry(dest, None if dest == END else key, True)
                        for dest in outgoing
                    ]
                )
            if key in self.branches:
                for branch in self.branches[key]:
                    nodes[edges_key] |= RunnableLambda(
                        branch.runnable, name=f"{key}_condition"
                    )

        nodes[START] = Channel.subscribe_to(
            f"{START}:inbox", tags=["langsmith:hidden"]
        ) | ChannelWrite(
            channels=[ChannelWriteEntry(START, None, False)] + update_channels
        )
        nodes[f"{START}:edges"] = ChannelInvoke(
            triggers=[START], tags=["langsmith:hidden"], channels=state_channels
        )
        if self.entry_point:
            nodes[f"{START}:edges"] |= Channel.write_to(f"{self.entry_point}:inbox")
        elif self.entry_point_branch:
            nodes[f"{START}:edges"] |= RunnableLambda(
                self.entry_point_branch.runnable, name=f"{START}_condition"
            )
        else:
            raise ValueError("No entry point set")

        return CompiledGraph(
            graph=self,
            nodes=nodes,
            channels={
                **self.channels,
                **node_inboxes,
                **node_outboxes,
                **waiting_edge_channels,
                END: LastValue(self.schema),
            },
            input=f"{START}:inbox",
            output=END,
            hidden=[f"{node}:inbox" for node in self.nodes] + [START] + state_keys,
            snapshot_channels=state_keys_read,
            checkpointer=checkpointer,
            interrupt_before_nodes=[f"{node}:inbox" for node in interrupt_before],
            interrupt_after_nodes=interrupt_after,
            debug=debug,
        )


def _coerce_state(schema: Type[Any], input: dict[str, Any]) -> dict[str, Any]:
    return schema(**input)


def _dict_getter(allowed_keys: list[str], key: str, input: dict) -> Any:
    if input is not None:
        if not isinstance(input, dict) or any(key not in allowed_keys for key in input):
            raise InvalidUpdateError(
                f"Invalid state update,"
                f" expected dict with one or more of {allowed_keys}, got {input}"
            )
        return input.get(key, SKIP_WRITE)
    else:
        return SKIP_WRITE


def _get_channels(schema: Type[dict]) -> dict[str, BaseChannel]:
    if not hasattr(schema, "__annotations__"):
        return {
            "__root__": _get_channel(schema),
        }

    channels: dict[str, BaseChannel] = {}
    for name, typ in schema.__annotations__.items():
        channels[name] = _get_channel(typ)

    return channels


def _get_channel(annotation: Any) -> Optional[BaseChannel]:
    if channel := _is_field_binop(annotation):
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
