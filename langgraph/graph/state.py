from collections import defaultdict
from functools import partial
from inspect import signature
from typing import Any, Optional, Type

from langchain_core.runnables import RunnableConfig, RunnableLambda

from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.last_value import LastValue
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph.graph import END, Graph
from langgraph.pregel import Channel, Pregel
from langgraph.pregel.read import ChannelRead
from langgraph.pregel.write import ChannelWrite

START = "__start__"


class StateGraph(Graph):
    def __init__(self, schema: Type[Any]) -> None:
        super().__init__()
        self.schema = schema
        self.channels = _get_channels(schema)
        if any(isinstance(c, BinaryOperatorAggregate) for c in self.channels.values()):
            self.support_multiple_edges = True

    def compile(self, checkpointer: Optional[BaseCheckpointSaver] = None) -> Pregel:
        self.validate()

        if any(key in self.nodes for key in self.channels):
            raise ValueError("Cannot use channel names as node names")

        state_keys = list(self.channels)

        outgoing_edges = defaultdict(list)
        for start, end in self.edges:
            outgoing_edges[start].append(f"{end}:inbox" if end != END else END)

        nodes = {
            key: (
                Channel.subscribe_to(f"{key}:inbox")
                | partial(_coerce_state, self.schema)  # coerce/validate using schema
                | node
                | _update_state
                | Channel.write_to(key)
            )
            for key, node in self.nodes.items()
        }

        for key in self.nodes:
            outgoing = outgoing_edges[key]
            edges_key = f"{key}:edges"
            if outgoing or key in self.branches:
                nodes[edges_key] = Channel.subscribe_to(
                    key, tags=["langsmith:hidden"]
                ) | ChannelRead(state_keys)
            if outgoing:
                nodes[edges_key] |= Channel.write_to(*[dest for dest in outgoing])
            if key in self.branches:
                for branch in self.branches[key]:
                    nodes[edges_key] |= RunnableLambda(
                        branch.runnable, name=f"{key}_condition"
                    )

        nodes[START] = (
            Channel.subscribe_to(f"{START}:inbox", tags=["langsmith:hidden"])
            | _update_state
            | Channel.write_to(START)
        )
        nodes[f"{START}:edges"] = (
            Channel.subscribe_to(START, tags=["langsmith:hidden"])
            | ChannelRead(state_keys)
            | Channel.write_to(f"{self.entry_point}:inbox")
        )

        return Pregel(
            nodes=nodes,
            channels=self.channels,
            input=f"{START}:inbox",
            output=END,
            hidden=[f"{node}:inbox" for node in self.nodes] + [START] + state_keys,
            checkpointer=checkpointer,
        )


def _coerce_state(schema: Type[Any], input: dict[str, Any]) -> dict[str, Any]:
    return schema(**input)


def _update_state(input: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    if input is not None:
        ChannelWrite.do_write(config, **input)
    return input


def _get_channels(schema: Type[dict]) -> dict[str, BaseChannel]:
    if not hasattr(schema, "__annotations__"):
        raise ValueError("Schema must be a class with type annotations")

    channels: dict[str, BaseChannel] = {}
    for name, typ in schema.__annotations__.items():
        if channel := _is_field_binop(typ):
            channels[name] = channel
        else:
            channels[name] = LastValue(typ)

    return channels


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
