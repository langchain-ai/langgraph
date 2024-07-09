from typing import Iterable, Mapping, Optional, Sequence, Union

from langgraph.channels.base import BaseChannel
from langgraph.constants import RESERVED
from langgraph.pregel.read import PregelNode
from langgraph.pregel.types import All


def validate_graph(
    nodes: Mapping[str, PregelNode],
    channels: dict[str, BaseChannel],
    input_channels: Union[str, Sequence[str]],
    output_channels: Union[str, Sequence[str]],
    stream_channels: Optional[Union[str, Sequence[str]]],
    interrupt_after_nodes: Union[All, Sequence[str]],
    interrupt_before_nodes: Union[All, Sequence[str]],
) -> None:
    for chan in channels:
        if chan in RESERVED:
            raise ValueError(f"Channel names {RESERVED} are reserved")

    subscribed_channels = set[str]()
    for name, node in nodes.items():
        if name in RESERVED:
            raise ValueError(f"Node names {RESERVED} are reserved")
        if isinstance(node, PregelNode):
            subscribed_channels.update(node.triggers)
        else:
            raise TypeError(
                f"Invalid node type {type(node)}, expected Channel.subscribe_to()"
            )

    for chan in subscribed_channels:
        if chan not in channels:
            raise ValueError(f"Subscribed channel '{chan}' not in 'channels'")

    if isinstance(input_channels, str):
        if input_channels not in channels:
            raise ValueError(f"Input channel '{input_channels}' not in 'channels'")
        if input_channels not in subscribed_channels:
            raise ValueError(
                f"Input channel {input_channels} is not subscribed to by any node"
            )
    else:
        for chan in input_channels:
            if chan not in channels:
                raise ValueError(f"Input channel '{chan}' not in 'channels'")
        if all(chan not in subscribed_channels for chan in input_channels):
            raise ValueError(
                f"None of the input channels {input_channels} are subscribed to by any node"
            )

    all_output_channels = set[str]()
    if isinstance(output_channels, str):
        all_output_channels.add(output_channels)
    else:
        all_output_channels.update(output_channels)
    if isinstance(stream_channels, str):
        all_output_channels.add(stream_channels)
    elif stream_channels is not None:
        all_output_channels.update(stream_channels)

    for chan in all_output_channels:
        if chan not in channels:
            raise ValueError(f"Output channel '{chan}' not in 'channels'")

    if interrupt_after_nodes != "*":
        for node in interrupt_after_nodes:
            if node not in nodes:
                raise ValueError(f"Node {node} not in nodes")
    if interrupt_before_nodes != "*":
        for node in interrupt_before_nodes:
            if node not in nodes:
                raise ValueError(f"Node {node} not in nodes")


def validate_keys(
    keys: Optional[Union[str, Sequence[str]]],
    valid: Iterable[str],
) -> None:
    if isinstance(keys, str):
        if keys not in valid:
            raise ValueError(f"Key {keys} not in list")
    elif keys is not None:
        for chan in keys:
            if chan not in valid:
                raise ValueError(f"Key {chan} not in list")
