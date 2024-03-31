from typing import Any, Mapping, Optional, Sequence, Union

from langgraph.channels.base import BaseChannel
from langgraph.channels.last_value import LastValue
from langgraph.constants import INTERRUPT
from langgraph.pregel.read import ChannelInvoke
from langgraph.pregel.reserved import ReservedChannels


def validate_graph(
    nodes: Mapping[str, ChannelInvoke],
    channels: dict[str, BaseChannel],
    input_channels: Union[str, Sequence[str]],
    output_channels: Union[str, Sequence[str]],
    stream_channels: Optional[Union[str, Sequence[str]]],
    interrupt_after_nodes: Sequence[str],
    interrupt_before_nodes: Sequence[str],
) -> None:
    subscribed_channels = set[str]()
    for name, node in nodes.items():
        if name == INTERRUPT:
            raise ValueError(f"Node name {INTERRUPT} is reserved")
        if isinstance(node, ChannelInvoke):
            subscribed_channels.update(node.channels.values())
        else:
            raise TypeError(
                f"Invalid node type {type(node)}, expected Channel.subscribe_to()"
            )

    for chan in subscribed_channels:
        if chan not in channels:
            channels[chan] = LastValue(Any)  # type: ignore[arg-type]

    if isinstance(input_channels, str):
        if input_channels not in channels:
            channels[input_channels] = LastValue(Any)  # type: ignore[arg-type]
        if input_channels not in subscribed_channels:
            raise ValueError(
                f"Input channel {input_channels} is not subscribed to by any node"
            )
    else:
        for chan in input_channels:
            if chan not in channels:
                channels[chan] = LastValue(Any)  # type: ignore[arg-type]
        if all(chan not in subscribed_channels for chan in input_channels):
            raise ValueError(
                f"None of the input channels {input_channels} are subscribed to by any node"
            )

    if isinstance(output_channels, str):
        if output_channels not in channels:
            channels[output_channels] = LastValue(Any)  # type: ignore[arg-type]
    else:
        for chan in output_channels:
            if chan not in channels:
                channels[chan] = LastValue(Any)  # type: ignore[arg-type]

    for chan in ReservedChannels:
        if chan not in channels:
            channels[chan] = LastValue(Any)  # type: ignore[arg-type]

    validate_keys(stream_channels, channels)
    validate_keys(interrupt_after_nodes, channels)
    validate_keys(interrupt_before_nodes, channels)


def validate_keys(
    keys: Optional[Union[str, Sequence[str]]],
    channels: Mapping[str, BaseChannel],
) -> None:
    if isinstance(keys, str):
        if keys not in channels:
            raise ValueError(f"Key {keys} not in channels")
    elif keys is not None:
        for chan in keys:
            if chan not in channels:
                raise ValueError(f"Key {chan} not in channels")
