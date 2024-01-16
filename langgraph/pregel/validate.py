from typing import Any, Mapping, Sequence, Union

from langgraph.channels.base import BaseChannel
from langgraph.channels.last_value import LastValue
from langgraph.pregel.read import ChannelBatch, ChannelInvoke
from langgraph.pregel.reserved import ReservedChannels


def validate_graph(
    nodes: Mapping[str, Union[ChannelInvoke, ChannelBatch]],
    channels: dict[str, BaseChannel],
    input: Union[str, Sequence[str]],
    output: Union[str, Sequence[str]],
    hidden: Sequence[str],
    interrupt: Sequence[str],
) -> None:
    subscribed_channels = set[str]()
    for node in nodes.values():
        if isinstance(node, ChannelInvoke):
            subscribed_channels.update(node.channels.values())
        elif isinstance(node, ChannelBatch):
            subscribed_channels.add(node.channel)
        else:
            raise TypeError(
                f"Invalid node type {type(node)}, expected Channel.subscribe_to() or Channel.subscribe_to_each()"
            )

    for chan in subscribed_channels:
        if chan not in channels:
            channels[chan] = LastValue(Any)  # type: ignore[arg-type]

    if isinstance(input, str):
        if input not in channels:
            channels[input] = LastValue(Any)  # type: ignore[arg-type]
        if input not in subscribed_channels:
            raise ValueError(f"Input channel {input} is not subscribed to by any node")
    else:
        for chan in input:
            if chan not in channels:
                channels[chan] = LastValue(Any)  # type: ignore[arg-type]
        if all(chan not in subscribed_channels for chan in input):
            raise ValueError(
                f"None of the input channels {input} are subscribed to by any node"
            )

    if isinstance(output, str):
        if output not in channels:
            channels[output] = LastValue(Any)  # type: ignore[arg-type]
    else:
        for chan in output:
            if chan not in channels:
                channels[chan] = LastValue(Any)  # type: ignore[arg-type]

    for chan in ReservedChannels:
        if chan not in channels:
            channels[chan] = LastValue(Any)  # type: ignore[arg-type]

    validate_keys(hidden, channels)
    validate_keys(interrupt, channels)


def validate_keys(
    keys: Union[str, Sequence[str]],
    channels: Mapping[str, BaseChannel],
) -> None:
    if isinstance(keys, str):
        if keys not in channels:
            raise ValueError(f"Key {keys} not in channels")
    else:
        for chan in keys:
            if chan not in channels:
                raise ValueError(f"Key {chan} not in channels")
