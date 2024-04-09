from typing import Any, Mapping, Optional, Sequence, Type, Union

from langgraph.channels.base import BaseChannel
from langgraph.constants import INTERRUPT
from langgraph.pregel.read import PregelNode


def validate_graph(
    nodes: Mapping[str, PregelNode],
    channels: dict[str, BaseChannel],
    input_channels: Union[str, Sequence[str]],
    output_channels: Union[str, Sequence[str]],
    stream_channels: Optional[Union[str, Sequence[str]]],
    interrupt_after_nodes: Sequence[str],
    interrupt_before_nodes: Sequence[str],
    default_channel_cls: Type[BaseChannel],
) -> None:
    subscribed_channels = set[str]()
    for name, node in nodes.items():
        if name == INTERRUPT:
            raise ValueError(f"Node name {INTERRUPT} is reserved")
        if isinstance(node, PregelNode):
            subscribed_channels.update(node.triggers)
        else:
            raise TypeError(
                f"Invalid node type {type(node)}, expected Channel.subscribe_to()"
            )

    for chan in subscribed_channels:
        if chan not in channels:
            channels[chan] = default_channel_cls(Any)  # type: ignore[call-arg]

    if isinstance(input_channels, str):
        if input_channels not in channels:
            channels[input_channels] = default_channel_cls(Any)  # type: ignore[call-arg]
        if input_channels not in subscribed_channels:
            raise ValueError(
                f"Input channel {input_channels} is not subscribed to by any node"
            )
    else:
        for chan in input_channels:
            if chan not in channels:
                channels[chan] = default_channel_cls(Any)  # type: ignore[call-arg]
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
            channels[chan] = default_channel_cls(Any)  # type: ignore[call-arg]

    for n in interrupt_after_nodes:
        if n not in nodes:
            raise ValueError(f"Node {n} not in nodes")
    for n in interrupt_before_nodes:
        if n not in nodes:
            raise ValueError(f"Node {n} not in nodes")


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
