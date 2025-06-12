from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.channels.base import BaseChannel
from langgraph.constants import RESERVED
from langgraph.managed.base import ManagedValueMapping
from langgraph.pregel.read import PregelNode
from langgraph.types import All


def validate_graph(
    nodes: Mapping[str, PregelNode],
    channels: dict[str, BaseChannel],
    managed: ManagedValueMapping,
    input_channels: str | Sequence[str],
    output_channels: str | Sequence[str],
    stream_channels: str | Sequence[str] | None,
    interrupt_after_nodes: All | Sequence[str],
    interrupt_before_nodes: All | Sequence[str],
) -> None:
    for chan in channels:
        if chan in RESERVED:
            raise ValueError(f"Channel name '{chan}' is reserved")
    for name in managed:
        if name in RESERVED:
            raise ValueError(f"Managed name '{name}' is reserved")

    subscribed_channels = set[str]()
    for name, node in nodes.items():
        if name in RESERVED:
            raise ValueError(f"Node name '{name}' is reserved")
        if isinstance(node, PregelNode):
            subscribed_channels.update(node.triggers)
            if isinstance(node.channels, str):
                if node.channels not in channels:
                    raise ValueError(
                        f"Node {name} reads channel '{node.channels}' "
                        f"not in known channels: '{repr(sorted(channels))[:100]}'"
                    )
            else:
                for chan in node.channels:
                    if chan not in channels and chan not in managed:
                        raise ValueError(
                            f"Node {name} reads channel '{chan}' "
                            f"not in known channels: '{repr(sorted(channels))[:100]}'"
                        )
        else:
            raise TypeError(
                f"Invalid node type {type(node)}, expected PregelNode or NodeBuilder"
            )

    for chan in subscribed_channels:
        if chan not in channels:
            raise ValueError(
                f"Subscribed channel '{chan}' not "
                f"in known channels: '{repr(sorted(channels))[:100]}'"
            )

    if isinstance(input_channels, str):
        if input_channels not in channels:
            raise ValueError(
                f"Input channel '{input_channels}' not "
                f"in known channels: '{repr(sorted(channels))[:100]}'"
            )
        if input_channels not in subscribed_channels:
            raise ValueError(
                f"Input channel {input_channels} is not subscribed to by any node"
            )
    else:
        for chan in input_channels:
            if chan not in channels:
                raise ValueError(
                    f"Input channel '{chan}' not in '{repr(sorted(channels))[:100]}'"
                )
        if all(chan not in subscribed_channels for chan in input_channels):
            raise ValueError(
                f"None of the input channels {input_channels} "
                f"are subscribed to by any node"
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
            raise ValueError(
                f"Output channel '{chan}' not "
                f"in known channels: '{repr(sorted(channels))[:100]}'"
            )

    if interrupt_after_nodes != "*":
        for n in interrupt_after_nodes:
            if n not in nodes:
                raise ValueError(f"Node {n} not in nodes")
    if interrupt_before_nodes != "*":
        for n in interrupt_before_nodes:
            if n not in nodes:
                raise ValueError(f"Node {n} not in nodes")


def validate_keys(
    keys: str | Sequence[str] | None,
    channels: Mapping[str, Any],
) -> None:
    if isinstance(keys, str):
        if keys not in channels:
            raise ValueError(f"Key {keys} not in channels")
    elif keys is not None:
        for chan in keys:
            if chan not in channels:
                raise ValueError(f"Key {chan} not in channels")
