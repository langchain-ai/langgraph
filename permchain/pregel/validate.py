from typing import Any, Mapping, Sequence

from permchain.channels.base import BaseChannel
from permchain.channels.last_value import LastValue
from permchain.pregel.read import ChannelBatch, ChannelInvoke


def validate_chains_channels(
    chains: Mapping[str, ChannelInvoke | ChannelBatch],
    channels: Mapping[str, BaseChannel],
    input: str | Sequence[str],
    output: str | Sequence[str],
) -> None:
    subscribed_channels = set[str]()
    for chain in chains.values():
        if isinstance(chain, ChannelInvoke):
            subscribed_channels.update(chain.channels.values())
        elif isinstance(chain, ChannelBatch):
            subscribed_channels.add(chain.channel)
        else:
            raise TypeError(
                f"Invalid chain type {type(chain)}, expected Channel.subscribe_to() or Channel.subscribe_to_each()"
            )

    for chan in subscribed_channels:
        if chan not in channels:
            channels[chan] = LastValue(Any)

    if isinstance(input, str):
        if input not in subscribed_channels:
            raise ValueError(f"Input channel {input} is not subscribed to by any chain")
    else:
        if all(chan not in subscribed_channels for chan in input):
            raise ValueError(
                f"None of the input channels {input} are subscribed to by any chain"
            )

    if isinstance(output, str):
        if output not in channels:
            channels[output] = LastValue(Any)
    else:
        for chan in output:
            if chan not in channels:
                channels[chan] = LastValue(Any)
