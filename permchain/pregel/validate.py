from typing import Any, Mapping, Sequence

from permchain.channels.base import BaseChannel
from permchain.channels.last_value import LastValue
from permchain.constants import CHECKPOINT_KEY_TS, CHECKPOINT_KEY_VERSION
from permchain.pregel.read import ChannelBatch, ChannelInvoke
from permchain.pregel.reserved import ReservedChannels

FORBIDDEN_CHANNEL_NAMES = {
    CHECKPOINT_KEY_TS,
    CHECKPOINT_KEY_VERSION,
}


def validate_chains_channels(
    chains: Mapping[str, ChannelInvoke | ChannelBatch],
    channels: dict[str, BaseChannel],
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
            channels[chan] = LastValue(Any)  # type: ignore[arg-type]

    if isinstance(input, str):
        if input not in channels:
            channels[input] = LastValue(Any)  # type: ignore[arg-type]
        if input not in subscribed_channels:
            raise ValueError(f"Input channel {input} is not subscribed to by any chain")
    else:
        for chan in input:
            if chan not in channels:
                channels[chan] = LastValue(Any)  # type: ignore[arg-type]
        if all(chan not in subscribed_channels for chan in input):
            raise ValueError(
                f"None of the input channels {input} are subscribed to by any chain"
            )

    if isinstance(output, str):
        if output not in channels:
            channels[output] = LastValue(Any)  # type: ignore[arg-type]
    else:
        for chan in output:
            if chan not in channels:
                channels[chan] = LastValue(Any)  # type: ignore[arg-type]

    for name in FORBIDDEN_CHANNEL_NAMES:
        if name in channels:
            raise ValueError(f"Channel name {name} is reserved")

    for chan in ReservedChannels:
        if chan not in channels:
            channels[chan] = LastValue(Any)  # type: ignore[arg-type]
