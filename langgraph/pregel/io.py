from typing import Any, Iterator, Mapping, Sequence

from langgraph.channels.base import BaseChannel
from langgraph.pregel.log import logger


def map_input(
    input_channels: str | Sequence[str], chunk: dict[str, Any] | Any | None
) -> Iterator[tuple[str, Any]]:
    """Map input chunk to a sequence of pending writes in the form (channel, value)."""
    if chunk is None:
        return
    elif isinstance(input_channels, str):
        yield (input_channels, chunk)
    else:
        if not isinstance(chunk, dict):
            raise TypeError(f"Expected chunk to be a dict, got {type(chunk).__name__}")
        for k in chunk:
            if k in input_channels:
                yield (k, chunk[k])
            else:
                logger.warning(f"Input channel {k} not found in {input_channels}")


def map_output(
    output_channels: str | Sequence[str],
    pending_writes: Sequence[tuple[str, Any]],
    channels: Mapping[str, BaseChannel],
) -> dict[str, Any] | Any | None:
    """Map pending writes (a sequence of tuples (channel, value)) to output chunk."""
    if isinstance(output_channels, str):
        if any(chan == output_channels for chan, _ in pending_writes):
            return channels[output_channels].get()
    else:
        if updated := {c for c, _ in pending_writes if c in output_channels}:
            return {chan: channels[chan].get() for chan in updated}
    return None
