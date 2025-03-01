from contextlib import contextmanager
from typing import Iterator, Mapping

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import Checkpoint


@contextmanager
def ChannelsManager(
    specs: Mapping[str, BaseChannel],
    checkpoint: Checkpoint,
) -> Iterator[Mapping[str, BaseChannel]]:
    """Manage channels for the lifetime of a Pregel invocation (multiple steps)."""
    channel_specs: dict[str, BaseChannel] = {}
    for k, v in specs.items():
        channel_specs[k] = v
    yield {
        k: v.from_checkpoint(checkpoint["channel_values"].get(k))
        for k, v in channel_specs.items()
    }
