from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Optional, Union

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.base.id import uuid6
from langgraph.constants import MISSING
from langgraph.managed.base import ManagedValueMapping, ManagedValueSpec

LATEST_VERSION = 4


def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=LATEST_VERSION,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
    )


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Optional[Mapping[str, BaseChannel]],
    step: int,
    *,
    id: Optional[str] = None,
) -> Checkpoint:
    """Create a checkpoint for the given channels."""
    ts = datetime.now(timezone.utc).isoformat()
    if channels is None:
        values = checkpoint["channel_values"]
    else:
        values = {}
        for k in channels:
            if k not in checkpoint["channel_versions"]:
                continue
            v = channels[k].checkpoint()
            if v is not MISSING:
                values[k] = v
    return Checkpoint(
        v=LATEST_VERSION,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=checkpoint["channel_versions"],
        versions_seen=checkpoint["versions_seen"],
    )


def channels_from_checkpoint(
    specs: Mapping[str, Union[BaseChannel, ManagedValueSpec]],
    checkpoint: Checkpoint,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
    """Get channels from a checkpoint."""
    channel_specs: dict[str, BaseChannel] = {}
    managed_specs: dict[str, ManagedValueSpec] = {}
    for k, v in specs.items():
        if isinstance(v, BaseChannel):
            channel_specs[k] = v
        else:
            managed_specs[k] = v
    return (
        {
            k: v.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
            for k, v in channel_specs.items()
        },
        managed_specs,
    )
