from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from langgraph.checkpoint.base import Checkpoint, EmptyChannelError
from langgraph.checkpoint.base.id import uuid6


class ChannelProtocol(Protocol):
    def checkpoint(self) -> Optional[Any]: ...


def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=1,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
    )


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Optional[Mapping[str, ChannelProtocol]],
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
        for k, v in channels.items():
            if k not in checkpoint["channel_versions"]:
                continue
            try:
                values[k] = v.checkpoint()
            except EmptyChannelError:
                pass
    return Checkpoint(
        v=1,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=checkpoint["channel_versions"],
        versions_seen=checkpoint["versions_seen"],
    )
