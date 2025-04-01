from datetime import datetime, timezone
from typing import Mapping, Optional

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import LATEST_VERSION, Checkpoint
from langgraph.checkpoint.base.id import uuid6
from langgraph.constants import MISSING


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
        pending_sends=checkpoint.get("pending_sends", []),
    )
