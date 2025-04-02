from datetime import datetime, timezone
from typing import Mapping, Optional

from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.base.id import uuid6
from langgraph.constants import MISSING
from langgraph.pregel.read import PregelNode

LATEST_VERSION = 2


def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=LATEST_VERSION,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
        pending_sends=[],
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
        pending_sends=checkpoint.get("pending_sends", []),
    )


def migrate_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    nodes: Mapping[str, PregelNode],
) -> None:
    """Migrate a checkpoint to new channel layout."""

    values = checkpoint["channel_values"]
    versions = checkpoint["channel_versions"]
    seen = checkpoint["versions_seen"]

    if any(k.startswith("start:") for k in versions):
        # Migrate from start:node to branch:to:node
        for k in values:
            if k.startswith("start:"):
                node = k.split(":")[1]
                new_k = f"branch:to:{node}"
                if node not in nodes:
                    continue
                v = versions.pop(k)
                s = seen.get(node, {}).pop(k, None)
                if s is None or s < v:
                    values[new_k] = values.pop(k)
                # TODO handle s == v
                # TODO handle new_k already in values

    # TODO Migrate from "node" to "branch:to:node"
