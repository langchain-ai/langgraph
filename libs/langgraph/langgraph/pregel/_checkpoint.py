from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    DeltaChainValue,
    DeltaValue,
)
from langgraph.checkpoint.base.id import uuid6

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel
from langgraph.managed.base import ManagedValueMapping, ManagedValueSpec

LATEST_VERSION = 4

logger = logging.getLogger(__name__)

_MISSING_SENTINEL = object()


def _assemble_delta_channels(
    checkpoint: "Checkpoint",
    config: RunnableConfig,
    checkpointer: BaseCheckpointSaver,
) -> dict[str, Any]:
    """Resolve any DeltaValue entries in checkpoint channel_values to DeltaChainValue.

    Returns a dict of only the channels that needed assembly (others are untouched).
    Tries get_channel_blob fast-path first; falls back to get_tuple traversal.
    """
    thread_id = str(config["configurable"]["thread_id"])
    checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
    assembled: dict[str, Any] = {}

    for channel, value in checkpoint["channel_values"].items():
        if not isinstance(value, DeltaValue):
            continue

        chain_deltas: list[list[Any]] = []
        base: list[Any] | None = None
        cursor: DeltaValue = value
        visited: set[str] = set()

        while True:
            chain_deltas.append(cursor.delta)
            prev_id = cursor.prev_checkpoint_id
            if prev_id is None:
                break  # chain root
            if prev_id in visited:
                logger.warning(
                    "DeltaChannel chain cycle at checkpoint %r for channel %r; breaking",
                    prev_id,
                    channel,
                )
                break
            visited.add(prev_id)

            # Fast path: saver has a dedicated blob store.
            blob = checkpointer.get_channel_blob(thread_id, checkpoint_ns, prev_id, channel)
            if blob is not NotImplemented:
                if isinstance(blob, DeltaValue):
                    cursor = blob
                    continue
                else:
                    base = blob  # plain list = snapshot root
                    break

            # Fallback: load the full checkpoint and extract channel value.
            parent_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": prev_id,
                }
            }
            parent_tuple = checkpointer.get_tuple(parent_config)
            if parent_tuple is None:
                logger.warning(
                    "DeltaChannel chain broken: checkpoint %r not found for channel %r",
                    prev_id,
                    channel,
                )
                break
            prev_val = parent_tuple.checkpoint["channel_values"].get(channel, _MISSING_SENTINEL)
            if prev_val is _MISSING_SENTINEL:
                break
            elif isinstance(prev_val, DeltaValue):
                cursor = prev_val
            else:
                base = prev_val
                break

        chain_deltas.reverse()
        assembled[channel] = DeltaChainValue(base=base, deltas=chain_deltas)

    return assembled


async def _aassemble_delta_channels(
    checkpoint: "Checkpoint",
    config: RunnableConfig,
    checkpointer: BaseCheckpointSaver,
) -> dict[str, Any]:
    """Async version of _assemble_delta_channels."""
    thread_id = str(config["configurable"]["thread_id"])
    checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
    assembled: dict[str, Any] = {}

    for channel, value in checkpoint["channel_values"].items():
        if not isinstance(value, DeltaValue):
            continue

        chain_deltas: list[list[Any]] = []
        base: list[Any] | None = None
        cursor: DeltaValue = value
        visited: set[str] = set()

        while True:
            chain_deltas.append(cursor.delta)
            prev_id = cursor.prev_checkpoint_id
            if prev_id is None:
                break
            if prev_id in visited:
                logger.warning(
                    "DeltaChannel chain cycle at checkpoint %r for channel %r; breaking",
                    prev_id,
                    channel,
                )
                break
            visited.add(prev_id)

            blob = await checkpointer.aget_channel_blob(
                thread_id, checkpoint_ns, prev_id, channel
            )
            if blob is not NotImplemented:
                if isinstance(blob, DeltaValue):
                    cursor = blob
                    continue
                else:
                    base = blob
                    break

            parent_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": prev_id,
                }
            }
            parent_tuple = await checkpointer.aget_tuple(parent_config)
            if parent_tuple is None:
                logger.warning(
                    "DeltaChannel chain broken: checkpoint %r not found for channel %r",
                    prev_id,
                    channel,
                )
                break
            prev_val = parent_tuple.checkpoint["channel_values"].get(channel, _MISSING_SENTINEL)
            if prev_val is _MISSING_SENTINEL:
                break
            elif isinstance(prev_val, DeltaValue):
                cursor = prev_val
            else:
                base = prev_val
                break

        chain_deltas.reverse()
        assembled[channel] = DeltaChainValue(base=base, deltas=chain_deltas)

    return assembled


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
    channels: Mapping[str, BaseChannel] | None,
    step: int,
    *,
    id: str | None = None,
    updated_channels: set[str] | None = None,
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
        updated_channels=None if updated_channels is None else sorted(updated_channels),
    )


def channels_from_checkpoint(
    specs: Mapping[str, BaseChannel | ManagedValueSpec],
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
    channels: dict[str, BaseChannel] = {}
    for k, v in channel_specs.items():
        ch = v.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
        ch.after_checkpoint(checkpoint["channel_versions"].get(k))
        channels[k] = ch
    return channels, managed_specs


def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],
        ts=checkpoint["ts"],
        id=checkpoint["id"],
        channel_values=checkpoint["channel_values"].copy(),
        channel_versions=checkpoint["channel_versions"].copy(),
        versions_seen={k: v.copy() for k, v in checkpoint["versions_seen"].items()},
        updated_channels=checkpoint.get("updated_channels", None),
    )
