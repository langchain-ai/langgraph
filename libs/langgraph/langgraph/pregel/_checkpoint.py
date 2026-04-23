from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import DELTA_SENTINEL, BaseCheckpointSaver, Checkpoint
from langgraph.checkpoint.base.id import uuid6

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel
from langgraph.channels.delta import DeltaChannel
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
    *,
    saver: BaseCheckpointSaver | None = None,
    config: RunnableConfig | None = None,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
    """Hydrate channels from a checkpoint.

    For most channels, `spec.from_checkpoint(checkpoint["channel_values"][k])`
    is sufficient — the stored value IS the reconstructed state.

    `DeltaChannel` is the exception: its stored value is a sentinel; the
    full state is spread across `checkpoint_writes` along the ancestor
    chain. When `saver` and `config` are provided, this function fetches
    that history via `saver._get_channel_writes_history` and folds it
    through the channel's reducer. Without them (static contexts — graph
    drawing, unit tests), delta channels fall back to empty.
    """
    channel_specs: dict[str, BaseChannel] = {}
    managed_specs: dict[str, ManagedValueSpec] = {}
    for k, v in specs.items():
        if isinstance(v, BaseChannel):
            channel_specs[k] = v
        else:
            managed_specs[k] = v

    channels: dict[str, BaseChannel] = {}
    for k, spec in channel_specs.items():
        ch: BaseChannel
        stored = checkpoint["channel_values"].get(k, MISSING)
        if (
            isinstance(spec, DeltaChannel)
            and saver is not None
            and config is not None
            and (stored is MISSING or stored is DELTA_SENTINEL)
        ):
            # Target's own blob is empty/sentinel — walk ancestors for
            # seed + writes. Skipping this when `stored` is a real value
            # preserves state written via `update_state` or sitting at the
            # tip of a pre-migration thread: the saver's ancestor walk
            # intentionally excludes the target's own blob, so without
            # this short-circuit we'd lose it.
            history = saver._get_channel_writes_history(config, k)
            delta_ch = spec.from_checkpoint(history.seed)
            delta_ch.replay_writes(history.writes)
            ch = delta_ch
        else:
            ch = spec.from_checkpoint(stored)
        channels[k] = ch
    return channels, managed_specs


async def achannels_from_checkpoint(
    specs: Mapping[str, BaseChannel | ManagedValueSpec],
    checkpoint: Checkpoint,
    *,
    saver: BaseCheckpointSaver | None = None,
    config: RunnableConfig | None = None,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
    """Async version of `channels_from_checkpoint`. See docstring there."""
    channel_specs: dict[str, BaseChannel] = {}
    managed_specs: dict[str, ManagedValueSpec] = {}
    for k, v in specs.items():
        if isinstance(v, BaseChannel):
            channel_specs[k] = v
        else:
            managed_specs[k] = v

    channels: dict[str, BaseChannel] = {}
    for k, spec in channel_specs.items():
        ch: BaseChannel
        stored = checkpoint["channel_values"].get(k, MISSING)
        if (
            isinstance(spec, DeltaChannel)
            and saver is not None
            and config is not None
            and (stored is MISSING or stored is DELTA_SENTINEL)
        ):
            history = await saver._aget_channel_writes_history(config, k)
            delta_ch = spec.from_checkpoint(history.seed)
            delta_ch.replay_writes(history.writes)
            ch = delta_ch
        else:
            ch = spec.from_checkpoint(stored)
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
