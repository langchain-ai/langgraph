from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import DELTA_SENTINEL, BaseCheckpointSaver, Checkpoint
from langgraph.checkpoint.base.id import uuid6

from langgraph._internal._typing import MISSING
from langgraph.channels.aggregate import AggregateChannel
from langgraph.channels.base import BaseChannel
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
    """Create a checkpoint for the given channels.

    For `AggregateChannel` spec with `snapshot_frequency != 1`, the stored
    blob alternates between the full value and `DELTA_SENTINEL` based on
    `is_snapshot_step(step)`. Non-snapshot steps store the sentinel; the
    value is reconstructed from ancestor writes at read time.
    """
    ts = datetime.now(timezone.utc).isoformat()
    if channels is None:
        values = checkpoint["channel_values"]
    else:
        values = {}
        for k in channels:
            if k not in checkpoint["channel_versions"]:
                continue
            ch = channels[k]
            if isinstance(ch, AggregateChannel) and not ch.is_snapshot_step(step):
                values[k] = DELTA_SENTINEL
                continue
            v = ch.checkpoint()
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


def _needs_replay(spec: BaseChannel, stored: object) -> bool:
    """True if `spec` is a delta-mode AggregateChannel and the stored
    blob is empty/sentinel, requiring an ancestor walk to reconstruct."""
    if not isinstance(spec, AggregateChannel):
        return False
    if spec.snapshot_frequency == 1:
        return False
    return stored is MISSING or stored is DELTA_SENTINEL


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

    `AggregateChannel` with `snapshot_frequency != 1` is the exception:
    its stored value on non-snapshot steps is `DELTA_SENTINEL`; the full
    state is spread across `checkpoint_writes` along the ancestor chain.
    When `saver` and `config` are provided, this function fetches that
    history via `saver._get_channel_writes_history` and folds it through
    the channel's operator. Without them (static contexts — graph
    drawing, unit tests), delta-mode channels fall back to empty.
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
        if _needs_replay(spec, stored) and saver is not None and config is not None:
            # Walk ancestors for seed + writes. The saver's walk stops at
            # the nearest non-sentinel blob (natural terminator under
            # snapshot_frequency > 1; pre-migration blobs also act as
            # terminators if the spec was changed mid-thread).
            history = saver._get_channel_writes_history(config, k)
            replay_ch = spec.from_checkpoint(history.seed)
            replay_ch.replay_writes(history.writes)
            ch = replay_ch
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
        if _needs_replay(spec, stored) and saver is not None and config is not None:
            history = await saver._aget_channel_writes_history(config, k)
            replay_ch = spec.from_checkpoint(history.seed)
            replay_ch.replay_writes(history.writes)
            ch = replay_ch
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
