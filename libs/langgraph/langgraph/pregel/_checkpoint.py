from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import DELTA_SENTINEL, BaseCheckpointSaver, Checkpoint
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import _DeltaSnapshot

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel
from langgraph.channels.delta import DeltaChannel
from langgraph.managed.base import ManagedValueMapping, ManagedValueSpec

LATEST_VERSION = 4

GetNextVersion = Callable[[Any, None], Any]


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
    get_next_version: GetNextVersion | None = None,
    force_delta_snapshot: bool = False,
) -> Checkpoint:
    """Create a checkpoint for the given channels.

    For `DeltaChannel` with `snapshot_frequency=N`, snapshot steps write a
    `_DeltaSnapshot` blob rather than `DELTA_SENTINEL`, bounding the ancestor
    walk to at most N steps. Snapshots are eager: even if the channel had no
    write this step, a version bump is forced (via `get_next_version`) so the
    blob is stored by `put()`. Without `get_next_version` (e.g. static
    contexts), snapshot steps gracefully fall back to sentinel.

    `force_delta_snapshot` writes available `DeltaChannel` values as snapshots
    regardless of `snapshot_frequency`. This is used by `durability="exit"`,
    where intermediate writes are not stored as ancestor `checkpoint_writes`.
    """
    ts = datetime.now(timezone.utc).isoformat()
    if channels is None:
        values = checkpoint["channel_values"]
        channel_versions = checkpoint["channel_versions"]
    else:
        values = {}
        channel_versions = dict(checkpoint["channel_versions"])
        for k in channels:
            if k not in channel_versions:
                continue
            ch = channels[k]
            if (
                isinstance(ch, DeltaChannel)
                and (force_delta_snapshot or ch.is_snapshot_step(step))
                and ch.is_available()
            ):
                # Eager snapshot: bump version if not already written this step
                # so put() includes this channel in new_versions and stores blob.
                if get_next_version is not None and (
                    updated_channels is None or k not in updated_channels
                ):
                    channel_versions[k] = get_next_version(channel_versions[k], None)
                values[k] = _DeltaSnapshot(ch.get())
            else:
                v = ch.checkpoint()
                if v is not MISSING:
                    values[k] = v
    return Checkpoint(
        v=LATEST_VERSION,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=channel_versions,
        versions_seen=checkpoint["versions_seen"],
        updated_channels=None if updated_channels is None else sorted(updated_channels),
    )


def _needs_replay(spec: BaseChannel, stored: object) -> bool:
    """True if `spec` is a `DeltaChannel` and the stored blob is a sentinel,
    requiring an ancestor walk to reconstruct.

    `_DeltaSnapshot` blobs and plain values (migration) resolve directly via
    `from_checkpoint` — only `DELTA_SENTINEL` / `MISSING` trigger replay.
    """
    if not isinstance(spec, DeltaChannel):
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
    is sufficient. `DeltaChannel` is the exception: sentinel blobs require an
    ancestor walk via `saver._get_all_delta_channels_writes_history`. All
    delta channels needing replay are batched into a single saver call to
    save K-1 redundant scans of `checkpoint_writes` (which has no channel
    index). The walk terminates per-channel at the nearest `_DeltaSnapshot`
    blob or pre-migration plain value, so read depth is bounded by
    `snapshot_frequency`.
    """
    channel_specs: dict[str, BaseChannel] = {}
    managed_specs: dict[str, ManagedValueSpec] = {}
    for k, v in specs.items():
        if isinstance(v, BaseChannel):
            channel_specs[k] = v
        else:
            managed_specs[k] = v

    delta_channels: list[str] = [
        k
        for k, spec in channel_specs.items()
        if _needs_replay(spec, checkpoint["channel_values"].get(k, MISSING))
    ]
    histories: Mapping[str, Any] = {}
    if delta_channels and saver is not None and config is not None:
        histories = saver._get_all_delta_channels_writes_history(
            config, delta_channels
        )

    channels: dict[str, BaseChannel] = {}
    for k, spec in channel_specs.items():
        ch: BaseChannel
        if k in histories:
            delta_spec = cast(DeltaChannel, spec)
            history = histories[k]
            replay_ch = delta_spec.from_checkpoint(history.seed)
            replay_ch.replay_writes(history.writes)
            ch = replay_ch
        else:
            ch = spec.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
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

    delta_channels: list[str] = [
        k
        for k, spec in channel_specs.items()
        if _needs_replay(spec, checkpoint["channel_values"].get(k, MISSING))
    ]
    histories: Mapping[str, Any] = {}
    if delta_channels and saver is not None and config is not None:
        histories = await saver._aget_all_delta_channels_writes_history(
            config, delta_channels
        )

    channels: dict[str, BaseChannel] = {}
    for k, spec in channel_specs.items():
        ch: BaseChannel
        if k in histories:
            delta_spec = cast(DeltaChannel, spec)
            history = histories[k]
            replay_ch = delta_spec.from_checkpoint(history.seed)
            replay_ch.replay_writes(history.writes)
            ch = replay_ch
        else:
            ch = spec.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
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
