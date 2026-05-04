from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
)
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


def _should_snapshot_delta(
    name: str,
    ch: DeltaChannel,
    updates_since_snapshot: Mapping[str, int],
    *,
    force: bool,
) -> bool:
    """Decide whether `ch` should write a `_DeltaSnapshot` this step.

    Triggers:
      * `force` — always snapshot (used by `durability="exit"`).
      * Update-count: this channel has accumulated at least
        `snapshot_frequency` updates since its last snapshot. The count
        is supplied by the caller via `updates_since_snapshot[name]` and
        is reset to `0` whenever a snapshot fires.

    Version-format-independent: works for `int`, `float`, and `str`
    versioning schemes alike.
    """
    if force:
        return True
    return updates_since_snapshot.get(name, 0) >= ch.snapshot_frequency


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel] | None,
    step: int,
    *,
    id: str | None = None,
    updated_channels: set[str] | None = None,
    get_next_version: GetNextVersion | None = None,
    force_delta_snapshot: bool = False,
    updates_since_snapshot: Mapping[str, int] | None = None,
    new_updates_since_snapshot: dict[str, int] | None = None,
) -> Checkpoint:
    """Create a checkpoint for the given channels.

    For each `DeltaChannel`, a `_DeltaSnapshot(value)` blob is written into
    `channel_values[k]` when this channel has accumulated at least
    `snapshot_frequency` updates since its last snapshot (counter supplied
    via `updates_since_snapshot`). Otherwise the channel is omitted from
    `channel_values`; its `channel_versions` entry still bumps so that the
    saver tracks the channel and the ancestor walk can replay writes.

    Snapshots are eager: even if the channel had no write this step, a
    version bump is forced (via `get_next_version`) so `put()` includes
    the channel in `new_versions` and stores the blob.

    `force_delta_snapshot` ignores the cadence and always snapshots —
    used by `durability="exit"` where intermediate writes are not stored
    as ancestor `checkpoint_writes`.

    If `new_updates_since_snapshot` is provided, the function resets the
    counter to `0` for any channel that snapshotted this step. Counters
    for channels that did not snapshot are left untouched (the caller is
    responsible for incrementing them based on `updated_channels`).
    """
    ts = datetime.now(timezone.utc).isoformat()
    counts = updates_since_snapshot or {}
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
                and ch.is_available()
                and _should_snapshot_delta(
                    k,
                    ch,
                    counts,
                    force=force_delta_snapshot,
                )
            ):
                # Eager snapshot: bump version if not already written this step
                # so put() includes this channel in new_versions and stores blob.
                if get_next_version is not None and (
                    updated_channels is None or k not in updated_channels
                ):
                    channel_versions[k] = get_next_version(channel_versions[k], None)
                values[k] = _DeltaSnapshot(ch.get())
                if new_updates_since_snapshot is not None:
                    new_updates_since_snapshot[k] = 0
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
    """True if `spec` is a `DeltaChannel` and no value is stored at this
    checkpoint, requiring an ancestor walk to reconstruct.

    `_DeltaSnapshot` blobs and plain values (migration) resolve directly via
    `from_checkpoint` — only absence (`MISSING`) triggers replay.
    """
    if not isinstance(spec, DeltaChannel):
        return False
    return stored is MISSING


def channels_from_checkpoint(
    specs: Mapping[str, BaseChannel | ManagedValueSpec],
    checkpoint: Checkpoint,
    *,
    saver: BaseCheckpointSaver | None = None,
    config: RunnableConfig | None = None,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
    """Hydrate channels from a checkpoint.

    For most channels, `spec.from_checkpoint(checkpoint["channel_values"][k])`
    is sufficient. `DeltaChannel` is the exception: when the channel is
    absent from `channel_values`, an ancestor walk via
    `saver.get_delta_channel_history` is required to find the nearest seed
    (`_DeltaSnapshot` blob or pre-migration plain value) and accumulate
    the writes between it and the target. All delta channels needing
    replay are batched into a single saver call.
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
        histories = saver.get_delta_channel_history(
            config=config, channels=delta_channels
        )

    channels: dict[str, BaseChannel] = {}
    for k, spec in channel_specs.items():
        ch: BaseChannel
        if k in histories:
            delta_spec = cast(DeltaChannel, spec)
            history = histories[k]
            replay_ch = delta_spec.from_checkpoint(history.get("seed", MISSING))
            replay_ch.replay_writes(history["writes"])
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
        histories = await saver.aget_delta_channel_history(
            config=config, channels=delta_channels
        )

    channels: dict[str, BaseChannel] = {}
    for k, spec in channel_specs.items():
        ch: BaseChannel
        if k in histories:
            delta_spec = cast(DeltaChannel, spec)
            history = histories[k]
            replay_ch = delta_spec.from_checkpoint(history.get("seed", MISSING))
            replay_ch.replay_writes(history["writes"])
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


def read_delta_updates_since_snapshot(
    metadata: CheckpointMetadata | None,
) -> dict[str, int]:
    """Read the per-channel update counter from checkpoint metadata.

    Returns an empty dict for missing/None metadata; the dict is
    `total=False` on `CheckpointMetadata`, so absence means "no prior
    delta-channel activity tracked."
    """
    if not metadata:
        return {}
    return dict(metadata.get("delta_updates_since_snapshot", {}) or {})
