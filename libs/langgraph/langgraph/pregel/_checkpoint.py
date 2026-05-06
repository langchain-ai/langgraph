from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, NamedTuple, cast

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


class CreateCheckpointResult(NamedTuple):
    checkpoint: Checkpoint
    snapshotted: set[str]


def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=LATEST_VERSION,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
    )


def decide_delta_snapshots(
    channels: Mapping[str, BaseChannel],
    counts: Mapping[str, int],
) -> set[str]:
    """Return the set of DeltaChannel names that should snapshot now.

    A channel snapshots when its accumulated update count (since the last
    snapshot) reaches or exceeds `snapshot_frequency`. This is a pure
    predicate — no mutation.
    """
    return {
        name
        for name, ch in channels.items()
        if isinstance(ch, DeltaChannel)
        and ch.is_available()
        and counts.get(name, 0) >= ch.snapshot_frequency
    }


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel] | None,
    step: int,
    *,
    id: str | None = None,
    updated_channels: set[str] | None = None,
    get_next_version: GetNextVersion | None = None,
    updates_since_snapshot: Mapping[str, int] | None = None,
) -> CreateCheckpointResult:
    """Build a new Checkpoint from the previous one and live channel state.

    For each `DeltaChannel`, a `_DeltaSnapshot(value)` blob is written into
    `channel_values[k]` when `decide_delta_snapshots` says the channel should
    snapshot (i.e. update count >= `snapshot_frequency`). Otherwise the
    channel is omitted from `channel_values` and the ancestor walk
    reconstructs state from `checkpoint_writes`.

    Returns a `CreateCheckpointResult` containing the checkpoint and the
    set of DeltaChannel names that were snapshotted (caller should reset
    their per-channel counters to 0 for these).
    """
    ts = datetime.now(timezone.utc).isoformat()
    counts = updates_since_snapshot or {}
    snapshotted: set[str] = set()
    if channels is None:
        values = checkpoint["channel_values"]
        channel_versions = checkpoint["channel_versions"]
    else:
        will_snapshot = decide_delta_snapshots(channels, counts)
        values = {}
        channel_versions = dict(checkpoint["channel_versions"])
        for k in channels:
            if k not in channel_versions:
                continue
            ch = channels[k]
            if k in will_snapshot:
                # In exit mode, the snapshot decision is deferred to exit
                # time (intermediate steps have do_checkpoint=False). The
                # channel's count may have reached snapshot_frequency over
                # several supersteps, but the LAST superstep may not have
                # written to this channel. In that case apply_writes()
                # (in _algo.py) didn't bump this channel's version, so
                # saver.put() wouldn't include it in new_versions and
                # the snapshot blob would be silently dropped. The manual
                # bump below closes the gap. In sync/async durability this
                # branch is effectively dead code (the step that pushes
                # the count to freq always writes the channel).
                if get_next_version is not None and (
                    updated_channels is None or k not in updated_channels
                ):
                    channel_versions[k] = get_next_version(channel_versions[k], None)
                values[k] = _DeltaSnapshot(ch.get())
                snapshotted.add(k)
            else:
                v = ch.checkpoint()
                if v is not MISSING:
                    values[k] = v
    return CreateCheckpointResult(
        checkpoint=Checkpoint(
            v=LATEST_VERSION,
            ts=ts,
            id=id or str(uuid6(clock_seq=step)),
            channel_values=values,
            channel_versions=channel_versions,
            versions_seen=checkpoint["versions_seen"],
            updated_channels=None
            if updated_channels is None
            else sorted(updated_channels),
        ),
        snapshotted=snapshotted,
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
