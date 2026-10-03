from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    PendingWrite,
)
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import _DeltaSnapshot

from langgraph._internal._config import DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT
from langgraph._internal._constants import NS_END, NS_SEP, PUSH, SNAPSHOT_BUMPS
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


def exit_delta_task_id(step: int, task_id: str) -> str:
    """Synthetic task id for exit-mode DeltaChannel writes.

    Embeds the superstep in the first UUID group so `ORDER BY task_id, idx`
    preserves chronological order while remaining a valid RFC UUID (required by
    Postgres `checkpoint_writes.task_id uuid` columns).
    """
    parts = str(uuid.UUID(task_id)).split("-")
    return f"{step:08d}-{parts[1]}-{parts[2]}-{parts[3]}-{parts[4]}"


def delta_channels_to_snapshot(
    channels: Mapping[str, BaseChannel],
    counters_since_delta_snapshot: Mapping[str, tuple[int, int]],
    channel_versions: ChannelVersions,
) -> set[str]:
    """Return the set of DeltaChannel names that should snapshot now.

    A channel snapshots when EITHER its accumulated update count reaches
    `snapshot_frequency` OR the total supersteps since its last snapshot
    reaches `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT`. A channel without a version
    was never written on this branch, so it has nothing to snapshot. This is a
    pure predicate — no mutation.
    """
    result: set[str] = set()
    for name, ch in channels.items():
        if (
            not isinstance(ch, DeltaChannel)
            or not ch.is_available()
            or name not in channel_versions
        ):
            continue
        updates, supersteps = counters_since_delta_snapshot.get(name, (0, 0))
        if (
            updates >= ch.snapshot_frequency
            or supersteps >= DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT
        ):
            result.add(name)
    return result


def get_updated_channels_from_tasks(
    run_tasks: Iterable[Any],
) -> set[str]:
    """Channel names written by an update_state superstep (excluding PUSH)."""
    return {c for task in run_tasks for c, _ in task.writes if c != PUSH}


def get_delta_channels_from_all_channels(
    channels: Mapping[str, BaseChannel],
    channel_versions: ChannelVersions,
) -> set[str]:
    """DeltaChannels to snapshot on the first update_state of a fresh thread:
    the ones it wrote, as the rest have no version and nothing to store."""
    return {
        k
        for k, ch in channels.items()
        if isinstance(ch, DeltaChannel) and ch.is_available() and k in channel_versions
    }


def delta_channels_with_pending_writes(
    specs: Mapping[str, Any],
    pending_writes: Iterable[PendingWrite] | None,
) -> set[str]:
    """DeltaChannels a branch starting from this checkpoint must snapshot.

    A checkpoint's pending writes belong to the child that consumed them, and
    nothing records which child that was. A new branch snapshots every delta
    channel they touch, so its ancestor walk never replays them.
    """
    return {
        ch
        for _, ch, _ in pending_writes or ()
        if isinstance(specs.get(ch), DeltaChannel)
    }


def create_metadata_for_update_state_api(
    channels: Mapping[str, BaseChannel],
    updated_channels: set[str],
    *,
    prev_metadata: Mapping[str, Any] | None,
) -> dict[str, tuple[int, int]]:
    """Advance ``counters_since_delta_snapshot`` for update_state on a non-fresh thread.

    Mirrors the per-superstep counter bump in ``_loop._put_checkpoint``.
    """
    prev_counters = dict(
        (prev_metadata or {}).get("counters_since_delta_snapshot") or {}
    )
    new_counters: dict[str, tuple[int, int]] = {}
    for ch_name, ch in channels.items():
        if not isinstance(ch, DeltaChannel):
            continue
        u, s = prev_counters.get(ch_name, (0, 0))
        s += 1
        if ch_name in updated_channels:
            u += 1
        new_counters[ch_name] = (u, s)
    return new_counters


def create_checkpoint_plan_for_update_state_api(
    channels: Mapping[str, BaseChannel],
    updated_channels: set[str],
    *,
    step: int,
    parents: dict[str, Any],
    saved_metadata: Mapping[str, Any] | None,
    is_fresh_thread: bool,
    fork_channels: set[str],
    channel_versions: ChannelVersions,
) -> tuple[set[str], dict[str, Any]]:
    """Return ``(channels_to_snapshot, metadata)`` for an update_state head."""
    metadata: dict[str, Any] = {
        "source": "update",
        "step": step,
        "parents": parents,
    }
    if is_fresh_thread:
        return get_delta_channels_from_all_channels(
            channels, channel_versions
        ), metadata

    new_counters = create_metadata_for_update_state_api(
        channels,
        updated_channels,
        prev_metadata=saved_metadata,
    )
    channels_to_snapshot = (
        delta_channels_to_snapshot(channels, new_counters, channel_versions)
        | fork_channels
    )
    for k in channels_to_snapshot:
        new_counters[k] = (0, 0)
    non_zero = {k: v for k, v in new_counters.items() if v != (0, 0)}
    if non_zero:
        metadata["counters_since_delta_snapshot"] = non_zero
    return channels_to_snapshot, metadata


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel] | None,
    step: int,
    *,
    id: str | None = None,
    updated_channels: set[str] | None = None,
    get_next_version: GetNextVersion | None = None,
    channels_to_snapshot: set[str] | None = None,
    stored_versions: ChannelVersions | None = None,
) -> Checkpoint:
    """Build a new Checkpoint from the previous one and live channel state.

    For each name in `channels_to_snapshot`, a `_DeltaSnapshot(value)` blob
    is written into `channel_values[k]`. Other delta channels are omitted
    from `channel_values` — the ancestor walk reconstructs their state
    from `checkpoint_writes`. Callers compute the set via
    `delta_channels_to_snapshot(channels, counters)`; defaults to empty
    (no snapshots) when not provided.

    `stored_versions` are the channel versions of the last checkpoint the
    saver stored. A snapshotted channel whose version has not moved since
    then is bumped.
    """
    ts = datetime.now(timezone.utc).isoformat()
    channels_to_snapshot = channels_to_snapshot or set()
    bumped: dict[str, tuple[Any, Any]] = {}
    if channels is None:
        values = checkpoint["channel_values"]
        channel_versions = checkpoint["channel_versions"]
    else:
        values = {}
        channel_versions = dict(checkpoint["channel_versions"])
        for k in channels:
            ch = channels[k]
            if k not in channel_versions:
                # A forced snapshot of a never-written channel still has to
                # land to stop the ancestor walk, and `put` only stores blobs
                # for versioned channels.
                if k in channels_to_snapshot and get_next_version is not None:
                    channel_versions[k] = get_next_version(None, None)
                    bumped[k] = (None, channel_versions[k])
                    values[k] = _DeltaSnapshot(ch.get())
                continue
            if k in channels_to_snapshot:
                # `put` only stores a blob for a channel whose version moved,
                # so snapshotting a channel this step did not write needs a
                # bump: exit mode reaching the cadence on a superstep that
                # skipped the channel, and a fork's first checkpoint.
                if (
                    get_next_version is not None
                    and stored_versions is not None
                    and channel_versions[k] == stored_versions.get(k)
                ):
                    old = channel_versions[k]
                    channel_versions[k] = get_next_version(old, None)
                    bumped[k] = (old, channel_versions[k])
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
        versions_seen=_mark_bumps_seen(checkpoint["versions_seen"], bumped),
        updated_channels=None if updated_channels is None else sorted(updated_channels),
    )


def _mark_bumps_seen(
    versions_seen: dict[str, ChannelVersions],
    bumped: Mapping[str, tuple[Any, Any]],
) -> dict[str, ChannelVersions]:
    """Advance whoever had seen a bumped channel's old version to the new one.

    A bump that only stores a snapshot is not a write. Left unseen, it would
    re-fire `interrupt_before` and rerun the channel's subscribers. For each
    entry it advances, `SNAPSHOT_BUMPS` keeps the new version and the one the
    node really read, so `versions_seen_without_bumps` can put the read back.
    """
    if not bumped:
        return versions_seen
    out = dict(versions_seen)
    marks = dict(versions_seen.get(SNAPSHOT_BUMPS, {}))
    for node, seen in versions_seen.items():
        if node == SNAPSHOT_BUMPS:
            continue
        for k, (old, new) in bumped.items():
            if seen.get(k) != old:
                continue
            advanced, read = _bump_keys(node, k)
            # If an earlier bump set `old`, the real read is already recorded.
            if old is not None and marks.get(advanced) != old:
                marks[read] = old
            marks[advanced] = new
            out[node] = {**out[node], k: new}
    if marks:
        out[SNAPSHOT_BUMPS] = marks
    return out


def _bump_keys(node: str, channel: str) -> tuple[str, str]:
    # Node names can't contain either separator, so the keys can't collide.
    return f"{node}{NS_SEP}{channel}", f"{node}{NS_END}{channel}"


def versions_seen_without_bumps(
    versions_seen: dict[str, ChannelVersions],
) -> dict[str, ChannelVersions]:
    """`versions_seen` as the nodes read it: an entry a bump advanced goes back
    to the version the node really read, or away if it never read one."""
    if not (marks := versions_seen.get(SNAPSHOT_BUMPS)):
        return versions_seen
    out: dict[str, ChannelVersions] = {}
    for node, seen in versions_seen.items():
        if node == SNAPSHOT_BUMPS:
            continue
        out[node] = {}
        for k, v in seen.items():
            advanced, read = _bump_keys(node, k)
            if marks.get(advanced) != v:
                out[node][k] = v
            elif read in marks:
                out[node][k] = marks[read]
    return out


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
