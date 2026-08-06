from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
)
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import _DeltaSnapshot

from langgraph._internal._config import DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT
from langgraph._internal._constants import PUSH
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
) -> set[str]:
    """Return the set of DeltaChannel names that should snapshot now.

    A channel snapshots when EITHER its accumulated update count reaches
    `snapshot_frequency` OR the total supersteps since its last snapshot
    reaches `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT`. This is a pure
    predicate — no mutation.
    """
    result: set[str] = set()
    for name, ch in channels.items():
        if not isinstance(ch, DeltaChannel) or not ch.is_available():
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
    *,
    include_unavailable: bool = False,
) -> set[str]:
    """Every available DeltaChannel.

    The set to snapshot whenever no ancestor walk can reconstruct these
    channels: the first update_state of a fresh thread (no ancestors at all),
    and the first checkpoint of a fork (whose base also holds the writes of the
    branch the fork abandons).
    """
    return {
        k
        for k, ch in channels.items()
        if isinstance(ch, DeltaChannel) and (include_unavailable or ch.is_available())
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
    is_fork: bool,
) -> tuple[set[str], dict[str, Any]]:
    """Return ``(channels_to_snapshot, metadata)`` for an update_state head.

    ``is_fork`` (the update was addressed at an explicit checkpoint) forces a
    full snapshot for the same reason ``is_fresh_thread`` does: the ancestor
    walk cannot reconstruct this head. The base a fork branches off keeps the
    pending writes of the branch being abandoned, and nothing records which
    child consumed which write, so the walk would replay them here too.
    Snapshotting terminates the walk at this checkpoint. Every delta channel
    snapshots, so no counters carry over.
    """
    metadata: dict[str, Any] = {
        "source": "update",
        "step": step,
        "parents": parents,
    }
    if is_fresh_thread or is_fork:
        return get_delta_channels_from_all_channels(
            channels, include_unavailable=is_fork
        ), metadata

    new_counters = create_metadata_for_update_state_api(
        channels,
        updated_channels,
        prev_metadata=saved_metadata,
    )
    channels_to_snapshot = delta_channels_to_snapshot(channels, new_counters)
    for k in channels_to_snapshot:
        new_counters[k] = (0, 0)
    non_zero = {k: v for k, v in new_counters.items() if v != (0, 0)}
    if non_zero:
        metadata["counters_since_delta_snapshot"] = non_zero
    return channels_to_snapshot, metadata


def create_fork_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    step: int,
    *,
    is_fork: bool,
    get_next_version: GetNextVersion,
) -> Checkpoint:
    """``create_checkpoint`` for an update_state path that bypasses the plan.

    The ``as_node`` INPUT and END paths write the fork's first checkpoint and
    return before ``create_checkpoint_plan_for_update_state_api`` runs. Left
    without a snapshot that checkpoint does not seal the fork, and the next
    superstep reconstructs its delta channels by walking through the shared
    base, picking up the abandoned branch's writes and then baking them into
    whatever it snapshots. Sealing has to happen on the fork's *first*
    checkpoint, which is this one.

    ``get_next_version`` is required for the same reason exit mode needs it:
    these paths apply writes to the input channel, not to the delta channel,
    so nothing bumps the delta channel's version and ``put`` would drop the
    blob as not-a-new-version. Callers must derive ``new_versions`` from the
    returned checkpoint rather than the one they passed in.
    """
    if not is_fork:
        return create_checkpoint(checkpoint, channels, step)
    return create_checkpoint(
        checkpoint,
        channels,
        step,
        get_next_version=get_next_version,
        channels_to_snapshot=get_delta_channels_from_all_channels(
            channels, include_unavailable=True
        ),
    )


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel] | None,
    step: int,
    *,
    id: str | None = None,
    updated_channels: set[str] | None = None,
    get_next_version: GetNextVersion | None = None,
    channels_to_snapshot: set[str] | None = None,
) -> Checkpoint:
    """Build a new Checkpoint from the previous one and live channel state.

    For each name in `channels_to_snapshot`, a `_DeltaSnapshot(value)` blob
    is written into `channel_values[k]`. Other delta channels are omitted
    from `channel_values` — the ancestor walk reconstructs their state
    from `checkpoint_writes`. Callers compute the set via
    `delta_channels_to_snapshot(channels, counters)`; defaults to empty
    (no snapshots) when not provided.
    """
    ts = datetime.now(timezone.utc).isoformat()
    channels_to_snapshot = channels_to_snapshot or set()
    if channels is None:
        values = checkpoint["channel_values"]
        channel_versions = checkpoint["channel_versions"]
    else:
        values = {}
        channel_versions = dict(checkpoint["channel_versions"])
        for k in channels:
            ch = channels[k]
            if k not in channel_versions:
                # Nothing was ever written to this channel on this branch, so
                # it has no version and `put` would drop any blob stored for
                # it. A *forced* snapshot still has to land: it is the only
                # thing that stops the ancestor walk running past this
                # checkpoint into a fork base that holds another branch's
                # writes. Mint a first version so the blob survives.
                if k in channels_to_snapshot and get_next_version is not None:
                    channel_versions[k] = get_next_version(None, None)
                    values[k] = _DeltaSnapshot(
                        ch.get() if ch.is_available() else ch.typ()
                    )
                continue
            if k in channels_to_snapshot:
                # Callers force a full snapshot blob here: exit mode when a
                # delta channel reaches its snapshot cadence, update_state on
                # a fresh thread (no ancestor to replay writes from), and a
                # fork (whose base also holds the abandoned branch's writes).
                # The manual version-bump below only applies to the exit-mode
                # case.
                #
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
