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
    """Return value of :func:`create_checkpoint`."""

    checkpoint: Checkpoint
    """The checkpoint to persist via ``saver.put()``."""
    snapshotted: set[str]
    """DeltaChannel names that were snapshotted this step.  The caller
    should reset their ``updates_since_snapshot`` counters to ``0``."""


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
    mutated_checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel] | None,
    step: int,
    *,
    id: str | None = None,
    updated_channels: set[str] | None = None,
    get_next_version: GetNextVersion | None = None,
    force_delta_snapshot: bool = False,
    updates_since_snapshot: Mapping[str, int] | None = None,
) -> CreateCheckpointResult:
    """Build a new ``Checkpoint`` from the previous one and live channel state.

    Args:
        mutated_checkpoint: The checkpoint that has been mutated in-place by
            ``apply_writes`` (its ``channel_versions`` and ``versions_seen``
            already reflect this superstep's writes).  The new checkpoint
            inherits these mutated values and builds new ``channel_values``
            from the live channels.
        channels: In-memory channel objects whose state was updated by
            ``apply_writes``.  Needed because ``mutated_checkpoint`` only
            carries updated version metadata — its ``channel_values`` still
            holds stale blobs from the previous checkpoint load.  This
            function calls ``ch.checkpoint()`` (or wraps in ``_DeltaSnapshot``
            for DeltaChannels) to produce fresh serialised ``channel_values``.
            Pass ``None`` to skip serialisation — the new checkpoint reuses
            ``mutated_checkpoint``'s ``channel_values`` as-is (used by
            ``durability="exit"`` intermediate steps or when no checkpointer
            is present).
        step: Superstep number, used to generate the checkpoint ``id`` via
            ``uuid6(clock_seq=step)`` when *id* is not provided explicitly.
        id: Explicit checkpoint id.  When supplied (e.g. during ``exiting``),
            overrides the ``uuid6``-based generation.
        updated_channels: Set of channel names written during this superstep
            (produced by ``apply_writes``).  Persisted as a sorted list in
            the new checkpoint for efficient cold-start scheduling.
        get_next_version: Version generator (e.g. ``saver.get_next_version``).
            ``None`` means version tracking is skipped (lightweight / no-
            checkpointer mode).
        force_delta_snapshot: When ``True``, every DeltaChannel is snapshotted
            regardless of ``snapshot_frequency``.  Used by ``durability="exit"``
            where intermediate ``checkpoint_writes`` are not stored, so ancestor
            replay would have nothing to replay from.
        updates_since_snapshot: *Read-only* counters — maps each DeltaChannel
            name to the number of updates since its last snapshot.  Used by
            ``_should_snapshot_delta`` to decide whether to snapshot now.

    Returns:
        A tuple of:
        - The checkpoint to persist via ``saver.put()``, with a fresh
          ``id``, ``ts``, and serialised ``channel_values`` /
          ``channel_versions``.
        - Names of DeltaChannels that were snapshotted this step.  The
          caller should reset their ``updates_since_snapshot`` counters
          to ``0``.
    """
    ts = datetime.now(timezone.utc).isoformat()
    counts = updates_since_snapshot or {}
    snapshotted: set[str] = set()
    if channels is None:
        values = mutated_checkpoint["channel_values"]
        channel_versions = mutated_checkpoint["channel_versions"]
    else:
        values = {}
        channel_versions = dict(mutated_checkpoint["channel_versions"])
        for k in channels:
            # Channel has never been written to (no version entry from
            # apply_writes), so there is no meaningful state to checkpoint.
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
                # Force-snapshot (durability="exit"): some channels may not
                # have been written this step, so apply_writes didn't bump
                # their version.  Manually bump so saver.put() persists the
                # blob.  Other channels in the same force-snapshot batch
                # *were* written this step and already bumped by
                # apply_writes — those are skipped below to avoid
                # double-bumping.  In the normal count-based path the
                # channel was necessarily written (otherwise count can't
                # reach snapshot_frequency), so this branch never fires.
                # TODO: force-snapshot on every exit is wasteful for short
                # runs — a 1-message run still serialises the full state.
                # The right fix is to persist checkpoint_writes for delta
                # channels in durability="exit" mode so ancestor replay
                # works, eliminating the need for force-snapshot entirely.
                if (
                    force_delta_snapshot
                    and get_next_version is not None
                    # Even in force-snapshot mode, some channels were
                    # written this step and already bumped by apply_writes.
                    # Skip those to avoid double-bumping.
                    and k not in (updated_channels or ())
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
            versions_seen=mutated_checkpoint["versions_seen"],
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
