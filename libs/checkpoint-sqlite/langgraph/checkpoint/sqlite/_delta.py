"""Shared helpers for `get_delta_channel_history` on sqlite savers.

Mirrors the two-stage shape of `BasePostgresSaver` (ancestor walk +
per-channel UNION ALL writes fetch), but adapted for sqlite's
constraints. The structural differences:

* No JSONB — to inspect `channel_values` for a checkpoint we must
  deserialize the full blob. Stage 1 streams the cursor row-by-row and
  deserializes only the rows the merged walk visits, freeing each blob
  before advancing.
* No separate blob table — `channel_values` lives inline in the
  checkpoint, so seeds come back from stage 1 with no second fetch.
* Single merged walk (not K independent walks): each visited cid is
  deserialized exactly once, regardless of how many channels are still
  seeking their seed.

The streaming design keeps peak in-flight memory at roughly one
deserialized checkpoint at a time, instead of holding the entire
ancestor chain's worth of raw blobs as a `fetchall()`-materialized list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.checkpoint.base import DeltaChannelHistory, PendingWrite

# Stage 1 streams ancestors of `target_cid` newest-first. The `<=`
# predicate keeps target itself in the stream so we can read its
# `parent_checkpoint_id` from the first row without a separate lookup;
# the caller skips target's own writes/seed (matches the
# `BaseCheckpointSaver` contract).
DELTA_STAGE1_SQL = (
    "SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint "
    "FROM checkpoints "
    "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id <= ? "
    "ORDER BY checkpoint_id DESC"
)


def build_delta_stage2_sql(*, chain_lens: Sequence[int]) -> str:
    """Stage-2 per-channel UNION ALL fetching writes from `writes`.

    One branch per channel with a non-empty chain. Each branch inlines its
    own `IN (?, ?, ...)` placeholder list because sqlite has no array-bind
    equivalent of postgres's `= ANY(%s)`. Caller passes parameters in
    matching order: `[thread_id, checkpoint_ns, channel, *chain_cids]` per
    branch.

    Returns an empty string when no channel has a chain (caller skips
    executing in that case). Per-channel UNION ALL avoids the over-fetch
    of a single `channel = ANY(channels)` filter when channels have
    different chain depths — same rationale as postgres.
    """
    branches: list[str] = []
    for n in chain_lens:
        cid_placeholders = ",".join("?" * n)
        branches.append(
            "SELECT checkpoint_id, channel, task_id, idx, type, value "
            "FROM writes "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? "
            f"AND checkpoint_id IN ({cid_placeholders})"
        )
    return " UNION ALL ".join(branches)


def step_walk_with_row(
    *,
    cid: str,
    parent_cid: str | None,
    type_tag: str,
    blob: bytes,
    target_id: str,
    serde: Any,
    chain_by_ch: dict[str, list[str]],
    seed_val_by_ch: dict[str, Any],
    walk_state: dict[str, Any],
    seeded: set[str],
    channels: Sequence[str],
) -> bool:
    """Process one streamed stage-1 row in the merged ancestor walk.

    The cursor returns (cid, parent_cid, type, blob) rows in
    `checkpoint_id` DESC order starting at target. The first row is
    target itself; we read its parent_cid to seed the walk and otherwise
    skip it (target's own writes/seed are not part of the contract).

    For each subsequent row, if `cid` matches the walk's current
    position, we deserialize the blob, append the cid to every
    not-yet-seeded channel's chain, and check `channel_values` for
    seeds. The deserialized checkpoint is dropped before advancing — no
    cross-row cache, so peak in-flight is one deserialized checkpoint.

    Off-path rows (different branch on the same thread) advance the
    cursor without doing any work.

    Returns True when every requested channel is seeded — the caller
    can stop iterating and close the cursor.
    """
    if "started" not in walk_state:
        if cid == target_id:
            walk_state["started"] = True
            walk_state["cur_cid"] = parent_cid
            walk_state["active"] = {ch for ch in channels if ch not in seeded}
        # Not target yet (or target not present): keep streaming.
        return False
    active: set[str] = walk_state["active"]
    if not active:
        return True
    if cid != walk_state["cur_cid"]:
        # Off-path row from a sibling branch — skip without deserializing.
        return False
    for ch in active:
        chain_by_ch[ch].append(cid)
    ckpt = serde.loads_typed((type_tag, blob))
    channel_values: Mapping[str, Any] = ckpt.get("channel_values") or {}
    for ch in [ch for ch in active if ch in channel_values]:
        seed_val_by_ch[ch] = channel_values[ch]
        seeded.add(ch)
        active.discard(ch)
    del ckpt, channel_values
    walk_state["cur_cid"] = parent_cid
    return not active


def build_delta_channels_writes_history(
    *,
    channels: Sequence[str],
    chain_by_ch: Mapping[str, list[str]],
    seed_val_by_ch: Mapping[str, Any],
    seeded: set[str],
    stage2_rows: Sequence[tuple[str, str, str, int, str, bytes]],
    serde: Any,
) -> dict[str, DeltaChannelHistory]:
    """Demux stage-2 rows per channel; produce per-channel histories.

    Stage-2 rows are `(checkpoint_id, channel, task_id, idx, type, value)`.
    Final write order is oldest→newest globally and `(task_id, idx)` within
    a checkpoint, matching the contract on `DeltaChannelHistory.writes`.

    `seed` is omitted when the walk reached a true root with no snapshot
    found (channel never entered `seeded`); consumers treat absence as
    "start empty".
    """
    writes_by_ch_by_cid: dict[str, dict[str, list[tuple[str, bytes, str, int]]]] = {
        ch: {} for ch in channels
    }
    for cid, ch, task_id, idx, type_tag, value_blob in stage2_rows:
        writes_by_ch_by_cid.setdefault(ch, {}).setdefault(cid, []).append(
            (type_tag, value_blob, task_id, idx)
        )
    for cid_map in writes_by_ch_by_cid.values():
        for ws in cid_map.values():
            ws.sort(key=lambda w: (w[2], w[3]))

    result: dict[str, DeltaChannelHistory] = {}
    for ch in channels:
        chain_cids = chain_by_ch.get(ch, [])
        cid_writes = writes_by_ch_by_cid.get(ch, {})
        collected: list[PendingWrite] = []
        # Chain is newest-first; iterate oldest-first for the public order.
        for cid in reversed(chain_cids):
            for type_tag, value_blob, task_id, _idx in cid_writes.get(cid, []):
                collected.append(
                    (task_id, ch, serde.loads_typed((type_tag, value_blob)))
                )
        entry: DeltaChannelHistory = {"writes": collected}
        if ch in seeded:
            entry["seed"] = seed_val_by_ch[ch]
        result[ch] = entry
    return result
