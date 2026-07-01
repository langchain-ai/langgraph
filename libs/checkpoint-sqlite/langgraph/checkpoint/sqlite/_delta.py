"""Shared helpers for `get_delta_channel_history` on sqlite savers.

Mirrors the supersteps-based two-pass shape of `BasePostgresSaver`
(ancestor walk bounded by `counters_since_delta_snapshot` + per-channel
UNION ALL writes fetch), adapted for sqlite's constraints:

* No JSONB — to inspect `channel_values` for a checkpoint we must
  deserialize the full blob. The WALK streams the cursor row-by-row and
  deserializes only the seed checkpoints (the ones at a channel's
  `supersteps` depth), freeing each blob before advancing.
* No separate blob table — `channel_values` lives inline in the
  checkpoint, so seeds come back from the WALK with no second fetch.
* Single shared parent-chain walk: each requested channel slices the
  same chain to its own `supersteps` depth.

Walk depth is driven by the *supersteps since last snapshot* counter,
not by scanning `channel_values` for the snapshot marker — the only
reliable way to locate seeds that aren't a `_DeltaSnapshot` sentinel
(e.g. legacy plain-value blobs from a thread migrated off a non-delta
channel).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.checkpoint.base import (
    DeltaChannelHistory,
    PendingWrite,
    _parse_supersteps_since_last_snapshot_by_channel,
)
from langgraph.checkpoint.serde.types import _DeltaSnapshot

logger = logging.getLogger(__name__)

# Re-exported under the package-local name used by the sqlite savers.
parse_supersteps_since_last_snapshot_by_channel = (
    _parse_supersteps_since_last_snapshot_by_channel
)

# The WALK streams ancestors of `target_id` newest-first. The `<=`
# predicate keeps target itself in the stream so we can read its
# `parent_checkpoint_id` from the first matching row without a separate
# lookup; target's own writes/seed are not part of the contract.
DELTA_WALK_SQL = (
    "SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint "
    "FROM checkpoints "
    "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id <= ? "
    "ORDER BY checkpoint_id DESC"
)


def build_delta_writes_fetch_sql(*, chain_lens: Sequence[int]) -> str:
    """Per-channel UNION ALL fetching writes from `writes`.

    One branch per channel with a non-empty chain. Each branch inlines its
    own `IN (?, ?, ...)` placeholder list because sqlite has no array-bind
    equivalent of postgres's `= ANY(?)`. Caller passes parameters in
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


def step_walk_supersteps(
    *,
    cid: str,
    parent_cid: str | None,
    type_tag: str,
    blob: bytes,
    target_id: str,
    serde: Any,
    shared_cpid_chain: list[str],
    walk_state: dict[str, Any],
    max_supersteps: int,
    needed_depths: set[int],
    seed_values_by_depth: dict[int, dict[str, Any]],
    channels: Sequence[str],
) -> bool:
    """Process one streamed WALK row, extending the shared parent chain.

    The cursor returns `(cid, parent_cid, type, checkpoint)` rows in
    `checkpoint_id` DESC order starting at target. The first row is target
    itself; we read its `parent_cid` to seed the walk and skip it (target's
    own writes/seed are not part of the contract). Off-path rows (a sibling
    branch on the same thread) advance the cursor without doing work.

    For each on-path ancestor we append its cid to `shared_cpid_chain`
    (newest first). When the ancestor sits at a depth some channel needs as
    its seed (`len(chain)` ∈ `needed_depths`), we deserialize it once and
    record its `channel_values` for the requested channels. The
    deserialized checkpoint is dropped immediately — peak in-flight is one
    deserialized checkpoint.

    Sets `walk_state["reached_root"]` when an ancestor has no parent.
    Returns True when the walk can stop: chain reached `max_supersteps`, or
    the root was reached.
    """
    if "started" not in walk_state:
        if cid == target_id:
            walk_state["started"] = True
            walk_state["cur_cid"] = parent_cid
            if parent_cid is None:
                walk_state["reached_root"] = True
                return True
        # Not target yet (or target not present): keep streaming.
        return False
    if len(shared_cpid_chain) >= max_supersteps:
        return True
    if cid != walk_state["cur_cid"]:
        # Off-path row from a sibling branch — skip without deserializing.
        return False
    shared_cpid_chain.append(cid)
    depth = len(shared_cpid_chain)  # 1-indexed position along the chain
    # Capture channel_values at any depth a channel may use as its seed: the
    # exact `supersteps` depths, plus the root-most checkpoint (the seed
    # candidate when the chain is shorter than `supersteps`).
    if depth in needed_depths or parent_cid is None:
        ckpt = serde.loads_typed((type_tag, blob))
        channel_values: Mapping[str, Any] = ckpt.get("channel_values") or {}
        seed_values_by_depth[depth] = {
            ch: channel_values[ch] for ch in channels if ch in channel_values
        }
        del ckpt, channel_values
    if parent_cid is None:
        walk_state["reached_root"] = True
        return True
    walk_state["cur_cid"] = parent_cid
    return len(shared_cpid_chain) >= max_supersteps


def resolve_delta_chains(
    *,
    channels: Sequence[str],
    supersteps_by_ch: Mapping[str, int],
    shared_cpid_chain: Sequence[str],
    seed_values_by_depth: Mapping[int, Mapping[str, Any]],
    has_reached_root: bool,
    thread_id: str,
) -> tuple[
    dict[str, list[str]],
    dict[str, str | None],
    dict[str, Any],
]:
    """Slice the shared parent chain into per-channel chain/seed mappings.

    For each channel the seed snapshot sits `supersteps` hops back: the seed
    checkpoint is `shared_cpid_chain[supersteps - 1]` and the chain is
    `shared_cpid_chain[:supersteps]` (newest first), with the seed's inline
    `channel_values[ch]` captured during the walk.

    When the chain is shorter than `supersteps` but the walk reached the
    root, the persisted chain is "compressed" relative to the logical
    superstep count (`durability="exit"`, or a thread that never
    snapshotted). The seed candidate is then the oldest persisted checkpoint
    (`shared_cpid_chain[-1]`); if its `channel_values[ch]` is absent the
    channel has no seed and replays the full chain on an empty baseline.
    """
    chained_cpid_by_ch: dict[str, list[str]] = {ch: [] for ch in channels}
    seed_cpid_by_ch: dict[str, str | None] = {ch: None for ch in channels}
    seed_value_by_ch: dict[str, Any] = {}
    for ch in channels:
        bound = supersteps_by_ch.get(ch, 0)
        if bound <= 0:
            continue
        if len(shared_cpid_chain) >= bound:
            seed_depth = bound
        elif has_reached_root:
            seed_depth = len(shared_cpid_chain)
        else:
            logger.warning(
                "cannot find seed snapshot for delta channel "
                "(thread_id=%s, channel=%s)",
                thread_id,
                ch,
            )
            continue
        if seed_depth <= 0:
            continue
        chained_cpid_by_ch[ch] = list(shared_cpid_chain[:seed_depth])
        seed_cpid_by_ch[ch] = shared_cpid_chain[seed_depth - 1]
        seed_vals = seed_values_by_depth.get(seed_depth, {})
        if ch in seed_vals:
            seed_value_by_ch[ch] = seed_vals[ch]
    return chained_cpid_by_ch, seed_cpid_by_ch, seed_value_by_ch


def build_delta_channels_writes_history(
    *,
    channels: Sequence[str],
    chained_cpid_by_ch: Mapping[str, Sequence[str]],
    seed_cpid_by_ch: Mapping[str, str | None],
    seed_value_by_ch: Mapping[str, Any],
    stage2_rows: Sequence[tuple[str, str, str, int, str, bytes]],
    serde: Any,
) -> dict[str, DeltaChannelHistory]:
    """Demux writes rows per channel; produce per-channel histories.

    `stage2_rows` are `(checkpoint_id, channel, task_id, idx, type, value)`.
    Final write order is oldest→newest globally and `(task_id, idx)` within
    a checkpoint, matching the contract on `DeltaChannelHistory.writes`.

    The seed checkpoint's own writes are replayed on top of a
    `_DeltaSnapshot` seed (the snapshot is the value *prior* to its own
    writes), but skipped for a migrated plain-value seed (a legacy
    non-delta blob already incorporates those writes). `seed` is omitted
    when no seed was located (implicit empty baseline); consumers treat
    absence as "start empty".
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
        entry: DeltaChannelHistory = {"writes": []}

        skip_seed_checkpoint_writes = False
        if ch in seed_value_by_ch:
            seed_value = seed_value_by_ch[ch]
            entry["seed"] = seed_value
            skip_seed_checkpoint_writes = not isinstance(seed_value, _DeltaSnapshot)

        cid_writes = writes_by_ch_by_cid.get(ch, {})
        if cid_writes:
            collected: list[PendingWrite] = []
            seed_cpid = seed_cpid_by_ch.get(ch)
            # Chain is newest→oldest; replay oldest→newest.
            for cid in reversed(list(chained_cpid_by_ch.get(ch, []))):
                if skip_seed_checkpoint_writes and cid == seed_cpid:
                    continue
                for type_tag, value_blob, task_id, _idx in cid_writes.get(cid, []):
                    collected.append(
                        (task_id, ch, serde.loads_typed((type_tag, value_blob)))
                    )
            entry["writes"] = collected
        result[ch] = entry
    return result
