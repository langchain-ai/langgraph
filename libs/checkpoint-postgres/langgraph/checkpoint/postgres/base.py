from __future__ import annotations

import logging
import random
import warnings
from collections.abc import Mapping, Sequence
from importlib.metadata import version as get_version
from typing import Any, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    DeltaChannelHistory,
    PendingWrite,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS, _DeltaSnapshot
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)

# Page size for the paged WALK scan in `get_delta_channel_history`. Internal
# constant — exposing this as a kwarg is left as a follow-up.
_DELTA_PAGE_SIZE = 1024

MetadataInput = dict[str, Any] | None

try:
    major, minor = get_version("langgraph").split(".")[:2]
    if int(major) == 0 and int(minor) < 5:
        warnings.warn(
            "You're using incompatible versions of langgraph and checkpoint-postgres. Please upgrade langgraph to avoid unexpected behavior.",
            DeprecationWarning,
            stacklevel=2,
        )
except Exception:
    # skip version check if running from source
    pass

"""
To add a new migration, add a new string to the MIGRATIONS list.
The position of the migration in the list is the version number.
"""
MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
);""",
    """CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);""",
    "ALTER TABLE checkpoint_blobs ALTER COLUMN blob DROP not null;",
    # NOTE: this is a no-op migration to ensure that the versions in the migrations table are correct.
    # This is necessary due to an empty migration previously added to the list.
    "SELECT 1;",
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id);
    """,
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
    """,
    """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
    """,
    """ALTER TABLE checkpoint_writes ADD COLUMN IF NOT EXISTS task_path TEXT NOT NULL DEFAULT '';""",
]

SELECT_SQL = """
select
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    (
        select array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob])
        from jsonb_each_text(checkpoint -> 'channel_versions')
        inner join checkpoint_blobs bl
            on bl.thread_id = checkpoints.thread_id
            and bl.checkpoint_ns = checkpoints.checkpoint_ns
            and bl.channel = jsonb_each_text.key
            and bl.version = jsonb_each_text.value
    ) as channel_values,
    (
        select
        array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] order by cw.task_id, cw.idx)
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns = checkpoints.checkpoint_ns
            and cw.checkpoint_id = checkpoints.checkpoint_id
    ) as pending_writes
from checkpoints """

SELECT_PENDING_SENDS_SQL = f"""
select
    checkpoint_id,
    array_agg(array[type::bytea, blob] order by task_path, task_id, idx) as sends
from checkpoint_writes
where thread_id = %s
    and checkpoint_id = any(%s)
    and channel = '{TASKS}'
group by checkpoint_id
"""

UPSERT_CHECKPOINT_BLOBS_SQL = """
    INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING
"""

UPSERT_CHECKPOINTS_SQL = """
    INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
    DO UPDATE SET
        checkpoint = EXCLUDED.checkpoint,
        metadata = EXCLUDED.metadata;
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
        channel = EXCLUDED.channel,
        type = EXCLUDED.type,
        blob = EXCLUDED.blob;
"""

INSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
"""


class _DeltaStage2Row(TypedDict, total=False):
    """One row from `_build_delta_fetch_sql` (a UNION ALL of writes and blobs)."""

    _kind: str  # "w" or "b"
    checkpoint_id: str | None  # "w" rows only
    channel: str | None  # set on both "w" and "b" rows
    type: str | None
    blob: bytes | None
    task_id: str | None  # "w" rows only
    idx: int | None  # "w" rows only
    version: str | None  # "b" rows only


# Multi-channel two-pass DeltaChannel reconstruction.
#
# A `DeltaChannel` does not store its full value at every checkpoint — it
# stores periodic full-value *snapshots* and accumulates intermediate
# *writes* between snapshots. To rebuild a channel's value at a target
# checkpoint we need:
#
#   - the **seed** — the most recent snapshot at-or-before the target
#     (a single blob row in `checkpoint_blobs`); and
#   - the **chain writes** — every write for this channel committed
#     between that snapshot and the target, in order
#     (rows in `checkpoint_writes`).
#
# Two passes, in order:
#
#   1. WALK — scan checkpoint metadata only (no blob bytes). For each
#      requested channel we read `channel_versions[ch]` (the seed's blob
#      version pointer). Then follow `parent_checkpoint_id` from the
#      target backwards in pages of `_DELTA_PAGE_SIZE` rows.
#
#      Walk depth is driven by the *supersteps since last snapshot*
#      counter — `metadata.counters_since_delta_snapshot[ch][1]` — read
#      from the target checkpoint. A channel's seed snapshot sits exactly
#      `supersteps` hops back along the parent chain; walking by the
#      counter (rather than scanning `channel_values` for the snapshot
#      marker) is the only reliable way to locate seeds that aren't a
#      `_DeltaSnapshot` sentinel — e.g. legacy plain-value blobs left by a
#      thread that migrated from a non-delta channel, which `put` stores
#      out of the inline `channel_values` map.
#
#   2. FETCH — given each channel's chained checkpoint ids and seed
#      version from WALK, pull only the rows we need: writes for those
#      exact checkpoint_ids and the seed blob at that exact version. One
#      roundtrip, per-channel UNION ALL — no over-fetch.
#
# Walking the *parent chain* (not `list(before=...)`) matters: forked
# threads have multiple branches, and only on-path ancestors contribute.


def _build_delta_walk_sql(channels: Sequence[str]) -> str:
    """Build the paged WALK SQL — scans checkpoint metadata only.

    Emits one row per checkpoint with K parallel JSONB key lookups (one
    `ver_i` column per requested channel — the channel's blob version, the
    pointer we dereference in FETCH if this checkpoint is the seed). No
    blob bytes; the result set fits a paged `LIMIT` cleanly.

    For channels=["messages", "files"] the result is::

        SELECT checkpoint_id, parent_checkpoint_id,
               checkpoint -> 'channel_versions' ->> %s AS ver_0,
               checkpoint -> 'channel_versions' ->> %s AS ver_1
        FROM checkpoints
        WHERE thread_id = %s AND checkpoint_ns = %s
          AND (%s::text IS NULL OR checkpoint_id < %s)
        ORDER BY checkpoint_id DESC
        LIMIT %s

    Channel names are passed as `%s` parameters (safe from SQL injection).
    Only the column aliases `ver_i` are interpolated into the SQL string
    (i is bounded by len(channels) and uses safe identifiers).

    Caller must extend params with `[ch_0, ch_1, ..., thread_id, ns,
    cursor, cursor, page_size]`. The `cursor` is the smallest
    `checkpoint_id` from the previous page (or `None` on the first page);
    `(%s::text IS NULL OR ...)` makes the first-page `WHERE` a no-op.
    """
    cols = [
        f"checkpoint -> 'channel_versions' ->> %s AS ver_{i}"
        for i in range(len(channels))
    ]
    return (
        "SELECT checkpoint_id, parent_checkpoint_id, "
        + ", ".join(cols)
        + " FROM checkpoints WHERE thread_id = %s AND checkpoint_ns = %s"
        " AND (%s::text IS NULL OR checkpoint_id < %s)"
        " ORDER BY checkpoint_id DESC LIMIT %s"
    )


def _build_delta_fetch_sql(
    *,
    channels_with_chain: Sequence[str],
    channels_with_seed: Sequence[str],
) -> str:
    """Build the FETCH SQL as a per-channel UNION ALL.

    For each channel with a non-empty chain, emit one branch reading
    `checkpoint_writes` for that specific channel + chain_cids. For each
    channel with a seed_version, emit one branch reading `checkpoint_blobs`
    for that channel + version. This avoids the over-fetch of a single
    `channel = ANY(channels) AND checkpoint_id = ANY(union)` form when
    channels have different chain depths.

    The caller must pass parameters in matching order:

        for ch in channels_with_chain:
            params += [thread_id, checkpoint_ns, ch, chain_cids[ch]]
        for ch in channels_with_seed:
            params += [thread_id, checkpoint_ns, ch, seed_version[ch]]

    Returns an empty SQL string if both channel lists are empty (caller
    must skip executing in that case).
    """
    branches: list[str] = []
    for _ in channels_with_chain:
        # NOTE: no ORDER BY on this branch — writes are sorted in assembly.
        branches.append(
            "SELECT 'w'::text AS _kind, "
            "checkpoint_id, channel, "
            "type, blob, task_id, idx, NULL::text AS version "
            "FROM checkpoint_writes "
            "WHERE thread_id = %s AND checkpoint_ns = %s AND channel = %s "
            "AND checkpoint_id = ANY(%s)"
        )
    for _ in channels_with_seed:
        branches.append(
            "SELECT 'b'::text AS _kind, NULL::text AS checkpoint_id, channel, "
            "type, blob, NULL::text AS task_id, NULL::int AS idx, version "
            "FROM checkpoint_blobs "
            "WHERE thread_id = %s AND checkpoint_ns = %s AND channel = %s "
            "AND version = %s"
        )
    return " UNION ALL ".join(branches)


def _ingest_walk_page(
    page_rows: Sequence[Mapping[str, Any]],
    channels: Sequence[str],
    parent_of: dict[str, str | None],
    ver_by_i_by_cid: list[dict[str, str | None]],
) -> str | None:
    """Fold one WALK page into `parent_of` + per-channel `ver_by_cid`.

    Returns the oldest checkpoint_id seen on this page (smallest, since
    pages come back DESC). Caller uses it as the cursor for the next page
    (`AND checkpoint_id < cursor`).
    """
    oldest: str | None = None
    for r in page_rows:
        cid = cast(str, r["checkpoint_id"])
        parent_of[cid] = cast("str | None", r["parent_checkpoint_id"])
        for i in range(len(channels)):
            ver_by_i_by_cid[i][cid] = cast("str | None", r.get(f"ver_{i}"))
        # Rows are DESC; the last one is the smallest cid in the page.
        oldest = cid
    return oldest


def _advance_shared_chain(
    target_id: str,
    parent_of: Mapping[str, str | None],
    shared_cpid_chain: list[str],
    max_supersteps: int,
) -> bool:
    """Extend the shared parent chain as far as ingested pages allow.

    The chain holds ancestors of the target, newest first: `chain[0]` is
    the target's parent, `chain[1]` its grandparent, and so on. A single
    chain is shared across all channels and grown to the maximum requested
    depth; each channel later slices `chain[:supersteps]`.

    Stops when:
      - the chain reaches `max_supersteps` hops, OR
      - the root is reached (parent is None) — returns True, OR
      - the next ancestor's cid isn't in `parent_of` yet (waits for the
        next page).

    Returns True iff the root was reached.
    """
    while len(shared_cpid_chain) < max_supersteps:
        top_of_chain = shared_cpid_chain[-1] if shared_cpid_chain else target_id
        if top_of_chain not in parent_of:
            return False  # wait for the next page
        parent = parent_of[top_of_chain]
        if parent is None:
            return True  # hit the root
        shared_cpid_chain.append(parent)
    return False


class BasePostgresSaver(BaseCheckpointSaver[str]):
    SELECT_SQL = SELECT_SQL
    SELECT_PENDING_SENDS_SQL = SELECT_PENDING_SENDS_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    supports_pipeline: bool

    def _migrate_pending_sends(
        self,
        pending_sends: list[tuple[bytes, bytes]],
        checkpoint: dict[str, Any],
        channel_values: list[tuple[bytes, bytes, bytes]],
    ) -> None:
        if not pending_sends:
            return
        # add to values
        enc, blob = self.serde.dumps_typed(
            [self.serde.loads_typed((c.decode(), b)) for c, b in pending_sends],
        )
        channel_values.append((TASKS.encode(), enc.encode(), blob))
        # add to versions
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    def _load_blobs(
        self, blob_values: list[tuple[bytes, bytes, bytes]]
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k.decode(): self.serde.loads_typed((t.decode(), v))
            for k, t, v in blob_values
            if t.decode() != "empty"
        }

    def _resolve_delta_chains(
        self,
        channels: Sequence[str],
        supersteps_by_ch: Mapping[str, int],
        shared_cpid_chain: Sequence[str],
        ver_by_i_by_cid: Sequence[Mapping[str, str | None]],
        has_reached_root: bool,
        thread_id: str,
    ) -> tuple[
        dict[str, list[str]],
        dict[str, str | None],
        dict[str, str | None],
    ]:
        """Slice the shared parent chain into per-channel chain/seed mappings.

        For each channel the seed snapshot sits `supersteps` hops back, so
        the seed checkpoint is `shared_cpid_chain[supersteps - 1]` and the
        chain is `shared_cpid_chain[:supersteps]` (newest first).

        When the chain is shorter than `supersteps` but the walk reached the
        root, the persisted chain is "compressed" relative to the logical
        superstep count — either because intermediate supersteps were never
        persisted (`durability="exit"`) or because the thread never produced
        a snapshot at all. In both cases the seed candidate is the oldest
        persisted checkpoint (`shared_cpid_chain[-1]`): FETCH loads its blob,
        and assembly keeps it only if non-empty (a real snapshot or migrated
        value) — otherwise it omits `seed` and replays the full chain on an
        empty baseline.
        """
        chained_cpid_by_ch: dict[str, list[str]] = {ch: [] for ch in channels}
        seed_cpid_by_ch: dict[str, str | None] = {ch: None for ch in channels}
        seed_ver_by_ch: dict[str, str | None] = {ch: None for ch in channels}
        for i, ch in enumerate(channels):
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
            seed_ver_by_ch[ch] = ver_by_i_by_cid[i].get(seed_cpid_by_ch[ch])
        return chained_cpid_by_ch, seed_cpid_by_ch, seed_ver_by_ch

    def _assemble_delta_history(
        self,
        *,
        channels: Sequence[str],
        chained_cpid_by_ch: Mapping[str, Sequence[str]],
        seed_cpid_by_ch: Mapping[str, str | None],
        seed_ver_by_ch: Mapping[str, str | None],
        fetch_rows: Sequence[_DeltaStage2Row],
    ) -> dict[str, DeltaChannelHistory]:
        """Demux FETCH rows per channel and produce per-channel histories.

        `fetch_rows` carry `channel` on every row. Write rows (`_kind = 'w'`)
        are bucketed per channel per checkpoint; seed-blob rows
        (`_kind = 'b'`) give each channel its snapshot value.

        The seed checkpoint's own writes are replayed on top of a
        `_DeltaSnapshot` seed (the snapshot is the value *prior* to its own
        writes), but skipped for a migrated plain-value seed (a legacy
        non-delta blob already incorporates those writes). The `seed` key is
        omitted when no seed was located or the blob is the "empty"
        tombstone — the consumer treats absence as "start empty".
        """
        # writes_by_ch_by_cid[channel][cid] = list of (type, blob, task_id, idx)
        writes_by_ch_by_cid: dict[str, dict[str, list[tuple[str, bytes, str, int]]]] = {
            ch: {} for ch in channels
        }
        seed_blob_by_ch: dict[str, tuple[str, bytes]] = {}

        for r in fetch_rows:
            ch = cast(str, r["channel"])
            if r["_kind"] == "w":
                cid = cast(str, r["checkpoint_id"])
                writes_by_ch_by_cid.setdefault(ch, {}).setdefault(cid, []).append(
                    cast(
                        "tuple[str, bytes, str, int]",
                        (r["type"], r["blob"], r["task_id"], r["idx"]),
                    )
                )
            else:  # _kind == "b" — the seed blob for this channel.
                seed_blob_by_ch[ch] = cast("tuple[str, bytes]", (r["type"], r["blob"]))

        # Within a checkpoint, writes apply oldest→newest by (task_id, idx).
        for cid_map in writes_by_ch_by_cid.values():
            for ws in cid_map.values():
                ws.sort(key=lambda w: (w[2], w[3]))

        result: dict[str, DeltaChannelHistory] = {}
        for ch in channels:
            entry: DeltaChannelHistory = {"writes": []}

            skip_seed_checkpoint_writes = False
            seed_blob = seed_blob_by_ch.get(ch)
            if seed_blob is not None and seed_blob[0] != "empty":
                seed_value = self.serde.loads_typed(seed_blob)
                entry["seed"] = seed_value
                # A migrated (non-delta) seed already includes the writes on
                # its own checkpoint; a `_DeltaSnapshot` does not.
                skip_seed_checkpoint_writes = not isinstance(seed_value, _DeltaSnapshot)

            cid_writes = writes_by_ch_by_cid.get(ch, {})
            if cid_writes:
                collected: list[PendingWrite] = []
                seed_cpid = seed_cpid_by_ch.get(ch)
                # Chain is newest→oldest; replay oldest→newest.
                for cid in reversed(chained_cpid_by_ch.get(ch, [])):
                    if skip_seed_checkpoint_writes and cid == seed_cpid:
                        continue
                    for type_tag, write_blob, task_id, _idx in cid_writes.get(cid, []):
                        val = self.serde.loads_typed((type_tag, write_blob))
                        collected.append((task_id, ch, val))
                entry["writes"] = collected
            result[ch] = entry
        return result

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, bytes | None]]:
        if not versions:
            return []

        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                cast(str, ver),
                *(
                    self.serde.dumps_typed(values[k])
                    if k in values
                    else ("empty", None)
                ),
            )
            for k, ver in versions.items()
        ]

    def _load_writes(
        self, writes: list[tuple[bytes, bytes, bytes, bytes]]
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid.decode(),
                    channel.decode(),
                    self.serde.loads_typed((t.decode(), v)),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _search_where(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None = None,
    ) -> tuple[str, list[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before.

        This method returns a tuple of a string and a tuple of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = $1 AND column2 IS $2". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = %s")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = %s ")
                param_values.append(checkpoint_id)

        # construct predicate for metadata filter
        if filter:
            wheres.append("metadata @> %s ")
            param_values.append(Jsonb(filter))

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < %s ")
            param_values.append(get_checkpoint_id(before))

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )
