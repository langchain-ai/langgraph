from __future__ import annotations

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
from langgraph.checkpoint.serde.types import TASKS
from psycopg.types.json import Jsonb

# Page size for stage-1 paged scan in `get_delta_channel_history`. Internal
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
    """One row from `_build_delta_stage2_sql` (a UNION ALL of writes and blobs)."""

    _kind: str  # "w" or "b"
    checkpoint_id: str | None  # "w" rows only
    channel: str | None  # set on both "w" and "b" rows
    type: str | None
    blob: bytes | None
    task_id: str | None  # "w" rows only
    idx: int | None  # "w" rows only
    version: str | None  # "b" rows only


# Multi-channel two-stage DeltaChannel reconstruction.
#
# Stage 1 scans checkpoint metadata (no blob bytes) and emits one row per
# checkpoint with K parallel JSONB key lookups (one column pair per
# requested delta channel: ver_i / hs_i).  No subqueries, no aggregation.
# Python walks the parent chain once across all channels.
#
# Stage 2 fetches all writes and the seed blobs for ALL channels in a
# single roundtrip via `channel = ANY(%s)` and chain/seed-version
# filtering.
#
# Empirical comparison vs an alternative "ship full channel_versions /
# channel_values JSONB and let Python pick" form (1000 checkpoints,
# 8 total channels in graph, 3 delta channels requested):
#
#   Postgres execution:    A=0.24ms vs B=0.38ms   (both negligible)
#   End-to-end latency:    A=6.83ms vs B=2.28ms   (B is 3.0x faster)
#   Wire payload:          A=836KB  vs B=330KB    (61% smaller)
#   Buffer hits:           identical (167 blocks)
#
# B (this dynamic-columns design) wins because it avoids JSONB
# serialization on the wire and JSONB-to-dict deserialization in
# psycopg.  Even at K=8 (8 delta channels = 16 dynamic columns), B
# still beats A end-to-end (4.2ms vs 6.8ms).


def _build_delta_stage1_sql(channels: Sequence[str], *, paged: bool) -> str:
    """Build stage 1 SQL with 2K parallel JSONB key lookups.

    For channels=["messages", "files"] (with `paged=True`) the result is::

        SELECT checkpoint_id, parent_checkpoint_id,
               checkpoint -> 'channel_versions' ->> %s AS ver_0,
               (checkpoint -> 'channel_values' -> %s) IS NOT NULL AS hs_0,
               checkpoint -> 'channel_versions' ->> %s AS ver_1,
               (checkpoint -> 'channel_values' -> %s) IS NOT NULL AS hs_1
        FROM checkpoints
        WHERE thread_id = %s AND checkpoint_ns = %s
          AND (%s::text IS NULL OR checkpoint_id < %s)
        ORDER BY checkpoint_id DESC
        LIMIT %s

    Channel names are passed as `%s` parameters (safe from SQL injection).
    Only the column aliases `ver_i` / `hs_i` are interpolated into the
    SQL string (i is bounded by len(channels) and uses safe identifiers).

    Caller must extend params with `[ch_0, ch_0, ch_1, ch_1, ...,
    thread_id, ns, cursor, cursor, page_size]` when `paged=True`.

    When `paged=False`, the WHERE has no cursor predicate and there's no
    LIMIT/ORDER BY — kept as a non-public helper for tests/diagnostics.
    """
    cols = []
    for i in range(len(channels)):
        cols.append(
            f"checkpoint -> 'channel_versions' ->> %s AS ver_{i}, "
            f"(checkpoint -> 'channel_values' -> %s) IS NOT NULL AS hs_{i}"
        )
    sql = (
        "SELECT checkpoint_id, parent_checkpoint_id, "
        + ", ".join(cols)
        + " FROM checkpoints WHERE thread_id = %s AND checkpoint_ns = %s"
    )
    if paged:
        sql += (
            " AND (%s::text IS NULL OR checkpoint_id < %s)"
            " ORDER BY checkpoint_id DESC LIMIT %s"
        )
    return sql


def _build_delta_stage2_sql(
    *,
    channels_with_chain: Sequence[str],
    channels_with_seed: Sequence[str],
) -> str:
    """Build stage 2 SQL as a per-channel UNION ALL.

    For each channel with a non-empty chain, emit one branch reading
    `checkpoint_writes` for that specific channel + chain_cids. For each
    channel with a seed_version, emit one branch reading `checkpoint_blobs`
    for that channel + version. This avoids the over-fetch of the prior
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


# Stage 1 rows are dynamic-shape dicts: {checkpoint_id, parent_checkpoint_id,
# ver_0, hs_0, ver_1, hs_1, ...}.  Walking is parameterized by the channel
# list to map indices back to channel names — no static TypedDict here.
# `dict[str, Any]` is the practical signature.


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

    @staticmethod
    def _ingest_stage1_page(
        stage1_rows: Sequence[Mapping[str, Any]],
        channels: Sequence[str],
        parent_of: dict[str, str | None],
        ver_by_i_by_cid: list[dict[str, str | None]],
        hs_by_i_by_cid: list[dict[str, bool]],
    ) -> str | None:
        """Fold one stage-1 page into the running walk-state mappings.

        Returns the oldest checkpoint_id seen on this page (smallest, since
        pages come back DESC). Caller uses it as the cursor for the next
        page (`AND checkpoint_id < cursor`).
        """
        oldest: str | None = None
        for r in stage1_rows:
            cid = cast(str, r["checkpoint_id"])
            parent_of[cid] = cast("str | None", r["parent_checkpoint_id"])
            for i in range(len(channels)):
                ver_by_i_by_cid[i][cid] = cast("str | None", r.get(f"ver_{i}"))
                hs_by_i_by_cid[i][cid] = bool(r.get(f"hs_{i}"))
            # Rows are DESC; the last one is the smallest cid in the page.
            oldest = cid
        return oldest

    @staticmethod
    def _try_advance_walks(
        target_id: str,
        channels: Sequence[str],
        parent_of: Mapping[str, str | None],
        ver_by_i_by_cid: Sequence[Mapping[str, str | None]],
        hs_by_i_by_cid: Sequence[Mapping[str, bool]],
        chain_by_ch: dict[str, list[str]],
        seed_ver_by_ch: dict[str, str | None],
        walk_cursor_by_ch: dict[str, str | None],
        seeded: set[str],
    ) -> None:
        """Advance each not-yet-seeded channel's walk as far as possible.

        Uses the partial `parent_of` map accumulated so far. A walk stops
        either because:
          (a) it found a snapshot for its channel (channel becomes seeded),
          (b) it reached a real root (parent_of[cid] is None — fully
              materialized at this point), or
          (c) the next ancestor cid isn't in `parent_of` yet (waiting for
              a later page; the cursor stays put).

        Mutates `chain_by_ch`, `seed_ver_by_ch`, `walk_cursor_by_ch`, and
        `seeded` in place.
        """
        for i, ch in enumerate(channels):
            if ch in seeded:
                continue
            # First-time entry: cursor starts at the target's parent.
            if ch not in walk_cursor_by_ch:
                walk_cursor_by_ch[ch] = parent_of.get(target_id)
            cur_cid = walk_cursor_by_ch[ch]
            ch_chain = chain_by_ch[ch]
            hs_i = hs_by_i_by_cid[i]
            ver_i = ver_by_i_by_cid[i]
            while cur_cid is not None:
                if cur_cid not in parent_of:
                    # Need more pages to continue this walk.
                    break
                ch_chain.append(cur_cid)
                if hs_i.get(cur_cid, False):
                    seed_ver_by_ch[ch] = ver_i.get(cur_cid)
                    seeded.add(ch)
                    cur_cid = None
                    break
                cur_cid = parent_of[cur_cid]
            walk_cursor_by_ch[ch] = cur_cid

    def _build_delta_channels_writes_history(
        self,
        *,
        channels: Sequence[str],
        chain_by_ch: Mapping[str, list[str]],
        seed_ver_by_ch: Mapping[str, str | None],
        stage2_rows: Sequence[_DeltaStage2Row],
    ) -> dict[str, DeltaChannelHistory]:
        """Demux stage 2 rows per channel; produce per-channel histories.

        stage2_rows carry `channel` on every row. We build per-channel
        `writes_by_cid` and per-channel `seed_blob` dicts, then assemble
        a `DeltaChannelHistory` per requested channel. The `seed` key is omitted
        when the walk reached root with no snapshot found, or when the
        seed blob is sentinel "empty" — in both cases the consumer treats
        absence as "start empty".
        """
        # writes_by_ch_by_cid[channel][cid] = list of (type, blob, task_id, idx)
        writes_by_ch_by_cid: dict[str, dict[str, list[tuple[str, bytes, str, int]]]] = {
            ch: {} for ch in channels
        }
        # seed_blob_by_ver[(channel, version)] = (type, blob)
        seed_blob_by_ver: dict[tuple[str, str], tuple[str, bytes]] = {}

        for r in stage2_rows:
            ch = cast(str, r["channel"])
            kind = r["_kind"]
            if kind == "w":
                cid = cast(str, r["checkpoint_id"])
                writes_by_ch_by_cid.setdefault(ch, {}).setdefault(cid, []).append(
                    cast(
                        "tuple[str, bytes, str, int]",
                        (r["type"], r["blob"], r["task_id"], r["idx"]),
                    )
                )
            else:  # kind == "b"
                ver = cast(str, r["version"])
                seed_blob_by_ver[(ch, ver)] = cast(
                    "tuple[str, bytes]", (r["type"], r["blob"])
                )

        # Sort writes per (channel, cid) newest-first by (task_id, idx)
        for cid_map in writes_by_ch_by_cid.values():
            for ws in cid_map.values():
                ws.sort(key=lambda w: (w[2], w[3]), reverse=True)

        result: dict[str, DeltaChannelHistory] = {}
        for ch in channels:
            chain_cids = chain_by_ch.get(ch, [])
            seed_version = seed_ver_by_ch.get(ch)

            collected: list[PendingWrite] = []
            cid_writes = writes_by_ch_by_cid.get(ch, {})
            for cid in chain_cids:
                for type_tag, write_blob, task_id, _idx in cid_writes.get(cid, []):
                    val = self.serde.loads_typed((type_tag, write_blob))
                    collected.append((task_id, ch, val))
            collected.reverse()

            entry: DeltaChannelHistory = {"writes": collected}
            if seed_version is not None:
                blob = seed_blob_by_ver.get((ch, seed_version))
                if blob is not None and blob[0] != "empty":
                    entry["seed"] = self.serde.loads_typed(blob)
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
