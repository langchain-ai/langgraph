from __future__ import annotations

import os
import random
import warnings
from collections.abc import Sequence
from importlib.metadata import version as get_version
from typing import Any, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    DELTA_SENTINEL,
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    PendingWrite,
    _ChannelWritesHistory,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS
from psycopg.types.json import Jsonb

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


class _DeltaCombinedRow(TypedDict, total=False):
    """One row from `SELECT_DELTA_COMBINED_SQL` (a UNION ALL of three tables).

    Every row carries `_kind` ("p" / "w" / "b") plus whichever columns are
    relevant for that kind; irrelevant columns are NULL and typed as `None`.
    """

    _kind: str  # always present: "p", "w", or "b"
    # checkpoint row ("p")
    checkpoint_id: str | None
    parent_checkpoint_id: str | None
    ver: str | None
    # write / blob rows ("w", "b")
    type: str | None
    blob: bytes | None
    # write row only ("w")
    task_id: str | None
    idx: int | None
    # blob row only ("b")
    version: str | None


# DeltaChannel reconstruction: one UNION ALL query fetches checkpoints,
# writes, and blobs for `channel` in one roundtrip; the ancestor walk runs
# in Python in `_build_delta_channel_writes_history`.
#
# Parameter order: (channel, thread_id, checkpoint_ns,
#                   thread_id, checkpoint_ns, channel,
#                   thread_id, checkpoint_ns, channel)
SELECT_DELTA_COMBINED_SQL = """
    SELECT 'p'::text       AS _kind,
           checkpoint_id,
           parent_checkpoint_id,
           checkpoint -> 'channel_versions' ->> %s AS ver,
           NULL::text       AS type,
           NULL::bytea      AS blob,
           NULL::text       AS task_id,
           NULL::int        AS idx,
           NULL::text       AS version
    FROM checkpoints
    WHERE thread_id = %s AND checkpoint_ns = %s
    UNION ALL
    SELECT 'w',
           checkpoint_id, NULL, NULL,
           type, blob, task_id, idx, NULL
    FROM checkpoint_writes
    WHERE thread_id = %s AND checkpoint_ns = %s AND channel = %s
    UNION ALL
    SELECT 'b',
           NULL, NULL, NULL,
           type, blob, NULL, NULL, version
    FROM checkpoint_blobs
    WHERE thread_id = %s AND checkpoint_ns = %s AND channel = %s
"""


# Two-stage DeltaChannel reconstruction.  Stage 1 scans checkpoint
# metadata (no blob bytes) to walk the parent chain and locate the
# nearest snapshot marker.  Stage 2 fetches only the chain-limited
# writes and the single seed snapshot blob.
#
# Parameter order:
#   stage1: (channel, channel, thread_id, checkpoint_ns)
#   stage2: (thread_id, checkpoint_ns, channel, chain_cids[],
#            thread_id, checkpoint_ns, channel, seed_versions[])

SELECT_DELTA_STAGE1_SQL = """
    SELECT checkpoint_id,
           parent_checkpoint_id,
           checkpoint -> 'channel_versions' ->> %s AS ver,
           (checkpoint -> 'channel_values' -> %s) IS NOT NULL AS has_snapshot
    FROM checkpoints
    WHERE thread_id = %s AND checkpoint_ns = %s
"""

SELECT_DELTA_STAGE2_SQL = """
    SELECT 'w'::text AS _kind,
           checkpoint_id,
           type, blob, task_id, idx, NULL::text AS version
    FROM checkpoint_writes
    WHERE thread_id = %s AND checkpoint_ns = %s AND channel = %s
      AND checkpoint_id = ANY(%s)
    UNION ALL
    SELECT 'b', NULL,
           type, blob, NULL, NULL, version
    FROM checkpoint_blobs
    WHERE thread_id = %s AND checkpoint_ns = %s AND channel = %s
      AND version = ANY(%s)
"""


class _DeltaStage1Row(TypedDict):
    """One row from `SELECT_DELTA_STAGE1_SQL`."""

    checkpoint_id: str
    parent_checkpoint_id: str | None
    ver: str | None
    has_snapshot: bool


def _two_stage_enabled() -> bool:
    """True when the two-stage DeltaChannel read path is opted-in."""
    return os.environ.get("LG_DELTA_TWO_STAGE_QUERY", "").lower() in (
        "1",
        "true",
        "yes",
    )


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

    def _build_delta_channel_writes_history(
        self,
        *,
        channel: str,
        target_id: str,
        rows: Sequence[_DeltaCombinedRow],
    ) -> _ChannelWritesHistory:
        """Reconstruct one delta channel's history from the combined UNION ALL rows.

        Pure data transform shared by sync (`PostgresSaver`) and async
        (`AsyncPostgresSaver`); both paths run `SELECT_DELTA_COMBINED_SQL`
        and feed the tagged rows here.

        Walk is newest → oldest from the target's parent. A non-sentinel
        blob in `checkpoint_blobs` (a pre-delta snapshot) terminates the
        walk and is returned as the seed so replay starts from it.

        Writes stored at `target_id` itself are pending writes for the next
        step and are excluded — the walk begins at the target's parent.
        """
        parent_of: dict[str, str | None] = {}
        ver_of: dict[str, str | None] = {}
        writes_by_cid: dict[str, list[tuple[str, bytes, str, int]]] = {}
        blob_by_ver: dict[str, tuple[str, bytes]] = {}

        for r in rows:
            kind = r["_kind"]
            if kind == "p":
                cid = cast(str, r["checkpoint_id"])
                parent_of[cid] = r["parent_checkpoint_id"]
                ver_of[cid] = r["ver"]
            elif kind == "w":
                cid = cast(str, r["checkpoint_id"])
                writes_by_cid.setdefault(cid, []).append(
                    cast(
                        "tuple[str, bytes, str, int]",
                        (r["type"], r["blob"], r["task_id"], r["idx"]),
                    )
                )
            else:  # kind == "b"
                blob_by_ver[cast(str, r["version"])] = cast(
                    "tuple[str, bytes]", (r["type"], r["blob"])
                )

        # newest write first per ancestor (task_id DESC, idx DESC)
        for ws in writes_by_cid.values():
            ws.sort(key=lambda w: (w[2], w[3]), reverse=True)

        ancestors: list[str] = []
        cur_cid: str | None = parent_of.get(target_id)
        while cur_cid is not None:
            ancestors.append(cur_cid)
            cur_cid = parent_of.get(cur_cid)
        if not ancestors:
            return _ChannelWritesHistory(seed=DELTA_SENTINEL, writes=[])

        collected: list[PendingWrite] = []  # newest first; reversed at the end
        for cid in ancestors:
            # Collect writes first — they encode the transition FROM this
            # ancestor's state to its child's and must be included even if
            # this ancestor is also the seed checkpoint.
            for type_tag, write_blob, task_id, _idx in writes_by_cid.get(cid, []):
                val = self.serde.loads_typed((type_tag, write_blob))
                collected.append((task_id, channel, val))
            # Then check seed terminator.
            ver = ver_of.get(cid)
            if ver is not None:
                seed_blob = blob_by_ver.get(ver)
                if seed_blob is not None and seed_blob[0] != "empty":
                    blob_value = self.serde.loads_typed(seed_blob)
                    if blob_value is not DELTA_SENTINEL:
                        collected.reverse()
                        return _ChannelWritesHistory(seed=blob_value, writes=collected)

        collected.reverse()  # oldest → newest
        return _ChannelWritesHistory(seed=DELTA_SENTINEL, writes=collected)

    @staticmethod
    def _walk_stage1(
        stage1_rows: Sequence[_DeltaStage1Row],
        target_id: str,
    ) -> tuple[list[str], str | None]:
        """Walk the parent chain from stage 1 metadata rows.

        Returns (chain_cids, seed_version):
          chain_cids: ancestor checkpoint IDs from target's parent down to
                      the seed (or root), in newest-first order.
          seed_version: the channel blob version at the nearest ancestor
                        with has_snapshot=True, or None if pure delta.
        """
        parent_of: dict[str, str | None] = {}
        ver_of: dict[str, str | None] = {}
        snapshot_of: dict[str, bool] = {}
        for r in stage1_rows:
            cid = r["checkpoint_id"]
            parent_of[cid] = r["parent_checkpoint_id"]
            ver_of[cid] = r["ver"]
            snapshot_of[cid] = r["has_snapshot"]

        chain_cids: list[str] = []
        seed_version: str | None = None
        cur_cid: str | None = parent_of.get(target_id)
        while cur_cid is not None:
            chain_cids.append(cur_cid)
            if snapshot_of.get(cur_cid, False):
                seed_version = ver_of.get(cur_cid)
                break
            cur_cid = parent_of.get(cur_cid)
        return chain_cids, seed_version

    def _build_delta_channel_writes_history_two_stage(
        self,
        *,
        channel: str,
        chain_cids: list[str],
        seed_version: str | None,
        stage2_rows: Sequence[_DeltaCombinedRow],
    ) -> _ChannelWritesHistory:
        """Reconstruct delta channel history from two-stage query results.

        chain_cids are in newest-first order (target's parent first).
        stage2_rows contain only writes for chain_cids and the single
        seed blob at seed_version.
        """
        writes_by_cid: dict[str, list[tuple[str, bytes, str, int]]] = {}
        seed_blob: tuple[str, bytes] | None = None

        for r in stage2_rows:
            kind = r["_kind"]
            if kind == "w":
                cid = cast(str, r["checkpoint_id"])
                writes_by_cid.setdefault(cid, []).append(
                    cast(
                        "tuple[str, bytes, str, int]",
                        (r["type"], r["blob"], r["task_id"], r["idx"]),
                    )
                )
            else:  # kind == "b"
                seed_blob = cast("tuple[str, bytes]", (r["type"], r["blob"]))

        for ws in writes_by_cid.values():
            ws.sort(key=lambda w: (w[2], w[3]), reverse=True)

        if not chain_cids:
            return _ChannelWritesHistory(seed=DELTA_SENTINEL, writes=[])

        collected: list[PendingWrite] = []
        for cid in chain_cids:
            for type_tag, write_blob, task_id, _idx in writes_by_cid.get(cid, []):
                val = self.serde.loads_typed((type_tag, write_blob))
                collected.append((task_id, channel, val))

        seed: Any = DELTA_SENTINEL
        if seed_blob is not None and seed_blob[0] != "empty":
            seed = self.serde.loads_typed(seed_blob)

        collected.reverse()
        return _ChannelWritesHistory(seed=seed, writes=collected)

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
