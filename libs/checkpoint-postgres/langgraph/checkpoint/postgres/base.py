from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from importlib.metadata import version as get_version
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    DELTA_SENTINEL,
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    DeltaChannelWrites,
    _overwrite_types,
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
        self,
        blob_values: Any,
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        result: dict[str, Any] = {}
        for k, t, v in blob_values:
            type_tag = t.decode()
            if type_tag != "empty":
                result[k.decode()] = self.serde.loads_typed((type_tag, v))
        return result

    def _resolve_delta_channels(
        self,
        config: RunnableConfig,
        channel_values: dict[str, Any],
        cur: Any,
    ) -> None:
        delta_channels = [ch for ch, v in channel_values.items() if v is DELTA_SENTINEL]
        if not delta_channels:
            return
        reconstructed = self._reconstruct_delta_channels_cur(
            thread_id=config["configurable"]["thread_id"],
            checkpoint_ns=config["configurable"].get("checkpoint_ns", ""),
            checkpoint_id=config["configurable"]["checkpoint_id"],
            channels=delta_channels,
            cur=cur,
        )
        for ch, writes in reconstructed.items():
            channel_values[ch] = writes

    def _reconstruct_delta_channels_cur(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        channels: Sequence[str],
        cur: Any,
    ) -> dict[str, DeltaChannelWrites]:
        """Reconstruct `DeltaChannelWrites` for every `channel` in `channels`.

        A single recursive CTE enumerates on-path ancestors (never siblings),
        left-joined once against `checkpoint_writes` and once against
        `checkpoint_blobs`. One roundtrip covers every delta channel in the
        `get_tuple` — avoids the N-channels × 3-queries blowup and fits the
        intent of the schema (`parent_checkpoint_id` is an indexed ancestor
        pointer; postgres's planner handles this CTE shape well for the
        ancestor depths seen in practice).

        The walk newest→oldest stops per channel at the first terminator:

          * a user-emitted `Overwrite` in `checkpoint_writes` — replaces
            prior history;
          * a non-sentinel blob in `checkpoint_blobs` — a pre-delta snapshot;
            bound as `DeltaChannelWrites.seed` so replay starts from it.

        Writes stored AT `checkpoint_id` are pending for the next step and
        excluded; the recursion starts from the target's parent.
        """
        if not channels:
            return {}

        overwrite_types = _overwrite_types()
        channels_list = list(channels)

        # depth=0 → target's parent (we never read writes or blob for the
        # target itself). A NULL parent stops the recursion.
        cur.execute(
            """
            WITH RECURSIVE ancestors(cid, parent, depth) AS (
                SELECT parent_checkpoint_id, NULL::text, 0
                FROM checkpoints
                WHERE thread_id = %s AND checkpoint_ns = %s
                  AND checkpoint_id = %s AND parent_checkpoint_id IS NOT NULL
              UNION ALL
                SELECT c.parent_checkpoint_id, a.cid, a.depth + 1
                FROM checkpoints c
                JOIN ancestors a ON c.checkpoint_id = a.cid
                WHERE c.thread_id = %s AND c.checkpoint_ns = %s
                  AND c.parent_checkpoint_id IS NOT NULL
            ),
            walk AS (
                -- Each ancestor plus the channel_versions mapping for blob join.
                SELECT a.cid, a.depth, c.checkpoint
                FROM ancestors a
                JOIN checkpoints c
                  ON c.thread_id = %s AND c.checkpoint_ns = %s
                 AND c.checkpoint_id = a.cid
            )
            SELECT w.cid, w.depth,
                   cw.channel AS write_channel, cw.type AS write_type,
                   cw.blob AS write_blob, cw.task_id, cw.idx,
                   bl.channel AS blob_channel, bl.type AS blob_type,
                   bl.blob AS blob_blob
            FROM walk w
            LEFT JOIN checkpoint_writes cw
              ON cw.thread_id = %s AND cw.checkpoint_ns = %s
             AND cw.checkpoint_id = w.cid AND cw.channel = ANY(%s)
            LEFT JOIN checkpoint_blobs bl
              ON bl.thread_id = %s AND bl.checkpoint_ns = %s
             AND bl.channel = ANY(%s)
             AND bl.version = (w.checkpoint->'channel_versions'->>bl.channel)
            ORDER BY w.depth ASC, cw.task_id DESC, cw.idx DESC
            """,
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,  # anchor
                thread_id,
                checkpoint_ns,  # recursion
                thread_id,
                checkpoint_ns,  # walk join
                thread_id,
                checkpoint_ns,
                channels_list,  # writes join
                thread_id,
                checkpoint_ns,
                channels_list,  # blobs join
            ),
        )

        # Group incoming rows by `cid` so we can, per ancestor, decide whether
        # to terminate (pre-delta blob) BEFORE processing writes at that
        # ancestor. Rows within a cid arrive in (task_id DESC, idx DESC)
        # order; we preserve that when building per-cid writes lists.
        rows_by_cid: dict[str, dict[str, Any]] = {}
        cid_order: list[str] = []  # newest → oldest
        seen_blob: set[tuple[str, str]] = set()
        seen_write: set[tuple[str, str, str, int]] = set()
        for row in cur.fetchall():
            cid = row["cid"]
            if cid not in rows_by_cid:
                rows_by_cid[cid] = {"writes_per_channel": {}, "blob_per_channel": {}}
                cid_order.append(cid)
            # Writes: dedupe via (cid, channel, task_id, idx).
            ch_w = row["write_channel"]
            if ch_w is not None:
                key = (cid, ch_w, row["task_id"], row["idx"])
                if key not in seen_write:
                    seen_write.add(key)
                    rows_by_cid[cid]["writes_per_channel"].setdefault(ch_w, []).append(
                        (row["write_type"], row["write_blob"])
                    )
            # Blobs: dedupe via (cid, channel).
            ch_b = row["blob_channel"]
            if ch_b is not None and (cid, ch_b) not in seen_blob:
                seen_blob.add((cid, ch_b))
                rows_by_cid[cid]["blob_per_channel"][ch_b] = (
                    row["blob_type"],
                    row["blob_blob"],
                )

        # Per-channel state. `collected[ch]` is newest→oldest during the walk.
        collected: dict[str, list[Any]] = {ch: [] for ch in channels_list}
        done: set[str] = set()
        seeds: dict[str, Any] = {}

        for cid in cid_order:  # newest → oldest
            bucket = rows_by_cid[cid]
            # At each ancestor, check the blob FIRST — a pre-delta blob
            # subsumes any writes stored under the same checkpoint, so we
            # must not fold those writes in before terminating.
            for ch in list(channels_list):
                if ch in done:
                    continue
                blob = bucket["blob_per_channel"].get(ch)
                if blob is None or blob[0] == "empty":
                    continue
                blob_value = self.serde.loads_typed(blob)
                if blob_value is DELTA_SENTINEL:
                    continue
                seeds[ch] = blob_value
                done.add(ch)
            # Then process per-channel writes for any channel still live.
            for ch in list(channels_list):
                if ch in done:
                    continue
                for type_tag, blob in bucket["writes_per_channel"].get(ch, []):
                    val = self.serde.loads_typed((type_tag, blob))
                    collected[ch].append(val)
                    if isinstance(val, overwrite_types):
                        done.add(ch)
                        break
            if len(done) == len(channels_list):
                break

        result: dict[str, DeltaChannelWrites] = {}
        for ch in channels_list:
            ch_writes = collected[ch]
            ch_writes.reverse()  # oldest → newest
            if ch in seeds:
                result[ch] = DeltaChannelWrites(writes=ch_writes, seed=seeds[ch])
            else:
                result[ch] = DeltaChannelWrites(writes=ch_writes)
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
