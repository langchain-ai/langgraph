from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from importlib.metadata import version as get_version
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS
from psycopg import sql
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


class BasePostgresSaver(BaseCheckpointSaver[str]):
    MIGRATIONS: list[sql.Composed]
    SELECT_SQL: sql.Composed
    SELECT_PENDING_SENDS_SQL: sql.Composed
    UPSERT_CHECKPOINT_BLOBS_SQL: sql.Composed
    UPSERT_CHECKPOINTS_SQL: sql.Composed
    UPSERT_CHECKPOINT_WRITES_SQL: sql.Composed
    INSERT_CHECKPOINT_WRITES_SQL: sql.Composed

    supports_pipeline: bool

    def _setup_queries(self, schema: str) -> None:
        self.MIGRATIONS = [
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (\n    v INTEGER PRIMARY KEY\n);"
            ).format(sql.Identifier(schema, "checkpoint_migrations")),
            sql.SQL("""CREATE TABLE IF NOT EXISTS {} (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);""").format(sql.Identifier(schema, "checkpoints")),
            sql.SQL("""CREATE TABLE IF NOT EXISTS {} (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);""").format(sql.Identifier(schema, "checkpoint_blobs")),
            sql.SQL("""CREATE TABLE IF NOT EXISTS {} (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);""").format(sql.Identifier(schema, "checkpoint_writes"), TASKS=sql.SQL(TASKS)),
            sql.SQL("ALTER TABLE {} ALTER COLUMN blob DROP not null;").format(
                sql.Identifier(schema, "checkpoint_blobs")
            ),
            sql.SQL("SELECT 1;"),
            sql.SQL(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoints_thread_id_idx ON {}(thread_id);"
            ).format(sql.Identifier(schema, "checkpoints")),
            sql.SQL(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_blobs_thread_id_idx ON {}(thread_id);"
            ).format(sql.Identifier(schema, "checkpoint_blobs")),
            sql.SQL(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_writes_thread_id_idx ON {}(thread_id);"
            ).format(sql.Identifier(schema, "checkpoint_writes")),
            sql.SQL(
                "ALTER TABLE {} ADD COLUMN IF NOT EXISTS task_path TEXT NOT NULL DEFAULT '';"
            ).format(sql.Identifier(schema, "checkpoint_writes")),
        ]

        self.SELECT_SQL = sql.SQL("""
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
        inner join {checkpoint_blobs} bl
            on bl.thread_id = {checkpoints}.thread_id
            and bl.checkpoint_ns = {checkpoints}.checkpoint_ns
            and bl.channel = jsonb_each_text.key
            and bl.version = jsonb_each_text.value
    ) as channel_values,
    (
        select
        array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] order by cw.task_id, cw.idx)
        from {checkpoint_writes} cw
        where cw.thread_id = {checkpoints}.thread_id
            and cw.checkpoint_ns = {checkpoints}.checkpoint_ns
            and cw.checkpoint_id = {checkpoints}.checkpoint_id
    ) as pending_writes
from {checkpoints} """).format(
            checkpoint_blobs=sql.Identifier(schema, "checkpoint_blobs"),
            checkpoint_writes=sql.Identifier(schema, "checkpoint_writes"),
            checkpoints=sql.Identifier(schema, "checkpoints"),
        )

        self.SELECT_PENDING_SENDS_SQL = sql.SQL("""
select
    checkpoint_id,
    array_agg(array[type::bytea, blob] order by task_path, task_id, idx) as sends
from {}
where thread_id = %s
    and checkpoint_id = any(%s)
    and channel = '{TASKS}'
group by checkpoint_id
""").format(sql.Identifier(schema, "checkpoint_writes"), TASKS=sql.SQL(TASKS))

        self.UPSERT_CHECKPOINT_BLOBS_SQL = sql.SQL("""
    INSERT INTO {} (thread_id, checkpoint_ns, channel, version, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING
""").format(sql.Identifier(schema, "checkpoint_blobs"))

        self.UPSERT_CHECKPOINTS_SQL = sql.SQL("""
    INSERT INTO {} (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
    DO UPDATE SET
        checkpoint = EXCLUDED.checkpoint,
        metadata = EXCLUDED.metadata;
""").format(sql.Identifier(schema, "checkpoints"))

        self.UPSERT_CHECKPOINT_WRITES_SQL = sql.SQL("""
    INSERT INTO {} (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
        channel = EXCLUDED.channel,
        type = EXCLUDED.type,
        blob = EXCLUDED.blob;
""").format(sql.Identifier(schema, "checkpoint_writes"), TASKS=sql.SQL(TASKS))

        self.INSERT_CHECKPOINT_WRITES_SQL = sql.SQL("""
    INSERT INTO {} (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
""").format(sql.Identifier(schema, "checkpoint_writes"), TASKS=sql.SQL(TASKS))

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
