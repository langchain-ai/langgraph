from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

import asyncpg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import TASKS

from langgraph.checkpoint.postgres_asyncpg import _ainternal
from langgraph.checkpoint.postgres_asyncpg.asyncpg_sql_scripts import (
    INSERT_CHECKPOINT_WRITES_SQL,
    MIGRATIONS,
    SELECT_PENDING_SENDS_SQL,
    SELECT_SQL,
    UPSERT_CHECKPOINT_BLOBS_SQL,
    UPSERT_CHECKPOINT_WRITES_SQL,
    UPSERT_CHECKPOINTS_SQL,
)

MetadataInput = dict[str, Any] | None


class AsyncPgPostgresSaver(BaseCheckpointSaver[str]):
    """Asynchronous checkpointer that stores checkpoints in a Postgres database."""

    lock: asyncio.Lock

    def __init__(
        self,
        conn: _ainternal.Conn,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

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

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: SerializerProtocol | None = None,
    ) -> AsyncIterator[AsyncPgPostgresSaver]:
        """Create a new AsyncPostgresSaver instance from a connection string.

        Args:
            conn_string: The Postgres connection info string.

        Returns:
            AsyncPostgresSaver: A new AsyncPostgresSaver instance.
        """
        conn = await asyncpg.connect(conn_string)
        try:
            yield cls(conn=conn, serde=serde)
        finally:
            await conn.close()

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        async with self._cursor(transaction=False) as cur:
            await cur.execute(MIGRATIONS[0])
            row = await cur.fetchrow(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
            )
            if row is None:
                version = -1
            else:
                version = row["v"]
            for v, migration in zip(
                range(version + 1, len(MIGRATIONS)),
                MIGRATIONS[version + 1 :],
                strict=False,
            ):
                await cur.execute(migration)
                await cur.execute(
                    "INSERT INTO checkpoint_migrations (v) VALUES ($1)", v
                )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the Postgres database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            An asynchronous iterator of matching checkpoint tuples.
        """
        where, args = self._search_where(config, filter, before)
        query = SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"

        async with self._cursor() as cur:
            values = await cur.fetch(query, *args)
            if not values:
                return
            # migrate pending sends if necessary
            for i, value in enumerate(values):
                values[i] = dict(value)
                values[i]["checkpoint"] = json.loads(values[i]["checkpoint"])
                values[i]["metadata"] = json.loads(values[i]["metadata"])
            if to_migrate := [
                v
                for v in values
                if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]
            ]:
                parent_ids = [v["parent_checkpoint_id"] for v in to_migrate]
                sends_data = await cur.fetch(
                    SELECT_PENDING_SENDS_SQL,
                    values[0]["thread_id"],
                    parent_ids,
                )
                grouped_by_parent = defaultdict(list)
                for value in to_migrate:
                    grouped_by_parent[value["parent_checkpoint_id"]].append(value)

                for sends in sends_data:
                    for value in grouped_by_parent[sends["checkpoint_id"]]:
                        if value["channel_values"] is None:
                            value["channel_values"] = []
                        self._migrate_pending_sends(
                            sends["sends"],
                            value["checkpoint"],
                            value["channel_values"],
                        )
            for value in values:
                yield await self._load_checkpoint_tuple(value)

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
        param_count = 0

        # construct predicate for config filter
        if config:
            param_count += 1
            wheres.append(f"thread_id = ${param_count}")
            param_values.append(config["configurable"]["thread_id"])

            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                param_count += 1
                wheres.append(f"checkpoint_ns = ${param_count}")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                param_count += 1
                wheres.append(f"checkpoint_id = ${param_count}")
                param_values.append(checkpoint_id)

        # construct predicate for metadata filter
        if filter:
            param_count += 1
            wheres.append(f"metadata @> ${param_count}")
            param_values.append(json.dumps(filter))

        # construct predicate for `before`
        if before is not None:
            param_count += 1
            wheres.append(f"checkpoint_id < ${param_count}")
            param_values.append(get_checkpoint_id(before))

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the Postgres database based on the
        provided config. If the config contains a `checkpoint_id` key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if checkpoint_id:
            where = "WHERE thread_id = $1 AND checkpoint_ns = $2 AND checkpoint_id = $3"
            args = (thread_id, checkpoint_ns, checkpoint_id)
        else:
            where = "WHERE thread_id = $1 AND checkpoint_ns = $2 ORDER BY checkpoint_id DESC LIMIT 1"
            args = (thread_id, checkpoint_ns)

        async with self._cursor() as cur:
            value = await cur.fetchrow(SELECT_SQL + where, *args)
            if value is None:
                return None

            value = dict(value)
            value["checkpoint"] = json.loads(value["checkpoint"])
            value["metadata"] = json.loads(value["metadata"])

            # migrate pending sends if necessary
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                sends_data = await cur.fetchrow(
                    SELECT_PENDING_SENDS_SQL,
                    thread_id,
                    [value["parent_checkpoint_id"]],
                )
                if sends_data:
                    if value["channel_values"] is None:
                        value["channel_values"] = []
                    self._migrate_pending_sends(
                        sends_data["sends"],
                        value["checkpoint"],
                        value["channel_values"],
                    )
            return await self._load_checkpoint_tuple(value)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously."""
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop("checkpoint_id", None)

        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        # inline primitive values in checkpoint table
        # others are stored in blobs table
        blob_values = {}
        for k, v in checkpoint["channel_values"].items():
            if v is None or isinstance(v, (str, int, float, bool)):
                pass
            else:
                blob_values[k] = copy["channel_values"].pop(k)

        async with self._cursor() as cur:
            if blob_versions := {
                k: v for k, v in new_versions.items() if k in blob_values
            }:
                blob_data = await asyncio.to_thread(
                    self._dump_blobs,
                    thread_id,
                    checkpoint_ns,
                    blob_values,
                    blob_versions,
                )
                await cur.executemany(UPSERT_CHECKPOINT_BLOBS_SQL, blob_data)

            checkpoint_json = json.dumps(copy)
            metadata_json = json.dumps(
                get_serializable_checkpoint_metadata(config, metadata)
            )

            await cur.execute(
                UPSERT_CHECKPOINTS_SQL,
                thread_id,
                checkpoint_ns,
                checkpoint["id"],
                checkpoint_id,
                checkpoint_json,
                metadata_json,
            )
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
        """
        query = (
            UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else INSERT_CHECKPOINT_WRITES_SQL
        )
        params = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        async with self._cursor() as conn:
            await conn.executemany(query, params)

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        async with self._cursor() as conn:
            await conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = $1",
                str(thread_id),
            )
            await conn.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = $1",
                str(thread_id),
            )
            await conn.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = $1",
                str(thread_id),
            )

    @asynccontextmanager
    async def _cursor(
        self, *, transaction: bool = True
    ) -> AsyncIterator[asyncpg.Connection]:
        """Create a database connection/cursor as a context manager.

        Args:
            transaction: whether to wrap operations in a transaction
        """
        async with self.lock, _ainternal.get_connection(self.conn) as conn:
            if transaction:
                async with conn.transaction():
                    yield conn
            else:
                yield conn

    async def _load_checkpoint_tuple(
        self, value: dict | asyncpg.Record
    ) -> CheckpointTuple:
        """
        Convert a database row into a CheckpointTuple object.

        Args:
            value: A row from the database containing checkpoint data.

        Returns:
            CheckpointTuple: A structured representation of the checkpoint,
            including its configuration, metadata, parent checkpoint (if any),
            and pending writes.
        """
        checkpoint_data = value["checkpoint"]

        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": value["thread_id"],
                    "checkpoint_ns": value["checkpoint_ns"],
                    "checkpoint_id": value["checkpoint_id"],
                }
            },
            {
                **checkpoint_data,
                "channel_values": {
                    **(checkpoint_data.get("channel_values") or {}),
                    **self._load_blobs(value["channel_values"]),
                },
            },
            value["metadata"],
            (
                {
                    "configurable": {
                        "thread_id": value["thread_id"],
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["parent_checkpoint_id"],
                    }
                }
                if value["parent_checkpoint_id"]
                else None
            ),
            await asyncio.to_thread(self._load_writes, value["pending_writes"]),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the Postgres database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            An iterator of matching checkpoint tuples.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncPostgresSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `checkpointer.alist(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),  # type: ignore[arg-type]  # noqa: F821
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the Postgres database based on the
        provided config. If the config contains a `checkpoint_id` key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncPostgresSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the Postgres database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncPostgresSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()


__all__ = ["AsyncPgPostgresSaver"]
