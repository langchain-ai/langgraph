"""Asynchronous Postgres checkpoint saver for LangGraph."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
    _DeltaSnapshot,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS
from psycopg import Capabilities, Connection, Cursor, Pipeline
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langgraph.checkpoint.postgres import _internal
from langgraph.checkpoint.postgres.base import (
    _DELTA_PAGE_SIZE,
    BasePostgresSaver,
    _build_delta_stage1_sql,
    _build_delta_stage2_sql,
    _DeltaStage2Row,
)
from langgraph.checkpoint.postgres.shallow import AsyncShallowPostgresSaver

from langgraph.checkpoint.postgres import _internal

Conn = _internal.Conn  # For backward compatibility


__all__ = ["AsyncPostgresSaver", "AsyncShallowPostgresSaver", "Conn"]


class AsyncPostgresSaver(BasePostgresSaver):
    """Asynchronous checkpointer that stores checkpoints in a Postgres database."""

    def __init__(
        self,
        conn: _internal.Conn,
        pipe: Pipeline | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        if isinstance(conn, ConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single Connection, not ConnectionPool."
            )

        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.supports_pipeline = Capabilities().has_pipeline()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, conn_string: str, *, pipeline: bool = False
    ) -> AsyncIterator[AsyncPostgresSaver]:
        """Create a new AsyncPostgresSaver instance from a connection string.

        Args:
            conn_string: The Postgres connection info string.
            pipeline: whether to use Pipeline

        Returns:
            AsyncPostgresSaver: A new AsyncPostgresSaver instance.
        """
        async with Connection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            if pipeline:
                async with conn.pipeline() as pipe:
                    yield cls(conn, pipe)
            else:
                yield cls(conn)

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        async with self._cursor() as cur:
            await cur.execute(self.MIGRATIONS[0])
            results = await cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
            )
            row = await results.fetchone()
            if row is None:
                version = -1
            else:
                version = row["v"]
            for i in range(version + 1, len(self.MIGRATIONS)):
                await cur.execute(self.MIGRATIONS[i])

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

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
            if isinstance(v, _DeltaSnapshot):
                blob_values[k] = copy["channel_values"].pop(k)
                copy["channel_values"][k] = True
            elif v is None or isinstance(v, (str, int, float, bool)):
                pass
            else:
                blob_values[k] = copy["channel_values"].pop(k)

        async with self._cursor(pipeline=True) as cur:
            if blob_versions := {
                k: v for k, v in new_versions.items() if k in blob_values
            }:
                await cur.executemany(
                    self.UPSERT_CHECKPOINT_BLOBS_SQL,
                    await asyncio.to_thread(
                        self._dump_blobs,
                        thread_id,
                        checkpoint_ns,
                        blob_values,
                        blob_versions,
                    ),
                )
            await cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    Jsonb(checkpoint),
                    Jsonb(metadata),
                    Jsonb(new_versions),
                ),
            )
            if self.pipe:
                await self.pipe.sync()

        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save intermediate writes associated with a checkpoint to the Postgres database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        async with self._cursor(pipeline=True) as cur:
            await cur.executemany(
                query,
                await asyncio.to_thread(
                    self._dump_writes,
                    config["configurable"]["thread_id"],
                    config["configurable"]["checkpoint_ns"],
                    config["configurable"]["checkpoint_id"],
                    task_id,
                    task_path,
                    writes,
                ),
            )

    async def aget_tuple(
        self,
        config: RunnableConfig,
    ) -> CheckpointTuple | None:
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
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")
        checkpoint_id = configurable.pop("checkpoint_id", None)

        if checkpoint_id:
            args = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        async with self._cursor() as cur:
            await cur.execute(
                self.SELECT_SQL + where,
                args,
                binary=True,
            )
            value = await cur.fetchone()
            if value is None:
                return None

            # migrate pending sends if necessary
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (thread_id, [value["parent_checkpoint_id"]]),
                )
                if sends := await cur.fetchone():
                    if value["channel_values"] is None:
                        value["channel_values"] = []
                    self._migrate_pending_sends(
                        sends["sends"],
                        value["checkpoint"],
                        value["channel_values"],
                    )

            return await self._load_checkpoint_tuple(value)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the Postgres database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            An async iterator of matching checkpoint tuples.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncPostgresSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.alist(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield await anext(aiter_)
            except StopAsyncIteration:
                break

    async def aget_tuple(
        self,
        config: RunnableConfig,
    ) -> CheckpointTuple | None:
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
        return await self.aget_tuple(config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

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
        return await self.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save intermediate writes associated with a checkpoint to the Postgres database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        return await self.aput_writes(config, writes, task_id, task_path)

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

    async def adelete_thread(self, thread_id: str) -> None:
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
        async with self._cursor(pipeline=True) as cur:
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (str(thread_id),),
            )
            await cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                (str(thread_id),),
            )
            await cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (str(thread_id),),
            )

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        """Prune checkpoints for given threads.

        Args:
            thread_ids: The thread IDs to prune.
            strategy: "keep_latest" keeps only the latest checkpoint per
                thread+namespace. "delete" removes everything.

        !!! warning "DeltaChannel"
            Threads using DeltaChannel are not pruned to avoid corrupting
            delta channel history. If any requested thread contains DeltaChannel
            state, the operation aborts and raises an error.

        Raises:
            ValueError: If an invalid strategy is provided.
            RuntimeError: If any thread contains DeltaChannel state and strategy
                is "keep_latest" (which cannot safely preserve DeltaChannel history).
        """
        if not thread_ids:
            return

        if strategy not in ("delete", "keep_latest"):
            raise ValueError(f"Invalid pruning strategy: {strategy}")

        # First, check for DeltaChannel state in any of the threads
        # if using keep_latest strategy
        if strategy == "keep_latest":
            async with self._cursor() as cur:
                for thread_id in thread_ids:
                    await cur.execute(
                        """
                        SELECT 1 FROM checkpoints
                        WHERE thread_id = %s
                        AND checkpoint_id IN (
                            SELECT checkpoint_id FROM checkpoints
                            WHERE thread_id = %s
                            AND channel_values @> '{"_delta_snapshot": true}'
                        )
                        LIMIT 1
                        """,
                        (str(thread_id), str(thread_id)),
                    )
                    if await cur.fetchone():
                        raise RuntimeError(
                            f"Thread {thread_id} contains DeltaChannel state. "
                            "keep_latest pruning is not supported for DeltaChannel threads. "
                            "Use 'delete' strategy or implement DeltaChannel-aware pruning."
                        )

        async with self._cursor(pipeline=True) as cur:
            if strategy == "delete":
                # Delete all checkpoints, writes, and blobs for the threads
                await cur.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ANY(%s)",
                    (list(thread_ids),),
                )
                await cur.execute(
                    "DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s)",
                    (list(thread_ids),),
                )
                await cur.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = ANY(%s)",
                    (list(thread_ids),),
                )
            elif strategy == "keep_latest":
                # Delete non-latest checkpoints, preserving writes and blobs for latest
                for thread_id in thread_ids:
                    # Get the latest checkpoint_id for each namespace
                    await cur.execute(
                        """
                        DELETE FROM checkpoints
                        WHERE thread_id = %s
                        AND (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
                            SELECT DISTINCT ON (thread_id, checkpoint_ns)
                                thread_id, checkpoint_ns, checkpoint_id
                            FROM checkpoints
                            WHERE thread_id = %s
                            ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC
                        )
                        """,
                        (str(thread_id), str(thread_id)),
                    )
                    # Delete writes for removed checkpoints
                    await cur.execute(
                        """
                        DELETE FROM checkpoint_writes
                        WHERE thread_id = %s
                        AND (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
                            SELECT thread_id, checkpoint_ns, checkpoint_id
                            FROM checkpoints
                            WHERE thread_id = %s
                        )
                        """,
                        (str(thread_id), str(thread_id)),
                    )
                    # Clean up orphaned blobs
                    await cur.execute(
                        """
                        DELETE FROM checkpoint_blobs
                        WHERE thread_id = %s
                        AND NOT EXISTS (
                            SELECT 1 FROM checkpoints c
                            WHERE c.thread_id = checkpoint_blobs.thread_id
                            AND c.checkpoint_ns = checkpoint_blobs.checkpoint_ns
                        )
                        """,
                        (str(thread_id),),
                    )

__all__ = ["AsyncPostgresSaver", "AsyncShallowPostgresSaver", "Conn"]