from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from collections.abc import AsyncIterator, Coroutine, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncTransaction

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import TASKS

from .base import BaseMemgraphSaver

if sys.version_info >= (3, 10):
    from builtins import anext
else:
    _sentinel = object()

    async def anext(iterator, default=_sentinel):
        """A backport of anext() for Python < 3.10."""
        try:
            return await iterator.__anext__()
        except StopAsyncIteration:
            if default is _sentinel:
                raise
            return default


class AsyncMemgraphSaver(BaseMemgraphSaver):
    """Asynchronous checkpointer that stores checkpoints in a Memgraph database.

    This class provides an asynchronous interface for persisting checkpoints to a Memgraph
    database. It leverages the `neo4j.AsyncDriver` for non-blocking database
    communication and is designed for use in asynchronous applications where
    blocking I/O operations are undesirable.

    The class manages an asyncio event loop and uses a lock to ensure that
    concurrent operations on the database are handled safely.

    Attributes:
        driver (AsyncDriver): An instance of `neo4j.AsyncDriver` used to interact
            with the Memgraph database.
        lock (asyncio.Lock): An asyncio lock to prevent race conditions during
            database writes.
        loop (asyncio.AbstractEventLoop): The asyncio event loop used for scheduling
            the asynchronous database operations.
    """

    driver: AsyncDriver
    lock: asyncio.Lock
    loop: asyncio.AbstractEventLoop

    def __init__(
        self,
        driver: AsyncDriver,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """
        Initialize the async Memgraph saver.

        Args:
            driver: The neo4j.AsyncDriver instance to connect to the database.
            serde: The serializer to use for checkpoint data. Defaults to a JSON serializer.
        """
        super().__init__(serde=serde)
        self.driver = driver
        self.lock = asyncio.Lock()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, conn_string: str, *, serde: SerializerProtocol | None = None
    ) -> AsyncIterator[AsyncMemgraphSaver]:
        """
        Create a new AsyncMemgraphSaver instance from a connection string.

        Args:
            conn_string: The Memgraph connection URI (e.g., "bolt://localhost:7687").
            serde: The serializer to use.

        Yields:
            An AsyncMemgraphSaver instance.
        """
        driver = AsyncGraphDatabase.driver(conn_string)
        try:
            yield cls(driver=driver, serde=serde)
        finally:
            await driver.close()

    async def setup(self) -> None:
        """
        Set up the checkpoint database asynchronously.

        This method creates the necessary constraints and indexes in Memgraph if they
        don't already exist and runs any pending database migrations.
        """
        async with self.driver.session() as session:
            try:
                result = await session.run(
                    "MATCH (m:Migration) RETURN m.v AS v ORDER BY m.v DESC LIMIT 1"
                )
                version_record = await result.single()
                version = version_record["v"] if version_record else -1
            except Exception:
                version = -1
            for v, migration in enumerate(self.MIGRATIONS):
                if v > version:
                    if not migration.startswith("//"):
                        await session.run(migration)
                    await session.run("MERGE (m:Migration {v: $v})", v=v)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """
        List checkpoints from the database asynchronously.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified one are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            An async iterator of matching checkpoint tuples.
        """
        where_clause, params = self._search_where_and_params(config, filter, before)
        query = self.SELECT_CYPHER.replace(
            "MATCH (c:Checkpoint)", f"MATCH (c:Checkpoint) {where_clause}"
        )
        query += " ORDER BY c.checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        async with self._session() as tx:
            result = await tx.run(query, params)
            records = [dict(record) async for record in result]
            if not records:
                return
            to_migrate = [
                r
                for r in records
                if r["checkpoint"].get("v", 0) < 4 and r["parent_checkpoint_id"]
            ]
            if to_migrate:
                thread_id = records[0]["thread_id"]
                parent_ids = list({r["parent_checkpoint_id"] for r in to_migrate})
                sends_result = await tx.run(
                    self.SELECT_PENDING_SENDS_CYPHER,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ids": parent_ids,
                        "tasks_channel": TASKS,
                    },
                )
                sends_records = [dict(record) async for record in sends_result]
                grouped_by_parent = defaultdict(list)
                for record in to_migrate:
                    grouped_by_parent[record["parent_checkpoint_id"]].append(record)
                for sends_record in sends_records:
                    parent_id = sends_record["checkpoint_id"]
                    for record in grouped_by_parent[parent_id]:
                        if record.get("channel_values") is None:
                            record["channel_values"] = []
                        self._migrate_pending_sends(
                            sends_record["sends"],
                            record["checkpoint"],
                            record["channel_values"],
                        )
            for record in records:
                yield await self._load_checkpoint_tuple(record)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """
        Get a checkpoint tuple from the database asynchronously.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            The checkpoint tuple, or None if not found.
        """
        where_clause, params = self._search_where_and_params(config, None, None)
        query = self.SELECT_CYPHER.replace(
            "MATCH (c:Checkpoint)", f"MATCH (c:Checkpoint) {where_clause}"
        )
        if "checkpoint_id" not in config["configurable"]:
            query += " ORDER BY c.checkpoint_id DESC LIMIT 1"
        async with self._session() as tx:
            result = await tx.run(query, params)
            record = await result.single()
            if record is None:
                return None
            record_dict = dict(record)
            if (
                record_dict["checkpoint"].get("v", 0) < 4
                and record_dict["parent_checkpoint_id"]
            ):
                thread_id = config["configurable"]["thread_id"]
                sends_result = await tx.run(
                    self.SELECT_PENDING_SENDS_CYPHER,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ids": [record_dict["parent_checkpoint_id"]],
                        "tasks_channel": TASKS,
                    },
                )
                sends_record = await sends_result.single()
                if sends_record:
                    if record_dict.get("channel_values") is None:
                        record_dict["channel_values"] = []
                    self._migrate_pending_sends(
                        sends_record["sends"],
                        record_dict["checkpoint"],
                        record_dict["channel_values"],
                    )
            return await self._load_checkpoint_tuple(record_dict)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Save a checkpoint to the database asynchronously.

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            The updated runnable config.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = get_checkpoint_id(config)
        checkpoint_blobs = await asyncio.to_thread(
            self._dump_blobs,
            thread_id,
            checkpoint_ns,
            checkpoint["channel_values"],
            new_versions,
        )
        async with self._session() as tx:
            if checkpoint_blobs:
                await tx.run(
                    self.UPSERT_CHECKPOINT_BLOBS_CYPHER,
                    blobs=checkpoint_blobs,
                )
            await tx.run(
                self.UPSERT_CHECKPOINTS_CYPHER,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint["id"],
                parent_checkpoint_id=parent_checkpoint_id,
                checkpoint=checkpoint,
                metadata=get_checkpoint_metadata(config, metadata),
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """
        Store intermediate writes linked to a checkpoint asynchronously.

        Args:
            config: Configuration of the related checkpoint.
            writes: A list of writes to store.
            task_id: The ID of the task creating the writes.
            task_path: The path of the task.
        """
        checkpoint_writes = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"].get("checkpoint_ns", ""),
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        if not checkpoint_writes:
            return
        upsert_mode = all(w[0] in WRITES_IDX_MAP for w in writes)
        query_template = (
            self.UPSERT_CHECKPOINT_WRITES_CYPHER
            if upsert_mode
            else self.INSERT_CHECKPOINT_WRITES_CYPHER
        )
        async with self._session() as tx:
            await tx.run(query_template, writes=checkpoint_writes)

    async def adelete_thread(self, thread_id: str) -> None:
        """
        Delete all data associated with a specific thread ID asynchronously.

        Args:
            thread_id: The ID of the thread to delete.
        """
        async with self._session() as tx:
            await tx.run(
                """
                MATCH (c:Checkpoint {thread_id: $thread_id})
                OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:Write)
                OPTIONAL MATCH (b:Blob {thread_id: $thread_id})
                DETACH DELETE c, w, b
                """,
                thread_id=thread_id,
            )

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncTransaction]:
        """A context manager for acquiring a Neo4j async session and transaction."""
        async with self.lock:
            session = self.driver.session()
            try:
                tx = await session.begin_transaction()
                async with tx:
                    yield tx
            finally:
                await session.close()

    async def _load_checkpoint_tuple(self, record: dict[str, Any]) -> CheckpointTuple:
        """
        Convert a database record into a CheckpointTuple asynchronously.

        Args:
            record: A dictionary-like object representing a database row.

        Returns:
            A structured CheckpointTuple.
        """
        channel_values, pending_writes = await asyncio.gather(
            asyncio.to_thread(self._load_blobs, record.get("channel_values") or []),
            asyncio.to_thread(self._load_writes, record.get("pending_writes") or []),
        )
        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": record["thread_id"],
                    "checkpoint_ns": record["checkpoint_ns"],
                    "checkpoint_id": record["checkpoint_id"],
                }
            },
            {
                **record["checkpoint"],
                "channel_values": channel_values,
            },
            record["metadata"],
            (
                {
                    "configurable": {
                        "thread_id": record["thread_id"],
                        "checkpoint_ns": record["checkpoint_ns"],
                        "checkpoint_id": record["parent_checkpoint_id"],
                    }
                }
                if record["parent_checkpoint_id"]
                else None
            ),
            pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread.

        Args:
            config: The config of the thread to list checkpoints for.
            filter: A filter to apply to the checkpoints.
            before: A checkpoint to list checkpoints before.
            limit: The maximum number of checkpoints to return.

        Returns:
            An iterator of checkpoint tuples.
        """
        if asyncio.get_running_loop() is self.loop:
            raise asyncio.InvalidStateError(
                "Sync `list` can't be called from the same event loop. Use `alist`."
            )
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    cast(Coroutine, anext(aiter_)), self.loop
                ).result()
            except StopAsyncIteration:
                break

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database.

        Args:
            config: The config of the checkpoint to get.

        Returns:
            The checkpoint tuple, or None if it doesn't exist.
        """
        if asyncio.get_running_loop() is self.loop:
            raise asyncio.InvalidStateError(
                "Sync `get_tuple` can't be called from the same event loop. Use `aget_tuple`."
            )
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
        """Put a checkpoint in the database.

        Args:
            config: The config of the checkpoint to put.
            checkpoint: The checkpoint to put.
            metadata: The metadata of the checkpoint.
            new_versions: The new versions of the channels.

        Returns:
            The config of the checkpoint that was put.
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
        """Put a list of writes in the database.

        Args:
            config: The config of the checkpoint to put.
            writes: A list of writes to put.
            task_id: The ID of the task that is performing the writes.
            task_path: The path of the task that is performing the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        """Delete a thread from the database.

        Args:
            thread_id: The ID of the thread to delete.
        """
        if asyncio.get_running_loop() is self.loop:
            raise asyncio.InvalidStateError(
                "Sync `delete_thread` can't be called from the same event loop. Use `adelete_thread`."
            )
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()


__all__ = ["AsyncMemgraphSaver"]
