import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterator, Optional, Sequence

from langchain_core.runnables import RunnableConfig

import duckdb
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.duckdb.base import BaseDuckDBSaver
from langgraph.checkpoint.serde.base import SerializerProtocol


class AsyncDuckDBSaver(BaseDuckDBSaver):
    lock: asyncio.Lock

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
    ) -> AsyncIterator["AsyncDuckDBSaver"]:
        """Create a new AsyncDuckDBSaver instance from a connection string.

        Args:
            conn_string (str): The DuckDB connection info string.

        Returns:
            AsyncDuckDBSaver: A new AsyncDuckDBSaver instance.
        """
        with duckdb.connect(conn_string) as conn:
            yield cls(conn)

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the DuckDB database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        async with self.lock:
            with self.conn.cursor() as cur:
                try:
                    await asyncio.to_thread(
                        cur.execute,
                        "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1",
                    )
                    row = await asyncio.to_thread(cur.fetchone)
                    if row is None:
                        version = -1
                    else:
                        version = row[0]
                except duckdb.CatalogException:
                    version = -1
                for v, migration in zip(
                    range(version + 1, len(self.MIGRATIONS)),
                    self.MIGRATIONS[version + 1 :],
                ):
                    await asyncio.to_thread(cur.execute, migration)
                    await asyncio.to_thread(
                        cur.execute,
                        "INSERT INTO checkpoint_migrations (v) VALUES (?)",
                        [v],
                    )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the DuckDB database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        # if we change this to use .stream() we need to make sure to close the cursor
        async with self._cursor() as cur:
            await asyncio.to_thread(cur.execute, query, args)
            results = await asyncio.to_thread(cur.fetchall)
            for value in results:
                (
                    thread_id,
                    checkpoint,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    metadata,
                    channel_values,
                    pending_writes,
                    pending_sends,
                ) = value
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint,
                        checkpoint,
                        channel_values,
                        pending_sends,
                    ),
                    self._load_metadata(metadata),
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        }
                        if parent_checkpoint_id
                        else None
                    ),
                    await asyncio.to_thread(self._load_writes, pending_writes),
                )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the DuckDBdatabase based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1"

        async with self._cursor() as cur:
            await asyncio.to_thread(
                cur.execute,
                self.SELECT_SQL + where,
                args,
            )

            value = await asyncio.to_thread(cur.fetchone)
            if value:
                (
                    thread_id,
                    checkpoint,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    metadata,
                    channel_values,
                    pending_writes,
                    pending_sends,
                ) = value
                return CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint,
                        checkpoint,
                        channel_values,
                        pending_sends,
                    ),
                    self._load_metadata(metadata),
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        }
                        if parent_checkpoint_id
                        else None
                    ),
                    await asyncio.to_thread(self._load_writes, pending_writes),
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the DuckDB database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )

        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        checkpoint_blobs = await asyncio.to_thread(
            self._dump_blobs,
            thread_id,
            checkpoint_ns,
            copy.pop("channel_values"),  # type: ignore[misc]
            new_versions,
        )
        async with self._cursor() as cur:
            if checkpoint_blobs:
                await asyncio.to_thread(
                    cur.executemany, self.UPSERT_CHECKPOINT_BLOBS_SQL, checkpoint_blobs
                )
            await asyncio.to_thread(
                cur.execute,
                self.UPSERT_CHECKPOINTS_SQL,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_id,
                    self._dump_checkpoint(copy),
                    self._dump_metadata(metadata),
                ),
            )

        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        params = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            writes,
        )
        async with self._cursor() as cur:
            await asyncio.to_thread(cur.executemany, query, params)

    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[duckdb.DuckDBPyConnection]:
        async with self.lock:
            with self.conn.cursor() as cur:
                yield cur

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the DuckDB database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
        """
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the DuckDB database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncDuckDBSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
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

        This method saves a checkpoint to the DuckDB database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

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
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id), self.loop
        ).result()
