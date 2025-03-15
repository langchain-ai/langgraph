import asyncio
import random
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional, TypeVar

import aiosqlite
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import ChannelProtocol
from langgraph.checkpoint.sqlite.utils import search_where

T = TypeVar("T", bound=Callable)


class AsyncSqliteSaver(BaseCheckpointSaver[str]):
    """An asynchronous checkpoint saver that stores checkpoints in a SQLite database.

    This class provides an asynchronous interface for saving and retrieving checkpoints
    using a SQLite database. It's designed for use in asynchronous environments and
    offers better performance for I/O-bound operations compared to synchronous alternatives.

    Attributes:
        conn (aiosqlite.Connection): The asynchronous SQLite database connection.
        serde (SerializerProtocol): The serializer used for encoding/decoding checkpoints.

    Tip:
        Requires the [aiosqlite](https://pypi.org/project/aiosqlite/) package.
        Install it with `pip install aiosqlite`.

    Warning:
        While this class supports asynchronous checkpointing, it is not recommended
        for production workloads due to limitations in SQLite's write performance.
        For production use, consider a more robust database like PostgreSQL.

    Tip:
        Remember to **close the database connection** after executing your code,
        otherwise, you may see the graph "hang" after execution (since the program
        will not exit until the connection is closed).

        The easiest way is to use the `async with` statement as shown in the examples.

        ```python
        async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
            # Your code here
            graph = builder.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": "thread-1"}}
            async for event in graph.astream_events(..., config, version="v1"):
                print(event)
        ```

    Examples:
        Usage within StateGraph:

        ```pycon
        >>> import asyncio
        >>>
        >>> from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> async def main():
        >>>     builder = StateGraph(int)
        >>>     builder.add_node("add_one", lambda x: x + 1)
        >>>     builder.set_entry_point("add_one")
        >>>     builder.set_finish_point("add_one")
        >>>     async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        >>>         graph = builder.compile(checkpointer=memory)
        >>>         coro = graph.ainvoke(1, {"configurable": {"thread_id": "thread-1"}})
        >>>         print(await asyncio.gather(coro))
        >>>
        >>> asyncio.run(main())
        Output: [2]
        ```
        Raw usage:

        ```pycon
        >>> import asyncio
        >>> import aiosqlite
        >>> from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        >>>
        >>> async def main():
        >>>     async with aiosqlite.connect("checkpoints.db") as conn:
        ...         saver = AsyncSqliteSaver(conn)
        ...         config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
        ...         checkpoint = {"ts": "2023-05-03T10:00:00Z", "data": {"key": "value"}, "id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}
        ...         saved_config = await saver.aput(config, checkpoint, {}, {})
        ...         print(saved_config)
        >>> asyncio.run(main())
        {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '0c62ca34-ac19-445d-bbb0-5b4984975b2a'}}
        ```
    """

    lock: asyncio.Lock
    is_setup: bool

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        serde: Optional[SerializerProtocol] = None,
    ):
        super().__init__(serde=serde)
        self.jsonplus_serde = JsonPlusSerializer()
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.is_setup = False

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, conn_string: str
    ) -> AsyncIterator["AsyncSqliteSaver"]:
        """Create a new AsyncSqliteSaver instance from a connection string.

        Args:
            conn_string (str): The SQLite connection string.

        Yields:
            AsyncSqliteSaver: A new AsyncSqliteSaver instance.
        """
        async with aiosqlite.connect(conn_string) as conn:
            yield cls(conn)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the SQLite database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
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
                    "Synchronous calls to AsyncSqliteSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the SQLite database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            # we don't check in other methods to avoid the overhead
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncSqliteSaver are only allowed from a "
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
                    anext(aiter_),  # noqa: F821
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the SQLite database. The checkpoint is associated
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
        task_path: str = "",
    ) -> None:
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the SQLite database if they don't
        already exist. It is called automatically when needed and should not be called
        directly by the user.
        """
        async with self.lock:
            if self.is_setup:
                return
            if not self.conn.is_alive():
                await self.conn
            async with self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                );
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    value BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                );
                """
            ):
                await self.conn.commit()

            self.is_setup = True

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the SQLite database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        await self.setup()
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        async with self.lock, self.conn.cursor() as cur:
            # find the latest checkpoint for the thread_id
            if checkpoint_id := get_checkpoint_id(config):
                await cur.execute(
                    "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        checkpoint_id,
                    ),
                )
            else:
                await cur.execute(
                    "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1",
                    (str(config["configurable"]["thread_id"]), checkpoint_ns),
                )
            # if a checkpoint is found, return it
            if value := await cur.fetchone():
                (
                    thread_id,
                    checkpoint_id,
                    parent_checkpoint_id,
                    type,
                    checkpoint,
                    metadata,
                ) = value
                if not get_checkpoint_id(config):
                    config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    }
                # find any pending writes
                await cur.execute(
                    "SELECT task_id, channel, type, value FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx",
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        str(config["configurable"]["checkpoint_id"]),
                    ),
                )
                # deserialize the checkpoint and metadata
                return CheckpointTuple(
                    config,
                    self.serde.loads_typed((type, checkpoint)),
                    self.jsonplus_serde.loads(metadata) if metadata is not None else {},
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
                    [
                        (task_id, channel, self.serde.loads_typed((type, value)))
                        async for task_id, channel, type, value in cur
                    ],
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

        This method retrieves a list of checkpoint tuples from the SQLite database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        await self.setup()
        where, params = search_where(config, filter, before)
        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
        FROM checkpoints
        {where}
        ORDER BY checkpoint_id DESC"""
        if limit:
            query += f" LIMIT {limit}"
        async with (
            self.lock,
            self.conn.execute(query, params) as cur,
            self.conn.cursor() as wcur,
        ):
            async for (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                type,
                checkpoint,
                metadata,
            ) in cur:
                await wcur.execute(
                    "SELECT task_id, channel, type, value FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    self.serde.loads_typed((type, checkpoint)),
                    self.jsonplus_serde.loads(metadata) if metadata is not None else {},
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
                    [
                        (task_id, channel, self.serde.loads_typed((type, value)))
                        async for task_id, channel, type, value in wcur
                    ],
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the SQLite database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.jsonplus_serde.dumps(
            get_checkpoint_metadata(config, metadata)
        )
        async with (
            self.lock,
            self.conn.execute(
                "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(config["configurable"]["thread_id"]),
                    checkpoint_ns,
                    checkpoint["id"],
                    config["configurable"].get("checkpoint_id"),
                    type_,
                    serialized_checkpoint,
                    serialized_metadata,
                ),
            ),
        ):
            await self.conn.commit()
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
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
        """
        query = (
            "INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else "INSERT OR IGNORE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        await self.setup()
        async with self.lock, self.conn.cursor() as cur:
            await cur.executemany(
                query,
                [
                    (
                        str(config["configurable"]["thread_id"]),
                        str(config["configurable"]["checkpoint_ns"]),
                        str(config["configurable"]["checkpoint_id"]),
                        task_id,
                        WRITES_IDX_MAP.get(channel, idx),
                        channel,
                        *self.serde.dumps_typed(value),
                    )
                    for idx, (channel, value) in enumerate(writes)
                ],
            )
            await self.conn.commit()

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        """Generate the next version ID for a channel.

        This method creates a new version identifier for a channel based on its current version.

        Args:
            current (Optional[str]): The current version identifier of the channel.
            channel (BaseChannel): The channel being versioned.

        Returns:
            str: The next version identifier, which is guaranteed to be monotonically increasing.
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
