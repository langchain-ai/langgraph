import asyncio
import functools
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import aiosqlite
from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.checkpoint.sqlite import JsonPlusSerializerCompat, search_where

T = TypeVar("T", bound=callable)


def not_implemented_sync_method(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError(
            "The AsyncSqliteSaver does not support synchronous methods. "
            "Consider using the SqliteSaver instead.\n"
            "from langgraph.checkpoint.sqlite import SqliteSaver\n"
            "See https://langchain-ai.github.io/langgraph/reference/checkpoints/langgraph.checkpoint.sqlite.SqliteSaver "
            "for more information."
        )

    return wrapper


class AsyncSqliteSaver(BaseCheckpointSaver, AbstractAsyncContextManager):
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
        >>> import aiosqlite
        >>>
        >>> from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> builder = StateGraph(int)
        >>> builder.add_node("add_one", lambda x: x + 1)
        >>> builder.set_entry_point("add_one")
        >>> builder.set_finish_point("add_one")
        >>> memory = AsyncSqliteSaver.from_conn_string("checkpoints.sqlite")
        >>> graph = builder.compile(checkpointer=memory)
        >>> coro = graph.ainvoke(1, {"configurable": {"thread_id": "thread-1"}})
        >>> asyncio.run(coro)
        Output: 2
        ```
        Raw usage:

        ```pycon
        >>> import asyncio
        >>> import aiosqlite
        >>> from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
        >>>
        >>> async def main():
        >>>     async with aiosqlite.connect("checkpoints.db") as conn:
        ...         saver = AsyncSqliteSaver(conn)
        ...         config = {"configurable": {"thread_id": "1"}}
        ...         checkpoint = {"ts": "2023-05-03T10:00:00Z", "data": {"key": "value"}}
        ...         saved_config = await saver.aput(config, checkpoint)
        ...         print(saved_config)
        >>> asyncio.run(main())
        {"configurable": {"thread_id": "1", "thread_ts": "2023-05-03T10:00:00Z"}}
        ```
    """

    serde = JsonPlusSerializerCompat()

    conn: aiosqlite.Connection
    lock: asyncio.Lock
    is_setup: bool

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        serde: Optional[SerializerProtocol] = None,
    ):
        super().__init__(serde=serde)
        self.conn = conn
        self.lock = asyncio.Lock()
        self.is_setup = False

    @classmethod
    def from_conn_string(cls, conn_string: str) -> "AsyncSqliteSaver":
        """Create a new AsyncSqliteSaver instance from a connection string.

        Args:
            conn_string (str): The SQLite connection string.

        Returns:
            AsyncSqliteSaver: A new AsyncSqliteSaver instance.
        """
        return AsyncSqliteSaver(conn=aiosqlite.connect(conn_string))

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        __exc_type: Optional[type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        if self.is_setup:
            return await self.conn.close()

    @not_implemented_sync_method
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        Note:
            This method is not implemented for the AsyncSqliteSaver. Use `aget` instead.
            Or consider using the [SqliteSaver][sqlitesaver] checkpointer.
        """

    @not_implemented_sync_method
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        Note:
            This method is not implemented for the AsyncSqliteSaver. Use `alist` instead.
            Or consider using the [SqliteSaver][sqlitesaver] checkpointer.
        """

    @not_implemented_sync_method
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """Save a checkpoint to the database. FOO"""

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
                    thread_ts TEXT NOT NULL,
                    parent_ts TEXT,
                    checkpoint BLOB,
                    metadata BLOB,
                    PRIMARY KEY (thread_id, thread_ts)
                );
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT NOT NULL,
                    thread_ts TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    value BLOB,
                    PRIMARY KEY (thread_id, thread_ts, task_id, idx)
                );
                """
            ):
                await self.conn.commit()

            self.is_setup = True

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the SQLite database based on the
        provided config. If the config contains a "thread_ts" key, the checkpoint with
        the matching thread ID and timestamp is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        await self.setup()
        async with self.conn.cursor() as cur:
            # find the latest checkpoint for the thread_id
            if config["configurable"].get("thread_ts"):
                await cur.execute(
                    "SELECT thread_id, thread_ts, parent_ts, checkpoint, metadata FROM checkpoints WHERE thread_id = ? AND thread_ts = ?",
                    (
                        str(config["configurable"]["thread_id"]),
                        str(config["configurable"]["thread_ts"]),
                    ),
                )
            else:
                await cur.execute(
                    "SELECT thread_id, thread_ts, parent_ts, checkpoint, metadata FROM checkpoints WHERE thread_id = ? ORDER BY thread_ts DESC LIMIT 1",
                    (str(config["configurable"]["thread_id"]),),
                )
            # if a checkpoint is found, return it
            if value := await cur.fetchone():
                if not config["configurable"].get("thread_ts"):
                    config = {
                        "configurable": {
                            "thread_id": value[0],
                            "thread_ts": value[1],
                        }
                    }
                # find any pending writes
                await cur.execute(
                    "SELECT task_id, channel, value FROM writes WHERE thread_id = ? AND thread_ts = ?",
                    (
                        str(config["configurable"]["thread_id"]),
                        str(config["configurable"]["thread_ts"]),
                    ),
                )
                # deserialize the checkpoint and metadata
                return CheckpointTuple(
                    config,
                    self.serde.loads(value[3]),
                    self.serde.loads(value[4]) if value[4] is not None else {},
                    (
                        {
                            "configurable": {
                                "thread_id": value[0],
                                "thread_ts": value[2],
                            }
                        }
                        if value[2]
                        else None
                    ),
                    [
                        (task_id, channel, self.serde.loads(value))
                        async for task_id, channel, value in cur
                    ],
                )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the SQLite database based
        on the provided config. The checkpoints are ordered by timestamp in descending order.

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        await self.setup()
        where, param_values = search_where(config, filter, before)
        query = f"""SELECT thread_id, thread_ts, parent_ts, checkpoint, metadata
        FROM checkpoints
        {where}
        ORDER BY thread_ts DESC"""
        if limit:
            query += f" LIMIT {limit}"
        async with self.conn.execute(query, param_values) as cursor:
            async for thread_id, thread_ts, parent_ts, value, metadata in cursor:
                yield CheckpointTuple(
                    {"configurable": {"thread_id": thread_id, "thread_ts": thread_ts}},
                    self.serde.loads(value),
                    self.serde.loads(metadata) if metadata is not None else {},
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": parent_ts,
                            }
                        }
                        if parent_ts
                        else None
                    ),
                )

    async def alist_subgraph_checkpoints(
        self, config: RunnableConfig
    ) -> AsyncIterator[CheckpointTuple]:
        async with self.conn.cursor() as cur:
            if config["configurable"].get("thread_ts"):
                cur.execute(
                    "SELECT thread_id, thread_ts, parent_ts, checkpoint, metadata FROM checkpoints WHERE thread_id LIKE ? || '%' AND thread_ts = ?",
                    (
                        str(config["configurable"]["thread_id"]),
                        str(config["configurable"]["thread_ts"]),
                    ),
                )
            else:
                cur.execute(
                    """SELECT checkpoints.thread_id, checkpoints.thread_ts, checkpoints.parent_ts, checkpoints.checkpoint, checkpoints.metadata
                    FROM checkpoints
                    INNER JOIN (
                        SELECT thread_id, MAX(thread_ts) as thread_ts
                        FROM checkpoints
                        WHERE thread_id LIKE ? || '%'
                        GROUP BY thread_id
                    ) latest_checkpoints
                    ON checkpoints.thread_id = latest_checkpoints.thread_id AND checkpoints.thread_ts = latest_checkpoints.thread_ts
                    ORDER BY checkpoints.thread_id, checkpoints.thread_ts DESC""",
                    (str(config["configurable"]["thread_id"]),),
                )
            async for thread_id, thread_ts, parent_ts, value, metadata in cur:
                yield CheckpointTuple(
                    {"configurable": {"thread_id": thread_id, "thread_ts": thread_ts}},
                    self.serde.loads(value),
                    self.serde.loads(metadata) if metadata is not None else {},
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": parent_ts,
                            }
                        }
                        if parent_ts
                        else None
                    ),
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the SQLite database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.

        Returns:
            RunnableConfig: The updated config containing the saved checkpoint's timestamp.
        """
        await self.setup()
        async with self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints (thread_id, thread_ts, parent_ts, checkpoint, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                str(config["configurable"]["thread_id"]),
                checkpoint["id"],
                config["configurable"].get("thread_ts"),
                self.serde.dumps(checkpoint),
                self.serde.dumps(metadata),
            ),
        ):
            await self.conn.commit()
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        await self.setup()
        async with self.conn.executemany(
            "INSERT OR REPLACE INTO writes (thread_id, thread_ts, task_id, idx, channel, value) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    str(config["configurable"]["thread_id"]),
                    str(config["configurable"]["thread_ts"]),
                    task_id,
                    idx,
                    channel,
                    self.serde.dumps(value),
                )
                for idx, (channel, value) in enumerate(writes)
            ],
        ):
            await self.conn.commit()
