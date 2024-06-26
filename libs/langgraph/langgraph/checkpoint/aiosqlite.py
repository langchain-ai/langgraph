import asyncio
import functools
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, AsyncIterator, Dict, Iterator, Optional, TypeVar

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

    Tip:
        Requires the [aiosqlite](https://pypi.org/project/aiosqlite/) package.
        Install it with `pip install aiosqlite`.

    Note:
        While this class does support asynchronous checkpointing, it is not recommended
        for production workloads, due to limitations in SQLite's write performance. For
        production workloads, consider using a more robust database like PostgreSQL.

    Args:
        conn (aiosqlite.Connection): The asynchronous SQLite database connection.
        serde (Optional[SerializerProtocol]): The serializer to use for serializing and deserializing checkpoints. Defaults to JsonPlusSerializerCompat.

    Examples:
        Usage within a StateGraph:
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
        if config["configurable"].get("thread_ts"):
            async with self.conn.execute(
                "SELECT checkpoint, parent_ts, metadata FROM checkpoints WHERE thread_id = ? AND thread_ts = ?",
                (
                    str(config["configurable"]["thread_id"]),
                    str(config["configurable"]["thread_ts"]),
                ),
            ) as cursor:
                if value := await cursor.fetchone():
                    return CheckpointTuple(
                        config,
                        self.serde.loads(value[0]),
                        self.serde.loads(value[2]) if value[2] is not None else {},
                        (
                            {
                                "configurable": {
                                    "thread_id": config["configurable"]["thread_id"],
                                    "thread_ts": value[1],
                                }
                            }
                            if value[1]
                            else None
                        ),
                    )
        else:
            async with self.conn.execute(
                "SELECT thread_id, thread_ts, parent_ts, checkpoint, metadata FROM checkpoints WHERE thread_id = ? ORDER BY thread_ts DESC LIMIT 1",
                (str(config["configurable"]["thread_id"]),),
            ) as cursor:
                if value := await cursor.fetchone():
                    return CheckpointTuple(
                        {
                            "configurable": {
                                "thread_id": value[0],
                                "thread_ts": value[1],
                            }
                        },
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
            config (RunnableConfig): The config to use for listing the checkpoints.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified timestamp are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of checkpoint tuples.
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
