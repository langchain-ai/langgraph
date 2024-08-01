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
    get_checkpoint_id,
)
from langgraph.checkpoint.sqlite.utils import search_where

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
        >>>
        >>> from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
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
        >>> from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        >>>
        >>> async def main():
        >>>     async with aiosqlite.connect("checkpoints.db") as conn:
        ...         saver = AsyncSqliteSaver(conn)
        ...         config = {"configurable": {"thread_id": "1"}}
        ...         checkpoint = {"ts": "2023-05-03T10:00:00Z", "data": {"key": "value"}}
        ...         saved_config = await saver.aput(config, checkpoint)
        ...         print(saved_config)
        >>> asyncio.run(main())
        {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}
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
        the matching thread ID and timestamp is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        await self.setup()
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        async with self.conn.cursor() as cur:
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
                    "SELECT task_id, channel, type, value FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
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
                    self.serde.loads_typed((type, metadata))
                    if metadata is not None
                    else {},
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
        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata
        FROM checkpoints
        {where}
        ORDER BY checkpoint_id DESC"""
        if limit:
            query += f" LIMIT {limit}"
        async with self.conn.execute(query, param_values) as cursor:
            async for (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                value,
                metadata,
            ) in cursor:
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    self.serde.loads_typed(("json", value)),
                    self.serde.loads_typed(("json", metadata))
                    if metadata is not None
                    else {},
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
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.serde.dumps_typed(metadata)[1]
        async with self.conn.execute(
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
            "INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    str(config["configurable"]["thread_id"]),
                    str(config["configurable"]["checkpoint_ns"]),
                    str(config["configurable"]["checkpoint_id"]),
                    task_id,
                    idx,
                    channel,
                    *self.serde.dumps_typed(value),
                )
                for idx, (channel, value) in enumerate(writes)
            ],
        ):
            await self.conn.commit()
