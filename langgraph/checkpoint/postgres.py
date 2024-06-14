import asyncio
import functools
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple

import asyncpg
from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.serde.jsonplus import JsonPlusSerializer

T = TypeVar("T", bound=callable)


def not_implemented_sync_method(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError(
            "The PostgresSaver does not support synchronous methods. "
            "Consider using the AsyncSqliteSaver instead."
        )
    return wrapper


class PostgresSaver(BaseCheckpointSaver, AbstractAsyncContextManager):
    """An asynchronous checkpoint saver that stores checkpoints in a PostgreSQL database.

    Tip:
        Requires the [asyncpg](https://pypi.org/project/asyncpg/) package.
        Install it with `pip install asyncpg`.

    Note:
        While this class does support asynchronous checkpointing, it is not recommended
        for production workloads, due to limitations in PostgreSQL's write performance for
        heavy concurrent workloads. For production workloads, consider using a more robust
        setup with connection pooling and proper indexing.

    Args:
        conn (asyncpg.Connection): The asynchronous PostgreSQL database connection.
        serde (Optional[SerializerProtocol]): The serializer to use for serializing and deserializing checkpoints. Defaults to JsonPlusSerializer.

    Examples:
        Usage within a StateGraph:
        ```python
        import asyncio
        import asyncpg

        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.graph import StateGraph

        async def main():
            conn = await asyncpg.connect("postgresql://user:password@localhost:5432/mydatabase")
            memory = PostgresSaver(conn)
            builder = StateGraph(int)
            builder.add_node("add_one", lambda x: x + 1)
            builder.set_entry_point("add_one")
            builder.set_finish_point("add_one")
            graph = builder.compile(checkpointer=memory)
            config = {"configurable": {"thread_id": "thread-1"}}
            result = await graph.ainvoke(1, config)
            print(result)  # Output: 2
            await conn.close()

        asyncio.run(main())
        ```

        Raw usage:
        ```python
        import asyncio
        import asyncpg
        from langgraph.checkpoint.postgres import PostgresSaver

        async def main():
            conn = await asyncpg.connect("postgresql://user:password@localhost:5432/mydatabase")
            saver = PostgresSaver(conn)
            config = {"configurable": {"thread_id": "1"}}
            checkpoint = {"ts": "2023-05-03T10:00:00Z",
                "data": {"key": "value"}}
            saved_config = await saver.aput(config, checkpoint)
            print(saved_config)
            await conn.close()

        asyncio.run(main())
        ```
    """

    serde = JsonPlusSerializer()

    conn: asyncpg.Connection
    is_setup: bool

    def __init__(
        self,
        conn: asyncpg.Connection,
        *,
        serde: Optional[SerializerProtocol] = None,
    ):
        super().__init__(serde=serde)
        self.conn = conn
        self.is_setup = False

    @classmethod
    async def from_conn_string(cls, conn_string: str) -> "PostgresSaver":
        """Create a new PostgresSaver instance from a connection string.

        Args:
            conn_string (str): The PostgreSQL connection string.

        Returns:
            PostgresSaver: A new PostgresSaver instance.
        """
        conn = await asyncpg.connect(conn_string)
        return PostgresSaver(conn)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        __exc_type: Optional[type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
        ) -> Optional[bool]:
                if self.is_setup:
                    await self.conn.close()

    @ not_implemented_sync_method
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        Note:
            This method is not implemented for the PostgresSaver. Use `aget` instead.
        """

    @ not_implemented_sync_method
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
            This method is not implemented for the PostgresSaver. Use `alist` instead.
        """

    @ not_implemented_sync_method
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """Save a checkpoint to the database."""

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the PostgreSQL database if they don't
        already exist. It is called automatically when needed and should not be called
        directly by the user.
        """
        if self.is_setup:
            return

        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                thread_ts TEXT NOT NULL,
                parent_ts TEXT,
                checkpoint BYTEA,
                metadata BYTEA,
                PRIMARY KEY (thread_id, thread_ts)
            );
            """
        )

        self.is_setup = True

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the PostgreSQL database based on the
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
            value = await self.conn.fetchrow(
                "SELECT checkpoint, parent_ts, metadata FROM checkpoints WHERE thread_id = $1 AND thread_ts = $2",
                str(config["configurable"]["thread_id"]),
                str(config["configurable"]["thread_ts"]),
            )
            if value:
                return CheckpointTuple(
                    config,
                    self.serde.loads(value["checkpoint"]),
                    self.serde.loads(value["metadata"]) if value["metadata"] is not None else {},
                    (
                        {
                            "configurable": {
                                "thread_id": config["configurable"]["thread_id"],
                                "thread_ts": value["parent_ts"],
                            }
                        }
                        if value["parent_ts"]
                        else None
                    ),
                )
        else:
            value = await self.conn.fetchrow(
                "SELECT thread_id, thread_ts, parent_ts, checkpoint, metadata FROM checkpoints WHERE thread_id = $1 ORDER BY thread_ts DESC LIMIT 1",
                str(config["configurable"]["thread_id"]),
            )
            if value:
                return CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "thread_ts": value["thread_ts"],
                        }
                    },
                    self.serde.loads(value["checkpoint"]),
                    self.serde.loads(value["metadata"]) if value["metadata"] is not None else {},
                    (
                        {
                            "configurable": {
                                "thread_id": value["thread_id"],
                                "thread_ts": value["parent_ts"],
                            }
                        }
                        if value["parent_ts"]
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

        This method retrieves a list of checkpoint tuples from the PostgreSQL database based
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
        async for value in self.conn.cursor(query, *param_values):
            yield CheckpointTuple(
                {"configurable": {"thread_id": value["thread_id"], "thread_ts": value["thread_ts"]}},
                self.serde.loads(value["checkpoint"]),
                self.serde.loads(value["metadata"]) if value["metadata"] is not None else {},
                (
                    {
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "thread_ts": value["parent_ts"],
                        }
                    }
                    if value["parent_ts"]
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

        This method saves a checkpoint to the PostgreSQL database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.

        Returns:
            RunnableConfig: The updated config containing the saved checkpoint's timestamp.
        """
        await self.setup()
        await self.conn.execute(
            "INSERT INTO checkpoints (thread_id, thread_ts, parent_ts, checkpoint, metadata) VALUES ($1, $2, $3, $4, $5) ON CONFLICT (thread_id, thread_ts) DO UPDATE SET checkpoint = EXCLUDED.checkpoint, metadata = EXCLUDED.metadata",
            str(config["configurable"]["thread_id"]),
            checkpoint["id"],
            config["configurable"].get("thread_ts"),
            self.serde.dumps(checkpoint),
            self.serde.dumps(metadata),
        )
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["id"],
            }
        }
