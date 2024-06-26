"""Implementation of a langgraph checkpoint saver using Postgres."""
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, AsyncIterator, Generator, Optional, Union, Tuple, List

import psycopg
from psycopg.types.json import Jsonb
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from psycopg_pool import AsyncConnectionPool, ConnectionPool


class JsonAndBinarySerializer(JsonPlusSerializer):
    def _default(self, obj):
        if isinstance(obj, (bytes, bytearray)):
            return self._encode_constructor_args(
                obj.__class__, method="fromhex", args=[obj.hex()]
            )
        return super()._default(obj)

    def dumps(self, obj: Any) -> tuple[str, bytes]:
        if isinstance(obj, bytes):
            return "bytes", obj
        elif isinstance(obj, bytearray):
            return "bytearray", obj

        return "json", super().dumps(obj)

    def loads(self, s: tuple[str, bytes]) -> Any:
        if s[0] == "bytes":
            return s[1]
        elif s[0] == "bytearray":
            return bytearray(s[1])
        elif s[0] == "json":
            return super().loads(s[1])
        else:
            raise NotImplementedError(f"Unknown serialization type: {s[0]}")


@contextmanager
def _get_sync_connection(
    connection: Union[psycopg.Connection, ConnectionPool, None],
) -> Generator[psycopg.Connection, None, None]:
    """Get the connection to the Postgres database."""
    if isinstance(connection, psycopg.Connection):
        yield connection
    elif isinstance(connection, ConnectionPool):
        with connection.connection() as conn:
            yield conn
    else:
        raise ValueError(
            "Invalid sync connection object. Please initialize the check pointer "
            f"with an appropriate sync connection object. "
            f"Got {type(connection)}."
        )


@asynccontextmanager
async def _get_async_connection(
    connection: Union[psycopg.AsyncConnection, AsyncConnectionPool, None],
) -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """Get the connection to the Postgres database."""
    if isinstance(connection, psycopg.AsyncConnection):
        yield connection
    elif isinstance(connection, AsyncConnectionPool):
        async with connection.connection() as conn:
            yield conn
    else:
        raise ValueError(
            "Invalid async connection object. Please initialize the check pointer "
            f"with an appropriate async connection object. "
            f"Got {type(connection)}."
        )


class PostgresSaver(BaseCheckpointSaver):
    """LangGraph checkpoint saver for Postgres.
    This implementation of a checkpoint saver uses a Postgres database to save
    and retrieve checkpoints. It uses the psycopg3 package to interact with the
    Postgres database.
    The checkpoint accepts either a sync_connection in the form of a psycopg.Connection
    or a psycopg.ConnectionPool object, or an async_connection in the form of a
    psycopg.AsyncConnection or psycopg.AsyncConnectionPool object.
    Usage:
    1. First time use: create schema in the database using the `create_tables` method or
       the async version `acreate_tables` method.
    2. Create a PostgresCheckpoint object with a serializer and an appropriate
       connection object.
       It's recommended to use a connection pool object for the connection.
       If using a connection object, you are responsible for closing the connection
       when done.
    Examples:
    Sync usage with a connection pool:
        .. code-block:: python
            from psycopg_pool import ConnectionPool
            from langchain_postgres import (
                PostgresCheckpoint, PickleCheckpointSerializer
            )
            pool = ConnectionPool(
                # Example configuration
                conninfo="postgresql://user:password@localhost:5432/dbname",
                max_size=20,
            )
            # Uses the pickle module for serialization
            # Make sure that you're only de-serializing trusted data
            # (e.g., payloads that you have serialized yourself).
            # Or implement a custom serializer.
            checkpoint = PostgresCheckpoint(
                serializer=PickleCheckpointSerializer(),
                sync_connection=pool,
            )
            # Use the checkpoint object to put, get, list checkpoints, etc.
    Async usage with a connection pool:
        .. code-block:: python
            from psycopg_pool import AsyncConnectionPool
            from langchain_postgres import (
                PostgresCheckpoint, PickleCheckpointSerializer
            )
            pool = AsyncConnectionPool(
                # Example configuration
                conninfo="postgresql://user:password@localhost:5432/dbname",
                max_size=20,
            )
            # Uses the pickle module for serialization
            # Make sure that you're only de-serializing trusted data
            # (e.g., payloads that you have serialized yourself).
            # Or implement a custom serializer.
            checkpoint = PostgresCheckpoint(
                serializer=PickleCheckpointSerializer(),
                async_connection=pool,
            )
            # Use the checkpoint object to put, get, list checkpoints, etc.
    Async usage with a connection object:
        .. code-block:: python
            from psycopg import AsyncConnection
            from langchain_postgres import (
                PostgresCheckpoint, PickleCheckpointSerializer
            )
            conninfo="postgresql://user:password@localhost:5432/dbname"
            # Take care of closing the connection when done
            async with AsyncConnection(conninfo=conninfo) as conn:
                # Uses the pickle module for serialization
                # Make sure that you're only de-serializing trusted data
                # (e.g., payloads that you have serialized yourself).
                # Or implement a custom serializer.
                checkpoint = PostgresCheckpoint(
                    serializer=PickleCheckpointSerializer(),
                    async_connection=conn,
                )
                # Use the checkpoint object to put, get, list checkpoints, etc.
                ...
    """

    sync_connection: Optional[Union[psycopg.Connection, ConnectionPool]] = None
    """The synchronous connection or pool to the Postgres database.
    
    If providing a connection object, please ensure that the connection is open
    and remember to close the connection when done.
    """
    async_connection: Optional[
        Union[psycopg.AsyncConnection, AsyncConnectionPool]
    ] = None
    """The asynchronous connection or pool to the Postgres database.
    
    If providing a connection object, please ensure that the connection is open
    and remember to close the connection when done.
    """

    def __init__(
        self,
        sync_connection: Optional[Union[psycopg.Connection, ConnectionPool]] = None,
        async_connection: Optional[
            Union[psycopg.AsyncConnection, AsyncConnectionPool]
        ] = None
        
    ):
        super().__init__(serde=JsonPlusSerializer())
        self.sync_connection = sync_connection
        self.async_connection = async_connection

    @contextmanager
    def _get_sync_connection(self) -> Generator[psycopg.Connection, None, None]:
        """Get the connection to the Postgres database."""
        with _get_sync_connection(self.sync_connection) as connection:
            yield connection

    @asynccontextmanager
    async def _get_async_connection(
        self,
    ) -> AsyncGenerator[psycopg.AsyncConnection, None]:
        """Get the connection to the Postgres database."""
        async with _get_async_connection(self.async_connection) as connection:
            yield connection

    @staticmethod
    def create_tables(connection: Union[psycopg.Connection, ConnectionPool], /) -> None:
        """Create the schema for the checkpoint saver."""
        with _get_sync_connection(connection) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        thread_ts TIMESTAMPTZ NOT NULL,
                        parent_ts TIMESTAMPTZ,
                        checkpoint BYTEA NOT NULL,
                        metadata BYTEA NOT NULL,
                        PRIMARY KEY (thread_id, thread_ts)
                    );
                    """
                )

    @staticmethod
    async def acreate_tables(
        connection: Union[psycopg.AsyncConnection, AsyncConnectionPool], /
    ) -> None:
        """Create the schema for the checkpoint saver."""
        async with _get_async_connection(connection) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        thread_ts TIMESTAMPTZ NOT NULL,
                        parent_ts TIMESTAMPTZ,
                        checkpoint BYTEA NOT NULL,
                        metadata BYTEA NOT NULL,
                        PRIMARY KEY (thread_id, thread_ts)
                    );
                    """
                )

    @staticmethod
    def drop_tables(connection: psycopg.Connection, /) -> None:
        """Drop the table for the checkpoint saver."""
        with connection.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS checkpoints;")

    @staticmethod
    async def adrop_tables(connection: psycopg.AsyncConnection, /) -> None:
        """Drop the table for the checkpoint saver."""
        async with connection.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS checkpoints;")

    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata) -> RunnableConfig:
        """Put the checkpoint for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
            checkpoint: The checkpoint to persist.
        Returns:
            The RunnableConfig that describes the checkpoint that was just created.
            It'll contain the `thread_id` and `thread_ts` of the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        parent_ts = config["configurable"].get("thread_ts")
        print((
                        thread_id,
                        checkpoint["ts"],
                        config["configurable"].get("thread_ts"),
                        self.serde.dumps(checkpoint),
                        self.serde.dumps(metadata),
                    )
        )
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                     """
                    INSERT INTO checkpoints 
                        (thread_id, thread_ts, parent_ts, checkpoint, metadata)
                    VALUES 
                        (%s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id, thread_ts)
                    DO UPDATE SET checkpoint = EXCLUDED.checkpoint,
                                  metadata = EXCLUDED.metadata;
                    """,
                    (
                        thread_id,
                        checkpoint["ts"],
                        parent_ts if parent_ts else None,
                        self.serde.dumps(checkpoint),
                        self.serde.dumps(metadata),
                    ),
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": checkpoint["ts"],
            },
        }

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata
    ) -> RunnableConfig:
        """Put the checkpoint for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
            checkpoint: The checkpoint to persist.
        Returns:
            The RunnableConfig that describes the checkpoint that was just created.
            It'll contain the `thread_id` and `thread_ts` of the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        parent_ts = config["configurable"].get("thread_ts")
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO checkpoints 
                        (thread_id, thread_ts, parent_ts, checkpoint, metadata)
                    VALUES 
                        (%s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id, thread_ts) 
                    DO UPDATE SET checkpoint = EXCLUDED.checkpoint,
                                  metadata = EXCLUDED.metadata;
                    """,
                    (
                        thread_id,
                        checkpoint["ts"],
                        parent_ts if parent_ts else None,
                        checkpoint,
                        metadata,
                    ),
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": checkpoint["ts"],
            },
        }

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Generator[CheckpointTuple, None, None]:
        """Get all the checkpoints for the given configuration."""
        where, args = self._search_where(config, filter, before)
        query = (
            """
            SELECT checkpoint, thread_ts, parent_ts
            FROM checkpoints
            {where}
            ORDER BY thread_ts DESC
            """
        )
        if limit:
            query += f" LIMIT {limit}"
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                thread_id = config["configurable"]["thread_id"]
                cur.execute(
                    query, 
                    {
                        "thread_id": thread_id,
                    },
                )
                for value in cur:
                    yield CheckpointTuple(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": value[1].isoformat(),
                            }
                        },
                        self.serde.loads(value[0]),
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": value[2].isoformat(),
                            }
                        }
                        if value[2]
                        else None,
                    )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Get all the checkpoints for the given configuration."""
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                thread_id = config["configurable"]["thread_id"]
                await cur.execute(
                    "SELECT checkpoint, thread_ts, parent_ts "
                    "FROM checkpoints "
                    "WHERE thread_id = %(thread_id)s "
                    "ORDER BY thread_ts DESC",
                    {
                        "thread_id": thread_id,
                    },
                )
                async for value in cur:
                    yield CheckpointTuple(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": value[1].isoformat(),
                            }
                        },
                        self.serde.loads(value[0]),
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": value[2].isoformat(),
                            }
                        }
                        if value[2]
                        else None,
                    )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get the checkpoint tuple for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
        Returns:
            The checkpoint tuple for the given configuration if it exists,
            otherwise None.
            If thread_ts is None, the latest checkpoint is returned if it exists.
        """
        thread_id = config["configurable"]["thread_id"]
        thread_ts = config["configurable"].get("thread_ts")
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                if thread_ts:
                    cur.execute(
                        "SELECT checkpoint, metadata, thread_ts, parent_ts "
                        "FROM checkpoints "
                        "WHERE thread_id = %(thread_id)s AND thread_ts = %(thread_ts)s",
                        {
                            "thread_id": thread_id,
                            "thread_ts": thread_ts,
                        },
                    )
                    value = cur.fetchone()
                    if value:
                        checkpoint, metadata, thread_ts, parent_ts = value
                    return CheckpointTuple(
                            config=config,
                            checkpoint=self.serde.loads(checkpoint),
                            metadata=self.serde.loads(metadata),
                            parent_config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": thread_ts.isoformat(),
                                }
                            }
                            if thread_ts
                            else None,
                        )
                else:
                    cur.execute(
                        "SELECT checkpoint, metadata, thread_ts, parent_ts "
                        "FROM checkpoints "
                        "WHERE thread_id = %(thread_id)s "
                        "ORDER BY thread_ts DESC LIMIT 1",
                        {
                            "thread_id": thread_id,
                        },
                    )
                    value = cur.fetchone()
                    if value:
                        checkpoint, metadata, thread_ts, parent_ts = value
                        return CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": thread_ts.isoformat(),
                                }
                            },
                            checkpoint=self.serde.loads(checkpoint),
                            metadata=self.serde.loads(metadata),
                            parent_config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": parent_ts.isoformat(),
                                }
                            }
                            if parent_ts
                            else None,
                        )
        return None

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get the checkpoint tuple for the given configuration.
        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
        Returns:
            The checkpoint tuple for the given configuration if it exists,
            otherwise None.
            If thread_ts is None, the latest checkpoint is returned if it exists.
        """
        thread_id = config["configurable"]["thread_id"]
        thread_ts = config["configurable"].get("thread_ts")
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                if thread_ts:
                    await cur.execute(
                        "SELECT checkpoint, parent_ts "
                        "FROM checkpoints "
                        "WHERE thread_id = %(thread_id)s AND thread_ts = %(thread_ts)s",
                        {
                            "thread_id": thread_id,
                            "thread_ts": thread_ts,
                        },
                    )
                    value = await cur.fetchone()
                    if value:
                        return CheckpointTuple(
                            config,
                            self.serde.loads(value[0]),
                            {
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": value[1].isoformat(),
                                }
                            }
                            if value[1]
                            else None,
                        )
                else:
                    await cur.execute(
                        "SELECT checkpoint, thread_ts, parent_ts "
                        "FROM checkpoints "
                        "WHERE thread_id = %(thread_id)s "
                        "ORDER BY thread_ts DESC LIMIT 1",
                        {
                            "thread_id": thread_id,
                        },
                    )
                    value = await cur.fetchone()
                    if value:
                        return CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": value[1].isoformat(),
                                }
                            },
                            checkpoint=self.serde.loads(value[0]),
                            parent_config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": value[2].isoformat(),
                                }
                            }
                            if value[2]
                            else None,
                        )

        return None
    
    def _search_where(
        self,
        config: Optional[RunnableConfig],
        filter: Optional[dict[str, Any]],
        before: Optional[RunnableConfig] = None,
    ) -> Tuple[str, List[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, cursor.

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

        # construct predicate for metadata filter
        if filter:
            wheres.append("metadata @> %s ")
            param_values.append(Jsonb(filter))

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < %s ")
            param_values.append(before["configurable"]["thread_ts"])

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )