import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from langchain_core.runnables import RunnableConfig
from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.postgres import SELECT_SQL, BasePostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol

MetadataInput = Optional[dict[str, Any]]


class AsyncPostgresSaver(BasePostgresSaver):
    lock: asyncio.Lock
    latest_tuple: Optional[CheckpointTuple]

    is_setup: bool

    def __init__(
        self,
        conn: AsyncConnection,
        pipe: Optional[AsyncPipeline] = None,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.latest_tuple: Optional[CheckpointTuple] = None
        self.is_setup = False

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, conn_string: str, *, pipeline: bool = False
    ) -> AsyncIterator["AsyncPostgresSaver"]:
        """Create a new PostgresSaver instance from a connection string.

        Args:
            conn_string (str): The Postgres connection info string.
            pipeline (bool): whether to use AsyncPipeline

        Returns:
            PostgresSaver: A new PostgresSaver instance.
        """
        async with await AsyncConnection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            if pipeline:
                async with conn.pipeline() as pipe:
                    yield AsyncPostgresSaver(conn, pipe)
            else:
                yield AsyncPostgresSaver(conn)

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the SQLite database if they don't
        already exist. It is called automatically when needed and should not be called
        directly by the user.
        """
        if self.is_setup:
            return
        async with self.lock:
            create_table_queries = [
                """CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint JSONB NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )""",
                """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL,
                    version TEXT NOT NULL,
                    type TEXT NOT NULL,
                    blob BYTEA NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                );""",
                """CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    blob BYTEA NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                );
                """,
            ]
            async with self.conn.cursor() as cur:
                for query in create_table_queries:
                    await cur.execute(query)

            if self.pipe:
                await self.pipe.sync()

            self.is_setup = True

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        await self.setup()
        where, args = self._search_where(config, filter, before)
        query = SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        # if we change this to use .stream() we need to make sure to close the cursor
        async for value in await self.conn.execute(query, args, binary=True):
            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": value["thread_id"],
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["checkpoint_id"],
                    }
                },
                {
                    **self._load_checkpoint(value["checkpoint"]),
                    "channel_values": await asyncio.to_thread(
                        self._load_blobs, value["channel_values"]
                    ),
                },
                value["metadata"],
                {
                    "configurable": {
                        "thread_id": value["thread_id"],
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["parent_checkpoint_id"],
                    }
                }
                if value["parent_checkpoint_id"]
                else None,
            )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        async with self._cursor() as cur:
            cur = await self.conn.execute(
                SELECT_SQL + where,
                args,
                binary=True,
            )

            async for value in cur:
                return CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    {
                        **self._load_checkpoint(value["checkpoint"]),
                        "channel_values": await asyncio.to_thread(
                            self._load_blobs, value["channel_values"]
                        ),
                    },
                    value["metadata"],
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": value["parent_checkpoint_id"],
                        }
                    }
                    if value["parent_checkpoint_id"]
                    else None,
                    await asyncio.to_thread(self._load_writes, value["pending_writes"]),
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        await self.setup()
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
        previous = (
            self.latest_tuple
            if self.latest_tuple
            and checkpoint_id
            and self.latest_tuple.config["configurable"]["thread_id"] == thread_id
            and self.latest_tuple.config["configurable"]["checkpoint_ns"]
            == checkpoint_ns
            and self.latest_tuple.config["configurable"]["checkpoint_id"]
            == checkpoint_id
            else None
        )
        self.latest_tuple = CheckpointTuple(
            config=next_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=config,
        )

        async with self._cursor(pipeline=True) as cur:
            await cur.executemany(
                """INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING""",
                await asyncio.to_thread(
                    self._dump_blobs,
                    thread_id,
                    checkpoint_ns,
                    copy.pop("channel_values"),
                    copy["channel_versions"],
                    previous.checkpoint["channel_versions"] if previous else None,
                ),
            )
            await cur.execute(
                """
                INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
                DO UPDATE SET
                    checkpoint = EXCLUDED.checkpoint,
                    metadata = EXCLUDED.metadata;""",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_id,
                    Jsonb(self._dump_checkpoint(copy)),
                    Jsonb(metadata),
                ),
            )
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        async with self._cursor() as cur:
            await cur.executemany(
                """INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING""",
                await asyncio.to_thread(
                    self._dump_writes,
                    config["configurable"]["thread_id"],
                    config["configurable"]["checkpoint_ns"],
                    config["configurable"]["checkpoint_id"],
                    task_id,
                    writes,
                ),
            )

    @asynccontextmanager
    async def _cursor(self, *, pipeline: bool = False) -> AsyncIterator[AsyncCursor]:
        await self.setup()
        if self.pipe:
            # a connection in pipeline mode can be used concurrently
            # in multiple threads/coroutines, but only one cursor can be
            # used at a time
            try:
                async with self.conn.cursor(binary=True) as cur:
                    yield cur
            finally:
                await self.pipe.sync()
        elif pipeline:
            # a connection not in pipeline mode can only be used by one
            # thread/coroutine at a time, so we acquire a lock
            async with self.lock, self.conn.pipeline(), self.conn.cursor(
                binary=True
            ) as cur:
                yield cur
        else:
            async with self.lock, self.conn.cursor(binary=True) as cur:
                yield cur
