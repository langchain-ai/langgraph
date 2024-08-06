import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Union

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
from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol


class AsyncPostgresSaver(BasePostgresSaver):
    lock: asyncio.Lock

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
                self.CREATE_CHECKPOINTS_SQL,
                self.CREATE_CHECKPOINT_BLOBS_SQL,
                self.CREATE_CHECKPOINT_WRITES_SQL,
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
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
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
                self.SELECT_SQL + where,
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
        new_versions: Optional[dict[str, Union[str, int, float]]] = None,
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

        async with self._cursor(pipeline=True) as cur:
            await cur.executemany(
                self.UPSERT_CHECKPOINT_BLOBS_SQL,
                await asyncio.to_thread(
                    self._dump_blobs,
                    thread_id,
                    checkpoint_ns,
                    copy.pop("channel_values"),
                    copy["channel_versions"],
                    new_versions,
                ),
            )
            await cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
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
                self.UPSERT_CHECKPOINT_WRITES_SQL,
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
