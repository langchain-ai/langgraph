import asyncio
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, AsyncIterator, Optional

import aiosqlite
from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.checkpoint.sqlite import JsonPlusSerializerCompat


class AsyncSqliteSaver(BaseCheckpointSaver, AbstractAsyncContextManager):
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

    async def setup(self) -> None:
        async with self.lock:
            if self.is_setup:
                return
            if not self.conn.is_alive():
                await self.conn
            async with self.conn.executescript(
                """
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
                        self.serde.loads(value[2]) if value[2] is not None else None,
                        {
                            "configurable": {
                                "thread_id": config["configurable"]["thread_id"],
                                "thread_ts": value[1],
                            }
                        }
                        if value[1]
                        else None,
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
                        self.serde.loads(value[4]) if value[4] is not None else None,
                        {
                            "configurable": {
                                "thread_id": value[0],
                                "thread_ts": value[2],
                            }
                        }
                        if value[2]
                        else None,
                    )

    async def alist(
        self,
        config: RunnableConfig,
        *,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        await self.setup()
        query = (
            "SELECT thread_id, thread_ts, parent_ts, checkpoint, metadata FROM checkpoints WHERE thread_id = ? ORDER BY thread_ts DESC"
            if before is None
            else "SELECT thread_id, thread_ts, parent_ts, checkpoint, metadata FROM checkpoints WHERE thread_id = ? AND thread_ts < ? ORDER BY thread_ts DESC"
        )
        if limit:
            query += f" LIMIT {limit}"
        async with self.conn.execute(
            query,
            (str(config["configurable"]["thread_id"]),)
            if before is None
            else (
                str(config["configurable"]["thread_id"]),
                str(before["configurable"]["thread_ts"]),
            ),
        ) as cursor:
            async for thread_id, thread_ts, parent_ts, value, metadata in cursor:
                yield CheckpointTuple(
                    {"configurable": {"thread_id": thread_id, "thread_ts": thread_ts}},
                    self.serde.loads(value),
                    self.serde.loads(metadata) if metadata is not None else None,
                    {"configurable": {"thread_id": thread_id, "thread_ts": parent_ts}}
                    if parent_ts
                    else None,
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: Optional[dict[str, Any]] = None,
    ) -> RunnableConfig:
        await self.setup()
        async with self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints (thread_id, thread_ts, parent_ts, checkpoint, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                str(config["configurable"]["thread_id"]),
                checkpoint["ts"],
                config["configurable"].get("thread_ts"),
                self.serde.dumps(checkpoint),
                self.serde.dumps(metadata) if metadata is not None else None,
            ),
        ):
            await self.conn.commit()
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["ts"],
            }
        }
