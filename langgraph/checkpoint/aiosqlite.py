import pickle
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import AsyncIterator, Optional

import aiosqlite
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointTuple


class AsyncSqliteSaver(BaseCheckpointSaver, AbstractAsyncContextManager):
    conn: aiosqlite.Connection

    is_setup: bool = Field(False, init=False, repr=False)

    class Config:
        arbitrary_types_allowed = True

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
        return await self.conn.close()

    async def setup(self) -> None:
        if self.is_setup:
            return

        await self.conn
        async with self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                thread_ts TEXT NOT NULL,
                parent_ts TEXT,
                checkpoint BLOB,
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
                "SELECT checkpoint, parent_ts FROM checkpoints WHERE thread_id = ? AND thread_ts = ?",
                (
                    config["configurable"]["thread_id"],
                    config["configurable"]["thread_ts"],
                ),
            ) as cursor:
                if value := await cursor.fetchone():
                    return CheckpointTuple(
                        config,
                        pickle.loads(value[0]),
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
                "SELECT thread_id, thread_ts, parent_ts, checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY thread_ts DESC LIMIT 1",
                (config["configurable"]["thread_id"],),
            ) as cursor:
                if value := await cursor.fetchone():
                    return CheckpointTuple(
                        {
                            "configurable": {
                                "thread_id": value[0],
                                "thread_ts": value[1],
                            }
                        },
                        pickle.loads(value[3]),
                        {
                            "configurable": {
                                "thread_id": value[0],
                                "thread_ts": value[2],
                            }
                        }
                        if value[2]
                        else None,
                    )

    async def alist(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
        await self.setup()
        async with self.conn.execute(
            "SELECT thread_id, thread_ts, parent_ts, checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY thread_ts DESC",
            (config["configurable"]["thread_id"],),
        ) as cursor:
            async for thread_id, thread_ts, parent_ts, value in cursor:
                yield CheckpointTuple(
                    {"configurable": {"thread_id": thread_id, "thread_ts": thread_ts}},
                    pickle.loads(value),
                    {"configurable": {"thread_id": thread_id, "thread_ts": parent_ts}}
                    if parent_ts
                    else None,
                )

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> RunnableConfig:
        await self.setup()
        async with self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints (thread_id, thread_ts, parent_ts, checkpoint) VALUES (?, ?, ?, ?)",
            (
                config["configurable"]["thread_id"],
                checkpoint["ts"],
                config["configurable"].get("thread_ts"),
                pickle.dumps(checkpoint),
            ),
        ):
            await self.conn.commit()
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["ts"],
            }
        }
