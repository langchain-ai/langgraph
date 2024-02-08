import pickle
from contextlib import asynccontextmanager
from typing import Optional, final

import aiosqlite
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint


class AsyncSqliteSaver(BaseCheckpointSaver):
    conn: aiosqlite.Connection

    is_setup: bool = Field(False, init=False, repr=False)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_conn_string(cls, conn_string: str) -> "AsyncSqliteSaver":
        return AsyncSqliteSaver(conn=aiosqlite.connect(conn_string))

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id="thread_id",
                annotation=str,
                name="Thread ID",
                description=None,
                default="",
                is_shared=True,
            ),
        ]

    async def setup(self) -> None:
        print("hello")
        if self.is_setup:
            return

        try:
            await self.conn
            await self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    checkpoint BLOB
                );
                """
            )
            await self.conn.commit()

            print("good bye")

            self.is_setup = True
        except BaseException as e:
            print(e)
            raise e

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        raise NotImplementedError

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        raise NotImplementedError

    async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
        await self.setup()
        async with self.conn.execute(
            "SELECT checkpoint FROM checkpoints WHERE thread_id = ?",
            (config["configurable"]["thread_id"],),
        ) as cursor:
            if value := await cursor.fetchone():
                return pickle.loads(value[0])

    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        await self.setup()
        await self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint) VALUES (?, ?)",
            (
                config["configurable"]["thread_id"],
                pickle.dumps(checkpoint),
            ),
        )
        await self.conn.commit()
