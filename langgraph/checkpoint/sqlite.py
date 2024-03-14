import pickle
import sqlite3
from contextlib import AbstractContextManager, contextmanager
from types import TracebackType
from typing import AsyncIterator, Iterator, Optional, Self

from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointTuple


class SqliteSaver(BaseCheckpointSaver, AbstractContextManager):
    conn: sqlite3.Connection

    is_setup: bool = Field(False, init=False, repr=False)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_conn_string(cls, conn_string: str) -> "SqliteSaver":
        return SqliteSaver(conn=sqlite3.connect(conn_string))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        return self.conn.close()

    def setup(self) -> None:
        if self.is_setup:
            return

        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                thread_ts TEXT NOT NULL,
                checkpoint BLOB,
                PRIMARY KEY (thread_id, thread_ts)
            );
            """
        )

        self.is_setup = True

    @contextmanager
    def cursor(self, transaction: bool = True):
        self.setup()
        cur = self.conn.cursor()
        try:
            yield cur
        finally:
            if transaction:
                self.conn.commit()
            cur.close()

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        with self.cursor(transaction=False) as cur:
            if config["configurable"].get("thread_ts"):
                cur.execute(
                    "SELECT checkpoint FROM checkpoints WHERE thread_id = ? AND thread_ts = ?",
                    (
                        config["configurable"]["thread_id"],
                        config["configurable"]["thread_ts"],
                    ),
                )
                if value := cur.fetchone():
                    return CheckpointTuple(config, pickle.loads(value[0]))
            else:
                cur.execute(
                    "SELECT thread_id, thread_ts, checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY thread_ts DESC LIMIT 1",
                    (config["configurable"]["thread_id"],),
                )
                if value := cur.fetchone():
                    return CheckpointTuple(
                        {
                            "configurable": {
                                "thread_id": value[0],
                                "thread_ts": value[1],
                            }
                        },
                        pickle.loads(value[2]),
                    )

    def list(self, config: RunnableConfig) -> Iterator[CheckpointTuple]:
        with self.cursor(transaction=False) as cur:
            cur.execute(
                "SELECT thread_id, thread_ts, checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY thread_ts DESC",
                (config["configurable"]["thread_id"],),
            )
            for thread_id, thread_ts, value in cur:
                yield CheckpointTuple(
                    {"configurable": {"thread_id": thread_id, "thread_ts": thread_ts}},
                    pickle.loads(value),
                )

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        with self.cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO checkpoints (thread_id, thread_ts, checkpoint) VALUES (?, ?, ?)",
                (
                    config["configurable"]["thread_id"],
                    checkpoint["ts"],
                    pickle.dumps(checkpoint),
                ),
            )
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["ts"],
            }
        }

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError

    async def alist(
        self, config: RunnableConfig
    ) -> AsyncIterator[tuple[RunnableConfig, Checkpoint]]:
        raise NotImplementedError

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> RunnableConfig:
        raise NotImplementedError
