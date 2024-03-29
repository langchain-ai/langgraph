import pickle
from contextlib import AbstractContextManager, contextmanager
from types import TracebackType
from typing import AsyncIterator, Iterator, Optional

import psycopg
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig
from typing_extensions import Self

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointTuple


class PostgresqlSaver(BaseCheckpointSaver, AbstractContextManager):
    conn: psycopg.Connection

    is_setup: bool = Field(False, init=False, repr=False)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_conn_string(cls, conn_string: str) -> "PostgresqlSaver":
        return PostgresqlSaver(conn=psycopg.connect(conn_string))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        __exc_type: Optional[type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return self.conn.close()

    def setup(self) -> None:
        if self.is_setup:
            return

        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    thread_ts TEXT NOT NULL,
                    parent_ts TEXT,
                    checkpoint BYTEA,
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
                    """
                    SELECT * 
                    FROM checkpoints 
                    WHERE thread_id = %s
                    AND thread_ts = %s
                    """,
                    (
                        config["configurable"]["thread_id"],
                        config["configurable"]["thread_ts"],
                    ),
                )
                if value := cur.fetchone():
                    return CheckpointTuple(
                        config=config,
                        checkpoint=pickle.loads(value[3]),
                        parent_config={
                            "configurable": {
                                "thread_id": config["configurable"]["thread_id"],
                                "thread_ts": value[1],
                            }
                        }
                        if value[1]
                        else None,
                    )
            else:
                cur.execute(
                    """
                    SELECT thread_id, thread_ts, parent_ts, checkpoint
                    FROM checkpoints 
                    WHERE thread_id = %s
                    ORDER BY thread_ts DESC 
                    LIMIT 1
                    """,
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

    def list(self, config: RunnableConfig) -> Iterator[CheckpointTuple]:
        with self.cursor(transaction=False) as cur:
            cur.execute(
                """
                SELECT thread_id, thread_ts, parent_ts, checkpoint 
                FROM checkpoints 
                WHERE thread_id = %s 
                ORDER BY thread_ts DESC
                """,
                (config["configurable"]["thread_id"],),
            )
            for thread_id, thread_ts, parent_ts, value in cur:
                yield CheckpointTuple(
                    {"configurable": {"thread_id": thread_id, "thread_ts": thread_ts}},
                    pickle.loads(value),
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "thread_ts": parent_ts,
                        }
                    }
                    if parent_ts
                    else None,
                )

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO checkpoints (thread_id, thread_ts, parent_ts, checkpoint)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (thread_id, thread_ts) DO UPDATE 
                SET parent_ts = EXCLUDED.parent_ts, checkpoint = EXCLUDED.checkpoint
                """,
                (
                    config["configurable"]["thread_id"],
                    checkpoint["ts"],
                    config["configurable"].get("thread_ts"),
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
        raise NotImplementedError("Use AsyncPostgresqlSaver instead")

    async def alist(
        self, config: RunnableConfig
    ) -> AsyncIterator[tuple[RunnableConfig, Checkpoint]]:
        raise NotImplementedError("Use AsyncPostgresqlSaver instead")

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> RunnableConfig:
        raise NotImplementedError("Use AsyncPostgresqlSaver instead")
