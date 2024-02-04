import pickle
import sqlite3
from contextlib import contextmanager
from typing import Optional

from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint


class SqliteSaver(BaseCheckpointSaver):
    conn: sqlite3.Connection

    is_setup: bool = Field(False, init=False, repr=False)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_conn_string(cls, conn_string: str) -> "SqliteSaver":
        return SqliteSaver(conn=sqlite3.connect(conn_string))

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

    def setup(self) -> None:
        if self.is_setup:
            return

        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT PRIMARY KEY,
                checkpoint BLOB
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

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        with self.cursor(transaction=False) as cur:
            cur.execute(
                "SELECT checkpoint FROM checkpoints WHERE thread_id = ?",
                (config["configurable"]["thread_id"],),
            )
            if value := cur.fetchone():
                return pickle.loads(value[0])

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        with self.cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint) VALUES (?, ?)",
                (
                    config["configurable"]["thread_id"],
                    pickle.dumps(checkpoint),
                ),
            )
