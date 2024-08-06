import threading
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Union

from langchain_core.runnables import RunnableConfig
from psycopg import Connection, Cursor, Pipeline
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.postgres.base import (
    BasePostgresSaver,
)
from langgraph.checkpoint.serde.base import SerializerProtocol


class PostgresSaver(BasePostgresSaver):
    lock: threading.Lock

    is_setup: bool

    def __init__(
        self,
        conn: Connection,
        pipe: Optional[Pipeline] = None,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.conn = conn
        self.pipe = pipe
        self.lock = threading.Lock()
        self.is_setup = False

    @classmethod
    @contextmanager
    def from_conn_string(
        cls, conn_string: str, *, pipeline: bool = False
    ) -> Iterator["PostgresSaver"]:
        """Create a new PostgresSaver instance from a connection string.

        Args:
            conn_string (str): The Postgres connection info string.
            pipeline (bool): whether to use Pipeline

        Returns:
            PostgresSaver: A new PostgresSaver instance.
        """
        with Connection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            if pipeline:
                with conn.pipeline() as pipe:
                    yield PostgresSaver(conn, pipe)
            else:
                yield PostgresSaver(conn)

    def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the SQLite database if they don't
        already exist. It is called automatically when needed and should not be called
        directly by the user.
        """
        if self.is_setup:
            return
        with self.lock:
            create_table_queries = [
                self.CREATE_CHECKPOINTS_SQL,
                self.CREATE_CHECKPOINT_BLOBS_SQL,
                self.CREATE_CHECKPOINT_WRITES_SQL,
            ]
            with self.conn.cursor(binary=True) as cur:
                for query in create_table_queries:
                    cur.execute(query)

            if self.pipe:
                self.pipe.sync()

            self.is_setup = True

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        self.setup()
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        # if we change this to use .stream() we need to make sure to close the cursor
        for value in self.conn.execute(query, args, binary=True):
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
                    "channel_values": self._load_blobs(value["channel_values"]),
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

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        with self._cursor() as cur:
            cur = self.conn.execute(
                self.SELECT_SQL + where,
                args,
                binary=True,
            )

            for value in cur:
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
                        "channel_values": self._load_blobs(value["channel_values"]),
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
                    self._load_writes(value["pending_writes"]),
                )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, Union[str, int, float]]] = None,
    ) -> RunnableConfig:
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

        with self._cursor(pipeline=True) as cur:
            cur.executemany(
                self.UPSERT_CHECKPOINT_BLOBS_SQL,
                self._dump_blobs(
                    thread_id,
                    checkpoint_ns,
                    copy.pop("channel_values"),
                    copy["channel_versions"],
                    new_versions,
                ),
            )
            cur.execute(
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

    def put_writes(
        self,
        config: RunnableConfig,
        writes: List[tuple[str, Any]],
        task_id: str,
    ) -> None:
        with self._cursor() as cur:
            cur.executemany(
                self.UPSERT_CHECKPOINT_WRITES_SQL,
                self._dump_writes(
                    config["configurable"]["thread_id"],
                    config["configurable"]["checkpoint_ns"],
                    config["configurable"]["checkpoint_id"],
                    task_id,
                    writes,
                ),
            )

    @contextmanager
    def _cursor(self, *, pipeline: bool = False) -> Iterator[Cursor]:
        self.setup()
        if self.pipe:
            # a connection in pipeline mode can be used concurrently
            # in multiple threads/coroutines, but only one cursor can be
            # used at a time
            try:
                with self.conn.cursor(binary=True) as cur:
                    yield cur
            finally:
                self.pipe.sync()
        elif pipeline:
            # a connection not in pipeline mode can only be used by one
            # thread/coroutine at a time, so we acquire a lock
            with self.lock, self.conn.pipeline(), self.conn.cursor(binary=True) as cur:
                yield cur
        else:
            with self.lock, self.conn.cursor(binary=True) as cur:
                yield cur
