from __future__ import annotations

import json
import threading
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, closing
from typing import Any

import pyodbc
from langchain_core.runnables import RunnableConfig
from pyodbc import Row, Cursor

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.sqlserver._sql_queries import (
    DELETE_THREAD_CHECKPOINT_BLOBS,
    DELETE_THREAD_CHECKPOINT_WRITES,
    DELETE_THREAD_CHECKPOINTS,
    INSERT_CHECKPOINT_WRITES_SQL,
    MIGRATIONS_SQL,
    SELECT_PENDING_SENDS_SQL,
    SELECT_SQL,
    UPSERT_CHECKPOINT_BLOBS_SQL,
    UPSERT_CHECKPOINT_WRITES_SQL,
    UPSERT_CHECKPOINTS_SQL,
)
from langgraph.checkpoint.sqlserver.base import BaseSQLServerSaver


class SQLServerSaver(BaseSQLServerSaver):
    """Checkpointer that stores checkpoints in a Microsoft SQL Server database."""

    lock: threading.Lock

    def __init__(
        self,
        conn: pyodbc.Connection,
        schema: str | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the SQLServerSaver with a connection and optional schema.

        Args:
            conn: The SQL Server connection object.
            schema: The schema to use for the checkpoint tables. Defaults to "dbo".
        """
        super().__init__(serde=serde)
        self.conn = conn
        self.schema = schema if schema else "dbo"
        self.lock = threading.Lock()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls, conn_string: str, schema: str | None = None
    ) -> Iterator[SQLServerSaver]:
        """Create a new SQLServerSaver instance from a connection string.

        Args:
            conn_string: The SQL Server connection info string.

        Returns:
            SQLServerSaver: A new SQLServerSaver instance.
        """
        with pyodbc.connect(conn_string) as conn:
            yield cls(conn, schema=schema)

    @contextmanager
    def _cursor(self) -> Iterator[Cursor]:
        """Create a thread-safe database cursor as a context manager.

        Returns:
            Iterator[Cursor]: A context manager that yields a database cursor.
        """
        with self.lock:
            with closing(self.conn.cursor()) as cur:
                yield cur

    def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the SQL Server database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        with self._cursor() as cur:
            for command in MIGRATIONS_SQL:
                command = command.replace("${SCHEMA}", self.schema)
                cur.execute(command)

        self.conn.commit()

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the SQLServer database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config: The config to use for listing the checkpoints.
            filter: Additional filtering criteria for metadata. Defaults to None.
            before: If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit: The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.

        Examples:
            >>> from langgraph.checkpoint.sqlserver import SQLServerSaver
            >>> DB_URI = "mssql+pyodbc://sa:yourStrong(!)Password@localhost:5432/tempdb?driver=ODBC+Driver+18+for+SQL+Server"
            >>> with SQLServerSaver.from_conn_string(DB_URI) as memory:
            ... # Run a graph, then list the checkpoints
            >>>     config = {"configurable": {"thread_id": "1"}}
            >>>     checkpoints = list(memory.list(config, limit=2))
            >>> print(checkpoints)
            [CheckpointTuple(...), CheckpointTuple(...)]

            >>> config = {"configurable": {"thread_id": "1"}}
            >>> before = {"configurable": {"checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875"}}
            >>> with SQLServerSaver.from_conn_string(DB_URI) as memory:
            ... # Run a graph, then list the checkpoints
            >>>     checkpoints = list(memory.list(config, before=before))
            >>> print(checkpoints)
            [CheckpointTuple(...), ...]
        """
        where, args = self._search_where(config, before)
        query = (
            SELECT_SQL.replace("${SCHEMA}", self.schema)
            + where
            + " ORDER BY checkpoint_id DESC"
        )
        if limit:
            query += f" OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY;"

        with self._cursor() as cur:
            cur.execute(query, args)
            values: list[dict[str, Any]] = self._postprocess_results(
                data=cur.fetchall(),
                cursor=cur,
                json_fields=[
                    "metadata",
                    "checkpoint",
                    "channel_values",
                    "pending_writes",
                ],
            )

            if not values:
                return

            if filter:
                values = [
                    v for v in values if self._filter_metadata(v["metadata"], filter)
                ]

            # migrate pending sends if necessary
            if to_migrate := [
                v
                for v in values
                if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]
            ]:
                grouped_by_parent = defaultdict(list)
                for value in to_migrate:
                    grouped_by_parent[value["parent_checkpoint_id"]].append(value)

                cur.execute(
                    SELECT_PENDING_SENDS_SQL.replace("${SCHEMA}", self.schema),
                    (
                        values[0]["thread_id"],
                        ",".join([v["parent_checkpoint_id"] for v in to_migrate]),
                    ),
                )
                sends_values = self._postprocess_results(
                    cur.fetchall(), cur, json_fields=["sends"]
                )
                for sends in sends_values:
                    for value in grouped_by_parent[sends["checkpoint_id"]]:
                        if value["channel_values"] is None:
                            value["channel_values"] = []
                        self._migrate_pending_sends(
                            sends["sends"],
                            value["checkpoint"],
                            value["channel_values"],
                        )

            for value in values:
                yield self._load_checkpoint_tuple(value)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the SQL Server database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and timestamp is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.

        Examples:

            Basic:
            >>> config = {"configurable": {"thread_id": "1"}}
            >>> checkpoint_tuple = memory.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(...)

            With timestamp:

            >>> config = {
            ...    "configurable": {
            ...        "thread_id": "1",
            ...        "checkpoint_ns": "",
            ...        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            ...    }
            ... }
            >>> checkpoint_tuple = memory.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(...)
        """  # noqa
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?"
        else:
            args = (thread_id, checkpoint_ns)
            where = """
                WHERE thread_id = ? AND checkpoint_ns = ? 
                ORDER BY checkpoint_id DESC 
                OFFSET 0 ROWS FETCH NEXT 1 ROWS ONLY;
            """

        with self._cursor() as cur:
            cur.execute(
                SELECT_SQL.replace("${SCHEMA}", self.schema) + where,
                args,
            )
            value = self._postprocess_results(
                cur.fetchone(),
                cur,
                json_fields=[
                    "checkpoint",
                    "metadata",
                    "channel_values",
                    "pending_writes",
                ],
            )

            if value is None:
                return

            # migrate pending sends if necessary
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                cur.execute(
                    SELECT_PENDING_SENDS_SQL.replace("${SCHEMA}", self.schema),
                    (thread_id, ",".join([value["parent_checkpoint_id"]])),
                )
                sends = self._postprocess_results(
                    cur.fetchone(),
                    cur,
                    json_fields=[
                        "sends",
                        "checkpoint",
                        "channel_values",
                        "pending_writes",
                    ],
                )

                if sends:
                    if value["channel_values"] is None:
                        value["channel_values"] = []
                    self._migrate_pending_sends(
                        sends["sends"],
                        value["checkpoint"],
                        value["channel_values"],
                    )

            return self._load_checkpoint_tuple(value)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the SQLServer database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Examples:

            >>> from langgraph.checkpoint.sqlserver import SQLServerSaver
            >>> DB_URI = "mssql+pyodbc://sa:yourStrong(!)Password@localhost:5432/tempdb?driver=ODBC+Driver+18+for+SQL+Server"
            >>> with SQLServerSaver.from_conn_string(DB_URI) as memory:
            >>>     config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            >>>     checkpoint = {"ts": "2024-05-04T06:32:42.235444+00:00", "id": "1ef4f797-8335-6428-8001-8a1503f9b875", "channel_values": {"key": "value"}}
            >>>     saved_config = memory.put(config, checkpoint, {"source": "input", "step": 1, "writes": {"key": "value"}}, {})
            >>> print(saved_config)
            {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef4f797-8335-6428-8001-8a1503f9b875'}}
        """
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

        with self._cursor() as cur:
            blobs = self._dump_blobs(
                thread_id,
                checkpoint_ns,
                copy.pop("channel_values"),  # type: ignore[misc]
                new_versions,
            )

            if blobs:
                cur.executemany(
                    UPSERT_CHECKPOINT_BLOBS_SQL.replace("${SCHEMA}", self.schema), blobs
                )

            cur.execute(
                UPSERT_CHECKPOINTS_SQL.replace("${SCHEMA}", self.schema),
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_id,
                    json.dumps(copy),
                    json.dumps(get_checkpoint_metadata(config, metadata)),
                ),
            )

        return next_config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the SQL Server database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
        """
        query = (
            UPSERT_CHECKPOINT_WRITES_SQL.replace("${SCHEMA}", self.schema)
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else INSERT_CHECKPOINT_WRITES_SQL.replace("${SCHEMA}", self.schema)
        )

        print(str(all(w[0] in WRITES_IDX_MAP for w in writes)))
        with self._cursor() as cur:
            cur.executemany(
                query,
                self._dump_writes(
                    config["configurable"]["thread_id"],
                    config["configurable"]["checkpoint_ns"],
                    config["configurable"]["checkpoint_id"],
                    task_id,
                    task_path,
                    writes,
                ),
            )

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.

        Returns:
            None
        """
        with self._cursor() as cur:
            cur.execute(
                DELETE_THREAD_CHECKPOINTS.replace("${SCHEMA}", self.schema),
                (str(thread_id),),
            )
            cur.execute(
                DELETE_THREAD_CHECKPOINT_BLOBS.replace("${SCHEMA}", self.schema),
                (str(thread_id),),
            )
            cur.execute(
                DELETE_THREAD_CHECKPOINT_WRITES.replace("${SCHEMA}", self.schema),
                (str(thread_id),),
            )

    def _load_checkpoint_tuple(self, value: Row) -> CheckpointTuple:
        """
        Convert a database row into a CheckpointTuple object.

        Args:
            value: A row from the database containing checkpoint data.

        Returns:
            CheckpointTuple: A structured representation of the checkpoint,
            including its configuration, metadata, parent checkpoint (if any),
            and pending writes.
        """
        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": value["thread_id"],
                    "checkpoint_ns": value["checkpoint_ns"],
                    "checkpoint_id": value["checkpoint_id"],
                }
            },
            {
                **value["checkpoint"],
                "channel_values": self._load_blobs(value["channel_values"]),
            },
            value["metadata"],
            (
                {
                    "configurable": {
                        "thread_id": value["thread_id"],
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["parent_checkpoint_id"],
                    }
                }
                if value["parent_checkpoint_id"]
                else None
            ),
            self._load_writes(value["pending_writes"]),
        )


__all__ = ["SQLServerSaver"]
