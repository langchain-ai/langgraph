from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from neo4j import Driver, GraphDatabase, Transaction

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
from langgraph.checkpoint.serde.types import TASKS

from .base import BaseMemgraphSaver


class MemgraphSaver(BaseMemgraphSaver):
    """A checkpointer that stores checkpoints in a Memgraph database.

    This class provides a checkpointer that uses a Memgraph graph database for
    persistent storage of langgraph checkpoints. It allows for saving, listing, and
    retrieving checkpoints, enabling the resumption of graph execution from a known
    state.

    It requires a `neo4j` driver instance to connect to the Memgraph database.
    Before its first use, the `setup()` method should be called to ensure the
    database schema (indexes, constraints) is correctly initialized.
    """

    driver: Driver
    lock: threading.Lock

    def __init__(
        self,
        driver: Driver,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """
        Initialize the Memgraph saver.

        Args:
            driver: The neo4j.Driver instance to connect to the database.
            serde: The serializer to use for checkpoint data. Defaults to a JSON serializer.
        """
        super().__init__(serde=serde)
        self.driver = driver
        self.lock = threading.Lock()

    @classmethod
    @contextmanager
    def from_conn_string(cls, conn_string: str) -> Iterator[MemgraphSaver]:
        """
        Create a new MemgraphSaver instance from a connection string.

        Args:
            conn_string: The Memgraph connection URI (e.g., "bolt://localhost:7687").

        Yields:
            A MemgraphSaver instance.
        """
        with GraphDatabase.driver(conn_string) as driver:
            yield cls(driver)

    def setup(self) -> None:
        """
        Set up the checkpoint database.

        This method creates the necessary constraints and indexes in Memgraph if they
        don't already exist and runs any pending database migrations. It should be
        called once before the checkpointer is used.
        """
        with self.driver.session() as session:
            try:
                result = session.run(
                    "MATCH (m:Migration) RETURN m.v AS v ORDER BY m.v DESC LIMIT 1"
                )
                version = result.single(strict=True)["v"]
            except Exception:
                version = -1
            for v, migration in enumerate(self.MIGRATIONS):
                if v > version:
                    if not migration.startswith("//"):  # Skip no-op comments
                        session.run(migration)
                    session.run("MERGE (m:Migration {v: $v})", v=v)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """
        List checkpoints from the database, ordered from newest to oldest.

        Args:
            config: The config to filter by (e.g., `{"configurable": {"thread_id": "..."}}`).
            filter: Additional metadata attributes to filter by.
            before: An optional config to list checkpoints created before the one specified.
            limit: The maximum number of checkpoints to return.

        Yields:
            An iterator of checkpoint tuples.
        """
        where_clause, params = self._search_where_and_params(config, filter, before)
        query = self.SELECT_CYPHER.replace(
            "MATCH (c:Checkpoint)", f"MATCH (c:Checkpoint) {where_clause}"
        )
        query += " ORDER BY c.checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        with self._session() as tx:
            records = [dict(r) for r in tx.run(query, params)]
            if not records:
                return
            to_migrate = [
                r
                for r in records
                if r["checkpoint"].get("v", 0) < 4 and r["parent_checkpoint_id"]
            ]
            if to_migrate:
                thread_id = records[0]["thread_id"]
                parent_ids = list({r["parent_checkpoint_id"] for r in to_migrate})
                sends_records = list(
                    tx.run(
                        self.SELECT_PENDING_SENDS_CYPHER,
                        {
                            "thread_id": thread_id,
                            "checkpoint_ids": parent_ids,
                            "tasks_channel": TASKS,
                        },
                    )
                )
                grouped_by_parent = defaultdict(list)
                for record in to_migrate:
                    grouped_by_parent[record["parent_checkpoint_id"]].append(record)
                for sends_record in sends_records:
                    parent_id = sends_record["checkpoint_id"]
                    for record in grouped_by_parent[parent_id]:
                        if record["channel_values"] is None:
                            record["channel_values"] = []
                        self._migrate_pending_sends(
                            sends_record["sends"],
                            record["checkpoint"],
                            record["channel_values"],
                        )
            for record in records:
                yield self._load_checkpoint_tuple(record)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """
        Get a specific checkpoint tuple from the database.

        If `checkpoint_id` is present in the config, it will be retrieved.
        Otherwise, the latest checkpoint for the given `thread_id` is returned.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            The checkpoint tuple, or None if not found.
        """
        where_clause, params = self._search_where_and_params(config, None, None)
        query = self.SELECT_CYPHER.replace(
            "MATCH (c:Checkpoint)", f"MATCH (c:Checkpoint) {where_clause}"
        )
        if "checkpoint_id" not in config["configurable"]:
            query += " ORDER BY c.checkpoint_id DESC LIMIT 1"
        with self._session() as tx:
            result = tx.run(query, params).single()
            if result is None:
                return None
            record = dict(result)
            # Perform pending sends migration if necessary
            if record["checkpoint"].get("v", 0) < 4 and record["parent_checkpoint_id"]:
                sends_result = tx.run(
                    self.SELECT_PENDING_SENDS_CYPHER,
                    {
                        "thread_id": config["configurable"]["thread_id"],
                        "checkpoint_ids": [record["parent_checkpoint_id"]],
                        "tasks_channel": TASKS,
                    },
                ).single()
                if sends_result:
                    if record.get("channel_values") is None:
                        record["channel_values"] = []
                    self._migrate_pending_sends(
                        sends_result["sends"],
                        record["checkpoint"],
                        record["channel_values"],
                    )
            return self._load_checkpoint_tuple(record)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Save a checkpoint to the database.

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            The updated runnable config containing the new checkpoint ID.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = get_checkpoint_id(config)
        checkpoint_blobs = self._dump_blobs(
            thread_id,
            checkpoint_ns,
            checkpoint["channel_values"],
            new_versions,
        )
        with self._session() as tx:
            # Batch upsert blobs using UNWIND
            if checkpoint_blobs:
                tx.run(self.UPSERT_CHECKPOINT_BLOBS_CYPHER, blobs=checkpoint_blobs)
            # Upsert checkpoint
            tx.run(
                self.UPSERT_CHECKPOINTS_CYPHER,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint["id"],
                parent_checkpoint_id=parent_checkpoint_id,
                checkpoint=checkpoint,
                metadata=get_checkpoint_metadata(config, metadata),
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """
        Store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration of the related checkpoint.
            writes: A list of writes to store, each a (channel, value) tuple.
            task_id: The ID of the task that produced the writes.
            task_path: The path of the task.
        """
        checkpoint_writes = self._dump_writes(
            config["configurable"]["thread_id"],
            config["configurable"].get("checkpoint_ns", ""),
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        if not checkpoint_writes:
            return
        upsert_mode = all(w[0] in WRITES_IDX_MAP for w in writes)
        query_template = (
            self.UPSERT_CHECKPOINT_WRITES_CYPHER
            if upsert_mode
            else self.INSERT_CHECKPOINT_WRITES_CYPHER
        )
        with self._session() as tx:
            tx.run(query_template, writes=checkpoint_writes)

    def delete_thread(self, thread_id: str) -> None:
        """
        Delete all data associated with a specific thread ID.

        Args:
            thread_id: The ID of the thread to delete.
        """
        with self._session() as tx:
            tx.run(
                """
                MATCH (c:Checkpoint {thread_id: $thread_id})
                OPTIONAL MATCH (c)-[:HAS_WRITE]->(w:Write)
                OPTIONAL MATCH (b:Blob {thread_id: $thread_id})
                DETACH DELETE c, w, b
                """,
                thread_id=thread_id,
            )

    @contextmanager
    def _session(self) -> Iterator[Transaction]:
        """A context manager for acquiring a Neo4j session and transaction."""
        # The neo4j.Driver is thread-safe
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                yield tx

    def _load_checkpoint_tuple(self, record: dict[str, Any]) -> CheckpointTuple:
        """
        Convert a database record into a CheckpointTuple.

        Args:
            record: A dictionary-like object representing a database row.

        Returns:
            A structured CheckpointTuple.
        """
        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": record["thread_id"],
                    "checkpoint_ns": record["checkpoint_ns"],
                    "checkpoint_id": record["checkpoint_id"],
                }
            },
            {
                **record["checkpoint"],
                "channel_values": self._load_blobs(record.get("channel_values")),
            },
            record["metadata"],
            (
                {
                    "configurable": {
                        "thread_id": record["thread_id"],
                        "checkpoint_ns": record["checkpoint_ns"],
                        "checkpoint_id": record["parent_checkpoint_id"],
                    }
                }
                if record["parent_checkpoint_id"]
                else None
            ),
            self._load_writes(record.get("pending_writes") or []),
        )


__all__ = ["MemgraphSaver", "BaseMemgraphSaver"]
