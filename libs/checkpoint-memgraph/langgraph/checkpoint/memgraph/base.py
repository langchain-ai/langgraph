from __future__ import annotations

import base64
import random
from collections.abc import Sequence
from typing import Any, Optional, cast

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS

MetadataInput = Optional[dict[str, Any]]

MEMGRAPH_MIGRATIONS = [
    # v0
    "CREATE CONSTRAINT ON (m:Migration) ASSERT m.v IS UNIQUE;",
    # v1‑v3
    "// No-op for composite constraint on Checkpoint.",
    "// No-op for composite constraint on Blob.",
    "// No-op for composite constraint on Write.",
    # v4
    "// In Memgraph, properties are nullable by default.",
    # v5
    "// No-op migration.",
    # v6‑v8
    "CREATE INDEX ON :Checkpoint(thread_id);",
    "CREATE INDEX ON :Blob(thread_id);",
    "CREATE INDEX ON :Write(thread_id);",
    # v9
    "// The task_path property will be added to Write nodes on creation.",
]

SELECT_CYPHER = """
MATCH (c:Checkpoint)
// Aggregate channel values
WITH c,
     [k IN (CASE WHEN c.checkpoint.channel_versions IS NULL
                 THEN []
                 ELSE keys(c.checkpoint.channel_versions) END)
      | {channel: k, version: c.checkpoint.channel_versions[k]}] AS versions
UNWIND (CASE WHEN size(versions) > 0
             THEN versions
             ELSE [{__dummy__: true}] END) AS v_data
OPTIONAL MATCH (bl:Blob {
    thread_id: c.thread_id,
    checkpoint_ns: c.checkpoint_ns,
    channel: v_data.channel,
    version: v_data.version
})
WITH c,
     collect(DISTINCT CASE WHEN bl IS NULL
                           THEN NULL
                           ELSE [bl.channel, bl.type, bl.blob] END) AS channel_values_raw
WITH c, [cv IN channel_values_raw WHERE cv IS NOT NULL] AS channel_values
// Aggregate pending writes
OPTIONAL MATCH (c)-[:HAS_WRITE]->(cw:Write)
WITH c, channel_values, cw
ORDER BY cw.task_id, cw.idx
WITH c,
     channel_values,
     collect(CASE WHEN cw IS NULL
                  THEN NULL
                  ELSE [cw.task_id, cw.channel, cw.type, cw.blob] END) AS pending_writes_raw
RETURN
    c.thread_id            AS thread_id,
    c.checkpoint           AS checkpoint,
    c.checkpoint_ns        AS checkpoint_ns,
    c.checkpoint_id        AS checkpoint_id,
    c.parent_checkpoint_id AS parent_checkpoint_id,
    c.metadata             AS metadata,
    channel_values,
    [w IN pending_writes_raw WHERE w IS NOT NULL] AS pending_writes
"""

SELECT_PENDING_SENDS_CYPHER = """
MATCH (w:Write)
WHERE w.thread_id     = $thread_id
  AND w.checkpoint_id IN $checkpoint_ids
  AND w.channel       = $tasks_channel
WITH w.checkpoint_id AS checkpoint_id, w
ORDER BY w.task_path, w.task_id, w.idx
RETURN checkpoint_id, collect([w.type, w.blob]) AS sends
"""

UPSERT_CHECKPOINT_BLOBS_CYPHER = """
UNWIND $blobs AS props
MERGE (b:Blob {
    thread_id:     props.thread_id,
    checkpoint_ns: props.checkpoint_ns,
    channel:       props.channel,
    version:       props.version
})
ON CREATE SET
    b.type = props.type,
    b.blob = props.blob
"""

UPSERT_CHECKPOINTS_CYPHER = """
MERGE (c:Checkpoint {
    thread_id:     $thread_id,
    checkpoint_ns: $checkpoint_ns,
    checkpoint_id: $checkpoint_id
})
ON CREATE SET
    c.parent_checkpoint_id = $parent_checkpoint_id,
    c.checkpoint           = $checkpoint,
    c.metadata             = $metadata
ON MATCH SET
    c.checkpoint = $checkpoint,
    c.metadata   = $metadata
"""

UPSERT_CHECKPOINT_WRITES_CYPHER = """
UNWIND $writes AS props
MERGE (c:Checkpoint {
    thread_id:     props.thread_id,
    checkpoint_ns: props.checkpoint_ns,
    checkpoint_id: props.checkpoint_id
})
MERGE (w:Write {
    thread_id:     props.thread_id,
    checkpoint_ns: props.checkpoint_ns,
    checkpoint_id: props.checkpoint_id,
    task_id:       props.task_id,
    idx:           props.idx
})
ON CREATE SET
    w.task_path = props.task_path,
    w.channel   = props.channel,
    w.type      = props.type,
    w.blob      = props.blob
ON MATCH SET
    w.channel = props.channel,
    w.type    = props.type,
    w.blob    = props.blob
MERGE (c)-[:HAS_WRITE]->(w)
"""

INSERT_CHECKPOINT_WRITES_CYPHER = """
UNWIND $writes AS props
MERGE (c:Checkpoint {
    thread_id:     props.thread_id,
    checkpoint_ns: props.checkpoint_ns,
    checkpoint_id: props.checkpoint_id
})
CREATE (w:Write {
    thread_id:     props.thread_id,
    checkpoint_ns: props.checkpoint_ns,
    checkpoint_id: props.checkpoint_id,
    task_id:       props.task_id,
    task_path:     props.task_path,
    idx:           props.idx,
    channel:       props.channel,
    type:          props.type,
    blob:          props.blob
})
MERGE (c)-[:HAS_WRITE]->(w)
"""


class BaseMemgraphSaver(BaseCheckpointSaver[str]):
    """A base class for saving LangGraph checkpoints to a Memgraph database.

    This class provides the core logic for interacting with Memgraph, serving as a
    foundation for both synchronous and asynchronous saver implementations. It defines
    the necessary Cypher queries for creating, reading, and updating checkpoint data,
    including the main checkpoint information, associated blobs (channel values),
    and pending writes.

    Attributes:
        SELECT_CYPHER (str): Cypher query to select checkpoint data.
        SELECT_PENDING_SENDS_CYPHER (str): Cypher query to select pending sends.
        MIGRATIONS (list[str]): A list of Cypher queries for database schema migrations.
        UPSERT_CHECKPOINT_BLOBS_CYPHER (str): Cypher query to upsert blob data.
        UPSERT_CHECKPOINTS_CYPHER (str): Cypher query to upsert checkpoint data.
        UPSERT_CHECKPOINT_WRITES_CYPHER (str): Cypher query to upsert write data.
        INSERT_CHECKPOINT_WRITES_CYPHER (str): Cypher query to insert write data.
        supports_pipeline (bool): Flag indicating if the saver supports pipelined operations.
    """

    SELECT_CYPHER = SELECT_CYPHER
    SELECT_PENDING_SENDS_CYPHER = SELECT_PENDING_SENDS_CYPHER
    MIGRATIONS = MEMGRAPH_MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_CYPHER = UPSERT_CHECKPOINT_BLOBS_CYPHER
    UPSERT_CHECKPOINTS_CYPHER = UPSERT_CHECKPOINTS_CYPHER
    UPSERT_CHECKPOINT_WRITES_CYPHER = UPSERT_CHECKPOINT_WRITES_CYPHER
    INSERT_CHECKPOINT_WRITES_CYPHER = INSERT_CHECKPOINT_WRITES_CYPHER

    supports_pipeline: bool

    @staticmethod
    def _encode_blob(value: Any) -> Any:
        """Return a value safe for Memgraph property storage."""
        if isinstance(value, (bytes, bytearray)):
            return "b64:" + base64.b64encode(value).decode("ascii")
        return value

    @staticmethod
    def _decode_blob(value: Any) -> Any:
        """Decode value previously encoded by `_encode_blob`."""
        if isinstance(value, str) and value.startswith("b64:"):
            try:
                return base64.b64decode(value[4:])
            except Exception:
                # Corrupted? fall through and return original string
                return value
        return value

    def _migrate_pending_sends(
        self,
        pending_sends: list[tuple[str, bytes | str]],
        checkpoint: dict[str, Any],
        channel_values: list[tuple[str, str, bytes | str]],
    ) -> None:
        """Move legacy pending sends into checkpoint.channel_values."""
        if not pending_sends:
            return
        # Decode blobs before deserializing
        deserialized_sends = [
            self.serde.loads_typed((type_, self._decode_blob(blob)))
            for type_, blob in pending_sends
        ]
        # Re-serialize the entire list of values for the new channel
        enc, blob = self.serde.dumps_typed(deserialized_sends)
        blob = self._encode_blob(blob)
        channel_values.append((TASKS, enc, blob))
        # Assign/bump version for the new channel
        if "channel_versions" not in checkpoint:
            checkpoint["channel_versions"] = {}
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    def _load_blobs(
        self,
        blob_values: list[tuple[str, str, bytes | str]],
    ) -> dict[str, Any]:
        """Decode rows from the Blob nodes table."""
        if not blob_values:
            return {}
        return {
            channel: self.serde.loads_typed((type_, self._decode_blob(blob)))
            for channel, type_, blob in blob_values
            if type_ != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[dict[str, Any]]:
        """Prepare blob rows for UNWIND insertion."""
        if not versions:
            return []
        blobs: list[dict[str, Any]] = []
        for channel, version in versions.items():
            type_, blob = (
                self.serde.dumps_typed(values[channel])
                if channel in values
                else ("empty", None)
            )
            blob = self._encode_blob(blob)
            blobs.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "channel": channel,
                    "version": cast(str, version),
                    "type": type_,
                    "blob": blob,
                }
            )
        return blobs

    def _load_writes(
        self,
        writes: list[tuple[str, str, str, bytes | str]],
    ) -> list[tuple[str, str, Any]]:
        """Decode Write node rows."""
        return (
            [
                (
                    task_id,
                    channel,
                    self.serde.loads_typed((type_, self._decode_blob(blob))),
                )
                for task_id, channel, type_, blob in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prepare Write nodes for batch insertion/upsert."""
        dumped: list[dict[str, Any]] = []
        for idx, (channel, value) in enumerate(writes):
            type_, blob = self.serde.dumps_typed(value)
            blob = self._encode_blob(blob)
            dumped.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "task_path": task_path,
                    "idx": WRITES_IDX_MAP.get(channel, idx),
                    "channel": channel,
                    "type": type_,
                    "blob": blob,
                }
            )
        return dumped

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Generate a new, monotonically increasing version string.

        This method is used to create version strings that are always greater than
        the previous ones. The version string consists of a zero-padded integer
        and a random float, separated by a dot.

        Args:
            current: The current version string (e.g., "1.234"). If None, the
                major version will be 1. Only the part before the first dot is
                considered for incrementing.
            channel: Unused in this implementation.

        Returns:
            A new version string, guaranteed to be lexicographically greater
            than the `current` one.
        """
        if current is None:
            current_major = 0
        elif isinstance(current, int):
            current_major = current
        else:
            current_major = int(current.split(".")[0])
        next_major = current_major + 1
        next_rand = random.random()
        return f"{next_major:032}.{next_rand:016}"

    def _search_where_and_params(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Compose Cypher WHERE clause & params from user filters."""
        wheres: list[str] = []
        params: dict[str, Any] = {}
        if config:
            if thread_id := config["configurable"].get("thread_id"):
                wheres.append("c.thread_id = $thread_id")
                params["thread_id"] = thread_id
            if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
                wheres.append("c.checkpoint_ns = $checkpoint_ns")
                params["checkpoint_ns"] = checkpoint_ns
            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("c.checkpoint_id = $checkpoint_id")
                params["checkpoint_id"] = checkpoint_id
        if filter:
            for key, value in filter.items():
                param_key = f"metadata_{key}"
                wheres.append(f"c.metadata.{key} = ${param_key}")
                params[param_key] = value
        if before is not None:
            if before_id := get_checkpoint_id(before):
                wheres.append("c.checkpoint_id < $before_checkpoint_id")
                params["before_checkpoint_id"] = before_id
        return ("WHERE " + " AND ".join(wheres)) if wheres else "", params
