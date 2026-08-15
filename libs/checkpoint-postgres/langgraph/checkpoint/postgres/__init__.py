from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
    _DeltaSnapshot,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import TASKS
from psycopg import Capabilities, Connection, Cursor, Pipeline
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from langgraph.checkpoint.postgres import _internal
from langgraph.checkpoint.postgres.base import (
    _DELTA_PAGE_SIZE,
    BasePostgresSaver,
    _build_delta_stage1_sql,
    _build_delta_stage2_sql,
    _DeltaStage2Row,
)
from langgraph.checkpoint.postgres.shallow import ShallowPostgresSaver

from langgraph.checkpoint.postgres import _internal

Conn = _internal.Conn  # For backward compatibility


__all__ = ["PostgresSaver", "BasePostgresSaver", "ShallowPostgresSaver", "Conn"]


def _has_delta_channel(checkpoint: Checkpoint) -> bool:
    """Check if a checkpoint uses DeltaChannel (has _DeltaSnapshot values)."""
    channel_values = checkpoint.get("channel_values", {})
    return any(isinstance(v, _DeltaSnapshot) for v in channel_values.values())


class PostgresSaver(BasePostgresSaver):
    """Checkpointer that stores checkpoints in a Postgres database."""

    lock: threading.Lock

    def __init__(
        self,
        conn: _internal.Conn,
        pipe: Pipeline | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        if isinstance(conn, ConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single Connection, not ConnectionPool."
            )

        self.conn = conn
        self.pipe = pipe
        self.lock = threading.Lock()
        self.supports_pipeline = Capabilities().has_pipeline()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls, conn_string: str, *, pipeline: bool = False
    ) -> Iterator[PostgresSaver]:
        """Create a new PostgresSaver instance from a connection string.

        Args:
            conn_string: The Postgres connection info string.
            pipeline: whether to use Pipeline

        Returns:
            PostgresSaver: A new PostgresSaver instance.
        """
        with Connection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            if pipeline:
                with conn.pipeline() as pipe:
                    yield cls(conn, pipe)
            else:
                yield cls(conn)

    def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time checkpointer is used.
        """
        with self._cursor() as cur:
            cur.execute(self.MIGRATIONS[0])
            results = cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
            )
            row = results.fetchone()
            if row is None:
                version = -1
            else:
                version = row["v"]
            for i in range(version + 1, len(self.MIGRATIONS)):
                cur.execute(self.MIGRATIONS[i])

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the Postgres database.

        This method saves a checkpoint to the Postgres database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop("checkpoint_id", None)

        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        # inline primitive values in checkpoint table
        # others are stored in blobs table
        blob_values = {}
        for k, v in checkpoint["channel_values"].items():
            if isinstance(v, _DeltaSnapshot):
                blob_values[k] = copy["channel_values"].pop(k)
                copy["channel_values"][k] = True
            elif v is None or isinstance(v, (str, int, float, bool)):
                pass
            else:
                blob_values[k] = copy["channel_values"].pop(k)

        with self._cursor(pipeline=True) as cur:
            if blob_versions := {
                k: v for k, v in new_versions.items() if k in blob_values
            }:
                cur.executemany(
                    self.UPSERT_CHECKPOINT_BLOBS_SQL,
                    self._dump_blobs(
                        thread_id,
                        checkpoint_ns,
                        blob_values,
                        blob_versions,
                    ),
                )
            cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    Jsonb(checkpoint),
                    Jsonb(metadata),
                    Jsonb(new_versions),
                ),
            )
            if self.pipe:
                self.pipe.sync()

        return next_config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save intermediate writes associated with a checkpoint to the Postgres database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task.
        """
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        with self._cursor(pipeline=True) as cur:
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
        with self._cursor(pipeline=True) as cur:
            cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (str(thread_id),),
            )
            cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                (str(thread_id),),
            )
            cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (str(thread_id),),
            )

    def prune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        """Prune checkpoints for given threads.

        Args:
            thread_ids: The thread IDs to prune.
            strategy: "keep_latest" keeps only the latest checkpoint per
                thread+namespace. "delete" removes everything.

        !!! warning "DeltaChannel"
            Threads using DeltaChannel are not pruned to avoid corrupting
            delta channel history. If any requested thread contains DeltaChannel
            state, the operation aborts and raises an error.

        Raises:
            ValueError: If an invalid strategy is provided.
            RuntimeError: If any thread contains DeltaChannel state and strategy
                is "keep_latest" (which cannot safely preserve DeltaChannel history).
        """
        if not thread_ids:
            return

        if strategy not in ("delete", "keep_latest"):
            raise ValueError(f"Invalid pruning strategy: {strategy}")

        # First, check for DeltaChannel state in any of the threads
        # if using keep_latest strategy
        if strategy == "keep_latest":
            with self._cursor() as cur:
                for thread_id in thread_ids:
                    cur.execute(
                        """
                        SELECT 1 FROM checkpoints
                        WHERE thread_id = %s
                        AND checkpoint_id IN (
                            SELECT checkpoint_id FROM checkpoints
                            WHERE thread_id = %s
                            AND channel_values @> '{"_delta_snapshot": true}'
                        )
                        LIMIT 1
                        """,
                        (str(thread_id), str(thread_id)),
                    )
                    if cur.fetchone():
                        raise RuntimeError(
                            f"Thread {thread_id} contains DeltaChannel state. "
                            "keep_latest pruning is not supported for DeltaChannel threads. "
                            "Use 'delete' strategy or implement DeltaChannel-aware pruning."
                        )

        with self._cursor(pipeline=True) as cur:
            if strategy == "delete":
                # Delete all checkpoints, writes, and blobs for the threads
                cur.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ANY(%s)",
                    (list(thread_ids),),
                )
                cur.execute(
                    "DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s)",
                    (list(thread_ids),),
                )
                cur.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = ANY(%s)",
                    (list(thread_ids),),
                )
            elif strategy == "keep_latest":
                # Delete non-latest checkpoints, preserving writes and blobs for latest
                for thread_id in thread_ids:
                    # Get the latest checkpoint_id for each namespace
                    cur.execute(
                        """
                        DELETE FROM checkpoints
                        WHERE thread_id = %s
                        AND (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
                            SELECT DISTINCT ON (thread_id, checkpoint_ns)
                                thread_id, checkpoint_ns, checkpoint_id
                            FROM checkpoints
                            WHERE thread_id = %s
                            ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC
                        )
                        """,
                        (str(thread_id), str(thread_id)),
                    )
                    # Delete writes for removed checkpoints
                    cur.execute(
                        """
                        DELETE FROM checkpoint_writes
                        WHERE thread_id = %s
                        AND (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
                            SELECT thread_id, checkpoint_ns, checkpoint_id
                            FROM checkpoints
                            WHERE thread_id = %s
                        )
                        """,
                        (str(thread_id), str(thread_id)),
                    )
                    # Clean up orphaned blobs
                    cur.execute(
                        """
                        DELETE FROM checkpoint_blobs
                        WHERE thread_id = %s
                        AND NOT EXISTS (
                            SELECT 1 FROM checkpoints c
                            WHERE c.thread_id = checkpoint_blobs.thread_id
                            AND c.checkpoint_ns = checkpoint_blobs.checkpoint_ns
                        )
                        """,
                        (str(thread_id),),
                    )

    @contextmanager
    def _cursor(self, *, pipeline: bool = False) -> Iterator[Cursor[DictRow]]:
        """Create a database cursor as a context manager.

        Args:
            pipeline: whether to use pipeline for the DB operations inside the context manager.
                Will be applied regardless of whether the PostgresSaver instance was initialized with a pipeline.
                If pipeline mode is not supported, will fall back to using transaction context manager.
        """
        with self.lock, _internal.get_connection(self.conn) as conn:
            if self.pipe:
                # a connection in pipeline mode can be used concurrently
                # in multiple threads/coroutines, but only one cursor can be
                # used at a time
                try:
                    with conn.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                finally:
                    if pipeline:
                        self.pipe.sync()
            elif pipeline:
                # a connection not in pipeline mode can only be used by one
                # thread/coroutine at a time, so we acquire a lock
                if self.supports_pipeline:
                    with (
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    # Use connection's transaction context manager when pipeline mode not supported
                    with (
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """Fast-path override of `BaseCheckpointSaver.get_delta_channel_history`.

        Two-stage query, both stages cover ALL requested channels:

        * Stage 1 (paged): dynamic SELECT over `checkpoints` with K parallel
          JSONB key lookups (one column pair per channel) — no subquery, no
          aggregation. Pages newest-first by `checkpoint_id` with a cursor;
          page size is `_DELTA_PAGE_SIZE`. Stops paging when every channel
          has found its seed or the chain is exhausted.

        * Stage 2 (per-channel UNION ALL): one branch per channel reading
          `checkpoint_writes` filtered to that channel's specific
          `chain_cids`, plus one branch per channel that has a seed reading
          `checkpoint_blobs` for that channel + version. Avoids the
          over-fetch of a single `channel = ANY(channels)` filter when
          channels have different chain depths.
        """
        if not channels:
            return {}
        channels = list(channels)
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id is None:
            target = self.get_tuple(config)
            if target is None:
                return {ch: {"writes": []} for ch in channels}
            checkpoint_id = target.config["configurable"]["checkpoint_id"]

        # Stage 1: paged K-JSONB-lookup scan, walking the parent chain in
        # Python after each page. Stops as soon as every channel has its seed.
        stage1_sql = _build_delta_stage1_sql(channels, paged=True)
        parent_of: dict[str, str | None] = {}
        ver_by_i_by_cid: list[dict[str, str | None]] = [{} for _ in channels]
        hb_by_i_by_cid: list[dict[str, bool]] = [{} for _ in channels]
        inline_by_i_by_cid: list[dict[str, Any]] = [{} for _ in channels]
        chain_by_ch: dict[str, list[str]] = {ch: [] for ch in channels}
        seed_ver_by_ch: dict[str, str | None] = {ch: None for ch in channels}
        seed_inline_by_ch: dict[str, Any] = {}
        walk_cursor_by_ch: dict[str, str | None] = {}
        seeded: set[str] = set()
        cursor: str | None = None

        with self._cursor() as cur:
            while True:
                stage1_params: list[Any] = []
                for ch in channels:
                    # ver_i, blob channel, blob version, inline_i
                    stage1_params.extend([ch, ch, ch, ch])
                stage1_params.extend(
                    [thread_id, checkpoint_ns, cursor, cursor, _DELTA_PAGE_SIZE]
                )
                cur.execute(stage1_sql, stage1_params)
                rows = cur.fetchall()
                if not rows:
                    break
                cursor = self._ingest_stage1_page(
                    rows, channels, parent_of, ver_by_i_by_cid,
                    hb_by_i_by_cid, inline_by_i_by_cid
                )
                self._try_advance_walks(
                    checkpoint_id, channels, parent_of, ver_by_i_by_cid,
                    hb_by_i_by_cid, inline_by_i_by_cid, chain_by_ch,
                    seed_ver_by_ch, seed_inline_by_ch, walk_cursor_by_ch,
                    seeded
                )
                if len(seeded) == len(channels):
                    break
            else:
                # Exhausted the chain without finding all seeds
                pass

        # Stage 2: collect writes + blobs for the chain_cids
        stage2_sql = _build_delta_stage2_sql(channels, chain_by_ch)
        stage2_params: list[Any] = []
        for ch in channels:
            stage2_params.append(ch)
        stage2_params.append(list(chain_by_ch.keys()))
        stage2_params.append(thread_id)
        stage2_params.append(checkpoint_ns)

        with self._cursor() as cur:
            cur.execute(stage2_sql, stage2_params)
            stage2_rows = cur.fetchall()

        return self._build_delta_channels_writes_history(
            channels=channels,
            chain_by_ch=chain_by_ch,
            seed_ver_by_ch=seed_ver_by_ch,
            seed_inline_by_ch=seed_inline_by_ch,
            stage2_rows=cast("list[_DeltaStage2Row]", stage2_rows),
        )

    def _load_checkpoint_tuple(self, value: DictRow) -> CheckpointTuple:
        """Convert a database row into a CheckpointTuple object.

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
                "channel_values": {
                    **(value["checkpoint"].get("channel_values") or {}),
                    **self._load_blobs(value["channel_values"]),
                },
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


__all__ = ["PostgresSaver", "BasePostgresSaver", "ShallowPostgresSaver", "Conn"]