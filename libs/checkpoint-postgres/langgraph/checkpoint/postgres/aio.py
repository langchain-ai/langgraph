from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from psycopg import (
    AsyncConnection,
    AsyncCursor,
    AsyncPipeline,
    Capabilities,
    InterfaceError,
    OperationalError,
)
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.postgres import _ainternal
from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.checkpoint.postgres.shallow import AsyncShallowPostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol

Conn = _ainternal.Conn  # For backward compatibility


class AsyncPostgresSaver(BasePostgresSaver):
    """Asynchronous checkpointer that stores checkpoints in a Postgres database."""

    lock: asyncio.Lock

    def __init__(
        self,
        conn: _ainternal.Conn,
        pipe: AsyncPipeline | None = None,
        serde: SerializerProtocol | None = None,
        *,
        retry_max_attempts: int = 5,
        retry_initial_backoff: float = 0.25,  # seconds
        retry_max_backoff: float = 3.0,
        reconnect_cb: Callable[[], Awaitable[AsyncConnection[dict[str, Any]]]]
        | None = None,
    ) -> None:
        super().__init__(serde=serde)
        if isinstance(conn, AsyncConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single AsyncConnection, not AsyncConnectionPool."
            )

        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.supports_pipeline = Capabilities().has_pipeline()

        # Retry config
        self.retry_max_attempts = retry_max_attempts
        self.retry_initial_backoff = retry_initial_backoff
        self.retry_max_backoff = retry_max_backoff

        # Reconnect factory for single-connection mode
        self._reconnect_cb = reconnect_cb

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        serde: SerializerProtocol | None = None,
        retry_max_attempts: int = 5,
        retry_initial_backoff: float = 0.25,
        retry_max_backoff: float = 3.0,
    ) -> AsyncIterator[AsyncPostgresSaver]:
        """Create a new AsyncPostgresSaver instance from a connection string."""

        async def _mkconn() -> AsyncConnection[dict[str, Any]]:
            return await AsyncConnection.connect(
                conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
            )

        async with await _mkconn() as conn:
            if pipeline:
                async with conn.pipeline() as pipe:
                    yield cls(
                        conn=conn,
                        pipe=pipe,
                        serde=serde,
                        retry_max_attempts=retry_max_attempts,
                        retry_initial_backoff=retry_initial_backoff,
                        retry_max_backoff=retry_max_backoff,
                        reconnect_cb=_mkconn,
                    )
            else:
                yield cls(
                    conn=conn,
                    serde=serde,
                    retry_max_attempts=retry_max_attempts,
                    retry_initial_backoff=retry_initial_backoff,
                    retry_max_backoff=retry_max_backoff,
                    reconnect_cb=_mkconn,
                )

    async def _ensure_connection_open(self) -> None:
        """If we are in single-connection mode and it's closed, recreate it."""
        if isinstance(self.conn, AsyncConnection):
            closed = bool(getattr(self.conn, "closed", False))
            if closed:
                if not self._reconnect_cb:
                    raise OperationalError(
                        "connection is closed and no reconnect callback provided"
                    )
                # Replace the dead connection; pipeline context becomes invalid
                self.conn = await self._reconnect_cb()
                self.pipe = None  # switch to per-op pipeline/transaction

    async def _reconnect_if_single_conn(self) -> None:
        """Force reconnection for single AsyncConnection setups."""
        if isinstance(self.conn, AsyncConnection) and self._reconnect_cb:
            try:
                self.conn = await self._reconnect_cb()
            finally:
                # Any previous long-lived pipeline is no longer valid
                self.pipe = None

    def _is_retryable_error(self, exc: BaseException) -> bool:
        # Never retry cancellations
        if isinstance(exc, asyncio.CancelledError):
            return False

        # psycopg3 transient client-side failures
        if isinstance(
            exc, (OperationalError, InterfaceError, TimeoutError, ConnectionResetError)
        ):
            msg = str(exc).lower()
            retry_signals = (
                "ssl connection has been closed unexpectedly",
                "could not receive data from server",
                "server closed the connection unexpectedly",
                "terminating connection due to administrator command",
                "connection not open",
                "connection already closed",
                "the connection is closed",
                "eof detected",
                "timeout expired",
                "connection reset by peer",
                "software caused connection abort",
            )
            return any(s in msg for s in retry_signals)
        return False

    async def _with_retry(
        self,
        op: Callable[[AsyncCursor[DictRow]], Awaitable[Any]],
        *,
        pipeline: bool = False,
    ) -> Any:
        delay = self.retry_initial_backoff
        attempts = self.retry_max_attempts
        last_exc: BaseException | None = None

        for i in range(attempts):
            try:
                await self._ensure_connection_open()
                async with self._cursor(pipeline=pipeline) as cur:
                    return await op(cur)
            except BaseException as e:
                last_exc = e
                if not self._is_retryable_error(e) or i == attempts - 1:
                    raise
                await self._reconnect_if_single_conn()
                await asyncio.sleep(delay + random.random() * min(0.25, delay))
                delay = min(delay * 2, self.retry_max_backoff)

        if last_exc:
            raise last_exc

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously."""

        async def _do(cur: AsyncCursor[DictRow]) -> None:
            await cur.execute(self.MIGRATIONS[0])
            results = await cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
            )
            row = await results.fetchone()
            version = -1 if row is None else row["v"]
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
            ):
                await cur.execute(migration)
                await cur.execute(f"INSERT INTO checkpoint_migrations (v) VALUES ({v})")

        await self._with_retry(_do, pipeline=False)
        if self.pipe:
            await self.pipe.sync()

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously."""
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"

        async def _do(cur: AsyncCursor[DictRow]) -> list[DictRow]:
            await cur.execute(query, args, binary=True)
            values = await cur.fetchall()
            if not values:
                return []

            # migrate pending sends if necessary
            to_migrate = [
                v
                for v in values
                if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]
            ]
            if to_migrate:
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (
                        values[0]["thread_id"],
                        [v["parent_checkpoint_id"] for v in to_migrate],
                    ),
                )
                grouped_by_parent = defaultdict(list)
                for value in to_migrate:
                    grouped_by_parent[value["parent_checkpoint_id"]].append(value)

                async for sends in cur:
                    for value in grouped_by_parent[sends["checkpoint_id"]]:
                        if value["channel_values"] is None:
                            value["channel_values"] = []
                        self._migrate_pending_sends(
                            sends["sends"],
                            value["checkpoint"],
                            value["channel_values"],
                        )
            return values

        values = await self._with_retry(_do, pipeline=False)
        for value in values:
            yield await self._load_checkpoint_tuple(value)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database asynchronously."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        async def _do(cur: AsyncCursor[DictRow]) -> DictRow | None:
            await cur.execute(self.SELECT_SQL + where, args, binary=True)
            value = await cur.fetchone()
            if value is None:
                return None

            # migrate pending sends if necessary
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (thread_id, [value["parent_checkpoint_id"]]),
                )
                sends = await cur.fetchone()
                if sends:
                    if value["channel_values"] is None:
                        value["channel_values"] = []
                    self._migrate_pending_sends(
                        sends["sends"],
                        value["checkpoint"],
                        value["channel_values"],
                    )
            return value

        value = await self._with_retry(_do, pipeline=False)
        if value is None:
            return None
        return await self._load_checkpoint_tuple(value)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously."""
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

        # inline primitive values in checkpoint table; blobs stored separately
        blob_values: dict[str, Any] = {}
        for k, v in checkpoint["channel_values"].items():
            if v is None or isinstance(v, (str, int, float, bool)):
                pass
            else:
                blob_values[k] = copy["channel_values"].pop(k)

        blob_versions = {k: v for k, v in new_versions.items() if k in blob_values}
        if blob_versions:
            params_blobs = await asyncio.to_thread(
                self._dump_blobs,
                thread_id,
                checkpoint_ns,
                blob_values,
                blob_versions,
            )
        else:
            params_blobs = None

        payload_checkpoint = (
            thread_id,
            checkpoint_ns,
            checkpoint["id"],
            checkpoint_id,
            Jsonb(copy),
            Jsonb(get_checkpoint_metadata(config, metadata)),
        )

        async def _do(cur: AsyncCursor[DictRow]) -> None:
            if params_blobs:
                await cur.executemany(self.UPSERT_CHECKPOINT_BLOBS_SQL, params_blobs)
            await cur.execute(self.UPSERT_CHECKPOINTS_SQL, payload_checkpoint)

        await self._with_retry(_do, pipeline=True)
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously."""
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        params = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )

        async def _do(cur: AsyncCursor[DictRow]) -> None:
            await cur.executemany(query, params)

        await self._with_retry(_do, pipeline=True)

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID."""

        async def _do(cur: AsyncCursor[DictRow]) -> None:
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (str(thread_id),),
            )
            await cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                (str(thread_id),),
            )
            await cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (str(thread_id),),
            )

        await self._with_retry(_do, pipeline=True)

    @asynccontextmanager
    async def _cursor(
        self, *, pipeline: bool = False
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        """Create a database cursor as a context manager."""
        async with self.lock, _ainternal.get_connection(self.conn) as conn:
            if self.pipe:
                try:
                    async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                finally:
                    if pipeline:
                        await self.pipe.sync()
            elif pipeline:
                if self.supports_pipeline:
                    async with (
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    async with (
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    async def _load_checkpoint_tuple(self, value: DictRow) -> CheckpointTuple:
        """Convert a database row into a CheckpointTuple object."""
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
                    **value["checkpoint"].get("channel_values"),
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
            await asyncio.to_thread(self._load_writes, value["pending_writes"]),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncPostgresSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `checkpointer.alist(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),  # type: ignore[arg-type]  # noqa: F821
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncPostgresSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database."""
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint."""
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncPostgresSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()


__all__ = ["AsyncPostgresSaver", "AsyncShallowPostgresSaver", "Conn"]
