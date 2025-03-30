import asyncio
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Callable, Optional, Union, cast

import orjson
from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline, Capabilities
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.postgres import _ainternal
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.postgres.base import (
    PLACEHOLDER,
    BasePostgresStore,
    PoolConfig,
    PostgresIndexConfig,
    Row,
    TTLConfig,
    _decode_ns_bytes,
    _ensure_index_config,
    _group_ops,
    _row_to_item,
    _row_to_search_item,
)

logger = logging.getLogger(__name__)


class AsyncPostgresStore(AsyncBatchedBaseStore, BasePostgresStore[_ainternal.Conn]):
    """Asynchronous Postgres-backed store with optional vector search using pgvector.

    !!! example "Examples"
        Basic setup and usage:
        ```python
        from langgraph.store.postgres import AsyncPostgresStore

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        async with AsyncPostgresStore.from_conn_string(conn_string) as store:
            await store.setup()  # Run migrations. Done once

            # Store and retrieve data
            await store.aput(("users", "123"), "prefs", {"theme": "dark"})
            item = await store.aget(("users", "123"), "prefs")
        ```

        Vector search using LangChain embeddings:
        ```python
        from langchain.embeddings import init_embeddings
        from langgraph.store.postgres import AsyncPostgresStore

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        async with AsyncPostgresStore.from_conn_string(
            conn_string,
            index={
                "dims": 1536,
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "fields": ["text"]  # specify which fields to embed. Default is the whole serialized value
            }
        ) as store:
            await store.setup()  # Run migrations. Done once

            # Store documents
            await store.aput(("docs",), "doc1", {"text": "Python tutorial"})
            await store.aput(("docs",), "doc2", {"text": "TypeScript guide"})
            await store.aput(("docs",), "doc3", {"text": "Other guide"}, index=False)  # don't index

            # Search by similarity
            results = await store.asearch(("docs",), query="programming guides", limit=2)
        ```

        Using connection pooling for better performance:
        ```python
        from langgraph.store.postgres import AsyncPostgresStore, PoolConfig

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        async with AsyncPostgresStore.from_conn_string(
            conn_string,
            pool_config=PoolConfig(
                min_size=5,
                max_size=20
            )
        ) as store:
            await store.setup()  # Run migrations. Done once
            # Use store with connection pooling...
        ```

    Warning:
        Make sure to:
        1. Call `setup()` before first use to create necessary tables and indexes
        2. Have the pgvector extension available to use vector search
        3. Use Python 3.10+ for async functionality

    Note:
        Semantic search is disabled by default. You can enable it by providing an `index` configuration
        when creating the store. Without this configuration, all `index` arguments passed to
        `put` or `aput` will have no effect.

    Note:
        If you provide a TTL configuration, you must explicitly call `start_ttl_sweeper()` to begin
        the background task that removes expired items. Call `stop_ttl_sweeper()` to properly
        clean up resources when you're done with the store.
    """

    __slots__ = (
        "_deserializer",
        "pipe",
        "lock",
        "supports_pipeline",
        "index_config",
        "embeddings",
        "ttl_config",
        "_ttl_sweeper_task",
        "_ttl_stop_event",
    )
    supports_ttl: bool = True

    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        pipe: Optional[AsyncPipeline] = None,
        deserializer: Optional[
            Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]
        ] = None,
        index: Optional[PostgresIndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> None:
        if isinstance(conn, AsyncConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single AsyncConnection, not AsyncConnectionPool."
            )
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.supports_pipeline = Capabilities().has_pipeline()
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None

        self.ttl_config = ttl
        self._ttl_sweeper_task: Optional[asyncio.Task[None]] = None
        self._ttl_stop_event = asyncio.Event()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        async with _ainternal.get_connection(self.conn) as conn:
            if self.pipe:
                async with self.pipe:
                    await self._execute_batch(grouped_ops, results, conn)
            else:
                await self._execute_batch(grouped_ops, results, conn)

        return results

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        pool_config: Optional[PoolConfig] = None,
        index: Optional[PostgresIndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> AsyncIterator["AsyncPostgresStore"]:
        """Create a new AsyncPostgresStore instance from a connection string.

        Args:
            conn_string (str): The Postgres connection info string.
            pipeline (bool): Whether to use AsyncPipeline (only for single connections)
            pool_config (Optional[PoolConfig]): Configuration for the connection pool.
                If provided, will create a connection pool and use it instead of a single connection.
                This overrides the `pipeline` argument.
            index (Optional[PostgresIndexConfig]): The embedding config.

        Returns:
            AsyncPostgresStore: A new AsyncPostgresStore instance.
        """
        if pool_config is not None:
            pc = pool_config.copy()
            async with cast(
                AsyncConnectionPool[AsyncConnection[DictRow]],
                AsyncConnectionPool(
                    conn_string,
                    min_size=pc.pop("min_size", 1),
                    max_size=pc.pop("max_size", None),
                    kwargs={
                        "autocommit": True,
                        "prepare_threshold": 0,
                        "row_factory": dict_row,
                        **(pc.pop("kwargs", None) or {}),
                    },
                    **cast(dict, pc),
                ),
            ) as pool:
                yield cls(conn=pool, index=index, ttl=ttl)
        else:
            async with await AsyncConnection.connect(
                conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
            ) as conn:
                if pipeline:
                    async with conn.pipeline() as pipe:
                        yield cls(conn=conn, pipe=pipe, index=index, ttl=ttl)
                else:
                    yield cls(conn=conn, index=index, ttl=ttl)

    async def setup(self) -> None:
        """Set up the store database asynchronously.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """

        async def _get_version(cur: AsyncCursor[DictRow], table: str) -> int:
            await cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    v INTEGER PRIMARY KEY
                )
            """
            )
            await cur.execute(f"SELECT v FROM {table} ORDER BY v DESC LIMIT 1")
            row = cast(dict, await cur.fetchone())
            if row is None:
                version = -1
            else:
                version = row["v"]
            return version

        async with self._cursor() as cur:
            version = await _get_version(cur, table="store_migrations")
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                await cur.execute(sql)
                await cur.execute("INSERT INTO store_migrations (v) VALUES (%s)", (v,))

            if self.index_config:
                version = await _get_version(cur, table="vector_migrations")
                for v, migration in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    sql = migration.sql
                    if migration.params:
                        params = {
                            k: v(self) if v is not None and callable(v) else v
                            for k, v in migration.params.items()
                        }
                        sql = sql % params
                    await cur.execute(sql)
                    await cur.execute(
                        "INSERT INTO vector_migrations (v) VALUES (%s)", (v,)
                    )

    async def sweep_ttl(self) -> int:
        """Delete expired store items based on TTL.

        Returns:
            int: The number of deleted items.
        """
        async with self._cursor() as cur:
            await cur.execute(
                """
                DELETE FROM store
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                """
            )
            deleted_count = cur.rowcount
            return deleted_count

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: Optional[int] = None
    ) -> asyncio.Task[None]:
        """Periodically delete expired store items based on TTL.

        Returns:
            Task that can be awaited or cancelled.
        """
        if not self.ttl_config:
            return asyncio.create_task(asyncio.sleep(0))

        if self._ttl_sweeper_task is not None and not self._ttl_sweeper_task.done():
            return self._ttl_sweeper_task

        self._ttl_stop_event.clear()

        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")

        async def _sweep_loop() -> None:
            while not self._ttl_stop_event.is_set():
                try:
                    try:
                        await asyncio.wait_for(
                            self._ttl_stop_event.wait(),
                            timeout=interval * 60,
                        )
                        break
                    except asyncio.TimeoutError:
                        pass

                    expired_items = await self.sweep_ttl()
                    if expired_items > 0:
                        logger.info(f"Store swept {expired_items} expired items")
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.exception("Store TTL sweep iteration failed", exc_info=exc)

        task = asyncio.create_task(_sweep_loop())
        task.set_name("ttl_sweeper")
        self._ttl_sweeper_task = task
        return task

    async def stop_ttl_sweeper(self, timeout: Optional[float] = None) -> bool:
        """Stop the TTL sweeper task if it's running.

        Args:
            timeout: Maximum time to wait for the task to stop, in seconds.
                If None, wait indefinitely.

        Returns:
            bool: True if the task was successfully stopped or wasn't running,
                False if the timeout was reached before the task stopped.
        """
        if self._ttl_sweeper_task is None or self._ttl_sweeper_task.done():
            return True

        logger.info("Stopping TTL sweeper task")
        self._ttl_stop_event.set()

        if timeout is not None:
            try:
                await asyncio.wait_for(self._ttl_sweeper_task, timeout=timeout)
                success = True
            except asyncio.TimeoutError:
                success = False
        else:
            await self._ttl_sweeper_task
            success = True

        if success:
            self._ttl_sweeper_task = None
            logger.info("TTL sweeper task stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper task to stop")

        return success

    async def __aenter__(self) -> "AsyncPostgresStore":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional["TracebackType"],
    ) -> None:
        # Ensure the TTL sweeper task is stopped when exiting the context
        if hasattr(self, "_ttl_sweeper_task") and self._ttl_sweeper_task is not None:
            # Set the event to signal the task to stop
            self._ttl_stop_event.set()
            # We don't wait for the task to complete here to avoid blocking
            # The task will clean up itself gracefully

    async def _execute_batch(
        self,
        grouped_ops: dict,
        results: list[Result],
        conn: AsyncConnection[DictRow],
    ) -> None:
        async with self._cursor(pipeline=True) as cur:
            if GetOp in grouped_ops:
                await self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]),
                    results,
                    cur,
                )

            if SearchOp in grouped_ops:
                await self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                await self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )

            if PutOp in grouped_ops:
                await self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]),
                    cur,
                )

    async def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            await cur.execute(query, params)
            rows = cast(list[Row], await cur.fetchall())
            key_to_row = {row["key"]: row for row in rows}
            for idx, key in items:
                row = key_to_row.get(key)
                if row:
                    results[idx] = _row_to_item(
                        namespace, row, loader=self._deserializer
                    )
                else:
                    results[idx] = None

    async def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        cur: AsyncCursor[DictRow],
    ) -> None:
        queries, embedding_request = self._prepare_batch_PUT_queries(put_ops)
        if embedding_request:
            if self.embeddings is None:
                # Should not get here since the embedding config is required
                # to return an embedding_request above
                raise ValueError(
                    "Embedding configuration is required for vector operations "
                    f"(for semantic search). "
                    f"Please provide an EmbeddingConfig when initializing the {self.__class__.__name__}."
                )
            query, txt_params = embedding_request
            vectors = await self.embeddings.aembed_documents(
                [param[-1] for param in txt_params]
            )
            queries.append(
                (
                    query,
                    [
                        p
                        for (ns, k, pathname, _), vector in zip(txt_params, vectors)
                        for p in (ns, k, pathname, vector)
                    ],
                )
            )

        for query, params in queries:
            await cur.execute(query, params)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)

        if embedding_requests and self.embeddings:
            vectors = await self.embeddings.aembed_documents(
                [query for _, query in embedding_requests]
            )
            for (idx, _), vector in zip(embedding_requests, vectors):
                _paramslist = queries[idx][1]
                for i in range(len(_paramslist)):
                    if _paramslist[i] is PLACEHOLDER:
                        _paramslist[i] = vector

        for (idx, _), (query, params) in zip(search_ops, queries):
            await cur.execute(query, params)
            rows = cast(list[Row], await cur.fetchall())
            items = [
                _row_to_search_item(
                    _decode_ns_bytes(row["prefix"]), row, loader=self._deserializer
                )
                for row in rows
            ]
            results[idx] = items

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        for (query, params), (idx, _) in zip(queries, list_ops):
            await cur.execute(query, params)
            rows = cast(list[dict], await cur.fetchall())
            namespaces = [_decode_ns_bytes(row["truncated_prefix"]) for row in rows]
            results[idx] = namespaces

    @asynccontextmanager
    async def _cursor(
        self, *, pipeline: bool = False
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        """Create a database cursor as a context manager.

        Args:
            pipeline: whether to use pipeline for the DB operations inside the context manager.
                Will be applied regardless of whether the PostgresStore instance was initialized with a pipeline.
                If pipeline mode is not supported, will fall back to using transaction context manager.
        """
        async with _ainternal.get_connection(self.conn) as conn:
            if self.pipe:
                # a connection in pipeline mode can be used concurrently
                # in multiple threads/coroutines, but only one cursor can be
                # used at a time
                try:
                    async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                finally:
                    if pipeline:
                        await self.pipe.sync()
            elif pipeline:
                # a connection not in pipeline mode can only be used by one
                # thread/coroutine at a time, so we acquire a lock
                if self.supports_pipeline:
                    async with (
                        self.lock,
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    async with (
                        self.lock,
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                async with (
                    self.lock,
                    conn.cursor(binary=True) as cur,
                ):
                    yield cur
