from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Callable, cast

import aiosqlite
import orjson
import sqlite_vec  # type: ignore[import-untyped]

from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
    TTLConfig,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.sqlite.base import (
    _PLACEHOLDER,
    BaseSqliteStore,
    SqliteIndexConfig,
    _decode_ns_text,
    _ensure_index_config,
    _group_ops,
    _row_to_item,
    _row_to_search_item,
)

logger = logging.getLogger(__name__)


class AsyncSqliteStore(AsyncBatchedBaseStore, BaseSqliteStore):
    """Asynchronous SQLite-backed store with optional vector search.

    This class provides an asynchronous interface for storing and retrieving data
    using a SQLite database with support for vector search capabilities.

    Examples:
        Basic setup and usage:
        ```python
        from langgraph.store.sqlite import AsyncSqliteStore

        async with AsyncSqliteStore.from_conn_string(":memory:") as store:
            await store.setup()  # Run migrations

            # Store and retrieve data
            await store.aput(("users", "123"), "prefs", {"theme": "dark"})
            item = await store.aget(("users", "123"), "prefs")
        ```

        Vector search using LangChain embeddings:
        ```python
        from langchain_openai import OpenAIEmbeddings
        from langgraph.store.sqlite import AsyncSqliteStore

        async with AsyncSqliteStore.from_conn_string(
            ":memory:",
            index={
                "dims": 1536,
                "embed": OpenAIEmbeddings(),
                "fields": ["text"]  # specify which fields to embed
            }
        ) as store:
            await store.setup()  # Run migrations once

            # Store documents
            await store.aput(("docs",), "doc1", {"text": "Python tutorial"})
            await store.aput(("docs",), "doc2", {"text": "TypeScript guide"})
            await store.aput(("docs",), "doc3", {"text": "Other guide"}, index=False)  # don't index

            # Search by similarity
            results = await store.asearch(("docs",), query="programming guides", limit=2)
        ```

    Warning:
        Make sure to call `setup()` before first use to create necessary tables and indexes.

    Note:
        This class requires the aiosqlite package. Install with `pip install aiosqlite`.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        deserializer: Callable[[bytes | str | orjson.Fragment], dict[str, Any]]
        | None = None,
        index: SqliteIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ):
        """Initialize the async SQLite store.

        Args:
            conn: The SQLite database connection.
            deserializer: Optional custom deserializer function for values.
            index: Optional vector search configuration.
            ttl: Optional time-to-live configuration.
        """
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.is_setup = False
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None
        self.ttl_config = ttl
        self._ttl_sweeper_task: asyncio.Task[None] | None = None
        self._ttl_stop_event = asyncio.Event()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: SqliteIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> AsyncIterator[AsyncSqliteStore]:
        """Create a new AsyncSqliteStore instance from a connection string.

        Args:
            conn_string: The SQLite connection string.
            index: Optional vector search configuration.
            ttl: Optional time-to-live configuration.

        Returns:
            An AsyncSqliteStore instance wrapped in an async context manager.
        """
        async with aiosqlite.connect(conn_string, isolation_level=None) as conn:
            yield cls(conn, index=index, ttl=ttl)

    async def setup(self) -> None:
        """Set up the store database.

        This method creates the necessary tables in the SQLite database if they don't
        already exist and runs database migrations. It should be called before first use.
        """
        async with self.lock:
            if self.is_setup:
                return

            # Create migrations table if it doesn't exist
            await self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS store_migrations (
                    v INTEGER PRIMARY KEY
                )
                """
            )

            # Check current migration version
            async with self.conn.execute(
                "SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1"
            ) as cur:
                row = await cur.fetchone()
                if row is None:
                    version = -1
                else:
                    version = row[0]

            # Apply migrations
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                await self.conn.executescript(sql)
                await self.conn.execute(
                    "INSERT INTO store_migrations (v) VALUES (?)", (v,)
                )

            # Apply vector migrations if index config is provided
            if self.index_config:
                # Create vector migrations table if it doesn't exist
                await self.conn.enable_load_extension(True)
                await self.conn.load_extension(sqlite_vec.loadable_path())
                await self.conn.enable_load_extension(False)
                await self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vector_migrations (
                        v INTEGER PRIMARY KEY
                    )
                    """
                )

                # Check current vector migration version
                async with self.conn.execute(
                    "SELECT v FROM vector_migrations ORDER BY v DESC LIMIT 1"
                ) as cur:
                    row = await cur.fetchone()
                    if row is None:
                        version = -1
                    else:
                        version = row[0]

                # Apply vector migrations
                for v, sql in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    await self.conn.executescript(sql)
                    await self.conn.execute(
                        "INSERT INTO vector_migrations (v) VALUES (?)", (v,)
                    )

            self.is_setup = True

    @asynccontextmanager
    async def _cursor(
        self, *, transaction: bool = True
    ) -> AsyncIterator[aiosqlite.Cursor]:
        """Get a cursor for the SQLite database.

        Args:
            transaction: Whether to use a transaction for database operations.

        Yields:
            An SQLite cursor object.
        """
        if not self.is_setup:
            await self.setup()
        async with self.lock:
            if transaction:
                await self.conn.execute("BEGIN")

            async with self.conn.cursor() as cur:
                try:
                    yield cur
                finally:
                    if transaction:
                        await self.conn.execute("COMMIT")

    async def sweep_ttl(self) -> int:
        """Delete expired store items based on TTL.

        Returns:
            int: The number of deleted items.
        """
        async with self._cursor() as cur:
            await cur.execute(
                """
                DELETE FROM store
                WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                """
            )
            deleted_count = cur.rowcount
            return deleted_count

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
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

    async def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
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

    async def __aenter__(self) -> AsyncSqliteStore:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Ensure the TTL sweeper task is stopped when exiting the context
        if hasattr(self, "_ttl_sweeper_task") and self._ttl_sweeper_task is not None:
            # Set the event to signal the task to stop
            self._ttl_stop_event.set()
            # We don't wait for the task to complete here to avoid blocking
            # The task will clean up itself gracefully

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations asynchronously.

        Args:
            ops: Iterable of operations to execute.

        Returns:
            List of operation results.
        """
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        async with self._cursor(transaction=True) as cur:
            if GetOp in grouped_ops:
                await self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results, cur
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
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), cur
                )

        return results

    async def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: aiosqlite.Cursor,
    ) -> None:
        """Process batch GET operations.

        Args:
            get_ops: Sequence of GET operations.
            results: List to store results in.
            cur: Database cursor.
        """
        # Group all queries by namespace to execute all operations for each namespace together
        namespace_queries = defaultdict(list)
        for prepared_query in self._get_batch_GET_ops_queries(get_ops):
            namespace_queries[prepared_query.namespace].append(prepared_query)

        # Process each namespace's operations
        for namespace, queries in namespace_queries.items():
            # Execute TTL refresh queries first
            for query in queries:
                if query.kind == "refresh":
                    try:
                        await cur.execute(query.query, query.params)
                    except Exception as e:
                        raise ValueError(
                            f"Error executing TTL refresh: \n{query.query}\n{query.params}\n{e}"
                        ) from e

            # Then execute GET queries and process results
            for query in queries:
                if query.kind == "get":
                    try:
                        await cur.execute(query.query, query.params)
                    except Exception as e:
                        raise ValueError(
                            f"Error executing GET query: \n{query.query}\n{query.params}\n{e}"
                        ) from e

                    rows = await cur.fetchall()
                    key_to_row = {
                        row[0]: {
                            "key": row[0],
                            "value": row[1],
                            "created_at": row[2],
                            "updated_at": row[3],
                            "expires_at": row[4] if len(row) > 4 else None,
                            "ttl_minutes": row[5] if len(row) > 5 else None,
                        }
                        for row in rows
                    }

                    # Process results for this query
                    for idx, key in query.items:
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
        cur: aiosqlite.Cursor,
    ) -> None:
        """Process batch PUT operations.

        Args:
            put_ops: Sequence of PUT operations.
            cur: Database cursor.
        """
        queries, embedding_request = self._prepare_batch_PUT_queries(put_ops)
        if embedding_request:
            if self.embeddings is None:
                # Should not get here since the embedding config is required
                # to return an embedding_request above
                raise ValueError(
                    "Embedding configuration is required for vector operations "
                    f"(for semantic search). "
                    f"Please provide an Embeddings when initializing the {self.__class__.__name__}."
                )

            query, txt_params = embedding_request
            # Update the params to replace the raw text with the vectors
            vectors = await self.embeddings.aembed_documents(
                [param[-1] for param in txt_params]
            )

            # Convert vectors to SQLite-friendly format
            vector_params = []
            for (ns, k, pathname, _), vector in zip(txt_params, vectors):
                vector_params.extend(
                    [ns, k, pathname, sqlite_vec.serialize_float32(vector)]
                )

            queries.append((query, vector_params))

        for query, params in queries:
            await cur.execute(query, params)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: aiosqlite.Cursor,
    ) -> None:
        """Process batch SEARCH operations.

        Args:
            search_ops: Sequence of SEARCH operations.
            results: List to store results in.
            cur: Database cursor.
        """
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)

        # Setup dot_product function if it doesn't exist
        if embedding_requests and self.embeddings:
            vectors = await self.embeddings.aembed_documents(
                [query for _, query in embedding_requests]
            )

            for (idx, _), embedding in zip(embedding_requests, vectors):
                _params_list: list = queries[idx][1]
                for i, param in enumerate(_params_list):
                    if param is _PLACEHOLDER:
                        _params_list[i] = sqlite_vec.serialize_float32(embedding)

        for (idx, _), (query, params) in zip(search_ops, queries):
            await cur.execute(query, params)
            rows = await cur.fetchall()

            if "score" in query:
                items = [
                    _row_to_search_item(
                        _decode_ns_text(row[0]),
                        {
                            "key": row[1],
                            "value": row[2],
                            "created_at": row[3],
                            "updated_at": row[4],
                            "expires_at": row[5] if len(row) > 5 else None,
                            "ttl_minutes": row[6] if len(row) > 6 else None,
                            "score": row[7] if len(row) > 7 else None,
                        },
                        loader=self._deserializer,
                    )
                    for row in rows
                ]
            else:  # Regular search query
                items = [
                    _row_to_search_item(
                        _decode_ns_text(row[0]),
                        {
                            "key": row[1],
                            "value": row[2],
                            "created_at": row[3],
                            "updated_at": row[4],
                            "expires_at": row[5] if len(row) > 5 else None,
                            "ttl_minutes": row[6] if len(row) > 6 else None,
                        },
                        loader=self._deserializer,
                    )
                    for row in rows
                ]

            results[idx] = items

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: aiosqlite.Cursor,
    ) -> None:
        """Process batch LIST NAMESPACES operations.

        Args:
            list_ops: Sequence of LIST NAMESPACES operations.
            results: List to store results in.
            cur: Database cursor.
        """
        queries = self._get_batch_list_namespaces_queries(list_ops)
        for (query, params), (idx, _) in zip(queries, list_ops):
            await cur.execute(query, params)

            rows = await cur.fetchall()
            results[idx] = [_decode_ns_text(row[0]) for row in rows]
