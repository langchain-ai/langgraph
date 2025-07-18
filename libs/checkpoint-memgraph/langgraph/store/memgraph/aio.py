from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Callable, cast
from urllib.parse import unquote, urlparse

import orjson
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession, AsyncTransaction
from neo4j.exceptions import Neo4jError

from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.memgraph.base import (
    BaseMemgraphStore,
    MemgraphIndexConfig,
    Record,
    TTLConfig,
    _decode_ns_text,
    _ensure_index_config,
    _group_ops,
    _namespace_to_text,
    _record_to_item,
    _record_to_search_item,
)

logger = logging.getLogger(__name__)


class AsyncMemgraphStore(AsyncBatchedBaseStore, BaseMemgraphStore[AsyncDriver]):
    """Asynchronous Memgraph-backed store using the neo4j async driver.

    This store interacts with a Memgraph database to persist, retrieve, and query
    data. It is designed for use in asynchronous applications and supports batch
    operations, vector-based semantic search, and Time-To-Live (TTL) data expiration.

    Args:
        conn: An asynchronous Memgraph database driver.
        database: The name of the database to connect to.
        deserializer: A function to deserialize stored data. Defaults to orjson.loads.
        index: Configuration for creating vector indexes for semantic search.
        ttl: Configuration for the Time-To-Live (TTL) feature for automatic data
            expiration.

    !!! example "Examples"
        Basic setup and usage:
        ```python
        from langgraph.store.memgraph import AsyncMemgraphStore

        conn_string = "bolt://user:pass@localhost:7687"

        async with AsyncMemgraphStore.from_conn_string(conn_string) as store:
            await store.setup()  # Run migrations. Done once

            # Store and retrieve data
            await store.aput(("users", "123"), {"prefs": {"theme": "dark"}}, key="default")
            item = await store.aget(("users", "123"), key="default")
        ```

    Warning:
        Make sure to call `setup()` before first use to create necessary constraints and indexes.
    """

    __slots__ = (
        "database",
        "_deserializer",
        "index_config",
        "embeddings",
        "ttl_config",
        "lock",
        "loop",
        "_ttl_sweeper_task",
        "_ttl_stop_event",
    )
    supports_ttl: bool = True

    def __init__(
        self,
        conn: AsyncDriver,
        *,
        database: str = "memgraph",
        deserializer: Callable[[str], dict[str, Any]] | None = None,
        index: MemgraphIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        super().__init__()
        self.database = database
        self._deserializer = deserializer or (lambda v: orjson.loads(v))
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
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
        database: str = "memgraph",
        index: MemgraphIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> AsyncIterator[AsyncMemgraphStore]:
        """Create a new AsyncMemgraphStore instance from a connection string.

        Args:
            conn_string: The Memgraph connection URI (e.g., "bolt://user:pass@host:7687").
            database: The database name to connect to.
            index: The index configuration for the store.
            ttl: The TTL configuration for the store.

        Returns:
            An AsyncMemgraphStore instance within an async context manager.
        """
        parsed = urlparse(conn_string)
        uri = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 7687}"
        auth = (unquote(parsed.username or ""), unquote(parsed.password or ""))
        driver = AsyncGraphDatabase.driver(uri, auth=auth)
        try:
            yield cls(driver, database=database, index=index, ttl=ttl)
        finally:
            await driver.close()

    async def __aenter__(self) -> AsyncMemgraphStore:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if hasattr(self, "_ttl_sweeper_task") and self._ttl_sweeper_task is not None:
            await self.stop_ttl_sweeper()
        if hasattr(self, "_task") and self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        async with self.conn.session(database=self.database) as session:
            yield session

    @asynccontextmanager
    async def _transaction(
        self, session: AsyncSession
    ) -> AsyncIterator[AsyncTransaction]:
        async with await session.begin_transaction() as tx:
            yield tx

    def _build_search_where_clause(self, op: SearchOp, params: dict[str, Any]) -> str:
        where_clauses = ["(n.expires_at IS NULL OR n.expires_at >= localdatetime())"]
        if op.namespace_prefix is not None:
            where_clauses.append("n.prefix STARTS WITH $prefix")
            params["prefix"] = _namespace_to_text(op.namespace_prefix)
        if op.filter:
            i = 0
            for key, value in op.filter.items():
                if isinstance(value, dict):
                    for op_key, op_val in value.items():
                        param_name = f"filter_val_{i}"
                        if op_key == "$gt":
                            where_clauses.append(f"n.{key} > ${param_name}")
                        elif op_key == "$gte":
                            where_clauses.append(f"n.{key} >= ${param_name}")
                        elif op_key == "$lt":
                            where_clauses.append(f"n.{key} < ${param_name}")
                        elif op_key == "$lte":
                            where_clauses.append(f"n.{key} <= ${param_name}")
                        elif op_key == "$ne":
                            where_clauses.append(f"n.{key} <> ${param_name}")
                        elif op_key == "$eq":
                            where_clauses.append(f"n.{key} = ${param_name}")
                        else:
                            logger.warning(
                                f"Unsupported filter operator '{op_key}' for key '{key}'."
                            )
                            continue
                        params[param_name] = op_val
                        i += 1
                else:
                    param_name = f"filter_val_{i}"
                    where_clauses.append(f"n.{key} = ${param_name}")
                    params[param_name] = value
                    i += 1
        return f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    def _get_batch_list_namespaces_queries(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
    ) -> list[tuple[str, dict[str, Any]]]:
        queries: list[tuple[str, dict[str, Any]]] = []
        for _, op in list_ops:
            params: dict[str, Any] = {
                "offset": op.offset,
                "max_depth": op.max_depth,
            }
            match_clauses = [
                "(n.expires_at IS NULL OR n.expires_at >= localdatetime())"
            ]
            if op.match_conditions:
                for i, condition in enumerate(op.match_conditions):
                    path_param = f"path_{i}"
                    params[path_param] = _namespace_to_text(condition.path)
                    if condition.match_type == "prefix":
                        match_clauses.append(f"n.prefix STARTS WITH ${path_param}")
                    elif condition.match_type == "suffix":
                        match_clauses.append(f"n.prefix ENDS WITH ${path_param}")
                    else:
                        logger.warning(
                            f"Unknown match_type in list_namespaces: {condition.match_type}"
                        )
            where_clause = f"WHERE {' AND '.join(match_clauses)}"
            limit_clause = ""
            if op.limit is not None:
                limit_clause = "LIMIT $limit"
                params["limit"] = op.limit
            query = f"""
                MATCH (n:StoreItem)
                {where_clause}
                WITH n.prefix AS full_prefix
                WITH full_prefix, split(full_prefix, '.') AS parts
                WITH full_prefix, parts,
                     CASE
                         WHEN $max_depth IS NOT NULL AND size(parts) > $max_depth
                         THEN substring(REDUCE(s = "", p IN parts[0..$max_depth - 1] | s + '.' + p), 1)
                         ELSE full_prefix
                     END AS truncated_prefix
                RETURN DISTINCT truncated_prefix
                ORDER BY truncated_prefix
                SKIP $offset
                {limit_clause}
            """
            queries.append((query, params))
        return queries

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Atomically executes a batch of operations against the store.

        This method groups operations by type and executes them within a single
        transaction, ensuring that all operations succeed or fail together.

        Args:
            ops: An iterable of operations (GetOp, PutOp, SearchOp, ListNamespacesOp)
                to be executed.

        Returns:
            A list of results corresponding to the input operations. The result for a
            PutOp will be None.
        """
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops
        async with self._session() as session:
            async with self.lock, self._transaction(session) as tx:
                if GetOp in grouped_ops:
                    await self._batch_get_ops(
                        cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]),
                        results,
                        tx,
                    )
                if SearchOp in grouped_ops:
                    await self._batch_search_ops(
                        cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                        results,
                        tx,
                    )
                if ListNamespacesOp in grouped_ops:
                    await self._batch_list_namespaces_ops(
                        cast(
                            Sequence[tuple[int, ListNamespacesOp]],
                            grouped_ops[ListNamespacesOp],
                        ),
                        results,
                        tx,
                    )
                if PutOp in grouped_ops:
                    await self._batch_put_ops(
                        cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), tx
                    )
        return results

    async def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        tx: AsyncTransaction,
    ) -> None:
        for (
            query,
            params,
            namespace,
            items,
        ) in self._get_batch_get_ops_queries(get_ops):
            result = await tx.run(query, params)
            key_to_idx = {item["key"]: item["idx"] for item in items}
            assert self._deserializer is not None
            async for rec in result:
                idx = key_to_idx.get(rec["key"])
                if idx is not None:
                    record: Record = {
                        "key": rec["key"],
                        "value": rec["value"],
                        "prefix": _namespace_to_text(namespace),
                        "created_at": rec["created_at"],
                        "updated_at": rec["updated_at"],
                    }
                    results[idx] = _record_to_item(
                        namespace,
                        record,
                        loader=self._deserializer,
                    )

    async def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        tx: AsyncTransaction,
    ) -> None:
        queries, embedding_request = self._prepare_batch_put_queries(put_ops)
        for query, params in queries:
            await tx.run(query, params)
        if embedding_request:
            if self.embeddings is None:
                raise ValueError(
                    "Embedding configuration is required for vector operations."
                )
            query, txt_params = embedding_request
            unique_texts = sorted({param[-1] for param in txt_params})
            vectors = await self.embeddings.aembed_documents(unique_texts)
            text_to_vector = dict(zip(unique_texts, vectors))
            embedding_batch = [
                {
                    "prefix": ns,
                    "key": k,
                    "text": text,
                    "embedding": text_to_vector[text],
                }
                for (ns, k, text) in txt_params
            ]
            await tx.run(query, {"batch": embedding_batch})

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        tx: AsyncTransaction,
    ) -> None:
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)
        op_idxs_requiring_embedding = {
            op_idx for op_idx, text in embedding_requests if text
        }
        op_idx_to_params = {
            op_idx: queries[i][1]
            for i, (op_idx, _) in enumerate(search_ops)
            if i < len(queries)
        }
        if embedding_requests and self.embeddings:
            unique_texts = sorted({text for _, text in embedding_requests if text})
            if unique_texts:
                embeddings = await self.embeddings.aembed_documents(unique_texts)
                text_to_embedding = dict(zip(unique_texts, embeddings))
                for op_idx, text in embedding_requests:
                    if text and op_idx in op_idx_to_params:
                        op_idx_to_params[op_idx]["embedding"] = text_to_embedding.get(
                            text
                        )
        for i, (op_idx, _op) in enumerate(search_ops):
            if i >= len(queries):
                continue
            query, params = queries[i]
            if op_idx in op_idxs_requiring_embedding and not params.get("embedding"):
                results[op_idx] = []
                continue
            result = await tx.run(query, params)
            search_items: list[SearchItem] = []
            assert self._deserializer is not None
            async for r in result:
                record: Record = {
                    "key": r["key"],
                    "value": r["value"],
                    "prefix": r["prefix"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "score": r.get("score"),
                }
                search_items.append(
                    _record_to_search_item(
                        _decode_ns_text(r["prefix"]), record, loader=self._deserializer
                    )
                )
            results[op_idx] = search_items

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        tx: AsyncTransaction,
    ) -> None:
        for (query, params), (idx, _) in zip(
            self._get_batch_list_namespaces_queries(list_ops), list_ops
        ):
            result = await tx.run(query, params)
            results[idx] = [
                _decode_ns_text(row["truncated_prefix"])
                async for row in result
                if row["truncated_prefix"]
            ]

    async def setup(self) -> None:
        """Set up the database schema and indexes.

        This method applies all necessary migrations to the database to ensure that
        the required constraints and indexes are in place for the store to
        function correctly. It also handles the setup of vector indexes if a
        vector search is configured.

        This method is idempotent and should be called once before the store is
        used.
        """

        async def _get_version(tx: AsyncTransaction, table: str) -> int:
            result = await tx.run(
                """
                MERGE (m:Migration {name: $table})
                ON CREATE SET m.version = -1
                RETURN m.version AS v
                """,
                {"table": table},
            )
            record = await result.single()
            return record["v"] if record else -1

        async def _set_version(tx: AsyncTransaction, table: str, version: int) -> None:
            await tx.run(
                """
                MATCH (m:Migration {name: $table})
                SET m.version = $version
                """,
                {"table": table, "version": version},
            )

        async with self._session() as session:
            async with self._transaction(session) as tx:
                version = await _get_version(tx, "store_migrations")
            for v, cypher in enumerate(
                self.MIGRATIONS[version + 1 :], start=version + 1
            ):
                try:
                    await session.run(cypher)
                    async with self._transaction(session) as tx:
                        await _set_version(tx, "store_migrations", v)
                except Neo4jError as e:
                    if "already exists" in str(e).lower():
                        async with self._transaction(session) as tx:
                            await _set_version(tx, "store_migrations", v)
                    else:
                        logger.error(
                            f"Failed to apply migration {v}.\\nCypher={cypher}\\nError={e}"
                        )
                        raise
            if self.index_config:
                async with self._transaction(session) as tx:
                    version = await _get_version(tx, "vector_migrations")
                for v, migration in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    if migration.condition and not migration.condition(self):
                        continue
                    cypher = migration.cypher
                    params = {}
                    if migration.params:
                        params = {
                            k: val(self) if callable(val) else val
                            for k, val in migration.params.items()
                        }
                    final_cypher = cypher.format(**params)
                    try:
                        await session.run(final_cypher)
                        async with self._transaction(session) as tx:
                            await _set_version(tx, "vector_migrations", v)
                    except Neo4jError as e:
                        if "already exists" in str(e).lower():
                            async with self._transaction(session) as tx:
                                await _set_version(tx, "vector_migrations", v)
                        else:
                            logger.error(
                                f"Failed to apply vector migration {v}.\\nCypher={final_cypher}\\nError={e}"
                            )
                            raise

    async def sweep_ttl(self) -> int:
        """Deletes expired items from the store based on their TTL.

        This method queries the database for items where the `expires_at` timestamp
        is in the past, and deletes them. It processes items in batches to avoid
        long-running transactions.

        Returns:
            The number of items that were deleted.
        """
        async with self._session() as session:
            result = await session.run(
                """
                MATCH (n:StoreItem)
                WHERE n.expires_at IS NOT NULL AND n.expires_at < localdatetime()
                WITH n LIMIT 10000
                DETACH DELETE n
                RETURN count(n) as deleted_count
                """
            )
            record = await result.single()
            return record["deleted_count"] if record else 0

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> asyncio.Task[None]:
        """Starts a background task to periodically delete expired items.

        This method initiates a sweeper that runs in a continuous loop, calling
        `sweep_ttl` at a specified interval. If a sweeper task is already
        running, this method will return the existing task.

        The TTL feature must be configured for the sweeper to run.

        Args:
            sweep_interval_minutes: The interval in minutes at which to sweep for
                expired items. If not provided, it defaults to the value in the
                TTLConfig, or 5 minutes.

        Returns:
            An asyncio.Task representing the running sweeper, which can be
            used to monitor or cancel it.
        """
        if not self.ttl_config:
            return asyncio.create_task(asyncio.sleep(0))
        if self._ttl_sweeper_task and not self._ttl_sweeper_task.done():
            logger.info("TTL sweeper task is already running")
            return self._ttl_sweeper_task
        self._ttl_stop_event.clear()
        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")

        async def _sweep_loop() -> None:
            while not self._ttl_stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._ttl_stop_event.wait(),
                        timeout=interval * 60,
                    )
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    break
                if self._ttl_stop_event.is_set():
                    break
                try:
                    expired_items = await self.sweep_ttl()
                    if expired_items > 0:
                        logger.info(f"Store swept {expired_items} expired items")
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.exception("Store TTL sweep iteration failed", exc_info=exc)

        task = asyncio.create_task(_sweep_loop())
        task.set_name("ttl-sweeper")
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
        try:
            await asyncio.wait_for(self._ttl_sweeper_task, timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        self._ttl_sweeper_task.cancel()
        try:
            await self._ttl_sweeper_task
        except asyncio.CancelledError:
            pass
        success = self._ttl_sweeper_task.done()
        if success:
            self._ttl_sweeper_task = None
            logger.info("TTL sweeper task stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper task to stop")
        return success
