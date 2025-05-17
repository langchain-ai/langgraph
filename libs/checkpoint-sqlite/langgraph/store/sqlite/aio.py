import asyncio
import datetime
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Callable, Optional, Union, cast

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
    get_text_at_path,
    tokenize_path,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.sqlite.base import (
    _PLACEHOLDER,
    MIGRATIONS,
    VECTOR_MIGRATIONS,
    PreparedGetQuery,
    SqliteIndexConfig,
    _decode_ns_text,
    _ensure_index_config,
    _group_ops,
    _namespace_to_text,
    _row_to_item,
    _row_to_search_item,
)

logger = logging.getLogger(__name__)


class AsyncSqliteStore(AsyncBatchedBaseStore):
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
            results = await store.asearch(("docs",), "programming guides", limit=2)
        ```

    Warning:
        Make sure to call `setup()` before first use to create necessary tables and indexes.

    Note:
        This class requires the aiosqlite package. Install with `pip install aiosqlite`.
    """

    MIGRATIONS = MIGRATIONS
    VECTOR_MIGRATIONS = VECTOR_MIGRATIONS
    supports_ttl = True

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        deserializer: Optional[
            Callable[[Union[bytes, str, orjson.Fragment]], dict[str, Any]]
        ] = None,
        index: Optional[SqliteIndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
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
        self._ttl_sweeper_task: Optional[asyncio.Task[None]] = None
        self._ttl_stop_event = asyncio.Event()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: Optional[SqliteIndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> AsyncIterator["AsyncSqliteStore"]:
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
        async with self.lock:
            if not self.is_setup:
                await self.setup()

            if transaction:
                await self.conn.execute("BEGIN")

            async with self.conn.cursor() as cur:
                try:
                    yield cur
                finally:
                    if transaction:
                        await self.conn.execute("COMMIT")

    def _get_batch_GET_ops_queries(
        self, get_ops: Sequence[tuple[int, GetOp]]
    ) -> list[PreparedGetQuery]:
        """
        Build queries to fetch (and optionally refresh the TTL of) multiple keys per namespace.

        Returns a list of PreparedGetQuery objects, which may include:
        - Queries with kind='refresh' for TTL refresh operations
        - Queries with kind='get' for data retrieval operations
        """
        namespace_groups = defaultdict(list)
        refresh_ttls = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key))
            refresh_ttls[op.namespace].append(getattr(op, "refresh_ttl", False))

        results = []
        for namespace, items in namespace_groups.items():
            _, keys = zip(*items)
            this_refresh_ttls = refresh_ttls[namespace]
            refresh_ttl_any = any(this_refresh_ttls)

            # Always add the main query to get the data
            select_query = f"""
                SELECT key, value, created_at, updated_at, expires_at, ttl_minutes
                FROM store
                WHERE prefix = ? AND key IN ({",".join(["?"] * len(keys))})
            """
            select_params = (_namespace_to_text(namespace), *keys)
            results.append(
                PreparedGetQuery(select_query, select_params, namespace, items, "get")
            )

            # Add a TTL refresh query if needed
            if (
                refresh_ttl_any
                and self.ttl_config
                and self.ttl_config.get("refresh_on_read", False)
            ):
                placeholders = ",".join(["?"] * len(keys))
                update_query = f"""
                    UPDATE store
                    SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+' || ttl_minutes || ' minutes')
                    WHERE prefix = ? 
                    AND key IN ({placeholders})
                    AND ttl_minutes IS NOT NULL
                """
                update_params = (_namespace_to_text(namespace), *keys)
                results.append(
                    PreparedGetQuery(
                        update_query, update_params, namespace, items, "refresh"
                    )
                )

        return results

    def _prepare_batch_PUT_queries(
        self, put_ops: Sequence[tuple[int, PutOp]]
    ) -> tuple[
        list[tuple[str, Sequence]],
        Optional[tuple[str, Sequence[tuple[str, str, str, str]]]],
    ]:
        # Last-write wins
        dedupped_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for _, op in put_ops:
            dedupped_ops[(op.namespace, op.key)] = op

        inserts: list[PutOp] = []
        deletes: list[PutOp] = []
        for op in dedupped_ops.values():
            if op.value is None:
                deletes.append(op)
            else:
                inserts.append(op)

        queries: list[tuple[str, Sequence]] = []

        if deletes:
            namespace_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for op in deletes:
                namespace_groups[op.namespace].append(op.key)
            for namespace, keys in namespace_groups.items():
                placeholders = ",".join(["?" for _ in keys])
                query = (
                    f"DELETE FROM store WHERE prefix = ? AND key IN ({placeholders})"
                )
                params = (_namespace_to_text(namespace), *keys)
                queries.append((query, params))

        embedding_request: Optional[tuple[str, Sequence[tuple[str, str, str, str]]]] = (
            None
        )
        if inserts:
            values = []
            insertion_params = []
            vector_values = []
            embedding_request_params = []
            now = datetime.datetime.now(datetime.timezone.utc)

            # First handle main store insertions
            for op in inserts:
                if op.ttl is None:
                    expires_at = None
                else:
                    expires_at = now + datetime.timedelta(minutes=op.ttl)
                values.append("(?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)")
                insertion_params.extend(
                    [
                        _namespace_to_text(op.namespace),
                        op.key,
                        orjson.dumps(cast(dict, op.value)),
                        expires_at,
                        op.ttl,
                    ]
                )

            # Then handle embeddings if configured
            if self.index_config:
                for op in inserts:
                    if op.index is False:
                        continue
                    value = op.value
                    ns = _namespace_to_text(op.namespace)
                    k = op.key

                    if op.index is None:
                        paths = self.index_config["__tokenized_fields"]
                    else:
                        paths = [(ix, tokenize_path(ix)) for ix in op.index]

                    for path, tokenized_path in paths:
                        texts = get_text_at_path(value, tokenized_path)
                        for i, text in enumerate(texts):
                            pathname = f"{path}.{i}" if len(texts) > 1 else path
                            vector_values.append(
                                "(?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                            )
                            embedding_request_params.append((ns, k, pathname, text))

            values_str = ",".join(values)
            query = f"""
                INSERT OR REPLACE INTO store (prefix, key, value, created_at, updated_at, expires_at, ttl_minutes)
                VALUES {values_str}
            """
            queries.append((query, insertion_params))

            if vector_values:
                values_str = ",".join(vector_values)
                query = f"""
                    INSERT OR REPLACE INTO store_vectors (prefix, key, field_name, embedding, created_at, updated_at)
                    VALUES {values_str}
                """
                embedding_request = (query, embedding_request_params)

        return queries, embedding_request

    def _prepare_batch_search_queries(
        self, search_ops: Sequence[tuple[int, SearchOp]]
    ) -> tuple[
        list[tuple[str, list[Union[None, str, list[float]]]]],  # queries, params
        list[tuple[int, str]],  # idx, query_text pairs to embed
    ]:
        """
        Build per-SearchOp SQL queries (with optional TTL refresh) plus embedding requests.
        Returns:
        - queries: list of (SQL, param_list)
        - embedding_requests: list of (original_index_in_search_ops, text_query)
        """
        queries = []
        embedding_requests = []

        for idx, (_, op) in enumerate(search_ops):
            # Build filter conditions first
            filter_params = []
            filter_conditions = []
            if op.filter:
                for key, value in op.filter.items():
                    if isinstance(value, dict):
                        for op_name, val in value.items():
                            condition, filter_params_ = self._get_filter_condition(
                                key, op_name, val
                            )
                            filter_conditions.append(condition)
                            filter_params.extend(filter_params_)
                    else:
                        # SQLite json_extract returns unquoted string values
                        if isinstance(value, str):
                            filter_conditions.append(
                                "json_extract(value, '$."
                                + key
                                + "') = '"
                                + value.replace("'", "''")
                                + "'"
                            )
                        elif value is None:
                            filter_conditions.append(
                                "json_extract(value, '$." + key + "') IS NULL"
                            )
                        elif isinstance(value, bool):
                            # SQLite JSON stores booleans as integers
                            filter_conditions.append(
                                "json_extract(value, '$."
                                + key
                                + "') = "
                                + ("1" if value else "0")
                            )
                        elif isinstance(value, (int, float)):
                            filter_conditions.append(
                                "json_extract(value, '$." + key + "') = " + str(value)
                            )
                        else:
                            # For complex objects, use param binding with JSON serialization
                            filter_conditions.append(
                                "json_extract(value, '$." + key + "') = ?"
                            )
                            filter_params.append(orjson.dumps(value))

            # Vector search branch
            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))

                # Choose the similarity function and score expression based on distance type
                distance_type = self.index_config.get("distance_type", "cosine")

                if distance_type == "cosine":
                    score_expr = "1.0 - vec_distance_cosine(sv.embedding, ?)"
                elif distance_type == "l2":
                    score_expr = "vec_distance_L2(sv.embedding, ?)"
                elif distance_type == "inner_product":
                    # For inner product, we want higher values to be better, so negate the result
                    # since inner product similarity is higher when vectors are more similar
                    score_expr = "-1 * vec_distance_L1(sv.embedding, ?)"
                else:
                    # Default to cosine similarity
                    score_expr = "1.0 - vec_distance_cosine(sv.embedding, ?)"

                filter_str = (
                    ""
                    if not filter_conditions
                    else " AND " + " AND ".join(filter_conditions)
                )
                if op.namespace_prefix:
                    prefix_filter_str = f"WHERE s.prefix LIKE ? {filter_str} "
                    ns_args: Sequence = (f"{_namespace_to_text(op.namespace_prefix)}%",)
                else:
                    ns_args = ()
                    if filter_str:
                        prefix_filter_str = f"WHERE {filter_str[5:]} "
                    else:
                        prefix_filter_str = ""

                # We use a CTE to compute scores, with a SQLite-compatible approach for distinct results
                base_query = f"""
                    WITH scored AS (
                        SELECT s.prefix, s.key, s.value, s.created_at, s.updated_at, s.expires_at, s.ttl_minutes,
                            {score_expr} AS score
                        FROM store s
                        JOIN store_vectors sv ON s.prefix = sv.prefix AND s.key = sv.key
                        {prefix_filter_str}
                            ORDER BY score DESC 
                        LIMIT ?
                    ),
                    ranked AS (
                        SELECT prefix, key, value, created_at, updated_at, expires_at, ttl_minutes, score,
                                ROW_NUMBER() OVER (PARTITION BY prefix, key ORDER BY score DESC) as rn
                        FROM scored
                    )
                    SELECT prefix, key, value, created_at, updated_at, expires_at, ttl_minutes, score
                    FROM ranked
                    WHERE rn = 1
                        ORDER BY score DESC
                    LIMIT ?
                    OFFSET ?
                    """
                params = [
                    _PLACEHOLDER,  # Vector placeholder
                    *ns_args,
                    *filter_params,
                    op.limit * 2,  # Expanded limit for better results
                    op.limit,
                    op.offset,
                ]
            # Regular search branch (no vector search)
            else:
                base_query = """
                    SELECT prefix, key, value, created_at, updated_at, expires_at, ttl_minutes, NULL as score
                    FROM store
                    WHERE prefix LIKE ?
                """
                params = [f"{_namespace_to_text(op.namespace_prefix)}%"]

                if filter_conditions:
                    params.extend(filter_params)
                    base_query += " AND " + " AND ".join(filter_conditions)

                base_query += " ORDER BY updated_at DESC"
                base_query += " LIMIT ? OFFSET ?"
                params.extend([op.limit, op.offset])

                # Debug the query
                logger.debug(f"Search query: {base_query}")
                logger.debug(f"Search params: {params}")

            # Handle TTL refresh if requested
            if (
                op.refresh_ttl
                and self.ttl_config
                and self.ttl_config.get("refresh_on_read", False)
            ):
                final_sql = f"""
                    WITH search_results AS (
                        {base_query}
                    ),
                    updated AS (
                        UPDATE store
                        SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+' || ttl_minutes || ' minutes')
                        WHERE (prefix, key) IN (SELECT prefix, key FROM search_results)
                        AND ttl_minutes IS NOT NULL
                    )
                    SELECT * FROM search_results
                """
                final_params = params[:]  # copy params
            else:
                final_sql = base_query
                final_params = params

            queries.append((final_sql, final_params))

        return queries, embedding_requests

    def _get_batch_list_namespaces_queries(
        self, list_ops: Sequence[tuple[int, ListNamespacesOp]]
    ) -> list[tuple[str, Sequence]]:
        """Generate batch LIST NAMESPACES operation queries.

        Args:
            list_ops: Sequence of LIST NAMESPACES operations.

        Returns:
            List of query and parameters tuples.
        """
        queries = []
        for _, op in list_ops:
            # In SQLite, use a different approach for namespace segmentation
            if op.max_depth is not None:
                # SQLite doesn't have a built-in function for string splitting/joining with depth limit
                query = """
                    WITH RECURSIVE split_prefix(prefix, remainder, depth) AS (
                        SELECT '', prefix || '.', 0 FROM (SELECT DISTINCT prefix FROM store) 
                        UNION ALL
                        SELECT 
                            CASE WHEN instr(remainder, '.') > 0 
                                THEN prefix || CASE WHEN prefix = '' THEN '' ELSE '.' END || substr(remainder, 1, instr(remainder, '.') - 1)
                                ELSE prefix || CASE WHEN prefix = '' THEN '' ELSE '.' END || remainder
                            END,
                            CASE WHEN instr(remainder, '.') > 0 
                                THEN substr(remainder, instr(remainder, '.') + 1)
                                ELSE ''
                            END,
                            depth + 1
                        FROM split_prefix
                        WHERE remainder != '' AND depth < ?
                    )
                    SELECT DISTINCT prefix FROM split_prefix WHERE depth > 0
                """
                params = [int(op.max_depth)]
            else:
                # If no max_depth is specified, use a simpler query
                query = "SELECT DISTINCT prefix FROM store"
                params = []

            conditions = []
            if op.match_conditions:
                for condition in op.match_conditions:
                    if condition.match_type == "prefix":
                        conditions.append("prefix LIKE ?")
                        # Convert string to int by using a hashed value if an int is actually expected
                        prefix_pattern = f"{_namespace_to_text(condition.path, handle_wildcards=True)}%"
                        # Type ignoring because the code is correct but mypy expects int
                        params.append(prefix_pattern)  # type: ignore
                    elif condition.match_type == "suffix":
                        conditions.append("prefix LIKE ?")
                        # Convert string to int by using a hashed value if an int is actually expected
                        suffix_pattern = f"%{_namespace_to_text(condition.path, handle_wildcards=True)}"
                        # Type ignoring because the code is correct but mypy expects int
                        params.append(suffix_pattern)  # type: ignore
                    else:
                        logger.warning(
                            f"Unknown match_type in list_namespaces: {condition.match_type}"
                        )

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY prefix LIMIT ? OFFSET ?"
            params.extend([int(op.limit), int(op.offset)])
            queries.append((query, tuple(params)))

        return queries

    def _get_filter_condition(self, key: str, op: str, value: Any) -> tuple[str, list]:
        """Helper to generate filter conditions.

        Args:
            key: The JSON field key to filter on.
            op: The operator to use.
            value: The value to compare against.

        Returns:
            Tuple of SQL condition and parameters.
        """
        if op == "$eq":
            return f"json_extract(value, '$.{key}') = ?", [
                value if isinstance(value, str) else orjson.dumps(value)
            ]
        elif op == "$gt":
            return f"json_extract(value, '$.{key}') > ?", [str(value)]
        elif op == "$gte":
            return f"json_extract(value, '$.{key}') >= ?", [str(value)]
        elif op == "$lt":
            return f"json_extract(value, '$.{key}') < ?", [str(value)]
        elif op == "$lte":
            return f"json_extract(value, '$.{key}') <= ?", [str(value)]
        elif op == "$ne":
            return f"json_extract(value, '$.{key}') != ?", [
                value if isinstance(value, str) else orjson.dumps(value)
            ]
        else:
            raise ValueError(f"Unsupported operator: {op}")

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

    async def __aenter__(self) -> "AsyncSqliteStore":
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
            # Generate embeddings for search queries
            vectors = await self.embeddings.aembed_documents(
                [query for _, query in embedding_requests]
            )

            # Replace placeholders with actual embeddings
            for (idx, _), embedding in zip(embedding_requests, vectors):
                _params_list: list = queries[idx][1]
                for i, param in enumerate(_params_list):
                    if param is _PLACEHOLDER:
                        _params_list[i] = sqlite_vec.serialize_float32(embedding)

        for (idx, _), (query, params) in zip(search_ops, queries):
            await cur.execute(query, params)
            rows = await cur.fetchall()

            if "score" in query:  # Vector search query
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
