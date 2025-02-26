import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional, Union, cast

import aiosqlite
import orjson

from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
    get_text_at_path,
    tokenize_path,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.sqlite.base import (
    _PLACEHOLDER,
    MIGRATIONS,
    VECTOR_MIGRATIONS,
    SqliteIndexConfig,
    _decode_ns_text,
    _dot_product,
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

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        deserializer: Optional[
            Callable[[Union[bytes, str, orjson.Fragment]], dict[str, Any]]
        ] = None,
        index: Optional[SqliteIndexConfig] = None,
    ):
        """Initialize the async SQLite store.

        Args:
            conn: The SQLite database connection.
            deserializer: Optional custom deserializer function for values.
            index: Optional vector search configuration.
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

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: Optional[SqliteIndexConfig] = None,
    ) -> AsyncIterator["AsyncSqliteStore"]:
        """Create a new AsyncSqliteStore instance from a connection string.

        Args:
            conn_string: The SQLite connection string.
            index: Optional vector search configuration.

        Returns:
            An AsyncSqliteStore instance wrapped in an async context manager.
        """
        async with aiosqlite.connect(conn_string, isolation_level=None) as conn:
            yield cls(conn, index=index)

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
    ) -> list[tuple[str, tuple, tuple[str, ...], list]]:
        """Generate batch GET operation queries.

        Args:
            get_ops: Sequence of GET operations.

        Returns:
            List of query information tuples.
        """
        namespace_groups: dict[tuple[str, ...], list[tuple[int, str]]] = {}
        for idx, op in get_ops:
            if op.namespace not in namespace_groups:
                namespace_groups[op.namespace] = []
            namespace_groups[op.namespace].append((idx, op.key))
        results = []
        for namespace, items in namespace_groups.items():
            _, keys = zip(*items)
            keys_to_query = ",".join(["?" for _ in keys])
            query = f"""
                SELECT key, value, created_at, updated_at
                FROM store
                WHERE prefix = ? AND key IN ({keys_to_query})
            """
            params = (_namespace_to_text(namespace), *keys)
            results.append((query, params, namespace, items))
        return results

    def _prepare_batch_PUT_queries(
        self, put_ops: Sequence[tuple[int, PutOp]]
    ) -> tuple[
        list[tuple[str, Sequence]],
        Optional[tuple[str, Sequence[tuple[str, str, str, str]]]],
    ]:
        """Generate batch PUT operation queries.

        Args:
            put_ops: Sequence of PUT operations.

        Returns:
            Tuple of regular queries and optional embedding request.
        """
        # Last-write wins
        dedupped_ops = {}
        for _, op in put_ops:
            dedupped_ops[(op.namespace, op.key)] = op

        inserts = []
        deletes = []
        for op in dedupped_ops.values():
            if op.value is None:
                deletes.append(op)
            else:
                inserts.append(op)

        queries = []

        if deletes:
            namespace_groups: dict[tuple[str, ...], list[str]] = {}
            for op in deletes:
                if op.namespace not in namespace_groups:
                    namespace_groups[op.namespace] = []
                namespace_groups[op.namespace].append(op.key)
            for namespace, keys in namespace_groups.items():
                placeholders = ",".join(["?" for _ in keys])
                query = (
                    f"DELETE FROM store WHERE prefix = ? AND key IN ({placeholders})"
                )
                params = (_namespace_to_text(namespace), *keys)
                queries.append((query, params))

        embedding_request = None
        if inserts:
            values = []
            insertion_params = []
            vector_values = []
            embedding_request_params = []

            # First handle main store insertions
            for op in inserts:
                values.append("(?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)")
                insertion_params.extend(
                    [
                        _namespace_to_text(op.namespace),
                        op.key,
                        json.dumps(cast(dict, op.value)),
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
                INSERT OR REPLACE INTO store (prefix, key, value, created_at, updated_at)
                VALUES {values_str}
            """
            queries.append((query, tuple(insertion_params)))

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
        """Generate batch SEARCH operation queries.

        Args:
            search_ops: Sequence of SEARCH operations.

        Returns:
            Tuple of search queries and embedding requests.
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
                        filter_conditions.append(
                            "json_extract(value, '$." + key + "') = ?"
                        )
                        filter_params.append(json.dumps(value))

            # Vector search branch
            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))

                # In SQLite we'll use a simpler vector search approach
                # with dot product between query embedding and stored embeddings

                filter_str = (
                    ""
                    if not filter_conditions
                    else " AND " + " AND ".join(filter_conditions)
                )
                if op.namespace_prefix:
                    prefix_filter_str = f"WHERE s.prefix LIKE ? {filter_str} "
                    ns_args = (f"{_namespace_to_text(op.namespace_prefix)}%",)
                else:
                    # Use a completely different variable name to avoid redefinition
                    empty_args: tuple[str, ...] = ()
                    if filter_str:
                        prefix_filter_str = f"WHERE {filter_str[5:]} "
                    else:
                        prefix_filter_str = ""
                    ns_args = cast(tuple[str], empty_args)

                # Use a CTE to compute scores
                base_query = f"""
                    WITH scored AS (
                        SELECT s.prefix, s.key, s.value, s.created_at, s.updated_at, 
                               dot_product(sv.embedding, ?) AS score
                        FROM store s
                        JOIN store_vectors sv ON s.prefix = sv.prefix AND s.key = sv.key
                        {prefix_filter_str}
                        ORDER BY score DESC 
                        LIMIT ?
                    )
                    SELECT prefix, key, value, created_at, updated_at, score
                    FROM scored
                    GROUP BY prefix, key  -- SQLite equivalent of DISTINCT ON
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
                    SELECT prefix, key, value, created_at, updated_at
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

            queries.append((base_query, params))

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
            return f"json_extract(value, '$.{key}') = ?", [json.dumps(value)]
        elif op == "$gt":
            return f"json_extract(value, '$.{key}') > ?", [str(value)]
        elif op == "$gte":
            return f"json_extract(value, '$.{key}') >= ?", [str(value)]
        elif op == "$lt":
            return f"json_extract(value, '$.{key}') < ?", [str(value)]
        elif op == "$lte":
            return f"json_extract(value, '$.{key}') <= ?", [str(value)]
        elif op == "$ne":
            return f"json_extract(value, '$.{key}') != ?", [json.dumps(value)]
        else:
            raise ValueError(f"Unsupported operator: {op}")

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
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            await cur.execute(query, params)
            rows = await cur.fetchall()
            key_to_row = {
                row[0]: {
                    "key": row[0],
                    "value": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                }
                for row in rows
            }
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
                vector_params.extend([ns, k, pathname, json.dumps(vector)])

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
            # Register the dot_product function with SQLite connection
            await self.conn.create_function(
                "dot_product", 2, _dot_product, deterministic=True
            )

            # Generate embeddings for search queries
            vectors = await self.embeddings.aembed_documents(
                [query for _, query in embedding_requests]
            )

            # Replace placeholders with actual embeddings
            for (idx, _), embedding in zip(embedding_requests, vectors):
                _params_list = queries[idx][1]
                for i, param in enumerate(_params_list):
                    if param is _PLACEHOLDER:
                        _params_list[i] = json.dumps(
                            embedding
                        )  # Convert to JSON string for SQLite

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
                            "score": row[5],
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

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Union[bool, list[str], None] = None,
    ) -> None:
        """Sync operation - not supported in AsyncSqliteStore.

        Use SqliteStore for synchronous operations.
        """
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncSqliteStore are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                )
        except RuntimeError:
            pass
        # We need to handle this method specially
        # Using a dummy implementation for type checking purposes
        raise NotImplementedError("Use async methods instead")
