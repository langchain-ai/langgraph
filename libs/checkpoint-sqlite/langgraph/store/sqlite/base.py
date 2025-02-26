import json
import logging
import math
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Callable, Literal, Optional, Union, cast

import orjson

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

_AIO_ERROR_MSG = (
    "The SqliteStore does not support async methods. "
    "Consider using AsyncSqliteStore instead.\n"
    "from langgraph.store.sqlite.aio import AsyncSqliteStore\n"
)

logger = logging.getLogger(__name__)

MIGRATIONS = [
    """
CREATE TABLE IF NOT EXISTS store (
    -- 'prefix' represents the doc's 'namespace'
    prefix text NOT NULL,
    key text NOT NULL,
    value text NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prefix, key)
);
""",
    """
-- For faster lookups by prefix
CREATE INDEX IF NOT EXISTS store_prefix_idx ON store (prefix);
""",
]

VECTOR_MIGRATIONS = [
    """
CREATE TABLE IF NOT EXISTS store_vectors (
    prefix text NOT NULL,
    key text NOT NULL,
    field_name text NOT NULL,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prefix, key, field_name),
    FOREIGN KEY (prefix, key) REFERENCES store(prefix, key) ON DELETE CASCADE
);
""",
]


class SqliteIndexConfig(IndexConfig):
    """Configuration for vector embeddings in SQLite store."""

    pass


def _namespace_to_text(
    namespace: tuple[str, ...], handle_wildcards: bool = False
) -> str:
    """Convert namespace tuple to text string."""
    if handle_wildcards:
        namespace = tuple("%" if val == "*" else val for val in namespace)
    return ".".join(namespace)


def _decode_ns_text(namespace: str) -> tuple[str, ...]:
    """Convert namespace string to tuple."""
    return tuple(namespace.split("."))


def _json_loads(content: Union[bytes, str, orjson.Fragment]) -> Any:
    if isinstance(content, orjson.Fragment):
        if hasattr(content, "buf"):
            content = content.buf
        else:
            if isinstance(content.contents, bytes):
                content = content.contents
            else:
                content = content.contents.encode()
        return orjson.loads(cast(bytes, content))
    elif isinstance(content, bytes):
        return orjson.loads(content)
    else:
        return json.loads(content)


def _row_to_item(
    namespace: tuple[str, ...],
    row: dict[str, Any],
    *,
    loader: Optional[
        Callable[[Union[bytes, str, orjson.Fragment]], dict[str, Any]]
    ] = None,
) -> Item:
    """Convert a row from the database into an Item."""
    val = row["value"]
    if not isinstance(val, dict):
        val = (loader or _json_loads)(val)

    kwargs = {
        "key": row["key"],
        "namespace": namespace,
        "value": val,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

    return Item(**kwargs)


def _row_to_search_item(
    namespace: tuple[str, ...],
    row: dict[str, Any],
    *,
    loader: Optional[
        Callable[[Union[bytes, str, orjson.Fragment]], dict[str, Any]]
    ] = None,
) -> SearchItem:
    """Convert a row from the database into a SearchItem."""
    loader = loader or _json_loads
    val = row["value"]
    score = row.get("score")
    if score is not None:
        try:
            score = float(score)
        except ValueError:
            logger.warning("Invalid score: %s", score)
            score = None
    return SearchItem(
        value=val if isinstance(val, dict) else loader(val),
        key=row["key"],
        namespace=namespace,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        score=score,
    )


def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    tot = 0
    for idx, op in enumerate(ops):
        grouped_ops[type(op)].append((idx, op))
        tot += 1
    return grouped_ops, tot


class SqliteStore(BaseStore):
    """SQLite-backed store with optional vector search capabilities.

    Examples:
        Basic setup and usage:
        ```python
        from langgraph.store.sqlite import SqliteStore
        import sqlite3

        conn = sqlite3.connect(":memory:")
        store = SqliteStore(conn)
        store.setup()  # Run migrations. Done once

        # Store and retrieve data
        store.put(("users", "123"), "prefs", {"theme": "dark"})
        item = store.get(("users", "123"), "prefs")
        ```

        Or using the convenient from_conn_string helper:
        ```python
        from langgraph.store.sqlite import SqliteStore

        with SqliteStore.from_conn_string(":memory:") as store:
            store.setup()

            # Store and retrieve data
            store.put(("users", "123"), "prefs", {"theme": "dark"})
            item = store.get(("users", "123"), "prefs")
        ```

        Vector search using LangChain embeddings:
        ```python
        from langchain.embeddings import OpenAIEmbeddings
        from langgraph.store.sqlite import SqliteStore

        with SqliteStore.from_conn_string(
            ":memory:",
            index={
                "dims": 1536,
                "embed": OpenAIEmbeddings(),
                "fields": ["text"]  # specify which fields to embed
            }
        ) as store:
            store.setup()  # Run migrations

            # Store documents
            store.put(("docs",), "doc1", {"text": "Python tutorial"})
            store.put(("docs",), "doc2", {"text": "TypeScript guide"})
            store.put(("docs",), "doc3", {"text": "Other guide"}, index=False)  # don't index

            # Search by similarity
            results = store.search(("docs",), "programming guides", limit=2)
        ```

    Note:
        Semantic search is disabled by default. You can enable it by providing an `index` configuration
        when creating the store. Without this configuration, all `index` arguments passed to
        `put` or `aput` will have no effect.

    Warning:
        Make sure to call `setup()` before first use to create necessary tables and indexes.
    """

    MIGRATIONS = MIGRATIONS
    VECTOR_MIGRATIONS = VECTOR_MIGRATIONS

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        deserializer: Optional[
            Callable[[Union[bytes, str, orjson.Fragment]], dict[str, Any]]
        ] = None,
        index: Optional[SqliteIndexConfig] = None,
    ):
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.lock = threading.Lock()
        self.is_setup = False
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: Optional[SqliteIndexConfig] = None,
    ) -> Iterator["SqliteStore"]:
        """Create a new SqliteStore instance from a connection string.

        Args:
            conn_string (str): The SQLite connection string.
            index (Optional[SqliteIndexConfig]): The index configuration for the store.

        Returns:
            SqliteStore: A new SqliteStore instance.
        """
        conn = sqlite3.connect(
            conn_string,
            check_same_thread=False,
            isolation_level=None,  # autocommit mode
        )
        try:
            yield cls(conn, index=index)
        finally:
            conn.close()

    @contextmanager
    def _cursor(self, *, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
        """Create a database cursor as a context manager.

        Args:
            transaction (bool): whether to use transaction for the DB operations
        """
        with self.lock:
            if not self.is_setup:
                self.setup()

            if transaction:
                self.conn.execute("BEGIN")

            cur = self.conn.cursor()
            try:
                yield cur
            finally:
                if transaction:
                    self.conn.execute("COMMIT")
                cur.close()

    def setup(self) -> None:
        """Set up the store database.

        This method creates the necessary tables in the SQLite database if they don't
        already exist and runs database migrations. It should be called before first use.
        """
        if self.is_setup:
            return

        with self.lock:
            # Create migrations table if it doesn't exist
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_migrations (
                    v INTEGER PRIMARY KEY
                )
                """
            )

            # Check current migration version
            cur = self.conn.execute(
                "SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                version = -1
            else:
                version = row[0]

            # Apply migrations
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                self.conn.executescript(sql)
                self.conn.execute("INSERT INTO store_migrations (v) VALUES (?)", (v,))

            # Apply vector migrations if index config is provided
            if self.index_config:
                # Create vector migrations table if it doesn't exist
                self.conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS vector_migrations (
                        v INTEGER PRIMARY KEY
                    )
                    """
                )

                # Check current vector migration version
                cur = self.conn.execute(
                    "SELECT v FROM vector_migrations ORDER BY v DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row is None:
                    version = -1
                else:
                    version = row[0]

                # Apply vector migrations
                for v, sql in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    self.conn.executescript(sql)
                    self.conn.execute(
                        "INSERT INTO vector_migrations (v) VALUES (?)", (v,)
                    )

            self.is_setup = True

    def _get_batch_GET_ops_queries(
        self, get_ops: Sequence[tuple[int, GetOp]]
    ) -> list[tuple[str, tuple, tuple[str, ...], list]]:
        namespace_groups = defaultdict(list)
        for idx, op in get_ops:
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
                            filter_params.append(json.dumps(value))

            # Vector search branch
            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))

                # Choose the similarity function and score expression based on distance type
                distance_type = self.index_config.get("distance_type", "cosine")

                if distance_type == "cosine":
                    score_expr = "cosine_similarity(sv.embedding, ?)"
                elif distance_type == "l2":
                    score_expr = "-1 * l2_distance(sv.embedding, ?)"
                elif distance_type == "inner_product":
                    score_expr = "dot_product(sv.embedding, ?)"
                else:
                    # Default to cosine similarity
                    score_expr = "cosine_similarity(sv.embedding, ?)"

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
                        SELECT s.prefix, s.key, s.value, s.created_at, s.updated_at, 
                               {score_expr} AS score
                        FROM store s
                        JOIN store_vectors sv ON s.prefix = sv.prefix AND s.key = sv.key
                        {prefix_filter_str}
                        ORDER BY score DESC 
                        LIMIT ?
                    ),
                    ranked AS (
                        SELECT prefix, key, value, created_at, updated_at, score,
                               ROW_NUMBER() OVER (PARTITION BY prefix, key ORDER BY score DESC) as rn
                        FROM scored
                    )
                    SELECT prefix, key, value, created_at, updated_at, score
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

                # Debug the query
                logger.debug(f"Search query: {base_query}")
                logger.debug(f"Search params: {params}")

            queries.append((base_query, params))

        return queries, embedding_requests

    def _get_batch_list_namespaces_queries(
        self, list_ops: Sequence[tuple[int, ListNamespacesOp]]
    ) -> list[tuple[str, Sequence]]:
        queries: list[tuple[str, Sequence]] = []
        for _, op in list_ops:
            # In SQLite, we need to use a different approach for namespace segmentation
            # since there's no direct equivalent to PostgreSQL's string aggregation
            if op.max_depth is not None:
                # SQLite doesn't have a built-in function for string splitting/joining with depth limit
                # We'll use a more basic approach
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
                params: list[Any] = [op.max_depth]
            else:
                # If no max_depth is specified, we can just use a simpler query
                query = "SELECT DISTINCT prefix FROM store"
                params = []

            conditions = []
            if op.match_conditions:
                for condition in op.match_conditions:
                    if condition.match_type == "prefix":
                        conditions.append("prefix LIKE ?")
                        params.append(
                            f"{_namespace_to_text(condition.path, handle_wildcards=True)}%"
                        )
                    elif condition.match_type == "suffix":
                        conditions.append("prefix LIKE ?")
                        params.append(
                            f"%{_namespace_to_text(condition.path, handle_wildcards=True)}"
                        )
                    else:
                        logger.warning(
                            f"Unknown match_type in list_namespaces: {condition.match_type}"
                        )

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY prefix LIMIT ? OFFSET ?"
            params.extend([op.limit, op.offset])
            queries.append((query, tuple(params)))

        return queries

    def _get_filter_condition(self, key: str, op: str, value: Any) -> tuple[str, list]:
        """Helper to generate filter conditions."""
        # We need to properly format values for SQLite JSON extraction comparison
        if op == "$eq":
            if isinstance(value, str):
                # Direct string comparison with proper quoting for unquoted json_extract result
                return (
                    f"json_extract(value, '$.{key}') = '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            elif value is None:
                return f"json_extract(value, '$.{key}') IS NULL", []
            elif isinstance(value, bool):
                # SQLite JSON stores booleans as integers
                return f"json_extract(value, '$.{key}') = {1 if value else 0}", []
            elif isinstance(value, (int, float)):
                return f"json_extract(value, '$.{key}') = {value}", []
            else:
                return f"json_extract(value, '$.{key}') = ?", [json.dumps(value)]
        elif op == "$gt":
            # For numeric values, SQLite needs to compare as numbers, not strings
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) > {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') > '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') > ?", [json.dumps(value)]
        elif op == "$gte":
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) >= {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') >= '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') >= ?", [json.dumps(value)]
        elif op == "$lt":
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) < {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') < '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') < ?", [json.dumps(value)]
        elif op == "$lte":
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) <= {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') <= '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') <= ?", [json.dumps(value)]
        elif op == "$ne":
            if isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') != '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            elif value is None:
                return f"json_extract(value, '$.{key}') IS NOT NULL", []
            elif isinstance(value, bool):
                return f"json_extract(value, '$.{key}') != {1 if value else 0}", []
            elif isinstance(value, (int, float)):
                return f"json_extract(value, '$.{key}') != {value}", []
            else:
                return f"json_extract(value, '$.{key}') != ?", [json.dumps(value)]
        else:
            raise ValueError(f"Unsupported operator: {op}")

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations.

        Args:
            ops (Iterable[Op]): List of operations to execute

        Returns:
            list[Result]: Results of the operations
        """
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        with self._cursor(transaction=True) as cur:
            if GetOp in grouped_ops:
                self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results, cur
                )

            if SearchOp in grouped_ops:
                self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )
            if PutOp in grouped_ops:
                self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), cur
                )

        return results

    def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: sqlite3.Cursor,
    ) -> None:
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            cur.execute(query, params)
            rows = cur.fetchall()
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

    def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        cur: sqlite3.Cursor,
    ) -> None:
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
            vectors = self.embeddings.embed_documents(
                [param[-1] for param in txt_params]
            )

            # Convert vectors to SQLite-friendly format
            vector_params = []
            for (ns, k, pathname, _), vector in zip(txt_params, vectors):
                vector_params.extend([ns, k, pathname, json.dumps(vector)])

            queries.append((query, vector_params))

        for query, params in queries:
            cur.execute(query, params)

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: sqlite3.Cursor,
    ) -> None:
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)

        # Setup similarity functions if they don't exist
        if embedding_requests and self.embeddings:
            # Register the dot_product function if not registered already
            self.conn.create_function(
                "dot_product", 2, _dot_product, deterministic=True
            )

            # Register the cosine_similarity function if not registered already
            self.conn.create_function(
                "cosine_similarity", 2, _cosine_similarity, deterministic=True
            )

            # Register the l2_distance function if not registered already
            self.conn.create_function(
                "l2_distance", 2, _l2_distance, deterministic=True
            )

            # Register the neg_l2_distance function if not registered already
            self.conn.create_function(
                "neg_l2_distance", 2, _neg_l2_distance, deterministic=True
            )

            # Register the vector_magnitude function if not registered already
            self.conn.create_function(
                "vector_magnitude", 1, _vector_magnitude, deterministic=True
            )

            # Generate embeddings for search queries
            embeddings = self.embeddings.embed_documents(
                [query for _, query in embedding_requests]
            )

            # Replace placeholders with actual embeddings
            for (idx, _), embedding in zip(embedding_requests, embeddings):
                _params_list = queries[idx][1]
                for i, param in enumerate(_params_list):
                    if param is _PLACEHOLDER:
                        _params_list[i] = json.dumps(
                            embedding
                        )  # Convert to JSON string for SQLite

        for (idx, _), (query, params) in zip(search_ops, queries):
            cur.execute(query, params)
            rows = cur.fetchall()

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

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: sqlite3.Cursor,
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        for (query, params), (idx, _) in zip(queries, list_ops):
            cur.execute(query, params)
            results[idx] = [_decode_ns_text(row[0]) for row in cur.fetchall()]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Async batch operation - not supported in SqliteStore.

        Use AsyncSqliteStore for async operations.
        """
        raise NotImplementedError(_AIO_ERROR_MSG)


# Helper functions


def _ensure_index_config(
    index_config: SqliteIndexConfig,
) -> tuple[Any, SqliteIndexConfig]:
    """Process and validate index configuration."""
    index_config = index_config.copy()
    tokenized: list[tuple[str, Union[Literal["$"], list[str]]]] = []
    tot = 0
    text_fields = index_config.get("text_fields") or ["$"]
    if isinstance(text_fields, str):
        text_fields = [text_fields]
    if not isinstance(text_fields, list):
        raise ValueError(f"Text fields must be a list or a string. Got {text_fields}")
    for p in text_fields:
        if p == "$":
            tokenized.append((p, "$"))
            tot += 1
        else:
            toks = tokenize_path(p)
            tokenized.append((p, toks))
            tot += len(toks)
    index_config["__tokenized_fields"] = tokenized
    index_config["__estimated_num_vectors"] = tot
    embeddings = ensure_embeddings(
        index_config.get("embed"),
    )
    return embeddings, index_config


def _dot_product(vec1_json: str, vec2_json: str) -> float:
    """SQLite function to compute dot product between two vectors stored as JSON strings."""
    vec1 = json.loads(vec1_json)
    vec2 = json.loads(vec2_json)

    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))


def _vector_magnitude(vec_json: str) -> float:
    """SQLite function to compute the magnitude of a vector stored as JSON string."""
    vec = json.loads(vec_json)
    return sum(v**2 for v in vec) ** 0.5


def _cosine_similarity(vec1_json: str, vec2_json: str) -> float:
    """Compute cosine similarity between two vectors stored as JSON strings.

    Returns the cosine similarity in the full range [-1, 1], where 1 is identical,
    -1 is completely opposite, and 0 indicates orthogonality.
    """
    vec1 = json.loads(vec1_json)
    vec2 = json.loads(vec2_json)

    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

    magnitude1 = math.sqrt(sum(v * v for v in vec1))
    magnitude2 = math.sqrt(sum(v * v for v in vec2))

    if magnitude1 < 1e-10 or magnitude2 < 1e-10:
        return 0.0
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    result = dot_product / (magnitude1 * magnitude2)
    return result


def _l2_distance(vec1_json: str, vec2_json: str) -> float:
    """SQLite function to compute L2 distance between two vectors stored as JSON strings."""
    vec1 = json.loads(vec1_json)
    vec2 = json.loads(vec2_json)

    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

    return sum((v1 - v2) ** 2 for v1, v2 in zip(vec1, vec2)) ** 0.5


def _neg_l2_distance(vec1_json: str, vec2_json: str) -> float:
    """Returns negative L2 distance for ranking purposes."""
    return -_l2_distance(vec1_json, vec2_json)


_PLACEHOLDER = object()
