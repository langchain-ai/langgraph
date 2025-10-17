from __future__ import annotations

import concurrent.futures
import datetime
import logging
import re
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal, NamedTuple, cast

import orjson
import sqlite_vec  # type: ignore[import-untyped]
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
    TTLConfig,
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
    """
-- Add expires_at column to store table
ALTER TABLE store
ADD COLUMN expires_at TIMESTAMP;
""",
    """
-- Add ttl_minutes column to store table
ALTER TABLE store
ADD COLUMN ttl_minutes REAL;
""",
    """
-- Add index for efficient TTL sweeping
CREATE INDEX IF NOT EXISTS idx_store_expires_at ON store (expires_at)
WHERE expires_at IS NOT NULL;
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


def _validate_filter_key(key: str) -> None:
    """Validate that a filter key is safe for use in SQL queries.

    Args:
        key: The filter key to validate

    Raises:
        ValueError: If the key contains invalid characters that could enable SQL injection
    """
    # Allow alphanumeric characters, underscores, dots, and hyphens
    # This covers typical JSON property names while preventing SQL injection
    if not re.match(r"^[a-zA-Z0-9_.-]+$", key):
        raise ValueError(
            f"Invalid filter key: '{key}'. Filter keys must contain only alphanumeric characters, underscores, dots, and hyphens."
        )


def _json_loads(content: bytes | str | orjson.Fragment) -> Any:
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
        return orjson.loads(content)


def _row_to_item(
    namespace: tuple[str, ...],
    row: dict[str, Any],
    *,
    loader: Callable[[bytes | str | orjson.Fragment], dict[str, Any]] | None = None,
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
    loader: Callable[[bytes | str | orjson.Fragment], dict[str, Any]] | None = None,
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


class PreparedGetQuery(NamedTuple):
    query: str  # Main query to execute
    params: tuple  # Parameters for the main query
    namespace: tuple[str, ...]  # Namespace info
    items: list  # List of items this query is for
    kind: Literal["get", "refresh"]


class BaseSqliteStore:
    """Shared base class for SQLite stores."""

    MIGRATIONS = MIGRATIONS
    VECTOR_MIGRATIONS = VECTOR_MIGRATIONS
    supports_ttl = True
    index_config: SqliteIndexConfig | None = None
    ttl_config: TTLConfig | None = None

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
            _, keys = zip(*items, strict=False)
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
        tuple[str, Sequence[tuple[str, str, str, str]]] | None,
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

        embedding_request: tuple[str, Sequence[tuple[str, str, str, str]]] | None = None
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
        list[
            tuple[str, list[None | str | list[float]], bool]
        ],  # queries, params, needs_refresh
        list[tuple[int, str]],  # idx, query_text pairs to embed
    ]:
        """
        Build per-SearchOp SQL queries (with optional TTL refresh flag) plus embedding requests.
        Returns:
        - queries: list of (SQL, param_list, needs_ttl_refresh_flag)
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
                    _validate_filter_key(key)

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
                            # Complex objects (list, dict, …) – compare JSON text
                            filter_conditions.append(
                                "json_extract(value, '$." + key + "') = ?"
                            )
                            # orjson.dumps returns bytes → decode to str so SQLite sees TEXT
                            filter_params.append(orjson.dumps(value).decode())

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

            # Determine if TTL refresh is needed
            needs_ttl_refresh = bool(
                op.refresh_ttl
                and self.ttl_config
                and self.ttl_config.get("refresh_on_read", False)
            )

            # The base_query is now the final_sql, and we pass the refresh flag
            final_sql = base_query
            final_params = params

            queries.append((final_sql, final_params, needs_ttl_refresh))

        return queries, embedding_requests

    def _get_batch_list_namespaces_queries(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
    ) -> list[tuple[str, Sequence]]:
        queries: list[tuple[str, Sequence]] = []

        for _, op in list_ops:
            where_clauses: list[str] = []
            params: list[Any] = []

            if op.match_conditions:
                for cond in op.match_conditions:
                    if cond.match_type == "prefix":
                        where_clauses.append("prefix LIKE ?")
                        params.append(
                            f"{_namespace_to_text(cond.path, handle_wildcards=True)}%"
                        )
                    elif cond.match_type == "suffix":
                        where_clauses.append("prefix LIKE ?")
                        params.append(
                            f"%{_namespace_to_text(cond.path, handle_wildcards=True)}"
                        )
                    else:
                        logger.warning(
                            "Unknown match_type in list_namespaces: %s", cond.match_type
                        )

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            if op.max_depth is not None:
                query = f"""
                    WITH RECURSIVE split(original, truncated, remainder, depth) AS (
                        SELECT
                            prefix          AS original,
                            ''              AS truncated,
                            prefix          AS remainder,
                            0               AS depth
                        FROM (SELECT DISTINCT prefix FROM store {where_sql})

                        UNION ALL

                        SELECT
                            original,
                            CASE
                                WHEN depth = 0
                                    THEN substr(remainder,
                                                1,
                                                CASE
                                                    WHEN instr(remainder, '.') > 0
                                                        THEN instr(remainder, '.') - 1
                                                    ELSE length(remainder)
                                                END)
                                ELSE
                                    truncated || '.' ||
                                    substr(remainder,
                                        1,
                                        CASE
                                            WHEN instr(remainder, '.') > 0
                                                THEN instr(remainder, '.') - 1
                                            ELSE length(remainder)
                                        END)
                            END                              AS truncated,
                            CASE
                                WHEN instr(remainder, '.') > 0
                                    THEN substr(remainder, instr(remainder, '.') + 1)
                                ELSE ''
                            END                              AS remainder,
                            depth + 1                       AS depth
                        FROM split
                        WHERE remainder <> ''
                            AND depth < ?
                    )
                    SELECT DISTINCT truncated AS prefix
                    FROM split
                    WHERE depth = ? OR remainder = ''
                    ORDER BY prefix
                    LIMIT ? OFFSET ?
                """
                params.extend([op.max_depth, op.max_depth, op.limit, op.offset])

            else:
                query = f"""
                    SELECT DISTINCT prefix
                    FROM store
                    {where_sql}
                    ORDER BY prefix
                    LIMIT ? OFFSET ?
                """
                params.extend([op.limit, op.offset])

            queries.append((query, tuple(params)))

        return queries

    def _get_filter_condition(self, key: str, op: str, value: Any) -> tuple[str, list]:
        """Helper to generate filter conditions."""
        _validate_filter_key(key)

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
                return f"json_extract(value, '$.{key}') = ?", [orjson.dumps(value)]
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
                return f"json_extract(value, '$.{key}') > ?", [orjson.dumps(value)]
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
                return f"json_extract(value, '$.{key}') >= ?", [orjson.dumps(value)]
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
                return f"json_extract(value, '$.{key}') < ?", [orjson.dumps(value)]
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
                return f"json_extract(value, '$.{key}') <= ?", [orjson.dumps(value)]
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
                return f"json_extract(value, '$.{key}') != ?", [orjson.dumps(value)]
        else:
            raise ValueError(f"Unsupported operator: {op}")


class SqliteStore(BaseSqliteStore, BaseStore):
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
            results = store.search(("docs",), query="programming guides", limit=2)
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
    supports_ttl = True

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        deserializer: Callable[[bytes | str | orjson.Fragment], dict[str, Any]]
        | None = None,
        index: SqliteIndexConfig | None = None,
        ttl: TTLConfig | None = None,
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
        self.ttl_config = ttl
        self._ttl_sweeper_thread: threading.Thread | None = None
        self._ttl_stop_event = threading.Event()

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
            _, keys = zip(*items, strict=False)
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

    def _get_filter_condition(self, key: str, op: str, value: Any) -> tuple[str, list]:
        """Helper to generate filter conditions."""
        _validate_filter_key(key)

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
                return f"json_extract(value, '$.{key}') = ?", [orjson.dumps(value)]
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
                return f"json_extract(value, '$.{key}') > ?", [orjson.dumps(value)]
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
                return f"json_extract(value, '$.{key}') >= ?", [orjson.dumps(value)]
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
                return f"json_extract(value, '$.{key}') < ?", [orjson.dumps(value)]
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
                return f"json_extract(value, '$.{key}') <= ?", [orjson.dumps(value)]
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
                return f"json_extract(value, '$.{key}') != ?", [orjson.dumps(value)]
        else:
            raise ValueError(f"Unsupported operator: {op}")

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: SqliteIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> Iterator[SqliteStore]:
        """Create a new SqliteStore instance from a connection string.

        Args:
            conn_string (str): The SQLite connection string.
            index (Optional[SqliteIndexConfig]): The index configuration for the store.
            ttl (Optional[TTLConfig]): The time-to-live configuration for the store.

        Returns:
            SqliteStore: A new SqliteStore instance.
        """
        conn = sqlite3.connect(
            conn_string,
            check_same_thread=False,
            isolation_level=None,  # autocommit mode
        )
        try:
            yield cls(conn, index=index, ttl=ttl)
        finally:
            conn.close()

    @contextmanager
    def _cursor(self, *, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
        """Create a database cursor as a context manager.

        Args:
            transaction (bool): whether to use transaction for the DB operations
        """
        if not self.is_setup:
            self.setup()
        with self.lock:
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

        with self.lock:
            if self.is_setup:
                return
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
                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                self.conn.enable_load_extension(False)
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

    def sweep_ttl(self) -> int:
        """Delete expired store items based on TTL.

        Returns:
            int: The number of deleted items.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                DELETE FROM store
                WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                """
            )
            deleted_count = cur.rowcount
            return deleted_count

    def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> concurrent.futures.Future[None]:
        """Periodically delete expired store items based on TTL.

        Returns:
            Future that can be waited on or cancelled.
        """
        if not self.ttl_config:
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return future

        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            logger.info("TTL sweeper thread is already running")
            # Return a future that can be used to cancel the existing thread
            future = concurrent.futures.Future()
            future.add_done_callback(
                lambda f: self._ttl_stop_event.set() if f.cancelled() else None
            )
            return future

        self._ttl_stop_event.clear()

        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")

        future = concurrent.futures.Future()

        def _sweep_loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break

                    try:
                        expired_items = self.sweep_ttl()
                        if expired_items > 0:
                            logger.info(f"Store swept {expired_items} expired items")
                    except Exception as exc:
                        logger.exception(
                            "Store TTL sweep iteration failed", exc_info=exc
                        )
                future.set_result(None)
            except Exception as exc:
                future.set_exception(exc)

        thread = threading.Thread(target=_sweep_loop, daemon=True, name="ttl-sweeper")
        self._ttl_sweeper_thread = thread
        thread.start()

        future.add_done_callback(
            lambda f: self._ttl_stop_event.set() if f.cancelled() else None
        )
        return future

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper thread if it's running.

        Args:
            timeout: Maximum time to wait for the thread to stop, in seconds.
                If `None`, wait indefinitely.

        Returns:
            bool: True if the thread was successfully stopped or wasn't running,
                False if the timeout was reached before the thread stopped.
        """
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True

        logger.info("Stopping TTL sweeper thread")
        self._ttl_stop_event.set()

        self._ttl_sweeper_thread.join(timeout)
        success = not self._ttl_sweeper_thread.is_alive()

        if success:
            self._ttl_sweeper_thread = None
            logger.info("TTL sweeper thread stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper thread to stop")

        return success

    def __del__(self) -> None:
        """Ensure the TTL sweeper thread is stopped when the object is garbage collected."""
        if hasattr(self, "_ttl_stop_event") and hasattr(self, "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)

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
                        cur.execute(query.query, query.params)
                    except Exception as e:
                        raise ValueError(
                            f"Error executing TTL refresh: \n{query.query}\n{query.params}\n{e}"
                        ) from e

            # Then execute GET queries and process results
            for query in queries:
                if query.kind == "get":
                    try:
                        cur.execute(query.query, query.params)
                    except Exception as e:
                        raise ValueError(
                            f"Error executing GET query: \n{query.query}\n{query.params}\n{e}"
                        ) from e

                    rows = cur.fetchall()
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
            for (ns, k, pathname, _), vector in zip(txt_params, vectors, strict=False):
                vector_params.extend(
                    [ns, k, pathname, sqlite_vec.serialize_float32(vector)]
                )

            queries.append((query, vector_params))

        for query, params in queries:
            cur.execute(query, params)

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: sqlite3.Cursor,
    ) -> None:
        prepared_queries, embedding_requests = self._prepare_batch_search_queries(
            search_ops
        )

        # Setup similarity functions if they don't exist
        if embedding_requests and self.embeddings:
            # Generate embeddings for search queries
            embeddings = self.embeddings.embed_documents(
                [query for _, query in embedding_requests]
            )

            # Replace placeholders with actual embeddings
            for (embed_req_idx, _), embedding in zip(
                embedding_requests, embeddings, strict=False
            ):
                if embed_req_idx < len(prepared_queries):
                    _params_list: list = prepared_queries[embed_req_idx][1]
                    for i, param in enumerate(_params_list):
                        if param is _PLACEHOLDER:
                            _params_list[i] = sqlite_vec.serialize_float32(embedding)
                else:
                    logger.warning(
                        f"Embedding request index {embed_req_idx} out of bounds for prepared_queries."
                    )

        for (original_op_idx, _), (query, params, needs_refresh) in zip(
            search_ops, prepared_queries, strict=False
        ):
            cur.execute(query, params)
            rows = cur.fetchall()

            if needs_refresh and rows and self.ttl_config:
                keys_to_refresh = []
                for row_data in rows:
                    keys_to_refresh.append((row_data[0], row_data[1]))

                if keys_to_refresh:
                    updates_by_prefix = defaultdict(list)
                    for prefix_text, key_text in keys_to_refresh:
                        updates_by_prefix[prefix_text].append(key_text)

                    for prefix_text, key_list in updates_by_prefix.items():
                        placeholders = ",".join(["?"] * len(key_list))
                        update_query = f"""
                            UPDATE store
                            SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+' || ttl_minutes || ' minutes')
                            WHERE prefix = ? AND key IN ({placeholders}) AND ttl_minutes IS NOT NULL
                        """
                        update_params = (prefix_text, *key_list)
                        try:
                            cur.execute(update_query, update_params)
                        except Exception as e:
                            logger.error(
                                f"Error during TTL refresh update for search: {e}"
                            )

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

            results[original_op_idx] = items

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: sqlite3.Cursor,
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        for (query, params), (idx, _) in zip(queries, list_ops, strict=False):
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
    tokenized: list[tuple[str, Literal["$"] | list[str]]] = []
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


_PLACEHOLDER = object()
