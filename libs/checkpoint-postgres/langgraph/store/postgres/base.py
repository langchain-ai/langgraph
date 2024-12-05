import asyncio
import json
import logging
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    cast,
)

import orjson
from psycopg import Capabilities, Connection, Cursor, Pipeline
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from typing_extensions import TypedDict

from langgraph.checkpoint.postgres import _ainternal as _ainternal
from langgraph.checkpoint.postgres import _internal as _pg_internal
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

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class Migration(NamedTuple):
    """A database migration with optional conditions and parameters."""

    sql: str
    params: Optional[dict[str, Any]] = None
    condition: Optional[Callable[["BasePostgresStore"], bool]] = None


MIGRATIONS: Sequence[str] = [
    """
CREATE TABLE IF NOT EXISTS store (
    -- 'prefix' represents the doc's 'namespace'
    prefix text NOT NULL,
    key text NOT NULL,
    value jsonb NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prefix, key)
);
""",
    """
-- For faster lookups by prefix
CREATE INDEX CONCURRENTLY IF NOT EXISTS store_prefix_idx ON store USING btree (prefix text_pattern_ops);
""",
]

VECTOR_MIGRATIONS: Sequence[Migration] = [
    Migration(
        """
CREATE EXTENSION IF NOT EXISTS vector;
""",
    ),
    Migration(
        """
CREATE TABLE IF NOT EXISTS store_vectors (
    prefix text NOT NULL,
    key text NOT NULL,
    field_name text NOT NULL,
    embedding %(vector_type)s(%(dims)s),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prefix, key, field_name),
    FOREIGN KEY (prefix, key) REFERENCES store(prefix, key) ON DELETE CASCADE
);
""",
        params={
            "dims": lambda store: store.index_config["dims"],
            "vector_type": lambda store: (
                cast(PostgresIndexConfig, store.index_config)
                .get("ann_index_config", {})
                .get("vector_type", "vector")
            ),
        },
    ),
    Migration(
        """
CREATE INDEX CONCURRENTLY IF NOT EXISTS store_vectors_embedding_idx ON store_vectors 
    USING %(index_type)s (embedding %(ops)s)%(index_params)s;
""",
        condition=lambda store: bool(
            store.index_config and _get_index_params(store)[0] != "flat"
        ),
        params={
            "index_type": lambda store: _get_index_params(store)[0],
            "ops": lambda store: _get_vector_type_ops(store),
            "index_params": lambda store: (
                " WITH ("
                + ", ".join(f"{k}={v}" for k, v in _get_index_params(store)[1].items())
                + ")"
                if _get_index_params(store)[1]
                else ""
            ),
        },
    ),
]


C = TypeVar("C", bound=Union[_pg_internal.Conn, _ainternal.Conn])


class PoolConfig(TypedDict, total=False):
    """Connection pool settings for PostgreSQL connections.

    Controls connection lifecycle and resource utilization:
    - Small pools (1-5) suit low-concurrency workloads
    - Larger pools handle concurrent requests but consume more resources
    - Setting max_size prevents resource exhaustion under load
    """

    min_size: int
    """Minimum number of connections maintained in the pool. Defaults to 1."""

    max_size: Optional[int]
    """Maximum number of connections allowed in the pool. None means unlimited."""

    kwargs: dict
    """Additional connection arguments passed to each connection in the pool.
    
    Default kwargs set automatically:
    - autocommit: True
    - prepare_threshold: 0
    - row_factory: dict_row
    """


class ANNIndexConfig(TypedDict, total=False):
    """Configuration for vector index in PostgreSQL store."""

    kind: Literal["hnsw", "ivfflat", "flat"]
    """Type of index to use: 'hnsw' for Hierarchical Navigable Small World, or 'ivfflat' for Inverted File Flat."""
    vector_type: Literal["vector", "halfvec"]
    """Type of vector storage to use.
    Options:
    - 'vector': Regular vectors (default)
    - 'halfvec': Half-precision vectors for reduced memory usage
    """


class HNSWConfig(ANNIndexConfig, total=False):
    """Configuration for HNSW (Hierarchical Navigable Small World) index."""

    kind: Literal["hnsw"]  # type: ignore[misc]
    m: int
    """Maximum number of connections per layer. Default is 16."""
    ef_construction: int
    """Size of dynamic candidate list for index construction. Default is 64."""


class IVFFlatConfig(ANNIndexConfig, total=False):
    """IVFFlat index divides vectors into lists, and then searches a subset of those lists that are closest to the query vector. It has faster build times and uses less memory than HNSW, but has lower query performance (in terms of speed-recall tradeoff).

    Three keys to achieving good recall are:
    1. Create the index after the table has some data
    2. Choose an appropriate number of lists - a good place to start is rows / 1000 for up to 1M rows and sqrt(rows) for over 1M rows
    3. When querying, specify an appropriate number of probes (higher is better for recall, lower is better for speed) - a good place to start is sqrt(lists)
    """

    kind: Literal["ivfflat"]  # type: ignore[misc]
    nlist: int
    """Number of inverted lists (clusters) for IVF index.
    
    Determines the number of clusters used in the index structure.
    Higher values can improve search speed but increase index size and build time.
    Typically set to the square root of the number of vectors in the index.
    """


class PostgresIndexConfig(IndexConfig, total=False):
    """Configuration for vector embeddings in PostgreSQL store with pgvector-specific options.

    Extends EmbeddingConfig with additional configuration for pgvector index and vector types.
    """

    ann_index_config: ANNIndexConfig
    """Specific configuration for the chosen index type (HNSW or IVF Flat)."""
    distance_type: Literal["l2", "inner_product", "cosine"]
    """Distance metric to use for vector similarity search:
    - 'l2': Euclidean distance
    - 'inner_product': Dot product
    - 'cosine': Cosine similarity
    """


class BasePostgresStore(Generic[C]):
    MIGRATIONS = MIGRATIONS
    VECTOR_MIGRATIONS = VECTOR_MIGRATIONS
    conn: C
    _deserializer: Optional[Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]]
    index_config: Optional[PostgresIndexConfig]

    def _get_batch_GET_ops_queries(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
    ) -> list[tuple[str, tuple, tuple[str, ...], list]]:
        namespace_groups = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key))
        results = []
        for namespace, items in namespace_groups.items():
            _, keys = zip(*items)
            keys_to_query = ",".join(["%s"] * len(keys))
            query = f"""
                SELECT key, value, created_at, updated_at
                FROM store
                WHERE prefix = %s AND key IN ({keys_to_query})
            """
            params = (_namespace_to_text(namespace), *keys)
            results.append((query, params, namespace, items))
        return results

    def _prepare_batch_PUT_queries(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
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
                placeholders = ",".join(["%s"] * len(keys))
                query = (
                    f"DELETE FROM store WHERE prefix = %s AND key IN ({placeholders})"
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
                values.append("(%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)")
                insertion_params.extend(
                    [
                        _namespace_to_text(op.namespace),
                        op.key,
                        Jsonb(cast(dict, op.value)),
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
                                "(%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                            )
                            embedding_request_params.append((ns, k, pathname, text))

            values_str = ",".join(values)
            query = f"""
                INSERT INTO store (prefix, key, value, created_at, updated_at)
                VALUES {values_str}
                ON CONFLICT (prefix, key) DO UPDATE
                SET value = EXCLUDED.value,
                    updated_at = CURRENT_TIMESTAMP
            """
            queries.append((query, insertion_params))

            if vector_values:
                values_str = ",".join(vector_values)
                query = f"""
                    INSERT INTO store_vectors (prefix, key, field_name, embedding, created_at, updated_at)
                    VALUES {values_str}
                    ON CONFLICT (prefix, key, field_name) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                """
                embedding_request = (query, embedding_request_params)

        return queries, embedding_request

    def _prepare_batch_search_queries(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
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
                        filter_conditions.append("value->%s = %s::jsonb")
                        filter_params.extend([key, json.dumps(value)])

            # Vector search branch
            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))

                score_operator, post_operator = _get_distance_operator(self)
                vector_type = (
                    cast(PostgresIndexConfig, self.index_config)
                    .get("ann_index_config", {})
                    .get("vector_type", "vector")
                )

                if (
                    vector_type == "bit"
                    and self.index_config.get("distance_type") == "hamming"
                ):
                    score_operator = score_operator % (
                        "%s",
                        self.index_config["dims"],
                    )
                else:
                    score_operator = score_operator % (
                        "%s",
                        vector_type,
                    )

                vectors_per_doc_estimate = self.index_config["__estimated_num_vectors"]
                expanded_limit = (op.limit * vectors_per_doc_estimate * 2) + 1

                # Vector search with CTE for proper score handling
                filter_str = (
                    ""
                    if not filter_conditions
                    else " AND " + " AND ".join(filter_conditions)
                )
                if op.namespace_prefix:
                    prefix_filter_str = f"WHERE s.prefix LIKE %s {filter_str} "
                    ns_args: Sequence = (f"{_namespace_to_text(op.namespace_prefix)}%",)
                else:
                    ns_args = ()
                    if filter_str:
                        prefix_filter_str = f"WHERE {filter_str} "
                    else:
                        prefix_filter_str = ""

                base_query = f"""
                    WITH scored AS (
                        SELECT s.prefix, s.key, s.value, s.created_at, s.updated_at, {score_operator} AS neg_score
                        FROM store s
                        JOIN store_vectors sv ON s.prefix = sv.prefix AND s.key = sv.key
                        {prefix_filter_str}
                        ORDER BY {score_operator} ASC 
                        LIMIT %s
                    )
                    SELECT * FROM (
                        SELECT DISTINCT ON (prefix, key) 
                            prefix, key, value, created_at, updated_at, {post_operator} as score 
                        FROM scored 
                        ORDER BY prefix, key, score DESC
                    ) AS unique_docs
                    ORDER BY score DESC
                    LIMIT %s
                    OFFSET %s
                """
                params = [
                    _PLACEHOLDER,  # Vector placeholder
                    *ns_args,
                    *filter_params,
                    _PLACEHOLDER,
                    expanded_limit,
                    op.limit,
                    op.offset,
                ]

            # Regular search branch
            else:
                base_query = """
                    SELECT prefix, key, value, created_at, updated_at
                    FROM store
                    WHERE prefix LIKE %s
                """
                params = [f"{_namespace_to_text(op.namespace_prefix)}%"]

                if filter_conditions:
                    params.extend(filter_params)
                    base_query += " AND " + " AND ".join(filter_conditions)

                base_query += " ORDER BY updated_at DESC"
                base_query += " LIMIT %s OFFSET %s"
                params.extend([op.limit, op.offset])

            queries.append((base_query, params))

        return queries, embedding_requests

    def _get_batch_list_namespaces_queries(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
    ) -> list[tuple[str, Sequence]]:
        queries: list[tuple[str, Sequence]] = []
        for _, op in list_ops:
            query = """
                SELECT DISTINCT ON (truncated_prefix) truncated_prefix, prefix
                FROM (
                    SELECT
                        prefix,
                        CASE
                            WHEN %s::integer IS NOT NULL THEN
                                (SELECT STRING_AGG(part, '.' ORDER BY idx)
                                 FROM (
                                     SELECT part, ROW_NUMBER() OVER () AS idx
                                     FROM UNNEST(REGEXP_SPLIT_TO_ARRAY(prefix, '\.')) AS part
                                     LIMIT %s::integer
                                 ) subquery
                                )
                            ELSE prefix
                        END AS truncated_prefix
                    FROM store
            """
            params: list[Any] = [op.max_depth, op.max_depth]

            conditions = []
            if op.match_conditions:
                for condition in op.match_conditions:
                    if condition.match_type == "prefix":
                        conditions.append("prefix LIKE %s")
                        params.append(
                            f"{_namespace_to_text(condition.path, handle_wildcards=True)}%"
                        )
                    elif condition.match_type == "suffix":
                        conditions.append("prefix LIKE %s")
                        params.append(
                            f"%{_namespace_to_text(condition.path, handle_wildcards=True)}"
                        )
                    else:
                        logger.warning(
                            f"Unknown match_type in list_namespaces: {condition.match_type}"
                        )

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += ") AS subquery "

            query += " ORDER BY truncated_prefix LIMIT %s OFFSET %s"
            params.extend([op.limit, op.offset])
            queries.append((query, tuple(params)))

        return queries

    def _get_filter_condition(self, key: str, op: str, value: Any) -> tuple[str, list]:
        """Helper to generate filter conditions."""
        if op == "$eq":
            return "value->%s = %s::jsonb", [key, json.dumps(value)]
        elif op == "$gt":
            return "value->>%s > %s", [key, str(value)]
        elif op == "$gte":
            return "value->>%s >= %s", [key, str(value)]
        elif op == "$lt":
            return "value->>%s < %s", [key, str(value)]
        elif op == "$lte":
            return "value->>%s <= %s", [key, str(value)]
        elif op == "$ne":
            return "value->%s != %s::jsonb", [key, json.dumps(value)]
        else:
            raise ValueError(f"Unsupported operator: {op}")


class PostgresStore(BaseStore, BasePostgresStore[_pg_internal.Conn]):
    """Postgres-backed store with optional vector search using pgvector.

    !!! example "Examples"
        Basic setup and key-value storage:
        ```python
        from langgraph.store.postgres import PostgresStore

        store = PostgresStore(
            connection_string="postgresql://user:pass@localhost:5432/dbname"
        )
        store.setup()

        # Store and retrieve data
        store.put(("users", "123"), "prefs", {"theme": "dark"})
        item = store.get(("users", "123"), "prefs")
        ```

        Vector search using LangChain embeddings:
        ```python
        from langchain.embeddings import init_embeddings
        from langgraph.store.postgres import PostgresStore

        store = PostgresStore(
            connection_string="postgresql://user:pass@localhost:5432/dbname",
            index={
                "dims": 1536,
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "fields": ["text"]  # specify which fields to embed. Default is the whole serialized value
            }
        )
        store.setup() # Do this once to run migrations

        # Store documents
        store.put(("docs",), "doc1", {"text": "Python tutorial"})
        store.put(("docs",), "doc2", {"text": "TypeScript guide"})
        store.put(("docs",), "doc2", {"text": "Other guide"}, index=False) # don't index

        # Search by similarity
        results = store.search(("docs",), query="python programming")
        ```

    Note:
        Semantic search is disabled by default. You can enable it by providing an `index` configuration
        when creating the store. Without this configuration, all `index` arguments passed to
        `put` or `aput`will have no effect.

    Warning:
        Make sure to call `setup()` before first use to create necessary tables and indexes.
        The pgvector extension must be available to use vector search.

    """

    __slots__ = (
        "_deserializer",
        "pipe",
        "lock",
        "supports_pipeline",
        "index_config",
        "embeddings",
    )

    def __init__(
        self,
        conn: _pg_internal.Conn,
        *,
        pipe: Optional[Pipeline] = None,
        deserializer: Optional[
            Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]
        ] = None,
        index: Optional[PostgresIndexConfig] = None,
    ) -> None:
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.pipe = pipe
        self.supports_pipeline = Capabilities().has_pipeline()
        self.lock = threading.Lock()
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
        pipeline: bool = False,
        pool_config: Optional[PoolConfig] = None,
        index: Optional[PostgresIndexConfig] = None,
    ) -> Iterator["PostgresStore"]:
        """Create a new PostgresStore instance from a connection string.

        Args:
            conn_string (str): The Postgres connection info string.
            pipeline (bool): whether to use Pipeline
            pool_config (Optional[PoolArgs]): Configuration for the connection pool.
                If provided, will create a connection pool and use it instead of a single connection.
                This overrides the `pipeline` argument.
            index (Optional[PostgresIndexConfig]): The index configuration for the store.

        Returns:
            PostgresStore: A new PostgresStore instance.
        """
        if pool_config is not None:
            pc = pool_config.copy()
            with cast(
                ConnectionPool[Connection[DictRow]],
                ConnectionPool(
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
                yield cls(conn=pool, index=index)
        else:
            with Connection.connect(
                conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
            ) as conn:
                if pipeline:
                    with conn.pipeline() as pipe:
                        yield cls(conn, pipe=pipe, index=index)
                else:
                    yield cls(conn, index=index)

    @contextmanager
    def _cursor(self, *, pipeline: bool = False) -> Iterator[Cursor[DictRow]]:
        """Create a database cursor as a context manager.

        Args:
            pipeline (bool): whether to use pipeline for the DB operations inside the context manager.
                Will be applied regardless of whether the PostgresStore instance was initialized with a pipeline.
                If pipeline mode is not supported, will fall back to using transaction context manager.
        """
        with _pg_internal.get_connection(self.conn) as conn:
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
                        self.lock,
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    with (
                        self.lock,
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        with self._cursor(pipeline=True) as cur:
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
        cur: Cursor[DictRow],
    ) -> None:
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            cur.execute(query, params)
            rows = cast(list[Row], cur.fetchall())
            key_to_row = {row["key"]: row for row in rows}
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
        cur: Cursor[DictRow],
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
            cur.execute(query, params)

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: Cursor[DictRow],
    ) -> None:
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)

        if embedding_requests and self.embeddings:
            embeddings = self.embeddings.embed_documents(
                [query for _, query in embedding_requests]
            )
            for (idx, _), embedding in zip(embedding_requests, embeddings):
                _paramslist = queries[idx][1]
                for i in range(len(_paramslist)):
                    if _paramslist[i] is _PLACEHOLDER:
                        _paramslist[i] = embedding

        for (idx, _), (query, params) in zip(search_ops, queries):
            cur.execute(query, params)
            rows = cast(list[Row], cur.fetchall())
            results[idx] = [
                _row_to_search_item(
                    _decode_ns_bytes(row["prefix"]), row, loader=self._deserializer
                )
                for row in rows
            ]

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: Cursor[DictRow],
    ) -> None:
        for (query, params), (idx, _) in zip(
            self._get_batch_list_namespaces_queries(list_ops), list_ops
        ):
            cur.execute(query, params)
            results[idx] = [_decode_ns_bytes(row["truncated_prefix"]) for row in cur]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)

    def setup(self) -> None:
        """Set up the store database.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """

        def _get_version(cur: Cursor[dict[str, Any]], table: str) -> int:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    v INTEGER PRIMARY KEY
                )
            """
            )
            cur.execute(f"SELECT v FROM {table} ORDER BY v DESC LIMIT 1")
            row = cast(dict, cur.fetchone())
            if row is None:
                version = -1
            else:
                version = row["v"]
            return version

        with self._cursor() as cur:
            version = _get_version(cur, table="store_migrations")
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                cur.execute(sql)
                cur.execute("INSERT INTO store_migrations (v) VALUES (%s)", (v,))

            if self.index_config:
                version = _get_version(cur, table="vector_migrations")
                for v, migration in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    if migration.condition and not migration.condition(self):
                        continue
                    sql = migration.sql
                    if migration.params:
                        params = {
                            k: v(self) if v is not None and callable(v) else v
                            for k, v in migration.params.items()
                        }
                        sql = sql % params
                    cur.execute(sql)
                    cur.execute("INSERT INTO vector_migrations (v) VALUES (%s)", (v,))


class Row(TypedDict):
    key: str
    value: Any
    prefix: str
    created_at: datetime
    updated_at: datetime


# Private utilities

_DEFAULT_ANN_CONFIG = ANNIndexConfig(
    vector_type="vector",
)


def _get_vector_type_ops(store: BasePostgresStore) -> str:
    """Get the vector type operator class based on config."""
    if not store.index_config:
        return "vector_cosine_ops"

    config = cast(PostgresIndexConfig, store.index_config)
    index_config = config.get("ann_index_config", _DEFAULT_ANN_CONFIG).copy()
    vector_type = cast(str, index_config.get("vector_type", "vector"))
    if vector_type not in ("vector", "halfvec"):
        raise ValueError(
            f"Vector type must be 'vector' or 'halfvec', got {vector_type}"
        )

    distance_type = config.get("distance_type", "cosine")

    # For regular vectors
    type_prefix = {"vector": "vector", "halfvec": "halfvec"}[vector_type]

    if distance_type not in ("l2", "inner_product", "cosine"):
        raise ValueError(
            f"Vector type {vector_type} only supports 'l2', 'inner_product', or 'cosine' distance, got {distance_type}"
        )

    distance_suffix = {
        "l2": "l2_ops",
        "inner_product": "ip_ops",
        "cosine": "cosine_ops",
    }[distance_type]

    return f"{type_prefix}_{distance_suffix}"


def _get_index_params(store: Any) -> tuple[str, dict[str, Any]]:
    """Get the index type and configuration based on config."""
    if not store.index_config:
        return "hnsw", {}

    config = cast(PostgresIndexConfig, store.index_config)
    index_config = config.get("ann_index_config", _DEFAULT_ANN_CONFIG).copy()
    kind = index_config.pop("kind", "hnsw")
    index_config.pop("vector_type", None)
    return kind, index_config


def _namespace_to_text(
    namespace: tuple[str, ...], handle_wildcards: bool = False
) -> str:
    """Convert namespace tuple to text string."""
    if handle_wildcards:
        namespace = tuple("%" if val == "*" else val for val in namespace)
    return ".".join(namespace)


def _row_to_item(
    namespace: tuple[str, ...],
    row: Row,
    *,
    loader: Optional[Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]] = None,
) -> Item:
    """Convert a row from the database into an Item.

    Args:
        namespace: Item namespace
        row: Database row
        loader: Optional value loader for non-dict values
    """
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
    row: Row,
    *,
    loader: Optional[Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]] = None,
) -> SearchItem:
    """Convert a row from the database into an Item."""
    loader = loader or _json_loads
    val = row["value"]
    score = row.get("score")
    if score is not None:
        try:
            score = float(score)  # type: ignore[arg-type]
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


def _json_loads(content: Union[bytes, orjson.Fragment]) -> Any:
    if isinstance(content, orjson.Fragment):
        if hasattr(content, "buf"):
            content = content.buf
        else:
            if isinstance(content.contents, bytes):
                content = content.contents
            else:
                content = content.contents.encode()
    return orjson.loads(cast(bytes, content))


def _decode_ns_bytes(namespace: Union[str, bytes, list]) -> tuple[str, ...]:
    if isinstance(namespace, list):
        return tuple(namespace)
    if isinstance(namespace, bytes):
        namespace = namespace.decode()[1:]
    return tuple(namespace.split("."))


def _get_distance_operator(store: Any) -> tuple[str, str]:
    """Get the distance operator and score expression based on config."""
    # Note: Today, we are not using ANN indices due to restrictions
    # on PGVector's support for mixing vector and non-vector filters
    # To use the index, PGVector expects:
    #  - ORDER BY the operator NOT an expression (even negation blocks it)
    #  - ASCENDING order
    #  - Any WHERE clause should be over a partial index.
    # If we violate any of these, it will use a sequential scan
    # See https://github.com/pgvector/pgvector/issues/216 and the
    # pgvector documentation for more details.
    if not store.index_config:
        raise ValueError(
            "Embedding configuration is required for vector operations "
            f"(for semantic search). "
            f"Please provide an Embeddings when initializing the {store.__class__.__name__}."
        )

    config = cast(PostgresIndexConfig, store.index_config)
    distance_type = config.get("distance_type", "cosine")

    # Return the operator and the score expression
    # The operator is used in the CTE and will be compatible with an ASCENDING ORDER
    # sort clause.
    # The score expression is used in the final query and will be compatible with
    # a DESCENDING ORDER sort clause and the user's expectations of what the similarity score
    # should be.
    if distance_type == "l2":
        # Final: "-(sv.embedding <-> %s::%s)"
        # We return the "l2 similarity" so that the sorting order is the same
        return "sv.embedding <-> %s::%s", "-scored.neg_score"
    elif distance_type == "inner_product":
        # Final: "-(sv.embedding <#> %s::%s)"
        return "sv.embedding <#> %s::%s", "-(scored.neg_score)"
    else:  # cosine similarity
        # Final:  "1 - (sv.embedding <=> %s::%s)"
        return "sv.embedding <=> %s::%s", "1 - scored.neg_score"


def _ensure_index_config(
    index_config: PostgresIndexConfig,
) -> tuple[Optional["Embeddings"], PostgresIndexConfig]:
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


_PLACEHOLDER = object()
