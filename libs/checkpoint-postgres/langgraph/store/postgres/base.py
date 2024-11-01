import asyncio
import json
import logging
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import orjson
from psycopg import BaseConnection, Connection, Cursor
from psycopg.errors import UndefinedTable
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from typing_extensions import TypedDict

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
)

logger = logging.getLogger(__name__)


MIGRATIONS = [
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
CREATE INDEX IF NOT EXISTS store_prefix_idx ON store USING btree (prefix text_pattern_ops);
""",
]

C = TypeVar("C", bound=BaseConnection)


class BasePostgresStore(Generic[C]):
    MIGRATIONS = MIGRATIONS
    conn: C
    _deserializer: Optional[Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]]

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

    def _get_batch_PUT_queries(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> list[tuple[str, Sequence]]:
        inserts: list[PutOp] = []
        deletes: list[PutOp] = []
        for _, op in put_ops:
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
        if inserts:
            values = []
            insertion_params = []
            for op in inserts:
                values.append("(%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)")
                insertion_params.extend(
                    [
                        _namespace_to_text(op.namespace),
                        op.key,
                        Jsonb(op.value),
                    ]
                )
            values_str = ",".join(values)
            query = f"""
                INSERT INTO store (prefix, key, value, created_at, updated_at)
                VALUES {values_str}
                ON CONFLICT (prefix, key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP
            """
            queries.append((query, insertion_params))

        return queries

    def _get_batch_search_queries(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
    ) -> list[tuple[str, Sequence]]:
        queries: list[tuple[str, Sequence]] = []
        for _, op in search_ops:
            query = """
                SELECT prefix, key, value, created_at, updated_at
                FROM store
                WHERE prefix LIKE %s
            """
            params: list = [f"{_namespace_to_text(op.namespace_prefix)}%"]

            if op.filter:
                filter_conditions = []
                for key, value in op.filter.items():
                    if isinstance(value, list):
                        filter_conditions.append("value->%s @> %s::jsonb")
                        params.extend([key, json.dumps(value)])
                    else:
                        filter_conditions.append("value->%s = %s::jsonb")
                        params.extend([key, json.dumps(value)])
                query += " AND " + " AND ".join(filter_conditions)

            # Note: we will need to not do this if sim/keyword search
            # is used
            query += " ORDER BY updated_at DESC LIMIT %s OFFSET %s"
            params.extend([op.limit, op.offset])

            queries.append((query, params))
        return queries

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
            queries.append((query, params))

        return queries


class PostgresStore(BaseStore, BasePostgresStore[Connection]):
    __slots__ = ("_deserializer",)

    def __init__(
        self,
        conn: Connection[Any],
        *,
        deserializer: Optional[
            Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]
        ] = None,
    ) -> None:
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        with self.conn.pipeline():
            if GetOp in grouped_ops:
                self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results
                )

            if PutOp in grouped_ops:
                self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp])
                )

            if SearchOp in grouped_ops:
                self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                )

            if ListNamespacesOp in grouped_ops:
                self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                )

        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)

    def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
    ) -> None:
        cursors = []
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            cur = self.conn.cursor(binary=True)
            cur.execute(query, params)
            cursors.append((cur, namespace, items))

        for cur, namespace, items in cursors:
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
    ) -> None:
        queries = self._get_batch_PUT_queries(put_ops)
        for query, params in queries:
            cur = self.conn.cursor(binary=True)
            cur.execute(query, params)

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
    ) -> None:
        queries = self._get_batch_search_queries(search_ops)
        cursors: list[tuple[Cursor[Any], int]] = []

        for (query, params), (idx, _) in zip(queries, search_ops):
            cur = self.conn.cursor(binary=True)
            cur.execute(query, params)
            cursors.append((cur, idx))

        for cur, idx in cursors:
            rows = cast(list[Row], cur.fetchall())
            items = [
                _row_to_item(
                    _decode_ns_bytes(row["prefix"]), row, loader=self._deserializer
                )
                for row in rows
            ]
            results[idx] = items

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        cursors: list[tuple[Cursor[Any], int]] = []
        for (query, params), (idx, _) in zip(queries, list_ops):
            cur = self.conn.cursor(binary=True)
            cur.execute(query, params)
            cursors.append((cur, idx))

        for cur, idx in cursors:
            rows = cast(list[dict], cur.fetchall())
            namespaces = [_decode_ns_bytes(row["truncated_prefix"]) for row in rows]
            results[idx] = namespaces

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
    ) -> Iterator["PostgresStore"]:
        """Create a new BasePostgresStore instance from a connection string.

        Args:
            conn_string (str): The Postgres connection info string.

        Returns:
            BasePostgresStore: A new BasePostgresStore instance.
        """
        with Connection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            yield cls(conn=conn)

    def setup(self) -> None:
        """Set up the store database.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """
        with self.conn.cursor(binary=True) as cur:
            try:
                cur.execute("SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1")
                row = cast(dict, cur.fetchone())
                if row is None:
                    version = -1
                else:
                    version = row["v"]
            except UndefinedTable:
                self.conn.rollback()
                version = -1
                # Create store_migrations table if it doesn't exist
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS store_migrations (
                        v INTEGER PRIMARY KEY
                    )
                """
                )
            for v, migration in enumerate(
                self.MIGRATIONS[version + 1 :], start=version + 1
            ):
                cur.execute(migration)
                cur.execute("INSERT INTO store_migrations (v) VALUES (%s)", (v,))


class Row(TypedDict):
    key: str
    value: Any
    prefix: str
    created_at: datetime
    updated_at: datetime


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
    """Convert a row from the database into an Item."""
    loader = loader or _json_loads
    val = row["value"]
    return Item(
        value=val if isinstance(val, dict) else loader(val),
        key=row["key"],
        namespace=namespace,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
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
