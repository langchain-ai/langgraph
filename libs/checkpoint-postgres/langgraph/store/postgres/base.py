import asyncio
import json
import logging
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generic, Iterable, Iterator, Sequence, TypeVar, Union, cast

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
CREATE EXTENSION IF NOT EXISTS ltree;
""",
    """
CREATE TABLE IF NOT EXISTS store (
    -- 'prefix' represents the doc's 'namespace'
    prefix ltree NOT NULL,
    key text NOT NULL,
    value jsonb NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prefix, key)
);
""",
    """
-- For faster listing of namespaces & lookups by namespace with prefix/suffix matching
CREATE INDEX IF NOT EXISTS store_prefix_idx ON store USING gist (prefix);
""",
]

C = TypeVar("C", bound=BaseConnection)


class BasePostgresStore(BaseStore, Generic[C]):
    MIGRATIONS = MIGRATIONS
    conn: C

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
            params = (_namespace_to_ltree(namespace), *keys)
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
                params = (_namespace_to_ltree(namespace), *keys)
                queries.append((query, params))
        if inserts:
            values = []
            insertion_params = []
            for op in inserts:
                values.append("(%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)")
                insertion_params.extend(
                    [
                        _namespace_to_ltree(op.namespace),
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
                SELECT key, value, created_at, updated_at
                FROM store
                WHERE prefix <@ %s
            """
            params: list = [_namespace_to_ltree(op.namespace_prefix)]

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

            query += " LIMIT %s OFFSET %s"
            params.extend([op.limit, op.offset])

            queries.append((query, params))
        return queries

    def _get_batch_list_namespaces_queries(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
    ) -> list[tuple[str, Sequence]]:
        queries: list[tuple[str, Sequence]] = []
        for _, op in list_ops:
            query = "SELECT DISTINCT subltree(prefix, 0, LEAST(nlevel(prefix), %s)) AS truncated_prefix FROM store"
            # https://www.postgresql.org/docs/current/ltree.html
            # The length of a label path cannot exceed 65535 labels.
            params: list[Any] = [op.max_depth if op.max_depth is not None else 65536]

            conditions = []
            if op.match_conditions:
                for condition in op.match_conditions:
                    if condition.match_type == "prefix":
                        conditions.append("prefix ~ %s::lquery")
                        lquery_pattern = f"{_namespace_to_ltree(condition.path)}.*"
                        params.append(lquery_pattern)
                    elif condition.match_type == "suffix":
                        conditions.append("prefix ~ %s::lquery")
                        lquery_pattern = f"*.{_namespace_to_ltree(condition.path)}"
                        params.append(lquery_pattern)
                    else:
                        logger.warning(
                            f"Unknown match_type in list_namespaces: {condition.match_type}"
                        )

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY truncated_prefix LIMIT %s OFFSET %s"
            params.extend([op.limit, op.offset])

            queries.append((query, params))
        return queries


class PostgresStore(BasePostgresStore[Connection]):
    def __init__(self, conn: Connection[Any]) -> None:
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
                    results[idx] = _row_to_item(namespace, row)
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
        cursors: list[tuple[Cursor[Any], int, SearchOp]] = []

        for (query, params), (idx, op) in zip(queries, search_ops):
            cur = self.conn.cursor(binary=True)
            cur.execute(query, params)
            cursors.append((cur, idx, op))

        for cur, idx, op in cursors:
            rows = cast(list[Row], cur.fetchall())
            items = [_row_to_item(op.namespace_prefix, row) for row in rows]
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
            namespaces = [
                tuple(row["truncated_prefix"].decode()[1:].split(".")) for row in rows
            ]
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
    prefix: bytes
    created_at: datetime
    updated_at: datetime


def _namespace_to_ltree(namespace: tuple[str, ...]) -> str:
    """Convert namespace tuple to ltree-compatible string."""
    return ".".join(namespace)


def _row_to_item(namespace: tuple[str, ...], row: Row) -> Item:
    """Convert a row from the database into an Item."""
    val = row["value"]
    return Item(
        value=val if isinstance(val, dict) else _json_loads(val),
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
