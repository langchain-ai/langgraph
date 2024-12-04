import asyncio
import json
import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import duckdb
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

logger = logging.getLogger(__name__)


MIGRATIONS = [
    """
CREATE TABLE IF NOT EXISTS store (
    prefix TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSON NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (prefix, key)
);
""",
    """
CREATE INDEX IF NOT EXISTS store_prefix_idx ON store (prefix);
""",
]

C = TypeVar("C", bound=duckdb.DuckDBPyConnection)


class BaseDuckDBStore(Generic[C]):
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
            keys_to_query = ",".join(["?"] * len(keys))
            query = f"""
                SELECT prefix, key, value, created_at, updated_at
                FROM store
                WHERE prefix = ? AND key IN ({keys_to_query})
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
                placeholders = ",".join(["?"] * len(keys))
                query = (
                    f"DELETE FROM store WHERE prefix = ? AND key IN ({placeholders})"
                )
                params = (_namespace_to_text(namespace), *keys)
                queries.append((query, params))
        if inserts:
            values = []
            insertion_params = []
            for op in inserts:
                values.append("(?, ?, ?, now(), now())")
                insertion_params.extend(
                    [
                        _namespace_to_text(op.namespace),
                        op.key,
                        json.dumps(op.value),
                    ]
                )
            values_str = ",".join(values)
            query = f"""
                INSERT INTO store (prefix, key, value, created_at, updated_at)
                VALUES {values_str}
                ON CONFLICT (prefix, key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = now()
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
                WHERE prefix LIKE ?
            """
            params: list = [f"{_namespace_to_text(op.namespace_prefix)}%"]

            if op.filter:
                filter_conditions = []
                for key, value in op.filter.items():
                    filter_conditions.append(f"json_extract(value, '$.{key}') = ?")
                    params.append(json.dumps(value))
                query += " AND " + " AND ".join(filter_conditions)

            query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
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
                WITH split_prefix AS (
                    SELECT
                        prefix,
                        string_split(prefix, '.') AS parts
                    FROM store
                )
                SELECT DISTINCT ON (truncated_prefix)
                    CASE
                        WHEN ? IS NOT NULL THEN
                        array_to_string(array_slice(parts, 1, ?), '.')
                        ELSE prefix
                    END AS truncated_prefix,
                    prefix
                FROM split_prefix
            """
            params: list[Any] = [op.max_depth, op.max_depth]

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
            queries.append((query, params))

        return queries


class DuckDBStore(BaseStore, BaseDuckDBStore[duckdb.DuckDBPyConnection]):
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        super().__init__()
        self.conn = conn

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        if GetOp in grouped_ops:
            self._batch_get_ops(
                cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results
            )

        if PutOp in grouped_ops:
            self._batch_put_ops(cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]))

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
            cur = self.conn.cursor()
            cur.execute(query, params)
            cursors.append((cur, namespace, items))

        for cur, namespace, items in cursors:
            rows = cur.fetchall()
            key_to_row = {row[1]: row for row in rows}
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
            cur = self.conn.cursor()
            cur.execute(query, params)

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
    ) -> None:
        queries = self._get_batch_search_queries(search_ops)
        cursors: list[tuple[duckdb.DuckDBPyConnection, int]] = []

        for (query, params), (idx, _) in zip(queries, search_ops):
            cur = self.conn.cursor()
            cur.execute(query, params)
            cursors.append((cur, idx))

        for cur, idx in cursors:
            rows = cur.fetchall()
            items = [_row_to_search_item(_convert_ns(row[0]), row) for row in rows]
            results[idx] = items

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        cursors: list[tuple[duckdb.DuckDBPyConnection, int]] = []
        for (query, params), (idx, _) in zip(queries, list_ops):
            cur = self.conn.cursor()
            cur.execute(query, params)
            cursors.append((cur, idx))

        for cur, idx in cursors:
            rows = cast(list[dict], cur.fetchall())
            namespaces = [_convert_ns(row[0]) for row in rows]
            results[idx] = namespaces

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
    ) -> Iterator["DuckDBStore"]:
        """Create a new BaseDuckDBStore instance from a connection string.

        Args:
            conn_string (str): The DuckDB connection info string.

        Returns:
            DuckDBStore: A new DuckDBStore instance.
        """
        with duckdb.connect(conn_string) as conn:
            yield cls(conn=conn)

    def setup(self) -> None:
        """Set up the store database.

        This method creates the necessary tables in the DuckDB database if they don't
        already exist and runs database migrations. It is called automatically when needed and should not be called
        directly by the user.
        """
        with self.conn.cursor() as cur:
            try:
                cur.execute("SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1")
                row = cast(dict, cur.fetchone())
                if row is None:
                    version = -1
                else:
                    version = row["v"]
            except duckdb.CatalogException:
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
                cur.execute("INSERT INTO store_migrations (v) VALUES (?)", (v,))


def _namespace_to_text(
    namespace: tuple[str, ...], handle_wildcards: bool = False
) -> str:
    """Convert namespace tuple to text string."""
    if handle_wildcards:
        namespace = tuple("%" if val == "*" else val for val in namespace)
    return ".".join(namespace)


def _row_to_item(
    namespace: tuple[str, ...],
    row: tuple,
) -> Item:
    """Convert a row from the database into an Item."""
    _, key, val, created_at, updated_at = row
    return Item(
        value=val if isinstance(val, dict) else json.loads(val),
        key=key,
        namespace=namespace,
        created_at=created_at,
        updated_at=updated_at,
    )


def _row_to_search_item(
    namespace: tuple[str, ...],
    row: tuple,
) -> SearchItem:
    """Convert a row from the database into an SearchItem."""
    # TODO: Add support for search
    _, key, val, created_at, updated_at = row
    return SearchItem(
        value=val if isinstance(val, dict) else json.loads(val),
        key=key,
        namespace=namespace,
        created_at=created_at,
        updated_at=updated_at,
    )


def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    tot = 0
    for idx, op in enumerate(ops):
        grouped_ops[type(op)].append((idx, op))
        tot += 1
    return grouped_ops, tot


def _convert_ns(namespace: Union[str, list]) -> tuple[str, ...]:
    if isinstance(namespace, list):
        return tuple(namespace)
    return tuple(namespace.split("."))
