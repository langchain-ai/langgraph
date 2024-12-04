import asyncio
import logging
from contextlib import asynccontextmanager
from typing import (
    AsyncIterator,
    Iterable,
    Sequence,
    cast,
)

import duckdb
from langgraph.store.base import GetOp, ListNamespacesOp, Op, PutOp, Result, SearchOp
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.duckdb.base import (
    BaseDuckDBStore,
    _convert_ns,
    _group_ops,
    _row_to_item,
)

logger = logging.getLogger(__name__)


class AsyncDuckDBStore(AsyncBatchedBaseStore, BaseDuckDBStore):
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        super().__init__()
        self.conn = conn
        self.loop = asyncio.get_running_loop()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        tasks = []

        if GetOp in grouped_ops:
            tasks.append(
                self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results
                )
            )

        if PutOp in grouped_ops:
            tasks.append(
                self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp])
                )
            )

        if SearchOp in grouped_ops:
            tasks.append(
                self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                )
            )

        if ListNamespacesOp in grouped_ops:
            tasks.append(
                self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                )
            )

        await asyncio.gather(*tasks)

        return results

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return asyncio.run_coroutine_threadsafe(self.abatch(ops), self.loop).result()

    async def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
    ) -> None:
        cursors = []
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            cur = self.conn.cursor()
            await asyncio.to_thread(cur.execute, query, params)
            cursors.append((cur, namespace, items))

        for cur, namespace, items in cursors:
            rows = await asyncio.to_thread(cur.fetchall)
            key_to_row = {row[1]: row for row in rows}
            for idx, key in items:
                row = key_to_row.get(key)
                if row:
                    results[idx] = _row_to_item(namespace, row)
                else:
                    results[idx] = None

    async def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> None:
        queries = self._get_batch_PUT_queries(put_ops)
        for query, params in queries:
            cur = self.conn.cursor()
            await asyncio.to_thread(cur.execute, query, params)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
    ) -> None:
        queries = self._get_batch_search_queries(search_ops)
        cursors: list[tuple[duckdb.DuckDBPyConnection, int]] = []

        for (query, params), (idx, _) in zip(queries, search_ops):
            cur = self.conn.cursor()
            await asyncio.to_thread(cur.execute, query, params)
            cursors.append((cur, idx))

        for cur, idx in cursors:
            rows = await asyncio.to_thread(cur.fetchall)
            items = [_row_to_item(_convert_ns(row[0]), row) for row in rows]
            results[idx] = items

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        cursors: list[tuple[duckdb.DuckDBPyConnection, int]] = []
        for (query, params), (idx, _) in zip(queries, list_ops):
            cur = self.conn.cursor()
            await asyncio.to_thread(cur.execute, query, params)
            cursors.append((cur, idx))

        for cur, idx in cursors:
            rows = cast(list[tuple], await asyncio.to_thread(cur.fetchall))
            namespaces = [_convert_ns(row[0]) for row in rows]
            results[idx] = namespaces

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
    ) -> AsyncIterator["AsyncDuckDBStore"]:
        """Create a new AsyncDuckDBStore instance from a connection string.

        Args:
            conn_string (str): The DuckDB connection info string.

        Returns:
            AsyncDuckDBStore: A new AsyncDuckDBStore instance.
        """
        with duckdb.connect(conn_string) as conn:
            yield cls(conn)

    async def setup(self) -> None:
        """Set up the store database asynchronously.

        This method creates the necessary tables in the DuckDB database if they don't
        already exist and runs database migrations. It is called automatically when needed and should not be called
        directly by the user.
        """
        cur = self.conn.cursor()
        try:
            await asyncio.to_thread(
                cur.execute, "SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1"
            )
            row = await asyncio.to_thread(cur.fetchone)
            if row is None:
                version = -1
            else:
                version = row[0]
        except duckdb.CatalogException:
            version = -1
            # Create store_migrations table if it doesn't exist
            await asyncio.to_thread(
                cur.execute,
                """
                CREATE TABLE IF NOT EXISTS store_migrations (
                    v INTEGER PRIMARY KEY
                )
                """,
            )
        for v, migration in enumerate(
            self.MIGRATIONS[version + 1 :], start=version + 1
        ):
            await asyncio.to_thread(cur.execute, migration)
            await asyncio.to_thread(
                cur.execute, "INSERT INTO store_migrations (v) VALUES (?)", (v,)
            )
