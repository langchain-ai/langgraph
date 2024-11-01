import asyncio
import logging
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterable,
    Optional,
    Sequence,
    Union,
    cast,
)

import orjson
from psycopg import AsyncConnection, AsyncCursor
from psycopg.errors import UndefinedTable
from psycopg.rows import dict_row

from langgraph.store.base import GetOp, ListNamespacesOp, Op, PutOp, Result, SearchOp
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.postgres.base import (
    BasePostgresStore,
    Row,
    _decode_ns_bytes,
    _group_ops,
    _row_to_item,
)

logger = logging.getLogger(__name__)


class AsyncPostgresStore(AsyncBatchedBaseStore, BasePostgresStore[AsyncConnection]):
    __slots__ = ("_deserializer",)

    def __init__(
        self,
        conn: AsyncConnection[Any],
        *,
        deserializer: Optional[
            Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]
        ] = None,
    ) -> None:
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.loop = asyncio.get_running_loop()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        async with self.conn.pipeline():
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
            cur = self.conn.cursor(binary=True)
            await cur.execute(query, params)
            cursors.append((cur, namespace, items))

        for cur, namespace, items in cursors:
            rows = cast(list[Row], await cur.fetchall())
            key_to_row = {row["key"]: row for row in rows}
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
    ) -> None:
        queries = self._get_batch_PUT_queries(put_ops)
        for query, params in queries:
            cur = self.conn.cursor(binary=True)
            await cur.execute(query, params)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
    ) -> None:
        queries = self._get_batch_search_queries(search_ops)
        cursors: list[tuple[AsyncCursor[Any], int]] = []

        for (query, params), (idx, _) in zip(queries, search_ops):
            cur = self.conn.cursor(binary=True)
            await cur.execute(query, params)
            cursors.append((cur, idx))

        for cur, idx in cursors:
            rows = cast(list[Row], await cur.fetchall())
            items = [
                _row_to_item(
                    _decode_ns_bytes(row["prefix"]), row, loader=self._deserializer
                )
                for row in rows
            ]
            results[idx] = items

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        cursors: list[tuple[AsyncCursor[Any], int]] = []
        for (query, params), (idx, _) in zip(queries, list_ops):
            cur = self.conn.cursor(binary=True)
            await cur.execute(query, params)
            cursors.append((cur, idx))

        for cur, idx in cursors:
            rows = cast(list[dict], await cur.fetchall())
            namespaces = [_decode_ns_bytes(row["truncated_prefix"]) for row in rows]
            results[idx] = namespaces

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
    ) -> AsyncIterator["AsyncPostgresStore"]:
        """Create a new AsyncPostgresStore instance from a connection string.

        Args:
            conn_string (str): The Postgres connection info string.

        Returns:
            AsyncPostgresStore: A new AsyncPostgresStore instance.
        """
        async with await AsyncConnection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            yield cls(conn=conn)

    async def setup(self) -> None:
        """Set up the store database asynchronously.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """
        async with self.conn.cursor() as cur:
            try:
                await cur.execute(
                    "SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1"
                )
                row = cast(dict, await cur.fetchone())
                if row is None:
                    version = -1
                else:
                    version = row["v"]
            except UndefinedTable:
                version = -1
                # Create store_migrations table if it doesn't exist
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS store_migrations (
                        v INTEGER PRIMARY KEY
                    )
                    """
                )
            for v, migration in enumerate(
                self.MIGRATIONS[version + 1 :], start=version + 1
            ):
                await cur.execute(migration)
                await cur.execute("INSERT INTO store_migrations (v) VALUES (%s)", (v,))
