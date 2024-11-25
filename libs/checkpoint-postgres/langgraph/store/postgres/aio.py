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
from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline, Capabilities
from psycopg.errors import UndefinedTable
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.postgres import _ainternal
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.postgres.base import (
    BasePostgresStore,
    EmbeddingConfig,
    Row,
    _decode_ns_bytes,
    _group_ops,
    _row_to_item,
    check_vector_available,
)

logger = logging.getLogger(__name__)


class AsyncPostgresStore(AsyncBatchedBaseStore, BasePostgresStore[_ainternal.Conn]):
    __slots__ = (
        "_deserializer",
        "pipe",
        "lock",
        "supports_pipeline",
        "embedding_config",
    )

    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        pipe: Optional[AsyncPipeline] = None,
        deserializer: Optional[
            Callable[[Union[bytes, orjson.Fragment]], dict[str, Any]]
        ] = None,
        embedding: Optional[EmbeddingConfig] = None,
    ) -> None:
        if isinstance(conn, AsyncConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single AsyncConnection, not AsyncConnectionPool."
            )
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.supports_pipeline = Capabilities().has_pipeline()
        self.embedding_config = embedding

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        async with _ainternal.get_connection(self.conn) as conn:
            if self.pipe:
                async with self.pipe:
                    await self._execute_batch(grouped_ops, results, conn)
            else:
                await self._execute_batch(grouped_ops, results, conn)

        return results

    async def _execute_batch(
        self,
        grouped_ops: dict,
        results: list[Result],
        conn: AsyncConnection[DictRow],
    ) -> None:
        async with self._cursor(pipeline=True) as cur:
            if GetOp in grouped_ops:
                await self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]),
                    results,
                    cur,
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
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]),
                    cur,
                )

    async def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            await cur.execute(query, params)
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
        cur: AsyncCursor[DictRow],
    ) -> None:
        queries, embedding_request = self._prepare_batch_PUT_queries(put_ops)
        if embedding_request:
            if self.embedding_config is None:
                # Should not get here since the embedding config is required
                # to return an embedding_request above
                raise ValueError(
                    "Embedding configuration is required for vector operations "
                    f"(for semantic search). "
                    f"Please provide an EmbeddingConfig when initializing the {self.__class__.__name__}."
                )
            query, txt_params = embedding_request
            # Update the params to replace the raw text with the vectors
            vectors = await self.embedding_config["embed"].aembed_documents(
                [param[-1] for param in txt_params]
            )
            queries.extend(
                [
                    (query, (ns, key, value, vector))
                    for (ns, key, value, _), vector in zip(txt_params, vectors)
                ]
            )

        for query, params in queries:
            await cur.execute(query, params)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)

        if embedding_requests and self.embedding_config:
            embeddings = await self.embedding_config["embed"].aembed_documents(
                [query for _, query in embedding_requests]
            )
            for (idx, _), embedding in zip(embedding_requests, embeddings):
                queries[idx][1][0] = embedding

        for (idx, _), (query, params) in zip(search_ops, queries):
            await cur.execute(query, params)
            rows = cast(list[Row], await cur.fetchall())
            items = [
                _row_to_item(
                    _decode_ns_bytes(row["prefix"]),
                    row,
                    loader=self._deserializer,
                    cls=SearchItem,
                )
                for row in rows
            ]
            results[idx] = items

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        for (query, params), (idx, _) in zip(queries, list_ops):
            await cur.execute(query, params)
            rows = cast(list[dict], await cur.fetchall())
            namespaces = [_decode_ns_bytes(row["truncated_prefix"]) for row in rows]
            results[idx] = namespaces

    @asynccontextmanager
    async def _cursor(
        self, *, pipeline: bool = False
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        """Create a database cursor as a context manager.

        Args:
            pipeline: whether to use pipeline for the DB operations inside the context manager.
                Will be applied regardless of whether the PostgresStore instance was initialized with a pipeline.
                If pipeline mode is not supported, will fall back to using transaction context manager.
        """
        async with _ainternal.get_connection(self.conn) as conn:
            if self.pipe:
                # a connection in pipeline mode can be used concurrently
                # in multiple threads/coroutines, but only one cursor can be
                # used at a time
                try:
                    async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                finally:
                    if pipeline:
                        await self.pipe.sync()
            elif pipeline:
                # a connection not in pipeline mode can only be used by one
                # thread/coroutine at a time, so we acquire a lock
                if self.supports_pipeline:
                    async with self.lock, conn.pipeline(), conn.cursor(
                        binary=True, row_factory=dict_row
                    ) as cur:
                        yield cur
                else:
                    async with self.lock, conn.transaction(), conn.cursor(
                        binary=True, row_factory=dict_row
                    ) as cur:
                        yield cur
            else:
                async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return asyncio.run_coroutine_threadsafe(self.abatch(ops), self.loop).result()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        min_size: int = 1,
        max_size: Optional[int] = None,
        use_pool: bool = False,
        embedding: Optional[EmbeddingConfig] = None,
    ) -> AsyncIterator["AsyncPostgresStore"]:
        """Create a new AsyncPostgresStore instance from a connection string.

        Args:
            conn_string (str): The Postgres connection info string.
            pipeline (bool): Whether to use AsyncPipeline (only for single connections)
            min_size (int): Minimum number of connections when using a pool
            max_size (Optional[int]): Maximum number of connections when using a pool
            use_pool (bool): Whether to use a connection pool
            embedding (Optional[EmbeddingConfig]): Configuration for vector embeddings

        Returns:
            AsyncPostgresStore: A new AsyncPostgresStore instance.
        """
        if use_pool:
            async with cast(
                AsyncConnectionPool[AsyncConnection[DictRow]],
                AsyncConnectionPool(
                    conn_string,
                    min_size=min_size,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "prepare_threshold": 0,
                        "row_factory": dict_row,
                    },
                ),
            ) as pool:
                yield cls(conn=pool, embedding=embedding)
        else:
            async with await AsyncConnection.connect(
                conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
            ) as conn:
                if pipeline:
                    async with conn.pipeline() as pipe:
                        yield cls(conn=conn, pipe=pipe, embedding=embedding)
                else:
                    yield cls(conn=conn, embedding=embedding)

    async def setup(self) -> None:
        """Set up the store database asynchronously.

        This method creates the necessary tables in the Postgres database if they don't
        already exist and runs database migrations. It MUST be called directly by the user
        the first time the store is used.
        """
        async with self._cursor() as cur:
            try:
                await cur.execute(
                    "SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1"
                )
                row = await cur.fetchone()
                if row is None:
                    version = -1
                else:
                    version = row["v"]
            except UndefinedTable:
                version = -1
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
                if isinstance(migration, str):
                    sql = migration
                else:
                    if migration.acondition and not (await migration.acondition(self)):
                        continue

                    sql = migration.sql
                    if migration.params:
                        params = {
                            k: v(self) if v is not None and callable(v) else v
                            for k, v in migration.params.items()
                        }
                        try:
                            sql = sql % params
                        except Exception as e:
                            logger.warning(f"Failed to format migration {v}: {e}")
                            if migration.condition == check_vector_available:
                                self.embedding_config = None
                            continue

                try:
                    await cur.execute(sql)
                    await cur.execute(
                        "INSERT INTO store_migrations (v) VALUES (%s)", (v,)
                    )
                except Exception as e:
                    logger.warning(f"Failed to run migration {v}: {e}")
                    if (
                        not isinstance(migration, str)
                        and migration.condition == check_vector_available
                    ):
                        self.embedding_config = None
                    continue
