from __future__ import annotations

import asyncio
import datetime
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline, Capabilities
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from langgraph.cache.base import FullKey, ValueT
from langgraph.cache.postgres.base import BasePostgresCache
from langgraph.checkpoint.postgres import _ainternal
from langgraph.checkpoint.serde.base import SerializerProtocol


class AsyncPostgresCache(BasePostgresCache):
    """File-based cache using Postgres."""

    def __init__(
        self,
        conn: _ainternal.Conn,
        pipe: AsyncPipeline | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        if isinstance(conn, AsyncConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single AsyncConnection, not AsyncConnectionPool."
            )

        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.supports_pipeline = Capabilities().has_pipeline()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        serde: SerializerProtocol | None = None,
    ) -> AsyncIterator[AsyncPostgresCache]:
        async with await AsyncConnection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            if pipeline:
                async with conn.pipeline() as pipe:
                    yield cls(conn=conn, pipe=pipe, serde=serde)
            else:
                yield cls(conn=conn, serde=serde)

    async def setup(self) -> None:
        """Run Setup for cache."""
        async with self._cursor() as cur:
            await cur.execute(self.MIGRATIONS[0])
            results = await cur.execute(
                "SELECT v FROM cache_migrations ORDER BY v DESC LIMIT 1"
            )
            row = await results.fetchone()
            if row is None:
                version = -1
            else:
                version = row["v"]
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
                strict=False,
            ):
                await cur.execute(migration)
                await cur.execute(f"INSERT INTO cache_migrations (v) VALUES ({v})")
        if self.pipe:
            await self.pipe.sync()

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Use PostgresCache Instead."""
        raise NotImplementedError("Please Use PostgresCache Instead.")

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""
        now = datetime.datetime.now(datetime.timezone.utc)
        out: dict[FullKey, ValueT] = {}
        if not keys:
            return out
        ns_list = [".".join(ns) for ns, _ in keys]
        key_list = [key for _, key in keys]
        async with self._cursor() as cur:
            await cur.execute(self.SELECT_SQL, (ns_list, key_list))
            to_delete: list[tuple[Any, Any]] = []
            if cur.pgresult is not None:
                async for row in cur:
                    exp: datetime.datetime | None = row["expiry"]
                    if exp is not None and exp.timestamp() < now.timestamp():
                        to_delete.append((row["ns"], row["key"]))
                        continue
                    ns = tuple(row["ns"].split("."))
                    val = self.serde.loads_typed((row["encoding"], row["val"]))
                    out[(ns, row["key"])] = val
                for ns, key in to_delete:
                    await cur.execute(self.DELETE_EXPIRED_SQL, (ns, key))
        return out

    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Use PostgresCache Instead."""
        raise NotImplementedError("Please Use PostgresCache Instead.")

    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""
        items = []
        for (ns, key), (val, ttl) in pairs.items():
            expiry = (
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(seconds=ttl)
                if ttl is not None
                else None
            )
            enc, blob = self.serde.dumps_typed(val)
            items.append((".".join(ns), key, expiry, enc, blob))
        async with self._cursor() as cur:
            await cur.executemany(self.UPSERT_SQL, items)

    def clear(self, namespaces: Sequence[tuple[str, ...]] | None = None) -> None:
        """Use PostgresCache Instead."""
        raise NotImplementedError("Please Use PostgresCache Instead.")

    async def aclear(self, namespaces: Sequence[tuple[str, ...]] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        async with self._cursor() as cur:
            if namespaces is None:
                await cur.execute(self.CLEAR_ALL_SQL)
            else:
                ns_list = [",".join(ns) for ns in namespaces]
                await cur.execute(self.CLEAR_NS_SQL, (ns_list,))

    @asynccontextmanager
    async def _cursor(
        self, *, pipeline: bool = False
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        """Create a database cursor as a context manager."""
        async with self.lock, _ainternal.get_connection(self.conn) as conn:
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
                    async with (
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    # Use connection's transaction context manager when pipeline mode not supported
                    async with (
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur
