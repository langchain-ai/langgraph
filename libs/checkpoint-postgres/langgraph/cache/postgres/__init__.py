from __future__ import annotations

import datetime
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from psycopg import Capabilities, Connection, Cursor, Pipeline
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from langgraph.cache.base import FullKey, ValueT
from langgraph.cache.postgres.base import BasePostgresCache
from langgraph.checkpoint.postgres import _internal
from langgraph.checkpoint.serde.base import SerializerProtocol


class PostgresCache(BasePostgresCache):
    """File-based cache using Postgres."""

    def __init__(
        self,
        conn: _internal.Conn,
        pipe: Pipeline | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        if isinstance(conn, ConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single Connection, not ConnectionPool."
            )

        self.conn = conn
        self.pipe = pipe
        self.lock = threading.Lock()
        self.supports_pipeline = Capabilities().has_pipeline()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls, conn_string: str, *, pipeline: bool = False
    ) -> Iterator[PostgresCache]:
        with Connection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            if pipeline:
                with conn.pipeline() as pipe:
                    yield cls(conn, pipe)
            else:
                yield cls(conn)

    def setup(self) -> None:
        """Run Setup for cache."""
        with self._cursor() as cur:
            cur.execute(self.MIGRATIONS[0])
            results = cur.execute(
                "SELECT v FROM cache_migrations ORDER BY v DESC LIMIT 1"
            )
            row = results.fetchone()
            if row is None:
                version = -1
            else:
                version = row["v"]
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
            ):
                cur.execute(migration)
                cur.execute(f"INSERT INTO cache_migrations (v) VALUES ({v})")
        if self.pipe:
            self.pipe.sync()

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""
        now = datetime.datetime.now(datetime.timezone.utc)
        out: dict[FullKey, ValueT] = {}
        if not keys:
            return out
        ns_list = [".".join(ns) for ns, _ in keys]
        key_list = [key for _, key in keys]
        with self._cursor() as cur:
            cur.execute(self.SELECT_SQL, (ns_list, key_list))
            to_delete: list[tuple[Any, Any]] = []
            if cur.pgresult is not None:
                for row in cur:
                    exp: datetime.datetime | None = row["expiry"]
                    if exp is not None and exp.timestamp() < now.timestamp():
                        to_delete.append((row["ns"], row["key"]))
                        continue
                    ns = tuple(row["ns"].split("."))
                    val = self.serde.loads_typed((row["encoding"], row["val"]))
                    out[(ns, row["key"])] = val
                for ns, key in to_delete:
                    cur.execute(self.DELETE_EXPIRED_SQL, (ns, key))
        return out

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Use AsyncPostgresCache Instead."""
        raise NotImplementedError("Please Use AsyncPostgresCache Instead.")

    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""
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
        with self._cursor() as cur:
            cur.executemany(self.UPSERT_SQL, items)

    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Use AsyncPostgresCache Instead."""
        raise NotImplementedError("Please Use AsyncPostgresCache Instead.")

    def clear(self, namespaces: Sequence[tuple[str, ...]] | None = None) -> None:
        """Delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        with self._cursor() as cur:
            if namespaces is None:
                cur.execute(self.CLEAR_ALL_SQL)
            else:
                ns_list = [",".join(ns) for ns in namespaces]
                cur.execute(self.CLEAR_NS_SQL, (ns_list,))

    async def aclear(self, namespaces: Sequence[tuple[str, ...]] | None = None) -> None:
        """Use AsyncPostgresCache Instead."""
        raise NotImplementedError("Please Use AsyncPostgresCache Instead.")

    @contextmanager
    def _cursor(self, *, pipeline: bool = False) -> Iterator[Cursor[DictRow]]:
        """Create a database cursor as a context manager."""
        with self.lock, _internal.get_connection(self.conn) as conn:
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
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    # Use connection's transaction context manager when pipeline mode not supported
                    with (
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur
