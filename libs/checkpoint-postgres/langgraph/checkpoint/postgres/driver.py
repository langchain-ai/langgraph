"""Driver adapters for PostgreSQL checkpoint savers.

Adapters isolate database-driver details such as connection lifecycle, cursor
creation, transactions, pipelines, and JSONB values. The saver SQL continues to
use psycopg-style placeholders; an alternative adapter can translate them in its
cursor implementation before delegating to another driver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SyncCursor(Protocol):
    """Cursor operations required by synchronous checkpoint savers."""

    def execute(
        self,
        query: str,
        params: Sequence[Any] | None = None,
        *,
        binary: bool = False,
    ) -> Any: ...

    def executemany(self, query: str, params_seq: Sequence[Sequence[Any]]) -> Any: ...

    def fetchone(self) -> Any: ...

    def fetchall(self) -> list[Any]: ...

    def __iter__(self) -> Iterator[Any]: ...


@runtime_checkable
class AsyncCursor(Protocol):
    """Cursor operations required by asynchronous checkpoint savers."""

    async def execute(
        self,
        query: str,
        params: Sequence[Any] | None = None,
        *,
        binary: bool = False,
    ) -> Any: ...

    async def executemany(
        self, query: str, params_seq: Sequence[Sequence[Any]]
    ) -> Any: ...

    async def fetchone(self) -> Any: ...

    async def fetchall(self) -> list[Any]: ...

    def __aiter__(self) -> AsyncIterator[Any]: ...


class PostgresDriverAdapter(ABC):
    """Common driver contract for PostgreSQL checkpoint savers.

    Implement a sync or async subclass to use a driver other than psycopg. The
    cursor returned by an adapter owns SQL execution and parameter adaptation.
    It must return mapping-like rows because savers address result columns by name.
    """

    @abstractmethod
    def is_pool(self, connection: Any) -> bool:
        """Return whether ``connection`` checks out a connection per operation."""

    def supports_pipeline(self) -> bool:
        """Return whether this driver supports a pipeline context manager."""
        return False

    def jsonb(self, value: Any) -> Any:
        """Convert a Python JSON-compatible value into a driver JSONB parameter."""
        return value


class SyncPostgresDriverAdapter(PostgresDriverAdapter, ABC):
    """Driver contract for :class:`PostgresSaver` and its shallow variant."""

    @abstractmethod
    def connect(self, conn_string: str) -> Any:
        """Return a context manager that opens and closes a connection."""

    @abstractmethod
    def get_connection(self, connection: Any) -> Any:
        """Return a context manager yielding a concrete connection."""

    @abstractmethod
    def cursor(self, connection: Any) -> Any:
        """Return a context manager yielding a :class:`SyncCursor`."""

    @abstractmethod
    def transaction(self, connection: Any) -> Any:
        """Return a transaction context manager."""

    @abstractmethod
    def pipeline(self, connection: Any) -> Any:
        """Return a pipeline context manager when pipelines are supported."""

    @abstractmethod
    def sync_pipeline(self, pipeline: Any) -> None:
        """Flush work submitted to an active pipeline."""


class AsyncPostgresDriverAdapter(PostgresDriverAdapter, ABC):
    """Driver contract for :class:`AsyncPostgresSaver` and its shallow variant."""

    @abstractmethod
    def connect(self, conn_string: str) -> Any:
        """Return an async context manager that opens and closes a connection."""

    @abstractmethod
    def get_connection(self, connection: Any) -> Any:
        """Return an async context manager yielding a concrete connection."""

    @abstractmethod
    def cursor(self, connection: Any) -> Any:
        """Return an async context manager yielding an :class:`AsyncCursor`."""

    @abstractmethod
    def transaction(self, connection: Any) -> Any:
        """Return an async transaction context manager."""

    @abstractmethod
    def pipeline(self, connection: Any) -> Any:
        """Return an async pipeline context manager when pipelines are supported."""

    @abstractmethod
    async def sync_pipeline(self, pipeline: Any) -> None:
        """Flush work submitted to an active pipeline."""


def _load_psycopg() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from psycopg import AsyncConnection, Capabilities, Connection
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import AsyncConnectionPool, ConnectionPool
    except ModuleNotFoundError as exc:
        if exc.name in {"psycopg", "psycopg_pool"}:
            raise ImportError(
                "The default psycopg driver is not installed. Install it with "
                '`pip install "langgraph-checkpoint-postgres[psycopg]"`, or '
                "pass a custom Postgres driver adapter."
            ) from exc
        raise
    return (
        Connection,
        AsyncConnection,
        ConnectionPool,
        AsyncConnectionPool,
        Capabilities,
        (dict_row, Jsonb),
    )


class PsycopgDriverAdapter(SyncPostgresDriverAdapter):
    """Default synchronous adapter backed by psycopg 3."""

    def is_pool(self, connection: Any) -> bool:
        _, _, ConnectionPool, _, _, _ = _load_psycopg()
        return isinstance(connection, ConnectionPool)

    def supports_pipeline(self) -> bool:
        _, _, _, _, Capabilities, _ = _load_psycopg()
        return Capabilities().has_pipeline()

    def jsonb(self, value: Any) -> Any:
        *_, helpers = _load_psycopg()
        _, Jsonb = helpers
        return Jsonb(value)

    @contextmanager
    def connect(self, conn_string: str) -> Iterator[Any]:
        Connection, _, _, _, _, helpers = _load_psycopg()
        dict_row, _ = helpers
        with Connection.connect(
            conn_string,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as connection:
            yield connection

    @contextmanager
    def get_connection(self, connection: Any) -> Iterator[Any]:
        Connection, _, ConnectionPool, _, _, _ = _load_psycopg()
        if isinstance(connection, Connection):
            yield connection
        elif isinstance(connection, ConnectionPool):
            with connection.connection() as checked_out:
                yield checked_out
        else:
            raise TypeError(f"Invalid connection type: {type(connection)}")

    @contextmanager
    def cursor(self, connection: Any) -> Iterator[SyncCursor]:
        *_, helpers = _load_psycopg()
        dict_row, _ = helpers
        with connection.cursor(binary=True, row_factory=dict_row) as cursor:
            yield cursor

    def transaction(self, connection: Any) -> Any:
        return connection.transaction()

    def pipeline(self, connection: Any) -> Any:
        return connection.pipeline()

    def sync_pipeline(self, pipeline: Any) -> None:
        pipeline.sync()


class AsyncPsycopgDriverAdapter(AsyncPostgresDriverAdapter):
    """Default asynchronous adapter backed by psycopg 3."""

    def is_pool(self, connection: Any) -> bool:
        _, _, _, AsyncConnectionPool, _, _ = _load_psycopg()
        return isinstance(connection, AsyncConnectionPool)

    def supports_pipeline(self) -> bool:
        _, _, _, _, Capabilities, _ = _load_psycopg()
        return Capabilities().has_pipeline()

    def jsonb(self, value: Any) -> Any:
        *_, helpers = _load_psycopg()
        _, Jsonb = helpers
        return Jsonb(value)

    @asynccontextmanager
    async def connect(self, conn_string: str) -> AsyncIterator[Any]:
        _, AsyncConnection, _, _, _, helpers = _load_psycopg()
        dict_row, _ = helpers
        async with await AsyncConnection.connect(
            conn_string,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as connection:
            yield connection

    @asynccontextmanager
    async def get_connection(self, connection: Any) -> AsyncIterator[Any]:
        _, AsyncConnection, _, AsyncConnectionPool, _, _ = _load_psycopg()
        if isinstance(connection, AsyncConnection):
            yield connection
        elif isinstance(connection, AsyncConnectionPool):
            async with connection.connection() as checked_out:
                yield checked_out
        else:
            raise TypeError(f"Invalid connection type: {type(connection)}")

    @asynccontextmanager
    async def cursor(self, connection: Any) -> AsyncIterator[AsyncCursor]:
        *_, helpers = _load_psycopg()
        dict_row, _ = helpers
        async with connection.cursor(binary=True, row_factory=dict_row) as cursor:
            yield cursor

    def transaction(self, connection: Any) -> Any:
        return connection.transaction()

    def pipeline(self, connection: Any) -> Any:
        return connection.pipeline()

    async def sync_pipeline(self, pipeline: Any) -> None:
        await pipeline.sync()


__all__ = [
    "AsyncCursor",
    "AsyncPostgresDriverAdapter",
    "AsyncPsycopgDriverAdapter",
    "PostgresDriverAdapter",
    "PsycopgDriverAdapter",
    "SyncCursor",
    "SyncPostgresDriverAdapter",
]
