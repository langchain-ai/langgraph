import sys
from contextlib import asynccontextmanager, contextmanager
from uuid import uuid4

import pytest

# Try to import psycopg and psycopg_pool
PSYCOPG_AVAILABLE = False
AsyncConnection = None
Connection = None
AsyncConnectionPool = None
ConnectionPool = None
PostgresSaver = None
AsyncPostgresSaver = None

try:
    from psycopg import AsyncConnection, Connection
    from psycopg_pool import AsyncConnectionPool, ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    PSYCOPG_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass # psycopg or dependent modules not available

SQLITE_AVAILABLE = False
SqliteSaver = None
AsyncSqliteSaver = None

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    SQLITE_AVAILABLE = True
except ImportError:
    pass # sqlite or dependent modules not available. SQLITE_AVAILABLE remains False.

from tests.memory_assert import MemorySaverAssertImmutable

DEFAULT_POSTGRES_URI = "postgres://postgres:postgres@localhost:5442/"


@contextmanager
def _checkpointer_memory():
    yield MemorySaverAssertImmutable()


@contextmanager
def _checkpointer_sqlite():
    if not SQLITE_AVAILABLE:
        pytest.skip("sqlite not available or langgraph.checkpoint.sqlite failed to import")
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield checkpointer


@contextmanager
def _checkpointer_postgres():
    if not PSYCOPG_AVAILABLE:
        pytest.skip("psycopg or psycopg_pool not available")
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        with PostgresSaver.from_conn_string(
            DEFAULT_POSTGRES_URI + database
        ) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@contextmanager
def _checkpointer_postgres_pipe():
    if not PSYCOPG_AVAILABLE:
        pytest.skip("psycopg or psycopg_pool not available")
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        with PostgresSaver.from_conn_string(
            DEFAULT_POSTGRES_URI + database
        ) as checkpointer:
            checkpointer.setup()
            # setup can't run inside pipeline because of implicit transaction
            with checkpointer.conn.pipeline() as pipe:
                checkpointer.pipe = pipe
                yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@contextmanager
def _checkpointer_postgres_pool():
    if not PSYCOPG_AVAILABLE:
        pytest.skip("psycopg or psycopg_pool not available")
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        with ConnectionPool(
            DEFAULT_POSTGRES_URI + database, max_size=10, kwargs={"autocommit": True}
        ) as pool:
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _checkpointer_sqlite_aio():
    if not SQLITE_AVAILABLE:
        pytest.skip("sqlite_aio not available or langgraph.checkpoint.sqlite.aio failed to import")
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield checkpointer


@asynccontextmanager
async def _checkpointer_postgres_aio():
    if not PSYCOPG_AVAILABLE:
        pytest.skip("psycopg or psycopg_pool not available")
    if sys.version_info < (3, 10):
        pytest.skip("Async Postgres tests require Python 3.10+")
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        async with AsyncPostgresSaver.from_conn_string(
            DEFAULT_POSTGRES_URI + database
        ) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _checkpointer_postgres_aio_pipe():
    if not PSYCOPG_AVAILABLE:
        pytest.skip("psycopg or psycopg_pool not available")
    if sys.version_info < (3, 10):
        pytest.skip("Async Postgres tests require Python 3.10+")
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        async with AsyncPostgresSaver.from_conn_string(
            DEFAULT_POSTGRES_URI + database
        ) as checkpointer:
            await checkpointer.setup()
            # setup can't run inside pipeline because of implicit transaction
            async with checkpointer.conn.pipeline() as pipe:
                checkpointer.pipe = pipe
                yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _checkpointer_postgres_aio_pool():
    if not PSYCOPG_AVAILABLE:
        pytest.skip("psycopg or psycopg_pool not available")
    if sys.version_info < (3, 10):
        pytest.skip("Async Postgres tests require Python 3.10+")
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        async with AsyncConnectionPool(
            DEFAULT_POSTGRES_URI + database, max_size=10, kwargs={"autocommit": True}
        ) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


__all__ = [
    "_checkpointer_memory",
    "_checkpointer_sqlite",
    "_checkpointer_postgres",
    "_checkpointer_postgres_pipe",
    "_checkpointer_postgres_pool",
    "_checkpointer_sqlite_aio",
    "_checkpointer_postgres_aio",
    "_checkpointer_postgres_aio_pipe",
    "_checkpointer_postgres_aio_pool",
]
