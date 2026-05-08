from contextlib import asynccontextmanager, contextmanager
import os
import sys
from uuid import uuid4

from psycopg import AsyncConnection, Connection
from psycopg.sql import SQL, Identifier

from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore, PostgresStore

DEFAULT_POSTGRES_URI = os.environ.get(
    "POSTGRES_URI", "postgres://postgres:postgres@localhost:5442/"
)

HITL_ENABLED = os.environ.get("HITL_DROP_DATABASE", "false").lower() == "true"


def _hitl_approve_drop(database: str) -> bool:
    """Human-in-the-Loop approval for DROP DATABASE operations."""
    if not sys.stdin.isatty():
        # In non-interactive environments, require explicit env var approval
        return HITL_ENABLED
    response = input(
        f"[HITL APPROVAL REQUIRED] About to DROP DATABASE '{database}'. "
        f"Type 'yes' to confirm: "
    )
    return response.strip().lower() == "yes"


def _safe_create_database(conn: Connection, database: str) -> None:
    conn.execute(SQL("CREATE DATABASE {}").format(Identifier(database)))


def _safe_drop_database(conn: Connection, database: str) -> None:
    if not _hitl_approve_drop(database):
        raise RuntimeError(
            f"HITL approval denied or not granted for DROP DATABASE '{database}'. "
            "Set HITL_DROP_DATABASE=true to allow automated drops in CI."
        )
    conn.execute(SQL("DROP DATABASE {}").format(Identifier(database)))


async def _async_safe_create_database(conn: AsyncConnection, database: str) -> None:
    await conn.execute(SQL("CREATE DATABASE {}").format(Identifier(database)))


async def _async_safe_drop_database(conn: AsyncConnection, database: str) -> None:
    if not _hitl_approve_drop(database):
        raise RuntimeError(
            f"HITL approval denied or not granted for DROP DATABASE '{database}'. "
            "Set HITL_DROP_DATABASE=true to allow automated drops in CI."
        )
    await conn.execute(SQL("DROP DATABASE {}").format(Identifier(database)))


@contextmanager
def _store_memory():
    store = InMemoryStore()
    yield store


@contextmanager
def _store_postgres():
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        _safe_create_database(conn, database)
    try:
        # yield store
        with PostgresStore.from_conn_string(DEFAULT_POSTGRES_URI + database) as store:
            store.setup()
            yield store
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            _safe_drop_database(conn, database)


@contextmanager
def _store_postgres_pipe():
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        _safe_create_database(conn, database)
    try:
        # yield store
        with PostgresStore.from_conn_string(DEFAULT_POSTGRES_URI + database) as store:
            store.setup()  # Run in its own transaction
        with PostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database, pipeline=True
        ) as store:
            yield store
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            _safe_drop_database(conn, database)


@contextmanager
def _store_postgres_pool():
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        _safe_create_database(conn, database)
    try:
        # yield store
        with PostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database, pool_config={"max_size": 10}
        ) as store:
            store.setup()
            yield store
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            _safe_drop_database(conn, database)


@asynccontextmanager
async def _store_postgres_aio():
    database = f"test_{uuid4().hex[:16]}"
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await _async_safe_create_database(conn, database)
    try:
        async with AsyncPostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database
        ) as store:
            await store.setup()
            yield store
    finally:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await _async_safe_drop_database(conn, database)


@asynccontextmanager
async def _store_postgres_aio_pipe():
    database = f"test_{uuid4().hex[:16]}"
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await _async_safe_create_database(conn, database)
    try:
        async with AsyncPostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database
        ) as store:
            await store.setup()  # Run in its own transaction
        async with AsyncPostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database, pipeline=True
        ) as store:
            yield store
    finally:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await _async_safe_drop_database(conn, database)


@asynccontextmanager
async def _store_postgres_aio_pool():
    database = f"test_{uuid4().hex[:16]}"
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await _async_safe_create_database(conn, database)
    try:
        async with AsyncPostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database,
            pool_config={"max_size": 10},
        ) as store:
            await store.setup()
            yield store
    finally:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await _async_safe_drop_database(conn, database)


__all__ = [
    "_store_memory",
    "_store_postgres",
    "_store_postgres_pipe",
    "_store_postgres_pool",
    "_store_postgres_aio",
    "_store_postgres_aio_pipe",
    "_store_postgres_aio_pool",
]