from contextlib import asynccontextmanager, contextmanager
import os
import re
from uuid import uuid4

from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore, PostgresStore
from psycopg import AsyncConnection, Connection

DEFAULT_POSTGRES_URI = os.environ.get(
    "POSTGRES_URI", "postgres://postgres@localhost:5442/"
)


def _get_safe_database_name() -> str:
    """Generate a safe database name using only alphanumeric characters."""
    hex_part = uuid4().hex[:16]
    # Validate that the name contains only safe characters
    if not re.match(r'^[a-f0-9]+$', hex_part):
        raise ValueError("Generated database name contains unsafe characters")
    return f"test_{hex_part}"


def _validate_identifier(name: str) -> str:
    """Validate that a database identifier contains only safe characters."""
    if not re.match(r'^[a-zA-Z0-9_]+$', name):
        raise ValueError(f"Unsafe database identifier: {name}")
    return name


def _request_hitl_approval(operation: str, target: str) -> bool:
    """Request Human-in-the-Loop approval for destructive operations."""
    auto_approve = os.environ.get("LANGGRAPH_TEST_AUTO_APPROVE_DROP", "").lower()
    if auto_approve in ("1", "true", "yes"):
        return True
    try:
        response = input(
            f"\n[HITL APPROVAL REQUIRED] About to execute: {operation} on '{target}'. "
            f"Type 'yes' to confirm: "
        ).strip().lower()
        return response == "yes"
    except (EOFError, OSError):
        # Non-interactive environment: deny by default unless env var is set
        return False


@contextmanager
def _store_memory():
    store = InMemoryStore()
    yield store


@contextmanager
def _store_postgres():
    database = _get_safe_database_name()
    safe_db = _validate_identifier(database)
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute("CREATE DATABASE " + safe_db)
    try:
        # yield store
        with PostgresStore.from_conn_string(DEFAULT_POSTGRES_URI + database) as store:
            store.setup()
            yield store
    finally:
        # drop unique db - requires HITL approval
        if _request_hitl_approval("DROP DATABASE", safe_db):
            with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
                conn.execute("DROP DATABASE " + safe_db)


@contextmanager
def _store_postgres_pipe():
    database = _get_safe_database_name()
    safe_db = _validate_identifier(database)
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute("CREATE DATABASE " + safe_db)
    try:
        # yield store
        with PostgresStore.from_conn_string(DEFAULT_POSTGRES_URI + database) as store:
            store.setup()  # Run in its own transaction
        with PostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database, pipeline=True
        ) as store:
            yield store
    finally:
        # drop unique db - requires HITL approval
        if _request_hitl_approval("DROP DATABASE", safe_db):
            with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
                conn.execute("DROP DATABASE " + safe_db)


@contextmanager
def _store_postgres_pool():
    database = _get_safe_database_name()
    safe_db = _validate_identifier(database)
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute("CREATE DATABASE " + safe_db)
    try:
        # yield store
        with PostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database, pool_config={"max_size": 10}
        ) as store:
            store.setup()
            yield store
    finally:
        # drop unique db - requires HITL approval
        if _request_hitl_approval("DROP DATABASE", safe_db):
            with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
                conn.execute("DROP DATABASE " + safe_db)


@asynccontextmanager
async def _store_postgres_aio():
    database = _get_safe_database_name()
    safe_db = _validate_identifier(database)
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute("CREATE DATABASE " + safe_db)
    try:
        async with AsyncPostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database
        ) as store:
            await store.setup()
            yield store
    finally:
        # drop unique db - requires HITL approval
        if _request_hitl_approval("DROP DATABASE", safe_db):
            async with await AsyncConnection.connect(
                DEFAULT_POSTGRES_URI, autocommit=True
            ) as conn:
                await conn.execute("DROP DATABASE " + safe_db)


@asynccontextmanager
async def _store_postgres_aio_pipe():
    database = _get_safe_database_name()
    safe_db = _validate_identifier(database)
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute("CREATE DATABASE " + safe_db)
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
        # drop unique db - requires HITL approval
        if _request_hitl_approval("DROP DATABASE", safe_db):
            async with await AsyncConnection.connect(
                DEFAULT_POSTGRES_URI, autocommit=True
            ) as conn:
                await conn.execute("DROP DATABASE " + safe_db)


@asynccontextmanager
async def _store_postgres_aio_pool():
    database = _get_safe_database_name()
    safe_db = _validate_identifier(database)
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute("CREATE DATABASE " + safe_db)
    try:
        async with AsyncPostgresStore.from_conn_string(
            DEFAULT_POSTGRES_URI + database,
            pool_config={"max_size": 10},
        ) as store:
            await store.setup()
            yield store
    finally:
        # drop unique db - requires HITL approval
        if _request_hitl_approval("DROP DATABASE", safe_db):
            async with await AsyncConnection.connect(
                DEFAULT_POSTGRES_URI, autocommit=True
            ) as conn:
                await conn.execute("DROP DATABASE " + safe_db)


__all__ = [
    "_store_memory",
    "_store_postgres",
    "_store_postgres_pipe",
    "_store_postgres_pool",
    "_store_postgres_aio",
    "_store_postgres_aio_pipe",
    "_store_postgres_aio_pool",
]