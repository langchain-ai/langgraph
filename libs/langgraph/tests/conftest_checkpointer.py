import os
from contextlib import asynccontextmanager, contextmanager
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from psycopg import AsyncConnection, Connection
from psycopg_pool import AsyncConnectionPool, ConnectionPool

pytest.register_assert_rewrite("tests.memory_assert")

from tests.memory_assert import (  # noqa: E402
    MemorySaverAssertImmutable,
    MemorySaverNeedsPendingSendsMigration,
)

DEFAULT_POSTGRES_URI = "postgres://postgres:postgres@localhost:5442/"
STRICT_MSGPACK = os.getenv("LANGGRAPH_STRICT_MSGPACK", "false").lower() in (
    "1",
    "true",
    "yes",
)


def _strict_msgpack_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=None)


def _apply_strict_msgpack(checkpointer) -> None:
    if not STRICT_MSGPACK:
        return
    serde = _strict_msgpack_serde()
    if hasattr(checkpointer, "serde"):
        checkpointer.serde = serde
    if hasattr(checkpointer, "saver") and hasattr(checkpointer.saver, "serde"):
        checkpointer.saver.serde = serde


@contextmanager
def _checkpointer_memory():
    if STRICT_MSGPACK:
        yield MemorySaverAssertImmutable(serde=_strict_msgpack_serde())
    else:
        yield MemorySaverAssertImmutable()


@contextmanager
def _checkpointer_memory_migrate_sends():
    checkpointer = MemorySaverNeedsPendingSendsMigration()
    _apply_strict_msgpack(checkpointer)
    yield checkpointer


@contextmanager
def _checkpointer_sqlite():
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        _apply_strict_msgpack(checkpointer)
        yield checkpointer


@contextmanager
def _checkpointer_sqlite_aes():
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        if STRICT_MSGPACK:
            checkpointer.serde = EncryptedSerializer.from_pycryptodome_aes(
                serde=_strict_msgpack_serde(), key=b"1234567890123456"
            )
        else:
            checkpointer.serde = EncryptedSerializer.from_pycryptodome_aes(
                key=b"1234567890123456"
            )
        yield checkpointer


@contextmanager
def _checkpointer_postgres():
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
            _apply_strict_msgpack(checkpointer)
            yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@contextmanager
def _checkpointer_postgres_pipe():
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
                _apply_strict_msgpack(checkpointer)
                yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@contextmanager
def _checkpointer_postgres_pool():
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
            _apply_strict_msgpack(checkpointer)
            yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _checkpointer_sqlite_aio():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        _apply_strict_msgpack(checkpointer)
        yield checkpointer


@asynccontextmanager
async def _checkpointer_postgres_aio():
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
            _apply_strict_msgpack(checkpointer)
            yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _checkpointer_postgres_aio_pipe():
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
                _apply_strict_msgpack(checkpointer)
                yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _checkpointer_postgres_aio_pool():
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
            _apply_strict_msgpack(checkpointer)
            yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


__all__ = [
    "_checkpointer_memory",
    "_checkpointer_memory_migrate_sends",
    "_checkpointer_sqlite",
    "_checkpointer_sqlite_aes",
    "_checkpointer_postgres",
    "_checkpointer_postgres_pipe",
    "_checkpointer_postgres_pool",
    "_checkpointer_sqlite_aio",
    "_checkpointer_postgres_aio",
    "_checkpointer_postgres_aio_pipe",
    "_checkpointer_postgres_aio_pool",
]
