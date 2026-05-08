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

DEFAULT_POSTGRES_URI = os.environ.get(
    "LANGGRAPH_TEST_POSTGRES_URI", "postgres://localhost:5442/"
)
AES_TEST_KEY = os.environb.get(b"LANGGRAPH_TEST_AES_KEY", None)

STRICT_MSGPACK = os.getenv("LANGGRAPH_STRICT_MSGPACK", "false").lower() in (
    "1",
    "true",
    "yes",
)


def _get_aes_key() -> bytes:
    if AES_TEST_KEY is None:
        raise ValueError(
            "LANGGRAPH_TEST_AES_KEY environment variable must be set to a 16, 24, or 32-byte value."
        )
    return AES_TEST_KEY


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


def _confirm_drop_database(database: str) -> bool:
    """HITL approval: prompt for confirmation before dropping a database."""
    confirm = input(
        f"HITL APPROVAL REQUIRED: Are you sure you want to DROP DATABASE '{database}'? "
        "This is a destructive operation. Type 'yes' to confirm: "
    )
    return confirm.strip().lower() == "yes"


async def _confirm_drop_database_async(database: str) -> bool:
    """HITL approval for async context: prompt for confirmation before dropping a database."""
    import asyncio

    loop = asyncio.get_event_loop()
    confirm = await loop.run_in_executor(
        None,
        lambda: input(
            f"HITL APPROVAL REQUIRED: Are you sure you want to DROP DATABASE '{database}'? "
            "This is a destructive operation. Type 'yes' to confirm: "
        ),
    )
    return confirm.strip().lower() == "yes"


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
        aes_key = _get_aes_key()
        if STRICT_MSGPACK:
            checkpointer.serde = EncryptedSerializer.from_pycryptodome_aes(
                serde=_strict_msgpack_serde(), key=aes_key
            )
        else:
            checkpointer.serde = EncryptedSerializer.from_pycryptodome_aes(
                key=aes_key
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
        if _confirm_drop_database(database):
            with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
                conn.execute(f"DROP DATABASE {database}")
        else:
            raise RuntimeError(
                f"DROP DATABASE '{database}' was not approved. Manual cleanup required."
            )


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
        if _confirm_drop_database(database):
            with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
                conn.execute(f"DROP DATABASE {database}")
        else:
            raise RuntimeError(
                f"DROP DATABASE '{database}' was not approved. Manual cleanup required."
            )


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
        if _confirm_drop_database(database):
            with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
                conn.execute(f"DROP DATABASE {database}")
        else:
            raise RuntimeError(
                f"DROP DATABASE '{database}' was not approved. Manual cleanup required."
            )


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
        if await _confirm_drop_database_async(database):
            async with await AsyncConnection.connect(
                DEFAULT_POSTGRES_URI, autocommit=True
            ) as conn:
                await conn.execute(f"DROP DATABASE {database}")
        else:
            raise RuntimeError(
                f"DROP DATABASE '{database}' was not approved. Manual cleanup required."
            )


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
        if await _confirm_drop_database_async(database):
            async with await AsyncConnection.connect(
                DEFAULT_POSTGRES_URI, autocommit=True
            ) as conn:
                await conn.execute(f"DROP DATABASE {database}")
        else:
            raise RuntimeError(
                f"DROP DATABASE '{database}' was not approved. Manual cleanup required."
            )


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
        if await _confirm_drop_database_async(database):
            async with await AsyncConnection.connect(
                DEFAULT_POSTGRES_URI, autocommit=True
            ) as conn:
                await conn.execute(f"DROP DATABASE {database}")
        else:
            raise RuntimeError(
                f"DROP DATABASE '{database}' was not approved. Manual cleanup required."
            )


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