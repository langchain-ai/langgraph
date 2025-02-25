import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from uuid import UUID, uuid4

import pytest
from langchain_core import __version__ as core_version
from packaging import version
from psycopg import AsyncConnection, Connection
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from pytest_mock import MockerFixture

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver, ShallowPostgresSaver
from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
    AsyncShallowPostgresSaver,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore, PostgresStore

pytest.register_assert_rewrite("tests.memory_assert")

DEFAULT_POSTGRES_URI = "postgres://postgres:postgres@localhost:5442/"
# TODO: fix this once core is released
IS_LANGCHAIN_CORE_030_OR_GREATER = version.parse(core_version) >= version.parse(
    "0.3.0.dev0"
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)


# checkpointer fixtures


@pytest.fixture(scope="function")
def checkpointer_memory():
    from tests.memory_assert import MemorySaverAssertImmutable

    yield MemorySaverAssertImmutable()


@pytest.fixture(scope="function")
def checkpointer_sqlite():
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield checkpointer


@asynccontextmanager
async def _checkpointer_sqlite_aio():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield checkpointer


@pytest.fixture(scope="function")
def checkpointer_postgres():
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


@pytest.fixture(scope="function")
def checkpointer_postgres_shallow():
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        # yield checkpointer
        with ShallowPostgresSaver.from_conn_string(
            DEFAULT_POSTGRES_URI + database
        ) as checkpointer:
            checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@pytest.fixture(scope="function")
def checkpointer_postgres_pipe():
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


@pytest.fixture(scope="function")
def checkpointer_postgres_pool():
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
async def _checkpointer_postgres_aio():
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
async def _checkpointer_postgres_aio_shallow():
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
        async with AsyncShallowPostgresSaver.from_conn_string(
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


@asynccontextmanager
async def awith_checkpointer(
    checkpointer_name: Optional[str],
) -> AsyncIterator[BaseCheckpointSaver]:
    if checkpointer_name is None:
        yield None
    elif checkpointer_name == "memory":
        from tests.memory_assert import MemorySaverAssertImmutable

        yield MemorySaverAssertImmutable()
    elif checkpointer_name == "sqlite_aio":
        async with _checkpointer_sqlite_aio() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_aio":
        async with _checkpointer_postgres_aio() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_aio_shallow":
        async with _checkpointer_postgres_aio_shallow() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_aio_pipe":
        async with _checkpointer_postgres_aio_pipe() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_aio_pool":
        async with _checkpointer_postgres_aio_pool() as checkpointer:
            yield checkpointer
    else:
        raise NotImplementedError(f"Unknown checkpointer: {checkpointer_name}")


@asynccontextmanager
async def _store_postgres_aio():
    if sys.version_info < (3, 10):
        pytest.skip("Async Postgres tests require Python 3.10+")
    database = f"test_{uuid4().hex[:16]}"
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
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
            await conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _store_postgres_aio_pipe():
    if sys.version_info < (3, 10):
        pytest.skip("Async Postgres tests require Python 3.10+")
    database = f"test_{uuid4().hex[:16]}"
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
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
            await conn.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _store_postgres_aio_pool():
    if sys.version_info < (3, 10):
        pytest.skip("Async Postgres tests require Python 3.10+")
    database = f"test_{uuid4().hex[:16]}"
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
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
            await conn.execute(f"DROP DATABASE {database}")


@pytest.fixture(scope="function")
def store_postgres():
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        # yield store
        with PostgresStore.from_conn_string(DEFAULT_POSTGRES_URI + database) as store:
            store.setup()
            yield store
    finally:
        # drop unique db
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


@pytest.fixture(scope="function")
def store_postgres_pipe():
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
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
            conn.execute(f"DROP DATABASE {database}")


@pytest.fixture(scope="function")
def store_postgres_pool():
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
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
            conn.execute(f"DROP DATABASE {database}")


@pytest.fixture(scope="function")
def store_in_memory():
    yield InMemoryStore()


@asynccontextmanager
async def awith_store(store_name: Optional[str]) -> AsyncIterator[BaseStore]:
    if store_name is None:
        yield None
    elif store_name == "in_memory":
        yield InMemoryStore()
    elif store_name == "postgres_aio":
        async with _store_postgres_aio() as store:
            yield store
    elif store_name == "postgres_aio_pipe":
        async with _store_postgres_aio_pipe() as store:
            yield store
    elif store_name == "postgres_aio_pool":
        async with _store_postgres_aio_pool() as store:
            yield store
    else:
        raise NotImplementedError(f"Unknown store {store_name}")


ALL_CHECKPOINTERS_SYNC = [
    "memory",
    "sqlite",
    "postgres",
    "postgres_pipe",
    "postgres_pool",
    "postgres_shallow",
]

ALL_CHECKPOINTERS_ASYNC = [
    "memory",
    "sqlite_aio",
    "postgres_aio",
    "postgres_aio_pipe",
    "postgres_aio_pool",
    "postgres_aio_shallow",
]
