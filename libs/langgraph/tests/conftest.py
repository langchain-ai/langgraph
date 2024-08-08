import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator, TypeVar
from uuid import UUID, uuid4

import pytest
from psycopg import AsyncConnection, Connection
from pytest_mock import MockerFixture

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from tests.memory_assert import MemorySaverAssertImmutable

DEFAULT_POSTGRES_URI = "postgres://postgres:postgres@localhost:5442/"


@pytest.fixture()
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)


"""
pytest-asyncio doesn't support calling async fixtures with getfixturevalue
so we need to use ThreadPoolExecutor to run the async fixture in a thread
https://github.com/pytest-dev/pytest-asyncio/issues/112#issuecomment-462062890
"""
T = TypeVar("T")


def close_loop(loop: asyncio.AbstractEventLoop) -> None:
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.run_until_complete(loop.shutdown_default_executor())
    asyncio.set_event_loop(None)
    loop.close()


@contextmanager
def agen_to_gen(agen: AsyncIterator[T]) -> Iterator[T]:
    with ThreadPoolExecutor(1) as bg:
        loop = asyncio.new_event_loop()
        bg.submit(asyncio.set_event_loop, loop).result()
        try:
            yield bg.submit(loop.run_until_complete, agen.__aenter__()).result()
        finally:
            bg.submit(
                loop.run_until_complete, agen.__aexit__(None, None, None)
            ).result()
            bg.submit(close_loop, loop).result()


# checkpointer fixtures


@pytest.fixture(scope="function")
def checkpointer_memory():
    yield MemorySaverAssertImmutable()


@pytest.fixture(scope="function")
def checkpointer_sqlite():
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield checkpointer


@pytest.fixture(scope="function")
def checkpointer_sqlite_aio():
    with agen_to_gen(_checkpointer_sqlite_aio()) as checkpointer:
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
def checkpointer_postgres_aio():
    if sys.version_info < (3, 10):
        pytest.skip("Async Postgres tests require Python 3.10+")
    with agen_to_gen(_checkpointer_postgres_aio()) as checkpointer:
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
            yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@pytest.fixture(scope="function")
def checkpointer_postgres_aio_pipe():
    if sys.version_info < (3, 10):
        pytest.skip("Async Postgres tests require Python 3.10+")
    with agen_to_gen(_checkpointer_postgres_aio_pipe()) as checkpointer:
        yield checkpointer


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
                yield checkpointer
    finally:
        # drop unique db
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")
