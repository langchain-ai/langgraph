from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Optional
from uuid import UUID

import pytest
from langchain_core import __version__ as core_version
from packaging import version
from pytest_mock import MockerFixture

from langgraph.cache.base import BaseCache
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from tests.conftest_checkpointer import (
    _checkpointer_memory,
    _checkpointer_postgres,
    _checkpointer_postgres_aio,
    _checkpointer_postgres_aio_pipe,
    _checkpointer_postgres_aio_pool,
    _checkpointer_postgres_aio_shallow,
    _checkpointer_postgres_pipe,
    _checkpointer_postgres_pool,
    _checkpointer_postgres_shallow,
    _checkpointer_sqlite,
    _checkpointer_sqlite_aes,
    _checkpointer_sqlite_aio,
)
from tests.conftest_store import (
    _store_memory,
    _store_postgres,
    _store_postgres_aio,
    _store_postgres_aio_pipe,
    _store_postgres_aio_pool,
    _store_postgres_pipe,
    _store_postgres_pool,
)

pytest.register_assert_rewrite("tests.memory_assert")

# TODO: fix this once core is released
IS_LANGCHAIN_CORE_030_OR_GREATER = version.parse(core_version) >= version.parse(
    "0.3.0.dev0"
)
SHOULD_CHECK_SNAPSHOTS = IS_LANGCHAIN_CORE_030_OR_GREATER


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)


@pytest.fixture(params=[True, False])
def checkpoint_during(request: pytest.FixtureRequest) -> bool:
    return request.param


# --- start of deprecated fixtures ---


@pytest.fixture(scope="function")
def checkpointer_memory():
    with _checkpointer_memory() as checkpointer:
        yield checkpointer


@pytest.fixture(scope="function")
def checkpointer_sqlite():
    with _checkpointer_sqlite() as checkpointer:
        yield checkpointer


@pytest.fixture(scope="function")
def checkpointer_sqlite_aes():
    with _checkpointer_sqlite_aes() as checkpointer:
        yield checkpointer


@pytest.fixture(scope="function")
def checkpointer_postgres():
    with _checkpointer_postgres() as checkpointer:
        yield checkpointer


@pytest.fixture(scope="function")
def checkpointer_postgres_shallow():
    with _checkpointer_postgres_shallow() as checkpointer:
        yield checkpointer


@pytest.fixture(scope="function")
def checkpointer_postgres_pipe():
    with _checkpointer_postgres_pipe() as checkpointer:
        yield checkpointer


@pytest.fixture(scope="function")
def checkpointer_postgres_pool():
    with _checkpointer_postgres_pool() as checkpointer:
        yield checkpointer


@asynccontextmanager
async def awith_checkpointer(
    checkpointer_name: Optional[str],
) -> AsyncIterator[BaseCheckpointSaver]:
    if checkpointer_name is None:
        yield None
    elif checkpointer_name == "memory":
        with _checkpointer_memory() as checkpointer:
            yield checkpointer
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


# --- end of deprecated fixtures ---


@pytest.fixture(scope="function", params=["sqlite", "memory"])
def cache(request: pytest.FixtureRequest) -> Iterator[BaseCache]:
    if request.param == "sqlite":
        yield SqliteCache(path=":memory:")
    elif request.param == "memory":
        yield InMemoryCache()
    else:
        raise ValueError(f"Unknown cache type: {request.param}")


@pytest.fixture(
    scope="function",
    params=["in_memory", "postgres", "postgres_pipe", "postgres_pool"],
)
def sync_store(request: pytest.FixtureRequest) -> Iterator[BaseStore]:
    store_name = request.param
    if store_name is None:
        yield None
    elif store_name == "in_memory":
        with _store_memory() as store:
            yield store
    elif store_name == "postgres":
        with _store_postgres() as store:
            yield store
    elif store_name == "postgres_pipe":
        with _store_postgres_pipe() as store:
            yield store
    elif store_name == "postgres_pool":
        with _store_postgres_pool() as store:
            yield store
    else:
        raise NotImplementedError(f"Unknown store {store_name}")


@pytest.fixture(
    scope="function",
    params=["in_memory", "postgres_aio", "postgres_aio_pipe", "postgres_aio_pool"],
)
async def async_store(request: pytest.FixtureRequest) -> AsyncIterator[BaseStore]:
    store_name = request.param
    if store_name is None:
        yield None
    elif store_name == "in_memory":
        with _store_memory() as store:
            yield store
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


@pytest.fixture(
    scope="function",
    params=[
        "memory",
        "sqlite",
        "sqlite_aes",
        "postgres",
        "postgres_pipe",
        "postgres_pool",
    ],
)
def sync_checkpointer(
    request: pytest.FixtureRequest,
) -> Iterator[BaseCheckpointSaver]:
    checkpointer_name = request.param
    if checkpointer_name == "memory":
        with _checkpointer_memory() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "sqlite":
        with _checkpointer_sqlite() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "sqlite_aes":
        with _checkpointer_sqlite_aes() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres":
        with _checkpointer_postgres() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_pipe":
        with _checkpointer_postgres_pipe() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_pool":
        with _checkpointer_postgres_pool() as checkpointer:
            yield checkpointer
    else:
        raise NotImplementedError(f"Unknown checkpointer: {checkpointer_name}")


@pytest.fixture(
    scope="function",
    params=[
        "memory",
        "sqlite_aio",
        "postgres_aio",
        "postgres_aio_pipe",
        "postgres_aio_pool",
    ],
)
async def async_checkpointer(
    request: pytest.FixtureRequest,
) -> AsyncIterator[BaseCheckpointSaver]:
    checkpointer_name = request.param
    if checkpointer_name == "memory":
        with _checkpointer_memory() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "sqlite_aio":
        async with _checkpointer_sqlite_aio() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_aio":
        async with _checkpointer_postgres_aio() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_aio_pipe":
        async with _checkpointer_postgres_aio_pipe() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_aio_pool":
        async with _checkpointer_postgres_aio_pool() as checkpointer:
            yield checkpointer
    else:
        raise NotImplementedError(f"Unknown checkpointer: {checkpointer_name}")


ALL_CHECKPOINTERS_SYNC = [
    "memory",
    "sqlite",
    "sqlite_aes",
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
