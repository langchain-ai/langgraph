import os
from collections.abc import AsyncIterator, Iterator
from uuid import UUID

import pytest
import redis
from langgraph.cache.base import BaseCache
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.redis import RedisCache
from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from pytest_mock import MockerFixture

from langgraph.types import Durability
from tests.conftest_checkpointer import (
    _checkpointer_memory,
    _checkpointer_memory_migrate_sends,
    _checkpointer_postgres,
    _checkpointer_postgres_aio,
    _checkpointer_postgres_aio_pipe,
    _checkpointer_postgres_aio_pool,
    _checkpointer_postgres_pipe,
    _checkpointer_postgres_pool,
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

NO_DOCKER = os.getenv("NO_DOCKER", "false") == "true"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)


@pytest.fixture(params=["sync", "async", "exit"])
def durability(request: pytest.FixtureRequest) -> Durability:
    return request.param


@pytest.fixture(
    scope="function",
    params=["sqlite", "memory"] if NO_DOCKER else ["sqlite", "memory", "redis"],
)
def cache(request: pytest.FixtureRequest) -> Iterator[BaseCache]:
    if request.param == "sqlite":
        yield SqliteCache(path=":memory:")
    elif request.param == "memory":
        yield InMemoryCache()
    elif request.param == "redis":
        # Get worker ID for parallel test isolation
        worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")

        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=False
        )
        # Use worker-specific prefix to avoid cache pollution between parallel tests
        cache = RedisCache(redis_client, prefix=f"test:cache:{worker_id}:")
        yield cache

        try:
            # Only clear keys with our specific prefix
            pattern = f"test:cache:{worker_id}:*"
            keys = redis_client.keys(pattern)
            if keys:
                redis_client.delete(*keys)
        except Exception:
            pass
    else:
        raise ValueError(f"Unknown cache type: {request.param}")


@pytest.fixture(
    scope="function",
    params=["in_memory"]
    if NO_DOCKER
    else ["in_memory", "postgres", "postgres_pipe", "postgres_pool"],
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
    params=["in_memory"]
    if NO_DOCKER
    else ["in_memory", "postgres_aio", "postgres_aio_pipe", "postgres_aio_pool"],
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
    ]
    if NO_DOCKER
    else [
        "memory",
        "memory_migrate_sends",
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
    elif checkpointer_name == "memory_migrate_sends":
        with _checkpointer_memory_migrate_sends() as checkpointer:
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
    ]
    if NO_DOCKER
    else [
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
