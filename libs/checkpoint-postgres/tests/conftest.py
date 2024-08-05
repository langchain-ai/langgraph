import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest
from psycopg import AsyncConnection
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool


def create_pool() -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    # parse connection string
    params = conninfo_to_dict(
        os.environ.get(
            "POSTGRES_URI",
            "postgres://postgres:postgres@localhost:5433/postgres?sslmode=disable",
        )
    )
    params.setdefault("options", "")
    params["options"] += " -c lock_timeout=1000"  # ms
    # create connection pool
    return AsyncConnectionPool(
        connection_class=AsyncConnection[DictRow],
        min_size=1,
        max_size=50,
        timeout=5,
        kwargs={
            **params,
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=False,
    )


@asynccontextmanager
async def connect() -> AsyncIterator[AsyncConnection[DictRow]]:
    async with create_pool() as pool:
        async with pool.connection() as conn:
            yield conn


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
async def conn(anyio_backend):
    async with connect() as conn:
        yield conn
