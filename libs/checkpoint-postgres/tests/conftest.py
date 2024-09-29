from typing import AsyncIterator

import pytest
from psycopg import AsyncConnection
from psycopg.errors import UndefinedTable
from psycopg.rows import DictRow, dict_row

DEFAULT_URI = "postgres://postgres:postgres@localhost:5441/postgres?sslmode=disable"


@pytest.fixture(scope="function")
async def conn() -> AsyncIterator[AsyncConnection[DictRow]]:
    async with await AsyncConnection.connect(
        DEFAULT_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        yield conn


@pytest.fixture(scope="function", autouse=True)
async def clear_test_db(conn: AsyncConnection[DictRow]) -> None:
    """Delete all tables before each test."""
    try:
        await conn.execute("DELETE FROM checkpoints")
        await conn.execute("DELETE FROM checkpoint_blobs")
        await conn.execute("DELETE FROM checkpoint_writes")
    except UndefinedTable:
        pass
