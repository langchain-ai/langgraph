from collections.abc import AsyncIterator

import pytest
from psycopg import AsyncConnection
from psycopg.errors import UndefinedTable
from psycopg.rows import DictRow, dict_row

from tests.embed_test_utils import CharacterEmbeddings

DEFAULT_POSTGRES_URI = "postgres://postgres:postgres@localhost:5441/"
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
        await conn.execute("DELETE FROM checkpoint_migrations")
    except UndefinedTable:
        pass
    try:
        await conn.execute("DELETE FROM store_migrations")
        await conn.execute("DELETE FROM store")
    except UndefinedTable:
        pass


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    return CharacterEmbeddings(dims=500)


VECTOR_TYPES = ["vector", "halfvec"]
