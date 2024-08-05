from uuid import UUID

import pytest
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from pytest_mock import MockerFixture


@pytest.fixture()
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)


DEFAULT_POSTGRES_URI = (
    "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable"
)


@pytest.fixture(scope="function")
async def conn():
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        yield conn


@pytest.fixture(scope="function", autouse=True)
async def clear_test_db(conn):
    """Delete all tables before each test."""
    await conn.execute("DROP TABLE IF EXISTS checkpoints")
    await conn.execute("DROP TABLE IF EXISTS checkpoint_blobs")
    await conn.execute("DROP TABLE IF EXISTS checkpoint_writes")


pytest.register_assert_rewrite("tests.memory_assert")
