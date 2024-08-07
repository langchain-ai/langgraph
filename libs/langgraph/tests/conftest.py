from uuid import UUID

import pytest
from psycopg import AsyncConnection
from psycopg.errors import UndefinedTable
from psycopg.rows import dict_row
from pytest_mock import MockerFixture

from langgraph.checkpoint.postgres import PostgresSaver

DEFAULT_POSTGRES_URI = (
    "postgres://postgres:postgres@localhost:5442/postgres?sslmode=disable"
)


@pytest.fixture()
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)


@pytest.fixture(scope="function")
async def conn():
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        yield conn


@pytest.fixture(scope="session", autouse=True)
def setup_before_all_tests():
    with PostgresSaver.from_conn_string(DEFAULT_POSTGRES_URI) as checkpointer:
        checkpointer.setup()


@pytest.fixture(scope="function", autouse=True)
async def clear_test_db(conn):
    """Delete all tables before each test."""
    try:
        await conn.execute("DELETE FROM checkpoints")
        await conn.execute("DELETE FROM checkpoint_blobs")
        await conn.execute("DELETE FROM checkpoint_writes")
    except UndefinedTable:
        pass


pytest.register_assert_rewrite("tests.memory_assert")
