"""Tests that `from_conn_string` forwards `serde` to the saver (no database needed)."""

from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@pytest.fixture(autouse=True)
def clear_test_db() -> None:
    """Override the conftest fixture: these tests mock the connection."""


@pytest.mark.parametrize("pipeline", [False, True])
def test_sync_from_conn_string_custom_serde(pipeline: bool) -> None:
    serde = JsonPlusSerializer()
    with patch("langgraph.checkpoint.postgres.Connection.connect") as connect:
        connect.return_value = MagicMock()
        conn = connect.return_value.__enter__.return_value
        conn.pipeline.return_value.__enter__.return_value = MagicMock()
        with PostgresSaver.from_conn_string(
            "postgresql://", pipeline=pipeline, serde=serde
        ) as saver:
            assert saver.serde is serde


def test_sync_from_conn_string_default_serde() -> None:
    with patch("langgraph.checkpoint.postgres.Connection.connect") as connect:
        connect.return_value = MagicMock()
        with PostgresSaver.from_conn_string("postgresql://") as saver:
            assert saver.serde is not None
            assert isinstance(saver.serde, type(PostgresSaver(MagicMock()).serde))


async def test_async_from_conn_string_custom_serde() -> None:
    serde = JsonPlusSerializer()
    with patch("langgraph.checkpoint.postgres.aio.AsyncConnection.connect") as connect:
        connect.return_value = MagicMock()
        connect.return_value.__aenter__.return_value = MagicMock()
        async with AsyncPostgresSaver.from_conn_string(
            "postgresql://", serde=serde
        ) as saver:
            assert saver.serde is serde
