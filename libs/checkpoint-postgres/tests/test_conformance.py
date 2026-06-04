"""Conformance tests for AsyncPostgresSaver."""
# mypy: disable-error-code="import-untyped"

from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.report import ProgressCallbacks
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from tests.conftest import DEFAULT_POSTGRES_URI


async def pg_lifespan() -> AsyncGenerator[None, None]:
    """No-op lifespan; databases are created per-checkpointer instance."""
    yield


@checkpointer_test(name="AsyncPostgresSaver", lifespan=pg_lifespan)
async def postgres_checkpointer() -> AsyncGenerator[AsyncPostgresSaver, None]:
    database = f"test_{uuid4().hex[:16]}"
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            saver = AsyncPostgresSaver(conn)
            await saver.setup()
            yield saver
    finally:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as conn:
            await conn.execute(f"DROP DATABASE {database}")


@pytest.mark.asyncio
async def test_full_conformance() -> None:
    """AsyncPostgresSaver passes ALL conformance tests."""
    report = await validate(
        postgres_checkpointer,
        progress=ProgressCallbacks.verbose(),
    )
    report.print_report()
    assert report.passed_all(), f"Conformance failed: {report.to_dict()}"
