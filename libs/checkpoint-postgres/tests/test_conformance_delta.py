"""Run delta-channel conformance capabilities against AsyncPostgresSaver."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.conformance import validate
from langgraph.checkpoint.conformance.initializer import checkpointer_test

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from tests.conftest import DEFAULT_URI


@pytest.mark.asyncio
async def test_delta_channel_conformance():
    @checkpointer_test(name="AsyncPostgresSaver")
    async def postgres_saver():
        async with AsyncPostgresSaver.from_conn_string(DEFAULT_URI) as saver:
            await saver.setup()
            yield saver

    report = await validate(
        postgres_saver,
        capabilities={
            "delta_channel_history",
        },
    )
    for cap, result in report.results.items():
        if result.passed is False:
            details = "\n".join(result.failures or [])
            pytest.fail(f"Capability {cap} failed:\n{details}")
