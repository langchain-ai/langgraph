"""Run delta-channel conformance capabilities against AsyncSqliteSaver."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "langgraph.checkpoint.conformance",
    reason="langgraph-checkpoint-conformance not installed",
)
pytest.importorskip("aiosqlite", reason="aiosqlite not installed")


@pytest.mark.asyncio
async def test_delta_channel_conformance():
    from langgraph.checkpoint.conformance import validate
    from langgraph.checkpoint.conformance.initializer import checkpointer_test

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    @checkpointer_test(name="AsyncSqliteSaver")
    async def sqlite_saver():
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            yield saver

    report = await validate(
        sqlite_saver,
        capabilities={
            "delta_channel_history",
        },
    )
    for cap, result in report.results.items():
        if result.passed is False:
            details = "\n".join(result.failures or [])
            pytest.fail(f"Capability {cap} failed:\n{details}")
