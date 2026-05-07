"""Run delta-channel conformance capabilities against AsyncSqliteSaver."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "langgraph.checkpoint.conformance",
    reason="langgraph-checkpoint-conformance not installed",
)
pytest.importorskip("aiosqlite", reason="aiosqlite not installed")

from langgraph.checkpoint.conformance import validate  # noqa: E402
from langgraph.checkpoint.conformance.initializer import checkpointer_test  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402


@checkpointer_test(name="AsyncSqliteSaver")
async def _sqlite_saver():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        yield saver


@pytest.mark.asyncio
async def test_delta_channel_conformance():
    report = await validate(
        _sqlite_saver,
        capabilities={
            "delta_channel_history",
            "delta_channel_keepset",
            "delta_channel_reconstruction",
        },
    )
    for cap, result in report.results.items():
        if result.passed is False:
            details = "\n".join(result.failures or [])
            pytest.fail(f"Capability {cap} failed:\n{details}")
