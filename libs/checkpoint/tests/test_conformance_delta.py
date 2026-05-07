"""Run delta-channel conformance capabilities against InMemorySaver."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip(
    "langgraph.checkpoint.conformance",
    reason="langgraph-checkpoint-conformance not installed",
)

from langgraph.checkpoint.conformance import validate  # noqa: E402
from langgraph.checkpoint.conformance.initializer import checkpointer_test  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402


@checkpointer_test(name="InMemorySaver")
async def _mem_saver():
    yield InMemorySaver()


@pytest.mark.asyncio
async def test_delta_channel_conformance():
    report = await validate(
        _mem_saver,
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
