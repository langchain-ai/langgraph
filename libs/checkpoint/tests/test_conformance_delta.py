"""Run delta-channel conformance capabilities against InMemorySaver."""

from __future__ import annotations

import pytest

conformance = pytest.importorskip(
    "langgraph.checkpoint.conformance",
    reason="langgraph-checkpoint-conformance not installed",
)


@pytest.mark.asyncio
async def test_delta_channel_conformance():
    from langgraph.checkpoint.conformance import validate
    from langgraph.checkpoint.conformance.initializer import checkpointer_test

    from langgraph.checkpoint.memory import InMemorySaver

    @checkpointer_test(name="InMemorySaver")
    async def mem_saver():
        yield InMemorySaver()

    report = await validate(
        mem_saver,
        capabilities={
            "delta_channel_history",
        },
    )
    for cap, result in report.results.items():
        if result.passed is False:
            details = "\n".join(result.failures or [])
            pytest.fail(f"Capability {cap} failed:\n{details}")
