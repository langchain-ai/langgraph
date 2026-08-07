"""Run delta-channel conformance capabilities against InMemorySaver."""

from __future__ import annotations

import pytest

conformance = pytest.importorskip(
    "langgraph.checkpoint.conformance",
    reason="langgraph-checkpoint-conformance not installed",
)


@pytest.mark.asyncio
async def test_delta_channel_conformance():
    # Imported inside the test: the module-level importorskip above is what
    # makes these safe, so they cannot move to the top of the file.
    from langgraph.checkpoint.conformance import validate  # noqa: PLC0415
    from langgraph.checkpoint.conformance.initializer import (  # noqa: PLC0415
        checkpointer_test,
    )

    from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415

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
