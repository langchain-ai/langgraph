"""Self-tests: run the conformance suite against InMemorySaver."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.checkpoint.conformance import (
    checkpointer_test,
    validate,
    validate_get_writes_history,
)


@checkpointer_test(name="InMemorySaver")
async def memory_checkpointer():
    yield InMemorySaver()


@pytest.mark.asyncio
async def test_validate_memory_base():
    """InMemorySaver passes all base capability tests."""
    report = await validate(memory_checkpointer)
    report.print_report()
    assert report.passed_all_base(), f"Base tests failed: {report.to_dict()}"


@pytest.mark.asyncio
async def test_validate_memory_get_writes_history():
    """InMemorySaver passes the `get_writes_history` conformance suite."""

    @asynccontextmanager
    async def factory():
        # Each scenario gets a fresh saver — no cross-scenario thread ID
        # collisions even though the helper itself uses uuid-prefixed IDs.
        yield InMemorySaver()

    await validate_get_writes_history(factory)
