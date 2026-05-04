"""Self-tests: run the conformance suite against InMemorySaver."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.checkpoint.conformance import checkpointer_test, validate


@checkpointer_test(name="InMemorySaver")
async def memory_checkpointer():
    yield InMemorySaver()


@pytest.mark.asyncio
async def test_validate_memory_base():
    """InMemorySaver passes all base capability tests."""
    report = await validate(memory_checkpointer)
    report.print_report()
    assert report.passed_all_base(), f"Base tests failed: {report.to_dict()}"
