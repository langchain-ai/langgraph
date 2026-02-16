"""Self-tests: run the validation suite against InMemorySaver."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.checkpoint.validation import checkpointer_test
from langgraph.checkpoint.validation.pytest import conformance_tests


@checkpointer_test(name="InMemorySaver")
async def memory_checkpointer():
    yield InMemorySaver()


# ---------------------------------------------------------------------------
# Option A: programmatic validate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_memory_base():
    """InMemorySaver passes all base capability tests."""
    from langgraph.checkpoint.validation import validate

    report = await validate(memory_checkpointer)
    report.print_report()
    assert report.passed_all_base(), f"Base tests failed: {report.to_dict()}"


# ---------------------------------------------------------------------------
# Option B: generated pytest functions via conformance_tests()
# ---------------------------------------------------------------------------

conformance_tests(memory_checkpointer)
