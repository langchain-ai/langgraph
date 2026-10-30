"""Tests for the conformance level reported by ``CapabilityReport``."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.checkpoint.conformance import (
    checkpointer_test,
    validate,
)
from langgraph.checkpoint.conformance.capabilities import Capability
from langgraph.checkpoint.conformance.report import (
    CapabilityReport,
    CapabilityResult,
)
from langgraph.checkpoint.conformance.validate import _RUNNERS


@checkpointer_test(name="NoopSaver")
async def noop_checkpointer():
    # Implements none of the methods the suite looks for.
    yield BaseCheckpointSaver()


@checkpointer_test(name="StoreOnlySaver")
async def store_only_checkpointer():
    class StoreOnlySaver(BaseCheckpointSaver):
        # Only the signatures matter here: every call returns nothing.
        async def aput(self, config, checkpoint, metadata, new_versions):
            return {}

        async def aget_tuple(self, config):
            return None

        async def alist(self, config, *, filter=None, before=None, limit=None):
            yield None

        async def aput_writes(self, config, writes, task_id, task_path=""):
            return None

        async def adelete_thread(self, thread_id):
            return None

    yield StoreOnlySaver()


async def test_noop_checkpointer_is_not_full():
    report = await validate(noop_checkpointer)

    assert not report.detected_capabilities
    assert report.conformance_level() == "NONE"
    assert report.to_dict()["conformance_level"] == "NONE"


async def test_all_detected_tests_passing_is_full():
    @checkpointer_test(name="InMemorySaver")
    async def memory_checkpointer():
        yield InMemorySaver()

    report = await validate(memory_checkpointer)

    assert report.passed_all()
    assert report.conformance_level() == "FULL"


async def test_implemented_base_with_failing_tests_is_reported():
    report = await validate(store_only_checkpointer)

    # Sanity check: the saver is detected, its tests just don't pass.
    assert report.detected_capabilities == [c.value for c in Capability if c in BASE]
    assert not report.passed_all()
    assert report.conformance_level() == "NONE"


BASE = [
    Capability.PUT,
    Capability.PUT_WRITES,
    Capability.GET_TUPLE,
    Capability.LIST,
    Capability.DELETE_THREAD,
]
EXTENDED = [Capability.COPY_THREAD, Capability.PRUNE]


def _report(*groups: dict[str, bool]) -> CapabilityReport:
    """Build a report where the given ``{capability: passed}`` maps were detected."""
    passed = {k: v for group in groups for k, v in group.items()}
    report = CapabilityReport(checkpointer_name="Fake")
    for cap in Capability:
        detected = cap.value in passed
        report.results[cap.value] = CapabilityResult(
            detected=detected,
            passed=passed.get(cap.value),
            tests_skipped=0 if detected else 1,
        )
    return report


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        ((), "NONE"),
        (({c.value: True for c in BASE if c is not Capability.DELETE_THREAD},), "BASE"),
        (({c.value: True for c in Capability},), "FULL"),
        (({c.value: True for c in BASE}, {EXTENDED[0].value: False}), "BASE+PARTIAL"),
        (
            (
                {c.value: True for c in BASE if c is not Capability.DELETE_THREAD},
                {Capability.DELETE_THREAD.value: False},
            ),
            "BASE",
        ),
        (({c.value: True for c in EXTENDED},), "NONE"),
        (({EXTENDED[0].value: True},), "NONE"),
        (({c.value: False for c in BASE},), "NONE"),
    ],
)
def test_levels(groups: tuple[dict[str, bool], ...], expected: str):
    assert _report(*groups).conformance_level() == expected


def test_every_capability_has_a_runner():
    # validate() silently records a capability as skipped when its runner is
    # missing, so a missing runner would make a saver look better than it is.
    assert set(_RUNNERS) == set(Capability)
