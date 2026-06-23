"""Phase 4: harden — parity, multi-version windows, progress events."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from langgraph.capability.errors import CapabilityContractError, CapabilityVersionError
from langgraph.capability.examples.research import RESEARCH_CAPABILITY, RESEARCH_SPEC
from langgraph.capability.examples.service_deploy import research_service_capability_for_tests
from langgraph.capability.parity import (
    DEFAULT_PARITY_DISCLAIMER,
    assert_io_compatible,
    compare_capability_parity,
    is_breaking_schema_change,
)
from langgraph.capability.progress import (
    ProgressPhase,
    emit_run_progress,
    iter_progress_dicts,
    progress_events_from_run_result,
)
from langgraph.capability.service import ServiceRunStatus
from langgraph.capability.versioning import (
    ServiceVersionPolicy,
    VersionWindow,
    bind_service_to_policy,
    default_n_minus_one_policy,
)
from langgraph.capability.contract import CapabilitySpec
from typing_extensions import TypedDict


def test_parity_identical_io_for_reference_research() -> None:
    svc = research_service_capability_for_tests()
    report = compare_capability_parity(RESEARCH_CAPABILITY, svc)
    assert report.level == "identical_io"
    assert report.same_input_schema and report.same_output_schema
    assert DEFAULT_PARITY_DISCLAIMER in report.disclaimer
    d = report.to_dict()
    assert d["capability_id"] == "langgraph.research"


def test_parity_package_only() -> None:
    report = compare_capability_parity(RESEARCH_CAPABILITY, None)
    assert report.level == "unknown"
    assert "No service" in report.notes[0]


def test_assert_io_compatible_and_breaking() -> None:
    class In(TypedDict):
        q: str

    class Out(TypedDict):
        a: str

    class Out2(TypedDict):
        b: str

    a = CapabilitySpec("demo.x", "1.0.0", In, Out)
    b = CapabilitySpec("demo.x", "1.1.0", In, Out)
    c = CapabilitySpec("demo.x", "2.0.0", In, Out2)
    assert_io_compatible(a, b)
    with pytest.raises(CapabilityContractError):
        assert_io_compatible(a, c, require_same_major=True)
    assert is_breaking_schema_change(a, c) is True
    assert is_breaking_schema_change(a, b) is False


def test_version_policy_n_minus_one() -> None:
    policy = default_n_minus_one_policy(
        "langgraph.research",
        "1.1.0",
        "1.0.0",
        previous_retired_after=date.today() + timedelta(days=30),
    )
    assert policy.supports("1.1.0")
    assert policy.supports("1.0.0")
    w = policy.resolve("1")
    assert w.version in {"1.1.0", "1.0.0"}
    assert policy.to_dict()["capability_id"] == "langgraph.research"


def test_version_policy_retired_window() -> None:
    policy = ServiceVersionPolicy(
        capability_id="langgraph.research",
        windows=[
            VersionWindow(version="1.0.0", retired_after=date.today() - timedelta(days=1)),
            VersionWindow(version="1.1.0", label="current"),
        ],
    )
    assert policy.supports("1.0.0") is False
    assert policy.resolve("1.1.0").version == "1.1.0"
    with pytest.raises(CapabilityVersionError):
        policy.resolve("9.9.9")


def test_bind_service_to_policy() -> None:
    svc = research_service_capability_for_tests()
    policy = default_n_minus_one_policy("langgraph.research", "1.0.0")
    pinned = bind_service_to_policy(svc, policy, "1.0.0")
    assert pinned.endpoint.version_label == "1.0.0"
    out = pinned.invoke({"query": "p", "max_sources": 1})
    assert out["sources"]


def test_progress_events_from_success() -> None:
    svc = research_service_capability_for_tests()
    result = svc.invoke_with_status({"query": "x", "max_sources": 1})
    events = progress_events_from_run_result(result)
    phases = [e.phase for e in events]
    assert ProgressPhase.ACCEPTED in phases
    assert ProgressPhase.SUCCEEDED in phases
    assert events[-1].percent == 100.0
    dicts = list(iter_progress_dicts(result))
    assert dicts[-1]["type"] == "capability_progress"


def test_progress_events_from_failure() -> None:
    from langgraph.capability.service import ServiceRunResult

    result = ServiceRunResult(
        output=None,
        status=ServiceRunStatus.FAILED,
        run_id="r1",
        capability_id="langgraph.research",
        version="1.0.0",
        error_message="nope",
    )
    events = progress_events_from_run_result(result)
    assert events[-1].phase is ProgressPhase.FAILED
    assert events[-1].message == "nope"

    seen: list[str] = []
    emit_run_progress(result, lambda ev: seen.append(ev.phase.value))
    assert "failed" in seen


def test_spec_still_exports_research() -> None:
    assert RESEARCH_SPEC.version == "1.0.0"
