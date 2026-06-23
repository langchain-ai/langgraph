"""Phase 2: service-mode capabilities (black-box remote delivery)."""

from __future__ import annotations

import pytest

from langgraph.capability.examples.parent_remote import build_parent_with_remote_research
from langgraph.capability.examples.research import RESEARCH_CAPABILITY, RESEARCH_SPEC
from langgraph.capability.examples.service_deploy import (
    LANGGRAPH_JSON_EXAMPLE,
    RESEARCH_SERVICE_GRAPH_ID,
    build_research_service_graph,
    research_service_capability_for_tests,
)
from langgraph.capability.service import (
    ServiceEndpoint,
    ServiceRunStatus,
    iter_boundary_events,
    local_service_invoker,
    service_capability,
    service_capability_from_package,
)


def test_service_invoke_via_local_invoker() -> None:
    svc = research_service_capability_for_tests(prefix="edge")
    out = svc.invoke({"query": "q", "max_sources": 1})
    assert out["sources"] == ["edge-source-1"]
    assert out["query"] == "q"


def test_service_invoke_with_status_and_boundary_events() -> None:
    svc = research_service_capability_for_tests()
    result = svc.invoke_with_status({"query": "x", "max_sources": 1})
    assert result.status is ServiceRunStatus.SUCCEEDED
    assert result.run_id
    assert result.capability_id == "langgraph.research"
    events = list(iter_boundary_events(result))
    assert events[0]["type"] == "capability_run"
    assert events[0]["status"] == "succeeded"


def test_service_failure_status() -> None:
    def bad_invoker(input, config):
        raise RuntimeError("boom")

    svc = service_capability(
        RESEARCH_SPEC,
        ServiceEndpoint(assistant_id="research", version_label="1.0.0"),
        invoker=bad_invoker,
    )
    result = svc.invoke_with_status({"query": "x", "max_sources": 1})
    assert result.status is ServiceRunStatus.FAILED
    assert "boom" in (result.error_message or "")


def test_service_metadata_delivery() -> None:
    svc = research_service_capability_for_tests()
    meta = svc.to_metadata()
    assert meta["delivery"] == "service"
    assert meta["endpoint"]["assistant_id"] == RESEARCH_SERVICE_GRAPH_ID


def test_deploy_entrypoint_matches_contract() -> None:
    graph = build_research_service_graph()
    out = graph.invoke({"query": "deploy", "max_sources": 1})
    assert "svc findings" in out["findings"]
    assert RESEARCH_SERVICE_GRAPH_ID in LANGGRAPH_JSON_EXAMPLE["graphs"]


def test_service_capability_from_package_shares_spec() -> None:
    svc = service_capability_from_package(
        RESEARCH_CAPABILITY,
        ServiceEndpoint(assistant_id="research"),
        invoker=local_service_invoker(RESEARCH_CAPABILITY, prefix="p"),
    )
    assert svc.spec is RESEARCH_CAPABILITY.spec
    assert svc.invoke({"query": "a", "max_sources": 1})["sources"][0].startswith("p-")


def test_parent_composes_service_research_and_package_review() -> None:
    app = build_parent_with_remote_research(research_prefix="remote")
    out = app.invoke(
        {
            "query": "mix",
            "findings": "",
            "sources": [],
            "approved": False,
            "notes": "",
        }
    )
    assert out["sources"]
    assert "remote" in out["findings"]
    assert out["approved"] is True


def test_service_missing_transport_raises_on_invoke() -> None:
    svc = service_capability(
        RESEARCH_SPEC,
        ServiceEndpoint(),  # no url/assistant/invoker
    )
    with pytest.raises(Exception):
        svc.invoke({"query": "x", "max_sources": 1})
