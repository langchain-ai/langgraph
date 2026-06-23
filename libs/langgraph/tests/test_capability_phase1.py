"""Phase 1: capability contracts + package delivery + local composition."""

from __future__ import annotations

import pytest

from langgraph.capability import (
    CapabilityContractError,
    CapabilitySchemaError,
    CapabilitySpec,
    CapabilityVersionError,
    SemVer,
    SideEffect,
    StateBoundary,
    attach_capability,
    graph_capability,
    select_capability_version,
)
from langgraph.capability.examples.parent_app import build_parent_graph
from langgraph.capability.examples.research import (
    RESEARCH_CAPABILITY,
    RESEARCH_SPEC,
    build_research_graph,
)
from langgraph.capability.examples.review import REVIEW_CAPABILITY, REVIEW_SPEC
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


def test_semver_parse_and_requests() -> None:
    v = SemVer.parse("1.2.3")
    assert v.major == 1 and v.minor == 2 and v.patch == 3
    assert v.compatible_with_request("1")
    assert v.compatible_with_request("1.2")
    assert v.compatible_with_request("1.2.3")
    assert not v.compatible_with_request("2")
    with pytest.raises(CapabilityVersionError):
        SemVer.parse("nope")


def test_shared_state_boundary_rejected() -> None:
    class In(TypedDict):
        x: int

    class Out(TypedDict):
        y: int

    with pytest.raises(CapabilityContractError, match="SHARED"):
        CapabilitySpec(
            capability_id="demo.bad",
            version="0.1.0",
            input_schema=In,
            output_schema=Out,
            state_boundary=StateBoundary.SHARED,
        )


def test_research_capability_invoke_boundary() -> None:
    result = RESEARCH_CAPABILITY.invoke({"query": "agents", "max_sources": 2})
    assert result["query"] == "agents"
    assert len(result["sources"]) == 2
    assert "findings" in result
    assert "scratch" not in result  # internal channel not in output schema path


def test_research_capability_missing_input_key() -> None:
    with pytest.raises(CapabilitySchemaError):
        RESEARCH_CAPABILITY.invoke({})  # type: ignore[arg-type]


def test_attach_capability_mapped_parent() -> None:
    class S(TypedDict):
        query: str
        findings: str

    g = StateGraph(S)
    attach_capability(
        g,
        "research",
        RESEARCH_CAPABILITY,
        public_params={"prefix": "t"},
        input_mapper=lambda s: {"query": s["query"], "max_sources": 1},
        output_mapper=lambda o: {"findings": o["findings"]},
    )
    g.add_edge(START, "research")
    g.add_edge("research", END)
    app = g.compile()
    out = app.invoke({"query": "x", "findings": ""})
    assert "t findings" in out["findings"]


def test_parent_app_composes_two_capabilities() -> None:
    app = build_parent_graph(research_prefix="demo")
    out = app.invoke(
        {
            "query": "compose",
            "findings": "",
            "sources": [],
            "approved": False,
            "notes": "",
        }
    )
    assert out["sources"]
    assert out["approved"] is True
    assert "criteria=" in out["notes"]


def test_select_capability_version() -> None:
    class In(TypedDict):
        q: str

    class Out(TypedDict):
        a: str

    specs = [
        CapabilitySpec("demo.x", "1.0.0", In, Out),
        CapabilitySpec("demo.x", "1.1.0", In, Out),
        CapabilitySpec("demo.x", "2.0.0", In, Out),
        CapabilitySpec("demo.y", "1.0.0", In, Out),
    ]
    picked = select_capability_version(specs, "demo.x", "1")
    assert picked.version == "1.1.0"
    with pytest.raises(CapabilityVersionError):
        select_capability_version(specs, "demo.x", "9")


def test_graph_capability_metadata() -> None:
    meta = RESEARCH_CAPABILITY.to_metadata()
    assert meta["capability_id"] == "langgraph.research"
    assert meta["delivery"] == "package"
    assert meta["state_boundary"] == "isolated"
    assert SideEffect.NONE.value in meta.get("side_effects", []) or meta[
        "side_effects"
    ] == []


def test_builder_entrypoint_direct() -> None:
    graph = build_research_graph(prefix="direct")
    out = graph.invoke({"query": "q", "max_sources": 1})
    assert out["sources"] == ["direct-source-1"]


def test_review_rejects_fail_content_via_parent_mapper() -> None:
    # Direct capability: content with "fail" is not approved
    out = REVIEW_CAPABILITY.invoke({"content": "fail please", "criteria": "c"})
    assert out["approved"] is False
