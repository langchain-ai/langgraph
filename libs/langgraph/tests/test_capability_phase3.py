"""Phase 3: ergonomics — catalog, config refs, add_capability_node."""

from __future__ import annotations

import pytest

from langgraph.capability.catalog import CapabilityCatalog, default_example_catalog
from langgraph.capability.compose import add_capability_node
from langgraph.capability.config_ref import (
    CONFIG_REF_EXAMPLES,
    parse_capability_ref,
    resolve_capability_ref,
    resolve_python_ref,
)
from langgraph.capability.examples.parent_catalog import build_parent_via_catalog
from langgraph.capability.examples.research import RESEARCH_CAPABILITY
from langgraph.capability.package import GraphCapability
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


def test_default_catalog_lists_reference_capabilities() -> None:
    cat = default_example_catalog()
    summary = cat.to_summary()
    ids = {row["capability_id"] for row in summary}
    assert "langgraph.research" in ids
    assert "langgraph.review" in ids
    research = cat.resolve("langgraph.research", "1")
    assert research.package is not None
    assert research.service is not None


def test_catalog_get_package_and_service() -> None:
    cat = default_example_catalog()
    pkg = cat.get_package("langgraph.research", "1.0.0")
    assert isinstance(pkg, GraphCapability)
    svc = cat.get_service("langgraph.research", "1")
    out = svc.invoke({"query": "c", "max_sources": 1})
    assert out["sources"]


def test_parse_capability_refs() -> None:
    py = parse_capability_ref(CONFIG_REF_EXAMPLES["python_capability"])
    assert py.kind == "python"
    assert py.module == "langgraph.capability.examples.research"
    svc = parse_capability_ref(CONFIG_REF_EXAMPLES["service_remote"])
    assert svc.kind == "service"
    assert svc.capability_id == "langgraph.research"
    assert svc.query.get("assistant_id") == "research"
    cat = parse_capability_ref(CONFIG_REF_EXAMPLES["catalog_package"])
    assert cat.kind == "catalog"
    assert cat.delivery == "package"


def test_resolve_python_ref_to_capability() -> None:
    obj = resolve_python_ref(CONFIG_REF_EXAMPLES["python_capability"])
    assert obj is RESEARCH_CAPABILITY


def test_resolve_catalog_ref() -> None:
    catalog = default_example_catalog()
    pkg = resolve_capability_ref(
        "catalog:langgraph.review@1:package", catalog=catalog
    )
    assert pkg.spec.capability_id == "langgraph.review"


def test_add_capability_node_with_object() -> None:
    class S(TypedDict):
        query: str
        findings: str

    g = StateGraph(S)
    add_capability_node(
        g,
        "research",
        RESEARCH_CAPABILITY,
        public_params={"prefix": "n"},
        input_mapper=lambda s: {"query": s["query"], "max_sources": 1},
        output_mapper=lambda o: {"findings": o["findings"]},
    )
    g.add_edge(START, "research")
    g.add_edge("research", END)
    out = g.compile().invoke({"query": "z", "findings": ""})
    assert "n findings" in out["findings"]


def test_add_capability_node_with_catalog_id() -> None:
    catalog = default_example_catalog()

    class S(TypedDict):
        query: str
        findings: str

    g = StateGraph(S)
    add_capability_node(
        g,
        "research",
        "langgraph.research",
        catalog=catalog,
        mode="package",
        public_params={"prefix": "id"},
        input_mapper=lambda s: {"query": s["query"], "max_sources": 1},
        output_mapper=lambda o: {"findings": o["findings"]},
    )
    g.add_edge(START, "research")
    g.add_edge("research", END)
    out = g.compile().invoke({"query": "z", "findings": ""})
    assert "id findings" in out["findings"]


def test_parent_via_catalog_end_to_end() -> None:
    app = build_parent_via_catalog()
    out = app.invoke(
        {
            "query": "ergo",
            "findings": "",
            "sources": [],
            "approved": False,
            "notes": "",
        }
    )
    assert out["approved"] is True
    assert "cat" in out["findings"]


def test_catalog_register_requires_delivery() -> None:
    from langgraph.capability.catalog import CatalogEntry
    from langgraph.capability.examples.research import RESEARCH_SPEC
    from langgraph.capability.errors import CapabilityContractError

    with pytest.raises(CapabilityContractError):
        CatalogEntry(spec=RESEARCH_SPEC)
