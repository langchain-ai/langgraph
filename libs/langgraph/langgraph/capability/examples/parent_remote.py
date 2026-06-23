"""Parent app composing a **service** capability (remote/black-box step)."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.capability.examples.review import REVIEW_CAPABILITY
from langgraph.capability.examples.service_deploy import research_service_capability_for_tests
from langgraph.capability.package import attach_capability
from langgraph.capability.service import attach_service_capability
from langgraph.graph import END, START, StateGraph


class ParentState(TypedDict):
    query: str
    findings: str
    sources: list[str]
    approved: bool
    notes: str


def build_parent_with_remote_research(*, research_prefix: str = "svc") -> Any:
    """Orchestrate: local seed → remote research service → local review package."""

    research_svc = research_service_capability_for_tests(prefix=research_prefix)
    g = StateGraph(ParentState)

    def seed(state: ParentState) -> dict[str, Any]:
        return {
            "findings": "",
            "sources": [],
            "approved": False,
            "notes": "",
        }

    g.add_node("seed", seed)
    attach_service_capability(
        g,
        "research",
        research_svc,
        input_mapper=lambda s: {"query": s["query"], "max_sources": 2},
        output_mapper=lambda o: {
            "findings": o["findings"],
            "sources": o["sources"],
        },
    )
    attach_capability(
        g,
        "review",
        REVIEW_CAPABILITY,
        input_mapper=lambda s: {
            "content": s.get("findings", ""),
            "criteria": "non-empty findings",
        },
        output_mapper=lambda o: {
            "approved": o["approved"],
            "notes": o["notes"],
        },
    )
    g.add_edge(START, "seed")
    g.add_edge("seed", "research")
    g.add_edge("research", "review")
    g.add_edge("review", END)
    return g.compile()
