"""Parent app composing package capabilities via attach_capability."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.capability.examples.research import RESEARCH_CAPABILITY
from langgraph.capability.examples.review import REVIEW_CAPABILITY
from langgraph.capability.package import attach_capability
from langgraph.graph import END, START, StateGraph


class ParentState(TypedDict):
    query: str
    findings: str
    sources: list[str]
    approved: bool
    notes: str


def build_parent_graph(*, research_prefix: str = "ref") -> Any:
    """Compose research + review capabilities without sharing internal channels."""

    g = StateGraph(ParentState)

    def seed(state: ParentState) -> dict[str, Any]:
        return {
            "findings": "",
            "sources": [],
            "approved": False,
            "notes": "",
        }

    g.add_node("seed", seed)
    attach_capability(
        g,
        "research",
        RESEARCH_CAPABILITY,
        public_params={"prefix": research_prefix},
        input_mapper=lambda s: {"query": s["query"], "max_sources": 3},
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
