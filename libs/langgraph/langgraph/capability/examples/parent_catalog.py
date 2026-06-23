"""Parent app using catalog + add_capability_node ergonomics."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.capability.catalog import default_example_catalog
from langgraph.capability.compose import add_capability_node
from langgraph.graph import END, START, StateGraph


class ParentState(TypedDict):
    query: str
    findings: str
    sources: list[str]
    approved: bool
    notes: str


def build_parent_via_catalog() -> Any:
    """Resolve capabilities by id from the example catalog."""
    catalog = default_example_catalog()
    g = StateGraph(ParentState)

    def seed(state: ParentState) -> dict[str, Any]:
        return {
            "findings": "",
            "sources": [],
            "approved": False,
            "notes": "",
        }

    g.add_node("seed", seed)
    add_capability_node(
        g,
        "research",
        "langgraph.research",
        mode="package",
        catalog=catalog,
        public_params={"prefix": "cat"},
        input_mapper=lambda s: {"query": s["query"], "max_sources": 2},
        output_mapper=lambda o: {
            "findings": o["findings"],
            "sources": o["sources"],
        },
    )
    add_capability_node(
        g,
        "review",
        "catalog:langgraph.review@1:package",
        catalog=catalog,
        input_mapper=lambda s: {
            "content": s.get("findings", ""),
            "criteria": "non-empty",
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
