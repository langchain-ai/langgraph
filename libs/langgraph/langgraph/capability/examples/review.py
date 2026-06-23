"""Reference ``langgraph.review`` capability — second library module for composition demos."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.capability.contract import CapabilitySpec, SideEffect, StateBoundary
from langgraph.capability.package import GraphCapability, graph_capability
from langgraph.graph import END, START, StateGraph


class ReviewInput(TypedDict):
    content: str
    criteria: str


class ReviewOutput(TypedDict):
    content: str
    approved: bool
    notes: str


class _ReviewState(TypedDict):
    content: str
    criteria: str
    approved: bool
    notes: str


def build_review_graph() -> Any:
    """Package entrypoint for the review capability."""

    def evaluate(state: _ReviewState) -> dict[str, Any]:
        content = state["content"]
        criteria = state["criteria"]
        approved = len(content) > 0 and "fail" not in content.lower()
        notes = f"criteria={criteria!r}; approved={approved}"
        return {"approved": approved, "notes": notes}

    g = StateGraph(
        _ReviewState,
        input_schema=ReviewInput,
        output_schema=ReviewOutput,
    )
    g.add_node("evaluate", evaluate)
    g.add_edge(START, "evaluate")
    g.add_edge("evaluate", END)
    return g.compile()


REVIEW_SPEC: CapabilitySpec[ReviewInput, ReviewOutput, None] = CapabilitySpec(
    capability_id="langgraph.review",
    version="1.0.0",
    input_schema=ReviewInput,
    output_schema=ReviewOutput,
    side_effects=frozenset({SideEffect.NONE}),
    state_boundary=StateBoundary.ISOLATED,
    description="Reference review capability (approve/reject with notes).",
    owner="langgraph",
)

REVIEW_CAPABILITY: GraphCapability[ReviewInput, ReviewOutput, None] = graph_capability(
    REVIEW_SPEC, build_review_graph
)
