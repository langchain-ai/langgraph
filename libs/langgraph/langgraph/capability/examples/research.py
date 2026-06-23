"""Reference ``langgraph.research`` capability — package delivery."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from langgraph.capability.contract import CapabilitySpec, SideEffect, StateBoundary
from langgraph.capability.package import GraphCapability, graph_capability
from langgraph.graph import END, START, StateGraph


class ResearchInput(TypedDict):
    """Public input contract (parents depend on this only)."""

    query: str
    max_sources: NotRequired[int]


class ResearchOutput(TypedDict):
    """Public output contract."""

    query: str
    findings: str
    sources: list[str]


class ResearchParams(TypedDict, total=False):
    """Caller-safe public params (no secrets / provider bindings)."""

    prefix: str
    max_sources_default: int


class _ResearchState(TypedDict):
    """Private internal state — not part of the capability contract."""

    query: str
    max_sources: int
    findings: str
    sources: list[str]
    scratch: str


def _normalize(state: _ResearchState) -> dict[str, Any]:
    max_sources = state.get("max_sources") or 3
    return {
        "query": state["query"],
        "max_sources": max_sources,
        "scratch": f"researching:{state['query']}",
    }


def _gather(state: _ResearchState, *, prefix: str) -> dict[str, Any]:
    n = state["max_sources"]
    sources = [f"{prefix}-source-{i}" for i in range(1, n + 1)]
    findings = f"{prefix} findings for {state['query']}: " + ", ".join(sources)
    return {"sources": sources, "findings": findings}


def build_research_graph(*, prefix: str = "ref", max_sources_default: int = 3) -> Any:
    """Package entrypoint: build an invokable research graph.

    Only accepts **public params**. Inject tools/models/secrets via a service
    adapter in service mode — not here.
    """

    def gather_node(state: _ResearchState) -> dict[str, Any]:
        return _gather(state, prefix=prefix)

    def ensure_max(state: ResearchInput) -> dict[str, Any]:
        return {
            "query": state["query"],
            "max_sources": state.get("max_sources", max_sources_default),
            "findings": "",
            "sources": [],
            "scratch": "",
        }

    g = StateGraph(
        _ResearchState,
        input_schema=ResearchInput,
        output_schema=ResearchOutput,
    )
    g.add_node("ensure", ensure_max)
    g.add_node("normalize", _normalize)
    g.add_node("gather", gather_node)
    g.add_edge(START, "ensure")
    g.add_edge("ensure", "normalize")
    g.add_edge("normalize", "gather")
    g.add_edge("gather", END)
    return g.compile()


RESEARCH_SPEC: CapabilitySpec[ResearchInput, ResearchOutput, ResearchParams] = (
    CapabilitySpec(
        capability_id="langgraph.research",
        version="1.0.0",
        input_schema=ResearchInput,
        output_schema=ResearchOutput,
        public_params_schema=ResearchParams,
        side_effects=frozenset({SideEffect.NONE}),
        state_boundary=StateBoundary.ISOLATED,
        description="Reference research capability with isolated internal state.",
        owner="langgraph",
    )
)

RESEARCH_CAPABILITY: GraphCapability[ResearchInput, ResearchOutput, ResearchParams] = (
    graph_capability(RESEARCH_SPEC, build_research_graph)
)
