"""Tests + labelled benchmark corpus for langgraph.graph.analysis.

Each corpus graph is labelled with exactly the rule names it should (for
adversarial graphs) or should not (for clean graphs) trigger. The
parametrized tests below turn that corpus into a real, reproducible
recall/false-positive measurement rather than an asserted claim:

    - test_adversarial_graph_detected: every injected defect must be
      caught (recall).
    - test_clean_graph_has_no_findings: no clean, realistic graph may
      produce a finding (false positives).
"""

from __future__ import annotations

from typing import Literal

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph.analysis import (
    Finding,
    Severity,
    analyze,
    detect_missing_checkpointer_with_cycle,
    detect_unbounded_cycles,
    detect_unreachable_nodes,
    detect_unresolved_branches,
    production_debt_score,
)
from langgraph.graph.state import StateGraph


class State(TypedDict):
    value: str


def _noop(_: State) -> dict:
    return {}


# ---------------------------------------------------------------------------
# Clean corpus: realistic graphs that must produce zero findings.
# ---------------------------------------------------------------------------


def clean_linear_chain() -> StateGraph:
    """START -> a -> b -> END, no cycles, no branches."""
    g = StateGraph(State)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g


def clean_fan_out_fan_in() -> StateGraph:
    """START -> {a, b} -> join -> END, using a waiting (fan-in) edge."""
    g = StateGraph(State)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_node("join", _noop)
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge(["a", "b"], "join")
    g.add_edge("join", END)
    return g


def clean_conditional_with_path_map() -> StateGraph:
    """A conditional edge with an explicit, fully-resolvable path_map."""

    def router(_: State) -> str:
        return "a"

    g = StateGraph(State)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_edge(START, "a")
    g.add_conditional_edges("a", router, {"a": "b", "b": END})
    g.add_edge("b", END)
    return g


def clean_conditional_with_literal_return() -> StateGraph:
    """A conditional edge resolvable via a Literal return-type annotation."""

    def router(_: State) -> Literal["b", "__end__"]:
        return "b"

    g = StateGraph(State)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_edge(START, "a")
    g.add_conditional_edges("a", router)
    g.add_edge("b", END)
    return g


def clean_bounded_retry_loop() -> StateGraph:
    """A cycle (retry loop) with a real exit edge to END, with a checkpointer."""

    def should_retry(_: State) -> str:
        return "retry"

    g = StateGraph(State)
    g.add_node("attempt", _noop)
    g.add_edge(START, "attempt")
    g.add_conditional_edges("attempt", should_retry, {"retry": "attempt", "done": END})
    return g


CLEAN_CORPUS = {
    "linear_chain": clean_linear_chain,
    "fan_out_fan_in": clean_fan_out_fan_in,
    "conditional_with_path_map": clean_conditional_with_path_map,
    "conditional_with_literal_return": clean_conditional_with_literal_return,
    "bounded_retry_loop": clean_bounded_retry_loop,
}


# ---------------------------------------------------------------------------
# Adversarial corpus: one deliberately injected defect each, labelled with
# the exact rule name that must fire.
# ---------------------------------------------------------------------------


def bad_orphan_node() -> StateGraph:
    """'orphan' is defined but never wired to anything."""
    g = StateGraph(State)
    g.add_node("a", _noop)
    g.add_node("orphan", _noop)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    return g


def bad_orphan_downstream_of_dead_node() -> StateGraph:
    """'b' is only reachable via 'a', which is itself unreachable."""
    g = StateGraph(State)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_node("live", _noop)
    g.add_edge("a", "b")  # dangling: 'a' has no incoming edge from START
    g.add_edge(START, "live")
    g.add_edge("live", END)
    return g


def bad_unbounded_self_loop() -> StateGraph:
    """A node that loops to itself with no exit."""

    def always_loop(_: State) -> str:
        return "loop"

    g = StateGraph(State)
    g.add_node("stuck", _noop)
    g.add_edge(START, "stuck")
    g.add_conditional_edges("stuck", always_loop, {"loop": "stuck"})
    return g


def bad_unbounded_two_node_cycle() -> StateGraph:
    """A two-node cycle (a <-> b) with no path out to END."""
    g = StateGraph(State)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    return g


def bad_cycle_without_checkpointer() -> StateGraph:
    """A properly-exiting retry loop, but intended to be compiled with checkpointer=None."""

    def should_retry(_: State) -> str:
        return "retry"

    g = StateGraph(State)
    g.add_node("attempt", _noop)
    g.add_edge(START, "attempt")
    g.add_conditional_edges("attempt", should_retry, {"retry": "attempt", "done": END})
    return g


def bad_unresolved_branch() -> StateGraph:
    """A router with no path_map and no inferable Literal return type."""

    def router(_: State):
        return "a"

    g = StateGraph(State)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_edge(START, "a")
    g.add_conditional_edges("a", router)
    g.add_edge("b", END)
    return g


ADVERSARIAL_CORPUS = {
    "orphan_node": (bad_orphan_node, "unreachable-node", None),
    "orphan_downstream_of_dead_node": (
        bad_orphan_downstream_of_dead_node,
        "unreachable-node",
        None,
    ),
    "unbounded_self_loop": (bad_unbounded_self_loop, "unbounded-cycle", None),
    "unbounded_two_node_cycle": (bad_unbounded_two_node_cycle, "unbounded-cycle", None),
    "cycle_without_checkpointer": (
        bad_cycle_without_checkpointer,
        "missing-checkpointer-with-cycle",
        None,  # checkpointer=None is the default; no override needed
    ),
    "unresolved_branch": (bad_unresolved_branch, "unresolvable-branch", None),
}


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CLEAN_CORPUS)
def test_clean_graph_has_no_findings(name: str) -> None:
    graph = CLEAN_CORPUS[name]()
    checkpointer = InMemorySaver() if name == "bounded_retry_loop" else None
    findings = analyze(graph, checkpointer=checkpointer)
    assert findings == [], f"{name} produced unexpected findings: {findings}"


@pytest.mark.parametrize("name", ADVERSARIAL_CORPUS)
def test_adversarial_graph_detected(name: str) -> None:
    factory, expected_rule, checkpointer = ADVERSARIAL_CORPUS[name]
    graph = factory()
    findings = analyze(graph, checkpointer=checkpointer)
    rules = {f.rule for f in findings}
    assert expected_rule in rules, (
        f"{name}: expected rule '{expected_rule}' not found in {rules}"
    )


def test_benchmark_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """Not an assertion -- prints the real recall/false-positive numbers.

    Run with `-s` to see the summary; the two tests above are what
    actually enforce recall == 100% and false positives == 0 per-case.
    """
    total_clean = len(CLEAN_CORPUS)
    fp_count = 0
    for name, factory in CLEAN_CORPUS.items():
        graph = factory()
        checkpointer = InMemorySaver() if name == "bounded_retry_loop" else None
        if analyze(graph, checkpointer=checkpointer):
            fp_count += 1

    total_adversarial = len(ADVERSARIAL_CORPUS)
    detected = 0
    for name, (factory, expected_rule, checkpointer) in ADVERSARIAL_CORPUS.items():
        graph = factory()
        rules = {f.rule for f in analyze(graph, checkpointer=checkpointer)}
        if expected_rule in rules:
            detected += 1

    print(
        f"\nbenchmark: recall={detected}/{total_adversarial} "
        f"false_positives={fp_count}/{total_clean}"
    )


# ---------------------------------------------------------------------------
# Unit-level tests for individual detectors and the score function
# ---------------------------------------------------------------------------


def test_production_debt_score_weights() -> None:
    findings = [
        Finding("r1", Severity.CRITICAL, None, "x"),
        Finding("r2", Severity.WARNING, None, "y"),
        Finding("r3", Severity.INFO, None, "z"),
    ]
    assert production_debt_score(findings) == 10 + 3 + 1
    assert production_debt_score([]) == 0


def test_detect_unreachable_nodes_finds_orphan() -> None:
    graph = bad_orphan_node()
    findings = detect_unreachable_nodes(graph)
    assert len(findings) == 1
    assert findings[0].node == "orphan"
    assert findings[0].severity == Severity.WARNING


def test_detect_unbounded_cycles_finds_self_loop() -> None:
    graph = bad_unbounded_self_loop()
    findings = detect_unbounded_cycles(graph)
    assert len(findings) == 1
    assert findings[0].node == "stuck"
    assert findings[0].severity == Severity.CRITICAL


def test_detect_unbounded_cycles_ignores_bounded_loop() -> None:
    graph = clean_bounded_retry_loop()
    assert detect_unbounded_cycles(graph) == []


def test_detect_missing_checkpointer_with_cycle() -> None:
    graph = bad_cycle_without_checkpointer()
    assert len(detect_missing_checkpointer_with_cycle(graph, None)) == 1
    assert detect_missing_checkpointer_with_cycle(graph, InMemorySaver()) == []


def test_detect_missing_checkpointer_with_cycle_false_also_flagged() -> None:
    # checkpointer=False ("explicitly no persistence, don't inherit") means
    # no persistence just as much as the None default does.
    graph = bad_cycle_without_checkpointer()
    assert len(detect_missing_checkpointer_with_cycle(graph, False)) == 1


def test_detect_missing_checkpointer_no_cycle_no_finding() -> None:
    graph = clean_linear_chain()
    assert detect_missing_checkpointer_with_cycle(graph, None) == []


def test_detect_unresolved_branches() -> None:
    graph = bad_unresolved_branch()
    findings = detect_unresolved_branches(graph)
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO


def test_detect_unresolved_branches_none_when_path_map_given() -> None:
    graph = clean_conditional_with_path_map()
    assert detect_unresolved_branches(graph) == []


def test_analyze_sorts_critical_first() -> None:
    graph = StateGraph(State)
    graph.add_node("stuck", _noop)
    graph.add_node("orphan", _noop)
    graph.add_edge(START, "stuck")
    graph.add_edge("stuck", "stuck")
    findings = analyze(graph)
    assert findings[0].severity == Severity.CRITICAL
