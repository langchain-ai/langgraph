"""Static production-readiness analysis for :class:`StateGraph` structures.

`StateGraph.compile()` (see `StateGraph.validate`) only checks that
every edge and conditional-edge endpoint refers to a real node, START, or
END. It does not check that every declared node is actually reachable
from START, that a cycle has any exit path to END, that a graph relying
on cycles was compiled with a checkpointer for crash recovery, or that a
conditional edge's destinations are even statically knowable. Those are
exactly the structural anti-patterns detected here -- gaps that compile
cleanly and run fine in a demo, then surface as hung agents, silently
dead nodes, or unrecoverable crashes once a graph is under real traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from langgraph.constants import END, START

if TYPE_CHECKING:
    from langgraph.graph.state import StateGraph


class Severity(str, Enum):
    """Risk tier of a :class:`Finding`.

    CRITICAL: can hang or crash a run (e.g. a cycle with no exit).
    WARNING: does not crash, but degrades reliability or leaves dead
        code in the graph (e.g. an unreachable node).
    INFO: an auditability/observability gap, not a functional defect.
    """

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One detected anti-pattern instance."""

    rule: str
    severity: Severity
    node: str | None
    message: str


_SEVERITY_WEIGHT: dict[Severity, int] = {
    Severity.CRITICAL: 10,
    Severity.WARNING: 3,
    Severity.INFO: 1,
}


def production_debt_score(findings: list[Finding]) -> int:
    """Severity-weighted sum of findings (critical=10, warning=3, info=1).

    This is a deliberately simple, transparent, reproducible heuristic for
    trending a graph's structural health over time and gating CI on
    regressions -- it is not a standardized industry metric.
    """
    return sum(_SEVERITY_WEIGHT[f.severity] for f in findings)


def _branch_ends(spec_ends: Any) -> set[str]:
    """Normalize a `StateNodeSpec.ends` value (tuple or dict) to a set.

    Mirrors `StateGraph.validate`'s own handling: a tuple's elements are
    target node names; for a dict, LangGraph's own validator only ever
    updates its target set with the dict's *keys*, so we match that.
    """
    if isinstance(spec_ends, dict):
        return set(spec_ends)
    return set(spec_ends)


def _adjacency(graph: StateGraph) -> dict[str, set[str]]:
    """Build a directed adjacency map over `{START} | graph.nodes | {END}`.

    Reuses `graph._all_edges` -- the same edge-flattening (plain edges
    plus fan-in `waiting_edges`) that `StateGraph.validate` itself uses
    at compile time -- so this stays correct if that internal
    representation changes.
    """
    adj: dict[str, set[str]] = {name: set() for name in graph.nodes}
    adj[START] = set()
    adj[END] = set()

    for start, end in graph._all_edges:
        adj.setdefault(start, set()).add(end)

    for source, branches in graph.branches.items():
        adj.setdefault(source, set())
        for branch in branches.values():
            if branch.ends is None:
                # No statically-resolvable path_map: the routing function's
                # real destinations are only known at runtime. Conservatively
                # assume it can reach any node (including END) so this
                # never produces a false-positive unreachable-node or
                # unbounded-cycle finding.
                adj[source] |= set(graph.nodes) | {END}
            else:
                adj[source] |= set(branch.ends.values())

    for name, spec in graph.nodes.items():
        ends = spec.ends
        if ends:
            adj.setdefault(name, set())
            adj[name] |= _branch_ends(ends)

    return adj


def _bfs_reachable(adj: dict[str, set[str]], sources: set[str]) -> set[str]:
    seen = set(sources)
    queue = list(sources)
    while queue:
        node = queue.pop()
        for nxt in adj.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _tarjan_scc(adj: dict[str, set[str]]) -> list[set[str]]:
    """Tarjan's strongly-connected-components algorithm.

    Any SCC of size > 1 is a cycle. (Single-node self-loops are cycles too
    but form size-1 SCCs, so callers must check those separately.)
    """
    index_counter = [0]
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    result: list[set[str]] = []

    def strongconnect(node: str) -> None:
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True

        for successor in adj.get(node, ()):
            if successor not in index:
                strongconnect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif on_stack.get(successor):
                lowlink[node] = min(lowlink[node], index[successor])

        if lowlink[node] == index[node]:
            component: set[str] = set()
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.add(w)
                if w == node:
                    break
            result.append(component)

    for node in adj:
        if node not in index:
            strongconnect(node)

    return result


def _cycles(adj: dict[str, set[str]]) -> list[set[str]]:
    cycles = [scc for scc in _tarjan_scc(adj) if len(scc) > 1]
    cycles += [{node} for node, targets in adj.items() if node in targets]
    return cycles


def detect_unreachable_nodes(graph: StateGraph) -> list[Finding]:
    """Nodes never reachable via any traversal starting at START.

    `compile()` never checks this: it only verifies edge endpoints
    exist, not that every declared node has an incoming path. A node
    added with `add_node` and then never wired to anything compiles and
    runs fine -- it is simply dead code that will never execute, silently
    diverging from what a reader of the graph definition would assume.
    """
    adj = _adjacency(graph)
    reachable = _bfs_reachable(adj, {START})
    findings = []
    for name in graph.nodes:
        if name not in reachable:
            findings.append(
                Finding(
                    rule="unreachable-node",
                    severity=Severity.WARNING,
                    node=name,
                    message=(
                        f"node '{name}' is never reachable from START via any "
                        "edge or conditional edge -- it is dead code that will "
                        "never execute"
                    ),
                )
            )
    return findings


def detect_unbounded_cycles(graph: StateGraph) -> list[Finding]:
    """Cycles (including self-loops) with no path from any member to END.

    `compile()` performs no cycle analysis at all. A cycle with no exit
    means any run that enters it can only be halted externally (a
    recursion-limit error or a manual kill) -- a hung-agent production
    incident waiting to happen, not a bug that will show up in a quick
    demo run.
    """
    adj = _adjacency(graph)
    findings = []
    for cycle in _cycles(adj):
        reachable_from_cycle = _bfs_reachable(adj, set(cycle))
        if END not in reachable_from_cycle:
            member = sorted(cycle)[0]
            findings.append(
                Finding(
                    rule="unbounded-cycle",
                    severity=Severity.CRITICAL,
                    node=member,
                    message=(
                        f"cycle containing {sorted(cycle)} has no path to END from "
                        "any node in it -- a run that enters this cycle can only "
                        "be stopped by a recursion-limit error, not by the graph "
                        "itself"
                    ),
                )
            )
    return findings


def detect_unresolved_branches(graph: StateGraph) -> list[Finding]:
    """Conditional edges whose destinations are not statically knowable.

    Occurs when `add_conditional_edges` is called without a
    `path_map`/list and the routing function has no `Literal` return
    type annotation LangGraph can infer from. Functionally fine at
    runtime, but it means no static tool -- this one included -- and no
    reviewer reading the graph definition can enumerate where that branch
    actually goes.
    """
    findings = []
    for source, branches in graph.branches.items():
        for name, branch in branches.items():
            if branch.ends is None:
                findings.append(
                    Finding(
                        rule="unresolvable-branch",
                        severity=Severity.INFO,
                        node=source,
                        message=(
                            f"conditional edge '{name}' from node '{source}' has "
                            "no statically-resolvable path_map -- pass path_map= "
                            "to add_conditional_edges, or annotate the routing "
                            "function's return type with Literal[...], so its "
                            "destinations can be audited"
                        ),
                    )
                )
    return findings


def detect_missing_checkpointer_with_cycle(
    graph: StateGraph, checkpointer: object | None
) -> list[Finding]:
    """A graph with a cycle compiled without a checkpointer.

    Cycles are how LangGraph expresses iterative agent loops (retries,
    ReAct-style tool loops). Without a checkpointer, a crash mid-loop
    loses all accumulated state and there is no way to inspect or resume
    the run -- a reliability gap invisible in local testing (which rarely
    crashes mid-loop) but a real incident in production.

    `checkpointer=None` (may inherit a parent's checkpointer when used as
    a subgraph) and `checkpointer=False` (explicitly no persistence, do
    not inherit) both mean "no persistence" for a top-level compiled
    graph, so both are treated as missing.
    """
    if checkpointer not in (None, False):
        return []
    adj = _adjacency(graph)
    if not _cycles(adj):
        return []
    return [
        Finding(
            rule="missing-checkpointer-with-cycle",
            severity=Severity.WARNING,
            node=None,
            message=(
                "graph contains a cycle but was compiled with no checkpointer "
                "-- a crash mid-loop loses all state and cannot be resumed"
            ),
        )
    ]


def analyze(graph: StateGraph, checkpointer: object | None = None) -> list[Finding]:
    """Run all detectors and return every finding, most-critical first."""
    findings: list[Finding] = []
    findings += detect_unbounded_cycles(graph)
    findings += detect_unreachable_nodes(graph)
    findings += detect_missing_checkpointer_with_cycle(graph, checkpointer)
    findings += detect_unresolved_branches(graph)
    order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    findings.sort(key=lambda f: (order[f.severity], f.rule, f.node or ""))
    return findings
