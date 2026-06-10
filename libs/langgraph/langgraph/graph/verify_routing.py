"""verify_routing — design-time sanity check for conditional edge logic.

LLMs write routing functions; LLMs also hallucinate on boolean satisfiability
at ~20% error rates (see https://github.com/Shrivastava-Aditya/boolean-algebra-engine).
This utility lets you express routing conditions as boolean expressions and
verifies them deterministically before ``graph.compile()``.

Checks performed
----------------
* **contradiction** — a route whose condition can never be true (dead route)
* **overlap** — two routes whose conditions can both hold simultaneously
  (ambiguous routing; the first-match wins but the intent is likely wrong)
* **gap** — a combination of variable values not covered by any route
  (the graph silently reaches an unhandled state)

Optional dependency
-------------------
Requires ``boolean-algebra-engine``::

    pip install boolean-algebra-engine

If not installed, :func:`verify_routing` raises ``ImportError`` with an
install hint. Nothing else in LangGraph depends on this package.

Quick start
-----------
.. code-block:: python

    from langgraph.graph.verify_routing import verify_routing

    issues = verify_routing({
        "tools":        "TOOL_CALL . !HUMAN",
        "human_review": "TOOL_CALL . HUMAN",
        "end":          "!TOOL_CALL",
    })
    # issues == [] — routing is sound

    for issue in issues:
        print(f"[{issue.kind}] {issue.message}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["RoutingIssue", "verify_routing"]


@dataclass
class RoutingIssue:
    """A single problem found in a routing condition set."""

    kind: str
    """``"contradiction"``, ``"overlap"``, or ``"gap"``."""

    routes: list[str]
    """Route names involved in this issue."""

    message: str
    """Human-readable description suitable for logging or raising."""

    details: dict = field(default_factory=dict)
    """Extra information (e.g. satisfying variable assignments)."""


def verify_routing(
    conditions: dict[str, str],
    *,
    raise_on_issues: bool = False,
) -> list[RoutingIssue]:
    """Verify that a set of routing conditions are logically sound.

    Useful as a design-time check before ``graph.compile()``. Express your
    routing function's branch conditions as boolean expressions; this utility
    verifies them deterministically using a SAT-based engine — no LLM
    inference, no sampling.

    Args:
        conditions: Mapping of route name → boolean expression string.
            Operators: AND (``"."``), OR (``"+"``), NOT (``"!"``).
            Variables must start with a letter and contain only letters,
            digits, and underscores. Example::

                {
                    "tools":        "TOOL_CALL . !HUMAN",
                    "human_review": "TOOL_CALL . HUMAN",
                    "end":          "!TOOL_CALL",
                }

        raise_on_issues: If ``True``, raise :exc:`ValueError` listing all
            issues when any are found. Defaults to ``False`` (return list).

    Returns:
        List of :class:`RoutingIssue` objects. An empty list means the
        routing conditions are sound — no contradictions, overlaps, or gaps.

    Raises:
        ImportError: If ``boolean-algebra-engine`` is not installed.
        ValueError: If ``raise_on_issues=True`` and issues are found.

    Examples:
        Sound routing — no issues::

            issues = verify_routing({
                "tools":        "TOOL_CALL . !HUMAN",
                "human_review": "TOOL_CALL . HUMAN",
                "end":          "!TOOL_CALL",
            })
            assert issues == []

        Catches a dead route (contradiction)::

            issues = verify_routing({"never": "A . !A", "always": "A + !A"})
            assert issues[0].kind == "contradiction"
            assert "never" in issues[0].routes

        Catches ambiguous routing (overlap)::

            issues = verify_routing({"a": "X", "b": "X . Y"})
            assert any(i.kind == "overlap" for i in issues)

        Catches a routing gap::

            issues = verify_routing({
                "tools": "TOOL_CALL",
                "done":  "END_FLAG",
            })
            assert any(i.kind == "gap" for i in issues)
    """
    try:
        from boolean_algebra_engine import evaluate
    except ImportError as exc:
        raise ImportError(
            "verify_routing requires boolean-algebra-engine. "
            "Install it with:  pip install boolean-algebra-engine\n"
            "Docs: https://github.com/Shrivastava-Aditya/boolean-algebra-engine"
        ) from exc

    if not conditions:
        return []

    issues: list[RoutingIssue] = []
    route_names = list(conditions.keys())
    exprs = list(conditions.values())

    # 1. Contradictions — each condition evaluated in isolation
    for name, expr in conditions.items():
        table, _ = evaluate(expr)
        if not table.satisfiable:
            issues.append(
                RoutingIssue(
                    kind="contradiction",
                    routes=[name],
                    message=(
                        f"Route '{name}' is a contradiction: "
                        f"'{expr}' can never be true. "
                        "This route is unreachable dead code."
                    ),
                )
            )

    # 2. Overlaps — pairwise AND across all route combinations
    for i in range(len(route_names)):
        for j in range(i + 1, len(route_names)):
            overlap_expr = f"({exprs[i]}).({exprs[j]})"
            table, _ = evaluate(overlap_expr)
            if table.satisfiable:
                issues.append(
                    RoutingIssue(
                        kind="overlap",
                        routes=[route_names[i], route_names[j]],
                        message=(
                            f"Routes '{route_names[i]}' and '{route_names[j]}' overlap: "
                            f"both conditions can be true simultaneously "
                            f"({len(table.minterms)} case(s)). "
                            "Routing will be ambiguous in those states."
                        ),
                        details={"satisfying_cases": table.minterms},
                    )
                )

    # 3. Gaps — NOT(union of all conditions) satisfiable → uncovered states
    union_expr = "+".join(f"({e})" for e in exprs)
    gap_expr = f"!({union_expr})"
    table, _ = evaluate(gap_expr)
    if table.satisfiable:
        issues.append(
            RoutingIssue(
                kind="gap",
                routes=route_names,
                message=(
                    f"Routing has {len(table.minterms)} uncovered state(s): "
                    "some variable combinations are not handled by any route. "
                    "The graph will reach an unhandled state in those cases."
                ),
                details={"uncovered_cases": table.minterms},
            )
        )

    if raise_on_issues and issues:
        lines = "\n".join(f"  [{i.kind}] {i.message}" for i in issues)
        raise ValueError(f"Routing verification failed:\n{lines}")

    return issues
