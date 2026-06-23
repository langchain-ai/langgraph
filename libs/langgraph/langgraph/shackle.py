"""
SHACKLE Guard for LangGraph
============================
Pre-execution circuit breaker as a LangGraph node.

Install: pip install shackle-guard
Usage:
    from langgraph.shackle import ShackleGuard
    guard = ShackleGuard(budget=0.25, max_repeat_calls=3)
    graph.add_node("guard", guard)
    graph.add_edge("agent", "guard")
    graph.add_conditional_edges("guard", guard.route, {
        "allow": "tools",
        "deny": "end",
        "hitl": "human_approval",
    })
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Literal

VerictType = Literal["allow", "deny", "hitl"]


class ShackleGuard:
    """Pre-execution guard node for LangGraph agents.

    Monitors tool calls, enforces budgets, detects loops,
    and routes to HITL when human approval is needed.

    The guard operates on LangGraph state messages — it inspects
    the last AI message for tool_calls and decides whether to allow,
    deny, or escalate each call.
    """

    def __init__(
        self,
        budget: float = 0.25,
        max_repeat_calls: int = 3,
        error_amplification: bool = True,
        timeout_seconds: int = 300,
        hitl_tools: list[str] | None = None,
    ) -> None:
        self.budget = budget
        self.max_repeat_calls = max_repeat_calls
        self.error_amplification = error_amplification
        self.timeout_seconds = timeout_seconds
        self.hitl_tools = hitl_tools or [
            "execute_code", "write_file", "delete_file",
            "run_shell", "deploy",
        ]

        self._budget_spent: float = 0.0
        self._total_calls: int = 0
        self._repeat_counts: dict[str, int] = {}
        self._last_tool_name: str = ""
        self._last_input_hash: str = ""
        self._circuit_tripped: bool = False
        self._circuit_reason: str = ""
        self._start_time: float = time.time()

        self._error_signals = (
            "401", "unauthorized", "403", "forbidden", "500",
            "timeout", "rate limit", "quota exceeded", "token expired",
        )

    def _hash_args(self, args: dict) -> str:
        return hashlib.sha256(
            json.dumps(args, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _cost(self, tool_name: str) -> float:
        return {
            "web_search": 0.001, "read_file": 0.0001,
            "write_file": 0.0005, "execute_code": 0.005,
            "query_db": 0.002, "call_api": 0.003,
            "create_agent": 0.01,
        }.get(tool_name, 0.001)

    def _error_in_args(self, args: dict) -> bool:
        arg_str = str(args).lower()
        return any(s in arg_str for s in self._error_signals)

    def __call__(self, state: dict) -> dict:
        """LangGraph node: inspects state, returns guarded state."""
        messages = state.get("messages", [])
        if not messages:
            return {**state, "shackle_verdict": "allow"}

        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", None) or []
        if not tool_calls:
            return {**state, "shackle_verdict": "allow"}

        # Evaluate each tool call
        verdicts = []
        for tc in tool_calls:
            v = self._evaluate(
                tc.get("name", "unknown"),
                tc.get("args", {}),
            )
            verdicts.append(v)

        # Worst-case verdict wins
        if any(v == "deny" for v in verdicts):
            final = "deny"
        elif any(v == "hitl" for v in verdicts):
            final = "hitl"
        else:
            final = "allow"

        return {**state, "shackle_verdict": final, "shackle_details": verdicts}

    def _evaluate(self, tool_name: str, args: dict) -> VerictType:
        if self._circuit_tripped:
            return "deny"

        elapsed = time.time() - self._start_time
        if elapsed > self.timeout_seconds:
            self._circuit_tripped = True
            self._circuit_reason = f"timeout ({self.timeout_seconds}s)"
            return "deny"

        remaining = self.budget - self._budget_spent
        if remaining <= 0:
            self._circuit_tripped = True
            self._circuit_reason = "budget_exhausted"
            return "deny"

        call_hash = self._hash_args(args)
        is_repeat = (tool_name == self._last_tool_name and
                     call_hash == self._last_input_hash)

        if is_repeat:
            self._repeat_counts[tool_name] = self._repeat_counts.get(tool_name, 0) + 1
            limit = self.max_repeat_calls
            if self.error_amplification and self._error_in_args(args):
                limit = max(1, limit - 1)
            if self._repeat_counts[tool_name] >= limit:
                return "deny"
        else:
            self._repeat_counts[tool_name] = 1

        if tool_name in self.hitl_tools:
            return "hitl"

        self._budget_spent += self._cost(tool_name)
        self._total_calls += 1
        self._last_tool_name = tool_name
        self._last_input_hash = call_hash
        return "allow"

    def route(self, state: dict) -> VerictType:
        """Conditional edge router for LangGraph."""
        return state.get("shackle_verdict", "allow")
