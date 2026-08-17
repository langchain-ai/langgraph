"""Deterministic Gate/Prove runtime boundary & hash-chained action ledger for LangGraph agent nodes.

Enforces zero-trust execution invariants:
1. never_equate_intent_to_approval: true (high model confidence is rejected as authorization for destructive tools).
2. Unattended destructive/provision/decommission tools are hard DENIED without an authorized HITL prove token.
3. Unapproved state-mutating actions default to SIMULATION mode.
4. Cryptographic SHA-256 hash-chained JSONL action ledger for ISO 42001, NIST AI RMF, and SOC 2 compliance.
5. Atomic kill-switch mechanism to pause agent tool dispatch instantly.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

GateMode = Literal["allow", "simulate", "deny"]
ToolTier = Literal["read", "write", "destructive", "provision", "decommission"]

HIGH_TIERS: set[ToolTier] = {"destructive", "provision", "decommission"}

DEFAULT_TIER_RULES: dict[str, ToolTier] = {
    "delete": "destructive",
    "remove": "destructive",
    "drop": "destructive",
    "terminate": "destructive",
    "destroy": "destructive",
    "shell.exec": "destructive",
    "bash": "destructive",
    "terminal": "destructive",
    "write": "write",
    "update": "write",
    "create": "write",
    "insert": "write",
    "patch": "write",
    "read": "read",
    "get": "read",
    "list": "read",
    "fetch": "read",
    "search": "read",
}

INSTANT_AUDIT_CTA = "https://a2zsoc.com/productized-services#instant-audit-tripwire"
CONSULTATION_CTA = "https://a2zsoc.com/consultation"


@dataclass(frozen=True)
class GateDecision:
    """Evaluation result for a proposed agent tool invocation."""

    tool: str
    tier: ToolTier
    mode: GateMode
    allowed: bool
    requires_hitl: bool
    never_equate_intent_to_approval: bool
    reason: str
    ledger_id: str
    receipt_hash: str
    kill_switch: bool = False
    confidence_rejected: bool = False
    replay_detected: bool = False
    actor_id: str = "agent-node"
    compliance_crosswalk: list[str] = field(
        default_factory=lambda: ["ISO_42001_A.6.2", "NIST_AI_RMF_GOVERN_1.2", "SOC2_CC6.8"]
    )
    instant_audit: str = INSTANT_AUDIT_CTA
    consultation: str = CONSULTATION_CTA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionLedger:
    """Append-only, SHA-256 hash-chained ledger for agent tool actions."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.entries: list[dict[str, Any]] = []
        self._last_hash = "0" * 64
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        self.entries = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self.entries.append(data)
                    self._last_hash = data.get("receipt_hash", self._last_hash)

    def record(
        self,
        actor_id: str,
        tool: str,
        tier: ToolTier,
        mode: GateMode,
        allowed: bool,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        ledger_id = str(uuid.uuid4())
        ts = time.time()
        payload = {
            "ledger_id": ledger_id,
            "timestamp": ts,
            "actor_id": actor_id,
            "tool": tool,
            "tier": tier,
            "mode": mode,
            "allowed": allowed,
            "reason": reason,
            "prev_hash": self._last_hash,
            "metadata": metadata or {},
        }
        raw = json.dumps(payload, sort_keys=True)
        receipt_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        payload["receipt_hash"] = receipt_hash
        self._last_hash = receipt_hash
        self.entries.append(payload)

        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")

        return ledger_id, receipt_hash

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for entry in self.entries:
            expected_prev = entry.get("prev_hash")
            if expected_prev != prev:
                return False
            payload = {k: v for k, v in entry.items() if k != "receipt_hash"}
            raw = json.dumps(payload, sort_keys=True)
            calc_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            if calc_hash != entry.get("receipt_hash"):
                return False
            prev = entry["receipt_hash"]
        return True


class ActionGate:
    """Deterministic Gate/Prove policy firewall for agent tool executions."""

    def __init__(
        self,
        prove_token: str = "",
        ledger: ActionLedger | None = None,
        custom_tier_rules: dict[str, ToolTier] | None = None,
    ) -> None:
        self.prove_token = prove_token or os.environ.get("AAG_PROVE_TOKEN", "")
        self.ledger = ledger if ledger is not None else ActionLedger()
        self.tier_rules = dict(DEFAULT_TIER_RULES)
        if custom_tier_rules:
            self.tier_rules.update(custom_tier_rules)

    def resolve_tier(self, tool_name: str, explicit_tier: str = "") -> ToolTier:
        if explicit_tier in ("read", "write", "destructive", "provision", "decommission"):
            return explicit_tier  # type: ignore[return-value]
        tool_lower = tool_name.lower()
        for pattern, tier in self.tier_rules.items():
            if pattern in tool_lower:
                return tier
        return "write"

    def is_kill_switch_engaged(self) -> bool:
        flag = os.environ.get("AAG_KILL_SWITCH", "").strip().lower()
        if flag in {"1", "true", "yes", "on"}:
            return True
        kill_file = Path(os.environ.get("AAG_KILL_SWITCH_FILE", "artifacts/KILL"))
        return kill_file.exists()

    def evaluate(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        tier: str = "",
        model_confidence: float | None = None,
        approved: bool = False,
        offered_token: str = "",
        simulate: bool = False,
        actor_id: str = "langgraph-agent",
    ) -> GateDecision:
        resolved_tier = self.resolve_tier(tool, tier)

        # 1. Kill Switch Check
        if self.is_kill_switch_engaged():
            lid, rhash = self.ledger.record(actor_id, tool, resolved_tier, "deny", False, "kill_switch_engaged")
            return GateDecision(
                tool=tool,
                tier=resolved_tier,
                mode="deny",
                allowed=False,
                requires_hitl=True,
                never_equate_intent_to_approval=True,
                reason="kill_switch_engaged",
                ledger_id=lid,
                receipt_hash=rhash,
                kill_switch=True,
                actor_id=actor_id,
            )

        # 2. Simulation Mode Check
        if simulate:
            lid, rhash = self.ledger.record(actor_id, tool, resolved_tier, "simulate", True, "simulation_mode_requested")
            return GateDecision(
                tool=tool,
                tier=resolved_tier,
                mode="simulate",
                allowed=True,
                requires_hitl=False,
                never_equate_intent_to_approval=True,
                reason="simulation_mode_requested",
                ledger_id=lid,
                receipt_hash=rhash,
                actor_id=actor_id,
            )

        # 3. Read Operations (Safe)
        if resolved_tier == "read":
            lid, rhash = self.ledger.record(actor_id, tool, resolved_tier, "allow", True, "read_operation_allowed")
            return GateDecision(
                tool=tool,
                tier=resolved_tier,
                mode="allow",
                allowed=True,
                requires_hitl=False,
                never_equate_intent_to_approval=True,
                reason="read_operation_allowed",
                ledger_id=lid,
                receipt_hash=rhash,
                actor_id=actor_id,
            )

        # 4. Destructive / High-Tier Operations (Strict Zero-Trust)
        if resolved_tier in HIGH_TIERS:
            token_valid = bool(
                self.prove_token
                and offered_token
                and hmac.compare_digest(self.prove_token.strip(), offered_token.strip())
            )
            if approved and token_valid:
                lid, rhash = self.ledger.record(actor_id, tool, resolved_tier, "allow", True, "hitl_token_verified")
                return GateDecision(
                    tool=tool,
                    tier=resolved_tier,
                    mode="allow",
                    allowed=True,
                    requires_hitl=True,
                    never_equate_intent_to_approval=True,
                    reason="hitl_token_verified",
                    ledger_id=lid,
                    receipt_hash=rhash,
                    actor_id=actor_id,
                )

            # Unattended destructive is denied even with 0.99 confidence
            lid, rhash = self.ledger.record(
                actor_id, tool, resolved_tier, "deny", False, "unattended_high_tier_denied"
            )
            return GateDecision(
                tool=tool,
                tier=resolved_tier,
                mode="deny",
                allowed=False,
                requires_hitl=True,
                never_equate_intent_to_approval=True,
                reason="unattended_high_tier_denied",
                ledger_id=lid,
                receipt_hash=rhash,
                confidence_rejected=bool(model_confidence and model_confidence > 0.5),
                actor_id=actor_id,
            )

        # 5. Standard Writes (Default to simulation if unapproved)
        if approved:
            lid, rhash = self.ledger.record(actor_id, tool, resolved_tier, "allow", True, "approved_write_allowed")
            return GateDecision(
                tool=tool,
                tier=resolved_tier,
                mode="allow",
                allowed=True,
                requires_hitl=False,
                never_equate_intent_to_approval=True,
                reason="approved_write_allowed",
                ledger_id=lid,
                receipt_hash=rhash,
                actor_id=actor_id,
            )

        # Unapproved write falls back to simulation mode
        lid, rhash = self.ledger.record(
            actor_id, tool, resolved_tier, "simulate", True, "unapproved_write_simulated"
        )
        return GateDecision(
            tool=tool,
            tier=resolved_tier,
            mode="simulate",
            allowed=True,
            requires_hitl=False,
            never_equate_intent_to_approval=True,
            reason="unapproved_write_simulated",
            ledger_id=lid,
            receipt_hash=rhash,
            actor_id=actor_id,
        )


class ActionBoundary:
    """Middleware decorator & wrapper for LangGraph agent nodes."""

    def __init__(
        self,
        gate: ActionGate | None = None,
        prove_token: str = "",
        ledger_path: str = "artifacts/action_ledger.jsonl",
    ) -> None:
        self.gate = gate or ActionGate(
            prove_token=prove_token,
            ledger=ActionLedger(Path(ledger_path) if ledger_path else None),
        )

    def guard_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        actor_id: str = "langgraph-agent",
        approved: bool = False,
        prove_token: str = "",
        simulate: bool = False,
        model_confidence: float | None = None,
    ) -> GateDecision:
        return self.gate.evaluate(
            tool=tool_name,
            arguments=arguments,
            approved=approved,
            offered_token=prove_token,
            simulate=simulate,
            actor_id=actor_id,
            model_confidence=model_confidence,
        )
