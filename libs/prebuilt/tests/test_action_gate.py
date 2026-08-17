from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys
from pathlib import Path

# Add action_gate module path
_module_dir = Path(__file__).resolve().parent.parent / "langgraph" / "prebuilt"
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

try:
    from action_gate import (
        ActionBoundary,
        ActionGate,
        ActionLedger,
        GateDecision,
    )
except ImportError:
    from langgraph.prebuilt.action_gate import (
        ActionBoundary,
        ActionGate,
        ActionLedger,
        GateDecision,
    )


class TestActionGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ActionGate(prove_token="secret-hitl-token", ledger=ActionLedger())

    def test_read_operation_allowed(self) -> None:
        decision = self.gate.evaluate(tool="read_file", arguments={"path": "README.md"})
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.mode, "allow")
        self.assertEqual(decision.tier, "read")
        self.assertFalse(decision.requires_hitl)

    def test_unattended_destructive_denied_even_with_high_confidence(self) -> None:
        decision = self.gate.evaluate(
            tool="shell.exec",
            arguments={"cmd": "rm -rf /"},
            model_confidence=0.99,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.mode, "deny")
        self.assertEqual(decision.tier, "destructive")
        self.assertTrue(decision.confidence_rejected)
        self.assertTrue(decision.never_equate_intent_to_approval)

    def test_destructive_allowed_with_valid_hitl_token(self) -> None:
        decision = self.gate.evaluate(
            tool="shell.exec",
            arguments={"cmd": "decommission_node"},
            approved=True,
            offered_token="secret-hitl-token",
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.mode, "allow")
        self.assertEqual(decision.reason, "hitl_token_verified")

    def test_destructive_denied_with_invalid_hitl_token(self) -> None:
        decision = self.gate.evaluate(
            tool="drop_database",
            approved=True,
            offered_token="wrong-token",
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.mode, "deny")

    def test_unapproved_write_defaults_to_simulation(self) -> None:
        decision = self.gate.evaluate(tool="update_record", arguments={"id": 123})
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.mode, "simulate")
        self.assertEqual(decision.reason, "unapproved_write_simulated")

    def test_action_boundary_wrapper(self) -> None:
        boundary = ActionBoundary(gate=self.gate)
        decision = boundary.guard_tool_call(
            tool_name="delete_account",
            arguments={"user_id": "u1"},
            model_confidence=0.95,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.mode, "deny")


class TestActionLedger(unittest.TestCase):
    def test_hash_chain_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_file = Path(tmp) / "ledger.jsonl"
            ledger = ActionLedger(ledger_file)

            ledger.record("agent-1", "fetch_data", "read", "allow", True, "ok")
            ledger.record("agent-1", "write_log", "write", "simulate", True, "simulated")
            ledger.record("agent-1", "rm_table", "destructive", "deny", False, "denied")

            self.assertEqual(len(ledger.entries), 3)
            self.assertTrue(ledger.verify_chain())

            # Verify persisted ledger
            reloaded = ActionLedger(ledger_file)
            self.assertEqual(len(reloaded.entries), 3)
            self.assertTrue(reloaded.verify_chain())


if __name__ == "__main__":
    unittest.main()
