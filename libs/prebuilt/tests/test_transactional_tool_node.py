import sys
import unittest
from pathlib import Path

# Add local prebuilt and langgraph packages to path
libs_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(libs_root / "prebuilt"))
sys.path.insert(0, str(libs_root / "langgraph"))

from langgraph.prebuilt.transactional_tool_node import (
    BlastRadiusExceededError,
    CompensatoryJournal,
    DestructiveActionBlockedError,
    TransactionalToolNode,
)


def dummy_tool(x: int) -> int:
    return x * 2


class TestTransactionalToolNode(unittest.TestCase):
    def setUp(self):
        self.node = TransactionalToolNode([dummy_tool], max_blast_radius=20.0)

    def test_journal_recording_and_lifo_rollback(self):
        journal = CompensatoryJournal()
        journal.record("provision_node", "node_1", {"type": "gpu"}, "deprovision_node", {"node_id": "node_1"})
        journal.record("attach_volume", "vol_1", {"size_gb": 100}, "detach_volume", {"vol_id": "vol_1"})

        rollback_seq = journal.generate_rollback_sequence()
        self.assertEqual(len(rollback_seq), 2)
        # Verify strict LIFO order
        self.assertEqual(rollback_seq[0]["tool_name"], "detach_volume")
        self.assertEqual(rollback_seq[0]["target_resource_id"], "vol_1")
        self.assertEqual(rollback_seq[1]["tool_name"], "deprovision_node")
        self.assertEqual(rollback_seq[1]["target_resource_id"], "node_1")

    def test_blast_radius_validation(self):
        # Within limit
        self.node.validate_action("cordon_node", {"node": "worker_1"}, simulated_blast_radius=10.0)

        # Exceeds limit
        with self.assertRaises(BlastRadiusExceededError):
            self.node.validate_action("mass_terminate", {}, simulated_blast_radius=25.0)

    def test_destructive_action_blocking(self):
        with self.assertRaises(DestructiveActionBlockedError):
            self.node.validate_action("drop_database", {})

        with self.assertRaises(DestructiveActionBlockedError):
            self.node.validate_action("safe_tool", {"destructive": True})


if __name__ == "__main__":
    unittest.main()
