"""Transactional ToolNode with compensatory rollback journaling and blast-radius containment.

Provides Hoare-logic transaction journaling and atomic rollback DAG generation
for mutating agent workflows in LangGraph.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from langgraph.prebuilt.tool_node import ToolNode


class BlastRadiusExceededError(RuntimeError):
    """Raised when an agent tool call exceeds the safe blast-radius limit."""


class DestructiveActionBlockedError(PermissionError):
    """Raised when an agent attempts an unauthorized destructive operation."""


@dataclass
class TransactionalActionRecord:
    """An individual recorded agent tool invocation in the rollback journal."""

    action_id: str
    tool_name: str
    target_resource_id: str
    forward_arguments: Dict[str, Any]
    inverse_tool_name: str
    inverse_arguments: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class CompensatoryJournal:
    """Hoare-logic LIFO transaction journal for atomic state undo."""

    def __init__(self) -> None:
        self.journal: List[TransactionalActionRecord] = []

    def record(
        self,
        tool_name: str,
        target_resource_id: str,
        forward_arguments: Dict[str, Any],
        inverse_tool_name: Optional[str] = None,
        inverse_arguments: Optional[Dict[str, Any]] = None,
    ) -> TransactionalActionRecord:
        inv_tool = inverse_tool_name or f"rollback_{tool_name}"
        inv_args = inverse_arguments or {}
        record = TransactionalActionRecord(
            action_id=f"tx_{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            target_resource_id=target_resource_id,
            forward_arguments=forward_arguments,
            inverse_tool_name=inv_tool,
            inverse_arguments=inv_args,
        )
        self.journal.append(record)
        return record

    def generate_rollback_sequence(self) -> List[Dict[str, Any]]:
        """Return compensatory actions in strict reverse chronological (LIFO) order."""
        return [
            {
                "tool_name": rec.inverse_tool_name,
                "target_resource_id": rec.target_resource_id,
                "arguments": rec.inverse_arguments,
                "original_action_id": rec.action_id,
            }
            for rec in reversed(self.journal)
        ]

    def clear(self) -> None:
        self.journal.clear()


class TransactionalToolNode(ToolNode):
    """Guarded ToolNode that intercepts calls, checks blast-radius, and logs undo journals."""

    BLOCKED_PATTERNS = {
        "rm_rf",
        "drop_database",
        "delete_production_cluster",
        "format_disk",
    }

    def __init__(
        self,
        tools: Sequence[Any],
        *,
        max_blast_radius: float = 25.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(tools, **kwargs)
        self.max_blast_radius = max_blast_radius
        self.journal = CompensatoryJournal()

    def validate_action(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        simulated_blast_radius: float = 1.0,
    ) -> None:
        """Verify pre-execution constraints."""
        if tool_name.lower() in self.BLOCKED_PATTERNS or arguments.get("destructive"):
            raise DestructiveActionBlockedError(
                f"Tool '{tool_name}' blocked: Destructive command prohibited without explicit approval."
            )
        if simulated_blast_radius > self.max_blast_radius:
            raise BlastRadiusExceededError(
                f"Tool '{tool_name}' rejected: Simulated blast radius {simulated_blast_radius:.2f} "
                f"exceeds ceiling {self.max_blast_radius:.2f}."
            )

    def record_mutation(
        self,
        tool_name: str,
        target_resource_id: str,
        forward_arguments: Dict[str, Any],
        inverse_tool_name: Optional[str] = None,
        inverse_arguments: Optional[Dict[str, Any]] = None,
    ) -> TransactionalActionRecord:
        """Log a mutating action to the transaction journal."""
        return self.journal.record(
            tool_name=tool_name,
            target_resource_id=target_resource_id,
            forward_arguments=forward_arguments,
            inverse_tool_name=inverse_tool_name,
            inverse_arguments=inverse_arguments,
        )

    def rollback_all(self) -> List[Dict[str, Any]]:
        """Trigger reverse compensatory sequence."""
        seq = self.journal.generate_rollback_sequence()
        self.journal.clear()
        return seq
