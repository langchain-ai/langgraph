"""
Human-in-the-loop approval helper built on existing LangGraph primitives.

This module provides a lightweight `human_approval()` function that:
- Pauses execution using `interrupt()`
- Resumes using `Command(resume=...)`
- Never executes a tool before approval
- Fails closed by default
- Tracks pending decisions with a strongly typed contract
"""

from __future__ import annotations

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Optional,
    Set,
    Tuple,
)

from langgraph.types import Command, interrupt


class DecisionState(Enum):
    """Possible states for a pending decision."""
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    CANCELLED = auto()
    EXPIRED = auto()
    EXECUTED = auto()


_VALID_TRANSITIONS: Dict[DecisionState, Set[DecisionState]] = {
    DecisionState.PENDING: {
        DecisionState.APPROVED,
        DecisionState.REJECTED,
        DecisionState.CANCELLED,
        DecisionState.EXPIRED,
    },
    DecisionState.APPROVED: {DecisionState.EXECUTED},
}


def _get_args_digest(args: Any) -> str:
    """Generate stable SHA-256 digest of arguments.

    Args:
        args: Arguments to hash (must be JSON-serializable).

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    normalized = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class DecisionStoreError(Exception):
    """Base error for decision store operations."""
    pass


class DecisionNotFoundError(DecisionStoreError):
    """Pending decision not found."""
    pass


class DecisionValidationError(Exception):
    """Validation failed for approval resume request."""
    pass


class DecisionStateError(DecisionValidationError):
    """Invalid state transition attempted."""
    pass


class DecisionExpiredError(DecisionValidationError):
    """Decision has expired."""
    pass


class DecisionCancelledError(DecisionValidationError):
    """Decision has been cancelled."""
    pass


@dataclass
class PendingDecision:
    """Contract for a pending human approval decision.

    This record contains all necessary metadata to validate, track,
    and audit approval requests and their outcomes.
    """
    checkpoint_id: str
    node_name: str
    tool_name: str
    tool_call_id: str
    args_digest: str
    decision_shape: Dict[str, Any]
    resume_command_id: str
    state: DecisionState = DecisionState.PENDING
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    terminal_state: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def is_expired(self) -> bool:
        """Check if decision has expired."""
        return datetime.now() > self.expires_at

    def can_transition_to(self, new_state: DecisionState) -> bool:
        """Check if transitioning to new_state is valid."""
        return new_state in _VALID_TRANSITIONS.get(self.state, set())

    def mark(self, new_state: DecisionState, terminal_state: Optional[str] = None) -> None:
        """Update state with validation.

        Args:
            new_state: Target state to transition to.
            terminal_state: Optional reason for terminal state.

        Raises:
            DecisionStateError: If transition is invalid.
        """
        if not self.can_transition_to(new_state):
            raise DecisionStateError(
                f"Cannot transition from {self.state.name} to {new_state.name}"
            )
        self.state = new_state
        self.terminal_state = terminal_state
        self.updated_at = datetime.now()


class DecisionStore(ABC):
    """Abstract base class for storing pending decisions."""

    @abstractmethod
    def save(self, decision: PendingDecision) -> None:
        """Save a pending decision.

        Args:
            decision: Decision to save.
        """
        pass

    @abstractmethod
    def get(self, resume_command_id: str) -> Optional[PendingDecision]:
        """Get a pending decision by resume_command_id.

        Args:
            resume_command_id: Unique ID for the resume command.

        Returns:
            Pending decision if found, None otherwise.
        """
        pass

    @abstractmethod
    def update(self, decision: PendingDecision) -> None:
        """Update an existing pending decision.

        Args:
            decision: Updated decision.

        Raises:
            DecisionNotFoundError: If decision doesn't exist.
        """
        pass

    @abstractmethod
    def delete(self, resume_command_id: str) -> None:
        """Delete a pending decision.

        Args:
            resume_command_id: ID of decision to delete.
        """
        pass


class InMemoryDecisionStore(DecisionStore):
    """In-memory decision store for testing and development."""

    def __init__(self) -> None:
        self._storage: Dict[str, PendingDecision] = {}

    def save(self, decision: PendingDecision) -> None:
        self._storage[decision.resume_command_id] = decision

    def get(self, resume_command_id: str) -> Optional[PendingDecision]:
        return self._storage.get(resume_command_id)

    def update(self, decision: PendingDecision) -> None:
        if decision.resume_command_id not in self._storage:
            raise DecisionNotFoundError(f"Decision {decision.resume_command_id} not found")
        self._storage[decision.resume_command_id] = decision

    def delete(self, resume_command_id: str) -> None:
        if resume_command_id in self._storage:
            del self._storage[resume_command_id]


# Global default store
_default_store: Optional[DecisionStore] = None


def get_default_store() -> DecisionStore:
    """Get or create the default in-memory store.

    Returns:
        Default decision store instance.
    """
    global _default_store
    if _default_store is None:
        _default_store = InMemoryDecisionStore()
    return _default_store


def set_default_store(store: DecisionStore) -> None:
    """Set the global default decision store.

    Args:
        store: Store to use as default.
    """
    global _default_store
    _default_store = store


@dataclass
class ApprovalResult:
    """Result returned from human_approval()."""
    action: str
    args: Dict[str, Any]
    decision: PendingDecision


def human_approval(
    tool_name: str,
    tool_call_id: str,
    args: Dict[str, Any],
    decision_shape: Dict[str, Any],
    node_name: str = "unknown",
    checkpoint_id: Optional[str] = None,
    store: Optional[DecisionStore] = None,
    allow_list: Optional[Set[str]] = None,
    deny_list: Optional[Set[str]] = None,
    expires_in: timedelta = timedelta(hours=24),
) -> ApprovalResult:
    """Human-in-the-loop approval helper.

    Args:
        tool_name: Name of the tool being approved.
        tool_call_id: Unique ID for this tool call.
        args: Arguments for the tool call.
        decision_shape: Shape of the decision (type, allowed_actions, etc.).
        node_name: Name of the graph node requesting approval.
        checkpoint_id: Checkpoint/thread ID (if available).
        store: DecisionStore implementation (uses default if not provided).
        allow_list: Set of tool names that bypass approval entirely.
        deny_list: Set of tool names that are always denied.
        expires_in: How long the pending decision is valid for.

    Returns:
        ApprovalResult containing action and final args.

    Raises:
        DecisionValidationError: If resume payload is invalid or expired/cancelled.
    """
    store = store or get_default_store()
    allow_list = allow_list or set()
    deny_list = deny_list or set()

    # Handle allow list first - execute immediately, no pending record
    if tool_name in allow_list:
        return ApprovalResult(
            action="approve",
            args=args,
            decision=PendingDecision(
                checkpoint_id=checkpoint_id or str(uuid.uuid4()),
                node_name=node_name,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                args_digest=_get_args_digest(args),
                decision_shape=decision_shape,
                resume_command_id=str(uuid.uuid4()),
                state=DecisionState.APPROVED,
            ),
        )

    # Handle deny list - terminal denial, no interrupt, no pending record
    if tool_name in deny_list:
        return ApprovalResult(
            action="reject",
            args=args,
            decision=PendingDecision(
                checkpoint_id=checkpoint_id or str(uuid.uuid4()),
                node_name=node_name,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                args_digest=_get_args_digest(args),
                decision_shape=decision_shape,
                resume_command_id=str(uuid.uuid4()),
                state=DecisionState.REJECTED,
                terminal_state="deny_listed",
            ),
        )

    args_digest = _get_args_digest(args)
    resume_payload = interrupt({
        "type": "approval_request",
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "args": args,
        "args_digest": args_digest,
        "decision_shape": decision_shape,
        "node_name": node_name,
        "checkpoint_id": checkpoint_id,
    })

    # Validate payload structure first
    if not isinstance(resume_payload, dict):
        raise DecisionValidationError("Resume payload must be a dictionary")

    required_fields = ["action", "resume_command_id", "tool_call_id", "args_digest"]
    for field in required_fields:
        if field not in resume_payload:
            raise DecisionValidationError(f"Missing required field: {field}")

    resume_command_id = resume_payload["resume_command_id"]
    pending_decision = store.get(resume_command_id)

    if pending_decision is None:
        # First pass: create pending record and interrupt again
        pending_decision = PendingDecision(
            checkpoint_id=checkpoint_id or str(uuid.uuid4()),
            node_name=node_name,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args_digest=args_digest,
            decision_shape=decision_shape,
            resume_command_id=resume_command_id,
            expires_at=datetime.now() + expires_in,
        )
        store.save(pending_decision)
        new_resume_payload = interrupt({
            "type": "approval_request",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "args": args,
            "args_digest": args_digest,
            "decision_shape": decision_shape,
            "node_name": node_name,
            "checkpoint_id": checkpoint_id,
            "resume_command_id": resume_command_id,
        })
        return human_approval(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args=args,
            decision_shape=decision_shape,
            node_name=node_name,
            checkpoint_id=checkpoint_id,
            store=store,
            allow_list=allow_list,
            deny_list=deny_list,
            expires_in=expires_in,
        )

    # Validate pending decision state
    if pending_decision.state != DecisionState.PENDING:
        if pending_decision.state == DecisionState.EXPIRED:
            raise DecisionExpiredError("Decision has expired")
        if pending_decision.state == DecisionState.CANCELLED:
            raise DecisionCancelledError("Decision has been cancelled")
        raise DecisionValidationError(f"Decision is in {pending_decision.state.name} state")

    if pending_decision.is_expired():
        pending_decision.mark(DecisionState.EXPIRED)
        store.update(pending_decision)
        raise DecisionExpiredError("Decision has expired")

    # Validate IDs match
    if pending_decision.tool_call_id != resume_payload["tool_call_id"]:
        raise DecisionValidationError("Tool call ID mismatch")
    if pending_decision.resume_command_id != resume_payload["resume_command_id"]:
        raise DecisionValidationError("Resume command ID mismatch")

    # Handle edited args
    final_args = resume_payload.get("args", args)
    final_digest = _get_args_digest(final_args)

    if final_digest != pending_decision.args_digest:
        # Close old decision as cancelled
        pending_decision.mark(DecisionState.CANCELLED, terminal_state="args_edited")
        store.update(pending_decision)
        # Create new decision with fresh ID and new args
        new_resume_id = str(uuid.uuid4())
        new_decision = PendingDecision(
            checkpoint_id=checkpoint_id or str(uuid.uuid4()),
            node_name=node_name,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args_digest=final_digest,
            decision_shape=decision_shape,
            resume_command_id=new_resume_id,
            expires_at=datetime.now() + expires_in,
        )
        store.save(new_decision)
        # Interrupt again for fresh approval of edited args
        new_resume_payload = interrupt({
            "type": "approval_request",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "args": final_args,
            "args_digest": final_digest,
            "decision_shape": decision_shape,
            "node_name": node_name,
            "checkpoint_id": checkpoint_id,
            "resume_command_id": new_resume_id,
        })
        return human_approval(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args=final_args,
            decision_shape=decision_shape,
            node_name=node_name,
            checkpoint_id=checkpoint_id,
            store=store,
            allow_list=allow_list,
            deny_list=deny_list,
            expires_in=expires_in,
        )

    # Validate action is allowed
    allowed_actions = decision_shape.get("allowed_actions", ["approve", "reject"])
    action = resume_payload["action"]
    if action not in allowed_actions:
        raise DecisionValidationError(f"Action '{action}' not in allowed set: {allowed_actions}")

    # Update decision state
    if action == "approve":
        pending_decision.mark(DecisionState.APPROVED)
    elif action == "reject":
        pending_decision.mark(DecisionState.REJECTED)

    store.update(pending_decision)
    return ApprovalResult(action=action, args=final_args, decision=pending_decision)


def mark_executed(
    resume_command_id: str,
    store: Optional[DecisionStore] = None,
) -> None:
    """Mark a decision as executed after tool has been run.

    Args:
        resume_command_id: Resume command ID from the decision.
        store: Decision store (uses default if not provided).

    Raises:
        DecisionNotFoundError: If decision doesn't exist.
        DecisionStateError: If transition is invalid.
    """
    store = store or get_default_store()
    decision = store.get(resume_command_id)
    if not decision:
        raise DecisionNotFoundError(f"Decision {resume_command_id} not found")
    decision.mark(DecisionState.EXECUTED)
    store.update(decision)


def resume_payload(
    resume_command_id: str,
    action: str,
    tool_call_id: str,
    args_digest: str,
    args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Helper to create a valid resume payload.

    Args:
        resume_command_id: Resume command ID from interrupt data.
        action: Decision action (e.g., "approve", "reject").
        tool_call_id: Tool call ID from interrupt data.
        args_digest: Args digest from interrupt data.
        args: Optional edited arguments (will generate new digest).

    Returns:
        Valid resume payload for Command(resume=...).
    """
    payload = {
        "action": action,
        "resume_command_id": resume_command_id,
        "tool_call_id": tool_call_id,
        "args_digest": args_digest,
    }
    if args is not None:
        payload["args"] = args
    return payload
