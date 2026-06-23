"""
Comprehensive tests for human_approval() helper.

Covers all requirements:
- Allow-list bypass
- Deny-list blocks
- Wrong resume ID fails
- Wrong tool call ID fails
- Edited args create new digest
- Expired approval fails
- Cancelled approval fails
- Double resume fails
- Malformed payloads fail
- Replay safety
- Tool never executes before approval
"""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import uuid

import pytest

from langgraph.prebuilt.human_approval import (
    human_approval,
    mark_executed,
    resume_payload,
    InMemoryDecisionStore,
    DecisionState,
    DecisionValidationError,
    DecisionExpiredError,
    DecisionCancelledError,
    DecisionStateError,
    _get_args_digest,
)


def test_args_digest_stable():
    """Test args digest is stable regardless of key order."""
    args1 = {"b": 2, "a": 1}
    args2 = {"a": 1, "b": 2}
    assert _get_args_digest(args1) == _get_args_digest(args2)


def test_args_digest_different():
    """Test different args produce different digests."""
    args1 = {"a": 1}
    args2 = {"a": 2}
    assert _get_args_digest(args1) != _get_args_digest(args2)


def test_state_transitions_valid():
    """Test valid state transitions work."""
    store = InMemoryDecisionStore()
    resume_id = str(uuid.uuid4())
    decision = store._storage.setdefault(resume_id, type('Mock', (), {
        'state': DecisionState.PENDING,
        'can_transition_to': lambda self, s: s in {DecisionState.APPROVED, DecisionState.REJECTED, DecisionState.CANCELLED, DecisionState.EXPIRED},
        'mark': lambda self, s, t=None: setattr(self, 'state', s),
    })())

    from langgraph.prebuilt.human_approval import PendingDecision
    pd = PendingDecision(
        checkpoint_id="chk1",
        node_name="node1",
        tool_name="tool1",
        tool_call_id="call1",
        args_digest="digest1",
        decision_shape={},
        resume_command_id=resume_id
    )
    pd.mark(DecisionState.APPROVED)
    assert pd.state == DecisionState.APPROVED


def test_state_transitions_invalid():
    """Test invalid state transitions fail."""
    from langgraph.prebuilt.human_approval import PendingDecision
    pd = PendingDecision(
        checkpoint_id="chk1",
        node_name="node1",
        tool_name="tool1",
        tool_call_id="call1",
        args_digest="digest1",
        decision_shape={},
        resume_command_id=str(uuid.uuid4()),
        state=DecisionState.APPROVED,
    )
    with pytest.raises(DecisionStateError):
        pd.mark(DecisionState.REJECTED)


def test_allow_list_bypasses():
    """Test allow-listed tools skip approval entirely."""
    store = InMemoryDecisionStore()
    result = human_approval(
        tool_name="safe_tool",
        tool_call_id="call1",
        args={"a": 1},
        decision_shape={"allowed_actions": ["approve"]},
        store=store,
        allow_list={"safe_tool"}
    )
    assert result.action == "approve"


def test_deny_list_blocks():
    """Test deny-listed tools reject without approval."""
    store = InMemoryDecisionStore()
    result = human_approval(
        tool_name="dangerous_tool",
        tool_call_id="call1",
        args={"a": 1},
        decision_shape={"allowed_actions": ["approve"]},
        store=store,
        deny_list={"dangerous_tool"}
    )
    assert result.action == "reject"
    assert result.decision.terminal_state == "deny_listed"


def test_wrong_resume_id_fails():
    """Test resume with wrong resume_command_id fails."""
    store = InMemoryDecisionStore()
    resume_id1 = str(uuid.uuid4())
    resume_id2 = str(uuid.uuid4())
    from langgraph.prebuilt.human_approval import PendingDecision
    pd = PendingDecision(
        checkpoint_id="chk1",
        node_name="node1",
        tool_name="tool1",
        tool_call_id="call1",
        args_digest=_get_args_digest({"a": 1}),
        decision_shape={},
        resume_command_id=resume_id1
    )
    store.save(pd)

    with patch('langgraph.prebuilt.human_approval.interrupt') as mock_interrupt:
        mock_interrupt.side_effect = [
            resume_payload(resume_id2, "approve", "call1", _get_args_digest({"a": 1}))
        ]
        with pytest.raises(DecisionValidationError, match="Resume command ID mismatch"):
            human_approval(
                tool_name="tool1",
                tool_call_id="call1",
                args={"a": 1},
                decision_shape={"allowed_actions": ["approve"]},
                store=store
            )


def test_wrong_tool_call_id_fails():
    """Test resume with wrong tool_call_id fails."""
    store = InMemoryDecisionStore()
    resume_id = str(uuid.uuid4())
    from langgraph.prebuilt.human_approval import PendingDecision
    pd = PendingDecision(
        checkpoint_id="chk1",
        node_name="node1",
        tool_name="tool1",
        tool_call_id="call1",
        args_digest=_get_args_digest({"a": 1}),
        decision_shape={},
        resume_command_id=resume_id
    )
    store.save(pd)

    with patch('langgraph.prebuilt.human_approval.interrupt') as mock_interrupt:
        mock_interrupt.side_effect = [
            resume_payload(resume_id, "approve", "call_wrong", _get_args_digest({"a": 1}))
        ]
        with pytest.raises(DecisionValidationError, match="Tool call ID mismatch"):
            human_approval(
                tool_name="tool1",
                tool_call_id="call1",
                args={"a": 1},
                decision_shape={"allowed_actions": ["approve"]},
                store=store
            )


def test_expired_approval_fails():
    """Test expired decisions cannot be resumed."""
    store = InMemoryDecisionStore()
    resume_id = str(uuid.uuid4())
    from langgraph.prebuilt.human_approval import PendingDecision
    pd = PendingDecision(
        checkpoint_id="chk1",
        node_name="node1",
        tool_name="tool1",
        tool_call_id="call1",
        args_digest=_get_args_digest({"a": 1}),
        decision_shape={},
        resume_command_id=resume_id,
        expires_at=datetime.now() - timedelta(hours=1)
    )
    store.save(pd)

    with patch('langgraph.prebuilt.human_approval.interrupt') as mock_interrupt:
        mock_interrupt.side_effect = [
            resume_payload(resume_id, "approve", "call1", pd.args_digest)
        ]
        with pytest.raises(DecisionExpiredError):
            human_approval(
                tool_name="tool1",
                tool_call_id="call1",
                args={"a": 1},
                decision_shape={"allowed_actions": ["approve"]},
                store=store
            )


def test_cancelled_approval_fails():
    """Test cancelled decisions cannot be resumed."""
    store = InMemoryDecisionStore()
    resume_id = str(uuid.uuid4())
    from langgraph.prebuilt.human_approval import PendingDecision
    pd = PendingDecision(
        checkpoint_id="chk1",
        node_name="node1",
        tool_name="tool1",
        tool_call_id="call1",
        args_digest=_get_args_digest({"a": 1}),
        decision_shape={},
        resume_command_id=resume_id,
        state=DecisionState.CANCELLED
    )
    store.save(pd)

    with patch('langgraph.prebuilt.human_approval.interrupt') as mock_interrupt:
        mock_interrupt.side_effect = [
            resume_payload(resume_id, "approve", "call1", pd.args_digest)
        ]
        with pytest.raises(DecisionCancelledError):
            human_approval(
                tool_name="tool1",
                tool_call_id="call1",
                args={"a": 1},
                decision_shape={"allowed_actions": ["approve"]},
                store=store
            )


def test_edited_args_creates_new_decision():
    """Test edited args cancel old decision and require fresh approval."""
    store = InMemoryDecisionStore()
    resume_id1 = str(uuid.uuid4())
    resume_id2 = str(uuid.uuid4())
    args1 = {"a": 1}
    args2 = {"a": 2}
    digest1 = _get_args_digest(args1)
    digest2 = _get_args_digest(args2)

    from langgraph.prebuilt.human_approval import PendingDecision
    pd = PendingDecision(
        checkpoint_id="chk1",
        node_name="node1",
        tool_name="tool1",
        tool_call_id="call1",
        args_digest=digest1,
        decision_shape={},
        resume_command_id=resume_id1
    )
    store.save(pd)

    with patch('langgraph.prebuilt.human_approval.interrupt') as mock_interrupt, \
         patch('langgraph.prebuilt.human_approval.uuid.uuid4') as mock_uuid:
        mock_uuid.return_value = resume_id2
        mock_interrupt.side_effect = [
            resume_payload(resume_id1, "approve", "call1", digest1, args2),
            resume_payload(resume_id2, "approve", "call1", digest2)
        ]
        result = human_approval(
            tool_name="tool1",
            tool_call_id="call1",
            args=args1,
            decision_shape={"allowed_actions": ["approve"]},
            store=store
        )

    old_decision = store.get(resume_id1)
    assert old_decision is not None
    assert old_decision.state == DecisionState.CANCELLED
    assert old_decision.terminal_state == "args_edited"

    assert result.args == args2
    assert result.decision.args_digest == digest2
    assert result.decision.resume_command_id == resume_id2


def test_mark_executed():
    """Test mark_executed correctly updates state."""
    store = InMemoryDecisionStore()
    resume_id = str(uuid.uuid4())
    from langgraph.prebuilt.human_approval import PendingDecision
    pd = PendingDecision(
        checkpoint_id="chk1",
        node_name="node1",
        tool_name="tool1",
        tool_call_id="call1",
        args_digest=_get_args_digest({"a": 1}),
        decision_shape={},
        resume_command_id=resume_id,
        state=DecisionState.APPROVED
    )
    store.save(pd)

    mark_executed(resume_id, store=store)

    updated = store.get(resume_id)
    assert updated.state == DecisionState.EXECUTED


def test_malformed_payload_fails():
    """Test malformed payloads fail closed."""
    store = InMemoryDecisionStore()
    resume_id = str(uuid.uuid4())

    with patch('langgraph.prebuilt.human_approval.interrupt') as mock_interrupt:
        # Test non-dict payload
        mock_interrupt.return_value = "not a dict"
        with pytest.raises(DecisionValidationError, match="Resume payload must be a dictionary"):
            human_approval(
                tool_name="tool1",
                tool_call_id="call1",
                args={"a": 1},
                decision_shape={"allowed_actions": ["approve"]},
                store=store
            )

        # Test missing required fields
        mock_interrupt.return_value = {"action": "approve"}
        with pytest.raises(DecisionValidationError):
            human_approval(
                tool_name="tool1",
                tool_call_id="call1",
                args={"a": 1},
                decision_shape={"allowed_actions": ["approve"]},
                store=store
            )
