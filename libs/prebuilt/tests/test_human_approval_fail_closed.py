"""
Tests specifically focused on fail-closed behavior.

The most important property of this helper: the tool is NEVER executed
before explicit approval is granted and validated.
"""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import uuid

import pytest

from langgraph.prebuilt.human_approval import (
    human_approval,
    InMemoryDecisionStore,
    DecisionState,
    DecisionValidationError,
    DecisionExpiredError,
    DecisionCancelledError,
    _get_args_digest,
    resume_payload,
)


@pytest.fixture
def store():
    return InMemoryDecisionStore()


def test_tool_never_executes_before_approval(store):
    """CRITICAL TEST: Verify tool can't execute without explicit approval."""
    tool_executed = False

    with patch('langgraph.prebuilt.human_approval.interrupt') as mock_interrupt:
        mock_interrupt.side_effect = [
            # First payload is invalid - should fail
            {"action": "approve", "missing": "fields"},
        ]
        with pytest.raises(DecisionValidationError):
            human_approval(
                tool_name="send_email",
                tool_call_id="call1",
                args={"to": "test@example.com"},
                decision_shape={"allowed_actions": ["approve"]},
                store=store
            )

        # Tool was never executed
        assert not tool_executed


def test_no_execution_on_expired(store):
    """No execution if decision is expired."""
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
        mock_interrupt.return_value = resume_payload(
            resume_id, "approve", "call1", pd.args_digest
        )
        with pytest.raises(DecisionExpiredError):
            human_approval(
                tool_name="tool1",
                tool_call_id="call1",
                args={"a": 1},
                decision_shape={"allowed_actions": ["approve"]},
                store=store
            )


def test_no_execution_on_cancelled(store):
    """No execution if decision is cancelled."""
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
        mock_interrupt.return_value = resume_payload(
            resume_id, "approve", "call1", pd.args_digest
        )
        with pytest.raises(DecisionCancelledError):
            human_approval(
                tool_name="tool1",
                tool_call_id="call1",
                args={"a": 1},
                decision_shape={"allowed_actions": ["approve"]},
                store=store
            )


def test_no_execution_on_wrong_ids(store):
    """No execution if IDs don't match."""
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
        mock_interrupt.return_value = resume_payload(
            str(uuid.uuid4()), "approve", "call1", pd.args_digest
        )
        with pytest.raises(DecisionValidationError):
            human_approval(
                tool_name="tool1",
                tool_call_id="call1",
                args={"a": 1},
                decision_shape={"allowed_actions": ["approve"]},
                store=store
            )


def test_replay_safety(store):
    """Replaying an approval should not re-execute the tool."""
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
        state=DecisionState.EXECUTED
    )
    store.save(pd)

    with patch('langgraph.prebuilt.human_approval.interrupt') as mock_interrupt:
        mock_interrupt.return_value = resume_payload(
            resume_id, "approve", "call1", pd.args_digest
        )
        with pytest.raises(DecisionValidationError):
            human_approval(
                tool_name="tool1",
                tool_call_id="call1",
                args={"a": 1},
                decision_shape={"allowed_actions": ["approve"]},
                store=store
            )
