"""Tests demonstrating safe human-in-the-loop approval patterns with interrupt().

When interrupt() is used for human approval, the resume value should carry only a
decision (approved/rejected) plus a proposal_id. The action and arguments to
execute should be re-derived from pre-interrupt state, not taken from the resume
payload. This prevents resume payload drift where an attacker or buggy client
approves a different action than the one proposed.
"""

import operator
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

pytestmark = pytest.mark.anyio


class ProposalState(TypedDict):
    """State carrying a pending approval proposal and execution log."""

    pending_proposal: Annotated[dict[str, Any] | None, lambda _curr, _prev: _prev]
    executed_action: Annotated[list[dict[str, Any]], operator.add]


PROPOSAL = {
    "proposal_id": "proposal-001",
    "proposed_action": "search_db",
    "proposed_args": {"query": "users:42"},
    "reviewer": "alice",
}


def _malicious_resume(action: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "proposal_id": "proposal-001",
        "approved": True,
        "approved_action": action,
        "approved_args": args,
    }


async def test_safe_pattern_re_derives_action_from_proposal_state() -> None:
    """Safe resume: re-derive action from pre-interrupt proposal state."""

    async def request_approval(state: ProposalState):
        proposal = PROPOSAL
        resume_value = interrupt(proposal)
        approved = resume_value.get("approved", False)
        return {
            "pending_proposal": proposal,
            "executed_action": [
                {
                    "proposal_id": proposal["proposal_id"],
                    "action": proposal["proposed_action"] if approved else "none",
                    "args": proposal["proposed_args"] if approved else {},
                    "source": "proposal_state",
                }
            ],
        }

    async def finalize(state: ProposalState):
        pass

    builder = StateGraph(ProposalState)
    builder.add_node("request_approval", request_approval)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "request_approval")
    builder.add_edge("request_approval", "finalize")
    builder.add_edge("finalize", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "hitl-safe-1"}}

    # First invoke — should interrupt at request_approval
    await graph.ainvoke({"pending_proposal": None, "executed_action": []}, config)
    state = await graph.aget_state(config)
    assert state.next == (
        "request_approval",
    ), f"Expected interrupt before request_approval, got next={state.next}"

    # Resume with a malicious payload
    malicious = _malicious_resume("delete_record", {"table": "users", "record_id": 42})
    final_state = await graph.ainvoke(Command(resume=malicious), config)

    executed = final_state["executed_action"][0]
    assert executed["action"] == "search_db"
    assert executed["args"] == {"query": "users:42"}
    assert executed["source"] == "proposal_state"


async def test_unsafe_pattern_executes_directly_from_resume_value() -> None:
    """Unsafe resume: reading action fields directly from the resume payload."""

    async def request_approval_unsafe(state: ProposalState):
        proposal = PROPOSAL
        resume_value = interrupt(proposal)
        return {
            "pending_proposal": proposal,
            "executed_action": [
                {
                    "proposal_id": resume_value.get("proposal_id"),
                    "action": resume_value.get("approved_action"),
                    "args": resume_value.get("approved_args", {}),
                    "source": "resume_payload",
                }
            ],
        }

    async def finalize(state: ProposalState):
        pass

    builder = StateGraph(ProposalState)
    builder.add_node("request_approval", request_approval_unsafe)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "request_approval")
    builder.add_edge("request_approval", "finalize")
    builder.add_edge("finalize", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "hitl-unsafe-1"}}

    await graph.ainvoke({"pending_proposal": None, "executed_action": []}, config)
    state = await graph.aget_state(config)
    assert state.next == (
        "request_approval",
    ), f"Expected interrupt, got next={state.next}"

    malicious = _malicious_resume("delete_record", {"table": "users", "record_id": 42})
    final_state = await graph.ainvoke(Command(resume=malicious), config)

    executed = final_state["executed_action"][0]
    assert executed["action"] == "delete_record"
    assert executed["args"] == {"table": "users", "record_id": 42}
    assert executed["source"] == "resume_payload"


async def test_resume_rejects_wrong_proposal_id() -> None:
    """Validate resume against proposal id before accepting the decision."""

    async def request_approval(state: ProposalState):
        proposal = PROPOSAL
        resume_value = interrupt(proposal)
        if resume_value.get("proposal_id") != proposal["proposal_id"]:
            return {
                "pending_proposal": proposal,
                "executed_action": [
                    {
                        "proposal_id": resume_value.get("proposal_id"),
                        "action": "rejected_mismatch",
                        "args": {},
                        "source": "validated_reject",
                    }
                ],
            }
        approved = resume_value.get("approved", False)
        return {
            "pending_proposal": proposal,
            "executed_action": [
                {
                    "proposal_id": proposal["proposal_id"],
                    "action": proposal["proposed_action"] if approved else "none",
                    "args": proposal["proposed_args"] if approved else {},
                    "source": "proposal_state",
                }
            ],
        }

    async def finalize(state: ProposalState):
        pass

    builder = StateGraph(ProposalState)
    builder.add_node("request_approval", request_approval)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "request_approval")
    builder.add_edge("request_approval", "finalize")
    builder.add_edge("finalize", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "hitl-validate-1"}}

    await graph.ainvoke({"pending_proposal": None, "executed_action": []}, config)

    # Resume with wrong proposal id
    wrong_id = {"proposal_id": "proposal-999", "approved": True}
    final_state = await graph.ainvoke(Command(resume=wrong_id), config)

    executed = final_state["executed_action"][0]
    assert executed["action"] == "rejected_mismatch"
    assert executed["source"] == "validated_reject"
