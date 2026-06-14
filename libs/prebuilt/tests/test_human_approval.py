"""Regression tests for human_approval ToolCallWrapper.

Pinned cases from PR review (rpelevin):

1.  allow-listed tool passes without creating a pending record.
2.  deny-listed tool returns a terminal denial and never interrupts.
3.  unclassified tool creates a pending record and does not execute before resume.
4.  resume for one pending tool call cannot approve a different tool call.
5.  edited arguments change the approved_args_digest and are visible in final state.
6.  expired or cancelled approval cannot be reused by replaying the same resume value.
7.  checkpoint/resume/reconnect preserves the pending record and terminal outcome.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command, Interrupt

from langgraph.prebuilt import ToolNode, human_approval
from langgraph.prebuilt.human_approval import (
    ApprovalDecision,
    PendingApproval,
    _canonical_digest,
    _validate_decision,
)
from langgraph.prebuilt.tool_node import ToolCallRequest

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Shared helpers (mirror test_on_tool_call.py conventions)
# ---------------------------------------------------------------------------


def _create_mock_runtime(store: BaseStore | None = None) -> Mock:
    mock_runtime = Mock()
    mock_runtime.store = store
    mock_runtime.context = None
    mock_runtime.stream_writer = lambda _: None
    mock_runtime.config = {"configurable": {"thread_id": "test-thread"}}
    return mock_runtime


def _create_config_with_runtime(
    thread_id: str = "test-thread",
    store: BaseStore | None = None,
) -> RunnableConfig:
    runtime = _create_mock_runtime(store)
    runtime.config = {"configurable": {"thread_id": thread_id}}
    return {
        "configurable": {
            "__pregel_runtime": runtime,
            "thread_id": thread_id,
        }
    }


def _ai_msg(tool_name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return f"contents of {path}"


@tool
def delete_file(path: str) -> str:
    """Delete a file."""
    return f"deleted {path}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"sent email to {to}"


@tool
def transfer_funds(account: str, amount: float) -> str:
    """Transfer funds."""
    return f"transferred {amount} to {account}"


ALL_TOOLS = [read_file, delete_file, send_email, transfer_funds]


# ---------------------------------------------------------------------------
# Minimal graph builder for interrupt tests
# ---------------------------------------------------------------------------


def _graph_with_wrapper(
    wrapper,
    checkpointer: BaseCheckpointSaver,
    tools=None,
) -> object:
    """Build START → tools → END graph with human_approval wrapper."""
    if tools is None:
        tools = ALL_TOOLS
    tool_node = ToolNode(tools, wrap_tool_call=wrapper)
    builder = StateGraph(MessagesState)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile(checkpointer=checkpointer)


# ===========================================================================
# Regression test 1 — allow-listed tool executes without a pending record
# ===========================================================================


def test_allow_listed_tool_executes_without_pending_record() -> None:
    """Allow-listed tool must execute immediately; interrupt() must NOT fire."""
    wrapper = human_approval(allow=["read_*"], deny=["delete_*"])
    tool_node = ToolNode([read_file], wrap_tool_call=wrapper)

    result = tool_node.invoke(
        {"messages": [_ai_msg("read_file", {"path": "/etc/hosts"})]},
        config=_create_config_with_runtime(),
    )

    messages = result["messages"]
    assert len(messages) == 1
    tm = messages[0]
    assert isinstance(tm, ToolMessage)
    assert tm.status != "error"
    assert "contents of" in tm.content
    # No pending-record metadata on an allow-listed call.
    assert "args_digest" not in tm.additional_kwargs


# ===========================================================================
# Regression test 2 — deny-listed tool returns terminal denial, never interrupts
# ===========================================================================


def test_deny_listed_tool_returns_terminal_denial_without_interrupt() -> None:
    """Deny-listed tool must be blocked immediately; execute must NOT run."""
    wrapper = human_approval(allow=["read_*"], deny=["delete_*"])
    tool_node = ToolNode([delete_file], wrap_tool_call=wrapper)

    result = tool_node.invoke(
        {"messages": [_ai_msg("delete_file", {"path": "/important"})]},
        config=_create_config_with_runtime(),
    )

    messages = result["messages"]
    assert len(messages) == 1
    tm = messages[0]
    assert isinstance(tm, ToolMessage)
    assert tm.status == "error"
    assert "denied by policy" in tm.content


# ===========================================================================
# Regression test 3 — unclassified tool creates pending record before execution
# ===========================================================================


def test_unclassified_tool_creates_pending_record_before_execution(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Unclassified tool must interrupt with PendingApproval; tool must NOT run."""
    wrapper = human_approval(allow=[], deny=[])
    executed: list[str] = []

    @tool
    def guarded_tool(x: str) -> str:
        """A guarded tool."""
        executed.append(x)
        return f"done {x}"

    graph = _graph_with_wrapper(wrapper, sync_checkpointer, tools=[guarded_tool])
    config: RunnableConfig = {"configurable": {"thread_id": "thread-3"}}
    ai_msg = _ai_msg("guarded_tool", {"x": "hello"}, call_id="tc-3")

    # First invoke: hits interrupt() inside the wrapper.
    graph.invoke({"messages": [ai_msg]}, config=config)

    # Tool was NOT executed.
    assert executed == []

    # State captures the interrupt.
    state = graph.get_state(config)
    assert state.next == ("tools",)
    task = state.tasks[0]
    assert len(task.interrupts) == 1

    pending_dict = task.interrupts[0].value
    assert pending_dict["tool_name"] == "guarded_tool"
    assert pending_dict["tool_call_id"] == "tc-3"
    assert pending_dict["policy_result"] == "requires_approval"
    assert pending_dict["terminal_state"] is None
    assert pending_dict["args_digest"] == _canonical_digest({"x": "hello"})
    assert pending_dict["resume_token"]  # non-empty


# ===========================================================================
# Regression test 4 — resume token bound to a specific tool_call_id
# ===========================================================================


def test_resume_for_one_call_cannot_approve_a_different_call() -> None:
    """ApprovalDecision for call A must be rejected when used against call B."""
    pending_a = PendingApproval(
        thread_id="t1",
        node_name="tools",
        tool_name="send_email",
        tool_call_id="call-A",
        args_digest=_canonical_digest({"to": "a@b.com", "subject": "hi", "body": ""}),
        policy_result="requires_approval",
        decision_shape="approve_reject_or_edit",
    )
    pending_b = PendingApproval(
        thread_id="t1",
        node_name="tools",
        tool_name="transfer_funds",
        tool_call_id="call-B",
        args_digest=_canonical_digest({"account": "12345", "amount": 500.0}),
        policy_result="requires_approval",
        decision_shape="approve_reject_or_edit",
    )

    # Decision correctly bound to A.
    decision_for_a = ApprovalDecision(
        token=pending_a.resume_token,
        tool_call_id=pending_a.tool_call_id,
        action="approve",
    )

    # Replaying it against B must fail — wrong token.
    with pytest.raises(ValueError, match="token mismatch"):
        _validate_decision(decision_for_a, pending_b)

    # Decision with correct token but wrong tool_call_id also fails.
    decision_wrong_id = ApprovalDecision(
        token=pending_b.resume_token,
        tool_call_id="call-WRONG",
        action="approve",
    )
    with pytest.raises(ValueError, match="tool_call_id mismatch"):
        _validate_decision(decision_wrong_id, pending_b)


# ===========================================================================
# Regression test 5 — edited args produce a distinct approved_args_digest
# ===========================================================================


def test_edited_arguments_change_digest_and_are_visible_in_final_state(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Edited args must produce their own digest, distinct from the original."""
    original_args = {"to": "boss@corp.com", "subject": "Hello", "body": "Hi"}
    edited_args = {"to": "boss@corp.com", "subject": "Hello — REVISED", "body": "Hi revised"}

    wrapper = human_approval(allow=[], deny=[])
    graph = _graph_with_wrapper(wrapper, sync_checkpointer, tools=[send_email])
    config: RunnableConfig = {"configurable": {"thread_id": "thread-5"}}

    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "send_email",
            "args": original_args,
            "id": "tc-5",
            "type": "tool_call",
        }],
    )

    # First invoke: creates PendingApproval, interrupts.
    graph.invoke({"messages": [ai_msg]}, config=config)

    state = graph.get_state(config)
    pending_dict = state.tasks[0].interrupts[0].value

    # Resume with edited args.
    decision = ApprovalDecision(
        token=pending_dict["resume_token"],
        tool_call_id=pending_dict["tool_call_id"],
        action="edit",
        edited_args=edited_args,
    )
    final_result = graph.invoke(Command(resume=decision.to_dict()), config=config)

    # Graph completed; inspect the ToolMessage.
    tool_messages = [m for m in final_result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1
    tm = tool_messages[-1]
    assert tm.status != "error"

    # The two digests must differ.
    original_digest = _canonical_digest(original_args)
    edited_digest = _canonical_digest(edited_args)
    assert original_digest != edited_digest

    # Both digests are stamped on the result.
    assert tm.additional_kwargs.get("approved_args_digest") == edited_digest
    assert tm.additional_kwargs.get("args_digest") == original_digest

    # The original digest was what was surfaced to the human.
    assert pending_dict["args_digest"] == original_digest


# ===========================================================================
# Regression test 6 — terminal approval cannot be replayed
# ===========================================================================


@pytest.mark.parametrize(
    "terminal",
    ["approved", "rejected", "edited", "expired", "cancelled", "executed"],
)
def test_terminal_approval_cannot_be_replayed(terminal: str) -> None:
    """Once resolved, any subsequent decision against the same record must fail."""
    pending = PendingApproval(
        thread_id="t1",
        node_name="tools",
        tool_name="send_email",
        tool_call_id="call-6",
        args_digest=_canonical_digest({"to": "x@y.com", "subject": "s", "body": "b"}),
        policy_result="requires_approval",
        decision_shape="approve_or_reject",
        terminal_state=terminal,  # type: ignore[arg-type]
    )
    decision = ApprovalDecision(
        token=pending.resume_token,
        tool_call_id=pending.tool_call_id,
        action="approve",
    )
    with pytest.raises(ValueError, match="terminal state"):
        _validate_decision(decision, pending)


# ===========================================================================
# Regression test 7 — checkpoint/resume/reconnect preserves pending record
# ===========================================================================


def test_checkpoint_preserves_pending_record_and_terminal_outcome(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """After interrupt, re-fetching state from checkpointer must preserve the
    pending record and allow successful resumption with correct approval."""
    wrapper = human_approval(allow=[], deny=[])
    graph = _graph_with_wrapper(wrapper, sync_checkpointer, tools=[send_email])
    config: RunnableConfig = {"configurable": {"thread_id": "thread-7"}}

    args = {"to": "cfo@corp.com", "subject": "Budget Q4", "body": "See attached"}
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "send_email",
            "args": args,
            "id": "tc-7",
            "type": "tool_call",
        }],
    )

    # Step 1 — first invocation hits interrupt.
    graph.invoke({"messages": [ai_msg]}, config=config)

    # Step 2 — simulate reconnect: re-fetch from checkpointer (same config).
    state_after_reconnect = graph.get_state(config)
    assert state_after_reconnect is not None
    assert state_after_reconnect.next == ("tools",)

    task = state_after_reconnect.tasks[0]
    assert len(task.interrupts) == 1
    pending_dict = task.interrupts[0].value

    # Pending record fields survive the checkpoint boundary.
    assert pending_dict["tool_name"] == "send_email"
    assert pending_dict["tool_call_id"] == "tc-7"
    assert pending_dict["thread_id"] == "thread-7"
    assert pending_dict["args_digest"] == _canonical_digest(args)
    assert pending_dict["terminal_state"] is None
    assert pending_dict["resume_token"]

    # Step 3 — resume with an approval decision.
    decision = ApprovalDecision(
        token=pending_dict["resume_token"],
        tool_call_id=pending_dict["tool_call_id"],
        action="approve",
    )
    final_result = graph.invoke(Command(resume=decision.to_dict()), config=config)

    # Graph completed; tool executed.
    tool_messages = [m for m in final_result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1
    tm = tool_messages[-1]
    assert tm.status != "error"
    assert tm.additional_kwargs.get("args_digest") == _canonical_digest(args)

    # Terminal: graph has no pending tasks.
    final_state = graph.get_state(config)
    assert final_state.next == ()


# ===========================================================================
# Unit tests for internal helpers
# ===========================================================================


def test_canonical_digest_is_order_independent() -> None:
    assert _canonical_digest({"b": 2, "a": 1}) == _canonical_digest({"a": 1, "b": 2})


def test_canonical_digest_changes_with_value() -> None:
    assert _canonical_digest({"a": 1}) != _canonical_digest({"a": 2})


def test_pending_approval_round_trips_to_dict() -> None:
    p = PendingApproval(
        thread_id="t",
        node_name="tools",
        tool_name="send_email",
        tool_call_id="c1",
        args_digest="abc",
        policy_result="requires_approval",
        decision_shape="approve_reject_or_edit",
    )
    assert PendingApproval.from_dict(p.to_dict()) == p


def test_validate_decision_wrong_token_raises() -> None:
    pending = PendingApproval(
        thread_id="t",
        node_name="tools",
        tool_name="send_email",
        tool_call_id="c1",
        args_digest="abc",
        policy_result="requires_approval",
        decision_shape="approve_or_reject",
    )
    decision = ApprovalDecision(
        token="completely-wrong-token",
        tool_call_id="c1",
        action="approve",
    )
    with pytest.raises(ValueError, match="token mismatch"):
        _validate_decision(decision, pending)


def test_validate_decision_edit_without_args_raises() -> None:
    pending = PendingApproval(
        thread_id="t",
        node_name="tools",
        tool_name="send_email",
        tool_call_id="c1",
        args_digest="abc",
        policy_result="requires_approval",
        decision_shape="approve_reject_or_edit",
    )
    decision = ApprovalDecision(
        token=pending.resume_token,
        tool_call_id="c1",
        action="edit",
        edited_args=None,
    )
    with pytest.raises(ValueError, match="edited_args"):
        _validate_decision(decision, pending)


def test_reject_returns_error_tool_message(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """A reject decision must produce an error ToolMessage without running the tool."""
    wrapper = human_approval(allow=[], deny=[])
    graph = _graph_with_wrapper(wrapper, sync_checkpointer, tools=[send_email])
    config: RunnableConfig = {"configurable": {"thread_id": "thread-reject"}}

    args = {"to": "x@y.com", "subject": "Hi", "body": "Test"}
    ai_msg = _ai_msg("send_email", args, call_id="tc-reject")
    graph.invoke({"messages": [ai_msg]}, config=config)

    pending_dict = graph.get_state(config).tasks[0].interrupts[0].value
    decision = ApprovalDecision(
        token=pending_dict["resume_token"],
        tool_call_id=pending_dict["tool_call_id"],
        action="reject",
    )
    final_result = graph.invoke(Command(resume=decision.to_dict()), config=config)

    tool_messages = [m for m in final_result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1
    tm = tool_messages[-1]
    assert tm.status == "error"
    assert "rejected by the operator" in tm.content


async def test_async_human_approval_allow() -> None:
    """async_human_approval allow-list path executes the tool immediately."""
    from langgraph.prebuilt.human_approval import async_human_approval

    wrapper = async_human_approval(allow=["read_*"], deny=[])
    tool_node = ToolNode([read_file], awrap_tool_call=wrapper)

    result = await tool_node.ainvoke(
        {"messages": [_ai_msg("read_file", {"path": "/tmp/x"})]},
        config=_create_config_with_runtime(),
    )
    tm = result["messages"][0]
    assert isinstance(tm, ToolMessage)
    assert tm.status != "error"
    assert "args_digest" not in tm.additional_kwargs


async def test_async_human_approval_deny() -> None:
    """async_human_approval deny-list path blocks without running the tool."""
    from langgraph.prebuilt.human_approval import async_human_approval

    wrapper = async_human_approval(allow=[], deny=["delete_*"])
    tool_node = ToolNode([delete_file], awrap_tool_call=wrapper)

    result = await tool_node.ainvoke(
        {"messages": [_ai_msg("delete_file", {"path": "/critical"})]},
        config=_create_config_with_runtime(),
    )
    tm = result["messages"][0]
    assert tm.status == "error"
    assert "denied by policy" in tm.content
