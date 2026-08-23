"""Tests for human_approval ToolCallWrapper (HITL pending-decision contract)."""

from __future__ import annotations

from typing import Annotated, Any
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph, add_messages
from langgraph.types import Command, Interrupt, interrupt
from typing_extensions import TypedDict

from langgraph.prebuilt.human_approval import (
    ApprovalDecision,
    ApprovalValidationError,
    PendingApproval,
    _ApprovalPolicy,
    async_human_approval,
    canonical_args_digest,
    compute_resume_token,
    human_approval,
    validate_decision,
)
from langgraph.prebuilt.tool_node import ToolNode
from tests.any_str import AnyStr


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return f"contents:{path}"


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent:{to}:{body}"


@tool
def drop_table(name: str) -> str:
    """Drop a table."""
    return f"dropped:{name}"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _build_graph(
    *,
    allow: list[str] | None = None,
    deny: list[str] | None = None,
    decision_shape: str = "approve_reject_or_edit",
    secret: str | None = None,
):
    wrapper = human_approval(
        allow=allow,
        deny=deny,
        decision_shape=decision_shape,  # type: ignore[arg-type]
        secret=secret,
    )
    tools = ToolNode([read_file, send_email, drop_table], wrap_tool_call=wrapper)
    builder = StateGraph(AgentState)
    builder.add_node("tools", tools)
    builder.add_edge(START, "tools")
    return builder.compile(checkpointer=InMemorySaver())


def _ai_tool_call(
    name: str, args: dict[str, Any], call_id: str = "call_1"
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _pending_from_result(result: dict[str, Any]) -> dict[str, Any]:
    interrupts = result.get("__interrupt__") or ()
    assert interrupts, f"expected interrupt, got {result!r}"
    value = interrupts[0].value
    assert isinstance(value, dict)
    return value


# --- unit helpers -----------------------------------------------------------


def test_canonical_args_digest_is_order_independent() -> None:
    a = canonical_args_digest({"b": 1, "a": 2})
    b = canonical_args_digest({"a": 2, "b": 1})
    assert a == b
    assert len(a) == 64


def test_resume_token_hmac_requires_secret() -> None:
    kwargs = dict(
        thread_id="t1",
        node_name="tools",
        tool_name="send_email",
        tool_call_id="c1",
        args_digest="abc",
    )
    plain = compute_resume_token(**kwargs)
    keyed = compute_resume_token(**kwargs, secret="s3cret")
    assert plain != keyed
    assert keyed == compute_resume_token(**kwargs, secret="s3cret")
    assert keyed != compute_resume_token(**kwargs, secret="other")


def test_policy_matching_is_case_sensitive() -> None:
    policy = _ApprovalPolicy(allow=["read_*"], deny=["delete_*"])
    assert policy.classify("read_file") == "allow"
    assert policy.classify("Read_file") == "requires_approval"
    assert policy.classify("delete_file") == "deny"
    assert policy.classify("Delete_file") == "requires_approval"


def test_allow_deny_lists_are_copied_at_construction() -> None:
    allow = ["read_*"]
    deny = ["drop_*"]
    graph = _build_graph(allow=allow, deny=deny)
    allow.append("send_*")
    deny.clear()

    config = {"configurable": {"thread_id": "copy-1"}}
    # send_email must still interrupt (not allow-listed at construction)
    result = graph.invoke(
        {
            "messages": [
                HumanMessage("hi"),
                _ai_tool_call("send_email", {"to": "a", "body": "b"}),
            ]
        },
        config,
    )
    assert "__interrupt__" in result

    # drop_table must still deny (deny list copied)
    result2 = graph.invoke(
        {
            "messages": [
                HumanMessage("hi"),
                _ai_tool_call("drop_table", {"name": "users"}, call_id="c2"),
            ]
        },
        {"configurable": {"thread_id": "copy-2"}},
    )
    msg = result2["messages"][-1]
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    assert "denied" in msg.content.lower()


def test_validate_decision_token_mismatch_hides_expected() -> None:
    pending = PendingApproval(
        thread_id="t",
        node_name="tools",
        tool_name="send_email",
        tool_call_id="c1",
        args_digest="d",
        policy_result="requires_approval",
        decision_shape="approve_reject_or_edit",
        resume_token="expected-token-value",
    )
    with pytest.raises(ApprovalValidationError, match="resume token mismatch") as exc:
        validate_decision(
            pending,
            ApprovalDecision(token="wrong", tool_call_id="c1", action="approve"),
        )
    assert "expected-token-value" not in str(exc.value)


def test_decision_shape_blocks_edit_when_approve_or_reject_only() -> None:
    pending = PendingApproval(
        thread_id="t",
        node_name="tools",
        tool_name="send_email",
        tool_call_id="c1",
        args_digest="d",
        policy_result="requires_approval",
        decision_shape="approve_or_reject",
        resume_token="tok",
    )
    with pytest.raises(ApprovalValidationError, match="edit not allowed"):
        validate_decision(
            pending,
            ApprovalDecision(
                token="tok",
                tool_call_id="c1",
                action="edit",
                edited_args={"to": "x", "body": "y"},
            ),
        )
    # approve still ok
    validate_decision(
        pending,
        ApprovalDecision(token="tok", tool_call_id="c1", action="approve"),
    )


def test_terminal_state_blocks_replay() -> None:
    for terminal in (
        "approved",
        "rejected",
        "edited",
        "expired",
        "cancelled",
        "executed",
    ):
        pending = PendingApproval(
            thread_id="t",
            node_name="tools",
            tool_name="send_email",
            tool_call_id="c1",
            args_digest="d",
            policy_result="requires_approval",
            decision_shape="approve_reject_or_edit",
            resume_token="tok",
            terminal_state=terminal,  # type: ignore[arg-type]
        )
        with pytest.raises(ApprovalValidationError, match="already resolved"):
            validate_decision(
                pending,
                ApprovalDecision(token="tok", tool_call_id="c1", action="approve"),
            )


# --- integration: allow / deny / interrupt ---------------------------------


def test_allow_listed_tool_passes_without_pending_record() -> None:
    graph = _build_graph(allow=["read_*"], deny=["drop_*"])
    config = {"configurable": {"thread_id": "allow-1"}}
    result = graph.invoke(
        {
            "messages": [
                HumanMessage("hi"),
                _ai_tool_call("read_file", {"path": "/tmp/a"}),
            ]
        },
        config,
    )
    assert "__interrupt__" not in result
    msg = result["messages"][-1]
    assert isinstance(msg, ToolMessage)
    assert msg.content == "contents:/tmp/a"
    assert "pending_approval" not in (msg.additional_kwargs or {})


def test_deny_listed_tool_returns_terminal_denial_without_interrupt() -> None:
    graph = _build_graph(allow=["read_*"], deny=["drop_*"])
    config = {"configurable": {"thread_id": "deny-1"}}
    result = graph.invoke(
        {
            "messages": [
                HumanMessage("hi"),
                _ai_tool_call("drop_table", {"name": "users"}),
            ]
        },
        config,
    )
    assert "__interrupt__" not in result
    msg = result["messages"][-1]
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    audit = msg.additional_kwargs["pending_approval"]
    assert audit["policy_result"] == "deny"
    assert audit["terminal_state"] == "rejected"


def test_unclassified_tool_creates_pending_record_and_does_not_execute() -> None:
    executed = {"n": 0}

    @tool
    def send_email_tracked(to: str, body: str) -> str:
        """Send an email."""
        executed["n"] += 1
        return f"sent:{to}:{body}"

    wrapper = human_approval(allow=["read_*"], deny=["drop_*"])
    tools = ToolNode([send_email_tracked], wrap_tool_call=wrapper)
    builder = StateGraph(AgentState)
    builder.add_node("tools", tools)
    builder.add_edge(START, "tools")
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "pending-1"}}
    result = graph.invoke(
        {
            "messages": [
                HumanMessage("hi"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "send_email_tracked",
                            "args": {"to": "a@b.com", "body": "hi"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        },
        config,
    )
    pending = _pending_from_result(result)
    assert pending["tool_name"] == "send_email_tracked"
    assert pending["tool_call_id"] == "call_1"
    assert pending["policy_result"] == "requires_approval"
    assert pending["terminal_state"] is None
    assert pending["args_digest"] == canonical_args_digest(
        {"to": "a@b.com", "body": "hi"}
    )
    assert executed["n"] == 0


def test_resume_for_one_call_cannot_approve_another() -> None:
    graph = _build_graph(secret="test-secret")
    config = {"configurable": {"thread_id": "cross-1"}}
    result = graph.invoke(
        {
            "messages": [
                HumanMessage("hi"),
                _ai_tool_call("send_email", {"to": "a", "body": "b"}),
            ]
        },
        config,
    )
    pending = _pending_from_result(result)

    # Forge a decision for a different tool_call_id with the real token.
    bad = ApprovalDecision(
        token=pending["resume_token"],
        tool_call_id="other_call",
        action="approve",
    )
    resumed = graph.invoke(Command(resume=bad.to_dict()), config)
    msg = resumed["messages"][-1]
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error"
    assert "tool_call_id mismatch" in msg.content


def test_edited_args_change_approved_digest_and_bind_execution() -> None:
    graph = _build_graph(secret="test-secret")
    config = {"configurable": {"thread_id": "edit-1"}}
    original_args = {"to": "a@b.com", "body": "draft"}
    result = graph.invoke(
        {
            "messages": [
                HumanMessage("hi"),
                _ai_tool_call("send_email", original_args),
            ]
        },
        config,
    )
    pending = _pending_from_result(result)
    edited = {"to": "safe@b.com", "body": "approved"}
    decision = ApprovalDecision(
        token=pending["resume_token"],
        tool_call_id=pending["tool_call_id"],
        action="edit",
        edited_args=edited,
    )
    resumed = graph.invoke(Command(resume=decision.to_dict()), config)
    msg = resumed["messages"][-1]
    assert isinstance(msg, ToolMessage)
    assert msg.content == "sent:safe@b.com:approved"
    audit = msg.additional_kwargs["pending_approval"]
    assert audit["args_digest"] == canonical_args_digest(original_args)
    assert audit["approved_args_digest"] == canonical_args_digest(edited)
    assert audit["args_digest"] != audit["approved_args_digest"]
    assert audit["supersedes_tool_call_id"] == pending["tool_call_id"]
    assert audit["terminal_state"] == "executed"


def test_reject_leaves_call_visible_but_non_dispatchable() -> None:
    """Dual-view invariant: history keeps the call; dispatch set has no executable original."""
    graph = _build_graph(secret="test-secret")
    config = {"configurable": {"thread_id": "reject-1"}}
    ai = _ai_tool_call("send_email", {"to": "a", "body": "b"})
    result = graph.invoke({"messages": [HumanMessage("hi"), ai]}, config)
    pending = _pending_from_result(result)

    decision = ApprovalDecision(
        token=pending["resume_token"],
        tool_call_id=pending["tool_call_id"],
        action="reject",
        message="Do not send.",
    )
    resumed = graph.invoke(Command(resume=decision.to_dict()), config)
    messages = resumed["messages"]

    # Audit / history view: original AI tool call still present
    assert any(
        isinstance(m, AIMessage)
        and any(tc.get("id") == "call_1" for tc in (m.tool_calls or []))
        for m in messages
    )

    # Dispatch / runtime view: a terminal ToolMessage consumed the call
    tool_msgs = [
        m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == "call_1"
    ]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "error"
    audit = tool_msgs[0].additional_kwargs["pending_approval"]
    assert audit["terminal_state"] == "rejected"
    assert audit["approved_tool_call_id"] is None

    # Graph finished tools node; nothing left to dispatch for call_1
    state = graph.get_state(config)
    assert state.next == ()


def test_modify_records_supersession_before_executing_successor() -> None:
    graph = _build_graph(secret="test-secret")
    config = {"configurable": {"thread_id": "modify-1"}}
    result = graph.invoke(
        {
            "messages": [
                HumanMessage("hi"),
                _ai_tool_call("send_email", {"to": "a", "body": "old"}),
            ]
        },
        config,
    )
    pending = _pending_from_result(result)
    decision = ApprovalDecision(
        token=pending["resume_token"],
        tool_call_id=pending["tool_call_id"],
        action="edit",
        edited_args={"to": "b", "body": "new"},
    )
    resumed = graph.invoke(Command(resume=decision.to_dict()), config)
    msg = resumed["messages"][-1]
    audit = msg.additional_kwargs["pending_approval"]
    assert audit["supersedes_tool_call_id"] == "call_1"
    assert audit["approved_tool_call_id"] == "call_1"
    assert msg.content == "sent:b:new"


def test_checkpoint_resume_preserves_pending_and_outcome() -> None:
    graph = _build_graph(secret="test-secret")
    config = {"configurable": {"thread_id": "ckpt-1"}}
    result = graph.invoke(
        {
            "messages": [
                HumanMessage("hi"),
                _ai_tool_call("send_email", {"to": "a", "body": "b"}),
            ]
        },
        config,
    )
    pending = _pending_from_result(result)

    # Reconnect: load state from checkpointer
    state = graph.get_state(config)
    assert state.tasks
    assert state.tasks[0].interrupts == (Interrupt(value=pending, id=AnyStr()),)

    decision = ApprovalDecision(
        token=pending["resume_token"],
        tool_call_id=pending["tool_call_id"],
        action="approve",
    )
    resumed = graph.invoke(Command(resume=decision.to_dict()), config)
    msg = resumed["messages"][-1]
    assert msg.content == "sent:a:b"
    assert msg.additional_kwargs["pending_approval"]["terminal_state"] == "executed"

    # Duplicate resume after terminal outcome is a no-op / does not re-execute
    again = graph.invoke(Command(resume=decision.to_dict()), config)
    tool_msgs = [
        m
        for m in again["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id == "call_1"
    ]
    assert len(tool_msgs) == 1


def test_hmac_token_not_forgeable_without_secret() -> None:
    graph = _build_graph(secret="server-secret")
    config = {"configurable": {"thread_id": "hmac-1"}}
    result = graph.invoke(
        {
            "messages": [
                HumanMessage("hi"),
                _ai_tool_call("send_email", {"to": "a", "body": "b"}),
            ]
        },
        config,
    )
    pending = _pending_from_result(result)
    forged = compute_resume_token(
        thread_id=pending["thread_id"],
        node_name=pending["node_name"],
        tool_name=pending["tool_name"],
        tool_call_id=pending["tool_call_id"],
        args_digest=pending["args_digest"],
        secret=None,  # attacker without server secret
    )
    assert forged != pending["resume_token"]
    resumed = graph.invoke(
        Command(
            resume=ApprovalDecision(
                token=forged,
                tool_call_id=pending["tool_call_id"],
                action="approve",
            ).to_dict()
        ),
        config,
    )
    msg = resumed["messages"][-1]
    assert msg.status == "error"
    assert "resume token mismatch" in msg.content


async def test_async_allow_and_deny_paths() -> None:
    def _config() -> RunnableConfig:
        runtime = Mock()
        runtime.store = None
        runtime.context = None
        runtime.stream_writer = lambda _: None
        return {"configurable": {"__pregel_runtime": runtime}}

    allow_wrapper = async_human_approval(allow=["read_*"], deny=["drop_*"])
    deny_wrapper = async_human_approval(allow=["read_*"], deny=["drop_*"])

    allow_node = ToolNode([read_file], awrap_tool_call=allow_wrapper)
    deny_node = ToolNode([drop_table], awrap_tool_call=deny_wrapper)

    allowed = await allow_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "x"},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_config(),
    )
    assert allowed["messages"][-1].content == "contents:x"

    denied = await deny_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "drop_table",
                            "args": {"name": "t"},
                            "id": "2",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_config(),
    )
    assert denied["messages"][-1].status == "error"


def test_wrap_tool_call_interrupt_propagates_with_bool_handle_errors() -> None:
    """GraphBubbleUp from wrap_tool_call must not be converted to ToolMessage."""

    def raising_wrapper(request, execute):
        return interrupt({"need": "approval"})

    node = ToolNode(
        [send_email], wrap_tool_call=raising_wrapper, handle_tool_errors=True
    )
    builder = StateGraph(AgentState)
    builder.add_node("tools", node)
    builder.add_edge(START, "tools")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "bubble-1"}}
    result = graph.invoke(
        {
            "messages": [
                HumanMessage("hi"),
                _ai_tool_call("send_email", {"to": "a", "body": "b"}),
            ]
        },
        config,
    )
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value == {"need": "approval"}
