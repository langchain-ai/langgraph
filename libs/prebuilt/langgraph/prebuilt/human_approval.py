"""Human-in-the-loop approval gate for `ToolNode` tool calls.

Provides a `human_approval()` / `async_human_approval()` factory that returns a
`ToolCallWrapper` for `ToolNode(wrap_tool_call=...)`.

Policy evaluation per tool call (deny → allow → interrupt):

- **deny**: return a terminal denial `ToolMessage` (no interrupt, no execution)
- **allow**: execute immediately (no pending record)
- **unclassified**: create a durable `PendingApproval` and pause via `interrupt()`

Resume accepts a structured `ApprovalDecision` bound to the pending record.
Edited arguments rebind execution to a new digest and mark the original call
as superseded. Audit fields are stamped on `ToolMessage.additional_kwargs`
while dispatchability is determined solely by whether a terminal tool result
was produced — history visibility is not execution authority.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal, cast

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from langgraph.prebuilt.tool_node import (
    AsyncToolCallWrapper,
    ToolCallRequest,
    ToolCallWrapper,
)

PolicyResult = Literal["allow", "deny", "requires_approval"]
DecisionShape = Literal["approve_or_reject", "approve_reject_or_edit"]
TerminalState = Literal[
    "approved",
    "rejected",
    "edited",
    "deferred",
    "expired",
    "cancelled",
    "executed",
]
DecisionAction = Literal["approve", "reject", "edit", "defer"]

_AUDIT_KW = "pending_approval"
_DENIAL_CONTENT = "Tool call denied by human_approval policy."
_REJECT_CONTENT = "Tool call rejected by human reviewer."
_VALIDATION_FAIL_CONTENT = "Approval decision rejected: {reason}"


def canonical_args_digest(args: dict[str, Any] | None) -> str:
    """Return a SHA-256 hex digest of tool arguments with stable key ordering."""
    payload = json.dumps(args or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resume_preimage(
    *,
    thread_id: str,
    node_name: str,
    tool_name: str,
    tool_call_id: str,
    args_digest: str,
) -> str:
    return "|".join((thread_id, node_name, tool_name, tool_call_id, args_digest))


def compute_resume_token(
    *,
    thread_id: str,
    node_name: str,
    tool_name: str,
    tool_call_id: str,
    args_digest: str,
    secret: str | None = None,
) -> str:
    """Derive a deterministic resume token for interrupt replay safety.

    LangGraph re-executes the node from the top on resume, so the token must
    be stable across that re-execution. When `secret` is set, the token is an
    HMAC-SHA-256 and is not forgeable without the secret.
    """
    preimage = _resume_preimage(
        thread_id=thread_id,
        node_name=node_name,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        args_digest=args_digest,
    )
    if secret is not None:
        return hmac.new(
            secret.encode("utf-8"),
            preimage.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()


def _thread_id_from_config(config: RunnableConfig | None) -> str:
    if not config:
        return ""
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id is not None else ""


def _node_name_from_request(request: ToolCallRequest) -> str:
    runtime = request.runtime
    if runtime is None:
        return ""
    config = getattr(runtime, "config", None) or {}
    metadata = config.get("metadata") or {}
    for key in ("langgraph_node", "checkpoint_ns"):
        value = metadata.get(key)
        if value:
            return str(value)
    return str(getattr(runtime, "tool_call_id", "") or "")


@dataclass
class PendingApproval:
    """Durable pending decision record for one gated tool call."""

    thread_id: str
    node_name: str
    tool_name: str
    tool_call_id: str
    args_digest: str
    policy_result: PolicyResult
    decision_shape: DecisionShape
    resume_token: str = ""
    terminal_state: TerminalState | None = None
    approved_args_digest: str | None = None
    supersedes_tool_call_id: str | None = None
    approved_tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if not self.resume_token:
            self.resume_token = compute_resume_token(
                thread_id=self.thread_id,
                node_name=self.node_name,
                tool_name=self.tool_name,
                tool_call_id=self.tool_call_id,
                args_digest=self.args_digest,
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingApproval:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class ApprovalDecision:
    """Structured human decision that must bind to a `PendingApproval`."""

    token: str
    tool_call_id: str
    action: DecisionAction
    edited_args: dict[str, Any] | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalDecision:
        return cls(
            token=str(data.get("token", "")),
            tool_call_id=str(data.get("tool_call_id", "")),
            action=cast("DecisionAction", data.get("action", "")),
            edited_args=data.get("edited_args"),
            message=data.get("message"),
        )


@dataclass
class _ApprovalPolicy:
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    decision_shape: DecisionShape = "approve_reject_or_edit"
    secret: str | None = None
    deny_on_error: bool = True

    def classify(self, tool_name: str) -> PolicyResult:
        if _match_patterns(tool_name, self.deny):
            return "deny"
        if _match_patterns(tool_name, self.allow):
            return "allow"
        return "requires_approval"


def _match_patterns(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


class ApprovalValidationError(ValueError):
    """Raised when a resume decision does not resolve the pending record."""


def validate_decision(
    pending: PendingApproval,
    decision: ApprovalDecision | dict[str, Any],
) -> ApprovalDecision:
    """Validate a resume decision against a pending approval record.

    Fail-closed: any mismatch or malformed payload raises
    `ApprovalValidationError` without revealing the expected token.
    """
    if not isinstance(decision, ApprovalDecision):
        if not isinstance(decision, dict):
            msg = "decision must be an ApprovalDecision or dict"
            raise ApprovalValidationError(msg)
        decision = ApprovalDecision.from_dict(decision)

    if pending.terminal_state is not None:
        msg = "approval request already resolved"
        raise ApprovalValidationError(msg)

    if not decision.token or not hmac.compare_digest(
        decision.token, pending.resume_token
    ):
        msg = "resume token mismatch"
        raise ApprovalValidationError(msg)

    if decision.tool_call_id != pending.tool_call_id:
        msg = "tool_call_id mismatch"
        raise ApprovalValidationError(msg)

    if decision.action not in {"approve", "reject", "edit", "defer"}:
        msg = "invalid decision action"
        raise ApprovalValidationError(msg)

    if decision.action == "edit" and pending.decision_shape == "approve_or_reject":
        msg = "edit not allowed for this decision_shape"
        raise ApprovalValidationError(msg)

    if decision.action == "edit" and not decision.edited_args:
        msg = "edited_args required when action is edit"
        raise ApprovalValidationError(msg)

    return decision


def _denial_message(
    request: ToolCallRequest,
    *,
    content: str,
    pending: PendingApproval | None = None,
    terminal_state: TerminalState = "rejected",
) -> ToolMessage:
    kwargs: dict[str, Any] = {}
    if pending is not None:
        record = PendingApproval(
            **{**pending.to_dict(), "terminal_state": terminal_state}
        )
        kwargs[_AUDIT_KW] = record.to_dict()
    return ToolMessage(
        content=content,
        name=request.tool_call["name"],
        tool_call_id=request.tool_call["id"],
        status="error",
        additional_kwargs=kwargs,
    )


def _stamp_audit(
    result: ToolMessage | Command,
    pending: PendingApproval,
) -> ToolMessage | Command:
    if isinstance(result, Command):
        return result
    audit = dict(result.additional_kwargs or {})
    audit[_AUDIT_KW] = pending.to_dict()
    result.additional_kwargs = audit
    return result


def _build_pending(
    request: ToolCallRequest,
    policy: _ApprovalPolicy,
) -> PendingApproval:
    tool_call = request.tool_call
    config = getattr(request.runtime, "config", None) if request.runtime else None
    thread_id = _thread_id_from_config(config)
    node_name = _node_name_from_request(request)
    tool_name = str(tool_call["name"])
    tool_call_id = str(tool_call["id"])
    args_digest = canonical_args_digest(tool_call.get("args"))
    return PendingApproval(
        thread_id=thread_id,
        node_name=node_name,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        args_digest=args_digest,
        policy_result="requires_approval",
        decision_shape=policy.decision_shape,
        resume_token=compute_resume_token(
            thread_id=thread_id,
            node_name=node_name,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args_digest=args_digest,
            secret=policy.secret,
        ),
    )


def _apply_decision(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage | Command],
    pending: PendingApproval,
    raw_decision: Any,
    *,
    deny_on_error: bool,
) -> ToolMessage | Command:
    try:
        decision = validate_decision(pending, raw_decision)
    except ApprovalValidationError as exc:
        if deny_on_error:
            return _denial_message(
                request,
                content=_VALIDATION_FAIL_CONTENT.format(reason=str(exc)),
                pending=pending,
                terminal_state="cancelled",
            )
        raise

    if decision.action == "defer":
        pending.terminal_state = None
        # Re-enter interrupt so the call stays pending with no execution.
        raw_decision = interrupt(pending.to_dict())
        return _apply_decision(
            request,
            execute,
            pending,
            raw_decision,
            deny_on_error=deny_on_error,
        )

    if decision.action == "reject":
        pending.terminal_state = "rejected"
        pending.approved_tool_call_id = None
        content = decision.message or _REJECT_CONTENT
        return _denial_message(
            request,
            content=content,
            pending=pending,
            terminal_state="rejected",
        )

    if decision.action == "edit":
        edited_args = cast("dict[str, Any]", decision.edited_args)
        approved_digest = canonical_args_digest(edited_args)
        pending.terminal_state = "edited"
        pending.approved_args_digest = approved_digest
        pending.supersedes_tool_call_id = pending.tool_call_id
        pending.approved_tool_call_id = pending.tool_call_id
        modified_call = {
            **request.tool_call,
            "args": edited_args,
        }
        result = execute(request.override(tool_call=modified_call))
        pending.terminal_state = "executed"
        return _stamp_audit(result, pending)

    # approve
    pending.terminal_state = "approved"
    pending.approved_args_digest = pending.args_digest
    pending.approved_tool_call_id = pending.tool_call_id
    result = execute(request)
    pending.terminal_state = "executed"
    return _stamp_audit(result, pending)


async def _aapply_decision(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    pending: PendingApproval,
    raw_decision: Any,
    *,
    deny_on_error: bool,
) -> ToolMessage | Command:
    try:
        decision = validate_decision(pending, raw_decision)
    except ApprovalValidationError as exc:
        if deny_on_error:
            return _denial_message(
                request,
                content=_VALIDATION_FAIL_CONTENT.format(reason=str(exc)),
                pending=pending,
                terminal_state="cancelled",
            )
        raise

    if decision.action == "defer":
        pending.terminal_state = None
        raw_decision = interrupt(pending.to_dict())
        return await _aapply_decision(
            request,
            execute,
            pending,
            raw_decision,
            deny_on_error=deny_on_error,
        )

    if decision.action == "reject":
        pending.terminal_state = "rejected"
        pending.approved_tool_call_id = None
        content = decision.message or _REJECT_CONTENT
        return _denial_message(
            request,
            content=content,
            pending=pending,
            terminal_state="rejected",
        )

    if decision.action == "edit":
        edited_args = cast("dict[str, Any]", decision.edited_args)
        approved_digest = canonical_args_digest(edited_args)
        pending.terminal_state = "edited"
        pending.approved_args_digest = approved_digest
        pending.supersedes_tool_call_id = pending.tool_call_id
        pending.approved_tool_call_id = pending.tool_call_id
        modified_call = {
            **request.tool_call,
            "args": edited_args,
        }
        result = await execute(request.override(tool_call=modified_call))
        pending.terminal_state = "executed"
        return _stamp_audit(result, pending)

    pending.terminal_state = "approved"
    pending.approved_args_digest = pending.args_digest
    pending.approved_tool_call_id = pending.tool_call_id
    result = await execute(request)
    pending.terminal_state = "executed"
    return _stamp_audit(result, pending)


def human_approval(
    allow: Sequence[str] | None = None,
    deny: Sequence[str] | None = None,
    *,
    decision_shape: DecisionShape = "approve_reject_or_edit",
    secret: str | None = None,
    deny_on_error: bool = True,
) -> ToolCallWrapper:
    """Create a sync tool-call wrapper that gates execution on human approval.

    Args:
        allow: Glob patterns for tools that execute without approval.
        deny: Glob patterns for tools that are blocked without interrupting.
            Deny is evaluated before allow.
        decision_shape: Whether reviewers may edit arguments, or only
            approve/reject.
        secret: Optional HMAC key for non-forgeable resume tokens. Required for
            production when interrupt payloads may be visible to untrusted
            clients. Token remains deterministic across node re-execution.
        deny_on_error: When `True` (default), invalid resume decisions fail
            closed with a denial `ToolMessage` instead of raising. Policy or
            validation failures never fall through to tool execution.

    Returns:
        A `ToolCallWrapper` suitable for `ToolNode(wrap_tool_call=...)`.

    Example:
        ```python
        from langgraph.prebuilt import ToolNode, human_approval

        wrapper = human_approval(allow=["read_*"], deny=["drop_*"], secret="...")
        node = ToolNode(tools, wrap_tool_call=wrapper)
        ```
    """
    policy = _ApprovalPolicy(
        allow=list(allow) if allow is not None else [],
        deny=list(deny) if deny is not None else [],
        decision_shape=decision_shape,
        secret=secret,
        deny_on_error=deny_on_error,
    )

    def wrapper(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        try:
            result = policy.classify(str(request.tool_call["name"]))
        except Exception:
            if policy.deny_on_error:
                return _denial_message(
                    request,
                    content="Approval policy evaluation failed; denying execution.",
                    terminal_state="cancelled",
                )
            raise

        if result == "deny":
            denied = _build_pending(request, policy)
            denied.policy_result = "deny"
            denied.terminal_state = "rejected"
            return _denial_message(
                request,
                content=_DENIAL_CONTENT,
                pending=denied,
                terminal_state="rejected",
            )

        if result == "allow":
            return execute(request)

        pending = _build_pending(request, policy)
        raw_decision = interrupt(pending.to_dict())
        return _apply_decision(
            request,
            execute,
            pending,
            raw_decision,
            deny_on_error=policy.deny_on_error,
        )

    return wrapper


def async_human_approval(
    allow: Sequence[str] | None = None,
    deny: Sequence[str] | None = None,
    *,
    decision_shape: DecisionShape = "approve_reject_or_edit",
    secret: str | None = None,
    deny_on_error: bool = True,
) -> AsyncToolCallWrapper:
    """Async variant of `human_approval` for `ToolNode(awrap_tool_call=...)`."""
    policy = _ApprovalPolicy(
        allow=list(allow) if allow is not None else [],
        deny=list(deny) if deny is not None else [],
        decision_shape=decision_shape,
        secret=secret,
        deny_on_error=deny_on_error,
    )

    async def wrapper(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        try:
            result = policy.classify(str(request.tool_call["name"]))
        except Exception:
            if policy.deny_on_error:
                return _denial_message(
                    request,
                    content="Approval policy evaluation failed; denying execution.",
                    terminal_state="cancelled",
                )
            raise

        if result == "deny":
            denied = _build_pending(request, policy)
            denied.policy_result = "deny"
            denied.terminal_state = "rejected"
            return _denial_message(
                request,
                content=_DENIAL_CONTENT,
                pending=denied,
                terminal_state="rejected",
            )

        if result == "allow":
            return await execute(request)

        pending = _build_pending(request, policy)
        raw_decision = interrupt(pending.to_dict())
        return await _aapply_decision(
            request,
            execute,
            pending,
            raw_decision,
            deny_on_error=policy.deny_on_error,
        )

    return wrapper


__all__ = [
    "ApprovalDecision",
    "ApprovalValidationError",
    "PendingApproval",
    "async_human_approval",
    "canonical_args_digest",
    "compute_resume_token",
    "human_approval",
    "validate_decision",
]
