"""human_approval — production-grade HITL ToolCallWrapper for ToolNode.

Factory that returns a :data:`ToolCallWrapper` (or :data:`AsyncToolCallWrapper`)
that intercepts every tool call, classifies it against a policy, and—for calls
that require human oversight—:

1. Creates a *durable* :class:`PendingApproval` record surfaced to the caller
   via :func:`~langgraph.types.interrupt`.
2. Waits for a ``Command(resume=<ApprovalDecision dict>)`` whose ``token`` is
   cryptographically bound to that exact record.
3. Validates the decision, marks the record terminal, then either executes the
   tool (with original or edited args) or returns a denial
   :class:`~langchain_core.messages.ToolMessage`.

The record is intentionally self-contained: it can be serialised to JSON,
stored in a database, and round-tripped through any LangGraph checkpointer
without loss of information.

Usage::

    from langgraph.prebuilt import ToolNode, human_approval

    wrapper = human_approval(
        allow=["read_*", "list_*"],
        deny=["drop_*"],
        # everything else → requires_approval
    )
    node = ToolNode(tools, wrap_tool_call=wrapper)

Resume protocol::

    # The interrupt value is the PendingApproval serialised to a dict.
    # Echo the token and tool_call_id unchanged to bind the decision.
    from langgraph.prebuilt.human_approval import ApprovalDecision
    from langgraph.types import Command

    graph.invoke(
        Command(resume=ApprovalDecision(
            token=pending["resume_token"],
            tool_call_id=pending["tool_call_id"],
            action="approve",           # or "reject" / "edit"
            edited_args={"key": "val"}, # required when action="edit"
        ).to_dict()),
        config,
    )
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

from langchain_core.messages import ToolMessage

from langgraph.prebuilt.tool_node import AsyncToolCallWrapper, ToolCallRequest, ToolCallWrapper
from langgraph.types import Command, interrupt

__all__ = [
    "human_approval",
    "async_human_approval",
    "PendingApproval",
    "ApprovalDecision",
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PolicyResult = Literal["allow", "deny", "requires_approval"]
TerminalState = Literal["approved", "rejected", "edited", "expired", "cancelled", "executed"]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PendingApproval:
    """Durable record created for every tool call that requires human sign-off.

    The record is surfaced via :func:`~langgraph.types.interrupt` and saved by
    the graph checkpointer so it survives reconnects and process restarts.

    Attributes:
        thread_id: LangGraph thread identifier (from ``config["configurable"]``).
        node_name: Name of the :class:`~langgraph.prebuilt.ToolNode` that owns
            this wrapper (for audit purposes).
        tool_name: Name of the tool being gated.
        tool_call_id: Unique identifier of this specific tool call (from the
            LLM message), used to prevent cross-call approval.
        args_digest: SHA-256 hex digest of the canonical JSON serialisation of
            the *original* tool arguments (keys sorted, no extra whitespace).
        policy_result: Classification produced before the interrupt was raised.
            Always ``"requires_approval"`` here; included for auditability.
        decision_shape: Which actions the human may take.
        resume_token: Hex digest that binds an :class:`ApprovalDecision` back
            to this exact record.  Derived deterministically from
            ``(thread_id, node_name, tool_name, tool_call_id, args_digest)``
            so that it is stable across node re-executions (LangGraph replays
            the node from the top when resuming after an interrupt).
        terminal_state: Set once the decision has been processed. ``None``
            while still pending.
        approved_args_digest: SHA-256 hex digest of the *edited* args when
            ``terminal_state == "edited"``; ``None`` otherwise.
    """

    thread_id: str
    node_name: str
    tool_name: str
    tool_call_id: str
    args_digest: str
    policy_result: PolicyResult
    decision_shape: Literal["approve_or_reject", "approve_reject_or_edit"]
    resume_token: str = field(default="")
    terminal_state: Optional[TerminalState] = None
    approved_args_digest: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.resume_token:
            # Derive deterministically so the same token is reproduced when
            # the node re-runs after the graph resumes from interrupt().
            key = ":".join([
                self.thread_id, self.node_name, self.tool_name,
                self.tool_call_id, self.args_digest,
            ])
            self.resume_token = hashlib.sha256(key.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON / checkpointer storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PendingApproval":
        """Reconstruct from a :meth:`to_dict` payload."""
        return cls(**d)


@dataclass
class ApprovalDecision:
    """Human decision returned via ``Command(resume=ApprovalDecision(...).to_dict())``.

    Attributes:
        token: Must equal :attr:`PendingApproval.resume_token`.
        tool_call_id: Must equal :attr:`PendingApproval.tool_call_id`.
            Prevents a captured approval from being replayed against a
            different tool call (cross-call replay prevention).
        action: ``"approve"`` | ``"reject"`` | ``"edit"``.
        edited_args: Required (and non-``None``) when ``action == "edit"``.
    """

    token: str
    tool_call_id: str
    action: Literal["approve", "reject", "edit"]
    edited_args: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ApprovalDecision":
        return cls(**d)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonical_digest(args: dict) -> str:
    """Return the SHA-256 hex digest of the canonical JSON of *args*.

    Keys are sorted and no extra whitespace is emitted, so the digest is
    independent of insertion order and formatting.
    """
    canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _match_patterns(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _classify(
    tool_name: str,
    allow_patterns: list[str],
    deny_patterns: list[str],
) -> PolicyResult:
    """Evaluate deny-list → allow-list → requires_approval."""
    if _match_patterns(tool_name, deny_patterns):
        return "deny"
    if _match_patterns(tool_name, allow_patterns):
        return "allow"
    return "requires_approval"


def _denial_message(tool_call_id: str, tool_name: str) -> ToolMessage:
    return ToolMessage(
        content=f"Tool call '{tool_name}' was denied by policy.",
        tool_call_id=tool_call_id,
        status="error",
    )


def _rejection_message(tool_call_id: str, tool_name: str) -> ToolMessage:
    return ToolMessage(
        content=f"Tool call '{tool_name}' was rejected by the operator.",
        tool_call_id=tool_call_id,
        status="error",
    )


def _validate_decision(decision: ApprovalDecision, pending: PendingApproval) -> None:
    """Raise :exc:`ValueError` if *decision* does not match *pending*.

    Checks performed in order:

    1. ``token`` must equal ``pending.resume_token``.
    2. ``tool_call_id`` must equal ``pending.tool_call_id``.
    3. ``pending.terminal_state`` must be ``None`` (not already resolved).
    4. When ``action == "edit"``, ``edited_args`` must be non-empty.
    """
    if decision.token != pending.resume_token:
        raise ValueError(
            f"Resume token mismatch for tool '{pending.tool_name}': "
            f"expected '{pending.resume_token}', got '{decision.token}'. "
            "This decision does not correspond to the pending approval record."
        )
    if decision.tool_call_id != pending.tool_call_id:
        raise ValueError(
            f"tool_call_id mismatch: expected '{pending.tool_call_id}', "
            f"got '{decision.tool_call_id}'. A captured approval cannot be "
            "replayed against a different tool call."
        )
    if pending.terminal_state is not None:
        raise ValueError(
            f"Cannot resume: approval for tool '{pending.tool_name}' "
            f"(tool_call_id='{pending.tool_call_id}') is already in "
            f"terminal state '{pending.terminal_state}'."
        )
    if decision.action == "edit" and not decision.edited_args:
        raise ValueError(
            "ApprovalDecision.edited_args must be non-empty when action='edit'."
        )


def _thread_id_from_config(config: dict) -> str:
    return (config or {}).get("configurable", {}).get("thread_id", "")


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def human_approval(
    *,
    allow: list[str] | None = None,
    deny: list[str] | None = None,
    decision_shape: Literal[
        "approve_or_reject", "approve_reject_or_edit"
    ] = "approve_reject_or_edit",
    node_name: str = "tools",
) -> ToolCallWrapper:
    """Return a :data:`~langgraph.prebuilt.tool_node.ToolCallWrapper` that
    gates tool execution on human approval.

    Each incoming tool call is classified against *deny* and *allow* glob
    patterns (evaluated in that order).  Calls that match neither pattern
    require an explicit human decision before the tool runs.

    Args:
        allow: Glob patterns for tool names that execute automatically without
            interruption.  Example: ``["read_*", "list_*"]``.
        deny: Glob patterns for tool names that are always blocked immediately.
            Example: ``["drop_*", "delete_*"]``.
        decision_shape: Which actions the human may take on a pending call.
            ``"approve_or_reject"`` or ``"approve_reject_or_edit"`` (default).
        node_name: Descriptive name of the owning
            :class:`~langgraph.prebuilt.ToolNode`; stored in the
            :class:`PendingApproval` record for audit purposes.

    Returns:
        A synchronous :data:`~langgraph.prebuilt.tool_node.ToolCallWrapper`
        suitable for ``ToolNode(wrap_tool_call=...)``.

    Example::

        wrapper = human_approval(allow=["search_*"], deny=["rm_*"])
        node = ToolNode(tools, wrap_tool_call=wrapper)
    """
    allow_patterns: list[str] = allow or []
    deny_patterns: list[str] = deny or []

    def _wrapper(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_call = request.tool_call
        tool_name: str = tool_call["name"]
        tool_call_id: str = tool_call["id"]
        args: dict = tool_call.get("args", {})
        config: dict = request.runtime.config or {}

        policy = _classify(tool_name, allow_patterns, deny_patterns)

        # ── allow: execute immediately, no pending record ──────────────────
        if policy == "allow":
            return execute(request)

        # ── deny: terminal rejection without interrupting ──────────────────
        if policy == "deny":
            return _denial_message(tool_call_id, tool_name)

        # ── requires_approval: create durable record, interrupt ────────────
        pending = PendingApproval(
            thread_id=_thread_id_from_config(config),
            node_name=node_name,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args_digest=_canonical_digest(args),
            policy_result="requires_approval",
            decision_shape=decision_shape,
        )

        # interrupt() raises GraphInterrupt (a GraphBubbleUp); it is
        # re-raised by ToolNode's error handler because _default_handle_tool_errors
        # only handles ToolInvocationError.  On resume the return value is
        # whatever the caller passed in Command(resume=...).
        raw_decision = interrupt(pending.to_dict())

        if isinstance(raw_decision, dict):
            decision = ApprovalDecision.from_dict(raw_decision)
        elif isinstance(raw_decision, ApprovalDecision):
            decision = raw_decision
        else:
            raise TypeError(
                f"Expected ApprovalDecision or dict, got {type(raw_decision).__name__}"
            )

        _validate_decision(decision, pending)

        # ── reject ─────────────────────────────────────────────────────────
        if decision.action == "reject":
            pending.terminal_state = "rejected"
            return _rejection_message(tool_call_id, tool_name)

        # ── edit: rebind to operator-supplied args ─────────────────────────
        if decision.action == "edit":
            edited_args: dict = decision.edited_args  # type: ignore[assignment]
            pending.approved_args_digest = _canonical_digest(edited_args)
            pending.terminal_state = "edited"
            modified_request = request.override(
                tool_call={**tool_call, "args": edited_args}
            )
            result = execute(modified_request)
            if isinstance(result, ToolMessage):
                result.additional_kwargs["approved_args_digest"] = pending.approved_args_digest
                result.additional_kwargs["args_digest"] = pending.args_digest
            return result

        # ── approve: execute with original args ────────────────────────────
        pending.terminal_state = "approved"
        result = execute(request)
        if isinstance(result, ToolMessage):
            result.additional_kwargs["args_digest"] = pending.args_digest
        return result

    return _wrapper


def async_human_approval(
    *,
    allow: list[str] | None = None,
    deny: list[str] | None = None,
    decision_shape: Literal[
        "approve_or_reject", "approve_reject_or_edit"
    ] = "approve_reject_or_edit",
    node_name: str = "tools",
) -> AsyncToolCallWrapper:
    """Async variant of :func:`human_approval`.

    Returns an :data:`~langgraph.prebuilt.tool_node.AsyncToolCallWrapper` for
    use with async graphs.  Shares identical approval logic.

    Example::

        wrapper = async_human_approval(allow=["search_*"], deny=["rm_*"])
        node = ToolNode(tools, awrap_tool_call=wrapper)
    """
    allow_patterns: list[str] = allow or []
    deny_patterns: list[str] = deny or []

    async def _awrapper(
        request: ToolCallRequest,
        execute: Callable,
    ) -> ToolMessage | Command:
        tool_call = request.tool_call
        tool_name: str = tool_call["name"]
        tool_call_id: str = tool_call["id"]
        args: dict = tool_call.get("args", {})
        config: dict = request.runtime.config or {}

        policy = _classify(tool_name, allow_patterns, deny_patterns)

        if policy == "allow":
            return await execute(request)

        if policy == "deny":
            return _denial_message(tool_call_id, tool_name)

        pending = PendingApproval(
            thread_id=_thread_id_from_config(config),
            node_name=node_name,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args_digest=_canonical_digest(args),
            policy_result="requires_approval",
            decision_shape=decision_shape,
        )

        raw_decision = interrupt(pending.to_dict())

        if isinstance(raw_decision, dict):
            decision = ApprovalDecision.from_dict(raw_decision)
        elif isinstance(raw_decision, ApprovalDecision):
            decision = raw_decision
        else:
            raise TypeError(
                f"Expected ApprovalDecision or dict, got {type(raw_decision).__name__}"
            )

        _validate_decision(decision, pending)

        if decision.action == "reject":
            pending.terminal_state = "rejected"
            return _rejection_message(tool_call_id, tool_name)

        if decision.action == "edit":
            edited_args = decision.edited_args  # type: ignore[assignment]
            pending.approved_args_digest = _canonical_digest(edited_args)
            pending.terminal_state = "edited"
            modified_request = request.override(
                tool_call={**tool_call, "args": edited_args}
            )
            result = await execute(modified_request)
            if isinstance(result, ToolMessage):
                result.additional_kwargs["approved_args_digest"] = pending.approved_args_digest
                result.additional_kwargs["args_digest"] = pending.args_digest
            return result

        pending.terminal_state = "approved"
        result = await execute(request)
        if isinstance(result, ToolMessage):
            result.additional_kwargs["args_digest"] = pending.args_digest
        return result

    return _awrapper
