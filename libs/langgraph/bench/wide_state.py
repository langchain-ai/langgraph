import operator
import hashlib
import hmac
import json
import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from random import choice
from typing import Annotated

# WARNING: langgraph is NOT on the organization's approved LLM/agent framework list.
# Usage of this framework must be reviewed and approved before production deployment.
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph

# Configure audit logger for forensic readiness
audit_logger = logging.getLogger("ai_audit")
audit_logger.setLevel(logging.INFO)
_audit_handler = logging.StreamHandler()
_audit_handler.setFormatter(
    logging.Formatter("%(asctime)s [AUDIT] %(message)s")
)
audit_logger.addHandler(_audit_handler)

# Session token signing secret (in production, load from a secure secrets manager)
_SESSION_SECRET = b"CHANGE_ME_USE_SECURE_SECRET_IN_PRODUCTION"
_SESSION_EXPIRY_SECONDS = 3600  # 1 hour

# Allowlist of permitted input value types for sanitization
_ALLOWED_SCALAR_TYPES = (str, int, float, bool, type(None))

# Patterns indicating potentially malicious content
_MALICIOUS_PATTERNS = [
    re.compile(r"(?i)(base64\s*,|data:.*base64)"),
    re.compile(r"(?i)(\beval\b|\bexec\b|\bos\.system\b|\bsubprocess\b)"),
    re.compile(r"(?i)(rm\s+-rf|chmod\s+|chown\s+|wget\s+|curl\s+.*\|)"),
    re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"),  # control characters
    re.compile(r"(?i)(\bignore previous instructions\b|\bforget your instructions\b)"),
]


def _sign_session_token(thread_id: str, issued_at: float, subject: str = "bench") -> str:
    """Create a signed session token with expiry and subject binding."""
    payload = f"{thread_id}:{issued_at}:{subject}"
    signature = hmac.new(_SESSION_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"


def _verify_session_token(token: str, subject: str = "bench") -> str:
    """Verify a signed session token and return the thread_id if valid."""
    parts = token.split(":")
    if len(parts) != 4:
        raise ValueError("Invalid session token format.")
    thread_id, issued_at_str, token_subject, signature = parts
    try:
        issued_at = float(issued_at_str)
    except ValueError:
        raise ValueError("Invalid session token: bad timestamp.")
    if token_subject != subject:
        raise ValueError("Invalid session token: subject mismatch.")
    now = time.time()
    if now - issued_at > _SESSION_EXPIRY_SECONDS:
        raise ValueError("Session token has expired.")
    expected_payload = f"{thread_id}:{issued_at_str}:{token_subject}"
    expected_sig = hmac.new(_SESSION_SECRET, expected_payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_sig):
        raise ValueError("Session token signature verification failed.")
    return thread_id


def _check_string_for_malicious_content(value: str) -> None:
    """Raise ValueError if the string contains potentially malicious content."""
    for pattern in _MALICIOUS_PATTERNS:
        if pattern.search(value):
            raise ValueError(
                f"Input validation failed: potentially malicious content detected matching pattern {pattern.pattern!r}."
            )


def _sanitize_and_validate_input(obj, _depth: int = 0) -> None:
    """
    Recursively validate and sanitize input to the AI model pipeline.
    Raises ValueError if any disallowed or malicious content is found.
    """
    if _depth > 50:
        raise ValueError("Input validation failed: input nesting depth exceeds limit.")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"Input validation failed: dict key must be str, got {type(k)}.")
            _check_string_for_malicious_content(k)
            _sanitize_and_validate_input(v, _depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _sanitize_and_validate_input(item, _depth + 1)
    elif isinstance(obj, str):
        _check_string_for_malicious_content(obj)
    elif not isinstance(obj, _ALLOWED_SCALAR_TYPES):
        raise ValueError(
            f"Input validation failed: disallowed type {type(obj)} in input."
        )


def _compute_input_hash(input_data: dict) -> str:
    """Compute a SHA-256 hash of the serialized input for audit purposes."""
    serialized = json.dumps(input_data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _audit_log(event: str, **kwargs) -> None:
    """Write an immutable audit log entry."""
    record = {
        "event": event,
        "timestamp": time.time(),
        "run_id": kwargs.get("run_id", ""),
        "thread_id": kwargs.get("thread_id", ""),
        "framework": "langgraph [NOT_IN_APPROVED_REGISTRY]",
        **{k: v for k, v in kwargs.items() if k not in ("run_id", "thread_id")},
    }
    audit_logger.info(json.dumps(record))


def wide_state(n: int) -> StateGraph:
    @dataclass(kw_only=True)
    class State:
        messages: Annotated[list, operator.add] = field(default_factory=list)
        trigger_events: Annotated[list, operator.add] = field(default_factory=list)
        """The external events that are converted by the graph."""
        primary_issue_medium: Annotated[str, lambda x, y: y or x] = field(
            default="email"
        )
        autoresponse: Annotated[dict | None, lambda _, y: y] = field(
            default=None
        )  # Always overwrite
        issue: Annotated[dict | None, lambda x, y: y if y else x] = field(default=None)
        relevant_rules: list[dict] | None = field(default=None)
        """SOPs fetched from the rulebook that are relevant to the current conversation."""
        memory_docs: list[dict] | None = field(default=None)
        """Memory docs fetched from the memory service that are relevant to the current conversation."""
        categorizations: Annotated[list[dict], operator.add] = field(
            default_factory=list
        )
        """The issue categorizations auto-generated by the AI."""
        responses: Annotated[list[dict], operator.add] = field(default_factory=list)
        """The draft responses recommended by the AI."""

        user_info: Annotated[dict | None, lambda x, y: y if y is not None else x] = (
            field(default=None)
        )
        """The current user state (by email)."""
        crm_info: Annotated[dict | None, lambda x, y: y if y is not None else x] = (
            field(default=None)
        )
        """The CRM information for organization the current user is from."""
        email_thread_id: Annotated[
            str | None, lambda x, y: y if y is not None else x
        ] = field(default=None)
        """The current email thread ID."""
        slack_participants: Annotated[dict, operator.or_] = field(default_factory=dict)
        """The growing list of current slack participants."""
        bot_id: str | None = field(default=None)
        """The ID of the bot user in the slack channel."""
        notified_assignees: Annotated[dict, operator.or_] = field(default_factory=dict)

    list_fields = {
        "messages",
        "trigger_events",
        "categorizations",
        "responses",
        "memory_docs",
        "relevant_rules",
    }
    dict_fields = {
        "user_info",
        "crm_info",
        "slack_participants",
        "notified_assignees",
        "autoresponse",
        "issue",
    }

    def read_write(read: str, write: Sequence[str], input: State) -> dict:
        val = getattr(input, read)
        val = {val: val} if isinstance(val, str) else val
        val_single = val[-1] if isinstance(val, list) else val
        val_list = val if isinstance(val, list) else [val]
        return {
            k: val_list
            if k in list_fields
            else val_single
            if k in dict_fields
            else "".join(choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
            for k in write
        }

    builder = StateGraph(State)
    builder.add_edge(START, "one")
    builder.add_node(
        "one",
        partial(read_write, "messages", ["trigger_events", "primary_issue_medium"]),
    )
    builder.add_edge("one", "two")
    builder.add_node(
        "two",
        partial(read_write, "trigger_events", ["autoresponse", "issue"]),
    )
    builder.add_edge("two", "three")
    builder.add_edge("two", "four")
    builder.add_node(
        "three",
        partial(read_write, "autoresponse", ["relevant_rules"]),
    )
    builder.add_node(
        "four",
        partial(
            read_write,
            "trigger_events",
            ["categorizations", "responses", "memory_docs"],
        ),
    )
    builder.add_node(
        "five",
        partial(
            read_write,
            "categorizations",
            [
                "user_info",
                "crm_info",
                "email_thread_id",
                "slack_participants",
                "bot_id",
                "notified_assignees",
            ],
        ),
    )
    builder.add_edge(["three", "four"], "five")
    builder.add_edge("five", "six")
    builder.add_node(
        "six",
        partial(read_write, "responses", ["messages"]),
    )
    builder.add_conditional_edges(
        "six", lambda state: END if len(state.messages) > n else "one"
    )

    return builder


if __name__ == "__main__":
    import asyncio

    import uvloop
    from langgraph.checkpoint.memory import InMemorySaver

    # WARNING: InMemorySaver is volatile and not forensic-ready.
    # In production, replace with a persistent, immutable checkpointer.
    graph = wide_state(1000).compile(checkpointer=InMemorySaver())
    input = {
        "messages": [
            {
                str(i) * 10: {
                    str(j) * 10: ["hi?" * 10, True, 1, 6327816386138, None] * 5
                    for j in range(50)
                }
                for i in range(50)
            }
        ]
    }

    # Validate and sanitize input before passing to the AI pipeline
    _sanitize_and_validate_input(input)

    # Create a signed, expiry-bound session token
    _raw_thread_id = str(uuid.uuid4())
    _issued_at = time.time()
    _signed_token = _sign_session_token(_raw_thread_id, _issued_at, subject="bench")
    # Verify the token and extract the thread_id
    _verified_thread_id = _verify_session_token(_signed_token, subject="bench")

    config = {"configurable": {"thread_id": _verified_thread_id}, "recursion_limit": 20000000000}

    # Compute input hash for audit trail
    _input_hash = _compute_input_hash(input)
    _run_id = str(uuid.uuid4())

    # Audit log: session and invocation start
    _audit_log(
        "ai_pipeline_invocation_start",
        run_id=_run_id,
        thread_id=_verified_thread_id,
        input_hash=_input_hash,
        framework="langgraph [NOT_IN_APPROVED_REGISTRY]",
        session_token_issued_at=_issued_at,
        session_expiry_seconds=_SESSION_EXPIRY_SECONDS,
    )

    async def run():
        chunk_count = 0
        try:
            async for c in graph.astream(input, config=config):
                chunk_keys = list(c.keys())
                chunk_count += 1
                # Audit log each decision/output chunk
                _audit_log(
                    "ai_pipeline_chunk",
                    run_id=_run_id,
                    thread_id=_verified_thread_id,
                    chunk_index=chunk_count,
                    chunk_keys=chunk_keys,
                )
                print(chunk_keys)
        except Exception as exc:
            _audit_log(
                "ai_pipeline_error",
                run_id=_run_id,
                thread_id=_verified_thread_id,
                error=str(exc),
            )
            raise
        finally:
            _audit_log(
                "ai_pipeline_invocation_end",
                run_id=_run_id,
                thread_id=_verified_thread_id,
                total_chunks=chunk_count,
            )

    uvloop.install()
    asyncio.run(run())