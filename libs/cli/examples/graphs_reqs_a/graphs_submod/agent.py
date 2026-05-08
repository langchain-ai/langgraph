import hashlib
import logging
import re
import unicodedata
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from urllib.parse import urlparse

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

# ---------------------------------------------------------------------------
# Logging / audit configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approved model registry
# ---------------------------------------------------------------------------
APPROVED_MODEL_REGISTRY = {
    # organisation-approved models only; unapproved models are blocked
}

# ---------------------------------------------------------------------------
# Tool allow-list
# ---------------------------------------------------------------------------
ALLOWED_TOOL_NAMES = {"tavily_search_results_json"}

# ---------------------------------------------------------------------------
# URL allow-list for outbound HTTP
# ---------------------------------------------------------------------------
ALLOWED_URL_HOSTNAMES = {
    "api.tavily.com",
}

# ---------------------------------------------------------------------------
# Dangerous primitives that must not appear in LLM output
# ---------------------------------------------------------------------------
DANGEROUS_PATTERNS = re.compile(
    r"\b(eval|exec|execfile|compile|__import__|subprocess|os\.system"
    r"|os\.popen|popen|shell=True|importlib)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Prompt injection / malicious content patterns
# ---------------------------------------------------------------------------
MALICIOUS_PROMPT_PATTERNS = re.compile(
    r"(ignore previous instructions|disregard.*instructions"
    r"|you are now|act as|pretend you|system prompt"
    r"|<\s*script|javascript:|data:text/html"
    r"|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}"
    r"|\bbase64\b|\beval\b|\bexec\b)",
    re.IGNORECASE,
)

# Maximum number of messages forwarded to the model (data minimisation)
MAX_CONTEXT_MESSAGES = 20
# Maximum character length of a single message content string
MAX_MESSAGE_LENGTH = 8000

# Correlation / trace identifier for the current process lifetime
_PROCESS_TRACE_ID = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sanitize_text(text: str) -> str:
    """Remove non-printable / invisible Unicode characters and strip."""
    normalized = unicodedata.normalize("NFKC", text)
    sanitized = "".join(ch for ch in normalized if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t"))
    return sanitized.strip()


def _check_for_malicious_content(text: str, source: str) -> None:
    """Raise ValueError if text contains known-malicious patterns."""
    if MALICIOUS_PROMPT_PATTERNS.search(text):
        logger.warning("Malicious content detected in %s", source)
        raise ValueError(f"Malicious content detected in {source}")


def _check_for_dangerous_code(text: str, source: str) -> None:
    """Raise ValueError if text contains dynamic code execution primitives."""
    if DANGEROUS_PATTERNS.search(text):
        logger.warning("Dangerous code primitive detected in %s", source)
        raise ValueError(f"Dangerous code primitive detected in {source}")


def _validate_and_sanitize_prompt(raw: str, source: str) -> str:
    sanitized = _sanitize_text(raw)
    _check_for_malicious_content(sanitized, source)
    _check_for_dangerous_code(sanitized, source)
    return sanitized


def _validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("https",) and parsed.hostname in ALLOWED_URL_HOSTNAMES
    except Exception:
        return False


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _add_provenance(message: BaseMessage, model_id: str) -> BaseMessage:
    """Attach provenance metadata to an AI-generated message."""
    timestamp = datetime.now(timezone.utc).isoformat()
    provenance = {
        "ai_generated": True,
        "model_id": model_id,
        "timestamp": timestamp,
        "trace_id": _PROCESS_TRACE_ID,
        "content_hash": _hash_content(str(message.content)),
        "label": "[AI-GENERATED CONTENT]",
    }
    # Attach as additional_kwargs so downstream consumers can inspect it
    if hasattr(message, "additional_kwargs"):
        message.additional_kwargs["provenance"] = provenance
    return message


def _minimise_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Apply data minimisation: truncate history and cap message length."""
    truncated = list(messages[-MAX_CONTEXT_MESSAGES:])
    result = []
    for msg in truncated:
        if isinstance(msg.content, str) and len(msg.content) > MAX_MESSAGE_LENGTH:
            object.__setattr__(msg, "content", msg.content[:MAX_MESSAGE_LENGTH])
        result.append(msg)
    return result


def _validate_tool_calls(message: BaseMessage) -> None:
    """Ensure every tool call requested by the model is on the allow-list."""
    tool_calls = getattr(message, "tool_calls", None) or []
    for tc in tool_calls:
        tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if tool_name not in ALLOWED_TOOL_NAMES:
            logger.warning(
                "AUDIT | trace_id=%s | DENIED tool call | tool=%s | reason=not_in_allowlist",
                _PROCESS_TRACE_ID,
                tool_name,
            )
            raise ValueError(f"Tool '{tool_name}' is not on the approved tool allow-list.")


def _check_privilege_escalation(message: BaseMessage) -> None:
    """Detect unexpected tool invocations that may indicate privilege escalation."""
    tool_calls = getattr(message, "tool_calls", None) or []
    for tc in tool_calls:
        tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if tool_name not in ALLOWED_TOOL_NAMES:
            logger.error(
                "SECURITY | trace_id=%s | Privilege escalation attempt detected | tool=%s",
                _PROCESS_TRACE_ID,
                tool_name,
            )
            raise PermissionError(f"Privilege escalation attempt: tool '{tool_name}' is not permitted.")


# ---------------------------------------------------------------------------
# Approved model registry check
# ---------------------------------------------------------------------------

def _assert_model_approved(model_key: str) -> None:
    """Block any model not present in the organisation's approved registry."""
    if model_key not in APPROVED_MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_key}' is not in the organisation's approved model registry. "
            "Update APPROVED_MODEL_REGISTRY to include an approved model before use."
        )


# ---------------------------------------------------------------------------
# Prompt loading with sanitization
# ---------------------------------------------------------------------------

def _load_and_validate_prompt(path: Path, source_label: str) -> str:
    raw = path.read_text(encoding="utf-8")
    return _validate_and_sanitize_prompt(raw, source_label)


prompt = _load_and_validate_prompt(
    Path(__file__).parent.parent / "prompt.txt", "prompt.txt"
)
subprompt = _load_and_validate_prompt(
    Path(__file__).parent / "subprompt.txt", "subprompt.txt"
)

# ---------------------------------------------------------------------------
# Model instantiation — blocked until approved models are registered
# ---------------------------------------------------------------------------
# Both ChatAnthropic (claude-3-sonnet-20240229) and ChatOpenAI are NOT in the
# organisation's approved model registry.  Attempting to use them raises an
# error at call_model time.  Replace the entries in APPROVED_MODEL_REGISTRY
# with organisation-approved model identifiers and instantiate accordingly.

tools = [TavilySearchResults(max_results=1)]


# ---------------------------------------------------------------------------
# Graph state / context
# ---------------------------------------------------------------------------

class AgentContext(TypedDict):
    model: Literal["anthropic", "openai"]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


def call_model(state, runtime: Runtime[AgentContext]):
    model_key = runtime.context.get("model", "anthropic")

    # Policy: only approved models may be used
    _assert_model_approved(model_key)

    # Retrieve the approved model instance from the registry
    model = APPROVED_MODEL_REGISTRY[model_key]

    raw_messages = state["messages"]

    # Validate and sanitize each input message
    for msg in raw_messages:
        if isinstance(msg.content, str):
            _check_for_malicious_content(msg.content, f"input message ({type(msg).__name__})")
            _check_for_dangerous_code(msg.content, f"input message ({type(msg).__name__})")

    # Data minimisation: truncate context
    messages = _minimise_messages(raw_messages)

    # Audit log: input
    input_hash = _hash_content(str([str(m.content) for m in messages]))
    logger.info(
        "AUDIT | trace_id=%s | event=llm_input | model=%s | message_count=%d | input_hash=%s | timestamp=%s",
        _PROCESS_TRACE_ID,
        model_key,
        len(messages),
        input_hash,
        datetime.now(timezone.utc).isoformat(),
    )

    response = model.invoke(messages)

    # Validate LLM output for dangerous primitives
    if isinstance(response.content, str):
        _check_for_dangerous_code(response.content, "llm_response")
        _check_for_malicious_content(response.content, "llm_response")

    # Validate tool calls against allow-list and check for privilege escalation
    _validate_tool_calls(response)
    _check_privilege_escalation(response)

    # Attach provenance / watermark metadata
    response = _add_provenance(response, model_key)

    # Audit log: output
    output_hash = _hash_content(str(response.content))
    logger.info(
        "AUDIT | trace_id=%s | event=llm_output | model=%s | output_hash=%s | "
        "tool_calls=%s | timestamp=%s",
        _PROCESS_TRACE_ID,
        model_key,
        output_hash,
        [
            (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None))
            for tc in (getattr(response, "tool_calls", None) or [])
        ],
        datetime.now(timezone.utc).isoformat(),
    )

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Tool node with allow-list enforcement
# ---------------------------------------------------------------------------

class _AllowListedToolNode:
    """Wraps ToolNode and enforces the tool allow-list before execution."""

    def __init__(self, inner: ToolNode, allowed: set) -> None:
        self._inner = inner
        self._allowed = allowed

    def __call__(self, state):
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            tool_calls = getattr(last, "tool_calls", None) or []
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name not in self._allowed:
                    logger.warning(
                        "AUDIT | trace_id=%s | DENIED tool execution | tool=%s | reason=not_in_allowlist",
                        _PROCESS_TRACE_ID,
                        name,
                    )
                    raise ValueError(f"Tool '{name}' is not on the approved tool allow-list.")
        logger.info(
            "AUDIT | trace_id=%s | event=tool_execution | timestamp=%s",
            _PROCESS_TRACE_ID,
            datetime.now(timezone.utc).isoformat(),
        )
        return self._inner(state)


_raw_tool_node = ToolNode(tools)
tool_node = _AllowListedToolNode(_raw_tool_node, ALLOWED_TOOL_NAMES)

# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState, context_schema=AgentContext)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")

graph = workflow.compile()