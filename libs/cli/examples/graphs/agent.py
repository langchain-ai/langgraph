import base64
import hashlib
import logging
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict
from urllib.parse import urlparse

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

# ---------------------------------------------------------------------------
# Logging / audit configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("agent")

# ---------------------------------------------------------------------------
# Approved model registry
# ---------------------------------------------------------------------------
APPROVED_MODEL_REGISTRY = {
    # Replace the entries below with models from the organisation's approved list.
    # Example placeholder — update to real approved identifiers before use.
    "approved-model-v1": "approved-model-v1",
}

APPROVED_MODEL_ID = "approved-model-v1"

# ---------------------------------------------------------------------------
# Tool allow-list
# ---------------------------------------------------------------------------
ALLOWED_TOOL_NAMES: frozenset[str] = frozenset({"tavily_search_results_json"})

# ---------------------------------------------------------------------------
# URL allow-list for outbound HTTP
# ---------------------------------------------------------------------------
ALLOWED_URL_HOSTNAMES: frozenset[str] = frozenset({
    "api.tavily.com",
})

# ---------------------------------------------------------------------------
# Dangerous patterns for input / output sanitisation
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS = re.compile(
    r"\b(eval|exec|subprocess|os\.system|__import__|compile|execfile"
    r"|open\s*\(|shutil|pickle\.loads|base64\.b64decode)\b",
    re.IGNORECASE,
)

_SHELL_COMMAND_PATTERNS = re.compile(
    r"(;\s*rm\s|&&\s*rm\s|\|\s*sh\b|\|\s*bash\b|`[^`]+`|\$\([^)]+\)"
    r"|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4})",
    re.IGNORECASE,
)

_MAX_MESSAGE_LENGTH = 32_000  # characters


def _check_for_dangerous_content(text: str, context: str) -> None:
    """Raise ValueError if dangerous patterns are detected."""
    if _DANGEROUS_PATTERNS.search(text):
        logger.warning("Dangerous pattern detected in %s", context)
        raise ValueError(f"Blocked: dangerous content detected in {context}")
    if _SHELL_COMMAND_PATTERNS.search(text):
        logger.warning("Shell command pattern detected in %s", context)
        raise ValueError(f"Blocked: shell command pattern detected in {context}")
    # Detect base64-encoded payloads heuristically
    b64_candidates = re.findall(r"[A-Za-z0-9+/]{40,}={0,2}", text)
    for candidate in b64_candidates:
        try:
            decoded = base64.b64decode(candidate).decode("utf-8", errors="ignore")
            if _DANGEROUS_PATTERNS.search(decoded) or _SHELL_COMMAND_PATTERNS.search(decoded):
                logger.warning("Base64-encoded dangerous content detected in %s", context)
                raise ValueError(
                    f"Blocked: base64-encoded dangerous content detected in {context}"
                )
        except Exception:
            pass


def _sanitize_message_content(content) -> str:
    """Return sanitised string content from a message content field."""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        text = " ".join(parts)
    else:
        text = str(content) if content is not None else ""
    if len(text) > _MAX_MESSAGE_LENGTH:
        logger.warning("Message content truncated from %d to %d chars", len(text), _MAX_MESSAGE_LENGTH)
        text = text[:_MAX_MESSAGE_LENGTH]
    return text


def validate_and_sanitize_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Validate and sanitise all input messages before sending to the model."""
    sanitized: list[BaseMessage] = []
    for msg in messages:
        text = _sanitize_message_content(msg.content)
        _check_for_dangerous_content(text, context="input message")
        # Reconstruct message with sanitised content
        sanitized.append(msg.__class__(content=text, **{
            k: v for k, v in msg.__dict__.items()
            if k not in ("content", "type")
        }))
    return sanitized


def validate_and_sanitize_response(response: BaseMessage, model_id: str) -> AIMessage:
    """Validate LLM output and attach provenance metadata."""
    text = _sanitize_message_content(response.content)
    _check_for_dangerous_content(text, context="LLM output")

    # Attach provenance / watermark metadata
    provenance = {
        "ai_generated": True,
        "model_id": model_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
    }
    additional_kwargs = dict(getattr(response, "additional_kwargs", {}) or {})
    additional_kwargs["provenance"] = provenance

    return AIMessage(content=text, additional_kwargs=additional_kwargs)


def check_tool_allowlist(tool_calls) -> None:
    """Raise if any requested tool is not on the allow-list."""
    for tc in tool_calls or []:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if name not in ALLOWED_TOOL_NAMES:
            logger.warning("Blocked disallowed tool invocation: %s", name)
            raise ValueError(f"Tool '{name}' is not on the approved tool allow-list.")


def check_url_allowlist(url: str) -> None:
    """Raise if the URL hostname is not on the allow-list."""
    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_URL_HOSTNAMES:
        logger.warning("Blocked outbound request to disallowed host: %s", parsed.hostname)
        raise ValueError(f"Outbound URL '{url}' is not on the approved allow-list.")


def require_authenticated_user(runtime: "Runtime") -> None:
    """Raise if the runtime context does not carry a verified user identity."""
    context = runtime.context if hasattr(runtime, "context") else {}
    user_id = context.get("user_id") if isinstance(context, dict) else getattr(context, "user_id", None)
    if not user_id:
        logger.error("Unauthenticated access attempt blocked.")
        raise PermissionError("Authentication required: no authenticated user identity found in runtime context.")
    logger.info("Authenticated user: %s", user_id)


# ---------------------------------------------------------------------------
# Approved model loader — raises if model is not in registry
# ---------------------------------------------------------------------------

def _load_approved_model(model_id: str):
    """Return a model instance only if model_id is in the approved registry."""
    if model_id not in APPROVED_MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_id}' is not in the organisation's approved model registry. "
            f"Approved models: {sorted(APPROVED_MODEL_REGISTRY.keys())}"
        )
    # NOTE: Replace the stub below with the actual approved model instantiation
    # once the organisation's registry is populated with real approved models.
    # Example:
    #   from langchain_openai import ChatOpenAI
    #   return ChatOpenAI(model=APPROVED_MODEL_REGISTRY[model_id], temperature=0)
    raise NotImplementedError(
        f"Model '{model_id}' is registered but no concrete instantiation is configured. "
        "Update _load_approved_model() with the approved provider and model name."
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
tools = [TavilySearchResults(max_results=1)]

# ---------------------------------------------------------------------------
# State / context schemas
# ---------------------------------------------------------------------------

class AgentContext(TypedDict):
    model: Literal["approved"]
    user_id: str  # required for authentication


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
    trace_id = str(uuid.uuid4())

    # 1. Authentication check
    require_authenticated_user(runtime)

    # 2. Resolve approved model
    model_id = APPROVED_MODEL_ID
    try:
        model = _load_approved_model(model_id)
        model = model.bind_tools([t for t in tools if t.name in ALLOWED_TOOL_NAMES])
    except NotImplementedError as exc:
        logger.error("[trace=%s] Approved model not configured: %s", trace_id, exc)
        raise

    # 3. Validate and sanitise input messages
    raw_messages = list(state["messages"])
    try:
        messages = validate_and_sanitize_messages(raw_messages)
    except ValueError as exc:
        logger.error("[trace=%s] Input validation failed: %s", trace_id, exc)
        raise

    # 4. Log the interaction (input)
    input_hash = hashlib.sha256(
        " ".join(_sanitize_message_content(m.content) for m in messages).encode()
    ).hexdigest()
    logger.info(
        "[trace=%s] LLM invocation | model=%s | input_hash=%s | msg_count=%d | timestamp=%s",
        trace_id,
        model_id,
        input_hash,
        len(messages),
        datetime.now(timezone.utc).isoformat(),
    )

    # 5. Invoke the model
    response = model.invoke(messages)

    # 6. Validate and sanitise output; attach provenance watermark
    try:
        response = validate_and_sanitize_response(response, model_id)
    except ValueError as exc:
        logger.error("[trace=%s] Output validation failed: %s", trace_id, exc)
        raise

    # 7. Check tool allow-list on requested tool calls
    try:
        check_tool_allowlist(getattr(response, "tool_calls", None))
    except ValueError as exc:
        logger.error("[trace=%s] Tool allow-list violation: %s", trace_id, exc)
        raise

    # 8. Log the interaction (output)
    output_hash = hashlib.sha256(
        _sanitize_message_content(response.content).encode()
    ).hexdigest()
    logger.info(
        "[trace=%s] LLM response | model=%s | output_hash=%s | timestamp=%s",
        trace_id,
        model_id,
        output_hash,
        datetime.now(timezone.utc).isoformat(),
    )

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Tool node with allow-list enforcement
# ---------------------------------------------------------------------------
tool_node = ToolNode([t for t in tools if t.name in ALLOWED_TOOL_NAMES])

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