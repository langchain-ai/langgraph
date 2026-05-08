import hashlib
import logging
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

# Approved model registry
APPROVED_MODEL_REGISTRY = {
    "claude-3-5-sonnet-20241022": "anthropic",
}

APPROVED_MODEL_NAME = "claude-3-5-sonnet-20241022"
APPROVED_MODEL_PROVIDER = "anthropic"

# Explicit tool allow list (empty means no tools are permitted)
TOOL_ALLOW_LIST: list[str] = []

tools = []

# Validate tools against allow list
for tool in tools:
    tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
    if tool_name not in TOOL_ALLOW_LIST:
        raise ValueError(
            f"Tool '{tool_name}' is not in the approved tool allow list. "
            f"Approved tools: {TOOL_ALLOW_LIST}"
        )

# Use approved model from registry
if APPROVED_MODEL_NAME not in APPROVED_MODEL_REGISTRY:
    raise RuntimeError(
        f"Model '{APPROVED_MODEL_NAME}' is not in the approved model registry."
    )

model_approved = ChatAnthropic(model=APPROVED_MODEL_NAME, temperature=0)
model_approved = model_approved.bind_tools(tools)

# Dynamic code execution primitives to detect in LLM output
DANGEROUS_PATTERNS = [
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bsubprocess\s*\.",
    r"\bos\.system\s*\(",
    r"\bos\.popen\s*\(",
    r"\b__import__\s*\(",
    r"\bcompile\s*\(",
    r"\bexecfile\s*\(",
]

# Prompt injection / malicious command patterns
MALICIOUS_PROMPT_PATTERNS = [
    r"(?i)(ignore\s+previous\s+instructions)",
    r"(?i)(system\s*prompt)",
    r"(?i)(\bbase64\b.*decode)",
    r"(?i)(rm\s+-rf)",
    r"(?i)(chmod\s+[0-7]+)",
    r"(?i)(curl\s+.*\|.*sh)",
    r"(?i)(wget\s+.*\|.*sh)",
    r"(?i)(;\s*drop\s+table)",
    r"(?i)(\bpowershell\b)",
    r"(?i)(\bcmd\.exe\b)",
]

AI_CONTENT_WATERMARK = "[AI-GENERATED-CONTENT]"


def _check_dangerous_content(text: str, context: str = "output") -> None:
    """Check for dangerous dynamic code execution primitives."""
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, text):
            logger.warning(
                "Dangerous pattern detected in LLM %s: pattern=%s", context, pattern
            )
            raise ValueError(
                f"LLM {context} contains forbidden dynamic code execution primitive "
                f"matching pattern: {pattern}"
            )


def _check_malicious_prompt(text: str) -> None:
    """Check for malicious command patterns in input messages."""
    for pattern in MALICIOUS_PROMPT_PATTERNS:
        if re.search(pattern, text):
            logger.warning(
                "Malicious prompt pattern detected in input: pattern=%s", pattern
            )
            raise ValueError(
                f"Input message contains potentially malicious content "
                f"matching pattern: {pattern}"
            )


def _sanitize_message_content(content) -> str:
    """Extract and sanitize text content from a message."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)


def _validate_and_sanitize_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """Validate and sanitize input messages before sending to LLM."""
    if not messages:
        raise ValueError("Messages sequence must not be empty.")
    for msg in messages:
        content_text = _sanitize_message_content(msg.content)
        _check_malicious_prompt(content_text)
    return messages


def _validate_llm_output(response: BaseMessage) -> BaseMessage:
    """Validate LLM output for dangerous content."""
    content_text = _sanitize_message_content(response.content)
    _check_dangerous_content(content_text, context="output")
    return response


def _add_provenance_metadata(response: BaseMessage, model_name: str, trace_id: str) -> BaseMessage:
    """Add provenance metadata to the LLM response."""
    timestamp = datetime.now(timezone.utc).isoformat()
    provenance = {
        "ai_generated": True,
        "model_id": model_name,
        "model_provider": APPROVED_MODEL_PROVIDER,
        "timestamp_utc": timestamp,
        "trace_id": trace_id,
        "watermark": AI_CONTENT_WATERMARK,
    }
    if not hasattr(response, "additional_kwargs"):
        response.additional_kwargs = {}
    response.additional_kwargs["provenance"] = provenance
    return response


def _require_authenticated_user(config: dict) -> str:
    """Enforce authentication: extract and validate user identity from config."""
    configurable = config.get("configurable", {}) if config else {}
    user_id = configurable.get("user_id") or configurable.get("authenticated_user")
    if not user_id:
        raise PermissionError(
            "Authentication required: 'user_id' or 'authenticated_user' must be "
            "provided in config['configurable'] before accessing the AI agent."
        )
    return str(user_id)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state, config):
    # Enforce authentication
    user_id = _require_authenticated_user(config)

    model = model_approved
    messages = state["messages"]

    # Validate and sanitize input messages
    messages = _validate_and_sanitize_messages(messages)

    # Generate a trace/correlation ID for this invocation
    trace_id = str(uuid.uuid4())

    # Compute input hash for audit trail
    input_repr = repr([_sanitize_message_content(m.content) for m in messages])
    input_hash = hashlib.sha256(input_repr.encode()).hexdigest()

    # Log the LLM interaction (input)
    logger.info(
        "LLM_INTERACTION_START: trace_id=%s user_id=%s model=%s input_hash=%s "
        "message_count=%d timestamp=%s",
        trace_id,
        user_id,
        APPROVED_MODEL_NAME,
        input_hash,
        len(messages),
        datetime.now(timezone.utc).isoformat(),
    )

    response = model.invoke(messages)

    # Validate LLM output for dangerous content
    response = _validate_llm_output(response)

    # Add provenance metadata / watermark
    response = _add_provenance_metadata(response, APPROVED_MODEL_NAME, trace_id)

    # Compute output hash for audit trail
    output_text = _sanitize_message_content(response.content)
    output_hash = hashlib.sha256(output_text.encode()).hexdigest()

    # Log the LLM interaction (output / audit trail)
    logger.info(
        "LLM_INTERACTION_END: trace_id=%s user_id=%s model=%s output_hash=%s "
        "has_tool_calls=%s timestamp=%s",
        trace_id,
        user_id,
        APPROVED_MODEL_NAME,
        output_hash,
        bool(getattr(response, "tool_calls", None)),
        datetime.now(timezone.utc).isoformat(),
    )

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def _enforce_tool_allow_list(tool_calls):
    """Enforce that only allow-listed tools are invoked."""
    for tool_call in tool_calls:
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
        if tool_name not in TOOL_ALLOW_LIST:
            logger.warning(
                "TOOL_DENIED: tool_name=%s not in allow list=%s",
                tool_name,
                TOOL_ALLOW_LIST,
            )
            raise ValueError(
                f"Tool '{tool_name}' is not in the approved tool allow list. "
                f"Approved tools: {TOOL_ALLOW_LIST}"
            )
        logger.info("TOOL_ALLOWED: tool_name=%s", tool_name)


# Define the function to execute tools
tool_node = ToolNode(tools)


class ContextSchema(TypedDict):
    model: Literal["anthropic", "openai"]


# Define a new graph
workflow = StateGraph(AgentState, context_schema=ContextSchema)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()