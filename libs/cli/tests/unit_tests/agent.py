import asyncio
import hashlib
import logging
import os
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph, add_messages

# check that env var is present
os.environ["SOME_ENV_VAR"]

# ---------------------------------------------------------------------------
# Logging / audit infrastructure
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Approved model registry
# ---------------------------------------------------------------------------
APPROVED_MODEL_REGISTRY = {
    "GenericFakeChatModel": {"version": "1.0.0", "provider": "langchain_core_test"},
}

MODEL_ID = "GenericFakeChatModel"
MODEL_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Approved tool allow list
# ---------------------------------------------------------------------------
APPROVED_TOOLS = {"tool"}

# ---------------------------------------------------------------------------
# Dangerous content patterns (input & output sanitization)
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS = re.compile(
    r"\b(eval|exec|subprocess|__import__|compile|open|os\.system"
    r"|os\.popen|importlib|pickle|marshal|base64)\b",
    re.IGNORECASE,
)

_SHELL_COMMAND_PATTERNS = re.compile(
    r"(&&|\|\||;|`|\$\(|>\s*/|<\s*/|/bin/|/usr/bin/|cmd\.exe|powershell)",
    re.IGNORECASE,
)


def _validate_registry(model_id: str) -> None:
    if model_id not in APPROVED_MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_id}' is not in the approved model registry. "
            "Only approved models may be used."
        )


def _sanitize_text(text: str, context: str = "input") -> str:
    if _DANGEROUS_PATTERNS.search(text):
        raise ValueError(
            f"Blocked {context}: contains forbidden dynamic code execution primitive."
        )
    if _SHELL_COMMAND_PATTERNS.search(text):
        raise ValueError(
            f"Blocked {context}: contains potential shell command injection."
        )
    return text


def _sanitize_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            _sanitize_text(content, context="input message")
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    _sanitize_text(str(part["text"]), context="input message part")
    return messages


def _sanitize_response(response: BaseMessage) -> BaseMessage:
    content = response.content
    if isinstance(content, str):
        _sanitize_text(content, context="LLM output")
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and "text" in part:
                _sanitize_text(str(part["text"]), context="LLM output part")
    return response


def _authenticate(config: dict) -> str:
    configurable = config.get("configurable", {}) if config else {}
    user_id = configurable.get("user_id") or configurable.get("auth_token")
    if not user_id:
        raise PermissionError(
            "Authentication required: no 'user_id' or 'auth_token' found in config. "
            "A user must authenticate before accessing the AI Agent."
        )
    return str(user_id)


def _audit_log(event: str, data: dict) -> None:
    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    logger.info("AUDIT: %s", record)


def _label_ai_output(response: BaseMessage, model_id: str, model_version: str, trace_id: str) -> AIMessage:
    content = response.content if isinstance(response.content, str) else str(response.content)
    provenance = (
        f"[AI-GENERATED | model={model_id} | version={model_version} "
        f"| trace_id={trace_id} | timestamp={datetime.now(timezone.utc).isoformat()}]"
    )
    labeled_content = f"{content}\n{provenance}"
    labeled = AIMessage(content=labeled_content)
    return labeled


# ---------------------------------------------------------------------------
# Registry validation at module load time
# ---------------------------------------------------------------------------
_validate_registry(MODEL_ID)


class AgentState(TypedDict):
    some_bytes: bytes
    some_byte_array: bytearray
    dict_with_bytes: dict[str, bytes]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sleep: int


async def call_model(state, config):
    # Authentication
    principal = _authenticate(config)

    trace_id = str(uuid.uuid4())

    if sleep := state.get("sleep"):
        await asyncio.sleep(sleep)

    messages = state["messages"]

    if len(messages) > 1:
        assert state["some_bytes"] == b"some_bytes"
        assert state["some_byte_array"] == bytearray(b"some_byte_array")
        assert state["dict_with_bytes"] == {"more_bytes": b"more_bytes"}

    # Sanitize and validate input messages
    _sanitize_messages(messages)

    # hacky way to reset model to the "first" response
    if isinstance(messages[-1], HumanMessage):
        model.i = 0

    # Compute input hash for audit trail
    input_repr = str([m.content for m in messages])
    input_hash = hashlib.sha256(input_repr.encode()).hexdigest()

    # Log LLM interaction (input)
    _audit_log("llm_invocation_start", {
        "trace_id": trace_id,
        "principal": principal,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "input_hash": input_hash,
        "message_count": len(messages),
    })

    response = await model.ainvoke(messages)

    # Validate and sanitize LLM output
    _sanitize_response(response)

    # Label / watermark AI-generated output with provenance metadata
    labeled_response = _label_ai_output(response, MODEL_ID, MODEL_VERSION, trace_id)

    output_hash = hashlib.sha256(str(labeled_response.content).encode()).hexdigest()

    # Log LLM interaction (output)
    _audit_log("llm_invocation_end", {
        "trace_id": trace_id,
        "principal": principal,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "output_hash": output_hash,
    })

    return {
        "messages": [labeled_response],
        "some_bytes": b"some_bytes",
        "some_byte_array": bytearray(b"some_byte_array"),
        "dict_with_bytes": {"more_bytes": b"more_bytes"},
    }


def call_tool(state):
    trace_id = str(uuid.uuid4())
    last_message = state["messages"][-1]
    last_message_content = last_message.content

    # Enforce tool allow list
    # Determine which tool is being invoked; here the node name is "tool"
    tool_name = "tool"
    if tool_name not in APPROVED_TOOLS:
        _audit_log("tool_invocation_denied", {
            "trace_id": trace_id,
            "tool_name": tool_name,
            "reason": "tool not in approved allow list",
        })
        raise ValueError(
            f"Tool '{tool_name}' is not in the approved tool allow list."
        )

    _audit_log("tool_invocation_start", {
        "trace_id": trace_id,
        "tool_name": tool_name,
        "input_content_hash": hashlib.sha256(str(last_message_content).encode()).hexdigest(),
    })

    result = ToolMessage(
        f"tool_call__{last_message_content}", tool_call_id="tool_call_id"
    )

    _audit_log("tool_invocation_end", {
        "trace_id": trace_id,
        "tool_name": tool_name,
        "output_content_hash": hashlib.sha256(str(result.content).encode()).hexdigest(),
    })

    return {"messages": [result]}


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # Strip provenance label before checking content
    content = last_message.content if isinstance(last_message.content, str) else str(last_message.content)
    # The original content is before the provenance label line
    original_content = content.split("\n[AI-GENERATED")[0].strip()
    if original_content == "end":
        return END
    else:
        next_node = "tool"
        if next_node not in APPROVED_TOOLS:
            raise ValueError(
                f"Attempted to route to unapproved tool node '{next_node}'."
            )
        return next_node


# NOTE: the model cycles through responses infinitely here
# Using GenericFakeChatModel as an approved test model substitute
model = GenericFakeChatModel(messages=iter([AIMessage(content="begin"), AIMessage(content="end")]))
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tool", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tool", "agent")

graph = workflow.compile()