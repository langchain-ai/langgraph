import base64
import hashlib
import logging
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Approved model registry
APPROVED_MODEL_REGISTRY = {
    "gpt-4o-2024-05-13": "openai",
}

# Approved tool allow list
APPROVED_TOOL_NAMES = {"tavily_search_results_json"}

# URL allowlist for outbound requests
ALLOWED_URL_PREFIXES = (
    "https://api.tavily.com/",
    "https://api.openai.com/",
)

# Dynamic code execution primitives to block in LLM output
DANGEROUS_PATTERNS = re.compile(
    r"\b(eval|exec|subprocess|os\.system|os\.popen|__import__|compile|execfile"
    r"|importlib|ctypes|pickle\.loads|marshal\.loads|open\s*\()\b",
    re.IGNORECASE,
)

# Prompt injection / malicious content patterns
MALICIOUS_PROMPT_PATTERNS = re.compile(
    r"(ignore previous instructions|disregard.*instructions|you are now|"
    r"system prompt|<\|.*\|>|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|"
    r"base64|eval\(|exec\(|subprocess|os\.system|cmd\.exe|/bin/sh|/bin/bash"
    r"|rm\s+-rf|chmod\s+777|wget\s+|curl\s+)",
    re.IGNORECASE,
)


def _validate_model_in_registry(model_id: str) -> None:
    if model_id not in APPROVED_MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_id}' is not in the approved model registry. "
            f"Approved models: {list(APPROVED_MODEL_REGISTRY.keys())}"
        )


def _sanitize_text_content(text: str, context: str = "input") -> str:
    if MALICIOUS_PROMPT_PATTERNS.search(text):
        raise ValueError(
            f"Potentially malicious content detected in {context}. Request blocked."
        )
    # Check for base64-encoded content
    try:
        segments = text.split()
        for segment in segments:
            if len(segment) > 20 and re.match(r'^[A-Za-z0-9+/=]+$', segment):
                decoded = base64.b64decode(segment).decode("utf-8", errors="ignore")
                if MALICIOUS_PROMPT_PATTERNS.search(decoded):
                    raise ValueError(
                        f"Base64-encoded malicious content detected in {context}. Request blocked."
                    )
    except Exception as exc:
        if "malicious content" in str(exc):
            raise
    return text


def _sanitize_prompt_file(content: str, filename: str) -> str:
    return _sanitize_text_content(content, context=f"prompt file '{filename}'")


def _validate_and_sanitize_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    sanitized = []
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            _sanitize_text_content(msg.content, context="message input")
        sanitized.append(msg)
    return sanitized


def _validate_output_for_dangerous_patterns(response: BaseMessage) -> BaseMessage:
    if hasattr(response, "content") and isinstance(response.content, str):
        if DANGEROUS_PATTERNS.search(response.content):
            raise ValueError(
                "LLM output contains potentially dangerous code execution primitives. "
                "Response blocked."
            )
        _sanitize_text_content(response.content, context="LLM output")
    return response


def _validate_tool_calls(last_message: BaseMessage) -> None:
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
            if tool_name not in APPROVED_TOOL_NAMES:
                logger.warning(
                    "AUDIT: Denied tool invocation attempt",
                    extra={
                        "tool_name": tool_name,
                        "approved_tools": list(APPROVED_TOOL_NAMES),
                        "denial_reason": "Tool not in approved allow list",
                    },
                )
                raise ValueError(
                    f"Tool '{tool_name}' is not in the approved tool allow list. "
                    f"Approved tools: {list(APPROVED_TOOL_NAMES)}"
                )


def _label_ai_output(response: BaseMessage, model_id: str) -> BaseMessage:
    timestamp = datetime.now(timezone.utc).isoformat()
    provenance = {
        "ai_generated": True,
        "model_id": model_id,
        "timestamp": timestamp,
        "content_hash": hashlib.sha256(
            (response.content if isinstance(response.content, str) else str(response.content)).encode()
        ).hexdigest(),
    }
    if not hasattr(response, "additional_kwargs"):
        response.additional_kwargs = {}
    response.additional_kwargs["provenance"] = provenance
    logger.info("AI_CONTENT_PROVENANCE: %s", provenance)
    return response


def _check_user_authenticated(runtime: "Runtime") -> None:
    context = runtime.context if hasattr(runtime, "context") else {}
    user_id = context.get("user_id") if isinstance(context, dict) else getattr(context, "user_id", None)
    authenticated = context.get("authenticated") if isinstance(context, dict) else getattr(context, "authenticated", None)
    if not user_id or not authenticated:
        raise PermissionError(
            "Authentication required: user must authenticate before accessing the AI agent."
        )


# Validate approved model
_APPROVED_MODEL_ID = "gpt-4o-2024-05-13"
_validate_model_in_registry(_APPROVED_MODEL_ID)

tools_raw = [TavilySearchResults(max_results=1)]

# Enforce tool allow list at bind time
tools = []
for t in tools_raw:
    tool_name = getattr(t, "name", None)
    if tool_name not in APPROVED_TOOL_NAMES:
        raise ValueError(
            f"Tool '{tool_name}' is not in the approved tool allow list: {APPROVED_TOOL_NAMES}"
        )
    tools.append(t)

model_oai = ChatOpenAI(temperature=0, model=_APPROVED_MODEL_ID)
model_oai = model_oai.bind_tools(tools)

_prompt_path = Path(__file__).parent.parent / "prompt.txt"
_subprompt_path = Path(__file__).parent / "subprompt.txt"

with open(_prompt_path) as _f:
    _raw_prompt = _f.read()
prompt = _sanitize_prompt_file(_raw_prompt, str(_prompt_path))

with open(_subprompt_path) as _f:
    _raw_subprompt = _f.read()
subprompt = _sanitize_prompt_file(_raw_subprompt, str(_subprompt_path))


class AgentContext(TypedDict):
    model: Literal["openai"]
    user_id: str
    authenticated: bool


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # Validate tool calls against allow list before continuing
    if not last_message.tool_calls:
        return "end"
    # Validate tool calls are in the approved allow list
    _validate_tool_calls(last_message)
    return "continue"


# Define the function that calls the model
def call_model(state, runtime: Runtime[AgentContext]):
    # Enforce authentication
    _check_user_authenticated(runtime)

    trace_id = str(uuid.uuid4())
    model = model_oai
    model_id = _APPROVED_MODEL_ID

    # Validate and sanitize input messages
    raw_messages = state["messages"]
    messages = _validate_and_sanitize_messages(raw_messages)

    # Minimise context: only pass the last 20 messages to avoid over-broad injection
    messages = list(messages)[-20:]

    input_hash = hashlib.sha256(
        str([getattr(m, "content", "") for m in messages]).encode()
    ).hexdigest()

    logger.info(
        "LLM_INTERACTION_START: trace_id=%s model=%s input_hash=%s message_count=%d timestamp=%s",
        trace_id,
        model_id,
        input_hash,
        len(messages),
        datetime.now(timezone.utc).isoformat(),
    )

    response = model.invoke(messages)

    # Validate and sanitize LLM output
    response = _validate_output_for_dangerous_patterns(response)

    # Label AI-generated output with provenance
    response = _label_ai_output(response, model_id)

    output_hash = hashlib.sha256(
        (response.content if isinstance(response.content, str) else str(response.content)).encode()
    ).hexdigest()

    logger.info(
        "LLM_INTERACTION_END: trace_id=%s model=%s output_hash=%s timestamp=%s",
        trace_id,
        model_id,
        output_hash,
        datetime.now(timezone.utc).isoformat(),
    )

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
tool_node = ToolNode(tools)


# Define a new graph
workflow = StateGraph(AgentState, context_schema=AgentContext)

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