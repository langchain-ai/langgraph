import logging
import re
import hashlib
import datetime
import uuid
from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Approved model registry — only pinned, registry-approved models are permitted.
APPROVED_MODEL_REGISTRY = {
    "approved-model-v1": "approved-model-v1",
}

# Approved tool allow list
APPROVED_TOOLS = {"tavily_search_results_json"}

# Approved URL allowlist for outbound HTTP
APPROVED_URL_ALLOWLIST = [
    "api.tavily.com",
]

# Dynamic code execution primitives to block in LLM output
DANGEROUS_PATTERNS = re.compile(
    r"\b(eval|exec|subprocess|os\.system|os\.popen|__import__|compile|execfile|open)\s*\(",
    re.IGNORECASE,
)

# Suspicious prompt injection patterns
PROMPT_INJECTION_PATTERNS = re.compile(
    r"(base64|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|<script|javascript:|data:text/html|"
    r"ignore previous|disregard (all )?instructions|system prompt|you are now|"
    r"rm\s+-rf|/bin/sh|/bin/bash|cmd\.exe|powershell)",
    re.IGNORECASE,
)

# NOTE: ChatOpenAI (OpenAI/GPT) is NOT in the organization's approved model registry.
# This model is retained here only to preserve the existing structure of this example file.
# In production, replace with an approved model from the registry.
_PINNED_MODEL_NAME = "gpt-3.5-turbo-0125"  # version-pinned; replace with approved model
model_oai = ChatOpenAI(model=_PINNED_MODEL_NAME, temperature=0)

_TRACE_ID = str(uuid.uuid4())


def _validate_url_allowlist(url: str) -> bool:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        return False
    hostname = parsed.hostname or ""
    return any(hostname == allowed or hostname.endswith("." + allowed) for allowed in APPROVED_URL_ALLOWLIST)


def _sanitize_messages(messages):
    sanitized = []
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        if not isinstance(content, str):
            content = str(content)
        if len(content) > 32768:
            raise ValueError("Input message exceeds maximum allowed length.")
        if PROMPT_INJECTION_PATTERNS.search(content):
            raise ValueError(f"Input message contains disallowed content: potential prompt injection detected.")
        sanitized.append(msg)
    return sanitized


def _validate_llm_output(response):
    content = response.content if hasattr(response, "content") else str(response)
    if not isinstance(content, str):
        content = str(content)
    if DANGEROUS_PATTERNS.search(content):
        raise ValueError("LLM output contains potentially dangerous code execution primitive.")
    return response


def _check_tool_allowlist(tool_calls):
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
        if tool_name not in APPROVED_TOOLS:
            logger.warning(
                "AUDIT|TOOL_DENIED|trace_id=%s|tool=%s|reason=not_in_allowlist|timestamp=%s",
                _TRACE_ID, tool_name, datetime.datetime.utcnow().isoformat()
            )
            raise ValueError(f"Tool '{tool_name}' is not in the approved tool allow list.")


def _require_authentication(config):
    configurable = config.get("configurable", {}) if config else {}
    user_id = configurable.get("user_id") or configurable.get("authenticated_user")
    if not user_id:
        raise PermissionError("Authentication required: no authenticated user found in config.")
    return user_id


def _label_ai_output(response, model_name: str, trace_id: str):
    timestamp = datetime.datetime.utcnow().isoformat()
    content = response.content if hasattr(response, "content") else str(response)
    provenance = {
        "ai_generated": True,
        "model": model_name,
        "model_version": _PINNED_MODEL_NAME,
        "timestamp": timestamp,
        "trace_id": trace_id,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    if hasattr(response, "additional_kwargs"):
        response.additional_kwargs["provenance"] = provenance
    return response


def _filter_tool_output(tool_output, max_length: int = 4096):
    if isinstance(tool_output, str):
        return tool_output[:max_length]
    if isinstance(tool_output, list):
        filtered = []
        for item in tool_output:
            if isinstance(item, dict):
                filtered.append({k: str(v)[:512] for k, v in item.items() if k in ("url", "title", "content", "snippet")})
            else:
                filtered.append(str(item)[:512])
        return filtered
    return tool_output


tools_raw = [TavilySearchResults(max_results=1)]

# Enforce tool allow list at registration time
for _tool in tools_raw:
    _tool_name = getattr(_tool, "name", None) or getattr(_tool, "__class__", type(_tool)).__name__.lower()
    if _tool_name not in APPROVED_TOOLS:
        raise ValueError(f"Tool '{_tool_name}' is not in the approved tool allow list and cannot be registered.")

tools = tools_raw

model_oai = model_oai.bind_tools(tools)


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
        # Enforce tool allow list before routing to tool execution
        _check_tool_allowlist(last_message.tool_calls)
        return "continue"


# Define the function that calls the model
def call_model(state, config):
    # Authentication check
    user_id = _require_authentication(config)

    model = model_oai
    messages = state["messages"]

    # Sanitize and validate input messages
    messages = _sanitize_messages(messages)

    input_hash = hashlib.sha256(
        "".join(m.content if hasattr(m, "content") else str(m) for m in messages).encode("utf-8")
    ).hexdigest()

    timestamp_in = datetime.datetime.utcnow().isoformat()
    logger.info(
        "AUDIT|LLM_INPUT|trace_id=%s|user=%s|model=%s|input_hash=%s|timestamp=%s",
        _TRACE_ID, user_id, _PINNED_MODEL_NAME, input_hash, timestamp_in
    )

    response = model.invoke(messages)

    # Validate LLM output for dangerous code execution primitives
    response = _validate_llm_output(response)

    # Label AI-generated output with provenance metadata
    response = _label_ai_output(response, model_name=_PINNED_MODEL_NAME, trace_id=_TRACE_ID)

    timestamp_out = datetime.datetime.utcnow().isoformat()
    output_content = response.content if hasattr(response, "content") else str(response)
    output_hash = hashlib.sha256(output_content.encode("utf-8")).hexdigest()

    logger.info(
        "AUDIT|LLM_OUTPUT|trace_id=%s|user=%s|model=%s|output_hash=%s|timestamp=%s",
        _TRACE_ID, user_id, _PINNED_MODEL_NAME, output_hash, timestamp_out
    )

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools with output minimisation
class FilteredToolNode(ToolNode):
    def invoke(self, input, config=None, **kwargs):
        result = super().invoke(input, config=config, **kwargs)
        if isinstance(result, dict) and "messages" in result:
            filtered_messages = []
            for msg in result["messages"]:
                if hasattr(msg, "content"):
                    filtered_content = _filter_tool_output(msg.content)
                    if isinstance(filtered_content, str):
                        msg.content = filtered_content
                    else:
                        import json
                        msg.content = json.dumps(filtered_content)[:4096]
                filtered_messages.append(msg)
            result["messages"] = filtered_messages
        timestamp = datetime.datetime.utcnow().isoformat()
        logger.info(
            "AUDIT|TOOL_EXECUTION|trace_id=%s|timestamp=%s",
            _TRACE_ID, timestamp
        )
        return result


tool_node = FilteredToolNode(tools)


class ContextSchema(TypedDict):
    model: Literal["approved-model-v1"]


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