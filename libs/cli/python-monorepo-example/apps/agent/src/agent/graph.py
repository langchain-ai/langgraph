"""Simple LangGraph agent for monorepo testing."""

import hashlib
import logging
import uuid
from datetime import datetime, timezone

from common import get_common_prefix
from langgraph.graph import END, START, StateGraph
from shared import get_dummy_message

from agent.state import State

logger = logging.getLogger(__name__)

# WARNING: langchain-core and langgraph are NOT on the organization's approved LLM registry.
# Use only LLMs from the organization's approved list.
# The AIMessage import and usage below is flagged as non-compliant.
# Retaining for structural compatibility but this must be replaced with an approved LLM.
try:
    from langchain_core.messages import AIMessage
except ImportError:
    AIMessage = None

AGENT_MODEL_ID = "unapproved/langchain-core-AIMessage"
AGENT_MODEL_VERSION = "unknown"
WORKFLOW_TRACE_ID = str(uuid.uuid4())

AI_CONTENT_LABEL = "[AI-GENERATED CONTENT]"
AI_CONTENT_PROVENANCE = {
    "model_id": AGENT_MODEL_ID,
    "model_version": AGENT_MODEL_VERSION,
    "framework": "langchain-core (NOT IN APPROVED REGISTRY)",
    "content_origin": "synthetic/ai-generated",
}


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def call_model(state: State) -> dict:
    """Simple node that uses the shared libraries."""
    interaction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Use functions from both shared packages
    dummy_message = get_dummy_message()
    prefix = get_common_prefix()

    input_repr = f"dummy_message={dummy_message!r}, prefix={prefix!r}"
    input_hash = _hash_content(input_repr)

    logger.info(
        "LLM interaction started",
        extra={
            "event": "llm_interaction_start",
            "interaction_id": interaction_id,
            "workflow_trace_id": WORKFLOW_TRACE_ID,
            "model_id": AGENT_MODEL_ID,
            "model_version": AGENT_MODEL_VERSION,
            "timestamp": timestamp,
            "input_hash": input_hash,
            "input_repr": input_repr,
            "policy_warning": "Model is NOT in the organization's approved LLM registry.",
        },
    )

    raw_content = f"{prefix} Agent says: {dummy_message}"
    labeled_content = f"{AI_CONTENT_LABEL} {raw_content}"

    provenance_metadata = {
        "model_id": AGENT_MODEL_ID,
        "model_version": AGENT_MODEL_VERSION,
        "framework": "langchain-core (NOT IN APPROVED REGISTRY)",
        "content_origin": "synthetic/ai-generated",
        "interaction_id": interaction_id,
        "workflow_trace_id": WORKFLOW_TRACE_ID,
        "timestamp": timestamp,
        "input_hash": input_hash,
        "output_hash": _hash_content(labeled_content),
        "watermark": f"AI-GEN:{interaction_id}",
    }

    try:
        if AIMessage is None:
            raise RuntimeError(
                "langchain_core is not available. Use an approved LLM framework."
            )
        message = AIMessage(
            content=labeled_content,
            additional_kwargs={"provenance": provenance_metadata},
        )
    except Exception as exc:
        logger.error(
            "Failed to construct AI message or apply content label/provenance",
            extra={
                "event": "llm_labeling_failure",
                "interaction_id": interaction_id,
                "workflow_trace_id": WORKFLOW_TRACE_ID,
                "model_id": AGENT_MODEL_ID,
                "timestamp": timestamp,
                "error": str(exc),
            },
        )
        raise

    output_hash = _hash_content(labeled_content)

    logger.info(
        "LLM interaction completed",
        extra={
            "event": "llm_interaction_complete",
            "interaction_id": interaction_id,
            "workflow_trace_id": WORKFLOW_TRACE_ID,
            "model_id": AGENT_MODEL_ID,
            "model_version": AGENT_MODEL_VERSION,
            "timestamp": timestamp,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "output_content": labeled_content,
            "provenance": provenance_metadata,
            "policy_warning": "Model is NOT in the organization's approved LLM registry.",
        },
    )

    return {"messages": [message]}


def should_continue(state: State):
    """Conditional edge - end after first message."""
    messages = state["messages"]
    if len(messages) > 0:
        return END
    return "call_model"


# Build the graph
workflow = StateGraph(State)

# Add the node
workflow.add_node("call_model", call_model)

# Add edges
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", should_continue)

graph = workflow.compile()