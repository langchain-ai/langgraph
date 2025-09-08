"""Simple LangGraph agent for monorepo testing."""

from common import get_common_prefix
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from shared import get_dummy_message

from agent.state import State


def call_model(state: State) -> dict:
    """Simple node that uses the shared libraries."""
    # Use functions from both shared packages
    dummy_message = get_dummy_message()
    prefix = get_common_prefix()

    message = AIMessage(content=f"{prefix} Agent says: {dummy_message}")

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
