from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

try:
    from langchain_ollama import ChatOllama
except ImportError:
    # Fallback or helpful error if user doesn't have it installed
    from langchain_community.chat_models import ChatOllama

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

# Initialize Tools
# Note: For full sovereignty, we would replace Tavily with a local search tool or similar. 
# Keeping Tavily for now as it's an external tool, not the brain itself.
tools = [TavilySearchResults(max_results=1)]

# Initialize Sovereign Model (Ollama)
# Requires: ollama pull llama3
model_local = ChatOllama(model="llama3", temperature=0)
model_local = model_local.bind_tools(tools)

class AgentContext(TypedDict):
    # Expanded to include 'local'
    model: Literal["anthropic", "openai", "local"]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

def call_model(state, runtime: Runtime[AgentContext]):
    # Default to local if not specified, or if specified as local
    model_type = runtime.context.get("model", "local")
    
    if model_type == "local":
        model = model_local
    # Fallbacks could be added here if imports were available
    else:
        model = model_local # Force sovereignty for this template

    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

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
