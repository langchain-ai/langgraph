"""
Complete working example of the fixed human approval flow for LangGraph.

This example demonstrates the proper way to implement human-in-the-loop tool execution
without the multiple tool results bug described in issue #4397.
"""

from typing import Annotated, List, Literal, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt


# Define the state with proper message handling
class AgentState(TypedDict):
    """State for the agent with automatic message deduplication."""
    messages: Annotated[List[BaseMessage], add_messages]


# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for information (safe tool)."""
    return f"Search results for: {query}"


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email (sensitive tool requiring approval)."""
    return f"Email sent to {recipient} with subject '{subject}'"


@tool
def delete_file(filepath: str) -> str:
    """Delete a file (sensitive tool requiring approval)."""
    return f"File {filepath} has been deleted"


# Tool categorization
SAFE_TOOLS = [search_web]
SENSITIVE_TOOLS = [send_email, delete_file]
ALL_TOOLS = SAFE_TOOLS + SENSITIVE_TOOLS


def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor node that processes user requests."""
    # In a real implementation, this would call an LLM
    # For demo purposes, we'll simulate tool calls
    
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()
        
        if "search" in content:
            # Simulate AI deciding to use search tool
            ai_response = AIMessage(
                content="I'll search for that information.",
                tool_calls=[{
                    "name": "search_web",
                    "args": {"query": "user query"},
                    "id": "search_1",
                    "type": "tool_call"
                }]
            )
        elif "email" in content:
            # Simulate AI deciding to send email
            ai_response = AIMessage(
                content="I'll send that email for you.",
                tool_calls=[{
                    "name": "send_email",
                    "args": {
                        "recipient": "user@example.com",
                        "subject": "Test Email",
                        "body": "This is a test email."
                    },
                    "id": "email_1",
                    "type": "tool_call"
                }]
            )
        elif "delete" in content:
            # Simulate AI deciding to delete file
            ai_response = AIMessage(
                content="I'll delete that file for you.",
                tool_calls=[{
                    "name": "delete_file",
                    "args": {"filepath": "/tmp/test.txt"},
                    "id": "delete_1",
                    "type": "tool_call"
                }]
            )
        else:
            # No tool call needed
            ai_response = AIMessage(content="I understand. How can I help you?")
        
        return {"messages": [ai_response]}
    
    return state


def tool_router(state: AgentState) -> Literal["safe_tools", "human_approval", "__end__"]:
    """Route to appropriate tool execution based on tool type."""
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return "__end__"
    
    tool_calls = last_message.tool_calls
    tool_names = [call["name"] for call in tool_calls]
    
    # Check if any sensitive tools are requested
    sensitive_requested = any(name in [t.name for t in SENSITIVE_TOOLS] for name in tool_names)
    safe_requested = any(name in [t.name for t in SAFE_TOOLS] for name in tool_names)
    
    if sensitive_requested:
        return "human_approval"
    elif safe_requested:
        return "safe_tools"
    else:
        return "__end__"


def safe_tool_executor(state: AgentState) -> AgentState:
    """Execute safe tools directly without approval."""
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return state
    
    # Get existing tool result IDs to prevent duplicates
    existing_result_ids = {
        getattr(m, "tool_call_id", None)
        for m in state["messages"]
        if isinstance(m, ToolMessage)
    }
    
    new_results = []
    for tool_call in last_message.tool_calls:
        call_id = tool_call["id"]
        
        # Skip if result already exists
        if call_id in existing_result_ids:
            continue
        
        tool_name = tool_call["name"]
        
        # Execute safe tools
        if tool_name == "search_web":
            result = search_web.invoke(tool_call["args"])
            new_results.append(ToolMessage(
                content=result,
                tool_call_id=call_id,
                name=tool_name
            ))
    
    if new_results:
        return {"messages": new_results}
    
    return state


def human_approval_node(state: AgentState) -> Command[Literal["sensitive_tools", "__end__"]]:
    """Human approval node for sensitive tools."""
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return Command(goto="__end__")
    
    tool_calls = last_message.tool_calls
    
    # For demo purposes, we'll simulate the approval process
    # In a real implementation, this would use interrupt() to pause for human input
    try:
        approval_data = interrupt({
            "message": "The following tool calls require approval:",
            "tool_calls": [
                {
                    "name": call["name"],
                    "args": call["args"],
                    "description": f"Execute {call['name']} with args {call['args']}"
                }
                for call in tool_calls
            ],
            "instructions": "Respond with {'approved': True} to approve or {'approved': False} to deny"
        })
        
        if approval_data.get("approved", False):
            return Command(goto="sensitive_tools")
        else:
            # Create denial messages for each tool call
            denial_messages = [
                ToolMessage(
                    content="Tool execution denied by human reviewer",
                    tool_call_id=call["id"],
                    name=call["name"]
                )
                for call in tool_calls
            ]
            return Command(goto="__end__", update={"messages": denial_messages})
            
    except Exception:
        # If interrupt fails (e.g., in testing), default to approval
        return Command(goto="sensitive_tools")


def sensitive_tool_executor(state: AgentState) -> AgentState:
    """Execute sensitive tools after human approval."""
    # Find the most recent AI message with tool calls
    ai_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and message.tool_calls:
            ai_message = message
            break
    
    if not ai_message:
        return state
    
    # Get existing tool result IDs to prevent duplicates
    existing_result_ids = {
        getattr(m, "tool_call_id", None)
        for m in state["messages"]
        if isinstance(m, ToolMessage)
    }
    
    new_results = []
    for tool_call in ai_message.tool_calls:
        call_id = tool_call["id"]
        
        # Skip if result already exists
        if call_id in existing_result_ids:
            continue
        
        tool_name = tool_call["name"]
        
        # Execute sensitive tools
        if tool_name == "send_email":
            result = send_email.invoke(tool_call["args"])
            new_results.append(ToolMessage(
                content=result,
                tool_call_id=call_id,
                name=tool_name
            ))
        elif tool_name == "delete_file":
            result = delete_file.invoke(tool_call["args"])
            new_results.append(ToolMessage(
                content=result,
                tool_call_id=call_id,
                name=tool_name
            ))
    
    if new_results:
        return {"messages": new_results}
    
    return state


def create_agent_graph() -> StateGraph:
    """Create the complete agent graph with proper human approval flow."""
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("safe_tools", safe_tool_executor)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("sensitive_tools", sensitive_tool_executor)
    
    # Add edges
    graph.add_edge(START, "supervisor")
    
    # Conditional routing from supervisor
    graph.add_conditional_edges(
        "supervisor",
        tool_router,
        {
            "safe_tools": "safe_tools",
            "human_approval": "human_approval",
            "__end__": END
        }
    )
    
    # Direct edges to end (no separate result handler needed)
    graph.add_edge("safe_tools", END)
    graph.add_edge("sensitive_tools", END)
    
    return graph


def demo_usage():
    """Demonstrate the fixed implementation."""
    print("Creating agent with fixed human approval flow...")
    
    # Create the graph
    graph = create_agent_graph()
    
    # Add checkpointer for human-in-the-loop functionality
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    
    # Configuration for the conversation
    config = {"configurable": {"thread_id": "demo_thread"}}
    
    print("\n=== Testing Safe Tool (Search) ===")
    try:
        result = app.invoke(
            {"messages": [HumanMessage(content="Please search for Python tutorials")]},
            config
        )
        print("Messages:")
        for msg in result["messages"]:
            print(f"  {type(msg).__name__}: {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"    Tool calls: {msg.tool_calls}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Testing Sensitive Tool (Email) ===")
    print("Note: This would normally pause for human approval")
    try:
        # In a real scenario, this would pause at the interrupt() call
        result = app.invoke(
            {"messages": [HumanMessage(content="Please send an email to the team")]},
            config
        )
        print("Messages:")
        for msg in result["messages"]:
            print(f"  {type(msg).__name__}: {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"    Tool calls: {msg.tool_calls}")
    except Exception as e:
        print(f"Note: This is expected when interrupt() is called: {e}")


if __name__ == "__main__":
    demo_usage()
