"""
Simple test to demonstrate the fix for the multiple tool results issue in human approval flow.

This is a simplified version of the fix for GitHub issue #4397 that demonstrates
the human approval flow without using interrupt() to avoid checkpointer complications.
This makes it easier to test the core duplicate prevention logic.

Related files:
- test_human_approval_tool_duplication_fix.py: Comprehensive test with interrupt()
- examples/human_in_the_loop/fixed_human_approval_example.py: Full working example
"""

from typing import Annotated, List, Literal, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command


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


# Tool categorization
SAFE_TOOLS = [search_web]
SENSITIVE_TOOLS = [send_email]


def create_ai_message_with_tool_call(tool_name: str, args: dict, call_id: str) -> AIMessage:
    """Helper to create AI message with tool call."""
    return AIMessage(
        content=f"I'll use the {tool_name} tool.",
        tool_calls=[{
            "name": tool_name,
            "args": args,
            "id": call_id,
            "type": "tool_call"
        }]
    )


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
    """Human approval node for sensitive tools (simplified for testing)."""
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return Command(goto="__end__")
    
    # For testing, we'll automatically approve
    print("🔍 Human approval requested for sensitive tool calls:")
    for call in last_message.tool_calls:
        print(f"  - {call['name']} with args: {call['args']}")
    print("✅ Auto-approved for testing")
    
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
            print(f"⚠️  Skipping duplicate tool call: {call_id}")
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
            print(f"✅ Executed {tool_name}: {result}")
    
    if new_results:
        return {"messages": new_results}
    
    return state


def create_test_graph() -> StateGraph:
    """Create the test graph with proper human approval flow."""
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("safe_tools", safe_tool_executor)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("sensitive_tools", sensitive_tool_executor)
    
    # Add edges
    graph.add_conditional_edges(
        START,
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


def test_safe_tool():
    """Test safe tool execution."""
    print("\n=== Testing Safe Tool (Search) ===")
    
    graph = create_test_graph()
    app = graph.compile()
    
    # Create initial state with AI message requesting safe tool
    ai_message = create_ai_message_with_tool_call(
        "search_web", 
        {"query": "Python tutorials"}, 
        "search_1"
    )
    
    initial_state = {
        "messages": [
            HumanMessage(content="Please search for Python tutorials"),
            ai_message
        ]
    }
    
    result = app.invoke(initial_state)
    
    print("Final messages:")
    for i, msg in enumerate(result["messages"]):
        print(f"  {i+1}. {type(msg).__name__}: {msg.content}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"     Tool calls: {msg.tool_calls}")
        if hasattr(msg, 'tool_call_id'):
            print(f"     Tool call ID: {msg.tool_call_id}")
    
    # Count tool results
    tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    print(f"Number of tool results: {len(tool_results)}")
    
    return len(tool_results) == 1


def test_sensitive_tool():
    """Test sensitive tool execution with human approval."""
    print("\n=== Testing Sensitive Tool (Email) ===")
    
    graph = create_test_graph()
    app = graph.compile()
    
    # Create initial state with AI message requesting sensitive tool
    ai_message = create_ai_message_with_tool_call(
        "send_email",
        {"recipient": "test@example.com", "subject": "Test", "body": "Hello"},
        "email_1"
    )
    
    initial_state = {
        "messages": [
            HumanMessage(content="Please send an email"),
            ai_message
        ]
    }
    
    result = app.invoke(initial_state)
    
    print("Final messages:")
    for i, msg in enumerate(result["messages"]):
        print(f"  {i+1}. {type(msg).__name__}: {msg.content}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"     Tool calls: {msg.tool_calls}")
        if hasattr(msg, 'tool_call_id'):
            print(f"     Tool call ID: {msg.tool_call_id}")
    
    # Count tool results
    tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    print(f"Number of tool results: {len(tool_results)}")
    
    return len(tool_results) == 1


def test_duplicate_prevention():
    """Test that duplicate tool execution is prevented."""
    print("\n=== Testing Duplicate Prevention ===")
    
    # Create a state that already has a tool result
    ai_message = create_ai_message_with_tool_call(
        "send_email",
        {"recipient": "test@example.com", "subject": "Test", "body": "Hello"},
        "email_1"
    )
    
    existing_tool_result = ToolMessage(
        content="Email sent to test@example.com with subject 'Test'",
        tool_call_id="email_1",
        name="send_email"
    )
    
    state_with_existing_result = {
        "messages": [
            HumanMessage(content="Please send an email"),
            ai_message,
            existing_tool_result
        ]
    }
    
    # Try to execute the tool again
    result_state = sensitive_tool_executor(state_with_existing_result)
    
    # Count tool results
    tool_results = [m for m in result_state["messages"] if isinstance(m, ToolMessage)]
    print(f"Number of tool results after duplicate prevention: {len(tool_results)}")
    
    return len(tool_results) == 1  # Should still be 1, not 2


if __name__ == "__main__":
    print("Testing Fixed Human Approval Flow")
    print("=" * 50)

    # Run tests
    safe_test_passed = test_safe_tool()
    sensitive_test_passed = test_sensitive_tool()
    duplicate_test_passed = test_duplicate_prevention()

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Safe tool test: {'✅ PASSED' if safe_test_passed else '❌ FAILED'}")
    print(f"Sensitive tool test: {'✅ PASSED' if sensitive_test_passed else '❌ FAILED'}")
    print(f"Duplicate prevention test: {'✅ PASSED' if duplicate_test_passed else '❌ FAILED'}")

    if all([safe_test_passed, sensitive_test_passed, duplicate_test_passed]):
        print("🎉 ALL TESTS PASSED - The fix works correctly!")
    else:
        print("❌ Some tests failed")
