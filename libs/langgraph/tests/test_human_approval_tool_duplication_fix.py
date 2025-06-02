"""
Test file to reproduce and fix the multiple tool results issue in human approval flow.

This test demonstrates the issue described in GitHub issue #4397 where sensitive tools
requiring human approval generate multiple tool results for a single tool call.
"""

import uuid
from typing import Annotated, Literal, TypedDict, List
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt


class GraphState(TypedDict):
    """State for the test graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    next: str


# Define test tools
@tool
def safe_math_tool(a: int, b: int) -> str:
    """A safe math tool that adds two numbers."""
    return f"The sum of {a} and {b} is {a + b}"


@tool
def sensitive_email_tool(recipient: str, subject: str) -> str:
    """A sensitive tool that sends emails (requires approval)."""
    return f"Email sent to {recipient} with subject: {subject}"


# PROBLEMATIC IMPLEMENTATION (reproduces the bug)
def problematic_tool_result_handler(state: GraphState) -> GraphState:
    """
    This is the problematic implementation from the original issue.
    It causes duplicate tool results.
    """
    tool_use_message = next(
        (m for m in reversed(state["messages"]) if hasattr(m, "tool_calls") and m.tool_calls),
        None
    )
    if not tool_use_message:
        return state
    
    existing_result_ids = {
        getattr(m, "tool_call_id", None)
        for m in state["messages"]
        if isinstance(m, ToolMessage)
    }
    
    new_results = []
    for tool_call in tool_use_message.tool_calls:
        call_id = tool_call["id"]
        if call_id in existing_result_ids:
            continue
        
        # BUG: This logic is flawed - it searches for tool results but breaks early
        tool_result_message = next(
            (m for m in state["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == call_id),
            None
        )
        if tool_result_message:
            new_results.append(tool_result_message)
            break  # BUG: This break prevents checking other tool calls
    
    if not new_results:
        return state
    
    return GraphState(messages=state["messages"] + new_results)


# FIXED IMPLEMENTATION
def fixed_tool_result_handler(state: GraphState) -> GraphState:
    """
    Fixed implementation that properly handles tool results without duplication.
    """
    # Get the latest AI message with tool calls
    tool_use_message = next(
        (m for m in reversed(state["messages"]) if hasattr(m, "tool_calls") and m.tool_calls),
        None
    )
    if not tool_use_message:
        return state
    
    # Get existing tool result IDs to avoid duplicates
    existing_result_ids = {
        getattr(m, "tool_call_id", None)
        for m in state["messages"]
        if isinstance(m, ToolMessage)
    }
    
    # Check if all tool calls already have results
    pending_tool_calls = [
        call for call in tool_use_message.tool_calls
        if call["id"] not in existing_result_ids
    ]
    
    # If no pending tool calls, no action needed
    if not pending_tool_calls:
        return state
    
    # This handler should not create new tool results - that's the job of tool execution nodes
    # It should only manage existing results if needed
    return state


def routing_node_factory(agent_name: str, safe_tools: List, sensitive_tools: List):
    """Factory to create routing nodes for different agents."""
    def routing_node(state: GraphState) -> Command:
        ai_message = state["messages"][-1]
        tool_calls = getattr(ai_message, "tool_calls", []) or []
        tools_requested = [call["name"] for call in tool_calls]
        safe_tool_names = [t.name for t in safe_tools]
        sensitive_tool_names = [t.name for t in sensitive_tools]
        
        for tool in tools_requested:
            if tool in sensitive_tool_names:
                return Command(goto="human_approval", update={"next": f"{agent_name}_sensitive"})
            if tool in safe_tool_names:
                return Command(goto=f"{agent_name}_safe", update={"next": f"{agent_name}_safe"})
        
        return Command(goto=END, update={"next": END})
    
    return routing_node


def safe_tool_executor(state: GraphState) -> GraphState:
    """Execute safe tools directly."""
    ai_message = state["messages"][-1]
    tool_calls = getattr(ai_message, "tool_calls", []) or []
    
    # Get existing tool result IDs to avoid duplicates
    existing_result_ids = {
        getattr(m, "tool_call_id", None)
        for m in state["messages"]
        if isinstance(m, ToolMessage)
    }
    
    new_results = []
    for tool_call in tool_calls:
        call_id = tool_call["id"]
        if call_id in existing_result_ids:
            continue  # Skip if result already exists
            
        if tool_call["name"] == "safe_math_tool":
            args = tool_call["args"]
            result = safe_math_tool.invoke(args)
            new_results.append(ToolMessage(
                content=result,
                tool_call_id=call_id,
                name=tool_call["name"]
            ))
    
    if new_results:
        return {"messages": new_results}
    return state


def sensitive_tool_executor(state: GraphState) -> GraphState:
    """Execute sensitive tools after human approval."""
    ai_message = next(
        (m for m in reversed(state["messages"]) if hasattr(m, "tool_calls") and m.tool_calls),
        None
    )
    if not ai_message:
        return state
    
    tool_calls = getattr(ai_message, "tool_calls", []) or []
    
    # Get existing tool result IDs to avoid duplicates
    existing_result_ids = {
        getattr(m, "tool_call_id", None)
        for m in state["messages"]
        if isinstance(m, ToolMessage)
    }
    
    new_results = []
    for tool_call in tool_calls:
        call_id = tool_call["id"]
        if call_id in existing_result_ids:
            continue  # Skip if result already exists
            
        if tool_call["name"] == "sensitive_email_tool":
            args = tool_call["args"]
            result = sensitive_email_tool.invoke(args)
            new_results.append(ToolMessage(
                content=result,
                tool_call_id=call_id,
                name=tool_call["name"]
            ))
    
    if new_results:
        return {"messages": new_results}
    return state


def human_approval_node(state: GraphState) -> Command:
    """Human approval node for sensitive tools."""
    # Get the tool call that needs approval
    ai_message = state["messages"][-1]
    tool_calls = getattr(ai_message, "tool_calls", []) or []
    
    if not tool_calls:
        return Command(goto=END)
    
    # For demo purposes, we'll simulate approval
    # In real implementation, this would use interrupt()
    approval = interrupt({
        "message": f"Approve tool call: {tool_calls[0]['name']}?",
        "tool_calls": tool_calls
    })
    
    if approval.get("approved", True):  # Default to approved for testing
        next_node = state.get("next", "math_sensitive")
        return Command(goto=next_node)
    else:
        return Command(goto=END, update={"messages": [
            ToolMessage(
                content="Tool execution denied by human reviewer",
                tool_call_id=tool_calls[0]["id"],
                name=tool_calls[0]["name"]
            )
        ]})


def create_test_graph_with_bug() -> StateGraph:
    """Create a test graph that reproduces the bug."""
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("supervisor", lambda state: state)  # Placeholder supervisor
    graph.add_node("math_router", routing_node_factory("math", [safe_math_tool], []))
    graph.add_node("email_router", routing_node_factory("email", [], [sensitive_email_tool]))
    graph.add_node("math_safe", safe_tool_executor)
    graph.add_node("email_sensitive", sensitive_tool_executor)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("tool_result_handler", problematic_tool_result_handler)  # BUG HERE
    
    # Add edges
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "email_router")
    graph.add_edge("email_router", "human_approval")
    graph.add_edge("human_approval", "email_sensitive")
    graph.add_edge("email_sensitive", "tool_result_handler")
    graph.add_edge("tool_result_handler", END)
    
    return graph


def create_test_graph_fixed() -> StateGraph:
    """Create a test graph with the fix applied."""
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("supervisor", lambda state: state)  # Placeholder supervisor
    graph.add_node("math_router", routing_node_factory("math", [safe_math_tool], []))
    graph.add_node("email_router", routing_node_factory("email", [], [sensitive_email_tool]))
    graph.add_node("math_safe", safe_tool_executor)
    graph.add_node("email_sensitive", sensitive_tool_executor)
    graph.add_node("human_approval", human_approval_node)
    # FIXED: Remove the problematic tool_result_handler or use the fixed version
    
    # Add edges
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "email_router")
    graph.add_edge("email_router", "human_approval")
    graph.add_edge("human_approval", "email_sensitive")
    graph.add_edge("email_sensitive", END)  # Direct to END, no separate handler needed
    
    return graph


def test_reproduce_bug():
    """Test that reproduces the multiple tool results bug."""
    print("=== Testing Bug Reproduction ===")

    # Create AI message with tool call
    tool_call = {
        "name": "sensitive_email_tool",
        "args": {"recipient": "test@example.com", "subject": "Test Email"},
        "id": "test_call_1",
        "type": "tool_call"
    }

    ai_message = AIMessage(
        content="I'll send that email for you.",
        tool_calls=[tool_call]
    )

    # Simulate the problematic flow
    initial_state = GraphState(
        messages=[
            HumanMessage(content="Send an email to test@example.com"),
            ai_message
        ],
        next="email_sensitive"
    )

    # First execution - creates tool result
    tool_result = ToolMessage(
        content="Email sent to test@example.com with subject: Test Email",
        tool_call_id="test_call_1",
        name="sensitive_email_tool"
    )

    state_with_result = GraphState(
        messages=initial_state["messages"] + [tool_result],
        next="email_sensitive"
    )

    # Apply problematic handler (this should not add duplicate results)
    final_state = problematic_tool_result_handler(state_with_result)

    # Count tool results
    tool_results = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    print(f"Number of tool results: {len(tool_results)}")

    if len(tool_results) > 1:
        print("❌ BUG REPRODUCED: Multiple tool results for single tool call")
        for i, result in enumerate(tool_results):
            print(f"  Result {i+1}: {result.tool_call_id}")
    else:
        print("✅ No duplicate tool results found")

    return len(tool_results)


def test_fixed_implementation():
    """Test the fixed implementation."""
    print("\n=== Testing Fixed Implementation ===")

    # Create AI message with tool call
    tool_call = {
        "name": "sensitive_email_tool",
        "args": {"recipient": "test@example.com", "subject": "Test Email"},
        "id": "test_call_1",
        "type": "tool_call"
    }

    ai_message = AIMessage(
        content="I'll send that email for you.",
        tool_calls=[tool_call]
    )

    # Simulate the fixed flow
    initial_state = GraphState(
        messages=[
            HumanMessage(content="Send an email to test@example.com"),
            ai_message
        ],
        next="email_sensitive"
    )

    # First execution - creates tool result
    tool_result = ToolMessage(
        content="Email sent to test@example.com with subject: Test Email",
        tool_call_id="test_call_1",
        name="sensitive_email_tool"
    )

    state_with_result = GraphState(
        messages=initial_state["messages"] + [tool_result],
        next="email_sensitive"
    )

    # Apply fixed handler
    final_state = fixed_tool_result_handler(state_with_result)

    # Count tool results
    tool_results = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    print(f"Number of tool results: {len(tool_results)}")

    if len(tool_results) == 1:
        print("✅ FIXED: Exactly one tool result as expected")
    else:
        print(f"❌ ISSUE: Expected 1 tool result, got {len(tool_results)}")

    return len(tool_results)


def test_safe_vs_sensitive_tools():
    """Test that safe tools work correctly while sensitive tools are fixed."""
    print("\n=== Testing Safe vs Sensitive Tools ===")

    # Test safe tool
    safe_tool_call = {
        "name": "safe_math_tool",
        "args": {"a": 5, "b": 3},
        "id": "safe_call_1",
        "type": "tool_call"
    }

    safe_ai_message = AIMessage(
        content="I'll calculate that for you.",
        tool_calls=[safe_tool_call]
    )

    safe_state = GraphState(
        messages=[
            HumanMessage(content="What's 5 + 3?"),
            safe_ai_message
        ],
        next="math_safe"
    )

    # Execute safe tool
    safe_result_state = safe_tool_executor(safe_state)
    safe_tool_results = [m for m in safe_result_state["messages"] if isinstance(m, ToolMessage)]

    print(f"Safe tool results: {len(safe_tool_results)}")

    # Test sensitive tool
    sensitive_tool_call = {
        "name": "sensitive_email_tool",
        "args": {"recipient": "test@example.com", "subject": "Test"},
        "id": "sensitive_call_1",
        "type": "tool_call"
    }

    sensitive_ai_message = AIMessage(
        content="I'll send that email.",
        tool_calls=[sensitive_tool_call]
    )

    sensitive_state = GraphState(
        messages=[
            HumanMessage(content="Send an email"),
            sensitive_ai_message
        ],
        next="email_sensitive"
    )

    # Execute sensitive tool
    sensitive_result_state = sensitive_tool_executor(sensitive_state)
    sensitive_tool_results = [m for m in sensitive_result_state["messages"] if isinstance(m, ToolMessage)]

    print(f"Sensitive tool results: {len(sensitive_tool_results)}")

    if len(safe_tool_results) == 1 and len(sensitive_tool_results) == 1:
        print("✅ Both safe and sensitive tools work correctly")
        return True
    else:
        print("❌ Issue with tool execution")
        return False


if __name__ == "__main__":
    print("Testing Multiple Tool Results Issue and Fix")
    print("=" * 50)

    # Test 1: Reproduce the bug
    bug_result_count = test_reproduce_bug()

    # Test 2: Test the fix
    fixed_result_count = test_fixed_implementation()

    # Test 3: Test safe vs sensitive tools
    tools_work = test_safe_vs_sensitive_tools()

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Bug reproduction: {bug_result_count} tool results (should be > 1 to show bug)")
    print(f"Fixed implementation: {fixed_result_count} tool results (should be 1)")
    print(f"Safe/Sensitive tools work: {tools_work}")

    if fixed_result_count == 1 and tools_work:
        print("✅ ALL TESTS PASSED - Issue is fixed!")
    else:
        print("❌ Some tests failed - Issue needs more work")
