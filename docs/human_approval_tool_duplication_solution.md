# Solution for Multiple Tool Results in Human Approval Flow (Issue #4397)

## Problem Summary

The issue occurs when using LangGraph's human approval flow with sensitive tools. The user reported getting the error:

```
InvokeModelWithResponseStream operation: messages.4: Too many tool_result blocks found: 3, expected 1. 
The number of tool_result blocks must match the number of tool_use blocks in the previous message.
```

This happens because the same tool is being executed multiple times, creating duplicate `ToolMessage` objects for a single tool call.

## Root Cause Analysis

After analyzing the user's code and LangGraph's architecture, the issue stems from several problems:

### 1. Flawed `tool_result_handler` Logic

The original `tool_result_handler` function has a critical bug:

```python
# PROBLEMATIC CODE
def tool_result_handler(state: GraphState) -> GraphState:
    # ... setup code ...
    for tool_call in tool_use_message.tool_calls:
        call_id = tool_call["id"]
        if call_id in existing_result_ids:
            continue
        tool_result_message = next(
            (m for m in state["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == call_id),
            None
        )
        if tool_result_message:
            new_results.append(tool_result_message)
            break  # BUG: This break prevents checking other tool calls
    # ...
```

**Issues:**
- The `break` statement prevents proper checking of all tool calls
- The logic searches for existing tool results but then re-adds them
- The function doesn't properly prevent duplicate executions

### 2. Graph Flow Issues

The human approval flow can cause tools to be executed multiple times:
1. Tool call is made → routed to human approval
2. After approval → tool is executed → result added
3. Flow continues → tool_result_handler processes the same tool call again
4. Multiple executions create duplicate results

### 3. Architectural Anti-Pattern

The `tool_result_handler` as implemented violates LangGraph's design principles:
- Tool execution should happen in dedicated tool nodes
- Result handling should be automatic via `add_messages`
- Manual result manipulation should be minimal

## Solution

### 1. Fixed Tool Result Handler

```python
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
```

### 2. Improved Tool Execution Nodes

```python
def safe_tool_executor(state: GraphState) -> GraphState:
    """Execute safe tools directly with duplicate prevention."""
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
            
        # Execute tool and create result
        if tool_call["name"] in SAFE_TOOLS:
            result = execute_tool(tool_call)
            new_results.append(ToolMessage(
                content=result,
                tool_call_id=call_id,
                name=tool_call["name"]
            ))
    
    if new_results:
        return {"messages": new_results}
    return state


def sensitive_tool_executor(state: GraphState) -> GraphState:
    """Execute sensitive tools after human approval with duplicate prevention."""
    # Find the most recent AI message with tool calls
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
            
        # Execute tool and create result
        if tool_call["name"] in SENSITIVE_TOOLS:
            result = execute_tool(tool_call)
            new_results.append(ToolMessage(
                content=result,
                tool_call_id=call_id,
                name=tool_call["name"]
            ))
    
    if new_results:
        return {"messages": new_results}
    return state
```

### 3. Simplified Graph Architecture

**Recommended approach:**

```python
def create_fixed_graph() -> StateGraph:
    """Create a graph with proper tool execution flow."""
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tool_router", tool_routing_node)
    graph.add_node("safe_tools", safe_tool_executor)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("sensitive_tools", sensitive_tool_executor)
    
    # Add edges - simplified flow
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "tool_router")
    
    # Conditional edges from router
    graph.add_conditional_edges(
        "tool_router",
        lambda state: route_tools(state),
        {
            "safe": "safe_tools",
            "sensitive": "human_approval",
            "end": END
        }
    )
    
    # Direct edges to end (no separate result handler needed)
    graph.add_edge("safe_tools", END)
    graph.add_edge("human_approval", "sensitive_tools")
    graph.add_edge("sensitive_tools", END)
    
    return graph
```

## Key Improvements

### 1. Duplicate Prevention
- Check existing tool result IDs before execution
- Skip tools that already have results
- Use proper tool call ID matching

### 2. Simplified Flow
- Remove unnecessary `tool_result_handler` node
- Let `add_messages` handle result aggregation automatically
- Direct flow from tool execution to END

### 3. Proper State Management
- Use LangGraph's built-in message handling
- Avoid manual message manipulation
- Leverage `add_messages` for automatic deduplication

### 4. Better Error Handling
- Validate tool calls before execution
- Handle missing or malformed tool calls gracefully
- Provide clear error messages

## Best Practices for Human-in-the-Loop Tool Execution

### 1. Use LangGraph's Built-in Patterns
```python
# Use ToolNode for standard tool execution
from langgraph.prebuilt import ToolNode

# For human approval, use interrupt() properly
def human_approval_node(state):
    approval = interrupt({
        "message": "Approve this tool call?",
        "tool_call": state["messages"][-1].tool_calls[0]
    })
    
    if approval.get("approved"):
        return Command(goto="execute_tool")
    else:
        return Command(goto=END)
```

### 2. Leverage State Annotations
```python
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # Auto-handles duplicates
    # other fields...
```

### 3. Minimal Custom Logic
- Let LangGraph handle message flow
- Only add custom logic where necessary
- Use built-in components when possible

## Testing the Fix

The provided test file (`test_human_approval_tool_duplication_fix.py`) demonstrates:
1. How the bug occurs with the original code
2. How the fix prevents duplicate tool results
3. Proper tool execution patterns

Run the test to verify the solution works correctly.
