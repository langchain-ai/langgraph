"""
Fixed version demonstrating proper human-in-the-loop implementation.

This example shows the correct way to implement human-in-the-loop:
1. Import InMemorySaver and compile with checkpointer=InMemorySaver()
2. Pass config={"configurable": {"thread_id": "<uuid>"}} when invoking
3. Keep the interrupt wrapper and HumanInterrupt payload structure unchanged

This ensures interrupts hit a persisted checkpoint that the UI can resume from.
"""

import uuid
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage

# Import from the correct path
try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    # Fallback for different versions  
    try:
        from langgraph.checkpoint.memory import MemorySaver as InMemorySaver
    except ImportError:
        from langgraph.checkpoint import MemorySaver as InMemorySaver


@tool
def book_hotel(hotel_name: str) -> str:
    """Book a hotel reservation with human-in-the-loop review."""
    # This interrupt will now work properly with checkpointer and thread_id
    user_approval = interrupt({
        "action": "book_hotel", 
        "args": {"hotel_name": hotel_name},
        "message": f"Do you want to book a stay at {hotel_name}?"
    })
    
    if user_approval:
        return f"Successfully booked a stay at {hotel_name}."
    else:
        return "Booking cancelled by user."


def chatbot(state: MessagesState):
    """Chatbot node that can call tools."""
    # For simplicity, we'll just simulate calling the tool directly
    message = state["messages"][-1]
    if "book" in message.content.lower() and "hotel" in message.content.lower():
        # Simulate the LLM deciding to call the book_hotel tool
        from langchain_core.messages import AIMessage
        tool_call = {
            "name": "book_hotel",
            "args": {"hotel_name": "Grand Hotel"},
            "id": "call_123",
            "type": "tool_call"
        }
        ai_message = AIMessage(content="I'll help you book a hotel.", tool_calls=[tool_call])
        return {"messages": [ai_message]}
    
    return {"messages": []}


# FIXED: Create checkpointer
checkpointer = InMemorySaver()

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode([book_hotel]))

builder.add_edge("__start__", "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

# FIXED: Compile with checkpointer to enable persistence for interrupts
graph = builder.compile(checkpointer=checkpointer)
graph.name = "MemoryAgent"  # Optional: set a name for the graph


if __name__ == "__main__":
    # FIXED: Pass thread_id in config for resumable runs
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print("=== Running with proper human-in-the-loop setup ===")
    
    # First invoke - should hit the interrupt
    result = graph.invoke(
        {"messages": [HumanMessage("Please book me a hotel")]}, 
        config=thread_config
    )
    
    print("First invoke result:", result)
    
    # Check if we got an interrupt
    if "__interrupt__" in result:
        print("\n✅ SUCCESS: Interrupt triggered! Human review panel should now show.")
        print("Interrupt details:", result["__interrupt__"])
        
        # Simulate human approval and resume
        print("\nSimulating human approval...")
        resume_result = graph.invoke(
            Command(resume=True),  # Human approves the action
            config=thread_config
        )
        print("Resume result:", resume_result)
    else:
        print("\n❌ No interrupt found. Something is still wrong.")
        
    print("\n=== Human-in-the-loop flow complete ===")
    print("The graph can now be used with Agent Chat UI or dev server.")
    print("Each chat session should use a unique thread_id for proper state management.")
