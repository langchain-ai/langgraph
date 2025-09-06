"""
Minimal reproduction of the human-in-the-loop bug.

This example demonstrates the bug where human-in-the-loop doesn't work
because the graph is compiled without a checkpointer and invokes without thread_id.

This causes interrupts to not be persisted and the UI has nothing resumable to render,
so tool calls proceed as a normal path instead of triggering the review panel.
"""

from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage


@tool
def book_hotel(hotel_name: str) -> str:
    """Book a hotel reservation."""
    # This interrupt should trigger a review panel in the UI, but it won't work
    # without a checkpointer and thread_id
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


# Build the graph WITHOUT a checkpointer (this is the bug)
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode([book_hotel]))

builder.add_edge("__start__", "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

# PROBLEM: Compiling without checkpointer - interrupts won't work!
graph = builder.compile()  # Missing checkpointer=InMemorySaver()

if __name__ == "__main__":
    # PROBLEM: Invoking without thread_id - interrupts can't be persisted!
    result = graph.invoke({"messages": [HumanMessage("Please book me a hotel")]})
    
    print("Result:", result)
    print("\nThis should have triggered an interrupt for human review, but it didn't!")
    print("The tool call proceeded without human intervention.")
