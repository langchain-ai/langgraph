import uuid
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ApprovalNode
from langgraph.types import Command

class State(TypedDict):
    customer_query: str
    draft_response: str
    approval_result: dict | None
    final_response: str

def draft_response(state: State):
    return {"draft_response": f"Dear Customer, thank you for your query: {state['customer_query']}. We will look into it."}

def send_response(state: State):
    response = state.get("approval_result", {}).get("data", state["draft_response"])
    print(f"Sending response: {response}")
    return {"final_response": response}

builder = StateGraph(State)
builder.add_node("draft", draft_response)
builder.add_node("approval", ApprovalNode(prompt="Review the draft response before sending."))
builder.add_node("send", send_response)

builder.add_edge(START, "draft")
builder.add_edge("draft", "approval")
builder.add_edge("approval", "send")
builder.add_edge("send", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "support-123"}}

print("--- Step 1: Drafting ---")
for chunk in graph.stream({"customer_query": "I want a refund."}, config):
    print(chunk)

print("\n--- Step 2: Human Review ---")
# Simulating a modification by human
resume_command = Command(resume={
    "action": "modify", 
    "data": "Dear Customer, we have approved your refund request."
})

for chunk in graph.stream(resume_command, config):
    print(chunk)

print("\n--- Workflow finished ---")
final_state = graph.get_state(config).values
print(f"Final Response: {final_state['final_response']}")
