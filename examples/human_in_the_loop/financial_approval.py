import uuid
from typing import Annotated, TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ApprovalNode
from langgraph.types import Command

# 1. Define State
class State(TypedDict):
    amount: float
    currency: str
    recipient: str
    status: str
    approval_result: dict | None

# 2. Define Nodes
def process_payment(state: State):
    print(f"Processing payment of {state['amount']} {state['currency']} to {state['recipient']}...")
    return {"status": "completed"}

def handle_rejection(state: State):
    print("Payment rejected by human.")
    return {"status": "rejected"}

# 3. Build Graph
builder = StateGraph(State)

# Add approval node
approval_node = ApprovalNode(
    prompt="Please approve this financial transaction.",
    state_key="approval_result"
)
builder.add_node("approval", approval_node)

builder.add_node("process_payment", process_payment)
builder.add_node("handle_rejection", handle_rejection)

# Define routing logic
def route_after_approval(state: State):
    if state["approval_result"]["action"] == "approve":
        return "process_payment"
    else:
        return "handle_rejection"

builder.add_edge(START, "approval")
builder.add_conditional_edges(
    "approval",
    route_after_approval,
    {
        "process_payment": "process_payment",
        "handle_rejection": "handle_rejection"
    }
)
builder.add_edge("process_payment", END)
builder.add_edge("handle_rejection", END)

# 4. Compile and Run
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

print("--- Starting workflow ---")
initial_input = {
    "amount": 1000.0,
    "currency": "USD",
    "recipient": "Alice"
}

for chunk in graph.stream(initial_input, config):
    print(chunk)

# The graph is now interrupted at the 'approval' node.
print("\n--- Human intervention required ---")
print("Simulating human approval...")

# Resume with approval
resume_command = Command(resume={"action": "approve", "data": {"notes": "Approved by CFO"}})

for chunk in graph.stream(resume_command, config):
    print(chunk)

print("\n--- Workflow finished ---")
final_state = graph.get_state(config).values
print(f"Final Status: {final_state['status']}")
print(f"Approval Result: {final_state['approval_result']}")
