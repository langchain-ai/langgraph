import uuid
from typing import TypedDict, List
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ApprovalNode
from langgraph.types import Command

class State(TypedDict):
    document_content: str
    extracted_data: dict
    approval_result: dict | None

def extract_data(state: State):
    # Simulated extraction
    return {"extracted_data": {"name": "John Doe", "total": 500.0}}

def save_to_db(state: State):
    data = state.get("approval_result", {}).get("data", state["extracted_data"])
    print(f"Saving validated data to DB: {data}")
    return {}

builder = StateGraph(State)
builder.add_node("extract", extract_data)
builder.add_node("validate", ApprovalNode(prompt="Validate the extracted data."))
builder.add_node("save", save_to_db)

builder.add_edge(START, "extract")
builder.add_edge("extract", "validate")
builder.add_edge("validate", "save")
builder.add_edge("save", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "doc-456"}}

print("--- Step 1: Extraction ---")
for chunk in graph.stream({"document_content": "Invoice for John Doe, total 500"}, config):
    print(chunk)

print("\n--- Step 2: Validation ---")
# Simulating human approval with modification
resume_command = Command(resume={
    "action": "modify",
    "data": {"name": "John Doe", "total": 550.0} # Corrected total
})

for chunk in graph.stream(resume_command, config):
    print(chunk)

print("\n--- Workflow finished ---")
