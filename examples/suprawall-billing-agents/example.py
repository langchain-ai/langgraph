import os
from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_suprawall import SupraWallCallbackHandler

# ---------------------------------------------------------------------------
# 1. Define Tools
# ---------------------------------------------------------------------------

@tool
def get_customer_history(customer_id: str):
    """Retrieves the interaction and purchase history for a customer."""
    return f"Customer {customer_id} has been a member for 2 years. Last purchase: $45.00."

@tool
def process_refund(customer_id: str, amount: float):
    """Processes a refund for a customer. Required for billing disputes."""
    return f"Refund of ${amount} processed for customer {customer_id}."

@tool
def delete_customer_record(customer_id: str):
    """DESTRUCTIVE: Permanently deletes a customer and all their data."""
    return f"Customer {customer_id} record deleted indefinitely."

# ---------------------------------------------------------------------------
# 2. Define State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    # Keep track of which agent is currently active for routing
    active_agent: str

# ---------------------------------------------------------------------------
# 3. Define Agent Nodes
# Each node uses the SupraWallCallbackHandler with a specific agent_id.
# This ensures SupraWall can enforce role-based policies per agent.
# ---------------------------------------------------------------------------

model = ChatOpenAI(model="gpt-4-turbo-preview")

def support_agent(state: AgentState):
    """General support agent. Can query history but not move money."""
    sw_callback = SupraWallCallbackHandler(agent_id="support-agent")
    tools = [get_customer_history]
    llm_with_tools = model.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"], config={"callbacks": [sw_callback]})
    return {"messages": [response], "active_agent": "support"}

def refund_agent(state: AgentState):
    """Refund specialist. Authorized for financial operations."""
    sw_callback = SupraWallCallbackHandler(agent_id="refund-agent")
    tools = [process_refund, get_customer_history]
    llm_with_tools = model.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"], config={"callbacks": [sw_callback]})
    return {"messages": [response], "active_agent": "refund"}

def ops_agent(state: AgentState):
    """Ops agent. Has destructive permissions, strictly guarded."""
    sw_callback = SupraWallCallbackHandler(agent_id="ops-agent")
    tools = [delete_customer_record, get_customer_history]
    llm_with_tools = model.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"], config={"callbacks": [sw_callback]})
    return {"messages": [response], "active_agent": "ops"}

# ---------------------------------------------------------------------------
# 4. Build the Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("support", support_agent)
workflow.add_node("refund", refund_agent)
workflow.add_node("ops", ops_agent)

# Simple Router logic based on user intent
def router(state: AgentState):
    last_message = state["messages"][-1].content.lower()
    if "refund" in last_message or "chargeback" in last_message:
        return "refund"
    if "delete" in last_message or "account cancellation" in last_message:
        return "ops"
    return "support"

workflow.set_entry_point("support")
workflow.add_conditional_edges("support", router, {"refund": "refund", "ops": "ops", "support": "support"})
workflow.add_edge("refund", END)
workflow.add_edge("ops", END)

app = workflow.compile()

# ---------------------------------------------------------------------------
# 5. Run Demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test Scenario 1: Safe query (Support Agent)
    print("\n--- Scenario 1: General History Query ---")
    for chunk in app.stream({"messages": [HumanMessage(content="What is the history for customer CUST-123?")]}):
        print(chunk)

    # Test Scenario 2: Authorized Refund (Refund Agent)
    print("\n--- Scenario 2: Processing a Refund ---")
    try:
        for chunk in app.stream({"messages": [HumanMessage(content="Please refund $40 for customer CUST-123")]}):
            print(chunk)
    except Exception as e:
        print(f"\n[BLOCKED BY SUPRAWALL] {e}")

    # Test Scenario 3: Blocked/Approval Required Action (Ops Agent)
    print("\n--- Scenario 3: Destructive Delete Action ---")
    try:
        for chunk in app.stream({"messages": [HumanMessage(content="Delete customer CUST-123 immediately.")]}):
            print(chunk)
    except Exception as e:
        print(f"\n[BLOCKED BY SUPRAWALL] {e}")
