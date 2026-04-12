import os
from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_suprawall import SupraWallCallbackHandler

# 1. Define Tools
@tool
def get_customer_history(customer_id: str):
    """Retrieves interaction history for a customer."""
    return f"Customer {customer_id} has 2 years tenure."

@tool
def process_refund(customer_id: str, amount: float):
    """Processes a refund. Authorized for billing disputes."""
    return f"Refund of ${amount} processed for {customer_id}."

@tool
def delete_customer_record(customer_id: str):
    """DESTRUCTIVE: Deletes customer data."""
    return f"Customer {customer_id} deleted."

# 2. Define State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]

# 3. Define Nodes with SupraWall Callback
model = ChatOpenAI(model="gpt-4-turbo-preview")

def support_node(state: AgentState):
    sw = SupraWallCallbackHandler(agent_id="support-agent")
    return {"messages": [model.bind_tools([get_customer_history]).invoke(state["messages"], config={"callbacks": [sw]})]}

def refund_node(state: AgentState):
    sw = SupraWallCallbackHandler(agent_id="refund-agent")
    return {"messages": [model.bind_tools([process_refund, get_customer_history]).invoke(state["messages"], config={"callbacks": [sw]})]}

def ops_node(state: AgentState):
    sw = SupraWallCallbackHandler(agent_id="ops-agent")
    return {"messages": [model.bind_tools([delete_customer_record, get_customer_history]).invoke(state["messages"], config={"callbacks": [sw]})]}

# 4. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("support", support_node)
workflow.add_node("refund", refund_node)
workflow.add_node("ops", ops_node)

def route(state: AgentState):
    last_msg = state["messages"][-1].content.lower()
    if "refund" in last_msg: return "refund"
    if "delete" in last_msg: return "ops"
    return END

workflow.set_entry_point("support")
workflow.add_conditional_edges("support", route, {"refund": "refund", "ops": "ops", "end": END})
workflow.add_edge("refund", END)
workflow.add_edge("ops", END)

app = workflow.compile()

if __name__ == "__main__":
    for chunk in app.stream({"messages": [HumanMessage(content="Refund $50 for CUST-1")]}):
        print(chunk)
