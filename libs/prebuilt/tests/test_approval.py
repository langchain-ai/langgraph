import uuid
from typing import TypedDict, Optional, Annotated
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from langgraph.prebuilt.approval import (
    ApprovalNode,
)

class State(TypedDict):
    approval_result: Optional[dict]
    data: str

def test_approval_node_approve():
    builder = StateGraph(State)
    builder.add_node("approval", ApprovalNode(prompt="Approve?"))
    builder.add_edge(START, "approval")
    
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # 1. Start execution
    events = list(graph.stream({"data": "test"}, config))
    assert len(events) == 1
    # Check for interrupt
    assert "__interrupt__" in events[0]
    interrupt_obj = events[0]["__interrupt__"][0]
    assert interrupt_obj.value["prompt"] == "Approve?"
    
    # 2. Resume with approve
    resumed_events = list(graph.stream(
        Command(resume={"action": "approve", "data": "approved_data"}),
        config
    ))
    
    # Verify state update
    final_state = graph.get_state(config).values
    assert final_state["approval_result"]["action"] == "approve"
    assert final_state["approval_result"]["data"] == "approved_data"

def test_approval_node_reject():
    builder = StateGraph(State)
    builder.add_node("approval", ApprovalNode(prompt="Approve?"))
    builder.add_edge(START, "approval")
    
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # 1. Start execution
    list(graph.stream({"data": "test"}, config))
    
    # 2. Resume with reject
    list(graph.stream(
        Command(resume={"action": "reject", "reason": "too risky"}),
        config
    ))
    
    # Verify state update
    final_state = graph.get_state(config).values
    assert final_state["approval_result"]["action"] == "reject"
    assert final_state["approval_result"]["reason"] == "too risky"

def test_approval_node_modify():
    builder = StateGraph(State)
    builder.add_node("approval", ApprovalNode(prompt="Approve?"))
    builder.add_edge(START, "approval")
    
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # 1. Start execution
    list(graph.stream({"data": "test"}, config))
    
    # 2. Resume with modify
    list(graph.stream(
        Command(resume={"action": "modify", "data": "modified_data"}),
        config
    ))
    
    # Verify state update
    final_state = graph.get_state(config).values
    assert final_state["approval_result"]["action"] == "modify"
    assert final_state["approval_result"]["data"] == "modified_data"

def test_graph_pause_resume():
    # Test the new Pregel.pause() and Pregel.resume() methods
    builder = StateGraph(State)
    def node_1(state): return {"data": "step1"}
    builder.add_node("node1", node_1)
    builder.add_edge(START, "node1")
    
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "pause-test"}}
    
    # Test resume convenience method
    # First, let's manually interrupt it
    def interrupt_node(state):
        from langgraph.types import interrupt
        val = interrupt("wait")
        return {"data": val}
        
    builder_with_int = StateGraph(State)
    builder_with_int.add_node("int", interrupt_node)
    builder_with_int.add_edge(START, "int")
    graph_int = builder_with_int.compile(checkpointer=checkpointer)
    
    list(graph_int.stream({"data": "test"}, config))
    
    # Resume using the new method
    graph_int.resume(config, value="resumed")
    
    assert graph_int.get_state(config).values["data"] == "resumed"
