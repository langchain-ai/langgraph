import time
import uuid
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class State(TypedDict):
    count: int

def node(state):
    return {"count": state["count"] + 1}

def test_pause_resume_latency():
    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Measure baseline invoke
    start = time.perf_counter()
    graph.invoke({"count": 0}, config)
    end = time.perf_counter()
    print(f"Baseline invoke latency: {(end - start) * 1000:.2f}ms")
    
    # Measure resume latency
    # 1. Create an interrupt
    def int_node(state):
        from langgraph.types import interrupt
        interrupt("pause")
        return {"count": state["count"] + 1}
        
    builder_int = StateGraph(State)
    builder_int.add_node("node", int_node)
    builder_int.add_edge(START, "node")
    builder_int.add_edge("node", END)
    graph_int = builder_int.compile(checkpointer=checkpointer)
    
    config_int = {"configurable": {"thread_id": str(uuid.uuid4())}}
    list(graph_int.stream({"count": 0}, config_int)) # Trigger interrupt
    
    start = time.perf_counter()
    graph_int.resume(config_int, value="ok")
    end = time.perf_counter()
    print(f"Resume latency: {(end - start) * 1000:.2f}ms")

if __name__ == "__main__":
    test_pause_resume_latency()
