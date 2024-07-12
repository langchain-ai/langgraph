from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


def test_multiple_interruptions():
    class State(TypedDict):
      input: str
    
    def step_1(_state):
       pass
      
    def step_2(_state):
        pass

    def step_3(_state):
        pass
      
    builder = StateGraph(State)
    builder.add_node("step_1", step_1)
    builder.add_node("step_2", step_2)
    builder.add_node("step_3", step_3)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)

    memory = MemorySaver()

    graph = builder.compile(checkpointer=memory, interrupt_after="*")

    initial_input = {"input": "hello world"}
    thread = {"configurable": {"thread_id": "1"}}

    graph.invoke(initial_input, thread, stream_mode="values")
    assert(graph.get_state(thread).next == ("step_2",))

    graph.invoke(None, thread, stream_mode="values")
    assert(graph.get_state(thread).next == ("step_3",))
