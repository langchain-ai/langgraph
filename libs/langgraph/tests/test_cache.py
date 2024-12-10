from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import CachePolicy
from typing import TypedDict



    
def test_basic_cache():
    cache = CachePolicy(cache_key=lambda x: "hi")
    builder = StateGraph(int)
    builder.add_node("add_two", lambda x: x + 2, cache=cache)
    builder.add_node("subtract_one", lambda x: x-1)
    builder.add_edge("add_two", "subtract_one")
    builder.add_conditional_edges("subtract_one", lambda x: END if x >= 10 else "add_two")
    builder.set_entry_point("add_two")
    config = {"configurable": {"thread_id": "thread-1"}}
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    graph.invoke(1, config) #, debug=True)

    # for elt in list(memory.list(None)):
    #     print(elt)
    #     print()

def test_dict_cache():

    class State(TypedDict):
        foo: int
        bar: int

    def cache_key(inputs: State):
        return str(inputs["bar"])

    cache = CachePolicy(cache_key=cache_key)
    builder = StateGraph(State)
    builder.add_node("add_two", lambda x: {"foo": x["foo"] + 2}, cache=cache)
    builder.add_node("subtract_one", lambda x: {"foo": x["foo"] - 1, "bar": (x["bar"] + 1) % 2})
    builder.add_edge("add_two", "subtract_one")
    builder.add_conditional_edges("subtract_one", lambda x: END if x["foo"] >= 10 else "add_two")
    builder.set_entry_point("add_two")
    config = {"configurable": {"thread_id": "thread-1"}}
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    graph.invoke({"foo": 1, "bar": 1}, config) #, debug=True)

    for elt in list(memory.list(None)):
        print(elt)
        print()