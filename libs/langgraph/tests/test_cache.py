from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, END
from langgraph.types import CachePolicy
from typing import TypedDict
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
import time
import random
import hashlib

@pytest.mark.parametrize("checkpointer_name", ["postgres"])
def test_cached_node_executes_once(request: pytest.FixtureRequest, checkpointer_name: str):
    """Test that an indefinitely cached node will not be executed more than once"""

    checkpointer: PostgresSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    call_count = 0
    class State(TypedDict):
        stop_condition: int

    def cached_node(input: State):
        nonlocal call_count
        call_count += 1
        pass

    def cache_key(input: State, config: RunnableConfig):
        return ""

    def dummy_node(input: State):
        return {"stop_condition": input["stop_condition"] + 1}
    
    config = {"configurable": {"thread_id": "thread-1"}}
    cache = CachePolicy(cache_key=cache_key)
    builder = StateGraph(State)
    builder.add_node("cached_node", cached_node, cache=cache)
    builder.add_node("dummy_node", dummy_node)
    builder.add_edge("cached_node", "dummy_node")
    builder.add_conditional_edges("dummy_node", lambda x: END if x["stop_condition"] >= 10 else "cached_node")
    builder.set_entry_point("cached_node")

    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"stop_condition": 0}, config)

    assert call_count == 1
    
@pytest.mark.parametrize("checkpointer_name", ["postgres"])
def test_cache_key_single_field(request: pytest.FixtureRequest, checkpointer_name: str):
    """Test that a cached node with a cache key specifying a single field of the input will 
    be executed only when a value of the field has not been seen before"""

    checkpointer: PostgresSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    call_count = 0
    target_call_count = random.randint(1, 10)

    class State(TypedDict):
        dependent_field: int
        stop_condition: int

    def cached_node(input: State):
        nonlocal call_count
        call_count += 1
        pass

    def cache_key(input: State, config: RunnableConfig):
        s = str(input["dependent_field"])
        return hashlib.md5(s.encode()).hexdigest()

    def dummy_node(input: State):
        return {
            "stop_condition": input["stop_condition"] + 1, 
            "dependent_field": (input["dependent_field"] + 1) % target_call_count
        }
    
    config = {"configurable": {"thread_id": "thread-1"}}
    cache = CachePolicy(cache_key=cache_key)
    builder = StateGraph(State)
    builder.add_node("cached_node", cached_node, cache=cache)
    builder.add_node("dummy_node", dummy_node)
    builder.add_edge("cached_node", "dummy_node")
    builder.add_conditional_edges("dummy_node", lambda x: END if x["stop_condition"] >= 10 else "cached_node")
    builder.set_entry_point("cached_node")

    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"stop_condition": 0, "dependent_field": 0}, config)

    assert call_count == target_call_count

@pytest.mark.parametrize("checkpointer_name", ["postgres"])
def test_cache_key_multiple_fields(request: pytest.FixtureRequest, checkpointer_name: str):
    """Test that a cached node with a cache key specifying a subset of input fields  will 
    be executed only when a value of the fields has not been seen before"""

    checkpointer: PostgresSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )
    call_count = 0

    class State(TypedDict):
        dependent_field_1: int
        dependent_field_2: int
        stop_condition: int

    def cached_node(input: State):
        nonlocal call_count
        call_count += 1
        pass

    # Pulls from cache only if values from 2 input fields have been seen before
    def cache_key(input: State, config: RunnableConfig):
        fields = ["dependent_field_1", "dependent_field_2"]
        s = "".join([f"{field}{val}" for field, val in input.items() if field in fields])
        return hashlib.md5(s.encode()).hexdigest()

    # Should only pull for values (0,0), (0,1), which occurs 4 times in 10 cycles
    def dummy_node(input: State):
        return {
            "stop_condition": input["stop_condition"] + 1, 
            "dependent_field_1": (input["dependent_field_1"] + 1) % 2,
            "dependent_field_2": (input["dependent_field_2"] + 1) % 4,
        }
    
    config = {"configurable": {"thread_id": "thread-1"}}
    cache = CachePolicy(cache_key=cache_key)
    builder = StateGraph(State)
    builder.add_node("cached_node", cached_node, cache=cache)
    builder.add_node("dummy_node", dummy_node)
    builder.add_edge("cached_node", "dummy_node")
    builder.add_conditional_edges("dummy_node", lambda x: END if x["stop_condition"] >= 10 else "cached_node")
    builder.set_entry_point("cached_node")

    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"stop_condition": 0, "dependent_field_1": 0, "dependent_field_2": 0}, config)

    assert call_count == 4

@pytest.mark.parametrize("checkpointer_name", ["postgres"])
def test_multiple_cached_nodes(request: pytest.FixtureRequest, checkpointer_name: str):
    """Test call counts of multiple cached nodes with different cache keys in one graph"""

    checkpointer: PostgresSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    call_count_1 = 0
    call_count_2 = 0

    class State(TypedDict):
        dependent_field_1: int
        dependent_field_2: int
        stop_condition: int

    def cached_node_1(input: State):
        nonlocal call_count_1
        call_count_1 += 1
        pass

    def cached_node_2(input: State):
        nonlocal call_count_2
        call_count_2 += 1
        pass

    # node 1 alternates bw 3 cached states
    def cache_key_1(input: State, config: RunnableConfig):
        s = ""
        if input["dependent_field_1"] % 3 == 0:
            s += "key_1"
        elif input["dependent_field_1"] % 3 == 1:
            s += "key_2"
        else:
            s += "key_3"
        return hashlib.md5(s.encode()).hexdigest()
    
    # node 1 alternates bw 2 cached states
    def cache_key_2(input: State, config: RunnableConfig):
        s = "key_4" if input["dependent_field_2"] % 2 == 0 else "key_5"
        return hashlib.md5(s.encode()).hexdigest()

    def dummy_node(input: State):
        return {
            "stop_condition": input["stop_condition"] + 1, 
            "dependent_field_1": input["dependent_field_1"] + 1,
            "dependent_field_2": input["dependent_field_2"] + 1
        }
    
    config: RunnableConfig = {"configurable": {"thread_id": "thread-1"}, "recursion_limit": 31}

    builder = StateGraph(State)
    builder.add_node("cached_node_1", cached_node_1, cache=CachePolicy(cache_key=cache_key_1))
    builder.add_node("cached_node_2", cached_node_2, cache=CachePolicy(cache_key=cache_key_2))
    builder.add_node("dummy_node", dummy_node)
    builder.add_edge("cached_node_1", "cached_node_2")
    builder.add_edge("cached_node_2", "dummy_node")
    builder.add_conditional_edges("dummy_node", lambda x: END if x["stop_condition"] >= 10 else "cached_node_1")
    builder.set_entry_point("cached_node_1")

    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"stop_condition": 0, "dependent_field_1": 0, "dependent_field_2": 0}, config)

    assert call_count_1 == 3
    assert call_count_2 == 2
    

@pytest.mark.parametrize("checkpointer_name", ["postgres"])
def test_cache_ttl(request: pytest.FixtureRequest, checkpointer_name: str):
    """Test that a cached node defined with ttl does not retrieve cached writes after they've 
    expired"""

    checkpointer: PostgresSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )
    call_count = 0

    class State(TypedDict):
        stop_condition: int

    def cached_node(input: State):
        nonlocal call_count
        call_count += 1
        pass

    def cache_key(input: State, config: RunnableConfig):
        return ""

    # node sleeps for half of the ttl
    def dummy_node(input: State):
        time.sleep(0.5)
        return {"stop_condition": input["stop_condition"] + 1}
    
    config = {"configurable": {"thread_id": "thread-1"}}
    cache = CachePolicy(cache_key=cache_key, ttl=1)
    builder = StateGraph(State)
    builder.add_node("cached_node", cached_node, cache=cache)
    builder.add_node("dummy_node", dummy_node)
    builder.add_edge("cached_node", "dummy_node")
    builder.add_conditional_edges("dummy_node", lambda x: END if x["stop_condition"] >= 4 else "cached_node")
    builder.set_entry_point("cached_node")

    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"stop_condition": 0}, config)

    # node should execute half of the time since cached write only lives for 2 cycles of the graph
    assert call_count == 2

@pytest.mark.parametrize("checkpointer_name", ["postgres"])
def test_cache_key_field_and_config(request: pytest.FixtureRequest, checkpointer_name: str):
    """Test that an indefinitely cached node will not be executed more than once"""

    checkpointer: PostgresSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    # First we run with config val A

    call_count = 0
    class State(TypedDict):
        stop_condition: int

    def cached_node(input: State):
        nonlocal call_count
        call_count += 1
        pass

    def dummy_node(input: State):
        return {"stop_condition": input["stop_condition"] + 1}
    
    config = {"configurable": {"thread_id": "thread-1"}}

    # Retrieve twice in 10 cycles for same config
    def cache_key(input: State, config: RunnableConfig = config):
        s = ""
        s += "key_1" if input["stop_condition"] % 5 == 0 else "key_2"
        s += config["configurable"]["thread_id"]
        return hashlib.md5(s.encode()).hexdigest()
    
    
    cache = CachePolicy(cache_key=cache_key)
    builder = StateGraph(State)
    builder.add_node("cached_node", cached_node, cache=cache)
    builder.add_node("dummy_node", dummy_node)
    builder.add_edge("cached_node", "dummy_node")
    builder.add_conditional_edges("dummy_node", lambda x: END if x["stop_condition"] >= 10 else "cached_node")
    builder.set_entry_point("cached_node")

    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"stop_condition": 0}, config)

    assert call_count == 2

    # Next run with new config should not be cached, so call count should = 2 again when reset
    call_count = 0

    config = {"configurable": {"thread_id": "thread-2"}}

    graph.invoke({"stop_condition": 0}, config)

    assert call_count == 2