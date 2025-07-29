#!/usr/bin/env python3
"""Simple test to check if basic subgraph functionality still works."""

import sys
import os

# Add the langgraph library to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'langgraph'))

from dataclasses import dataclass
from langgraph.graph.state import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


@dataclass
class Context:
    username: str

class State(TypedDict):
    my_key: str

# Simple subgraph without runtime context
def simple_node(state: State):
    return {'my_key': state['my_key'] + ' processed'}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node('simple_node', simple_node)
subgraph_builder.set_entry_point('simple_node')
subgraph_builder.set_finish_point('simple_node')
subgraph = subgraph_builder.compile()

# Parent graph
def main_node(state: State):
    return {'my_key': 'hello'}

builder = StateGraph(State)
builder.add_node('main_node', main_node)
builder.add_node('subgraph_node', subgraph)
builder.set_entry_point('main_node')
builder.add_edge('main_node', 'subgraph_node')
builder.set_finish_point('subgraph_node')
graph = builder.compile()

# Test basic functionality
try:
    result = graph.invoke({'my_key': 'start'})
    print(f"Basic subgraph test SUCCESS: {result}")
    
    # Test graph drawing
    try:
        mermaid = graph.get_graph().draw_mermaid(with_styles=False)
        print("Graph drawing SUCCESS")
        print(f"Mermaid: {mermaid[:100]}...")
    except Exception as e:
        print(f"Graph drawing FAILED: {e}")
    
    sys.exit(0)
except Exception as e:
    print(f"Basic subgraph test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
