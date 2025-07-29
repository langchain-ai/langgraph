#!/usr/bin/env python3
"""Test script to verify the runtime context propagation fix for subgraphs."""

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
    foo: str

# Subgraph
def subgraph_node_1(state: State, runtime: Runtime[Context]):
    return {'foo': 'hi! ' + runtime.context.username}

subgraph_builder = StateGraph(State, context_schema=Context)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.set_entry_point('subgraph_node_1')
subgraph = subgraph_builder.compile()

# Parent graph
def main_node(state: State, runtime: Runtime[Context]):
    return {'foo': 'hello ' + runtime.context.username}

builder = StateGraph(State, context_schema=Context)
builder.add_node(main_node)
builder.add_node('node_1', subgraph)
builder.set_entry_point('main_node')
builder.add_edge('main_node', 'node_1')
graph = builder.compile()

# Test the fix
try:
    context = Context(username='Alice')
    result = graph.invoke({'foo': 'world'}, context=context)
    print(f"SUCCESS: {result}")
    print("Runtime context is now properly propagated to subgraphs!")
    sys.exit(0)
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
