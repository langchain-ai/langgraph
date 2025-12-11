"""
Test script to demonstrate versions_seen in StateGraph

Graph structure:
    nodeA -> nodeB + nodeC -> nodeD

State:
    fieldA: str
    fieldB: str
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from pprint import pprint
import json


class State(TypedDict):
    fieldA: str
    fieldB: str


class StateOnlyA(TypedDict):
    """Input schema for nodeB - only reads fieldA"""
    fieldA: str


class StateOnlyB(TypedDict):
    """Input schema for nodeC - only reads fieldB"""
    fieldB: str


def nodeA(state: State) -> dict:
    """Reads fieldA + fieldB"""
    print(f"  [nodeA] Reading: fieldA='{state['fieldA']}', fieldB='{state['fieldB']}'")
    return {"fieldA": state["fieldA"] + "->A", "fieldB": state["fieldB"] + "->A"}


def nodeB(state: StateOnlyA) -> dict:
    """Reads only fieldA"""
    print(f"  [nodeB] Reading: fieldA='{state['fieldA']}'")
    return {"fieldA": state["fieldA"] + "->B"}


def nodeC(state: StateOnlyB) -> dict:
    """Reads only fieldB"""
    print(f"  [nodeC] Reading: fieldB='{state['fieldB']}'")
    return {"fieldB": state["fieldB"] + "->C"}


def nodeD(state: State) -> dict:
    """Reads fieldA + fieldB"""
    print(f"  [nodeD] Reading: fieldA='{state['fieldA']}', fieldB='{state['fieldB']}'")
    return {"fieldA": state["fieldA"] + "->D", "fieldB": state["fieldB"] + "->D"}


# Build the graph
graph = StateGraph(State)

graph.add_node("nodeA", nodeA)  # reads fieldA + fieldB (default: full state)
graph.add_node("nodeB", nodeB, input_schema=StateOnlyA)  # reads only fieldA
graph.add_node("nodeC", nodeC, input_schema=StateOnlyB)  # reads only fieldB
graph.add_node("nodeD", nodeD)  # reads fieldA + fieldB (default: full state)

graph.add_edge(START, "nodeA")
graph.add_edge("nodeA", "nodeB")
graph.add_edge("nodeA", "nodeC")
graph.add_edge(["nodeB", "nodeC"], "nodeD")
graph.add_edge("nodeD", END)

# Compile with checkpointer
checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Print compiled graph info
print("=" * 60)
print("COMPILED GRAPH INFO")
print("=" * 60)
print("\nChannels created:")
for name, channel in app.channels.items():
    print(f"  - {name}: {type(channel).__name__}")

print("\nNodes with their triggers and channels:")
for name, node in app.nodes.items():
    print(f"  - {name}:")
    print(f"      triggers: {node.triggers}")
    print(f"      channels: {node.channels}")

# Run the graph
print("\n" + "=" * 60)
print("EXECUTION")
print("=" * 60)

config = {"configurable": {"thread_id": "test-1"}}
input_state = {"fieldA": "Hello", "fieldB": "World"}

print(f"\nInput: {input_state}\n")

# Run the graph to completion
result = app.invoke(input_state, config)
print(f"Final result: {result}\n")

# Now use get_state_history to get all checkpoints in order
print("=" * 60)
print("CHECKPOINT HISTORY (using get_state_history)")
print("=" * 60)

# get_state_history returns checkpoints in reverse order (newest first)
history = list(app.get_state_history(config))
history.reverse()  # Reverse to get oldest first

for idx, state_snapshot in enumerate(history):
    metadata = state_snapshot.metadata
    
    print(f"\n{'='*60}")
    print(f"Step {metadata.get('step', '?')} - Source: {metadata.get('source', '?')}")
    print(f"{'='*60}")
    
    # Show which node(s) wrote this checkpoint
    if "writes" in metadata and metadata["writes"]:
        print(f"Writes by: {list(metadata['writes'].keys())}")
    
    print(f"\nState values: {state_snapshot.values}")
    
    # Access the actual checkpoint data
    checkpoint_tuple = checkpointer.get_tuple(state_snapshot.config)
    if checkpoint_tuple:
        cp = checkpoint_tuple.checkpoint
        
        # Helper to simplify version string
        def simplify_version(ver):
            return str(ver).split(".")[0][-2:] if "." in str(ver) else str(ver)
        
        # Pretty print checkpoint with simplified versions
        print("\nCheckpoint (raw):")
        print(f"  v: {cp['v']}")
        print(f"  id: {cp['id'][:20]}...")
        print(f"  ts: {cp['ts']}")
        print(f"  updated_channels: {cp.get('updated_channels')}")
        
        print(f"\n  channel_values:")
        for ch, val in sorted(cp["channel_values"].items()):
            val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
            print(f"    {ch}: {val_str}")
        
        print(f"\n  channel_versions:")
        for ch, ver in sorted(cp["channel_versions"].items()):
            print(f"    {ch}: v{simplify_version(ver)}")
        
        print(f"\n  versions_seen:")
        for node_name, seen in sorted(cp["versions_seen"].items()):
            if seen:
                print(f"    {node_name}:")
                for ch, ver in sorted(seen.items()):
                    print(f"        {ch}: v{simplify_version(ver)}")
            else:
                print(f"    {node_name}: {{}}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key observations:
1. versions_seen only records TRIGGER channels (branch:to:*, join:*)
2. State channels (fieldA, fieldB) are NEVER in versions_seen
3. Each node only records the trigger channel that activated it
""")

