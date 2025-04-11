# StateGraph API Specification

## Overview

`StateGraph` is the primary high-level API in LangGraph for building stateful computation graphs. It represents a graph structure where nodes communicate by reading and writing to a shared state, enabling complex multi-step workflows with LLMs, tools, and other components.

## Constructor

```python
def __init__(
    self,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
    *,
    input: Optional[Type[Any]] = None,
    output: Optional[Type[Any]] = None,
) -> None
```

### Parameters

- **state_schema**: The schema defining the state structure, typically a TypedDict or Pydantic model
- **config_schema**: Optional schema defining configuration parameters
- **input**: Optional schema for graph inputs (defaults to state_schema)
- **output**: Optional schema for graph outputs (defaults to state_schema)

## Core Methods

### Node Management

```python
def add_node(
    self,
    node: Union[str, RunnableLike],
    action: Optional[RunnableLike] = None,
    *,
    metadata: Optional[dict[str, Any]] = None,
    input: Optional[Type[Any]] = None,
    retry: Optional[RetryPolicy] = None,
    destinations: Optional[Union[dict[str, str], tuple[str]]] = None,
    subgraphs: list[PregelProtocol] = EMPTY_SEQ,
) -> Self
```

Adds a new node to the graph. The node can be specified as a string ID with an action callable, or as a Runnable.

```python
def add_sequence(
    self,
    nodes: Sequence[Union[RunnableLike, tuple[str, RunnableLike]]],
) -> Self
```

Adds a sequence of nodes to be executed in order, automatically creating edges between them.

### Edge Management

```python
def add_edge(self, start_key: Union[str, list[str]], end_key: str) -> Self
```

Adds a directed edge from start node to end node. The start node can be a single node or a list of nodes.

```python
def add_conditional_edges(
    self,
    source: str,
    path: Union[Callable[..., Union[Hashable, list[Hashable]]], Runnable[Any, Union[Hashable, list[Hashable]]]],
    path_map: Optional[Union[dict[Hashable, str], list[str]]] = None,
    then: Optional[str] = None,
) -> Self
```

Adds conditional routing logic between nodes. The path callable examines the state and returns a value that determines the next node to execute.

### Graph Entry/Exit

```python
def set_entry_point(self, key: str) -> Self
```

Defines the starting node for graph execution. Only needed if not using START node.

```python
def set_conditional_entry_point(
    self,
    path: Union[Callable[..., Union[Hashable, list[Hashable]]], Runnable[Any, Union[Hashable, list[Hashable]]]],
    path_map: Optional[Union[dict[Hashable, str], list[str]]] = None,
    then: Optional[str] = None,
) -> Self
```

Sets a conditional starting point based on the input state.

```python
def set_finish_point(self, key: str) -> Self
```

Marks a node as an exit point for the graph.

### Validation and Compilation

```python
def validate(self, interrupt: Optional[Sequence[str]] = None) -> Self
```

Checks the graph for correctness, ensuring there are no disconnected nodes or unreachable states. Called by compile.

```python
def compile(
    self,
    checkpointer: Checkpointer = None,
    *,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[Union[All, list[str]]] = None,
    interrupt_after: Optional[Union[All, list[str]]] = None,
    name: Optional[str] = None,
) -> "CompiledStateGraph"
```

Transforms the graph into an executable CompiledStateGraph. The checkpointer enables state persistence.

## Constants

Two special constants are provided for graph construction:

- **START**: Special value representing the entry point to the graph
- **END**: Special value representing an exit point from the graph

## Implementation Details

When a `StateGraph` is compiled, it is transformed into a Pregel instance with:

1. **Node Translation**: Each graph node becomes a `PregelNode` with associated actions
2. **Channel Creation**: State fields are represented as channels with appropriate behaviors
3. **Edge Mapping**: Graph edges determine message routing between nodes
4. **Branch Handling**: Conditional edges are implemented as special routing logic
5. **Checkpoint Configuration**: If provided, enables state persistence and resumption

The StateGraph API handles the complexities of the underlying Pregel execution model, providing a more intuitive interface for building stateful workflows.

## Example Usage

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Define the state schema
class State(TypedDict):
    count: int
    message: str

# Create a StateGraph with our schema
graph = StateGraph(State)

# Add nodes
def increment(state: State):
    return {"count": state["count"] + 1}

def check(state: State):
    if state["count"] >= 3:
        return "finish"
    return "increment"

def finish(state: State):
    return {"message": f"Finished with count {state['count']}"}

graph.add_node("increment", increment)
graph.add_node("check", check)
graph.add_node("finish", finish)

# Add edges
graph.add_edge(START, "increment")
graph.add_edge("increment", "check")
graph.add_conditional_edges("check", check, {"finish": "finish"})
graph.add_edge("finish", END)

# Compile and run
compiled_graph = graph.compile()
result = compiled_graph.invoke({"count": 0, "message": ""})
# result will be {"count": 3, "message": "Finished with count 3"}
```
