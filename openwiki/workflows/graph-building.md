---
type: Workflows & Patterns
title: Graph Building Patterns
description: Guide to constructing StateGraph instances with nodes, edges, branching, error handling, and subgraphs for stateful LangGraph workflows.
tags: [stategraph, nodes, edges, conditional-routing, send, error-handling, subgraphs, method-chaining, messages-state]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-9b01a2a24dd7f7a0fb7e6f05
    resource: repo://libs/langgraph/langgraph/graph/message.py
  - id: openwiki-source-36058e684ab11833a28af83b
    resource: repo://libs/langgraph/langgraph/graph/state.py
  - id: openwiki-source-32221bce0f3eead36a7c7e36
    resource: repo://libs/langgraph/langgraph/types.py
  - id: openwiki-source-359b1259bb077515edbf1b05
    resource: repo://libs/langgraph/tests/test_pregel.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

StateGraph is the primary builder for LangGraph applications. It provides a fluent API for defining stateful workflows where nodes communicate through shared, typed state. This guide covers construction patterns, routing strategies, error recovery, and composition techniques for building robust agentic systems.

Key responsibilities of StateGraph:

- **Node registration** with optional per-node input schemas, retry policies, cache policies, and error handlers
- **Deterministic and conditional edge routing** between nodes, including dynamic fan-out via `Send`
- **State schema definition** with type-safe field definitions and reducer functions for concurrent updates
- **Graph compilation** to an executable `CompiledStateGraph` with optional checkpointing, interrupts, and debug mode
- **Method chaining API** for readable, fluent graph construction

---

## Basic Graph Construction

A minimal StateGraph requires three components: a state schema, at least one node, and at least one edge from START.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int

def increment_node(state: State) -> dict:
    return {"value": state["value"] + 1}

builder = StateGraph(State)
builder.add_node("increment", increment_node)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)

graph = builder.compile()
result = graph.invoke({"value": 0})  # {'value': 1}
```

### StateGraph Constructor

The `StateGraph` constructor accepts:

- **`state_schema`** (required): A `TypedDict`, dataclass, or Pydantic model defining the shared state all nodes read and write.
- **`context_schema`** (optional): Schema for immutable, request-scoped context (e.g., `user_id`, database connections).
- **`input_schema`** (optional): Schema for graph inputs. Defaults to `state_schema`.
- **`output_schema`** (optional): Schema for graph outputs. Defaults to `state_schema`.

```python
from dataclasses import dataclass
from langgraph.graph import StateGraph

@dataclass
class Context:
    user_id: str

class State(TypedDict):
    messages: list
    counter: int

graph = StateGraph(
    state_schema=State,
    context_schema=Context,
    input_schema=InputState,  # May be a subset of State
    output_schema=OutputState,  # May be a subset of State
)
```

---

## Node Registration and Input Schemas

### Basic Node Addition

Nodes are registered with `add_node(name_or_callable, action=None)`. The node name is inferred from the function's `__name__` if not explicitly provided.

```python
def my_node(state: State) -> dict:
    return {"field": state["field"] + 1}

# Infer name from function
builder.add_node(my_node)  # Node name is "my_node"

# Explicit name
builder.add_node("my_custom_name", my_node)
```

### Per-Node Input Schemas

By default, a node receives the full state schema. Use `input_schema` to restrict a node's input to a subset of state keys:

```python
class SubsetState(TypedDict):
    field_a: int
    field_b: str

def specialized_node(state: SubsetState) -> dict:
    return {"field_a": state["field_a"] * 2}

# This node only reads field_a and field_b
builder.add_node("specialized", specialized_node, input_schema=SubsetState)
```

When a custom `input_schema` is provided, the node receives only those keys from the state, allowing schema evolution and multi-schema composition in subgraphs.

### Node Options

Nodes can be configured with:

- **`retry_policy`**: Automatic retry on transient failures (see Error Handling).
- **`cache_policy`**: Cache node results based on inputs (deterministic functions only).
- **`error_handler`**: Node-level exception handler for recovery or fallback logic.
- **`timeout`**: Hard wall-clock or idle timeout for async nodes.
- **`defer`**: Delay node execution until the graph is about to finish (useful for deferred cleanup).
- **`trace_policy`**: Control how node inputs/outputs are recorded in traces.
- **`destinations`**: For graph visualization only; documents where a node can route using `Command`.

```python
def node_with_options(state: State) -> dict:
    return state

builder.add_node(
    "resilient_node",
    node_with_options,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5),
    timeout=30.0,
    error_handler=my_error_handler,
)
```

---

## Edges and Routing

### Unconditional Edges

Unconditional edges create deterministic control flow from one node to another:

```python
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)
```

Multiple nodes can feed into one node via waiting edges:

```python
builder.add_edge(["node_a", "node_b"], "node_c")  # Wait for both to complete
```

### Conditional Edges

Use `add_conditional_edges(source, path, path_map=None)` to route based on state inspection. The `path` function receives the node's output (or full state if no explicit reader) and returns a node name or list of node names.

```python
def should_continue(state: State) -> str:
    if state["value"] > 10:
        return "finish"
    else:
        return "loop"

builder.add_conditional_edges(
    "check_node",
    should_continue,
    {
        "loop": "check_node",
        "finish": END,
    }
)
```

The `path_map` is optional. If omitted, the routing function must return node names directly:

```python
def router(state: State) -> str:
    return "node_a" if state["done"] else "node_b"

builder.add_conditional_edges("decision_node", router)
```

---

## Dynamic Routing with Send

Return `Send` objects from conditional edges or nodes to **dynamically fork execution to multiple targets in the same superstep**. Each `Send` routes a custom state to a target node.

### Send in Conditional Edges

```python
from langgraph.types import Send

class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]

def continue_to_jokes(state: OverallState) -> list[Send]:
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

builder = StateGraph(OverallState)
builder.add_node("generate_joke", lambda state: {"jokes": [f"Joke about {state['subject']}"]})
builder.add_conditional_edges(START, continue_to_jokes)
builder.add_edge("generate_joke", END)

graph = builder.compile()
result = graph.invoke({"subjects": ["cats", "dogs"]})
# {'subjects': ['cats', 'dogs'], 'jokes': ['Joke about cats', 'Joke about dogs']}
```

Each `Send` creates a new task to the target node with the provided input. All generated tasks execute in parallel before proceeding.

### Send in Node Returns

Nodes can also return `Send` objects to trigger dynamic routing:

```python
def splitter_node(state: State) -> list[Send]:
    return [
        Send("worker", {"item": item})
        for item in state["items"]
    ]

builder.add_node("splitter", splitter_node)
```

---

## Messages State Pattern

For conversational agents, use the **messages state pattern** with the `add_messages` reducer, which deduplicates messages by ID while maintaining append-only semantics:

```python
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# In a node:
def agent_node(state: AgentState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
```

The `add_messages` reducer:
- Appends new messages to the list
- Replaces any message with a matching `id` (enabling message editing)
- Filters out `RemoveMessage` objects
- Maintains order

This pattern is ideal for multi-turn conversations, tool-using agents, and any workflow where message history is central.

---

## Branching with Conditional Edges

### Simple Router with Path Map

Define a router function and explicit path mapping:

```python
def router(state: State) -> str:
    if state["query_type"] == "search":
        return "search"
    elif state["query_type"] == "database":
        return "database"
    else:
        return "default"

builder.add_conditional_edges(
    "classifier",
    router,
    {
        "search": "search_node",
        "database": "db_node",
        "default": "default_node",
    }
)
```

### Type-Hinted Return Values

Annotate the router's return type with `Literal` to allow the graph builder to infer the path map:

```python
from typing import Literal

def router(state: State) -> Literal["search", "database"]:
    return "search" if state["is_search"] else "database"

# No path_map needed; inferred from Literal annotation
builder.add_conditional_edges("classifier", router)
```

### Multi-Path Routing

Return a list of node names to route to multiple nodes:

```python
def multi_router(state: State) -> list[str]:
    targets = []
    if state["needs_search"]:
        targets.append("search")
    if state["needs_db"]:
        targets.append("database")
    return targets

builder.add_conditional_edges("classifier", multi_router)
```

---

## Error Handling and Recovery

### Per-Node Error Handlers

Attach an error handler to a node to recover from exceptions:

```python
def fallback_handler(state: State, info: dict) -> dict:
    # info contains {"exception": <exception>, ...}
    return {"error_count": state.get("error_count", 0) + 1}

def risky_node(state: State) -> dict:
    if random.random() > 0.5:
        raise ValueError("Random failure")
    return {"success": True}

builder.add_node(
    "risky",
    risky_node,
    error_handler=fallback_handler,
)
```

When an error handler is triggered, it receives the current state and error context. Its return value updates the state, and execution continues at the next node (not a retry by default).

### Global Default Error Handler

Set a default error handler for all nodes via `set_node_defaults()`:

```python
def default_fallback(state: State, info: dict) -> dict:
    logger.error(f"Node failed: {info['exception']}")
    return {"failed": True}

builder.set_node_defaults(error_handler=default_fallback)
```

Per-node handlers override the default.

### Retry Policies

Automatically retry nodes on transient failures using `RetryPolicy`:

```python
from langgraph.types import RetryPolicy

# Exponential backoff with jitter
retry_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,
    backoff_factor=2.0,
    jitter=True,
)

builder.add_node("flaky_api", call_flaky_api, retry_policy=retry_policy)
```

Retries are independent of error handlers. If all retry attempts fail, the error handler (if configured) is invoked.

---

## Entry and Exit Points

### Setting Entry Points

Graphs must have at least one edge from START:

```python
builder.add_edge(START, "node_a")  # Implicit entry point

# Or explicit:
builder.set_entry_point("node_a")
```

Multiple nodes can be entry points via conditional edges:

```python
def route_entry(state: State) -> str:
    return "fast_path" if state["priority"] else "slow_path"

builder.add_conditional_edges(START, route_entry)
builder.add_node("fast_path", ...)
builder.add_node("slow_path", ...)
```

### Setting Finish Points

Graphs end when reaching the END node or any node with no outgoing edges. Explicitly mark finish points:

```python
builder.set_finish_point("final_node")  # Equivalent to builder.add_edge("final_node", END)
```

---

## Method Chaining API

StateGraph supports fluent, chainable construction for readable workflows:

```python
graph = (
    StateGraph(State)
    .add_node("step_a", node_a)
    .add_node("step_b", node_b)
    .add_edge(START, "step_a")
    .add_edge("step_a", "step_b")
    .set_finish_point("step_b")
    .compile()
)
```

All builder methods return `Self`, enabling this pattern. Method chaining improves readability for complex graphs.

---

## Subgraphs and Nesting

Compiled StateGraph instances can be used as nodes in a parent graph, enabling modular, reusable workflows.

### Basic Subgraph Pattern

```python
# Define subgraph
class SubState(TypedDict):
    input_data: str

def sub_node(state: SubState) -> dict:
    return {"output_data": state["input_data"].upper()}

subgraph = (
    StateGraph(SubState)
    .add_node("process", sub_node)
    .add_edge(START, "process")
    .set_finish_point("process")
    .compile()
)

# Use subgraph as a node in parent graph
class ParentState(TypedDict):
    data: str

def call_subgraph(state: ParentState) -> dict:
    result = subgraph.invoke({"input_data": state["data"]})
    return {"data": result["output_data"]}

parent_graph = (
    StateGraph(ParentState)
    .add_node("sub", call_subgraph)
    .add_edge(START, "sub")
    .set_finish_point("sub")
    .compile()
)
```

### Direct Subgraph Registration

Register a compiled graph directly as a node:

```python
parent_graph = StateGraph(ParentState)
parent_graph.add_node("subgraph", subgraph)  # Compiled subgraph as node
parent_graph.add_edge(START, "subgraph")
```

The parent graph handles input/output schema adaptation automatically.

### Checkpointer Inheritance

By default, subgraphs inherit the parent graph's checkpointer. Disable checkpointing in a subgraph via `compile(checkpointer=False)`:

```python
subgraph = (
    StateGraph(SubState)
    .add_node("step", sub_node)
    .add_edge(START, "step")
    .compile(checkpointer=False)  # No checkpointing in subgraph
)
```

Or enable independent checkpointing with `compile(checkpointer=InMemorySaver())`.

---

## Graph Compilation

The `compile()` method converts a StateGraph builder into a `CompiledStateGraph`, the executable runtime supporting invoke, stream, batch, and async operations.

```python
graph = builder.compile()
```

### Compilation Parameters

- **`checkpointer`**: Optional checkpoint saver (dict, InMemorySaver, PostgresCheckpointer, etc.) for pause-resume.
- **`cache`**: Optional cache implementation for deterministic node result caching.
- **`store`**: Optional storage backend for long-lived data (separate from state).
- **`interrupt_before`**: List of node names to interrupt execution before running.
- **`interrupt_after`**: List of node names to interrupt execution after running.
- **`debug`**: Enable verbose debug output during execution.
- **`name`**: Human-readable name for the compiled graph.

```python
from langgraph.checkpoint.memory import InMemorySaver

compiled = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["user_input"],
    interrupt_after=["agent"],
    debug=True,
)
```

### Invocation

Invoke the compiled graph via `invoke()`, `stream()`, `batch()`, or async variants:

```python
result = compiled.invoke({"value": 0})

# Stream updates from each node
for update in compiled.stream({"value": 0}):
    print(update)

# With checkpointing
config = {"configurable": {"thread_id": "my-thread"}}
result = compiled.invoke({"value": 0}, config=config)

# Resume from checkpoint
result = compiled.invoke({"value": 0}, config=config)  # Reuses prior state
```

---

## Validation and Common Patterns

### Graph Validation

Graphs are validated at `compile()` time. Common validation errors:

- **"Found edge starting at unknown node"**: An edge references a non-existent source node.
- **"Graph must have an entrypoint"**: No edge from START is defined.
- **"Found edge ending at unknown node"**: An edge targets a non-existent node.

Validate early by calling `compile()` before using the graph.

### Parallel Node Execution

Multiple nodes can execute in parallel if they don't write to the same unannotated state key:

```python
class State(TypedDict):
    a: int
    b: int  # No reducer; last-value-wins
    c: Annotated[list, operator.add]  # Reducer; can have concurrent writes

builder.add_edge(START, ["node_1", "node_2"])  # Both run in parallel
builder.add_edge(["node_1", "node_2"], "node_3")  # node_3 waits for both
```

For uninitialized fields with no reducer, concurrent writes raise `InvalidUpdateError`.

### Avoiding Common Pitfalls

1. **Forgetful entry points**: Always add an edge from START.
2. **Dead nodes**: Every node should eventually lead to END.
3. **Infinite loops**: Without explicit termination (e.g., via conditional edge), loops continue indefinitely.
4. **Concurrent writes**: Use reducers (`Annotated`) for fields written by multiple nodes.
5. **Type mismatches**: Ensure node return types are dicts compatible with the state schema.

---

## Example: ReAct Agent

A complete example combining patterns above:

```python
from typing import Annotated
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"

tools = [search]

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent(state: AgentState) -> dict:
    model = init_chat_model("gpt-4o-mini")
    response = model.invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

agent_graph = builder.compile()

# Invoke
result = agent_graph.invoke({"messages": [{"role": "user", "content": "What is Python?"}]})
```

---

## References

- **Core Classes**: `StateGraph` (repo://libs/langgraph/langgraph/graph/state.py#L131-L200), `CompiledStateGraph` (repo://libs/langgraph/langgraph/graph/state.py#L1404-L1420)
- **Branching**: `BranchSpec` (repo://libs/langgraph/langgraph/graph/_branch.py#L83-L120), `Send` (repo://libs/langgraph/langgraph/types.py#L704-L793)
- **Error Handling**: `RetryPolicy` (repo://libs/langgraph/langgraph/types.py), per-node handlers via `add_node(..., error_handler=...)`
- **Messages**: `add_messages` reducer (repo://libs/langgraph/langgraph/graph/message.py#L60-L150)
- **Tests**: Comprehensive patterns in (repo://libs/langgraph/tests/test_pregel.py), including Send, subgraphs, conditional routing
