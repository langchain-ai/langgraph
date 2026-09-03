---
type: Concept
title: Subgraphs and Nesting
description: Composing LangGraph applications by nesting compiled graphs as nodes, with schema adaptation, isolated state, and independent checkpointing enabling modular workflows and multi-agent coordination.
tags: [subgraph, nesting, composition, modularity, schema-mapping, namespace-isolation, state-isolation, checkpointing, recursion, multi-agent]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-3f311c44cb199db866ef9977
    resource: repo://libs/langgraph/bench/fanout_to_subgraph.py
  - id: openwiki-source-6b1e3dc6732527936bf6e128
    resource: repo://libs/langgraph/langgraph/_internal/_replay.py
  - id: openwiki-source-6ba6113c604a5fc7a18ea2e2
    resource: repo://libs/langgraph/langgraph/callbacks.py
  - id: openwiki-source-38dc4e3fe1af9d8f3d241cc6
    resource: repo://libs/langgraph/langgraph/errors.py
  - id: openwiki-source-36058e684ab11833a28af83b
    resource: repo://libs/langgraph/langgraph/graph/state.py
  - id: openwiki-source-d30667fe1721764c2d67aebd
    resource: repo://libs/langgraph/langgraph/pregel/_algo.py
  - id: openwiki-source-32221bce0f3eead36a7c7e36
    resource: repo://libs/langgraph/langgraph/types.py
  - id: openwiki-source-cf9645f3c211482255b6ab55
    resource: repo://libs/langgraph/tests/test_parent_command.py
  - id: openwiki-source-3ca34e33428766d841de969a
    resource: repo://libs/langgraph/tests/test_runtime.py
  - id: openwiki-source-d86958f5aabde646e90882fd
    resource: repo://libs/langgraph/tests/test_subgraph_detection.py
  - id: openwiki-source-683ca76faece52e57e12784e
    resource: repo://libs/langgraph/tests/test_subgraph_persistence.py
  - id: openwiki-source-443dcfdc244fb40215e19eea
    resource: repo://libs/langgraph/tests/test_time_travel.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

**Subgraph nesting** is a composition pattern that enables modular, reusable LangGraph workflows. Instead of building a single monolithic graph, you can compose multiple `StateGraph`s—some specialized, some delegating to others—by adding a compiled graph as a node in a parent graph. This enables:

- **Modularity**: Package logic into focused, reusable subgraph units.
- **Multi-agent coordination**: One agent delegates work to another agent (e.g., an agent calling a tool that itself is an agent).
- **Schema adaptation**: Subgraph input/output schemas can differ from the parent's state, with automatic mapping.
- **State isolation**: Subgraph state is independent; parent nodes cannot directly access internal subgraph state.
- **Namespace isolation**: Each subgraph invocation gets its own checkpoint namespace, enabling independent time-travel.
- **Configurable persistence**: Control whether subgraph state accumulates (stateful), resets per invocation (stateless), or is discarded (no checkpointing).
- **Recursion**: Subgraphs can contain other subgraphs, bounded by a recursion limit.

---

## Core Pattern: Adding a Compiled Graph as a Node

The simplest subgraph pattern is to compile a `StateGraph` and add it to a parent graph using `add_node()`:

```python
from langgraph.graph import StateGraph, START, END

# Build a subgraph
subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("step_a", node_a)
subgraph_builder.add_edge(START, "step_a")
subgraph_builder.set_finish_point("step_a")
subgraph = subgraph_builder.compile()

# Add the compiled subgraph as a node in the parent graph
parent_builder = StateGraph(ParentState)
parent_builder.add_node("subgraph_node", subgraph)
parent_builder.add_edge(START, "subgraph_node")
parent_builder.add_edge("subgraph_node", END)

parent_app = parent_builder.compile(checkpointer=InMemorySaver())
```

When the parent graph executes "subgraph_node", it invokes the subgraph's compiled `CompiledStateGraph` as a callable `Runnable`. The subgraph runs its own superstep loop internally, then returns its final state to the parent.

---

## Schema Mapping and Input/Output Adaptation

A subgraph's internal state schema can differ from the parent's state. LangGraph handles the mapping automatically using `input_schema` and `output_schema`.

### Definition

- **`input_schema`**: The subset of state (or custom schema) that the subgraph expects as input. If not specified, defaults to the subgraph's `state_schema`.
- **`output_schema`**: The subset of state (or custom schema) that the subgraph produces as output. If not specified, defaults to the subgraph's `state_schema`.

### Automatic Mapping

When a subgraph is invoked, the parent supplies inputs that are mapped to the subgraph's `input_schema`, and the subgraph's outputs are mapped back to the parent state using `output_schema`.

### Example: Fanout with Schema Adaptation

```python
from typing_extensions import TypedDict

class ParentState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]

class JokeInput(TypedDict):
    subject: str

class JokeOutput(TypedDict):
    jokes: list[str]

class JokeState(JokeInput, JokeOutput):
    pass

# Subgraph expects only "subject", outputs only "jokes"
subgraph = StateGraph(
    JokeState,
    input_schema=JokeInput,
    output_schema=JokeOutput
)
subgraph.add_node("generate", lambda s: {"jokes": [f"Joke: {s['subject']}"]})
subgraph.set_entry_point("generate")
subgraph.set_finish_point("generate")
compiled_subgraph = subgraph.compile()

# Parent sends individual subjects via Send objects
def router(state: ParentState):
    return [
        Send("generate_joke", {"subject": s})
        for s in state["subjects"]
    ]

parent = StateGraph(ParentState)
parent.add_node("generate_joke", compiled_subgraph)
parent.add_conditional_edges(START, router)
parent.add_edge("generate_joke", END)

# Parent state {"subjects": ["cats", "dogs"], "jokes": []}
# Each Send call: subgraph receives {"subject": "cats"}, 
#   processes with input_schema JokeInput, 
#   returns {"jokes": [...]}, merged into parent's "jokes" list via output_schema
```

Reference: `repo://libs/langgraph/langgraph/graph/state.py#L1515-L1547` (input/output channel mapping in CompiledStateGraph).

---

## State Isolation

### Independent State Space

A subgraph maintains its own internal state channels, separate from the parent's state. When the parent invokes a subgraph:

1. **Input mapping**: Parent state is projected to the subgraph's `input_schema` (if specified).
2. **Independent execution**: The subgraph's nodes operate on the subgraph's channels, not parent channels.
3. **Output mapping**: The subgraph's `output_schema` is extracted and merged back into parent state.

The parent **cannot directly read** internal subgraph channels that are not in the `output_schema`. This isolation enables clean separation of concerns.

### Parent-Child Communication

The primary mechanism for parent-child communication is the mapping of state through `input_schema` and `output_schema`:

```python
# Parent has: state = {"user_id": "alice", "task_list": [...]}
# Subgraph has: input_schema = {"task": str}, output_schema = {"result": str}

# When parent calls subgraph_node:
# 1. Parent state {"user_id": "alice", "task_list": [...]}) 
#    is mapped to input_schema {"task": "item_1"}
# 2. Subgraph processes {"task": "item_1"} internally
# 3. Subgraph returns {"result": "processed"}
# 4. Parent state is updated: {"user_id": "alice", "task_list": [...], "result": "processed"}
```

---

## Checkpoint Namespace and Time-Travel Isolation

Each subgraph invocation receives its own **checkpoint namespace**—a scoped, independent segment of the checkpoint store.

### How It Works

When a subgraph is invoked as a node in the parent graph, the executor automatically creates a namespace for that invocation:

```
parent_thread_id = "my-thread"
parent_namespace = ("parent_node",)

# When subgraph_node executes:
subgraph_namespace = ("parent_node", "subgraph_node:<task_id>", "inner_node:<task_id>")
```

The `<task_id>` suffix ensures that multiple invocations of the same subgraph in the same parent run maintain separate checkpoint histories.

### Independent Replay and Fork

With namespace isolation, you can:

- **Time-travel within the subgraph** without affecting the parent's history.
- **Fork from a subgraph checkpoint** to explore an alternative execution path inside the subgraph.
- **Replay from before the subgraph invocation** and re-execute the subgraph with fresh state.

Reference: `repo://libs/langgraph/langgraph/_internal/_replay.py` (subgraph checkpoint loading during time-travel).

---

## Checkpointer Configuration: Stateful, Stateless, and None

The `checkpointer` parameter controls how subgraph state is persisted.

### `checkpointer=None` (Stateless, Default)

The subgraph **does not persist state across separate parent invocations** but **inherits** the parent's checkpointer for interrupt/resume support.

**Use case**: An agent invoked from a tool, where each tool call starts with a clean state.

```python
# Subgraph doesn't persist; resets state each time parent invokes it
subgraph = subgraph_builder.compile(checkpointer=None)

# Parent can still interrupt/resume the subgraph via inherited checkpointer
parent = parent_builder.compile(checkpointer=sync_checkpointer)

# Each time parent runs, subgraph starts fresh
result1 = parent.invoke({"task": "task1"})
result2 = parent.invoke({"task": "task2"})  # subgraph has no memory of task1
```

Reference: `repo://libs/langgraph/tests/test_subgraph_persistence.py#L85-L144` (tests demonstrating stateless behavior).

### `checkpointer=True` (Stateful)

The subgraph **accumulates state across invocations on the same thread**, with its own persistent checkpoint namespace.

**Use case**: A multi-turn agent that remembers conversation history across separate parent calls.

```python
# Subgraph persists state across invocations
subgraph = subgraph_builder.compile(checkpointer=True)

parent = parent_builder.compile(checkpointer=sync_checkpointer)

config = {"configurable": {"thread_id": "shared-thread"}}

# 1st invocation: subgraph sees empty state
result1 = parent.invoke({"input": "hello"}, config)

# 2nd invocation: subgraph loads accumulated state from 1st invocation
result2 = parent.invoke({"input": "how are you?"}, config)
```

Reference: `repo://libs/langgraph/tests/test_subgraph_persistence.py#L297-L373` (stateful subgraph tests).

### `checkpointer=False` (No Checkpointing)

The subgraph **does not use or inherit any checkpointer**, even if the parent has one. Useful for stateless, fire-and-forget subgraphs.

```python
# Subgraph has no checkpointer, even if parent does
subgraph = subgraph_builder.compile(checkpointer=False)
```

Reference: `repo://libs/langgraph/tests/test_subgraph_persistence.py#L233-L293` (tests with checkpointer=False).

---

## Error Propagation via ParentCommand

When a subgraph encounters an error or raises an exception during execution, the parent graph can decide how to handle it.

### ParentCommand Exception

If a subgraph (or node inside a subgraph) issues a `Command` with `graph=Command.PARENT`, the exception `ParentCommand` is raised, allowing the parent to intercept and route control flow:

```python
class ChildState(TypedDict):
    jump: bool

def child_node(state: ChildState):
    if state["jump"]:
        # Signal parent to jump to a different node
        return Command(graph=Command.PARENT, goto="parent_second")
    return state

subgraph = StateGraph(ChildState)
subgraph.add_node("node", child_node)
subgraph.set_entry_point("node")
compiled_subgraph = subgraph.compile()

def parent_first(state: ParentState):
    # Invoke subgraph; if it issues Command.PARENT, exception is raised here
    result = compiled_subgraph.invoke({"jump": True})
    # Exception ParentCommand bubbles up, parent handles routing
    return state

parent = StateGraph(ParentState)
parent.add_node("parent_first", parent_first)
parent.add_node("parent_second", lambda s: s)
parent.add_edge(START, "parent_first")
parent.add_edge("parent_second", END)
```

The parent's exception handler (or error handler node) can catch `ParentCommand` and decide whether to proceed, retry, or navigate to a different node.

Reference: `repo://libs/langgraph/langgraph/errors.py#L129-L134` (ParentCommand definition); `repo://libs/langgraph/tests/test_parent_command.py#L9-L53` (end-to-end example).

---

## Context Propagation (Read-Only)

A parent graph's `context_schema` is **read-only** and propagated to subgraph scopes via `RunnableConfig`.

### Pattern

```python
from langgraph.runtime import Runtime

class ParentContext(TypedDict):
    user_id: str
    db_connection: Any

# Subgraph nodes can read parent context via Runtime
def subgraph_node(state: SubgraphState, runtime: Runtime[ParentContext]):
    user_id = runtime.context["user_id"]
    # Use user_id for lookups, logging, etc.
    return state

subgraph = StateGraph(SubgraphState)
subgraph.add_node("process", subgraph_node)
subgraph.set_entry_point("process")
compiled_subgraph = subgraph.compile()

parent = StateGraph(ParentState, context_schema=ParentContext)
parent.add_node("subgraph_node", compiled_subgraph)
# ...
parent.compile()

# Invoke with context
result = parent.invoke(
    {"data": ...},
    context={"user_id": "alice", "db_connection": db},
)
```

The context flows down to subgraph nodes but cannot be modified by subgraph nodes. This ensures that cross-cutting concerns (authentication, logging, shared resources) are available but protected from mutation.

Reference: `repo://libs/langgraph/langgraph/graph/state.py#L184-L199` (context example in docstring); `repo://libs/langgraph/tests/test_runtime.py#L271-L300` (runtime context propagation test).

---

## Recursion: Subgraphs Containing Subgraphs

Subgraphs can be arbitrarily nested—a subgraph can itself add compiled subgraphs as nodes. Execution depth is bounded by `recursion_limit` (default: 25 supersteps per graph per root thread).

### Example: Three-Level Nesting

```python
# Level 3: innermost subgraph
inner_subgraph = StateGraph(InnerState).add_node("step", node_fn).compile()

# Level 2: middle subgraph containing innermost
middle_subgraph_builder = StateGraph(MiddleState)
middle_subgraph_builder.add_node("inner", inner_subgraph)
middle_subgraph = middle_subgraph_builder.compile()

# Level 1: parent graph containing middle subgraph
parent_builder = StateGraph(ParentState)
parent_builder.add_node("middle", middle_subgraph)
parent_app = parent_builder.compile(checkpointer=sync_checkpointer)

# Each level maintains its own checkpoint namespace
# Namespace structure: ("middle", "inner:<task_id>", "step:<task_id>")
```

Time-travel and replay work correctly across all nesting levels, with each subgraph maintaining independent checkpoint history within its scope.

Reference: `repo://libs/langgraph/tests/test_time_travel.py#L1585-L1650` (three-level nested subgraph tests).

---

## Multi-Agent Coordination

Subgraph nesting enables agent-to-agent delegation patterns, where one agent invokes another agent as a tool or delegated task.

### Pattern: Agent Calls Agent via Tool

```python
from langchain_core.messages import HumanMessage, AIMessage

# Inner agent (e.g., research agent)
inner_graph = StateGraph(MessagesState)
inner_graph.add_node("researcher", research_node)
inner_graph.set_entry_point("researcher")
inner_graph.set_finish_point("researcher")
compiled_inner = inner_graph.compile()

# Outer agent that delegates to inner agent
def delegate_to_researcher(state: OuterState):
    # Invoke inner agent as a tool
    result = compiled_inner.invoke({
        "messages": [HumanMessage(content="Research X")]
    })
    return {"messages": [AIMessage(content=f"Result: {result['messages'][-1]}...")]}

outer_graph = StateGraph(OuterState)
outer_graph.add_node("researcher_tool", delegate_to_researcher)
outer_graph.set_entry_point("researcher_tool")

outer_app = outer_graph.compile(checkpointer=checkpointer)
```

Each agent maintains its own state (messages, tools available, model), and the parent agent decides when and how to delegate.

Reference: `repo://libs/langgraph/tests/test_multiple_subgraphs.py` (multiple subgraph invocation patterns).

---

## Use Cases and Patterns

### 1. Tool with Internal Workflow

An LLM agent uses a tool that itself contains a workflow:

```python
# Tool: a subgraph that processes a request
request_processor = StateGraph(ProcessState).add_node(...).compile()

def tool_execute(input: str):
    result = request_processor.invoke({"request": input})
    return result

# The agent calls this tool
```

### 2. Multi-Stage Pipeline

Break a large workflow into logical stages, each a subgraph:

```python
# Extract → Validate → Transform → Load
extract_sg = StateGraph(State).add_node("extract", ...).compile()
validate_sg = StateGraph(State).add_node("validate", ...).compile()
transform_sg = StateGraph(State).add_node("transform", ...).compile()

pipeline = StateGraph(PipelineState)
pipeline.add_node("extract", extract_sg)
pipeline.add_node("validate", validate_sg)
pipeline.add_node("transform", transform_sg)
pipeline.add_edge(START, "extract")
pipeline.add_edge("extract", "validate")
pipeline.add_edge("validate", "transform")
```

### 3. Dynamic Subgraph Selection

Choose which subgraph to invoke based on runtime conditions:

```python
def router(state: MainState):
    if state["type"] == "A":
        return "subgraph_a"
    elif state["type"] == "B":
        return "subgraph_b"
    return "default"

main = StateGraph(MainState)
main.add_node("subgraph_a", compiled_a)
main.add_node("subgraph_b", compiled_b)
main.add_node("default", compiled_default)
main.add_conditional_edges(START, router)
```

### 4. Distributed / Remote Subgraphs

With `RemoteRunnable`, a subgraph can execute on a remote server while maintaining the same composition pattern (synchronous calls hide network latency).

---

## Key Responsibilities

### Parent Graph

- **State mapping**: Transform parent state to subgraph `input_schema`; merge subgraph output back to parent state.
- **Invocation**: Call the compiled subgraph as a `Runnable`; treat it like any other node.
- **Error handling**: Catch `ParentCommand` exceptions if subgraph issues `Command.PARENT`.
- **Checkpointer provision**: Provide a checkpointer if the parent needs persistence; subgraph inherits it unless overridden.

### Subgraph

- **Independent execution**: Run its own superstep loop, with its own nodes and channels.
- **Output production**: Return state (or values) according to `output_schema`.
- **Interrupt support**: If the parent has a checkpointer, the subgraph can support interrupt/resume (if `checkpointer=None`).
- **Namespace awareness**: Operate within its own checkpoint namespace (for stateful subgraphs).

---

## Invariants and Failure Modes

### Checkpoint Inheritance

- If subgraph is compiled with `checkpointer=None` (default), it inherits the parent's checkpointer for interrupt/resume **only**. State does not accumulate across parent invocations.
- If subgraph is compiled with `checkpointer=True`, it maintains its own persistent checkpoint namespace and accumulates state.
- If subgraph is compiled with `checkpointer=False`, no checkpointing occurs, even if the parent has one.

### Schema Mismatch

- If `input_schema` is not provided, the subgraph uses its `state_schema` (all fields required as input).
- If parent sends a `Send` object that doesn't match subgraph's `input_schema`, a validation error is raised at execution time.
- If `output_schema` omits a field that subgraph produces, that field is discarded; parent state is updated only with `output_schema` fields.

### Namespace Isolation

- Subgraph interrupts **do not** bubble up to the parent; they are handled within the subgraph's scope.
- Parent interrupts **do** suspend the subgraph invocation (the parent loop stops).
- Time-travel to a subgraph checkpoint does not affect parent checkpoints at other namespaces.

---

## Testing and Introspection

### Streaming with `subgraphs=True`

To see subgraph internal events during execution:

```python
config = {"configurable": {"thread_id": "my-thread"}}

for event in parent_app.stream(
    {"input": ...},
    config,
    stream_mode="updates",
    subgraphs=True,  # Include subgraph updates
):
    print(event)
```

This emits updates from both parent and subgraph nodes, helping with debugging and monitoring.

### Checkpoint Inspection

```python
state = parent_app.get_state(config, subgraphs=True)
# state.tasks[0].state.config provides subgraph checkpoint config
```

Reference: `repo://libs/langgraph/tests/test_time_travel.py#L2669-L2715` (get_state with subgraphs).

---

## Summary

Subgraph nesting is a powerful composition mechanism for building modular, multi-agent workflows. By isolating state, providing schema adaptation, and enabling independent checkpointing, subgraphs allow you to:

- Decompose large graphs into reusable, testable units.
- Coordinate multiple agents with clean separation of concerns.
- Support persistent, interruptible, and time-travelable execution at every nesting level.

The pattern integrates seamlessly with LangGraph's core abstractions (state, channels, nodes, edges) and supports all streaming, debugging, and checkpoint features.
