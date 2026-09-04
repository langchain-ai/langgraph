---
type: Architecture & Design
title: Core Concepts
description: Foundational LangGraph concepts including the Pregel dataflow algorithm, nodes, edges, channels, state, and task execution model.
tags: [pregel, nodes, edges, channels, state, dataflow, superstep, reducer, streaming]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-6c95109f667df245389a281a
    resource: repo://libs/checkpoint/langgraph/checkpoint/base/__init__.py
  - id: openwiki-source-95bb198e6253ec30c7591a90
    resource: repo://libs/langgraph/langgraph/channels/__init__.py
  - id: openwiki-source-9ac63269e899a21d67302cbf
    resource: repo://libs/langgraph/langgraph/channels/base.py
  - id: openwiki-source-1358f0ad7cc3f870325b6135
    resource: repo://libs/langgraph/langgraph/channels/delta.py
  - id: openwiki-source-33278185a14e90f69cfefcb2
    resource: repo://libs/langgraph/langgraph/channels/last_value.py
  - id: openwiki-source-9b01a2a24dd7f7a0fb7e6f05
    resource: repo://libs/langgraph/langgraph/graph/message.py
  - id: openwiki-source-36058e684ab11833a28af83b
    resource: repo://libs/langgraph/langgraph/graph/state.py
  - id: openwiki-source-d30667fe1721764c2d67aebd
    resource: repo://libs/langgraph/langgraph/pregel/_algo.py
  - id: openwiki-source-4ae17bf912e4007bb8b83bef
    resource: repo://libs/langgraph/langgraph/pregel/main.py
  - id: openwiki-source-832ee3c88a1c4fabb7818587
    resource: repo://libs/langgraph/langgraph/pregel/protocol.py
  - id: openwiki-source-32221bce0f3eead36a7c7e36
    resource: repo://libs/langgraph/langgraph/types.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

LangGraph is built on a **dataflow-inspired execution model** centered on the **Pregel algorithm**. Instead of traditional sequential function calls, LangGraph graphs execute as **directed acyclic networks where nodes are functions or runnables, communication flows through typed state channels, and execution proceeds through atomic supersteps**. This design enables parallelism, stateful composition, and human-in-the-loop interrupts.

The three core abstractions are:

- **Nodes**: Functions or runnables that read and write state.
- **Edges**: Control flow paths connecting nodes, with optional conditional routing.
- **Channels**: Typed, versioned storage containers for state values, with pluggable reducer semantics.

Together, these enable **persistent, resumable, time-travelable graph execution** where every state snapshot can be checkpointed, interrupted for human input, and replayed.

---

## StateGraph: The Primary API

`StateGraph` (repo://libs/langgraph/langgraph/graph/state.py#L131-L200) is the primary builder for LangGraph applications. It uses a **typed state schema** (TypedDict, dataclass, or Pydantic model) to define the shared state all nodes read and write.

### State Schema Definition

State schemas are declared using standard Python type hints. Each field can be annotated with a **reducer function** that aggregates concurrent updates:

```python
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    count: int                              # No reducer; last value wins
    messages: Annotated[list, operator.add]  # Reducer: concatenate lists
    done: bool
```

When multiple nodes write to the same key in the same superstep, the reducer is invoked: `new_value = reducer(current, updates)`.

**Special Case: Messages.** LangGraph provides `add_messages`, a reducer pattern for append-only message streams with deduplication by ID (repo://libs/langgraph/langgraph/graph/message.py#L61-L200). This enables natural multi-turn conversation workflows without duplicate handling logic.

### StateGraph Methods

**Graph Construction:**

- `add_node(name, function)`: Register a node by name (or infer from function's `__name__`).
- `add_edge(source, target)`: Add a deterministic edge.
- `add_conditional_edges(source, condition_fn, edge_map)`: Route based on state inspection. The condition function receives state (and optional config) and returns a next node name or a list of `Send` objects for dynamic fan-out.
- `set_entry_point(name)`: Mark the starting node(s).
- `set_finish_point(name)`: Mark the ending node(s).

**Policies:**

- `set_node_defaults(retry_policy=..., cache_policy=..., error_handler=..., timeout=...)`: Apply default policies to all nodes at compile time. Per-node settings override defaults.

**Compilation:**

- `compile(checkpointer=None, cache=None, store=None)`: Convert the builder into a `CompiledStateGraph`, the executable runtime.

Example:

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)
graph.add_node("step_a", node_a)
graph.add_node("step_b", node_b)
graph.add_edge(START, "step_a")
graph.add_edge("step_a", "step_b")
graph.add_edge("step_b", END)

compiled = graph.compile()
result = compiled.invoke({"count": 0, "messages": []})
```

---

## Nodes and Control Flow

### Node Semantics

A **node** is a function or `Runnable` that reads state and returns state updates. The signature is `(State | partial<State>) -> dict[str, Any]` or `(State, config) -> dict[str, Any]`.

**Return Type:**

- Nodes return a **partial state update** (a dict with only the keys to modify), not the full state. This enables parallel execution: multiple nodes can run simultaneously if they don't both write to the same unannotated key.
- Return values are merged into state via the channel reducer for each key. No reducer = last-value-wins.

**Types of Returns:**

- **Dict:** `{"count": 5}` — a plain update.
- **Send:** `[Send("node_b", {"input": ...})]` — dynamically route to another node with custom input in the same superstep.
- **Command:** `Command(update={...}, goto="node_b")` — advanced: update state, interrupt, resume interrupts, and navigate in one primitive.

### Conditional Edges

A conditional edge (repo://libs/langgraph/langgraph/graph/state.py#L700+) uses a function to decide the next node(s) based on current state.

```python
def router(state: State) -> str | list[Send]:
    if state["count"] > 5:
        return "finish"
    else:
        return "loop"

graph.add_conditional_edges("step_a", router, {"finish": END, "loop": "step_b"})
```

If the router returns `Send` objects, it enables **dynamic branching**: execute the same node multiple times in parallel with different inputs.

```python
def map_step(state: State):
    return [Send("process", {"item": item}) for item in state["items"]]
```

This is the foundation for map-reduce and fan-out patterns.

---

## The Pregel Algorithm: Execution Model

Pregel is a **synchronous, bulk-synchronous dataflow model** inspired by Google's Pregel and Apache Beam. LangGraph's `Pregel` class (repo://libs/langgraph/langgraph/pregel/main.py) implements the runtime.

### Supersteps: Atomic Units of Execution

Execution proceeds in **supersteps**—atomic synchronization barriers where:

1. **Read Phase:** All triggered nodes read the current checkpoint state and their input channels (which were updated in the previous superstep).
2. **Execution Phase:** All nodes execute in parallel (or sequentially depending on the loop implementation).
3. **Write Phase:** All writes are collected and applied to channels atomically.
4. **Advance Phase:** Channel versions are incremented; the next superstep is scheduled if any channel was updated or any node requests further execution.

**State Snapshot:** At each superstep, the state is a **complete snapshot** of all channel values at that logical point. Versions track which channels have been updated, enabling nodes to trigger only when their inputs change.

### Channel Versioning

Each channel has a version (integer, string, or float) that increments each superstep if the channel is updated. Nodes record the versions they have seen in `checkpoint["versions_seen"]`. On the next superstep, only nodes whose triggers (input channels) have newer versions are scheduled.

This enables efficient incremental execution: if node A writes to channel X and node B reads X, node B is triggered. But if neither changes, subsequent nodes don't run.

### Prepare and Execute Phases

`prepare_next_tasks()` (repo://libs/langgraph/langgraph/pregel/_algo.py#L392-L513) determines which nodes to run based on:

1. **PUSH tasks** (from `Send` or `TASKS` channel): Explicit fan-out instructions.
2. **PULL tasks** (triggered by channel updates): Nodes whose read channels have new versions.

The **trigger_to_nodes** map (built at compile time) maps channel names to the set of nodes that read them. This optimization avoids checking all nodes if only a subset have input.

Once tasks are prepared, the executor runs them (in parallel via `ThreadPoolExecutor`, `asyncio`, or a custom runtime). Results are written back to the checkpoint, and the loop continues until no tasks are scheduled.

---

## Channels: Typed State Containers

A **channel** is a versioned, typed container for a single state value. `BaseChannel` (repo://libs/langgraph/langgraph/channels/base.py) defines the abstraction.

### Channel Implementations

**`LastValue`** (repo://libs/langgraph/langgraph/channels/last_value.py): Stores the most recent value. Multiple updates in one step raise an error (last-value-wins requires single assignment per step). Used for most unannotated fields.

**`DeltaChannel`** (repo://libs/langgraph/langgraph/channels/delta.py): Accumulates updates via a reducer. Updates are stored as writes in the checkpoint; state is reconstructed by replaying writes through the reducer. Efficient for frequently-updated channels (e.g., token-by-token message streams, growing lists). Includes configurable snapshot frequency to bound replay depth.

**`Topic`** (repo://libs/langgraph/langgraph/channels/topic.py): A queue-like channel for fan-out. Stores a deque of messages; each read consumes them. Used internally for `Send` (dynamic routing).

**`BinaryOperatorAggregate`** (repo://libs/langgraph/langgraph/channels/binop.py): Applies a binary operator (e.g., `operator.add`, `operator.or_`) to aggregate updates.

**`NamedBarrierValue`** (repo://libs/langgraph/langgraph/channels/named_barrier_value.py): Synchronization primitive; collects named updates until all expected keys are present, then releases a single value.

**`EphemeralValue`** (repo://libs/langgraph/langgraph/channels/ephemeral_value.py): Available only during the current superstep; cleared after reads.

**`AnyValue` & `UntrackedValue`**: Untyped or unversioned containers for advanced use cases.

### Channel Selection and Automatic Inference

When you define a state schema with `Annotated[Type, reducer]`, StateGraph automatically creates the appropriate channel:

```python
from langgraph.channels import DeltaChannel

class State(TypedDict):
    messages: Annotated[list, DeltaChannel(operator.add)]  # DeltaChannel
    count: int  # LastValue (implicit, no annotation)
```

If no reducer is present, `LastValue` is used. If a reducer is a function (or `operator.*` callable), either `BinaryOperatorAggregate` (for pairs) or `DeltaChannel` (for sequences) is selected.

### Channel API

**Reading:**

- `channel.get()`: Return the current value, or raise `EmptyChannelError`.
- `channel.is_available()`: Return `True` if the channel has been written since initialization.

**Writing:**

- `channel.update(values: Sequence[Update]) -> bool`: Apply updates; return `True` if the channel changed.
- `channel.consume() -> bool`: Notify the channel it was read; some channels (like `Topic`) clear state after consumption.
- `channel.finish()`: Notify the channel execution is (tentatively) ending; used for cleanup or final flushes.

**Serialization:**

- `channel.checkpoint() -> CheckpointValue`: Serialize state for persistence (often just the value itself, or a sentinel for delta channels).
- `channel.from_checkpoint(data) -> Channel`: Deserialize and restore a channel from persisted data.

---

## State and Versions

The **state** at any point is a complete snapshot of all channel values. It is represented as a dict mapping channel names to values, or as a TypedDict/Pydantic instance if a schema is provided.

### Checkpoint Structure

A checkpoint (repo://libs/checkpoint/langgraph/checkpoint/base/__init__.py#L93-L125) is a dict containing:

```python
{
  "v": 1,                          # Format version
  "id": "uuid6",                   # Unique checkpoint ID
  "ts": "2024-01-15T...",          # ISO 8601 timestamp
  "channel_values": {              # Deserialized channel state
    "count": 5,
    "messages": [...]
  },
  "channel_versions": {            # Current version of each channel
    "count": 3,
    "messages": 3
  },
  "versions_seen": {               # Per-node: versions it has processed
    "node_a": {"count": 2, "messages": 2},
    "node_b": {"count": 3}
  },
  "updated_channels": ["count"]    # Which channels changed in this step
}
```

**versions_seen** is crucial: it allows the scheduler to determine which nodes have unprocessed inputs. A node runs only if at least one of its trigger channels has a newer version than recorded in `versions_seen[node_name]`.

### State Snapshots and Time Travel

Via `get_state(config)` and `get_state_history(config)`, users can retrieve state at any point in the execution trace. Passing a `checkpoint_id` to `invoke()` resumes from that checkpoint, enabling time-travel debugging and conditional resumption.

---

## Dynamic Routing: Send and Command

### Send: Explicit Task Dispatch

`Send` (repo://libs/langgraph/langgraph/types.py#L704-L793) is a primitive that explicitly enqueues a task for a specific node with custom input state.

```python
Send(node="process_item", arg={"item": item}, timeout=None)
```

When a node returns a list of `Send` objects from a conditional edge, those tasks are added to the execution queue for the next superstep, enabling fan-out.

**Use Case:** Map-reduce, multi-branch processing, or heterogeneous routing based on dynamic data.

### Command: Unified Control

`Command` (repo://libs/langgraph/langgraph/types.py#L798-L849) is a dataclass that wraps multiple control directives:

```python
Command(
    update={"state_key": new_value},  # Update to apply
    goto="next_node",                 # Node to navigate to
    resume={interrupt_id: value},     # Resume pending interrupts
    graph=None                        # Which graph (None = current, "parent" = parent)
)
```

**update:** Merges state like a regular node return.

**goto:** Can be a node name, a list of names, or `Send` objects. Overrides normal edge routing.

**resume:** Resumes one or more pending interrupts with values. Used in human-in-the-loop workflows (see Interrupts below).

**graph:** Routes the command to a specific graph scope (parent, sibling subgraph, etc.).

---

## Reducers and Accumulation

When a state field is annotated with a **reducer function**, concurrent writes are merged via that function instead of overwriting.

### Reducer Signature and Semantics

A reducer is `(accumulated_value, list[updates]) -> new_value`.

**Key Property:** Reducers must be **batching-invariant** (associative):

```
reducer(reducer(state, xs), ys) == reducer(state, xs + ys)
```

This allows LangGraph to replay writes in larger batches without changing the outcome.

### Built-in Reducers

- `operator.add` (list concatenation, dict merging, etc.): `[1, 2] + [3, 4] = [1, 2, 3, 4]`
- `operator.or_` (set union, bool OR): `{1, 2} | {3} = {1, 2, 3}`
- `operator.mul`, `operator.and_`, etc.

### Custom Reducers

```python
def merge_dicts(current, updates):
    """Merge updates into current, with later updates overwriting earlier."""
    result = current.copy()
    for u in updates:
        result.update(u)
    return result

class State(TypedDict):
    metadata: Annotated[dict, merge_dicts]
```

### Messages: add_messages Reducer Pattern

`add_messages` is a special reducer for message lists that:

1. Appends new messages.
2. Deduplicates by message ID (later messages with the same ID replace earlier ones).
3. Handles `RemoveMessage` to tombstone entries.

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

This is the idiomatic pattern for multi-turn conversation, ensuring message histories remain consistent across node executions.

---

## Task Execution and Parallelism

### PregelExecutableTask

Once tasks are prepared, each becomes a `PregelExecutableTask` (repo://libs/langgraph/langgraph/types.py#L666-L681) containing:

- **name:** Node name.
- **input:** The state values read from channels.
- **proc:** The actual runnable (function or `Runnable`).
- **writes:** A deque for task to record output updates.
- **config:** RunnableConfig with checkpoint ID, thread ID, etc.
- **triggers:** Which channels this node read (for version tracking).
- **retry_policy, cache_policy, timeout:** Per-task policies.

### Execution Modes

**Sync Execution:** Tasks run sequentially via `SyncPregelLoop` (repo://libs/langgraph/langgraph/pregel/_loop.py). Useful for testing, debugging, or single-threaded applications.

**Async Execution:** Tasks run concurrently via `asyncio` tasks in `AsyncPregelLoop`. Ideal for I/O-bound nodes (LLM calls, API queries).

**Custom Runtime:** Via the `runtime` parameter, users can provide a custom `Runtime` (repo://libs/langgraph/langgraph/runtime.py) implementation to control task scheduling, resource limits, and integration with external executors.

### Error Handling and Retries

Nodes can specify:

- **retry_policy:** Automatic retries with exponential backoff on specified exceptions.
- **error_handler:** A special node that runs if the primary node raises and is not caught by retry.

Error handlers receive the exception and the state, and can return an update, a `Command` to resume/goto, or re-raise.

---

## Configuration, Context, and Injection

### Context Schema

The optional `context_schema` (repo://libs/langgraph/langgraph/graph/state.py#L217-L219) allows passing immutable, run-scoped data to nodes without polluting state:

```python
from langgraph.runtime import Runtime

class Context(TypedDict):
    user_id: str
    db_conn: Any

def node(state: State, runtime: Runtime[Context]) -> State:
    user = runtime.context["user_id"]
    # ...
    return {...}

graph = StateGraph(State, context_schema=Context)
# ...
compiled = graph.compile()
result = compiled.invoke(input, context={"user_id": "123", "db_conn": conn})
```

### Config Injection

Nodes can request `RunnableConfig` as a parameter:

```python
def node(state: State, config: RunnableConfig) -> State:
    # config contains thread_id, checkpoint_id, etc.
    return {...}
```

The config is automatically populated by the runtime.

### Managed Values

Managed values (repo://libs/langgraph/langgraph/managed/base.py) are injection points for deterministic state (e.g., a counter, a timestamp). They don't affect state versioning but can be read in nodes.

---

## Superstep and Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Superstep N                                                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. READ: Inspect versions_seen[node] vs channel_versions      │
│    Schedule nodes whose triggers have new versions             │
│                                                                 │
│ 2. PREPARE: Build PregelExecutableTask for each scheduled node│
│    Include reads from channels (snapshot for local exec)       │
│                                                                 │
│ 3. EXECUTE: Run tasks in parallel (or serially)               │
│    Each task collects writes in deque                          │
│                                                                 │
│ 4. WRITE: Merge all writes to channels via reducers           │
│    Update channel_versions, versions_seen                      │
│                                                                 │
│ 5. ADVANCE: If any channel updated or task returned Send/goto│
│    Create checkpoint and schedule Superstep N+1               │
│    Otherwise, finish execution                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Example: Complete Graph

```python
from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class State(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]
    done: bool

def map_items(state: State):
    """Fan out: send each item to 'process' node."""
    return [Send("process", {"item": item}) for item in state["items"]]

def process(state: State):
    """Process a single item."""
    result = state["item"].upper()
    return {"results": [result]}

def reduce_results(state: State):
    """After all items processed, aggregate."""
    return {"done": True}

# Build graph
builder = StateGraph(State)
builder.add_node("map", map_items)
builder.add_node("process", process)
builder.add_node("reduce", reduce_results)

builder.add_edge(START, "map")
builder.add_edge("map", "process")  # dynamic routing via Send
builder.add_edge("process", "reduce")
builder.add_edge("reduce", END)

graph = builder.compile()

# Execute
result = graph.invoke({
    "items": ["apple", "banana"],
    "results": [],
    "done": False
})

print(result["results"])  # ["APPLE", "BANANA"]
```

---

## Key Invariants and Guarantees

1. **Atomicity per Superstep:** All reads in a superstep observe the same state snapshot. Writes are applied atomically.

2. **Determinism:** Same input and same sequence of `Send` routing decisions → same output (unless nodes have inherent randomness).

3. **Versioning Prevents Infinite Loops:** A node runs only if its triggers have new versions. Once no channel updates, execution halts.

4. **Checkpoint Consistency:** Checkpoints are created before superstep boundaries. Resuming from a checkpoint re-executes the next superstep, not the node that created the checkpoint.

5. **Reducer Associativity:** Writes are batched arbitrarily; reducers must handle any order and batch size without changing the result.

---

## References

- **StateGraph**: repo://libs/langgraph/langgraph/graph/state.py#L131-L200
- **Pregel Runtime**: repo://libs/langgraph/langgraph/pregel/main.py
- **Pregel Algorithm**: repo://libs/langgraph/langgraph/pregel/_algo.py#L155-L345
- **Channel Abstraction**: repo://libs/langgraph/langgraph/channels/base.py#L19-L100
- **Send Type**: repo://libs/langgraph/langgraph/types.py#L704-L793
- **Command Type**: repo://libs/langgraph/langgraph/types.py#L798-L849
- **add_messages Reducer**: repo://libs/langgraph/langgraph/graph/message.py#L61-L200
- **Tests**: repo://libs/langgraph/tests/test_state.py, repo://libs/langgraph/tests/test_pregel.py
