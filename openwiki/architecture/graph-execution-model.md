---
type: Architecture & Design
title: Graph Execution Model
description: The synchronous bulk-parallel (Pregel) superstep algorithm, task scheduling, node invocation, state advancement, and error handling that power LangGraph execution.
tags: [pregel, superstep, task-scheduling, state-advancement, node-execution, bulk-synchronous-parallel]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-38dc4e3fe1af9d8f3d241cc6
    resource: repo://libs/langgraph/langgraph/errors.py
  - id: openwiki-source-d30667fe1721764c2d67aebd
    resource: repo://libs/langgraph/langgraph/pregel/_algo.py
  - id: openwiki-source-29a807096a263cef9f8bafe1
    resource: repo://libs/langgraph/langgraph/pregel/_executor.py
  - id: openwiki-source-2f8b73594e09a8a8dcb396c5
    resource: repo://libs/langgraph/langgraph/pregel/_loop.py
  - id: openwiki-source-7070fd1f8c7df259a4e5b657
    resource: repo://libs/langgraph/langgraph/pregel/_read.py
  - id: openwiki-source-73a7ae761087eaf044a3a24c
    resource: repo://libs/langgraph/langgraph/pregel/_runner.py
  - id: openwiki-source-3228893922d89be064177f17
    resource: repo://libs/langgraph/langgraph/pregel/_write.py
  - id: openwiki-source-4ae17bf912e4007bb8b83bef
    resource: repo://libs/langgraph/langgraph/pregel/main.py
  - id: openwiki-source-32221bce0f3eead36a7c7e36
    resource: repo://libs/langgraph/langgraph/types.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

LangGraph executes graphs using a **synchronous bulk-parallel execution model** inspired by Google's Pregel and Apache Beam. Execution proceeds in **supersteps**—atomic synchronization barriers where all nodes that have triggered inputs read state, execute in parallel, and write results back to channels. This design ensures determinism, checkpointability, and composability of concurrent updates.

The execution model unifies:

- **Superstep algorithm** (`pregel/_algo.py`): Determines which nodes run and coordinates state advancement.
- **Task scheduling** (`prepare_next_tasks`): Maps trigger conditions and channel updates to executable tasks.
- **Node invocation** (`pregel/_call.py`, `pregel/_runner.py`): Wraps nodes in PregelNode containers, handles sync/async execution, retry, cache, and timeout policies.
- **Concurrent execution** (`pregel/_executor.py`): Manages thread pools, async contexts, and future collection.
- **Error handling** (`errors.py`, error handler nodes): Routes failures to handler nodes or propagates with retry.
- **State persistence** (`pregel/_checkpoint.py`): Snapshots channel state, versions, and pending writes at each superstep.

---

## The Superstep Algorithm

A **superstep** is an atomic unit of execution. The algorithm is implemented in `pregel/_algo.py` and orchestrated by `pregel/_loop.py` (SyncPregelLoop/AsyncPregelLoop).

### Four-Phase Superstep Cycle

**1. Prepare Phase** (`prepare_next_tasks`)

Determine which nodes will execute by:

- **PUSH tasks** (explicit routing): Scan the TASKS channel for `Send` objects from conditional edges. Each Send specifies a node and custom input.
- **PULL tasks** (implicit routing): For each node, check if any of its trigger channels have been updated since the node last processed them (version comparison in `versions_seen`).

The **trigger_to_nodes** optimization maps each channel name to nodes that subscribe to it, avoiding a full node scan if only a subset of channels changed.

```
trigger_to_nodes["messages"] = ["agent", "summarizer"]
updated_channels = {"messages", "count"}
→ candidate_nodes = {"agent", "summarizer"}  # fetch only these
```

If no PUSH or PULL tasks exist, the superstep completes (no more nodes to run).

**2. Execution Phase** (PregelRunner: `_runner.py`)

Execute all prepared tasks in parallel:

- **Sync context**: ThreadPoolExecutor, one thread per task, with background thread for checkpointer writes.
- **Async context**: asyncio task group, one task per node, with concurrent persistence.

During execution:

- Each task reads its input channels (immutable snapshots from the checkpoint).
- The node function runs and produces writes (to channels via ChannelWrite, to the TASKS channel via Send, or control signals via Command).
- Writes are collected in a deque without applying to the checkpoint yet.

**3. Write Phase** (`apply_writes`)

After all tasks complete:

- Collect writes from all tasks grouped by channel.
- For each channel, invoke its reducer if multiple tasks wrote to the same channel.
- Update channel versions (increment the version number for channels that changed).
- Update `versions_seen[node_name]` for each node to record the channel versions it has now processed.

```python
# Two tasks wrote to "messages"
writes_by_channel = {
    "messages": [msg1_from_task_a, msg2_from_task_b],
    "count": [5]
}
# Reducer merges messages: messages = reducer(current, [msg1, msg2])
# Version bumped: channel_versions["messages"] = 3
# Record consumption: versions_seen["agent"]["messages"] = 3
```

**4. State Advancement & Termination**

- Channel versions are incremented atomically.
- The checkpoint is saved (or writes are buffered, depending on durability mode).
- The next superstep is scheduled if:
  - Any channel was updated (triggering nodes that depend on it).
  - The graph has not reached END or exceeded the recursion limit.

If no tasks are scheduled, the graph terminates. If `step > recursion_limit`, a `GraphRecursionError` is raised.

---

## Task Scheduling

### Task Types: PUSH vs PULL

**PUSH tasks** (from `prepare_next_tasks`, lines 442–466 in `_algo.py`):

- Originate from the TASKS channel, enqueued by `Send` objects returned from conditional edges or nodes.
- Each Send specifies the target node, input state, and optional timeout.
- PUSH tasks execute immediately in the same superstep, enabling **dynamic fan-out**.

```python
def router(state):
    return [
        Send("process_item", {"item": item})
        for item in state["items"]
    ]
```

All returned Send objects spawn PUSH tasks in the next superstep.

**PULL tasks** (from `prepare_next_tasks`, lines 488–512 in `_algo.py`):

- A node is pulled when at least one of its trigger channels has a newer version than the node's `versions_seen` record.
- Trigger channels are declared at graph compile time; the `trigger_to_nodes` map accelerates lookup.
- PULL tasks execute in the same superstep as PUSH tasks.

### Trigger Mechanism

When a node is added to a StateGraph, its `triggers` are inferred from its `subscribe_to` declarations (in NodeBuilder or @entrypoint). The scheduler uses `_triggers()` (in `_algo.py`) to check:

```python
def _triggers(channels, channel_versions, versions_seen, null_version, proc):
    """Return True if proc should run in the next superstep."""
    seen = versions_seen or {}
    for chan in proc.triggers:
        seen_version = seen.get(chan, null_version)
        current_version = channel_versions.get(chan, null_version)
        if current_version > seen_version:  # type: ignore[operator]
            return True  # At least one trigger channel is new
    return False
```

### Task Identity and Caching

Each task receives a unique `task_id` (generated by xxh3_128_hexdigest or uuid5, depending on checkpoint version). The ID encodes:

- Checkpoint namespace and node name.
- Step number.
- Task path (PUSH vs PULL, Send index, etc.).

Task IDs enable:

- **Input caching**: If a task with the same arguments ran in a prior step, cache its output and skip execution (controlled by `cache_policy`).
- **Deterministic replay**: On resume, task IDs are stable, allowing the runner to re-execute only tasks that failed.
- **Error handler routing**: Failed tasks are linked to error handlers via the source task ID.

---

## Node Invocation and Execution

### PregelNode: The Container

A `PregelNode` (in `pregel/_read.py`) wraps a user function or Runnable and declares:

- **bound**: The actual function or Runnable to execute.
- **triggers**: Input channel names.
- **channels**: Output channel names (write destinations).
- **error_handler_node**: Optional error handler name.
- **tags, metadata, retry_policy, cache_policy, timeout**: Policies applied at execution.

PregelNode does not itself execute; instead, it provides the metadata needed to construct a `PregelExecutableTask`.

### PregelExecutableTask: The Executable Unit

A `PregelExecutableTask` (in `types.py`, lines 666–681) is the runtime representation:

```python
@dataclass
class PregelExecutableTask:
    name: str                          # Node name
    input: dict[str, Any]             # Read state (channel snapshot)
    proc: Runnable                    # The bound runnable
    writes: deque[tuple[str, Any]]    # Accumulated writes (mutable)
    config: RunnableConfig            # Full context (read fn, send fn, etc.)
    id: str                           # Unique task ID
    path: tuple[str | int | tuple, ...] # Task path (PUSH/PULL, index, etc.)
```

The `config` (RunnableConfig) injects:

- **CONFIG_KEY_READ**: A partial function (`local_read`) to read channels during task execution.
- **CONFIG_KEY_SEND**: A callback to append writes to `task.writes`.
- **CONFIG_KEY_TASK_ID, CONFIG_KEY_CHECKPOINT_NS**: For tracing and error routing.

### Execution Pipeline

**ChannelRead & ChannelWrite abstractions** (in `pregel/_read.py` and `pregel/_write.py`):

- **ChannelRead**: Injected runnables that read channels on-demand. Called by conditional edges or within node logic via `config[CONF][CONFIG_KEY_READ]`.
- **ChannelWrite**: Injects writes to channels. Each node's output is wrapped in ChannelWrite, which invokes the send callback.

The user's node function is wrapped in a sequence:

```
[node_function] → [ChannelWrite(channels)]
```

When the node returns state updates, ChannelWrite batches them and appends to task.writes.

**PregelRunner** (`pregel/_runner.py`, _sync and _async):

- Iterates over tasks.
- Applies retry policies, timeout, and cache logic.
- Invokes `node.proc.invoke()` or `node.proc.ainvoke()` with the task's config.
- Catches exceptions and routes to error handlers if defined.
- Accumulates results in task.writes.

---

## State Advancement and Channel Versions

### Checkpoint Structure

A checkpoint (from `checkpoint/base/__init__.py`) is immutable once saved and contains:

```python
{
    "v": 1,                              # Version
    "id": "uuid-hex",                   # Checkpoint ID
    "ts": "2024-01-15T...",             # Timestamp
    "channel_values": {                 # Deserialized channel state
        "messages": [...],
        "count": 5
    },
    "channel_versions": {               # Current version of each channel
        "messages": 3,
        "count": 3
    },
    "versions_seen": {                  # Per-node: what it has processed
        "agent": {"messages": 2, "count": 2},
        "summarizer": {"messages": 3}
    },
    "updated_channels": ["messages"]    # Which channels changed in this step
}
```

### Version Semantics

**Channels** have versions (int, float, or string). Each time `apply_writes` updates a channel, the version increments (via `get_next_version`, typically `increment(v) = v + 1 if v else 1`).

**Nodes** record `versions_seen[node_name][channel]` after processing. This allows the scheduler to skip a node if all its trigger channels are at versions it has already seen.

```python
# After node "agent" ran on superstep 2:
versions_seen["agent"] = {"messages": 2, "count": 2}

# On superstep 3, "count" is updated to version 3 but "messages" stays 2
channel_versions = {"messages": 2, "count": 3}

# Should "agent" run?
# agent.triggers = ["messages", "count"]
# agent has seen messages@2 (current is 2 ✗) and count@2 (current is 3 ✓)
# At least one is new → trigger "agent" in superstep 4
```

### Managing Concurrent Updates

When multiple nodes write to the same channel in one superstep:

- All writes are batched in `pending_writes_by_channel[chan]`.
- The channel's reducer is invoked: `new_value = channel.reducer(current, [write1, write2, ...])`.
- The reducer must be associative (batching-invariant) so replaying is safe.

**Example: messages channel with `add_messages` reducer**

```python
current = [msg1, msg2]
writes = [[msg3, msg4], [msg5]]  # From two tasks
result = add_messages(current, [msg3, msg4, msg5])
# Deduplicates by ID, appends new messages, handles RemoveMessage
```

---

## Runtime Flow: Sync and Async

### SyncPregelLoop (Synchronous Execution)

Instantiated by `Pregel.invoke()`:

1. **__enter__** (`_first`): Load checkpoint from saver, initialize channels, apply input writes.
2. **tick loop** (in runner):
   - Call `loop.tick()`: Prepare tasks, check interrupt_before, emit debug events.
   - Call `runner.invoke()`: Execute tasks in ThreadPoolExecutor.
   - Call `loop.after_tick()`: Apply writes, save checkpoint, check interrupt_after.
   - Repeat until no tasks or recursion limit.
3. **__exit__**: Handle graph termination, durability, delta-channel snapshots, emit final events.

### AsyncPregelLoop (Asynchronous Execution)

Instantiated by `Pregel.ainvoke()`:

- Uses asyncio task groups instead of ThreadPoolExecutor.
- Calls `loop.atick()` and `runner.ainvoke()` in async context.
- Supports concurrent writes to checkpointer (durability="async").

### Error Handling in Runner

PregelRunner.invoke/ainvoke wraps task execution in try-except:

1. **Task execution fails** with exception `exc`.
2. Check if the node has an `error_handler_node`.
3. If yes: Write ERROR and ERROR_SOURCE_NODE markers to checkpoint_pending_writes, then schedule the error handler as a new task (same superstep).
4. If no: Raise NodeError (which may trigger graph-level error handling or propagate).

Error handlers receive the failed task's output and the exception. They can update state, invoke retry, or gracefully terminate.

---

## Durability Modes

The `Durability` enum controls when writes are persisted:

- **"sync"** (default): After each superstep completes, `apply_writes` is called, checkpoint is saved, and the next superstep begins. Guarantees all writes are durable before proceeding.
  
- **"async"**: Writes are persisted in the background (next superstep may execute before writes are confirmed durable). Uses `checkpointer_put_after_previous` to ensure order: writes from superstep N must be durable before superstep N+1's checkpoint is finalized.

- **"exit"**: Writes are accumulated in `_exit_delta_writes` and persisted only when the graph exits (all supersteps complete). Minimal overhead but no intermediate checkpoints.

### Delta Channels and Snapshots

DeltaChannel (for efficient list/message accumulation) stores writes as individual entries and reconstructs state by replaying writes through the reducer. To bound replay depth, snapshots are taken at configurable intervals (e.g., every 20 supersteps). On graph exit in "exit" mode, all delta writes are persisted under an anchor checkpoint.

---

## Interrupts and Control Flow

### Graph Interrupts

Two interrupt points are checked each superstep:

- **interrupt_before**: Before task execution, if any trigger channels were updated.
- **interrupt_after**: After write application, same condition.

If triggered, `GraphInterrupt` is raised, checkpoint is saved, and the graph pauses. The caller can inspect the state, insert a resume value, and resume the graph by calling `invoke(..., {"interrupt_id": resume_value})`.

### Command: Multi-Directional Control

A node can return a `Command` to:

- **update**: Merge state changes (like a regular return).
- **goto**: Override normal edge routing and jump to specific node(s).
- **resume**: Resume one or more pending interrupts.
- **graph**: Route the command to a parent or sibling subgraph scope.

Commands are converted to writes via `map_command` (in `pregel/_io.py`) and processed like regular updates.

---

## Concurrency and Parallelism

### Task-Level Parallelism

All tasks in a superstep execute in parallel (no dependencies within a superstep). Nodes that would create circular dependencies are not allowed at compile time; the graph is a DAG.

### Channel Updates are Atomic

Writes from a superstep are applied atomically: either all succeed or none (except in async mode, where durability is eventually consistent). This ensures no node sees a partial update.

### Managed Values

Some state is managed externally (e.g., via a Context channel that manages a resource lifecycle). ManagedValueMapping handles get/set operations, ensuring consistency across the superstep.

---

## Key Invariants and Guarantees

1. **Determinism**: Given the same input and checkpoint, the execution produces the same output.
2. **Atomicity**: All writes from a superstep are applied together; no node sees mid-superstep state.
3. **Progress**: The graph either reaches END, exceeds the recursion limit, or is interrupted. No infinite loops on well-formed graphs.
4. **Checkpointability**: State is saved at every superstep boundary, enabling resume and time-travel debugging.
5. **Version-driven scheduling**: Nodes are triggered only when their input channels change, reducing redundant execution.

---

## Extension Points and Configuration

- **Retry policies** (pregel/_retry.py): Exponential backoff, jitter, max retries, custom predicates.
- **Cache policies** (pregel/_cache.py): Skip re-execution if input matches a prior task.
- **Timeout policies** (pregel/_timeout.py): Interrupt a task if execution exceeds a deadline.
- **Trace policies** (types.py): Control what events are logged (nodes, edges, steps, etc.).
- **Error handlers**: Node-scoped or graph-scoped handlers intercept and respond to failures.
- **Managed values**: Lifecycle-aware state (e.g., HTTP client connection pool).
- **Custom channels**: Implement reducer logic for domain-specific accumulation (e.g., vector aggregation).

---

## Representative Usage

```python
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: Annotated[list, operator.add]
    count: int

def node_a(state):
    return {"messages": [f"Step {state['count']}"], "count": state["count"] + 1}

def node_b(state):
    return {"messages": [f"Processed: {state['count']}"]}

graph = StateGraph(State)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", END)

compiled = graph.compile()

# Invocation triggers the superstep loop:
# Superstep 0: START → node_a (PULL)
# Superstep 1: node_b (PULL, triggered by "count" update)
# Superstep 2: No triggers → END
result = compiled.invoke({"messages": [], "count": 0})
```

---

## See Also

- [Core Concepts](./core-concepts.md): Nodes, edges, channels, and the StateGraph API.
- [Checkpoint Persistence](../operations/checkpoint-persistence.md): Saving and resuming from checkpoints.
- Source files:
  - `pregel/_algo.py`: Core superstep and task scheduling logic.
  - `pregel/_loop.py`: Sync/async loop orchestration and state advancement.
  - `pregel/_runner.py`: Node invocation, retry, cache, timeout.
  - `pregel/_executor.py`: Concurrent execution contexts (threads, asyncio).
  - `errors.py`: Error types and handling.
