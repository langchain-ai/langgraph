# Pregel Execution Engine Specification

## Overview

Pregel is the foundational execution engine for LangGraph, implementing a Bulk Synchronous Parallel (BSP) computation model. Inspired by Google's original Pregel system, it provides a framework for executing computational graphs with stateful communication between nodes while maintaining strict invariants around execution order, state updates, and error handling.

## Conceptual Model

Pregel follows the Bulk Synchronous Parallel model, where computation proceeds in a series of synchronized steps called "supersteps." Each superstep consists of three distinct phases:

1. **Plan**: Determine which actors to execute based on pending channel updates
2. **Execute**: Run selected actors in parallel, collecting their outputs
3. **Update**: Apply all updates to channels atomically at the end of the step

This model ensures several critical properties:

- **Determinism**: Given the same input, execution produces the same output
- **Isolation**: Node executions within a superstep cannot observe each other's updates until the next superstep
- **Atomicity**: All channel updates from a superstep are applied at once
- **Checkpoint-ability**: The system state can be captured at superstep boundaries

## Core Components

### PregelProtocol

Abstract interface defining the contract for all Pregel implementations:

```python
class PregelProtocol(Protocol):
    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any: ...

    def stream(
        self,
        input: Any,
        config: Optional[dict] = None,
        stream_mode: Optional[StreamMode] = None,
        **kwargs: Any,
    ) -> Iterator[Any]: ...

    def get_state(self, thread_id: Optional[str] = None) -> Any: ...

    def update_state(self, thread_id: str, state: Any) -> None: ...

    def get_state_history(self, thread_id: str) -> list[Any]: ...
```

### PregelNode

Represents an actor in the system with the following properties:

```python
class PregelNode:
    def __init__(
        self,
        name: str,
        action: PregelExecutable,
        *,
        subscribe: Optional[Collection[str]] = None,
        trigger: Optional[str] = None,
        writers: Optional[Collection[str]] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None: ...
```

Key properties:

- **name**: Unique identifier for the node
- **action**: Function to execute when the node is triggered
- **subscribe**: Channel names this node listens to for updates
- **trigger**: Special condition for node execution
- **writers**: Channels this node can write to (for validation)
- **retry_policy**: Strategy for handling execution failures

### Channels

Communication paths that store values and propagate them between nodes:

```python
class Channel(Protocol):
    def get_value(self) -> Any: ...

    def update(self, value: Any) -> bool: ...

    def checkpoint(self) -> Any: ...

    def from_checkpoint(self, value: Any) -> None: ...
```

Each channel type implements this interface with specific behaviors:

1. **LastValue**: Stores only the most recent value; rejects multiple updates in a single step
2. **AnyValue**: Permits multiple updates within a step, storing the last one
3. **EphemeralValue**: Temporary storage that clears after being read
4. **UntrackedValue**: Like LastValue but excluded from checkpoints
5. **NamedBarrierValue**: Synchronization mechanism requiring all named values to be received
6. **BinaryOperatorAggregate**: Applies operations to combine values (sum, join, etc.)
7. **Topic**: PubSub topic supporting multiple subscribers and values

### Tasks

Units of work representing computations to execute:

```python
class PregelTask:
    def __init__(
        self,
        node: str,
        trigger: Optional[str] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None: ...

class PregelExecutableTask:
    def __init__(
        self,
        task: PregelTask,
        inputs: dict[str, Any],
        context: dict[str, Any],
    ) -> None: ...
```

### Checkpoints

Snapshots of execution state at superstep boundaries:

```python
class Checkpoint:
    def __init__(self, channel_values: dict[str, Any]) -> None: ...

    def get_values(self) -> dict[str, Any]: ...

    def update(self, channel_values: dict[str, Any]) -> None: ...
```

## Execution Flow

When `invoke()` or `stream()` is called, Pregel performs the following steps:

1. **Initialization**:

   - Create or retrieve a checkpoint for the specified thread
   - Load initial channel values from the checkpoint
   - Transform input into channel updates

2. **Superstep Loop**:

   - **Plan**: Identify nodes to execute based on channel updates
   - **Execute**: Run selected nodes in parallel, collecting updates
   - **Update**: Apply all updates to channels atomically
   - **Checkpoint**: Save the current state if checkpointing is enabled
   - Repeat until no more nodes are active or an END token is received

3. **Termination**:
   - Extract output from designated output channels
   - Return result or stream updates based on the stream mode

This process is depicted in the diagram below:

```
┌──────────────────────────────────┐
│            INPUT                 │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│        INITIALIZE STATE          │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│       SUPERSTEP LOOP             │
│  ┌────────────────────────────┐  │
│  │         PLAN               │  │
│  │  (Identify active nodes)   │  │
│  └────────────┬───────────────┘  │
│               │                   │
│               ▼                   │
│  ┌────────────────────────────┐  │
│  │        EXECUTE             │  │
│  │  (Run nodes in parallel)   │  │
│  └────────────┬───────────────┘  │
│               │                   │
│               ▼                   │
│  ┌────────────────────────────┐  │
│  │         UPDATE             │  │
│  │  (Apply channel updates)   │  │
│  └────────────┬───────────────┘  │
│               │                   │
│               ▼                   │
│  ┌────────────────────────────┐  │
│  │       CHECKPOINT           │  │
│  │  (Save current state)      │  │
│  └────────────┬───────────────┘  │
│               │                   │
│               ▼                   │
│      [More nodes active?]         │
│      /                \           │
│    Yes                No          │
│     │                  │          │
│     └──────────────────┘          │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│            OUTPUT                │
└──────────────────────────────────┘
```

## Stream Modes

Pregel supports different streaming options to provide visibility into execution:

- **values**: Stream the complete state after each superstep
- **updates**: Stream state deltas after each node execution
- **debug**: Stream comprehensive execution information for debugging

Each mode has different performance characteristics and use cases.

## Critical Invariants

Based on LangGraph's test suite, Pregel maintains the following invariants:

1. **Execution Determinism**:

   - Given the same input and thread ID, execution produces identical results
   - Channel update order within a superstep does not affect the final state

2. **State Isolation**:

   - Updates from one node are not visible to other nodes within the same superstep
   - Nodes cannot observe partial updates from incomplete executions

3. **Checkpoint Consistency**:

   - Checkpoints capture the complete system state at superstep boundaries
   - Restored checkpoints resume execution with the exact same state

4. **Task Management**:

   - Tasks execute in a deterministic order based on node dependencies
   - Parallel execution optimizes performance but maintains consistency

5. **Error Handling**:

   - Node execution failures can be handled with retry policies
   - Unrecoverable errors propagate without corrupting the execution state
   - Failures during checkpointing do not corrupt previous checkpoints

6. **Termination Guarantees**:
   - Execution always terminates for acyclic graphs
   - Cyclic graphs require explicit exit conditions to ensure termination
   - Execution timeouts prevent infinite loops

## Implementation Notes

### Type Safety

Pregel enforces type safety through:

- Input/output schema validation
- Channel type checking for updates
- Runtime validation of node return values

### Concurrency Model

Pregel balances parallelism with determinism:

- Nodes within a superstep can execute in parallel
- Channel updates are collected and applied sequentially
- Execution order is deterministic despite parallel processing

### Performance Optimizations

- **Eager Planning**: Pregel identifies all active nodes at the start of a superstep
- **Task Batching**: Similar tasks can be batched for efficient execution
- **Lazy Checkpointing**: Only modified channels are included in checkpoints
- **Channel-specific optimizations**: Different channel types use specialized storage strategies

### Testing Approach

The test suite for Pregel focuses on:

1. **Functional correctness**: Ensuring proper node execution and state updates
2. **Concurrency safety**: Verifying parallel execution does not affect determinism
3. **Error handling**: Confirming failures are properly managed
4. **Checkpoint fidelity**: Validating checkpoint creation and restoration
5. **Edge cases**: Testing unusual graph topologies and execution patterns

## Reimplementation Guidance

When reimplementing Pregel from scratch, consider the following approach:

1. Start with a simplified sequential execution model that maintains basic invariants
2. Add channel implementations one at a time, focusing on correctness
3. Implement the checkpoint system with proper serialization
4. Add parallel execution with careful attention to update ordering
5. Implement error handling and retry policies
6. Optimize for performance and resource usage

The most challenging aspects are:

- Maintaining determinism with parallel execution
- Ensuring checkpoint consistency
- Properly handling error cases
- Managing memory usage for large state objects

## Example Usage

```python
# Define node functions
def process_input(state):
    # Process input data
    return {"output_channel": processed_data}

def make_decision(state):
    # Make a decision based on processed data
    return {"decision_channel": decision}

# Create PregelNodes
input_node = PregelNode(
    name="input_processor",
    action=process_input,
    subscribe=["input_channel"],
    writers=["output_channel"]
)

decision_node = PregelNode(
    name="decision_maker",
    action=make_decision,
    subscribe=["output_channel"],
    writers=["decision_channel"]
)

# Create channels
channels = {
    "input_channel": LastValue(),
    "output_channel": LastValue(),
    "decision_channel": LastValue()
}

# Create Pregel instance
pregel = Pregel(
    nodes={"input_processor": input_node, "decision_maker": decision_node},
    channels=channels,
    checkpoint_factories={"memory": memory_checkpointer_factory}
)

# Invoke the graph
result = pregel.invoke(
    {"input_channel": input_data},
    config={"thread_id": "conversation_123"}
)

# Stream execution with updates
for update in pregel.stream(
    {"input_channel": input_data},
    config={"thread_id": "conversation_456"},
    stream_mode="updates"
):
    print(update)
```

## Related Components

Pregel interacts closely with:

- **StateGraph**: High-level API that compiles to Pregel
- **Channels**: Communication primitives used by Pregel
- **Checkpoint System**: Provides persistence for Pregel execution
- **Human-in-the-Loop**: Uses Pregel's checkpointing for interruption/resumption
