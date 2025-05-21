# Pregel Execution Engine Specification

## Overview

Pregel is the foundational execution engine for LangGraph, implementing a Bulk Synchronous Parallel (BSP) computation model. Inspired by Google's original Pregel system, it provides a framework for executing computational graphs with stateful communication between nodes while maintaining strict invariants around execution order, state updates, and error handling.

Before exploring the conceptual model, it's helpful to understand a few core terms. In Pregel, **nodes** are functions that process data and produce outputs. **Channels** serve as communication pathways that store and propagate values between nodes. During execution, the system organizes work into **tasks** and captures state in **checkpoints** at defined synchronization points.

## Conceptual Model

Pregel follows the Bulk Synchronous Parallel model, where computation proceeds in a series of synchronized steps called "supersteps." Each superstep consists of three distinct phases:

1. **Plan**: Determine which actors (nodes) to execute based on pending channel updates
2. **Execute**: Run selected actors in parallel, collecting their outputs
3. **Update**: Apply all updates to channels atomically at the end of the superstep

This model ensures several critical properties:

- **Determinism**: Given identical inputs, the system produces the same control flow. This isolates variability to the LLM outputs, not execution order or timing.
- **Isolation**: Node executions within a superstep cannot observe each other's updates until the next superstep. This controls exactly when information flows between LLMs, preventing premature or inconsistent observations that would lead to non-deterministic behavior when timing varies between runs.
- **Atomicity**: All channel updates from a superstep happen at once. This maintains consistent state transitions, preventing scenarios where some updates apply while related ones don't (or where updates apply in variable order). In multi-agent systems, this allows LLMs to coordinate their work on related tasks asynchronously.
- **Checkpoint-ability**: The system captures state at superstep boundaries, allowing processes to pause, resume, and recover without losing progress. This capability makes possible human review of intermediary results, recovery from infrastructure failures, and coordination between agents operating on different timescales.

## Core Components

### PregelProtocol

Abstract interface defining the contract for all Pregel implementations:

```python
class PregelProtocol(Protocol):
    def invoke(self, input: Any, config: Optional[dict] = None) -> Any: ...

    def stream(
        self,
        input: Any,
        config: Optional[dict] = None,
        stream_mode: StreamMode = 'updates',
    ) -> Iterator[Any]: ...

    def get_state(self, thread_id: Optional[str] = None) -> Any: ...

    def update_state(self, thread_id: str, state: Any) -> None: ...

    def get_state_history(self, thread_id: str) -> list[Any]: ...
```

### PregelNode

Represents an actor in the system with the following properties:

```python
class PregelNode:
    # name of the node, unique within the graph
    name: str
    # function to execute when the node is triggered
    action: Runnable
    # functions that convert node output into channel updates
    writers: Sequence[Runnable]
    # channels that trigger the node when updated
    trigger_channels: Sequence[str]
    # channels the node reads when triggered
    read_channels: Sequence[str]
    # retry policy for the node
    retry_policy: RetryPolicy
    # cache policy for the node
    cache_policy: CachePolicy
```

### Channels

Communication paths that store values and propagate them between nodes:

```python
class Channel(Protocol, Generic[V, U, C]):
    def get(self) -> V: ...

    def update(self, updates: Sequence[U]) -> bool: ...

    def checkpoint(self) -> C: ...

    @classmethod
    def from_checkpoint(cls, value: C) -> Channel: ...
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
class PregelExecutableTask:
    # task parameters (primary key)
    path: tuple[Union[str, int, tuple], ...]
    # hash of task parameters
    id: str
    # name of the node
    name: str
    # function to execute
    proc: Runnable
    # input to the node (1st arg)
    input: Any
    # config for the node (2nd arg)
    config: RunnableConfig
    # functions that convert node output into channel updates
    writers: Sequence[Runnable] = ()
    # accumulates channel updates, tuple (channel, value)
    writes: deque[tuple[str, Any]]
    # retry policy for the task
    retry_policy: Optional[RetryPolicy]
    # cache policy for the task
    cache_policy: Optional[CachePolicy]
```

### Checkpoints

Snapshots of execution state at superstep boundaries:

```python
class SendProtocol(Protocol):
    # node to execute
    node: str
    # input to the node
    arg: Any

class Checkpoint:
    """State snapshot at a given point in time."""

    v: int
    """The version of the checkpoint format. Currently 2."""
    id: str
    """The ID of the checkpoint. This is both unique and monotonically
    increasing, so can be used for sorting checkpoints from first to last."""
    ts: str
    """The timestamp of the checkpoint in ISO 8601 format."""
    channel_values: dict[str, Any]
    """The values of the channels at the time of the checkpoint.
    Mapping from channel name to deserialized channel snapshot value.
    """
    channel_versions: dict[str, str]
    """The versions of the channels at the time of the checkpoint.
    The keys are channel names and the values are monotonically increasing
    version strings for each channel.
    """
    versions_seen: dict[str, dict[str, str]]
    """Map from node ID to map from channel name to version seen.
    This keeps track of the versions of the channels that each node has seen.
    Used to determine which nodes to execute next.
    """
    pending_sends: List[SendProtocol]
    """List of inputs pushed to nodes but not yet processed.
    Cleared by the next checkpoint."""
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
│  │ (Identify active nodes:    │  │
│  │  pulls from triggered      │  │
│  │  channels/edges & pushes   │  │
│  │  from explicit Sends)      │  │
│  └────────────┬───────────────┘  │
│               │                  │
│               ▼                  │
│  ┌────────────────────────────┐  │
│  │        EXECUTE             │  │
│  │  (Run nodes in parallel)   │  │
│  └────────────┬───────────────┘  │
│               │                  │
│               ▼                  │
│  ┌────────────────────────────┐  │
│  │         UPDATE             │  │
│  │  (Apply channel updates)   │  │
│  └────────────┬───────────────┘  │
│               │                  │
│               ▼                  │
│  ┌────────────────────────────┐  │
│  │       CHECKPOINT           │  │
│  │  (Save current state)      │  │
│  └────────────┬───────────────┘  │
│               │                  │
│               ▼                  │
│      [More nodes active?]        │
│      /                \          │
│    Yes                No         │
│     │                  │         │
│     └──────────────────┘         │
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
- **messages**: Stream LLM messages token-by-token
- **custom**: Stream user-defined chunks emitted by nodes
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
   - Execution timeouts and step limits prevent infinite loops

## Example Usage: StateGraph

```python
class State(TypedDict):
    input: str
    output: str
    decision: str

# Define node functions
def process_input(state: State) -> dict[str, Any]:
    # Process input data
    return {
        "output": ...,
    }

def make_decision(state: State) -> dict[str, Any]:
    # Make a decision based on processed data
    return {"decision": ...}

# StateGraph definition
builder = StateGraph(State)
builder.add_node("process_input", process_input)
builder.add_node("make_decision", make_decision)
builder.add_edge(START, "process_input")
builder.add_edge("process_input", "make_decision")
```

### How is it compiled to Pregel?

```python
# Create channels
channels = {
    # the reserved entrypoint channel
    START: EphemeralValue(),
    # one channel per key in state schema
    "input": LastValue(),
    "output": LastValue(),
    "decision": LastValue(),
    # each node gets an "inbox" channel to trigger it
    "branch:to:process_input": EphemeralValue(),
    "branch:to:make_decision": EphemeralValue(),
}

# Create PregelNodes
process_input_node = PregelNode(
    name="process_input",
    action=process_input,
    trigger_channels=["branch:to:process_input"],
    read_channels=["input", "output", "decision"],
)

make_decision_node = PregelNode(
    name="make_decision",
    action=make_decision,
    trigger_channels=["branch:to:make_decision"],
    read_channels=["input", "output", "decision"],
)

# Define the start node

def __start__(state: State) -> Sequence[tuple[str, Any]]:
    # Trigger the first node, update the state channels
    return (
        ("branch:to:process_input", ...),
        ("input", state["input"]),
        ("output", state["output"]),
        ("decision", state["decision"]),
    )

__start__node = PregelNode(
    name=START,
    action=__start__,
    trigger_channels=[START],
    read_channels=[START],
)

# Create Pregel instance
pregel = Pregel(
    channels=channels,
    nodes={
        "process_input": process_input_node,
        "make_decision": make_decision_node,
        START: __start__node,
    },
    # the entrypoint channel
    input_channels=[START],
    # the keys in state schema are the output channels
    output_channels=["input", "output", "decision"],
)

# Invoke the graph
result = pregel.invoke(
    {"input": input_data},
)
```

## Execution Walkthrough

When you call `pregel.invoke({"input": input_data})`, the instance uses your input data to update the START channel. The system then examines all channels to find which ones have changed, discovering that the START channel now has new data. This change activates the `__start__` node, which is configured to run when the `__start__` channel updates.

The `__start__` node executes next, reading from the START channel and producing several outputs. It places your input into the "input" channel, initializes the "output" and "decision" channels with default values, and sends a message to the "branch:to:process_input" channel.

With the first round of updates applied, Pregel again checks which channels have changed. It finds that the "branch:to:process_input" channel now has new data, which triggers the process_input node (since it has `"branch:to:process_input"` as one of its `trigger_channels`). This node runs, reading from the "input", "output" and "decision" channels (the whole state). After applying its wrapped `process_input` function, it produces an update for a single channel (`"output"`). Its update is applied to the channel at the end of the superstep. Since we defined a directed edge from `"process_input"` to `"make_decision"`, Pregel also activates the "branch:to:make_decision" channel to ensure that that node is executed in the next superstep. Since no other nodes were triggered, the superstep ends.

In the next superstep, Pregel identifies that the "branch:to:make_decision" channel has changed, which activates the `make_decision` node (since it has `"branch:to:make_decision"` as one of its `trigger_channels`). This node runs, reading the complete state again. After applying its wrapped `make_decision` function, it produces an update for the "decision" channel. This update is applied at the end of the superstep. Pregel then checks if any more nodes should run, but finds none - no channels have new updates that would trigger additional nodes, and no more edges were defined in the graph. Since no other nodes were triggered, this superstep ends.

With no more work to do, Pregel concludes the execution and returns the final values from the output channels specified when creating the graph: "input", "output", and "decision".