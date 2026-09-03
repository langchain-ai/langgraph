---
type: Persistence & State
title: Checkpointing and Memory
description: Architecture of checkpoint persistence, state snapshots, and durable graph execution enabling resumption from interrupts and time-travel debugging.
tags: [checkpointing, persistence, durability, state-snapshot, resumption, thread-id]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-7dd1ecd6e49bec7af09e11e7
    resource: repo://libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py
  - id: openwiki-source-6c95109f667df245389a281a
    resource: repo://libs/checkpoint/langgraph/checkpoint/base/__init__.py
  - id: openwiki-source-0bb72cb2f8e9b84c7909edba
    resource: repo://libs/checkpoint/langgraph/checkpoint/memory/__init__.py
  - id: openwiki-source-16071c666268b16a8eb57a30
    resource: repo://libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py
  - id: openwiki-source-a37efa6c36fc4469731bb7be
    resource: repo://libs/langgraph/langgraph/_internal/_constants.py
  - id: openwiki-source-c6341920f9103722ec2f4354
    resource: repo://libs/langgraph/langgraph/pregel/_checkpoint.py
  - id: openwiki-source-2f8b73594e09a8a8dcb396c5
    resource: repo://libs/langgraph/langgraph/pregel/_loop.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

Checkpointing in LangGraph provides durable state snapshots that enable graphs to persist across execution interruptions, resume from failures, and support time-travel inspection. The checkpoint system is built on the `BaseCheckpointSaver` abstraction, which defines a common interface for storing and retrieving graph state across multiple runtime backends: in-memory (testing only), SQLite, PostgreSQL, and custom implementations.

Checkpoints capture the complete execution state—channel values, version metadata, and task coordination—at logical points in the graph execution loop. By coupling checkpoint persistence with thread IDs and checkpoint IDs, users can build multi-session conversational state, replay from any historical point, and handle interrupts (user-initiated pauses for human input or decisions) transparently.

---

## Checkpoint Abstraction

### BaseCheckpointSaver Interface

All checkpoint persistence in LangGraph flows through `BaseCheckpointSaver` (repo://libs/checkpoint/langgraph/checkpoint/base/__init__.py#L177-L723), a generic base class defining the core contract for checkpoint storage and retrieval.

**Core Methods:**

- **`put(config, checkpoint, metadata, new_versions)`**: Atomically write a checkpoint snapshot along with metadata and updated channel versions. Returns an updated `RunnableConfig` so the caller can track the checkpoint ID for later retrieval.
- **`get_tuple(config)`**: Retrieve a checkpoint tuple (checkpoint data + metadata + parent reference) given a `RunnableConfig` containing a thread ID and optional checkpoint ID for time-travel.
- **`list(config, filter=None, before=None, limit=None)`**: Enumerate checkpoints matching criteria (e.g., all checkpoints on a thread or before a specific point).
- **`delete_thread(thread_id)`**: Erase all checkpoints and associated writes for a thread.
- **`put_writes(config, writes, task_id)`**: Store intermediate writes (pending state changes) that succeeded in a node but whose containing checkpoint failed; enables replay of successful side effects on resume.

**Async Variants:** Each core method has an async equivalent (`aget_tuple`, `aput`, etc.) for non-blocking I/O in async execution contexts.

**Delta Channel Support:** The optional `get_delta_channel_history(config, channels)` method walks the parent checkpoint chain to reconstruct delta-channel state from incremental writes and snapshots, enabling efficient storage of frequently-updated channels.

All implementations default to `JsonPlusSerializer`, which serializes checkpoints to msgpack with fallback support for LangChain types (tools, messages, dates). The serializer is configurable to support custom types via allowlist (see "Serialization Security" below).

### Checkpoint Structure

A `Checkpoint` (repo://libs/checkpoint/langgraph/checkpoint/base/__init__.py#L93-L125) is a TypedDict containing a complete state snapshot:

```python
{
  "v": 1,                      # Format version (currently 1)
  "id": "uuid6-timestamp",      # Unique, monotonically increasing checkpoint ID
  "ts": "2024-01-15T...",       # ISO 8601 timestamp
  "channel_values": {...},      # Deserialized channel state dict[name -> value]
  "channel_versions": {...},    # Version tracking dict[name -> int|str|float]
  "versions_seen": {...},       # Per-node channel version history
  "updated_channels": [...]     # List of channels modified at this checkpoint
}
```

- **channel_values**: The value of each channel at checkpoint time. For delta channels, contains either a `_DeltaSnapshot` blob (a periodic full snapshot) or is omitted (state reconstructed from ancestor writes).
- **channel_versions**: Monotonically increasing version identifiers for each channel, used to determine which nodes have unprocessed updates.
- **versions_seen**: A per-node map tracking which channel versions each node has already consumed, enabling incremental task scheduling.
- **updated_channels**: Optimization hint listing which channels were modified since the previous checkpoint.

### CheckpointTuple and Metadata

A `CheckpointTuple` (repo://libs/checkpoint/langgraph/checkpoint/base/__init__.py#L140-L148) packages a checkpoint with its operational context:

```python
CheckpointTuple(
  config: RunnableConfig,           # Checkpoint coordinate (thread_id, checkpoint_ns, checkpoint_id)
  checkpoint: Checkpoint,            # The snapshot itself
  metadata: CheckpointMetadata,      # Source, step, parent IDs, delta channel counters
  parent_config: RunnableConfig,     # Config of the previous checkpoint (for parent chain walks)
  pending_writes: list[PendingWrite] # Writes from failed tasks to replay on resume
)
```

`CheckpointMetadata` (repo://libs/checkpoint/langgraph/checkpoint/base/__init__.py#L39-L87) captures operation semantics:

- **source**: `"input"` (initial state), `"loop"` (execution step), `"update"` (manual state update), or `"fork"` (copied checkpoint).
- **step**: -1 for input, 0+ for subsequent steps.
- **parents**: Mapping of nested graph namespaces to parent checkpoint IDs (for forked/subgraph contexts).
- **counters_since_delta_snapshot**: Per-delta-channel (updates, supersteps) counters determining when to snapshot (beta feature).

---

## Configuration and Namespacing

### Thread ID and Checkpoint ID

A `thread_id` is the primary key for isolating execution contexts. The same graph invocation repeatedly with the same `thread_id` accumulates state on that thread (e.g., chat history in a multi-turn conversation). A unique `thread_id` per invocation yields independent execution traces.

Threads are passed via config:

```python
config = {"configurable": {"thread_id": "user-123"}}
graph.invoke(inputs, config)
```

A `checkpoint_id` (generated by the saver, typically a UUID6) uniquely identifies a snapshot and enables time-travel: invoke with a previous checkpoint ID to resume from mid-execution rather than the latest state.

**Configuration Keys** (repo://libs/langgraph/langgraph/_internal/_constants.py#L52-L59):

- `CONFIG_KEY_THREAD_ID` (`"thread_id"`): Thread identifier.
- `CONFIG_KEY_CHECKPOINT_ID` (`"checkpoint_id"`): Current checkpoint ID.
- `CONFIG_KEY_CHECKPOINT_NS` (`"checkpoint_ns"`): Namespace for nested graphs; empty string (`""`) for root.

### Checkpoint Namespace

A `checkpoint_ns` isolates state for subgraphs and nested execution contexts, formatted as a pipe-separated hierarchy (e.g., `"outer|inner"` for nested subgraphs). The root graph uses an empty namespace. Namespacing allows a single thread to maintain separate checkpoint chains for independent subgraph branches, enabling forking and parallel composition without state collision.

---

## Checkpoint Lifecycle in Execution

### When Checkpoints Are Created

The execution loop in `PregelLoop` (repo://libs/langgraph/langgraph/pregel/_loop.py) creates and saves checkpoints at strategic points determined by the **durability mode**:

1. **`durability="sync"`** (default): Save after every superstep (node execution).
2. **`durability="async"`**: Save concurrently during the next superstep; no blocking on I/O.
3. **`durability="exit"`**: Save only when the graph completes or encounters an interrupt.

The `_put_checkpoint(metadata)` method (repo://libs/langgraph/langgraph/pregel/_loop.py#L1081-L1220) orchestrates checkpoint creation:

- **Channel state capture**: For each channel, call `channel.checkpoint()` to obtain its serializable state.
- **Version advancement**: Call `saver.get_next_version(current_version, None)` to generate new version identifiers (defaults to incrementing integers).
- **Delta snapshots**: Periodically capture full state blobs for `DeltaChannel` instances (whose normal checkpoints contain only incremental writes).
- **Metadata assembly**: Build `CheckpointMetadata` with source, step counter, and delta counters.
- **Asynchronous persistence**: Submit the save operation (via `self._checkpointer_put_after_previous`) ensuring saves are ordered and don't block task execution.

### Resumption Flow

When a graph resumes (e.g., after an interrupt), the loop calls `channels_from_checkpoint` (repo://libs/langgraph/langgraph/pregel/_checkpoint.py#L229-L277):

1. **Fetch checkpoint**: Retrieve the target checkpoint tuple via `saver.get_tuple(config)`.
2. **Hydrate regular channels**: Call `spec.from_checkpoint(checkpoint["channel_values"][name])` for each non-delta channel.
3. **Replay delta channels**: For delta channels absent from `channel_values`, invoke `saver.get_delta_channel_history(config, channels)` to walk ancestors, find a seed (full snapshot), and replay intermediate writes.
4. **Apply pending writes**: Integrate any `pending_writes` from the checkpoint tuple (writes from failed tasks that succeeded).

Async resumption uses `achannels_from_checkpoint` with the same logic.

---

## Pending Writes and Fault Recovery

When a node task succeeds but the checkpoint save fails, the execution loop stores those writes as **pending writes** in `CheckpointTuple.pending_writes`. These are PendingWrite tuples: `(task_id, channel_name, value)`.

On resumption, the loop replays pending writes before scheduling the next tasks, ensuring that successful side effects are not lost due to downstream failures. The `put_writes(config, writes, task_id)` method allows the loop to persist intermediate writes separately, decoupling node execution from checkpoint durability for better throughput.

---

## Implementations

### InMemorySaver

`InMemorySaver` (repo://libs/checkpoint/langgraph/checkpoint/memory/__init__.py#L33-L430) stores checkpoints in process memory using nested dicts:

- **Thread → Namespace → Checkpoint ID → (checkpoint_data, metadata_data, parent_id)**: Three-level hierarchy matching the logical structure.
- **Blobs**: Separate storage for channel value blobs, keyed by (thread_id, namespace, channel_name, version).
- **Writes**: Pending writes indexed by (thread_id, namespace, checkpoint_id) and (task_id, write_index).

**Use cases:** Testing, debugging, single-machine prototypes. Not suitable for production or multi-process systems (state is lost on restart).

**Features:**
- Full `DeltaChannel` support with optimized `get_delta_channel_history` that walks the chain once for all channels.
- Context manager support (`__enter__` / `__exit__`, `__aenter__` / `__aexit__`) for lifecycle management.
- `setup()` is a no-op (no tables to create).

### PostgresSaver / AsyncPostgresSaver

Provided by `langgraph-checkpoint-postgres`, these implementations persist checkpoints to PostgreSQL, enabling production multi-tenant and high-availability deployments.

**Schema:** Checkpoints, metadata, and writes are stored in relational tables with optimizations for delta channels (separate `checkpoint_writes` table for incremental updates).

**Async support:** AsyncPostgresSaver uses async-await for I/O, suitable for concurrent graph execution.

**Setup:** `saver.setup()` creates tables if they don't exist (idempotent).

### SQLiteSaver / AsyncSQLiteSaver

Provided by `langgraph-checkpoint-sqlite`, these offer a lightweight persistent option for single-machine or small-team deployments.

**Suitable for:** Development, small services, local multi-threading scenarios.

**Setup:** `saver.setup()` initializes the database file.

### Custom Implementations

Subclass `BaseCheckpointSaver` and implement the core methods. Key patterns:

- Store `(thread_id, checkpoint_ns, checkpoint_id)` tuples as composite keys.
- Maintain parent chain links (via `parent_config`) to enable ancestor walks for delta channels.
- Respect the `pending_writes` contract: return them in `CheckpointTuple` so the loop can replay on resume.
- Ensure `put` is atomic or idempotent (retries must not corrupt state).

---

## Serialization and Security

The `JsonPlusSerializer` (repo://libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py#L82-L95) encodes checkpoints to msgpack by default, with fallback handling for common LangChain types:

- **Type Support**: dates, times, decimals, UUIDs, IP addresses, enums, dataclasses, LangChain messages/tools, custom Pydantic models.
- **Msgpack Encoding**: Provides compact binary format and native type hints.
- **Fallback**: JSON encoding for compatibility with pre-msgpack checkpoint versions.

**Security Note**: The serializer should not be used on untrusted data. If an attacker can write directly to your checkpoint storage, deserialization can execute arbitrary code. Mitigate with:

- **Strict Mode** (`LANGGRAPH_STRICT_MSGPACK=true` environment variable): Restrict deserialization to a built-in allowlist of safe types.
- **Custom Allowlist**: Pass `allowed_msgpack_modules` to `JsonPlusSerializer` to explicitly allow your application's types.
- **Encrypted Storage**: Use `EncryptedSerializer` (repo://libs/checkpoint/langgraph/checkpoint/serde/encrypted.py) to wrap the serializer and encrypt at rest.

---

## Delta Channels (Beta)

For channels that update frequently (e.g., message histories), storing the full state at every checkpoint is wasteful. `DeltaChannel` (repo://libs/langgraph/langgraph/channels/delta.py) stores only deltas:

- **Snapshots**: Full state blobs (`_DeltaSnapshot`) written every `snapshot_frequency` updates or when `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` is reached.
- **Incremental Writes**: Between snapshots, only the delta is stored in `checkpoint_writes` rows.
- **Reconstruction**: `get_delta_channel_history` walks ancestors to find the nearest snapshot seed and replays writes to reconstitute current state.

**Metadata Tracking**: `CheckpointMetadata["counters_since_delta_snapshot"]` maintains `(updates, supersteps)` counters per delta channel to drive snapshotting decisions.

**Pruning Caveat**: Naive deletion of intermediate checkpoints can break delta channel reconstruction (no snapshot ancestor to restore from). Safe pruning strategies include walking back from kept checkpoints to preserve all ancestors back to a snapshot, or forcing a fresh snapshot before pruning.

---

## Interrupts and Checkpoints

When a node raises a `GraphInterrupt`, the loop saves the current checkpoint (if `durability != "exit"`) and halts execution, returning an interrupt event to the caller. The caller can then resume later with the same `thread_id` and `checkpoint_id`:

```python
# First invocation hits an interrupt
try:
    result = graph.invoke(inputs, {"configurable": {"thread_id": "user-1"}})
except GraphInterrupt:
    # Graph is paused; checkpoint is saved
    pass

# Resume from the interrupt checkpoint
result = graph.invoke(
    resume_value,
    {"configurable": {"thread_id": "user-1"}}  # Resumes from latest checkpoint
)
```

The loop automatically loads the checkpoint, hydrates channels, replays pending writes, and continues execution from the next scheduled task.

---

## Durability Modes and Guarantees

### Sync Durability

- **Timing**: Checkpoint persists **before** the next node executes.
- **Guarantee**: Strong. Every step is durable; no data loss on process crash mid-step.
- **Trade-off**: Synchronous I/O blocks the execution loop; suitable for fast storage or when durability is critical.

### Async Durability

- **Timing**: Checkpoint persists **during** the next node execution (concurrent).
- **Guarantee**: Moderate. Task scheduling proceeds without waiting for I/O; crash between task start and checkpoint completion may lose the step, but not the task itself (retryable).
- **Trade-off**: Better throughput; slightly lower safety margin.

### Exit Durability

- **Timing**: Checkpoint persists **only at graph completion** or on interrupt.
- **Guarantee**: Weak within a run (intermediate crashes lose in-flight steps), but strong across boundaries. Final state is always saved.
- **Trade-off**: Minimum I/O overhead; suitable for short-lived graphs or when failure recovery is acceptable.

---

## Operational Patterns

### Multi-Session Conversational Memory

Reuse the same `thread_id` across multiple `invoke()` calls to accumulate state in a thread:

```python
thread_id = "user-123"
for query in user_queries:
    result = graph.invoke(
        {"question": query},
        {"configurable": {"thread_id": thread_id}}
    )
    # State (e.g., chat history) persists on thread_id
```

### Time-Travel Debugging

Resume from any historical checkpoint:

```python
# List all checkpoints on a thread
for checkpoint_tuple in saver.list(
    {"configurable": {"thread_id": "user-123"}}
):
    print(f"Checkpoint {checkpoint_tuple.checkpoint['id']} at step {checkpoint_tuple.metadata['step']}")

# Resume from a specific historical checkpoint
graph.invoke(
    resume_input,
    {
        "configurable": {
            "thread_id": "user-123",
            "checkpoint_id": "target-checkpoint-id"
        }
    }
)
```

### Stateful Agents

Agents can maintain conversation history and tool execution logs across invocations on a single thread, with periodic pruning (via `saver.prune()`) to prevent unbounded checkpoint accumulation.

---

## Configuration and Setup

### Minimal Configuration

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver

builder = StateGraph(State)
# ... add nodes, edges ...

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Invoke with thread_id
result = graph.invoke(
    inputs,
    {"configurable": {"thread_id": "my-thread"}}
)
```

### Production Setup with PostgreSQL

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://...")
checkpointer.setup()  # Create tables

graph = builder.compile(checkpointer=checkpointer)
```

### Custom Serializer

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

cipher = YourCipher()  # Implement CipherProtocol
serde = EncryptedSerializer(cipher, JsonPlusSerializer())
checkpointer = PostgresSaver(..., serde=serde)
```

---

## Invariants and Failure Semantics

1. **Parent Chain Integrity**: Each checkpoint maintains a link to its parent via `parent_config`. Ancestor walks must never break (required for delta channel reconstruction).
2. **Checkpoint ID Uniqueness**: IDs are unique within a thread and namespace; duplicate puts should be idempotent or rejected.
3. **Version Monotonicity**: Channel versions must be strictly increasing; no rewinds or gaps.
4. **Pending Write Visibility**: Writes stored at a checkpoint must be returned in `CheckpointTuple.pending_writes` on retrieval, ensuring replay on resume.
5. **Durability Boundary**: A checkpoint is durable only after `put()` completes and is acknowledged by the saver backend.

---

## See Also

- **Graph Execution Model** (`/openwiki/architecture/graph-execution-model.md`): How checkpoints integrate with task scheduling and channel updates.
- **Checkpoint Persistence Operations** (`/openwiki/operations/checkpoint-persistence.md`): Operational procedures for backup, pruning, and migrating checkpoints.
