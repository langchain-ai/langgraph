---
type: Architecture & Design
title: State and Channels
description: Deep dive into how state is defined, typed, and updated through channels, including channel types, reducer semantics, managed values, and input/output schemas.
tags: [state, channels, reducers, delta, last-value, managed-values, ephemeral, barrier, input-schema, output-schema, private-channels]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-a37efa6c36fc4469731bb7be
    resource: repo://libs/langgraph/langgraph/_internal/_constants.py
  - id: openwiki-source-9ac63269e899a21d67302cbf
    resource: repo://libs/langgraph/langgraph/channels/base.py
  - id: openwiki-source-c9e4dd1489dd92f56a9be7fd
    resource: repo://libs/langgraph/langgraph/channels/binop.py
  - id: openwiki-source-1358f0ad7cc3f870325b6135
    resource: repo://libs/langgraph/langgraph/channels/delta.py
  - id: openwiki-source-7cedc7be33f5b714b17bce1c
    resource: repo://libs/langgraph/langgraph/channels/ephemeral_value.py
  - id: openwiki-source-33278185a14e90f69cfefcb2
    resource: repo://libs/langgraph/langgraph/channels/last_value.py
  - id: openwiki-source-21f69261967311ecc39afb4e
    resource: repo://libs/langgraph/langgraph/channels/named_barrier_value.py
  - id: openwiki-source-e8fc48643d5576114cb8c3c9
    resource: repo://libs/langgraph/langgraph/channels/topic.py
  - id: openwiki-source-9b01a2a24dd7f7a0fb7e6f05
    resource: repo://libs/langgraph/langgraph/graph/message.py
  - id: openwiki-source-36058e684ab11833a28af83b
    resource: repo://libs/langgraph/langgraph/graph/state.py
  - id: openwiki-source-6cc6fc4ab2c58b65c5cbd03e
    resource: repo://libs/langgraph/langgraph/managed/base.py
  - id: openwiki-source-4ae17bf912e4007bb8b83bef
    resource: repo://libs/langgraph/langgraph/pregel/main.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

State is the core abstraction through which LangGraph nodes communicate. Rather than passing data through function parameters and returns, all nodes read from and write to a **typed, versioned state**. This enables:

- **Concurrent writes** from multiple nodes in the same superstep (merged via reducers)
- **State snapshots** at each superstep for checkpointing, resumption, and time-travel debugging
- **Incremental execution** via version tracking (nodes only run when their inputs change)

A **channel** is a typed, versioned container for one state field. LangGraph provides multiple channel types, each with different update semantics: `LastValue` (overwrite), `DeltaChannel` (accumulate), `Topic` (queue), `EphemeralValue` (one-step), and `NamedBarrierValue` (synchronization).

---

## State Schema Definition

State is declared as a **TypedDict, dataclass, or Pydantic model**. Each field can be annotated with a **reducer function** that controls how concurrent updates to that field are merged.

### Basic State Schema

```python
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    count: int                              # No reducer; last value wins
    messages: Annotated[list, operator.add]  # Reducer: concatenate lists
    done: bool
```

When no reducer is specified (unannotated field), the channel is `LastValue`: if multiple nodes write to the same key in one superstep, an error is raised. This prevents accidental concurrent writes to non-aggregatable fields.

When a reducer is specified via `Annotated[Type, reducer_func]`, the channel automatically selects the appropriate implementation based on the reducer's signature and type:

- **Binary function** (two parameters): `BinaryOperatorAggregate` channel
- **Sequence function** (accumulates via reducer): `DeltaChannel` (for append-only or list-like accumulation)
- **Callable class** (channel implementation): used directly

### Reducer Semantics

A reducer merges concurrent writes from multiple nodes in the same superstep:

```python
new_value = reducer(current_value, list_of_updates)
```

**Key Property:** Reducers must be **batching-invariant** (associative):

```
reducer(reducer(state, xs), ys) == reducer(state, xs + ys)
```

This allows LangGraph to replay writes in larger batches during checkpoint recovery without changing the reconstructed state.

### Built-in Reducers

- `operator.add`: List concatenation, dict merging (`[1, 2] + [3] = [1, 2, 3]`)
- `operator.or_`: Set union, bool OR (`{1, 2} | {3} = {1, 2, 3}`)
- `operator.mul`, `operator.and_`, etc.: Standard binary operators

### Custom Reducers

```python
def merge_dicts(current, updates):
    """Merge updates into current, with later updates overwriting earlier."""
    result = current.copy() if current else {}
    for u in updates:
        if u is not None:
            result.update(u)
    return result

class State(TypedDict):
    metadata: Annotated[dict, merge_dicts]
```

---

## Channel Types

Each state field is backed by a **channel** that implements the update semantics. StateGraph automatically selects the channel based on type hints and annotations.

### LastValue

**Use case:** Single-valued fields that should not be concurrently written.

```python
class State(TypedDict):
    count: int  # Implicitly LastValue[int]
```

- **Invariant:** Raises `InvalidUpdateError` if more than one update arrives in a superstep.
- **Semantics:** Stores the single value; reads return the value directly.
- **Checkpoint:** Value itself.

### DeltaChannel

**Use case:** Append-only sequences (messages, growing lists) where all updates should be merged via a reducer.

```python
from langgraph.channels import DeltaChannel

class State(TypedDict):
    messages: Annotated[list[Message], DeltaChannel(operator.add)]
```

- **Semantics:** Applies the reducer to accumulated state and a batch of updates: `new_state = reducer(state, [update1, update2, ...])`.
- **Snapshot Frequency:** Configurable to bound replay depth. Default is 1000 updates before writing a full snapshot blob; additional snapshots are taken if 5000 supersteps pass since the last snapshot.
- **Checkpoint:** Stores only a sentinel; state is reconstructed by replaying ancestor writes through the reducer.
- **Use in Messages:** Ideal for token-by-token streaming LLM outputs appended to a message list in multiple supersteps.

### BinaryOperatorAggregate

**Use case:** Fields with simple binary operators (add, or, max, etc.).

```python
class State(TypedDict):
    items: Annotated[list, operator.add]
    flags: Annotated[set, operator.or_]
```

- **Semantics:** For N updates, applies the binary operator pairwise: `(...((a ⊕ b) ⊕ c) ⊕ d)`.
- **Checkpoint:** Value itself.

### EphemeralValue

**Use case:** Temporary state available only during the current superstep; cleared after reads.

```python
class State(TypedDict):
    input_data: Annotated[dict, EphemeralValue]
```

- **Semantics:** Value set in superstep N is readable in superstep N but cleared afterward. Each superstep starts empty unless new data is written.
- **Guard Option:** By default (`guard=True`), raises `InvalidUpdateError` if multiple updates arrive; `guard=False` allows multiple updates (keeps the last).
- **Use Case:** Data passed via `Send(..., arg={...})` to dynamically spawned nodes.

### Topic

**Use case:** Queue-like streaming of fan-out tasks (internal use for `Send` dispatch).

```python
from langgraph.channels import Topic

task_channel = Topic(Send, accumulate=False)
```

- **Semantics:** Stores a deque of messages. Each `update()` flattens and appends. If `accumulate=False`, the channel is emptied after reads.
- **Internal Use:** Stores `Send` objects for dynamic node dispatch.

### NamedBarrierValue

**Use case:** Synchronization: collect named updates until all expected sources have written.

```python
from langgraph.channels import NamedBarrierValue

class State(TypedDict):
    result: Annotated[None, NamedBarrierValue({"node_a", "node_b", "node_c"})]
```

- **Semantics:** Tracks which named sources have written. Once all names in the set have written at least once, `is_available()` returns `True`.
- **Consume:** After a read, resets the seen set.
- **Use Case:** Ensure all parallel branches have completed before proceeding.

### AnyValue & UntrackedValue

- **AnyValue:** Untyped container; accepts any value without validation.
- **UntrackedValue:** Values stored but not versioned (don't trigger dependent nodes).

---

## Channel API and Lifecycle

### Reading

- `channel.get() -> Value`: Return the current value. Raises `EmptyChannelError` if never written.
- `channel.is_available() -> bool`: Return `True` if the channel contains data.

### Writing

- `channel.update(values: Sequence[Update]) -> bool`: Apply updates and return whether the channel changed.

### Lifecycle

- `channel.consume() -> bool`: Notify the channel it was read. Some channels (e.g., `Topic`) clear state after consumption.
- `channel.finish() -> bool`: Notify execution is tentatively ending. Used for final flushes or availability changes (e.g., `LastValueAfterFinish` becomes available only after finish).

### Checkpointing

- `channel.checkpoint() -> CheckpointValue`: Serialize the channel's state for persistence (often the value itself, or a sentinel for delta channels).
- `channel.from_checkpoint(data) -> Channel`: Restore a channel from persisted data, optionally replaying writes if stored as delta.

---

## Input, State, and Output Schemas

StateGraph supports three schemas to control what data is accepted, processed, and returned:

### State Schema

The **required** master schema defining all fields in the graph's state.

```python
class State(TypedDict):
    query: str
    results: Annotated[list, operator.add]
    is_done: bool
```

All nodes read and write to this schema. Channels are created for each field.

### Input Schema

**Optional** schema defining the structure of data passed to `invoke()` or `stream()`.

```python
class Input(TypedDict):
    query: str
    limit: int  # Not in state; used only during input mapping
```

- **Default:** If not specified, equals `state_schema`.
- **Mapping:** Input is mapped to state via a schema mapper that extracts matching field names.
- **Use Case:** Accept more fields than state contains (e.g., configuration parameters), or fewer fields (e.g., simplify user-facing API).

### Output Schema

**Optional** schema defining the structure of data returned from `invoke()` or final state in streaming.

```python
class Output(TypedDict):
    results: list
    metadata: dict
```

- **Default:** If not specified, equals `state_schema`.
- **Filtering:** Only fields in `output_schema` are returned; others remain hidden (no mutation).
- **Use Case:** Hide internal state fields (e.g., intermediate embeddings) from users.

### Private Channels

Fields in `state_schema` but **not** in `output_schema` are **private**: they're maintained internally but never returned to the user.

```python
class State(TypedDict):
    query: str
    embeddings: list  # Private intermediate data
    results: list     # Public output

class Output(TypedDict):
    results: list     # Only this is returned
```

---

## Managed Values

**Managed values** are special state fields auto-populated by the runtime with deterministic data. They do not participate in version tracking or trigger dependent nodes.

### Example: is_last_step

```python
from langgraph.managed import IsLastStep

class State(TypedDict):
    messages: list
    is_final_node: Annotated[bool, IsLastStep]

def my_node(state: State) -> dict:
    if state["is_final_node"]:
        return {"messages": [final_message]}
    return {}
```

**Semantics:**
- Available only during node execution (not in checkpoints).
- Set to `True` when the node is the last to execute in the run.
- Allows conditional final cleanup without explicit orchestration.

---

## Context Schema

The **context schema** is an optional, immutable, run-scoped dictionary passed to all nodes without polluting state. Use it for read-only configuration or shared resources.

```python
class Context(TypedDict):
    user_id: str
    db_connection: Any

def my_node(state: State, runtime: Runtime[Context]) -> dict:
    user = runtime.context["user_id"]
    # ... use user_id without storing in state
    return {"result": ...}

graph = StateGraph(State, context_schema=Context)
compiled = graph.compile()
result = compiled.invoke(
    {"messages": []},
    context={"user_id": "123", "db_connection": conn}
)
```

---

## Automatic Channel Selection

When you define a state field with `Annotated[Type, reducer]`, StateGraph infers the channel type:

```python
from typing import Annotated
from langgraph.channels import DeltaChannel, EphemeralValue
import operator

class State(TypedDict):
    # Explicit channel
    messages: Annotated[list, DeltaChannel(operator.add)]
    
    # Binary operator → BinaryOperatorAggregate
    items: Annotated[list, operator.add]
    
    # Callable class → used directly
    ephemeral: Annotated[dict, EphemeralValue]
    
    # No annotation → LastValue
    count: int
```

**Selection Logic:**
1. If annotation is a `BaseChannel` instance → use directly.
2. If annotation is a `BaseChannel` subclass → instantiate with the type.
3. If annotation is a callable (function) with exactly 2 positional parameters → create `BinaryOperatorAggregate`.
4. Otherwise → create `LastValue`.

---

## The `add_messages` Reducer

`add_messages` is a specialized reducer for message lists that handles deduplication and removal semantics:

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**Semantics:**
1. Appends new messages.
2. Deduplicates by message ID: if a message with the same ID already exists, the new one replaces it.
3. Supports `RemoveMessage` to tombstone entries.

**Example:**

```python
from langchain_core.messages import HumanMessage, AIMessage

msgs1 = [HumanMessage(content="Hello", id="1")]
msgs2 = [AIMessage(content="Hi!", id="2")]
add_messages(msgs1, msgs2)
# Result: [HumanMessage(..., id="1"), AIMessage(..., id="2")]

# Overwrite by ID
msgs3 = [HumanMessage(content="Hello again", id="1")]
add_messages(msgs1, msgs3)
# Result: [HumanMessage(..., id="1", content="Hello again")]
```

This is the **idiomatic pattern** for multi-turn conversations, ensuring message histories remain consistent across node executions.

---

## State Updates and Reducers: Examples

### Example 1: LastValue (Unannotated)

```python
class State(TypedDict):
    count: int

# Node 1 in superstep N
return {"count": 5}

# Node 2 in same superstep N
return {"count": 10}

# Error: InvalidUpdateError (count is LastValue, only one update allowed)
```

### Example 2: Reducer (Annotated with operator.add)

```python
class State(TypedDict):
    items: Annotated[list, operator.add]

# Node 1 in superstep N
return {"items": [1, 2]}

# Node 2 in same superstep N
return {"items": [3, 4]}

# Result: reducer([1, 2], [3, 4]) = [1, 2, 3, 4]
```

### Example 3: DeltaChannel with Messages

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[Message], add_messages]

# Token-by-token LLM streaming in node 1
return {"messages": [AIMessage(content="Hello")]}

# Same superstep, another node streams more
return {"messages": [AIMessage(content=" world")]}

# Reducer applies add_messages: messages are merged with ID deduplication
# Result: [HumanMessage(...), AIMessage(content="Hello world")]
```

### Example 4: DeltaChannel vs. LastValue

**DeltaChannel semantics:** Good for append-only data where all writes should be preserved.

```python
# Superstep 1: messages = [msg1]
# Superstep 2: add msg2 → messages = [msg1, msg2]
# Superstep 3: add msg3 → messages = [msg1, msg2, msg3]
```

**LastValue semantics:** Overwrites; only the final value matters.

```python
# Superstep 1: status = "processing"
# Superstep 2: status = "done"
# Final state: status = "done" (previous value discarded)
```

---

## Version Tracking and Triggering

Each channel has a **version** (integer, string, or float) that increments each superstep if the channel is updated.

### Checkpoint Versions

The checkpoint stores:

```python
{
    "channel_values": {
        "count": 5,
        "messages": [...]
    },
    "channel_versions": {
        "count": 3,          # Version after superstep 3
        "messages": 3
    },
    "versions_seen": {
        "node_a": {"count": 2, "messages": 2},  # node_a last read versions 2
        "node_b": {"count": 3}
    },
    "updated_channels": ["count"]  # Which channels changed in this step
}
```

### Triggering Rules

A node is scheduled to run in superstep N+1 only if:

1. **Explicitly triggered:** Via `add_edge()`, `add_conditional_edges()`, or `Send()`.
2. **Input-triggered:** At least one of its input channels has a newer version than recorded in `versions_seen[node_name]`.

This prevents infinite loops and wasted computation: once no channel updates, execution halts.

---

## Reducer Batching and Replay

DeltaChannel writes are replayed during checkpoint recovery. To optimize replay, DeltaChannel supports **snapshots**:

```python
class State(TypedDict):
    messages: Annotated[list[Message], DeltaChannel(add_messages, snapshot_frequency=500)]
```

- **snapshot_frequency:** Number of updates before writing a full snapshot blob (default 1000).
- **Replay:** On recovery, if a snapshot exists, state is restored from the snapshot; only newer writes are replayed.
- **Bounded Depth:** Even for inactive channels, snapshots are taken every 5000 supersteps system-wide.

This bounds the cost of checkpoint deserialization and replay.

---

## Node Input and Output Mapping

Nodes can have a custom **input_schema** different from the graph state schema:

```python
class State(TypedDict):
    full_context: dict
    counter: int

class NodeInput(TypedDict):
    counter: int  # Subset of state fields

def my_node(input: NodeInput) -> dict:
    # Only receives {"counter": ...}, not full_context
    return {"counter": input["counter"] + 1}

graph.add_node("step", my_node, input_schema=NodeInput)
```

**Mapping Logic:**
- At compile time, a **mapper** is created that extracts matching field names from state and passes only those to the node.
- The node returns a partial state dict; all keys are merged back into state.
- This allows nodes to ignore irrelevant state fields and simplifies function signatures.

---

## Checkpoint and Serialization

### Checkpoint Format

Channels are responsible for serialization via `checkpoint()` and `from_checkpoint()`:

```python
# DeltaChannel
checkpoint_value = channel.checkpoint()
# Returns: _DeltaSnapshot(value=...) or plain value for backwards compat

# LastValue
checkpoint_value = channel.checkpoint()
# Returns: the value itself
```

### Recovery

When resuming from a checkpoint:

```python
checkpoint = load_checkpoint(checkpoint_id)

for channel_name, channel in graph_channels.items():
    channel = channel.from_checkpoint(checkpoint["channel_values"][channel_name])
    # Channel now contains state from the checkpoint
```

For DeltaChannel, if the checkpoint is a sentinel, the checkpoint loader (repo://libs/checkpoint) replays ancestor writes through the reducer.

---

## Examples: Complete State Definition

### Example 1: Simple Chat

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
```

- `messages`: Append-only via `add_messages`.
- `user_id`: LastValue (doesn't change per superstep).

### Example 2: Agentic Loop with Intermediate State

```python
from typing import Annotated
from langgraph.channels import DeltaChannel, EphemeralValue
from langgraph.graph.message import add_messages
import operator

class State(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    intermediate_steps: Annotated[list, DeltaChannel(operator.add)]  # Append-only
    tool_input: Annotated[dict, EphemeralValue]  # Available only this superstep
    is_done: bool

class Output(TypedDict):
    messages: list
    is_done: bool
```

- `query`: LastValue (set once).
- `messages`: Append-only conversation history.
- `intermediate_steps`: Accumulates tool calls and results via DeltaChannel.
- `tool_input`: Ephemeral; passed via `Send(...)` to tool node.
- `is_done`: LastValue; indicates completion.
- Output hides `query`, `intermediate_steps`, and `tool_input` from users.

### Example 3: Map-Reduce with Barrier

```python
from typing import Annotated
from langgraph.channels import NamedBarrierValue
import operator

class State(TypedDict):
    items: list[str]
    results: Annotated[list, operator.add]
    all_done: Annotated[None, NamedBarrierValue({"map", "reduce"})]
```

- `items`: List of items to process.
- `results`: Accumulates results from parallel workers via `operator.add`.
- `all_done`: Barrier that signals when both map and reduce phases are complete.

---

## Common Patterns and Anti-Patterns

### Pattern: Incremental Accumulation

Use DeltaChannel for metrics, logs, or growing lists:

```python
class State(TypedDict):
    logs: Annotated[list[str], DeltaChannel(operator.add)]

def log_node(state: State) -> dict:
    return {"logs": [f"Step: {len(state['logs'])"]}
```

Each node's logs are accumulated; no logs are lost.

### Pattern: State Filtering

Use `output_schema` to hide sensitive or implementation data:

```python
class State(TypedDict):
    api_key: str
    internal_cache: dict
    public_result: str

class Output(TypedDict):
    public_result: str  # Only field returned to user
```

### Anti-Pattern: Mixing Semantics

Don't use LastValue for fields that will be written by multiple nodes in the same superstep. Instead, annotate with a reducer:

```python
# ❌ BAD
class State(TypedDict):
    all_results: list  # Multiple nodes write to this

# ✅ GOOD
class State(TypedDict):
    all_results: Annotated[list, operator.add]  # Merge via reducer
```

### Anti-Pattern: Overusing EphemeralValue

EphemeralValue is for temporary data passed via `Send(...)`. Don't use it for state that should persist:

```python
# ❌ BAD
class State(TypedDict):
    messages: Annotated[list, EphemeralValue]  # Would be cleared each step!

# ✅ GOOD
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Persistent, append-only
```

---

## Reference and Implementation Details

### File Structure

- **State schema handling:** repo://libs/langgraph/langgraph/graph/state.py
- **Channel base and implementations:** repo://libs/langgraph/langgraph/channels/
  - `base.py`: BaseChannel abstract interface
  - `last_value.py`: LastValue and LastValueAfterFinish
  - `delta.py`: DeltaChannel with replay and snapshots
  - `binop.py`: BinaryOperatorAggregate
  - `ephemeral_value.py`: EphemeralValue
  - `topic.py`: Topic (queue)
  - `named_barrier_value.py`: NamedBarrierValue synchronization
- **Message reducer:** repo://libs/langgraph/langgraph/graph/message.py (add_messages)
- **Managed values:** repo://libs/langgraph/langgraph/managed/base.py
- **Field utilities:** repo://libs/langgraph/langgraph/_internal/_fields.py

### Key Abstractions

- **BaseChannel:** Generic interface for all channel implementations. Every channel must implement `get()`, `update()`, `checkpoint()`, and `from_checkpoint()`.
- **StateGraph._add_schema():** Introspects a schema (TypedDict, dataclass, Pydantic) and creates channels for each field based on type hints and annotations.
- **Reducer Functions:** User-provided callables that merge updates; must be batching-invariant.

---

## Summary: State and Channels in Context

State is **the single source of truth** for graph execution:

1. **Definition:** Typed schema (TypedDict, dataclass, Pydantic).
2. **Channels:** Each field is backed by a channel implementing update semantics.
3. **Reducers:** Merge concurrent writes via user-specified or inferred functions.
4. **Versions:** Track channel updates to trigger dependent nodes.
5. **Checkpoints:** Serialize state for resumption and time-travel debugging.
6. **Schemas:** Input, state, and output schemas allow flexible data shapes.
7. **Managed Values:** Runtime-injected deterministic data.

Together, these enable **declarative, composable, resumable graph programs** where state flows through channels and nodes process incremental updates.
