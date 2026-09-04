---
type: Workflows & Patterns
title: Execution and Streaming
description: How to invoke graphs, stream results in different modes, handle interrupts, consume output, and use callbacks for observability and control.
tags: [streaming, invoke, stream, interrupts, callbacks, checkpoint, human-in-the-loop, v3-streaming]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-6ba6113c604a5fc7a18ea2e2
    resource: repo://libs/langgraph/langgraph/callbacks.py
  - id: openwiki-source-6d1627d1aa023712eac096a3
    resource: repo://libs/langgraph/langgraph/config.py
  - id: openwiki-source-4ae17bf912e4007bb8b83bef
    resource: repo://libs/langgraph/langgraph/pregel/main.py
  - id: openwiki-source-3eb3bed875edf55c9ce25cee
    resource: repo://libs/langgraph/langgraph/stream/run_stream.py
  - id: openwiki-source-32221bce0f3eead36a7c7e36
    resource: repo://libs/langgraph/langgraph/types.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

LangGraph provides multiple execution modes to run graphs and consume their output:

- **`invoke()`** / **`ainvoke()`**: Single synchronous/asynchronous invocation returning final output or collected stream parts.
- **`stream()`** / **`astream()`**: Pull-based streaming in different modes (`values`, `updates`, `messages`, `custom`, `tasks`, `debug`, `checkpoints`); returns/yields dictionaries or tuples depending on version.
- **`stream_events(version="v3")`** / **`astream_events(version="v3")`**: Experimental caller-driven streaming with typed projections, no background thread.
- **Callbacks** via `RunnableConfig`: Observability hooks for chain lifecycle, graph-specific interrupt/resume, and custom handlers.
- **Human-in-the-loop interrupts**: Pause execution with `interrupt()`, inspect state, optionally modify, and resume with `Command`.

This page covers the execution API, streaming modes, interrupt/resume mechanics, configuration, and practical patterns.

---

## Invocation: `invoke()` and `ainvoke()`

`invoke()` and `ainvoke()` run the graph to completion and return output. They are synchronous and asynchronous variants of the same pattern.

### Basic Invocation

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    value: str

def node_a(state: State) -> dict:
    return {"value": state["value"] + "_a"}

builder = StateGraph(State, input_schema=State)
builder.add_node("node_a", node_a)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", END)
graph = builder.compile()

# Sync invocation
output = graph.invoke({"value": "x"})
# output: {'value': 'x_a'}
```

### Configuration and State Snapshots

Invocation config includes:

- **`thread_id`**: Identifies the persisted state thread (required if checkpointer is set).
- **`checkpoint_id`**: (Optional) Resume from a specific checkpoint, enabling time-travel.
- **`recursion_limit`**: Maximum supersteps (default 25); raises `GraphRecursionError` if exceeded.
- **`timeout`**: Overall execution timeout.
- **`callbacks`**: Handlers for lifecycle, interrupt/resume, and chain events.

```python
config = {
    "configurable": {
        "thread_id": "thread-1",
        "checkpoint_id": "checkpoint-xyz",  # Resume from mid-execution
    }
}
output = graph.invoke({"value": "x"}, config)
```

To retrieve snapshots of the state at any point:

```python
# Get the current state
state = graph.get_state(config)
# state.values: dict of current channel values
# state.next: tuple of nodes to run next
# state.config: config used to fetch this snapshot
# state.interrupts: pending interrupts, if any

# Get history of all checkpoints for a thread
for snapshot in graph.get_state_history(config):
    print(snapshot.values)
```

### `invoke()` with Stream Modes

The `stream_mode` parameter collects stream output and returns it instead of just the final state:

```python
# Collect all values snapshots as a list
output = graph.invoke(
    {"value": "x"},
    stream_mode="values"  # default; returns final state
)
# output: {'value': 'x_a'}

# Collect node updates as a list (v1 behavior)
output = graph.invoke(
    {"value": "x"},
    stream_mode="updates",
    version="v1"
)
# output: [('node_a', {'value': 'x_a'}), ...]

# Return typed stream parts (v2 format)
output = graph.invoke(
    {"value": "x"},
    stream_mode="updates",
    version="v2"
)
# output: [
#     {"type": "updates", "ns": ("node_a",), "data": {"value": "x_a"}},
#     ...
# ]
```

### Async Invocation

```python
output = await graph.ainvoke({"value": "x"}, config)
```

---

## Streaming: `stream()` and `astream()`

`stream()` and `astream()` are pull-based iterators that yield output as the graph executes. Callers drive the pump—no background thread.

### Stream Modes

The `stream_mode` parameter controls what is emitted:

| Mode | Behavior |
|------|----------|
| `"values"` | Full state snapshots after each superstep. Default. |
| `"updates"` | Node-level state deltas (only channels written by each node). |
| `"messages"` | LLM message tokens and metadata during model invocations. |
| `"custom"` | User-emitted events via `StreamWriter` inside nodes. |
| `"tasks"` | Task lifecycle events (start, finish, error). |
| `"debug"` | Comprehensive debug events with execution context. |
| `"checkpoints"` | Checkpoint creation events. |

Multiple modes can be streamed simultaneously by passing a list:

```python
for chunk in graph.stream(
    {"value": "x"},
    stream_mode=["values", "updates"]
):
    # Yields tuples: (mode, data) or (namespace, mode, data) for subgraphs
    mode, data = chunk
    print(f"{mode}: {data}")
```

### Values Mode (Default)

Emits the complete state after each superstep:

```python
for chunk in graph.stream({"value": "x"}):
    print(chunk)  # {'value': 'x_a'} then final state
```

**v1 format** (default):
```python
for chunk in graph.stream({"value": "x"}, version="v1"):
    print(chunk)  # dict
```

**v2 format** (typed stream parts):
```python
for chunk in graph.stream({"value": "x"}, stream_mode="values", version="v2"):
    print(chunk)
    # {
    #     "type": "values",
    #     "ns": ("node_a",),  # namespace (empty tuple for main graph)
    #     "data": {"value": "x_a"},
    #     "interrupts": ()  # pending interrupts, if any
    # }
```

### Updates Mode

Emits only the channels modified by each node in each superstep:

```python
for chunk in graph.stream({"value": "x"}, stream_mode="updates"):
    print(chunk)
    # (v1) ('node_a', {'value': 'x_a'})
    # or (v2) {"type": "updates", "ns": ("node_a",), "data": {"value": "x_a"}}
```

### Messages Mode

Streams LLM token-by-token output from `BaseChatModel.invoke()` calls:

```python
from langchain_core.language_models.chat_model import BaseChatModel

# Inside a node:
def my_node(state):
    response = model.invoke(state["messages"])
    return {"messages": response}

# When streaming with messages mode:
for chunk in graph.stream(input, stream_mode="messages"):
    token, metadata = chunk  # (v1 format)
    print(token)  # "hello", " ", "world", ...
```

### Custom Mode

Emit arbitrary data from inside nodes using `StreamWriter`:

```python
from langgraph.config import get_stream_writer

def my_node(state):
    writer = get_stream_writer()
    writer({"step": "start"})
    # ... node logic ...
    writer({"step": "end"})
    return state

for chunk in graph.stream(input, stream_mode="custom"):
    print(chunk)  # {"step": "start"} then {"step": "end"}
```

### Multiple Stream Modes

```python
for chunk in graph.stream(
    {"value": "x"},
    stream_mode=["values", "updates"],
    version="v2"
):
    # v2: typed stream parts with "type", "ns", "data", optional "interrupts"
    # v1: tuples (ns_tuple, mode, data) or (mode, data)
    print(chunk)
```

### Subgraph Streaming

When a graph contains subgraphs (nested graph nodes), use `subgraphs=True` to receive events from inside them:

```python
for chunk in graph.stream(
    input,
    stream_mode="updates",
    subgraphs=True
):
    # v1: (namespace_tuple, mode, data) where namespace is e.g. ("parent_node:<task_id>", "child_node:<task_id>")
    # v2: namespace tuple in the stream part
    print(chunk)
```

### Debug Mode

Emit comprehensive debugging information:

```python
for chunk in graph.stream(input, stream_mode="debug"):
    print(chunk)
    # Full execution context: task info, writes, channel versions, etc.
```

---

## Stream Events v3: Caller-Driven Streaming (Experimental)

**`stream_events(version="v3")`** and **`astream_events(version="v3")`** provide an experimental, caller-driven streaming API with **no background thread** and **typed projections**.

### Key Differences from v1/v2

- **Caller pumps the run**: Iteration on any projection (`values`, `messages`, etc.) drives the graph forward.
- **Single-consumer projections**: Each projection can be iterated once; use `.tee(n)` for fan-out.
- **Typed projections**: `run.values`, `run.messages`, `run.lifecycle`, `run.subgraphs` are type-aware `StreamChannel` objects.
- **No background thread**: Simpler, more predictable memory and concurrency model.

### Basic Usage

```python
# Sync version
run = graph.stream_events({"value": "x"}, version="v3")

# Iterate values projection to drive the run forward
for state in run.values:
    print(state)  # {'value': 'x_a'}, ...

# Get final state and interrupt status
final_state = run.output
was_interrupted = run.interrupted
interrupts = run.interrupts
```

**Async version:**

```python
run = await graph.astream_events({"value": "x"}, version="v3")
async for state in run.values:
    print(state)

final = run.output
is_int = await run.interrupted
ints = await run.interrupts
```

### Projections

Built-in projections available on `GraphRunStream`:

- **`run.values`**: Full state snapshots after each superstep (always available).
- **`run.messages`**: LLM message tokens (if messages transformer registered).
- **`run.lifecycle`**: Interrupt and resume events (always available).
- **`run.subgraphs`**: Nested run streams from subgraph nodes.
- **`extensions[key]`**: Optional custom transformers registered at compile time.

### Interleaving Projections

Receive events from multiple projections in arrival order:

```python
for name, item in run.interleave("values", "messages"):
    if name == "values":
        print("state:", item)
    else:
        print("token:", item)
```

### Context Manager Pattern

```python
with graph.stream_events({"value": "x"}, version="v3") as run:
    for state in run.values:
        process(state)
    if run.interrupted:
        handle_interrupt()
```

### Configuring Transformers

Register custom transformers at compile time or supply them at call time:

```python
from langgraph.stream.transformers import StreamTransformer

class MyTransformer(StreamTransformer):
    required_stream_modes = ("custom",)
    
    def init(self):
        return {}
    
    def process(self, event):
        # Return True to keep the event, False to drop it
        return True

# At compile time
graph = builder.compile(stream_transformers=[MyTransformer()])

# Or at call time
run = graph.stream_events(
    input,
    version="v3",
    transformers=[MyTransformer()]
)
```

---

## Human-in-the-Loop: Interrupts and Resumption

LangGraph supports pausing graph execution to request human input or approval, then resuming with the user's response.

### Triggering an Interrupt

Inside a node, call `interrupt(value)` to pause execution:

```python
from langgraph.types import interrupt

def approval_node(state):
    # Pause and send a message to the client
    user_input = interrupt("Please approve this action")
    
    # After resume, user_input contains the value passed by Command
    print(f"User said: {user_input}")
    return {"status": "approved"}
```

**Requirements:**

- A checkpointer must be configured (interrupts rely on persistence).
- `interrupt()` raises a `GraphInterrupt` exception, halting the node.
- The execution pauses at the **current superstep**.
- The state is persisted; clients can inspect it via `get_state()`.

### Resuming from an Interrupt

Use `Command` to specify the value to resume with:

```python
from langgraph.types import Command

# Get the current state
state = graph.get_state(config)
if state.interrupts:
    interrupt_obj = state.interrupts[0]
    print(f"Interrupted with: {interrupt_obj.value}")

# Resume, providing a value for the interrupt
config_with_resume = {
    "configurable": {
        "thread_id": "thread-1",
    }
}
output = graph.invoke(Command(resume="user's approval"), config_with_resume)
```

### Multiple Interrupts in One Node

If a node contains multiple `interrupt()` calls, they match resume values by order:

```python
def my_node(state):
    age = interrupt("What is your age?")      # interrupt #1
    name = interrupt("What is your name?")    # interrupt #2
    return {"age": age, "name": name}

# Resume with a list of values (in order)
output = graph.invoke(Command(resume=[25, "Alice"]), config)
```

### Interrupt Metadata

Each `Interrupt` object carries metadata for client routing:

```python
class Interrupt:
    value: Any        # The interrupt payload
    id: str           # Unique ID for this interrupt
```

Retrieve interrupts from state snapshots:

```python
state = graph.get_state(config)
if state.interrupts:
    for interrupt_obj in state.interrupts:
        print(f"Interrupt {interrupt_obj.id}: {interrupt_obj.value}")
```

### Observing Interrupts and Resumes

Register a `GraphCallbackHandler` to observe interrupt and resume lifecycle events:

```python
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent

class MyGraphHandler(GraphCallbackHandler):
    def on_interrupt(self, event: GraphInterruptEvent):
        print(f"Interrupted: {event.interrupts}")
        print(f"Checkpoint: {event.checkpoint_id}")
    
    def on_resume(self, event: GraphResumeEvent):
        print(f"Resumed from: {event.checkpoint_id}")

config = {
    "callbacks": [MyGraphHandler()]
}
output = graph.invoke(input, config)
```

---

## Callbacks and Configuration

### RunnableConfig

`RunnableConfig` is a dictionary-like object that carries execution context and options:

```python
from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {
    "configurable": {
        "thread_id": "thread-1",
        "checkpoint_id": "checkpoint-xyz",  # resume from here
        "checkpoint_ns": "main",  # namespace for nested graphs
    },
    "run_id": UUID(...),  # Unique run ID
    "callbacks": [handler1, handler2],  # Callback handlers
    "tags": ["production"],
    "metadata": {"user_id": "123"},
}

output = graph.invoke(input, config)
```

### Callback Handlers

LangChain callbacks are passed via `config["callbacks"]`. Common handlers:

- **`BaseCallbackHandler`**: Generic chain lifecycle (on_chain_start, on_chain_end, on_llm_new_token, etc.).
- **`GraphCallbackHandler`**: LangGraph-specific lifecycle (on_interrupt, on_resume).

```python
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.callbacks import GraphCallbackHandler

class MyChainHandler(BaseCallbackHandler):
    def on_chain_start(self, run_id, inputs, **kwargs):
        print(f"Chain start: {inputs}")
    
    def on_chain_end(self, run_id, outputs, **kwargs):
        print(f"Chain end: {outputs}")

class MyGraphHandler(GraphCallbackHandler):
    def on_interrupt(self, event):
        print(f"Graph interrupted: {event.interrupts}")

config = {
    "callbacks": [MyChainHandler(), MyGraphHandler()]
}
output = graph.invoke(input, config)
```

### Durability Modes

Control when state changes are persisted via the `durability` parameter:

| Mode | Behavior |
|------|----------|
| `"async"` | (Default) Persist asynchronously while the next superstep executes. |
| `"sync"` | Persist synchronously before the next superstep starts. |
| `"exit"` | Persist only when the graph completes or interrupts. |

```python
graph.invoke(
    input,
    durability="sync"  # Synchronous persistence
)
```

---

## Practical Examples

### UI Streaming: Live State Updates

Stream values to a UI client as the graph executes:

```python
def stream_to_ui(graph, input_data, websocket):
    for chunk in graph.stream(
        input_data,
        stream_mode="values",
        version="v2"
    ):
        # Emit typed stream part to frontend
        websocket.send_json(chunk)
        # chunk: {"type": "values", "ns": (...), "data": {...}, "interrupts": ()}
```

### Token Streaming: Live LLM Output

Stream LLM tokens as they arrive:

```python
def stream_tokens(graph, input_data):
    for chunk in graph.stream(
        input_data,
        stream_mode="messages",
        version="v1"
    ):
        token, metadata = chunk
        yield token  # Stream to UI
```

### Human-in-the-Loop Workflow

```python
from langgraph.types import interrupt, Command

def review_node(state):
    # Pause and ask for approval
    approval = interrupt(
        value={"action": "publish", "content": state["draft"]},
        id="approval_required"
    )
    if approval:
        return {"status": "published"}
    else:
        return {"status": "rejected"}

# Client side:
state = graph.get_state(config)
if state.interrupts:
    # Show approval dialog
    user_approved = show_dialog(state.interrupts[0].value)
    # Resume
    graph.invoke(Command(resume=user_approved), config)
```

### Streaming with Interrupts (v3)

```python
run = graph.stream_events({"value": "x"}, version="v3")

for state in run.values:
    print(f"State: {state}")

if run.interrupted:
    interrupts = run.interrupts
    print(f"Interrupted: {interrupts}")
    # Handle interrupt—perhaps modify state or request user input
```

### Checkpoint History and Replay

```python
# Get all checkpoints for a thread
history = list(graph.get_state_history(config, limit=10))

for i, snapshot in enumerate(history):
    print(f"Step {i}: {snapshot.values}")

# Resume from a specific checkpoint
old_snapshot = history[3]
config["configurable"]["checkpoint_id"] = old_snapshot.config["configurable"]["checkpoint_id"]
output = graph.invoke(input, config)
```

### Combining Callbacks and Streaming

```python
from langgraph.callbacks import GraphCallbackHandler

class LoggingHandler(GraphCallbackHandler):
    def on_interrupt(self, event):
        print(f"[LOG] Interrupted: {event.interrupts}")

config = {"callbacks": [LoggingHandler()]}

for chunk in graph.stream(input, config=config):
    process(chunk)
```

---

## Key Invariants and Behaviors

### Execution Model

- Graphs execute in **supersteps** (atomic synchronization barriers).
- Each superstep runs all triggered nodes in parallel, then applies writes and increments channel versions.
- The next superstep is scheduled if any channel was updated and the graph has not reached END or exceeded recursion_limit.
- Streaming yields output **after each superstep completes**.

### Interrupt Semantics

- **On interrupt**: The current node's execution halts; state is persisted.
- **On resume**: The interrupted node re-executes from the beginning (deterministically, due to task IDs and checkpointing).
- **Multiple interrupts**: Matched to resume values by order in the node.
- **State isolation**: Each task has its own interrupt list scoped to its execution.

### Checkpoint and Time-Travel

- Checkpoints snapshot channel state at each superstep.
- Passing `checkpoint_id` in config resumes execution from that point.
- Prior supersteps are replayed deterministically (task IDs remain stable).
- `get_state_history()` retrieves all checkpoints for a thread.

### Stream Versions

- **v1** (default for `stream()`): Plain dicts and tuples; untyped.
- **v2** (typed stream parts): `{"type": ..., "ns": ..., "data": ..., "interrupts": ...}`.
- **v3** (experimental for `stream_events()`): Caller-driven, no background thread, typed projections.

---

## Configuration Reference

### Config Keys

Core config keys (use via `config["configurable"]`):

- `thread_id` (str): Thread identifier for checkpoint persistence.
- `checkpoint_id` (str, optional): Resume from this checkpoint.
- `checkpoint_ns` (str, optional): Namespace for nested graph checkpoints.
- `recursion_limit` (int, default 25): Max supersteps before error.
- `timeout` (float, optional): Execution timeout in seconds.
- `callbacks` (list): Callback handlers.
- `tags` (list): Run tags for metadata.
- `metadata` (dict): Custom metadata.

### RunnableConfig Structure

```python
config: RunnableConfig = {
    "configurable": {
        "thread_id": "...",
        "checkpoint_id": "...",
        # ... other configurable keys
    },
    "run_id": UUID(...),
    "callbacks": [...],
    "tags": [...],
    "metadata": {...},
}
```

---

## Related Concepts

- **Graph Execution Model**: Supersteps, task scheduling, channel versioning, and state advancement.
- **Checkpointing**: Persisting and replaying state via `BaseCheckpointSaver`.
- **Channels**: The storage and reducer mechanism for graph state.
- **Error Handling**: Error handlers, retry policies, and graceful degradation.
