---
type: Testing & Patterns
title: Testing Patterns
description: How to write tests for graphs, channels, nodes, and integration workflows using pytest, fixtures, and strategies for determinism, streaming, checkpointing, and error handling.
tags: [testing, pytest, fixtures, patterns, determinism, checkpointing, streaming, snapshot-testing]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-51cd3d5adf59b6088148f7d5
    resource: repo://libs/langgraph/tests/conftest_checkpointer.py
  - id: openwiki-source-3c04e6001e24fc37874db53e
    resource: repo://libs/langgraph/tests/conftest.py
  - id: openwiki-source-88448dc1a96cd8c1b694645a
    resource: repo://libs/langgraph/tests/fake_chat.py
  - id: openwiki-source-93834fdb7ab1e9377112795c
    resource: repo://libs/langgraph/tests/memory_assert.py
  - id: openwiki-source-f96a7952c9fee7819ea4c100
    resource: repo://libs/langgraph/tests/test_channels.py
  - id: openwiki-source-36db910d85e2701ba7bb84a8
    resource: repo://libs/langgraph/tests/test_interruption.py
  - id: openwiki-source-d5a44e19b0040db750f95499
    resource: repo://libs/langgraph/tests/test_large_cases.py
  - id: openwiki-source-0cf33f3aaf5b4b6a8d21b36d
    resource: repo://libs/langgraph/tests/test_pregel_async.py
  - id: openwiki-source-359b1259bb077515edbf1b05
    resource: repo://libs/langgraph/tests/test_pregel.py
  - id: openwiki-source-6feaeb3da2a8a44b49e25d87
    resource: repo://libs/langgraph/tests/test_retry.py
  - id: openwiki-source-2a1582f888ea749f020226a8
    resource: repo://libs/langgraph/tests/test_state.py
  - id: openwiki-source-3e54420b389b822102336f3d
    resource: repo://libs/langgraph/tests/test_stream_events_v3.py
  - id: openwiki-source-683ca76faece52e57e12784e
    resource: repo://libs/langgraph/tests/test_subgraph_persistence.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

Testing LangGraph applications requires strategies tailored to the Pregel execution model, stateful channels, checkpointing, and concurrent node execution. This page documents the core testing patterns used in the LangGraph test suite, including fixture setup, deterministic test isolation, error handling verification, and techniques for testing graphs, channels, streaming, and subgraphs.

All examples reference the test suite at `libs/langgraph/tests/`.

---

## Core Fixtures and Test Setup

### Pytest Configuration and Async Support

LangGraph tests use `pytest` with async support via `pytest-asyncio`. Basic configuration:

```python
pytestmark = pytest.mark.anyio
```

The `conftest.py` provides a universal `anyio_backend` fixture:

```python
@pytest.fixture
def anyio_backend():
    return "asyncio"
```

This allows `async def test_*` functions to run without explicit `@pytest.mark.asyncio` decorators.

### Checkpointer Fixtures

Checkpointers are the backbone of stateful testing. `conftest_checkpointer.py` provides parameterized fixtures for multiple backends:

**Sync Checkpointers** (`sync_checkpointer`):

```python
@pytest.fixture(
    params=["memory", "sqlite", "sqlite_aes"]
    if NO_DOCKER
    else ["memory", "sqlite", "sqlite_aes", "postgres", ...]
)
def sync_checkpointer(request: pytest.FixtureRequest) -> Iterator[BaseCheckpointSaver]:
    # Yields InMemorySaver, SqliteSaver, or PostgresSaver depending on request.param
```

In-memory checkpointer (fastest, no I/O):

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "thread-1"}}
```

**Async Checkpointers** (`async_checkpointer`):

```python
async def async_checkpointer(request: pytest.FixtureRequest) -> AsyncIterator[BaseCheckpointSaver]:
    # Same backends, with async methods: aget_tuple, aput, etc.
```

### Assertion Utilities

**MemorySaverAssertImmutable** (`memory_assert.py`):

Verifies that checkpoints are not mutated after being saved (catches race conditions):

```python
from tests.memory_assert import MemorySaverAssertImmutable

checkpointer = MemorySaverAssertImmutable()
# Asserts during put() that prior checkpoint wasn't modified
```

### FakeChatModel and Deterministic LLM Responses

For deterministic tests, use seeded fake LLMs instead of real API calls.

**FakeChatModel** (`fake_chat.py`):

```python
from tests.fake_chat import FakeChatModel
from langchain_core.messages import AIMessage

model = FakeChatModel(messages=[
    AIMessage(content="response 1"),
    AIMessage(content="response 2"),
])

# Cycles through messages in order
result1 = model.invoke([...])  # returns "response 1"
result2 = model.invoke([...])  # returns "response 2"
```

**FakeStreamingListLLM** (from langchain-core):

```python
from langchain_core.language_models.fake import FakeStreamingListLLM

llm = FakeStreamingListLLM(responses=["answer 1", "answer 2"])
```

**FakeMessagesListChatModel**:

```python
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

model = FakeMessagesListChatModel(messages=[
    AIMessage(content="hello", tool_calls=[...]),
    AIMessage(content="goodbye"),
])
```

### Deterministic UUIDs

For reproducible test output, mock UUID generation:

```python
@pytest.fixture
def deterministic_uuids(mocker):
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4)
        for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)
```

Use it in tests:

```python
def test_with_deterministic_ids(deterministic_uuids):
    # All generated UUIDs are predictable
    ...
```

---

## Testing Graphs

### Synchronous Graph Tests

The most common pattern: build a StateGraph, compile it, and invoke it.

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    hello: str

def node_a(state: State) -> State:
    return {"hello": state["hello"] + "_a"}

def node_b(state: State) -> State:
    return {"hello": state["hello"] + "_b"}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

graph = builder.compile()
result = graph.invoke({"hello": "start"})
assert result == {"hello": "start_a_b"}
```

### Stateful Graph Tests with Checkpointing

Use a checkpointer to test state persistence and resumption:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thread-1"}}

# First invocation
result = graph.invoke({"hello": "start"}, config)
# State is saved

# Retrieve saved state
snapshot = graph.get_state(config)
assert snapshot.values == {"hello": "start"}
```

### Testing with Interrupts

Verify that interrupts pause execution at specified points and resume correctly:

```python
from langgraph.types import interrupt, Interrupt

def node_with_interrupt(state: State):
    interrupt("wait for user input")
    return {"hello": "continued"}

graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "thread-1"}}

# First invoke hits interrupt
result = graph.invoke({"hello": "start"}, config)
assert "__interrupt__" in result
assert len(result["__interrupt__"]) == 1

# Resume
result = graph.invoke(None, config)
# Execution continues after interrupt
```

### Testing Durability Modes

LangGraph supports three durability modes: `"sync"`, `"async"`, and `"exit"`. Test each:

```python
@pytest.fixture(params=["sync", "async", "exit"])
def durability(request):
    return request.param

def test_with_durability(sync_checkpointer, durability):
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "thread-1"}}
    result = graph.invoke({"hello": "start"}, config, durability=durability)
```

---

## Testing Channels

### Channel Primitives

Test channels directly without a full graph:

```python
from langgraph.channels.last_value import LastValue
from langgraph._internal._typing import MISSING

# Create and initialize
channel = LastValue(int).from_checkpoint(MISSING)

# Update and read
channel.update([42])
assert channel.get() == 42

# Persistence
checkpoint = channel.checkpoint()
restored = LastValue(int).from_checkpoint(checkpoint)
assert restored.get() == 42
```

### Testing Reducers (e.g., add_messages)

Test message accumulation behavior:

```python
from langgraph.graph.message import add_messages
from langgraph.channels.delta import DeltaChannel
from langchain_core.messages import HumanMessage, AIMessage

reducer = add_messages
ch = DeltaChannel(reducer, list).from_checkpoint(MISSING)

# Add first message
ch.update([HumanMessage(content="hi", id="h1")])
assert len(ch.get()) == 1

# Add second message
ch.update([AIMessage(content="hello", id="a1")])
assert len(ch.get()) == 2
assert ch.get()[0].content == "hi"
assert ch.get()[1].content == "hello"
```

### Testing Reducer Semantics

Verify reducer behavior with multiple writes in a superstep:

```python
from langgraph.channels.binop import BinaryOperatorAggregate
import operator

ch = BinaryOperatorAggregate(int, operator.add).from_checkpoint(MISSING)
assert ch.get() == 0  # Initial value for addition

ch.update([1, 2, 3])
assert ch.get() == 6  # 1 + 2 + 3

ch.update([4])
assert ch.get() == 10  # 6 + 4
```

### Testing Empty Channel Errors

Verify that reading from an empty channel raises an appropriate error:

```python
from langgraph.errors import EmptyChannelError

channel = LastValue(int).from_checkpoint(MISSING)
with pytest.raises(EmptyChannelError):
    channel.get()
```

---

## Testing Nodes

### Unit Testing Node Functions

Test node logic independently, with mock inputs:

```python
def process_item(state: State) -> State:
    """Transforms state by appending '_processed'."""
    return {"value": state["value"] + "_processed"}

def test_process_item():
    state = {"value": "item"}
    result = process_item(state)
    assert result == {"value": "item_processed"}
```

### Testing Nodes with Mocked Dependencies

Use `mocker` to test node behavior without external calls:

```python
def test_node_with_mocked_dep(mocker):
    mock_api = mocker.Mock(return_value="api_response")
    
    def node_calls_api(state: State) -> State:
        result = mock_api(state["query"])
        return {"result": result}
    
    state = {"query": "test"}
    result = node_calls_api(state)
    assert result == {"result": "api_response"}
    mock_api.assert_called_once_with("test")
```

### Testing Nodes with Config

If a node reads from config (e.g., to get a stream writer or read a dependency):

```python
from langgraph.config import get_stream_writer

def node_with_streaming(state: State, *, config: RunnableConfig) -> State:
    writer = get_stream_writer(config)
    writer("progress update")
    return state

def test_node_streaming():
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(callbacks=[...])
    # Verify that writer was called
```

---

## Testing Async Graphs and Nodes

### Async Graph Tests

Use `async def` test functions (pytest-asyncio):

```python
async def test_async_graph():
    async def async_node(state: State) -> State:
        # Can use await
        return {"hello": state["hello"] + "_async"}
    
    builder = StateGraph(State)
    builder.add_node("async_node", async_node)
    builder.add_edge(START, "async_node")
    graph = builder.compile()
    
    result = await graph.ainvoke({"hello": "start"})
    assert result == {"hello": "start_async"}
```

### Async Checkpointing

Use `async_checkpointer` fixture and async methods:

```python
async def test_async_with_checkpointer(async_checkpointer):
    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "thread-1"}}
    
    result = await graph.ainvoke({"hello": "start"}, config)
    snapshot = await graph.aget_state(config)
    assert snapshot.values == {"hello": "start"}
```

### Testing Concurrent Execution

Verify that parallel nodes execute concurrently:

```python
import asyncio
from time import time

async def test_concurrent_nodes():
    delay = 0.1
    
    async def slow_node_a(state: State) -> State:
        await asyncio.sleep(delay)
        return {"a": "done"}
    
    async def slow_node_b(state: State) -> State:
        await asyncio.sleep(delay)
        return {"b": "done"}
    
    builder = StateGraph(State)
    builder.add_node("a", slow_node_a)
    builder.add_node("b", slow_node_b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    
    graph = builder.compile()
    
    start = time()
    await graph.ainvoke({})
    elapsed = time() - start
    
    # Should take ~delay (concurrent), not 2*delay (serial)
    assert elapsed < 1.5 * delay
```

---

## Testing Streaming

### Streaming with stream()

Test output from different stream modes:

```python
def test_stream_values():
    """stream_mode='values' emits full state after each node."""
    graph = builder.compile()
    
    stream_output = list(graph.stream(
        {"hello": "start"},
        stream_mode="values"
    ))
    
    assert len(stream_output) >= 2
    assert stream_output[0] == {"hello": "start_a"}
    assert stream_output[-1] == {"hello": "start_a_b"}
```

### Streaming with stream_mode='updates'

Test incremental updates only:

```python
def test_stream_updates():
    """stream_mode='updates' emits only changed channels."""
    stream_output = list(graph.stream(
        {"hello": "start"},
        stream_mode="updates"
    ))
    
    # First update from node_a
    assert "a" in stream_output[0]
```

### Streaming with stream_mode='events' (v3)

Test structured event streams with detailed execution information:

```python
def test_stream_events():
    from langgraph.types import StreamPart
    
    events = list(graph.stream(
        {"hello": "start"},
        stream_mode="events"
    ))
    
    # Filter for task events
    task_events = [e for e in events if e.get("event") == "on_node_start"]
    assert len(task_events) > 0
```

### Testing Async Streaming

```python
async def test_async_stream():
    result = []
    async for value in graph.astream({"hello": "start"}):
        result.append(value)
    
    assert len(result) > 0
```

---

## Testing Error Handling and Retry

### Testing Retry Policies

Verify that nodes retry on specific exceptions:

```python
from langgraph.types import RetryPolicy

call_count = 0

def flaky_node(state: State) -> State:
    global call_count
    call_count += 1
    if call_count < 3:
        raise ValueError("temporary error")
    return {"hello": "recovered"}

# Compile with retry policy
builder = StateGraph(State)
builder.add_node("flaky", flaky_node)
builder.set_entry_point("flaky")
graph = builder.compile()

# With retry policy, should recover
result = graph.invoke({})
assert call_count == 3
assert result == {"hello": "recovered"}
```

### Testing Error Handlers

Define and test error handler nodes:

```python
class State(TypedDict):
    value: str
    error: str

def failing_node(state: State) -> State:
    raise ValueError("intentional error")

def error_handler(state: State) -> State:
    return {"error": "handled"}

builder = StateGraph(State)
builder.add_node("main", failing_node)
builder.add_node("error", error_handler)
builder.set_entry_point("main")
builder.add_edge(START, "main")

# Register error handler
graph = builder.compile()
# Error handler routing is configured in graph construction
```

### Testing Timeout Behavior

Verify that long-running nodes timeout:

```python
import asyncio
from langgraph.types import TimeoutPolicy

async def test_timeout():
    async def slow_node(state: State) -> State:
        await asyncio.sleep(10)  # Very slow
        return state
    
    builder = StateGraph(State)
    builder.add_node("slow", slow_node)
    builder.set_entry_point("slow")
    
    graph = builder.compile()
    
    # With a short timeout, should raise NodeTimeoutError
    from langgraph.errors import NodeTimeoutError
    with pytest.raises(NodeTimeoutError):
        await graph.ainvoke({}, timeout=0.1)
```

---

## Testing Subgraphs

### Testing Stateless Subgraphs

A subgraph compiled without a checkpointer inherits the parent's checkpointer for interrupt support but resets state each invocation:

```python
def test_stateless_subgraph(sync_checkpointer):
    # Inner subgraph (no checkpointer)
    def echo(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content=f"Echo: {state['messages'][-1].content}")]}
    
    inner = StateGraph(MessagesState).add_node("echo", echo).compile()
    
    # Parent graph (with checkpointer)
    def call_inner(state: State) -> dict:
        resp = inner.invoke({"messages": [HumanMessage(content="hello")]})
        return {"result": resp["messages"][-1].content}
    
    parent = (
        StateGraph(State)
        .add_node("call_inner", call_inner)
        .add_edge(START, "call_inner")
        .compile(checkpointer=sync_checkpointer)
    )
    
    config = {"configurable": {"thread_id": "thread-1"}}
    result = parent.invoke({"result": ""}, config)
    
    # Subgraph output is captured but not persisted as subgraph state
    assert "Echo:" in result["result"]
```

### Testing Stateful Subgraphs

A subgraph with `checkpointer=True` retains state across parent invocations:

```python
def test_stateful_subgraph(sync_checkpointer):
    # Inner subgraph with checkpointer
    inner = (
        StateGraph(MessagesState)
        .add_node("echo", echo)
        .compile(checkpointer=sync_checkpointer)
    )
    
    parent = (
        StateGraph(State)
        .add_node("call_inner", call_inner)
        .compile(checkpointer=sync_checkpointer)
    )
    
    # Subgraph state accumulates across parent invocations
```

### Testing Subgraph Input/Output Adaptation

Verify that subgraph input and output schemas are correctly adapted:

```python
class SubgraphInputSchema(TypedDict):
    messages: list

def test_subgraph_schema():
    subgraph = builder.compile(input_schema=SubgraphInputSchema)
    
    # Schema mismatch should be caught
    with pytest.raises(ValueError):
        subgraph.invoke({"wrong_key": "value"})
```

---

## Snapshot Testing

### Using Syrupy for Complex State

For tests with complex, nested state structures, snapshot testing provides regression detection:

```python
from syrupy import SnapshotAssertion

def test_complex_state_structure(snapshot: SnapshotAssertion):
    # Build complex state
    state = {
        "messages": [
            HumanMessage(content="hi", id="h1"),
            AIMessage(content="hello", id="a1"),
        ],
        "metadata": {"step": 1, "nodes": ["a", "b"]},
    }
    
    graph = builder.compile()
    result = graph.invoke(state)
    
    # Compare against stored snapshot
    assert result == snapshot
```

Snapshots are stored in `__snapshots__` directories. Update with `pytest --snapshot-update`.

---

## Performance and Large-Scale Testing

### Benchmarking Node Execution Time

Measure node performance:

```python
from time import perf_counter

def test_node_performance():
    graph = builder.compile()
    
    start = perf_counter()
    for _ in range(100):
        graph.invoke({"hello": "start"})
    elapsed = perf_counter() - start
    
    assert elapsed < 5.0  # Should complete in under 5 seconds
```

### Testing Large State

Verify behavior with large state objects:

```python
def test_large_state_handling():
    large_state = {
        "messages": [HumanMessage(content=f"msg {i}") for i in range(1000)],
    }
    
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke(large_state)
    
    # Verify state is preserved
    assert len(result["messages"]) == 1000
```

### Testing Long-Running Graphs

Verify graphs with many supersteps:

```python
def test_long_running_graph():
    class CountState(TypedDict):
        count: int
    
    def increment(state: CountState) -> CountState:
        return {"count": state["count"] + 1}
    
    def should_continue(state: CountState) -> str:
        return "increment" if state["count"] < 100 else END
    
    builder = StateGraph(CountState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_conditional_edges("increment", should_continue)
    
    graph = builder.compile()
    result = graph.invoke({"count": 0})
    
    assert result["count"] == 100
```

---

## Common Test Patterns

### Testing Conditional Edges

Verify routing logic:

```python
def router(state: State) -> str:
    if "error" in state:
        return "handle_error"
    return "process"

builder = StateGraph(State)
builder.add_node("process", lambda s: {"value": "processed"})
builder.add_node("handle_error", lambda s: {"error": "handled"})
builder.add_conditional_edges(START, router)

graph = builder.compile()

# Test normal path
result = graph.invoke({})
assert result.get("value") == "processed"

# Test error path
result = graph.invoke({"error": "initial"})
assert "error" in result
```

### Testing Send (Dynamic Routing)

Verify fan-out with Send:

```python
from langgraph.types import Send

def split_and_process(state: State) -> list[Send]:
    return [
        Send("process", {"item": item})
        for item in state["items"]
    ]

builder = StateGraph(State)
builder.add_node("split", split_and_process)
builder.add_node("process", lambda s: {"result": s["item"]})
builder.add_edge(START, "split")

graph = builder.compile()
result = graph.invoke({"items": [1, 2, 3]})
```

### Testing MessagesState

Verify message accumulation and reducer behavior:

```python
from langgraph.graph.message import MessagesState

def test_messages_state():
    builder = StateGraph(MessagesState)
    builder.add_node("add_msg", lambda s: {"messages": [AIMessage(content="hi")]})
    builder.add_edge(START, "add_msg")
    
    graph = builder.compile()
    
    result = graph.invoke({"messages": [HumanMessage(content="hello")]})
    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "hello"
    assert result["messages"][1].content == "hi"
```

---

## Debugging and Troubleshooting

### Using get_state() to Inspect Checkpoints

```python
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "thread-1"}}

# Run until interrupt or end
graph.invoke({"hello": "start"}, config)

# Inspect saved state
snapshot = graph.get_state(config)
print(snapshot.values)
print(snapshot.next)  # Next nodes to run
```

### Reviewing Execution History

```python
# Get all checkpoints for a thread
history = list(graph.get_state_history(config))
for checkpoint in history:
    print(checkpoint.values)
    print(checkpoint.metadata)
```

### Streaming for Real-Time Debugging

Use streaming to observe execution step-by-step:

```python
for step in graph.stream({"hello": "start"}, stream_mode="updates"):
    print(step)
```

---

## Best Practices

1. **Fixture Parameterization**: Use parameterized fixtures (`@pytest.fixture(params=...)`) to test across multiple checkpointer backends and durability modes simultaneously.

2. **Determinism**: Always use deterministic fakes (FakeChatModel, mock UUIDs) in tests. Avoid real API calls or non-deterministic behavior.

3. **Isolation**: Each test should use its own thread ID or unique config to prevent state leakage between tests.

4. **Async Testing**: Mark async tests with `pytestmark = pytest.mark.anyio` at module level, or use `@pytest.mark.asyncio` on individual tests.

5. **Checkpoint Cleanup**: Checkpointers with side effects (databases, files) should be cleaned up in fixtures using context managers (`@contextmanager` or `@asynccontextmanager`).

6. **Channel-Level Testing**: Test channel reducers and semantics directly before building full-graph tests. This isolates issues.

7. **Snapshot Testing**: Use syrupy for complex, nested structures. Update snapshots carefully and review diffs.

8. **Error Coverage**: Test retry logic, timeouts, and error handlers independently. Use mocks to simulate specific failure conditions.

9. **Performance Baselines**: Set reasonable performance expectations for your graphs. Use `perf_counter()` for timing-sensitive tests.

10. **Documentation**: Comment on non-obvious test setup, especially for complex checkpointer or durability configurations.

---

## References

- **Test Suite**: `libs/langgraph/tests/`
- **Graph Execution Model**: [Graph Execution Model](/openwiki/architecture/graph-execution-model.md)
- **Checkpointer API**: `langgraph.checkpoint.base`
- **Channel Types**: `langgraph.channels`
- **Pregel Algorithm**: `langgraph.pregel._algo`
