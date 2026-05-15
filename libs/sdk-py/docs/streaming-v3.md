# Thread-Centric Streaming (v3)

`client.threads.stream(thread_id, assistant_id)` returns an `AsyncThreadStream` (or
`SyncThreadStream`) context manager that owns the SSE connection for one thread. All
typed projections — values, messages, tool calls, custom events — share a single
underlying SSE session opened with the union of the channels they need.

## Basic Usage

```python
from langgraph_sdk import get_client

client = get_client()

async with client.threads.stream(
    thread_id="my-thread",
    assistant_id="agent",
) as thread:
    await thread.run.start(input={"messages": [{"role": "user", "content": "hi"}]})

    async for snapshot in thread.values:
        print("state snapshot:", snapshot)

    final = await thread.output   # terminal state values
```

Sync equivalent:

```python
from langgraph_sdk import get_sync_client

client = get_sync_client()

with client.threads.stream(thread_id="my-thread", assistant_id="agent") as thread:
    thread.run.start(input={"messages": [{"role": "user", "content": "hi"}]})
    final = thread.output
```

## Typed Projections

### Values (`thread.values`)

Iterates state snapshots: the current REST state is yielded first, then live values
events from the SSE.

```python
async for snapshot in thread.values:
    print(snapshot)   # dict of graph state

# Or await for the terminal value (same as thread.output):
final = await thread.values
```

### Messages (`thread.messages`)

Each message-start event yields one `AsyncChatModelStream` (from `langchain_core`).
The stream receives token deltas and is fully usable after the `async for` loop
completes.

```python
# Collect all streams; each is fully dispatched once the loop finishes.
streams = [s async for s in thread.messages]
for stream in streams:
    print(await stream.text)       # "Hello, world!"
    msg = await stream.output      # AIMessage
```

### Tool Calls (`thread.tool_calls`)

```python
async for call in thread.tool_calls:
    print(call.name, call.call_id, call.status)
    if call.output is not None:
        print(call.output)
```

### Custom Events (`thread.extensions["name"]`)

Custom events emitted by your graph under a specific name are surfaced as plain dicts.

```python
async for payload in thread.extensions["progress"]:
    print(payload)   # {"name": "progress", "step": 1, ...}
```

### Terminal Output (`thread.output`)

Awaits lifecycle completion, then fetches durable thread state.

```python
final_values = await thread.output
```

## Concurrent Consumers

All consumers that need different channels should start concurrently so they share
one SSE connection. Sequential iteration triggers SSE rotations that the client-side
dedup set defends against, but the recommended pattern is `asyncio.gather`:

```python
import asyncio

async with client.threads.stream(thread_id="t", assistant_id="agent") as thread:
    await thread.run.start(input={...})

    async def get_messages():
        return [s async for s in thread.messages]

    async def get_tool_calls():
        return [c async for c in thread.tool_calls]

    async def get_progress():
        return [p async for p in thread.extensions["progress"]]

    messages, tool_calls, progress = await asyncio.gather(
        get_messages(), get_tool_calls(), get_progress()
    )
    for s in messages:
        print(await s.text)
```

## Human-in-the-Loop

When the server requests human input, `thread.interrupted` becomes `True` and
`thread.interrupts` contains the pending payloads.

```python
async with client.threads.stream(thread_id="t", assistant_id="agent") as thread:
    await thread.run.start(input={...})
    await thread._wait_for_run_done()   # blocks until interrupt or completion

    if thread.interrupted:
        print(thread.interrupts[0]["value"])
        await thread.run.respond("my answer")
        await thread._wait_for_run_done()   # wait for resumed run
```

## Agent Graph Introspection

`thread.agent.get_tree()` fetches the assistant's graph structure. Session headers
(e.g. auth) are forwarded automatically.

```python
async with client.threads.stream(thread_id="t", assistant_id="agent") as thread:
    graph = await thread.agent.get_tree(xray=True)
    for node in graph["nodes"]:
        print(node["id"])
```

## Subgraph Projections

`ScopedStreamHandle` scopes projections to a specific subgraph namespace:

```python
from langgraph_sdk._async.stream import ScopedStreamHandle

handle = ScopedStreamHandle(
    thread=thread,
    path=("worker:abc",),
    graph_name="worker",
    trigger_call_id=None,
)
async for call in handle.tool_calls:
    print(call.name)   # only tool calls from the "worker:abc" subgraph
```

## Transport

By default the client uses SSE. WebSocket transport is also available:

```python
async with client.threads.stream(
    thread_id="t",
    assistant_id="agent",
    transport="websocket",
) as thread:
    ...
```
