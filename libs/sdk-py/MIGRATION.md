# Migration Guide: v2 → v3 Streaming

`client.runs.stream()` (v2) remains fully supported. This guide covers how to
adopt the new `client.threads.stream()` (v3) surface when you want typed
projections, shared SSE fan-out, or WebSocket transport.

## Minimal before/after

**v2 — `client.runs.stream()`**

```python
from langgraph_sdk import get_client

client = get_client()

thread = await client.threads.create()
async for chunk in client.runs.stream(
    thread["thread_id"],
    "agent",
    input={"messages": [{"role": "user", "content": "hello"}]},
    stream_mode="messages",
):
    print(chunk.event, chunk.data)
```

**v3 — `client.threads.stream()`**

```python
from langgraph_sdk import get_client
import asyncio

client = get_client()

async with client.threads.stream(assistant_id="agent") as thread:
    await thread.run.start(input={"messages": [{"role": "user", "content": "hello"}]})

    async for stream in thread.messages:
        print(await stream.text)
```

## Key differences

| | v2 `client.runs.stream()` | v3 `client.threads.stream()` |
|---|---|---|
| Thread creation | Explicit `client.threads.create()` | Lazy (minted client-side if omitted) |
| Connection per run | Yes | No — shared SSE for the session |
| Typed projections | No (raw `StreamPart`) | Yes (`messages`, `tool_calls`, `values`, …) |
| Subgraph streaming | Not supported | `thread.subgraphs` / `thread.subagents` |
| WebSocket transport | No | Yes (`transport="websocket"`, async only) |
| Interrupt handling | Manual polling | `thread.interrupted` / `thread.run.respond()` |
| Terminal state | Included in stream | `await thread.output` |

## Reattaching to an existing thread

```python
async with client.threads.stream(
    thread_id="existing-thread-id",
    assistant_id="agent",
) as thread:
    # If the run already completed, thread.output resolves immediately.
    result = await thread.output
```

## Consuming multiple projections concurrently

All projections share one SSE connection. Use `asyncio.gather` (or
`asyncio.TaskGroup`) to start multiple consumers before any single projection
has finished — the fan-out task routes events to all subscribers in parallel.

```python
async with client.threads.stream(assistant_id="agent") as thread:
    await thread.run.start(input={"messages": [{"role": "user", "content": "hi"}]})

    async def collect_messages():
        return [s async for s in thread.messages]

    async def collect_tool_calls():
        return [c async for c in thread.tool_calls]

    messages, tool_calls = await asyncio.gather(
        collect_messages(),
        collect_tool_calls(),
    )
```

## Human-in-the-loop (interrupts)

```python
async with client.threads.stream(assistant_id="agent") as thread:
    await thread.run.start(input={"messages": [{"role": "user", "content": "book a flight"}]})

    # Wait for the run to pause at an interrupt node.
    # thread.interrupted becomes True when input.requested arrives.
    while not thread.interrupted:
        await asyncio.sleep(0.1)

    # Resume with a human response (unambiguous when only one interrupt is outstanding).
    await thread.run.respond("yes, confirm booking")

    result = await thread.output
```

## Sync client

The sync client mirrors the async API without `async`/`await`:

```python
from langgraph_sdk import get_sync_client

client = get_sync_client()

with client.threads.stream(assistant_id="agent") as thread:
    thread.run.start(input={"messages": [{"role": "user", "content": "hello"}]})
    for stream in thread.messages:
        print(stream.text)
```

The sync client uses SSE only (`transport="websocket"` is not supported).
