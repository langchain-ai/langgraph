# LangGraph Python SDK

This repository contains the Python SDK for interacting with the LangSmith Deployment REST API.

## Quick Start

To get started with the Python SDK, [install the package](https://pypi.org/project/langgraph-sdk/)

```bash
pip install -U langgraph-sdk
```

You will need a running LangGraph API server. If you're running a server locally using `langgraph-cli`, SDK will automatically point at `http://localhost:8123`, otherwise
you would need to specify the server URL when creating a client.

```python
from langgraph_sdk import get_client

# If you're using a remote server, initialize the client with `get_client(url=REMOTE_URL)`
client = get_client()

# List all assistants
assistants = await client.assistants.search()

# We auto-create an assistant for each graph you register in config.
agent = assistants[0]

# Start a new thread
thread = await client.threads.create()

# Start a streaming run
input = {"messages": [{"role": "human", "content": "what's the weather in la"}]}
async for chunk in client.runs.stream(thread['thread_id'], agent['assistant_id'], input=input):
    print(chunk)
```

## Known Limitations

- **WebSocket transport** requires `websockets>=14` and is only available on the async client (`AsyncThreadStream`). The sync client (`SyncThreadStream`) uses SSE exclusively.
- **`thread.extensions[name]`** opens a new subscription each time the same name is accessed. Assign the projection to a variable and reuse it within a single session rather than re-indexing across multiple iterations.
- **Sync streaming** drives the lifecycle watcher in a background thread. Long-lived sync sessions will hold that thread open until the context manager exits.
- **Reconnect attempts** are limited to 5 by default for both the shared SSE fan-out and the lifecycle watcher. Persistent network partitions will surface as `RuntimeError` on in-flight projections.

## Thread-Centric Streaming (v3)

`client.threads.stream()` returns a context manager that owns the SSE session for one
thread. Typed projections — values snapshots, message streams, tool calls, custom
events — all share the same underlying connection.

```python
from langgraph_sdk import get_client
import asyncio

client = get_client()

async with client.threads.stream(
    thread_id="my-thread",
    assistant_id="agent",
) as thread:
    await thread.run.start(input={"messages": [{"role": "user", "content": "hi"}]})

    # Start all consumers concurrently so they share one SSE connection.
    async def get_messages():
        return [s async for s in thread.messages]

    async def get_tool_calls():
        return [c async for c in thread.tool_calls]

    messages, tool_calls = await asyncio.gather(get_messages(), get_tool_calls())

    for stream in messages:
        print(await stream.text)          # accumulated text

    final = await thread.output           # terminal state values
```
