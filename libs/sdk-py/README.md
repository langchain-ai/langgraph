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
