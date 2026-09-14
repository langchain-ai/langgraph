---
type: Integration
title: SDK and API Integration
description: Programming interfaces for interacting with LangGraph servers via SDKs and REST APIs, supporting async execution, streaming, thread management, and human-in-the-loop workflows.
tags: [sdk, api, client, async, streaming, threads, python, javascript]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-0432c8982de468b84bfe45bf
    resource: repo://libs/cli/examples/langgraph.json
  - id: openwiki-source-267e2c951319a772291005e0
    resource: repo://libs/cli/langgraph_cli/cli.py
  - id: openwiki-source-75f1e442bb6ed78ddb1df39d
    resource: repo://libs/cli/README.md
  - id: openwiki-source-bd2660848c9f79f281b2cc27
    resource: repo://libs/sdk-py/langgraph_sdk/__init__.py
  - id: openwiki-source-7e98e631021478f789d6e722
    resource: repo://libs/sdk-py/langgraph_sdk/_async/assistants.py
  - id: openwiki-source-929fa28a693aeed7ed8d92ec
    resource: repo://libs/sdk-py/langgraph_sdk/_async/client.py
  - id: openwiki-source-9c1e48d971f33eb839dd8605
    resource: repo://libs/sdk-py/langgraph_sdk/_async/http.py
  - id: openwiki-source-abd12d83a44b12e8079f6e83
    resource: repo://libs/sdk-py/langgraph_sdk/_async/runs.py
  - id: openwiki-source-5e9ef0e3e4ad05838f948cad
    resource: repo://libs/sdk-py/langgraph_sdk/_async/stream.py
  - id: openwiki-source-f4e8a5738154e36ff2a660e8
    resource: repo://libs/sdk-py/langgraph_sdk/_async/threads.py
  - id: openwiki-source-0757d3f687c6bc2ec5f03949
    resource: repo://libs/sdk-py/langgraph_sdk/_sync/client.py
  - id: openwiki-source-f8a06ac0bd7d480c3016622d
    resource: repo://libs/sdk-py/langgraph_sdk/client.py
  - id: openwiki-source-cf0a2d1242c22c6014dacfbf
    resource: repo://libs/sdk-py/langgraph_sdk/errors.py
  - id: openwiki-source-8e1e64b86767cf70d250cb77
    resource: repo://libs/sdk-py/langgraph_sdk/runtime.py
  - id: openwiki-source-789ea08912740cf67cfc3ed9
    resource: repo://libs/sdk-py/langgraph_sdk/schema.py
  - id: openwiki-source-0ef83e09a2317f0a2bb94f23
    resource: repo://libs/sdk-py/langgraph_sdk/sse.py
  - id: openwiki-source-db88622f6b74bb9adbcdfc38
    resource: repo://libs/sdk-py/langgraph_sdk/stream/transport/__init__.py
  - id: openwiki-source-5205229b49576f2f6978328e
    resource: repo://libs/sdk-py/MIGRATION.md
  - id: openwiki-source-583d6fb6d1a776a95837b56a
    resource: repo://libs/sdk-py/README.md
  - id: openwiki-source-8c56d9098c74859d386c17e0
    resource: repo://libs/sdk-py/tests/integration/test_lifecycle.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

LangGraph provides programmatic access to graphs through **SDKs** (Python and JavaScript) and a **REST API**. The SDK and API enable applications to:

- **Connect to a running LangGraph server** (local or remote)
- **Discover and execute graphs** as registered assistants
- **Manage sessions** via threads (persistent conversation state)
- **Stream execution results** in multiple formats (values, messages, events, tool calls)
- **Implement human-in-the-loop workflows** with interrupts and resumption
- **Configure execution** with custom input, context, and metadata
- **Handle network resilience** with automatic reconnection logic

The SDK is **async-first**, meaning all client methods are asynchronous (`async`/`await` in Python). The REST API uses HTTP (SSE for streaming) and supports both individual REST calls and bidirectional protocols (WebSocket for advanced streaming modes).

---

## Core Concepts

### Assistants

An **assistant** is a registered graph configured on the server, auto-created when you define a graph in `langgraph.json`. Each assistant has:

- A unique `assistant_id` (typically the graph ID)
- Version tracking and metadata
- Input/output schemas (queryable via the API)
- Graph definition (topology, subgraphs) accessible for UI rendering

### Threads

A **thread** represents a **conversation session** — the persistent state shared across multiple invocations of a graph. Threads:

- Are identified by `thread_id`
- Store accumulated graph state (channels, messages, checkpoints)
- Support metadata and custom extensions (`thread.extensions`)
- Persist across client reconnections and server restarts
- Enable branching and history inspection via checkpoints

### Runs

A **run** is a single invocation of a graph within a thread context. Runs:

- Are stateful (belong to a thread) or stateless
- Accept input, config (overrides), and metadata
- Stream results in multiple modes (`values`, `messages`, `events`, `tasks`)
- Report status (`pending`, `running`, `success`, `error`, `interrupted`, `timeout`)
- Support interrupts (human-in-the-loop pausing and resumption)

### Streaming Modes

The SDK and API support multiple streaming modes to handle different use cases:

- **`values`** — state snapshots as execution progresses
- **`messages`** — complete message objects from the LLM
- **`events`** — low-level execution events (node start/end, etc.)
- **`tasks`** — task start/finish lifecycle events
- **`checkpoints`** — checkpoint creation events for recovery
- **`custom`** — application-specific events

### Thread Streaming (v3)

**`client.threads.stream()`** (v3 protocol) provides a unified interface:

- Single shared **SSE connection** per thread session (fan-out to multiple consumers)
- **Typed projections**: `thread.values`, `thread.messages`, `thread.tool_calls`, `thread.extensions`
- **Command dispatch**: `thread.run.start(...)`, `thread.run.respond(...)`
- **Lifecycle watcher**: `thread.interrupted`, `thread.interrupts`, `thread.output`
- **WebSocket transport** option for reduced latency (async only)

---

## Client Initialization

### Python SDK: Async Client

```python
from langgraph_sdk import get_client

# Default: connect to local server at http://localhost:8123
client = get_client()

# Or specify a remote URL
client = get_client(url="http://my-server:8123")

# Optional: configure API key (auto-loads from env by default)
client = get_client(api_key="your-api-key")

# Optional: skip env-based API key loading
client = get_client(api_key=None)
```

**Async context:** All SDK methods must be awaited; use within an `async def` or `asyncio.run()`.

### Python SDK: Sync Client

```python
from langgraph_sdk import get_sync_client

# Synchronous variant for scripts and non-async contexts
client = get_sync_client(url="http://localhost:8123")

# Sync client works in synchronous code (no async/await needed)
assistant = client.assistants.get("agent")
```

### In-Process Connection

When the client runs inside a LangGraph server (e.g., in a subagent node), omit the URL:

```python
client = get_client(url=None)  # Infers ASGI transport; works within server only
```

### API Key Resolution

The SDK auto-loads API keys in this priority order:

1. Explicit `api_key` argument
2. `LANGGRAPH_API_KEY` environment variable
3. `LANGSMITH_API_KEY` environment variable
4. `LANGCHAIN_API_KEY` environment variable

### HTTP Client Configuration

Customize timeouts and headers:

```python
import httpx
from langgraph_sdk import get_client

client = get_client(
    url="http://localhost:8123",
    timeout=httpx.Timeout(connect=10, read=600, write=300, pool=5),
    headers={"Custom-Header": "value"}
)
```

Default timeouts: connect=5s, read=300s, write=300s, pool=5s.

---

## Core Resources

### Assistants Client

```python
# Discover available graphs
assistants = await client.assistants.search(limit=10)

# Get specific assistant
assistant = await client.assistants.get("agent")

# Fetch graph definition (nodes, edges)
graph_def = await client.assistants.get_graph("agent")

# Get input/output schemas
schema = await client.assistants.get_schema("agent")

# Fetch subgraph definitions (with xray parameter)
subgraphs = await client.assistants.get_subgraphs("agent", xray=2)
```

### Threads Client

```python
# Create a new thread
thread = await client.threads.create(
    metadata={"user_id": "123", "session": "important"}
)
thread_id = thread["thread_id"]

# Fetch existing thread state
state = await client.threads.get(thread_id)

# Update thread state directly (without executing graph)
await client.threads.update_state(
    thread_id,
    values={"messages": [...]},  # values to merge
    as_node="my_node"  # which node to attribute the write to
)

# List threads with filtering
threads = await client.threads.search(
    metadata_filter={"user_id": "123"},
    limit=20
)

# Delete a thread
await client.threads.delete(thread_id)
```

### Runs Client (v2 Protocol)

```python
# Simple stream (v2)
async for chunk in client.runs.stream(
    thread_id=thread_id,
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "hello"}]},
    stream_mode="values"
):
    print(chunk)

# Wait for completion (blocking)
result = await client.runs.wait(
    thread_id=thread_id,
    assistant_id="agent",
    input={"messages": [...]}
)

# Get run status
run = await client.runs.get(thread_id, run_id)
print(run["status"])  # "pending", "running", "success", etc.

# Cancel a run
await client.runs.cancel(thread_id, run_id)
```

### Threads Client Streaming (v3 Protocol)

```python
import asyncio

async with client.threads.stream(
    thread_id="my-thread",  # optional; auto-creates if omitted
    assistant_id="agent"
) as thread:
    # Start a new run
    await thread.run.start(
        input={"messages": [{"role": "user", "content": "What is 2+2?"}]},
        config={"configurable": {"model": "gpt-4"}},
        metadata={"trace_id": "abc123"}
    )
    
    # Consume projections concurrently (share one SSE)
    async def collect_messages():
        return [msg async for msg in thread.messages]
    
    async def collect_tool_calls():
        return [call async for call in thread.tool_calls]
    
    async def collect_values():
        return [val async for val in thread.values]
    
    messages, tool_calls, values = await asyncio.gather(
        collect_messages(),
        collect_tool_calls(),
        collect_values()
    )
    
    # Get terminal output (waits for "completed" lifecycle event)
    output = await thread.output
    print("Final state:", output)

# Async context manager automatically closes SSE connection on exit
```

### Human-in-the-Loop Interrupts

```python
async with client.threads.stream(assistant_id="agent") as thread:
    await thread.run.start(input={"messages": [...]})
    
    # Wait for an interrupt (loop until thread.interrupted is True)
    while not thread.interrupted:
        async for _ in thread.values:
            if thread.interrupted:
                break
        if not thread.interrupted:
            await asyncio.sleep(0.1)
    
    # Access interrupt details
    for interrupt in thread.interrupts:
        print(f"Interrupt ID: {interrupt['interrupt_id']}")
        print(f"Namespace: {interrupt['namespace']}")
        print(f"Value: {interrupt['value']}")
    
    # Resume with human response
    # (if only one interrupt, omit interrupt_id)
    await thread.run.respond(
        response="yes, proceed",
        interrupt_id="specific-interrupt-id"  # optional
    )
    
    # Continue consuming events until completion
    final = await thread.output
```

---

## Streaming and Event Handling

### Stream Modes and Event Structure

The server emits events via SSE or WebSocket. The v2 protocol wraps events in `StreamPart` objects:

```python
# StreamPart has: event (string), data (dict)
async for chunk in client.runs.stream(...):
    print(chunk.event)  # "values", "messages", "updates", etc.
    print(chunk.data)   # event-specific data (dict or None)
```

The v3 protocol (typed projections) automatically decodes events:

```python
async with client.threads.stream(...) as thread:
    # Typed projection: automatically handles decoding
    async for message in thread.messages:
        print(message)  # Already parsed message object
    
    async for tool_call in thread.tool_calls:
        print(tool_call)  # Already parsed tool call
```

### Single-Consumer Constraint

**Important:** Each projection (`thread.messages`, `thread.values`, etc.) is a **single-consumer** stream. If you iterate twice, the second iteration receives no events. Instead:

1. **Collect into memory** before iterating:
   ```python
   messages = [m async for m in thread.messages]
   for m in messages:
       print(m)
   ```

2. **Or use `asyncio.gather`** to start all consumers concurrently:
   ```python
   messages, values = await asyncio.gather(
       async_list(thread.messages),
       async_list(thread.values)
   )
   ```

### Thread Extensions

Access custom extension channels via projections:

```python
async with client.threads.stream(assistant_id="agent") as thread:
    await thread.run.start(input={...})
    
    # Access custom extension channel
    async for event in thread.extensions["my_custom_channel"]:
        print(event)
```

---

## Configuration and Customization

### Passing Config to Runs

Override graph config (e.g., LLM model, temperatures) per run:

```python
await thread.run.start(
    input={...},
    config={
        "configurable": {
            "model": "gpt-4",
            "temperature": 0.7,
            "retriever_type": "hybrid"
        }
    }
)
```

The graph's `config_schema` defines available overrides. Check with:

```python
schema = await client.assistants.get_schema("agent")
print(schema.get("config_schema"))  # Defines valid config keys
```

### Context and Metadata

Pass execution context and metadata:

```python
# v2 protocol (runs.stream)
async for chunk in client.runs.stream(
    thread_id=thread_id,
    assistant_id="agent",
    input={...},
    context={"user_id": "123"},  # Custom context (graph sees via ServerRuntime)
    metadata={"trace_id": "xyz"}  # Tracked in run records
):
    pass

# v3 protocol (threads.stream)
await thread.run.start(
    input={...},
    metadata={"trace_id": "xyz"}
)
```

---

## Error Handling

### API Errors

The SDK wraps HTTP errors in typed exceptions:

```python
from langgraph_sdk.errors import (
    APIError,
    APIStatusError,
    APIConnectionError,
    APIResponseValidationError
)

try:
    assistant = await client.assistants.get("nonexistent")
except APIStatusError as e:
    print(f"HTTP {e.status_code}: {e.message}")
    print(f"Error code: {e.code}")
    print(f"Request ID: {e.request_id}")
except APIConnectionError as e:
    print(f"Connection failed: {e.message}")
except APIError as e:
    print(f"API error: {e.message}")
```

### Reconnection Logic

The HTTP client automatically **retries up to 5 times** on transient failures (with exponential backoff). For long-lived streams (SSE/WebSocket):

- The lifecycle watcher (inside `AsyncThreadStream`) reconnects if the SSE drops
- Both the shared fan-out and the lifecycle watcher respect the 5-retry limit
- Persistent network partitions surface as `RuntimeError` on in-flight projections

Example: catching reconnection failures:

```python
async with client.threads.stream(assistant_id="agent") as thread:
    try:
        await thread.run.start(input={...})
        async for msg in thread.messages:
            print(msg)
    except RuntimeError as e:
        if "reconnect" in str(e).lower():
            print("Stream lost and could not reconnect")
        raise
```

---

## Transport Modes

### SSE (Server-Sent Events) — Default

- HTTP-based streaming protocol
- Works with both async and sync clients
- Reliable for single-direction server-to-client events
- Suitable for long-lived connections (minutes to hours)

### WebSocket — Advanced

- Async only; requires `websockets>=14`
- Lower latency; supports bidirectional communication
- Better for interactive, low-latency workflows

Enable WebSocket transport:

```python
async with client.threads.stream(
    assistant_id="agent",
    transport="websocket"  # Use WebSocket instead of SSE
) as thread:
    await thread.run.start(input={...})
    async for msg in thread.messages:
        print(msg)
```

---

## Server Discovery and Deployment

### langgraph.json Configuration

Define graphs for the server via `langgraph.json`:

```json
{
  "dependencies": [
    "langchain_openai",
    "langchain_anthropic",
    "."
  ],
  "graphs": {
    "agent": "./my_package/agent.py:graph",
    "tools_agent": "./my_package/tools_agent.py:compiled_graph",
    "rag": "./my_package/rag.py:rag_graph"
  },
  "env": "./.env",
  "python_version": "3.11",
  "databases": [
    {
      "name": "postgres_db",
      "dialect": "postgres",
      "url_env": "POSTGRES_URI"
    }
  ],
  "services": [
    {
      "name": "redis",
      "image": "redis:7"
    }
  ]
}
```

### Local Development: `langgraph dev`

Run the API server locally with hot reloading:

```bash
langgraph dev -c langgraph.json
```

- Watches for file changes and auto-reloads graphs
- Listens on `http://127.0.0.1:2024` by default
- SDK defaults to `http://localhost:8123` if not overridden

### Docker Deployment: `langgraph up`

Run the server in Docker (production-ready):

```bash
langgraph up -c langgraph.json -p 8123
```

- Builds a Docker container with dependencies, graphs, and config
- Exposes the API on port 8123
- Integrates databases and services defined in `langgraph.json`

### Building Custom Images: `langgraph build`

Generate a Docker image for CI/CD or custom deployments:

```bash
langgraph build -t my-langgraph-app:1.0.0 -c langgraph.json
docker push my-langgraph-app:1.0.0
```

---

## Runtime Behavior

### Graph Lifecycle During API Calls

The graph factory function is invoked in different contexts:

- **`threads.create_run`** — Execute the graph fully (node functions + edge evaluation)
- **`threads.update_state`** — Apply state mutations without executing nodes
- **`threads.get_state`** — Format state snapshots (graph structure informs which tasks are pending)
- **`assistants.get`** — Introspect graph structure (schemas, subgraphs, visualization)

Use `ServerRuntime.execution_runtime` to conditionally set up expensive resources only during execution:

```python
from langgraph_sdk.runtime import ServerRuntime

def graph_factory(runtime: ServerRuntime) -> CompiledGraph:
    if runtime.execution_runtime:
        # Initialize expensive resources only during actual execution
        llm = ChatOpenAI(model="gpt-4")
        retriever = load_retriever()
    else:
        # Lightweight versions for introspection
        llm = None
        retriever = None
    
    # Build graph using llm, retriever
    return graph
```

---

## Example Workflows

### Simple Chat

```python
from langgraph_sdk import get_client
import asyncio

async def main():
    client = get_client(url="http://localhost:8123")
    
    # Create a thread
    thread = await client.threads.create()
    
    # Stream a run
    final_state = None
    async for chunk in client.runs.stream(
        thread["thread_id"],
        "agent",
        input={"messages": [{"role": "user", "content": "What is LangGraph?"}]},
        stream_mode="values"
    ):
        print(f"Event: {chunk.event}, Data keys: {chunk.data.keys() if chunk.data else None}")
        if chunk.event == "values":
            final_state = chunk.data
    
    if final_state:
        print("Final messages:", final_state.get("messages"))

asyncio.run(main())
```

### Multi-Turn Conversation (v3)

```python
from langgraph_sdk import get_client
import asyncio

async def multi_turn():
    client = get_client()
    
    async with client.threads.stream(assistant_id="agent") as thread:
        # Turn 1
        await thread.run.start(
            input={"messages": [{"role": "user", "content": "Hi, what can you do?"}]}
        )
        final = await thread.output
        print("Assistant:", final["messages"][-1]["content"])
        
        # Turn 2 (same thread, conversation continues)
        await thread.run.start(
            input={"messages": final["messages"] + [
                {"role": "user", "content": "Tell me more about tool calling"}
            ]}
        )
        final = await thread.output
        print("Assistant:", final["messages"][-1]["content"])

asyncio.run(multi_turn())
```

### Human-in-the-Loop Agent

```python
from langgraph_sdk import get_client
import asyncio

async def human_in_loop():
    client = get_client()
    
    async with client.threads.stream(assistant_id="approval_agent") as thread:
        await thread.run.start(input={"request": "approve large purchase"})
        
        # Wait for interrupt (human approval required)
        while not thread.interrupted:
            async for _ in thread.values:
                if thread.interrupted:
                    break
            if not thread.interrupted:
                await asyncio.sleep(0.5)
        
        # Get interrupt details
        interrupt = thread.interrupts[0] if thread.interrupts else None
        if interrupt:
            print(f"Awaiting human approval: {interrupt['value']}")
            user_input = input("Approve? (yes/no): ")
            
            # Resume the run
            await thread.run.respond(user_input)
            final = await thread.output
            print("Final result:", final)

asyncio.run(human_in_loop())
```

---

## Best Practices

1. **Reuse client instances** — `get_client()` is lightweight; reuse the same client across requests.

2. **Use v3 streaming for concurrent projections** — `client.threads.stream()` shares one SSE connection, improving resource usage.

3. **Handle interrupts explicitly** — Don't assume runs complete; check `thread.interrupted` and call `thread.run.respond()`.

4. **Collect projections into memory if needed** — Single-consumer streams prevent double-iteration; use `asyncio.gather()` or list comprehensions.

5. **Configure timeouts for long-running graphs** — Adjust `timeout` parameter in `get_client()` if your graphs take >5 min to initialize.

6. **Use context manager for streaming** — `async with client.threads.stream(...) as thread:` ensures proper cleanup.

7. **Provide meaningful metadata** — Include `trace_id`, `user_id`, `session` in metadata for debugging and auditing.

8. **Test reconnection behavior** — Simulate network failures to verify your error handling works.

---

## API Reference Links

- **Python SDK**: [langgraph-sdk API reference](https://reference.langchain.com/python/langgraph-sdk/)
- **CLI**: [langgraph-cli API reference](https://reference.langchain.com/python/langgraph-cli/)
- **Full Docs**: [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview)
