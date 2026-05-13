# Python SDK v3 streaming — high-level design

---

## 1. What it is

A thread-centric streaming surface for `langgraph-sdk` (Python). One context manager per thread, typed command dispatch, typed projections. Mirrors `@langchain/langgraph-sdk` (JS).

```python
async with client.threads.stream(assistant_id="agent") as thread:
    await thread.run.start(input={"x": 1})
    async for message in thread.messages:
        ...
    final_state = await thread.values
```

Additive — `client.runs.stream(...)` and `client.threads.join_stream(...)` stay.

---

## 2. Architecture

```
   User code
     │
     │  async with client.threads.stream(...) as thread:
     │      await thread.run.start(...)
     │      async for msg in thread.messages: ...
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Public surface — langgraph_sdk._async.stream                    │
│                                                                  │
│   AsyncThreadStream                                              │
│     .run.start    .input.respond    .agent.get_tree              │
│     .events                                                      │
│     .values  .messages  .tool_calls  .subgraphs  .subagents      │
│     .extensions["name"]                                          │
│     .interrupted  .interrupts                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Internals — langgraph_sdk.stream                                 │
│                                                                  │
│   StreamController         (event router, per-run dispatch)      │
│      ├─ projections        (values / messages / tool_calls / …)  │
│      └─ MultiCursorBuffer  (per-projection replay log)           │
│                                                                  │
│   Subscription matcher     (channel + namespace filters)         │
│   Union-filter SSE         (one shared connection per thread)    │
│   Lifecycle watcher SSE    (always-on, surfaces interrupts)      │
│                                                                  │
│   ProtocolSseTransport ──── WebSocketTransport                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                       v3 protocol over HTTP/SSE or WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ langgraph-api server                                             │
│   POST /threads/{thread_id}/commands                             │
│   POST /threads/{thread_id}/stream/events                        │
│   WS   /threads/{thread_id}/stream/events                        │
└─────────────────────────────────────────────────────────────────┘
```

Each `AsyncThreadStream` holds one union-filter SSE (subscription set rotates as projections come and go) plus one always-on lifecycle SSE — two HTTP connections per active thread. `AsyncThreadStream` is transport-agnostic; SSE and WebSocket implement the same internal `TransportAdapter` contract.

---

## 3. User-facing surface

| API | Notes |
|---|---|
| `client.threads.stream(thread_id=None, *, assistant_id, headers=None)` | Entry point. Returns an `AsyncThreadStream` async context manager. Mints `uuid.uuid4()` when `thread_id` is None. |
| `thread.run.start(input=, config=, metadata=)` | Dispatches `run.start`. Returns `{"run_id": "..."}`. |
| `thread.run.respond(...)` | Resume after interrupt. |
| `thread.agent.get_tree(...)` | Agent introspection. |
| `thread.events` | Raw `AsyncIterator[Event]` over every channel. |
| `thread.values` | `AsyncIterator[snapshot]` plus `Awaitable[final_state]`. |
| `thread.messages` | `AsyncIterator[StreamingMessageHandle]` — typed over `langchain-core` `BaseMessage`. |
| `thread.tool_calls` | `AsyncIterator[ToolCallHandle]`. |
| `thread.subgraphs` / `thread.subagents` | Nested handles for graph composition. |
| `thread.extensions["name"]` | Per-extension dispatch on `custom:<name>`. |
| `thread.interrupted` / `thread.interrupts` | Lifecycle state, always current. |

### 3.1 Nested handles — subgraphs and subagents

`thread.subgraphs` and `thread.subagents` are streams of **invocations**, not static lists of registered components. Each iteration yields one handle scoped to one execution:

```python
async with client.threads.stream(assistant_id="agent") as thread:
    await thread.run.start(input={...})

    async for subgraph in thread.subgraphs:
        async for message in subgraph.messages:
            ...
        async for call in subgraph.tool_calls:
            ...
```

A `SubgraphHandle` (or `SubagentHandle`) exposes the same projection surface as the top-level thread — `messages`, `tool_calls`, `subgraphs`, `subagents`, media — filtered to events whose namespace matches that invocation's path. Nesting composes: a subgraph that itself invokes another subgraph yields a fresh handle from `subgraph.subgraphs`. There is no static registration step on the SDK side; the set of subgraphs/subagents that appear is discovered at runtime from event namespaces emitted by the graph.

---

## 4. Migration

Pre-v3 (untyped, one run per call):

```python
async for chunk in client.runs.stream(
    thread_id, assistant_id, input={...}, stream_mode="messages",
):
    # chunk.event and chunk.data are untyped
    ...
```

v3 (typed, thread-centric, multiple runs per thread):

```python
async with client.threads.stream(thread_id, assistant_id="agent") as thread:
    await thread.run.start(input={...})
    async for message in thread.messages:
        ...
```

Differences:

- The thread is the context manager — multiple `run.start` calls compose inside one session.
- Projections replace `stream_mode` — pick the typed iterable you need.
- Reattach is automatic — `client.threads.stream(thread_id="existing-id", ...)` replays buffered events and goes live. Replaces `client.threads.join_stream(...)`.
- The protocol is parsed once at the SDK boundary; projections expose Python objects, not raw frames.

### 4.1 Available projections

The complete set, all defined in `langgraph_sdk._async.stream`:

- `thread.events` — raw `Event` dicts over every channel; untyped, useful for debug or to drop below the typed surface.
- `thread.values` — state snapshots plus final state. Replaces `stream_mode="values"`.
- `thread.messages` — `StreamingMessageHandle` typed over `langchain-core` `BaseMessage`. Replaces `stream_mode="messages"`.
- `thread.tool_calls` — `ToolCallHandle` per tool invocation.
- `thread.subgraphs` / `thread.subagents` — nested handles per invocation (see §3.1).
- `thread.extensions["name"]` — per-extension events on `custom:<name>` channels. Replaces `stream_mode="custom"`.

The set is closed at the SDK boundary; `extensions["name"]` is the open-ended escape hatch for server-side transformers that emit on custom channels.

---

## 5. Dependencies

- `langchain-protocol>=0.0.15` — CDDL-derived `TypedDict`s for the protocol wire shape.
- `langchain-core>=1.4.0,<2` — required by the messages projection to construct `BaseMessage` subclasses. Matches the JS SDK's hard-dep stance.
- `httpx`, `orjson` — already required.
- Python 3.10+.
- Server: `langgraph-api` with `FF_V2_EVENT_STREAMING` enabled and ≥ 0.9.0rc1.
