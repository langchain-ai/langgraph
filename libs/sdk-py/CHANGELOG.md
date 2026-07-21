# Changelog

## Unreleased

### Added

- **Thread-centric streaming (v3)** — `client.threads.stream()` returns an
  `AsyncThreadStream` (or `SyncThreadStream`) context manager that owns one
  SSE or WebSocket connection for the lifetime of a thread session.

- **Typed projections** — `thread.messages`, `thread.tool_calls`,
  `thread.values`, and `thread.extensions[name]` all share the same underlying
  transport connection. Iterating multiple projections concurrently expands the
  server-side filter union without opening additional connections.

- **Scoped subgraph handles** — `thread.subgraphs` (alias `thread.subagents`)
  yields one `ScopedStreamHandle` per direct child invocation, each exposing
  `.messages`, `.tool_calls`, and `.subgraphs` scoped to that namespace.

- **WebSocket transport** — pass `transport="websocket"` to
  `client.threads.stream()` to use a WebSocket connection instead of SSE
  (async client only).

- **Automatic reconnect** — the shared SSE fan-out and the lifecycle watcher
  both reconnect on transport drops, replaying missed events via a `since`
  cursor and deduplicating by `event_id`.

- **`thread.agent.get_tree()`** — fetches the assistant graph definition for
  the current session's `assistant_id` with optional `xray` depth control.

- **`thread.run.respond()`** — resumes a run after a server-side interrupt,
  resolving the outstanding `InterruptPayload` by `interrupt_id`.

- **`thread.output`** — awaitable that resolves to the terminal thread state
  `values` dict after the run lifecycle completes.

### Changed

- `client.threads.stream()` now accepts `transport="sse"` (default) or
  `transport="websocket"` in place of the previous transport-agnostic default.

### Notes

- The v3 streaming surface (`AsyncThreadStream`, `SyncThreadStream`, and all
  projection classes) is **new** in this release. The existing
  `client.runs.stream()` (v2) surface is unchanged and remains fully supported.
- `thread_id` is minted client-side (UUIDv4) when not provided; the server
  creates the thread row lazily on the first `run.start`.
