# Changelog

## 0.4.10

### Removed

- **`PrioritySettings.fairness_weight`** — removed. The server stores
  `priority` and `fairness_group` on the thread, and every fairness group has
  weight 1 for now. A callback that still returns `fairness_weight` keeps
  working: the server ignores the key and logs a warning.

### Changed

- **`PrioritySettings.priority`** — now says -5 runs first and 5 runs last,
  matching how the server orders runs. The 0.4.9 docstring said the reverse.

## 0.4.9 and earlier

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

- **`Priority` (beta)** — register an async function with `@priority.assign`
  and name the object in the `priority` section of `langgraph.json`. The
  server calls the function with a `PriorityContext` at the first run on a
  thread with no settings and stores the `PrioritySettings` it returns
  (`priority`, `fairness_group`, `fairness_weight`) on the thread. Requires
  `langgraph-api` 0.18.0 or later and a `langgraph-cli` that supports the
  section; older versions of either ignore it.

### Changed

- `client.threads.stream()` now accepts `transport="sse"` (default) or
  `transport="websocket"` in place of the previous transport-agnostic default.

### Fixed

- Resource-scoped auth decorators now honor `actions=` and reject empty or
  invalid action lists. Because unmatched custom-auth paths remain allowed,
  deployments using action-scoped handlers should configure a global
  default-deny handler; `langgraph-api` 0.10+ warns about uncovered paths at
  startup. Resource-specific decorators retain matching `resources=` selectors
  for backward compatibility; use `@auth.on(resources=...)` for other resources.

### Notes

- The v3 streaming surface (`AsyncThreadStream`, `SyncThreadStream`, and all
  projection classes) is **new** in this release. The existing
  `client.runs.stream()` (v2) surface is unchanged and remains fully supported.
- `thread_id` is minted client-side (UUIDv4) when not provided; the server
  creates the thread row lazily on the first `run.start`.
