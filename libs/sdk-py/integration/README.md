# v3 streaming integration — Docker harness

End-to-end exercise of the `langgraph_sdk` v3 streaming surface against
a real `langgraph-api` server running locally via docker compose. The
suite ships in two forms:

- **`libs/sdk-py/tests/integration/`** — pytest tests behind the
  `integration` marker. Run with `pytest -m integration` from
  `libs/sdk-py`. Excluded from `make test` by default; the autouse
  fixture skips them when the API at `http://localhost:2024` is
  unreachable.
- **`libs/sdk-py/integration/scripts/`** — equivalent standalone
  scripts kept for hands-on debugging. Run directly with
  `uv run python integration/scripts/test_values.py` (etc.).

## Layout

```
integration/
├── docker-compose.yml       # postgres + redis + api
├── Dockerfile               # thin layer on langchain/langgraph-api:latest-py3.12
├── langgraph.json           # registers all three example graphs
├── graph/
│   ├── streaming_graph.py   # plain StateGraph (registered as `agent`)
│   ├── tools_agent.py       # `create_agent` + tool node (registered as `tools_agent`)
│   └── deep_agent.py        # `create_deep_agent` + SubAgent (registered as `deep_agent`)
└── scripts/                 # standalone runnables that mirror the pytest tests
```

## How the image works

The `api` service is built from `Dockerfile` as a thin layer on top of
`langchain/langgraph-api:latest-py3.12`. That base image bundles
everything the API needs (the Python `langgraph_api`,
`langgraph_runtime_postgres`, `langgraph_license`, `langgraph_grpc_common`
packages, plus the Go `core-api-grpc` binary and its entrypoint at
`/storage/entrypoint.sh`).

Our layer installs the graph dependencies (`langchain`, `deepagents`)
that aren't in the base image, then copies the project graphs and
`langgraph.json`. No license override, no Anthropic dep — `deep_agent`
uses `FakeMessagesListChatModel` for both supervisor and researcher.

The base image is the `licensed` variant, so it requires a real
`LANGSMITH_API_KEY` (or `LANGGRAPH_CLOUD_LICENSE_KEY`) at runtime. The
compose file passes it through from the host shell.

Pinning: `latest-py3.12` tracks whatever was most recently published
to Docker Hub (currently the same digest as `0.9.0.dev3-py3.12`). This
intentionally surfaces upstream regressions early. To pin a specific
revision, edit `Dockerfile` to a tagged or `@sha256:<digest>` form.

## Start the stack

```bash
export LANGSMITH_API_KEY=...   # required by the licensed image
cd libs/sdk-py/integration
docker compose build
docker compose up -d
```

The api container exposes the LangGraph API at `http://localhost:2024`.
Postgres binds to `localhost:5443` and Redis to `localhost:6380` (offset
from common defaults to avoid clashing with other local services).

Wait until the api healthcheck passes:

```bash
docker compose ps   # api should report (healthy) after ~10-30s
```

Tail logs while it boots:

```bash
docker compose logs -f api
```

## Run the pytest suite (preferred for CI)

From `libs/sdk-py`:

```bash
uv run pytest tests/integration/ -m integration
```

Each test file targets one projection or feature; both async and sync
paths are covered. The autouse fixture in `tests/integration/conftest.py`
hits `/ok` once at session start and skips everything if the API is
unreachable.

## Run a standalone script

From `libs/sdk-py`:

```bash
uv run python integration/scripts/test_values.py
uv run python integration/scripts/test_messages.py
# ... etc.
```

Each script verifies the API is reachable, runs the async path, runs
the sync path, and asserts the invariants for that projection. Exit
code is `0` on success.

## What the suite covers

| Surface | pytest test | Script |
|---|---|---|
| `thread.values` | `test_values.py` | `scripts/test_values.py` |
| `thread.messages` (+ inner `.text`) | `test_messages.py` | `scripts/test_messages.py` |
| `thread.tool_calls` against `tools_agent` | `test_tools.py` | `scripts/test_tools.py` |
| `thread.subgraphs` against `agent` and `deep_agent` | `test_subgraphs.py` | `scripts/test_subgraphs.py` |
| `thread.extensions["progress"]` | `test_extensions.py` | `scripts/test_extensions.py` |
| `thread.interrupted` / `run.respond` | `test_lifecycle.py` | `scripts/test_lifecycle.py` |
| Mid-iteration SSE close + REST fallback | `test_reconnect.py` | `scripts/test_reconnect.py` |
| WebSocket transport | `test_websocket.py` | `scripts/test_websocket.py` |
| `agent.get_tree` + extensions cache identity | `test_helpers.py` | `scripts/test_helpers.py` |
| Mid-run cancel via `runs.cancel` | `test_cancel.py` | `scripts/test_cancel.py` |
| Concurrent `threads.stream()` contexts | `test_concurrent.py` | `scripts/test_concurrent.py` |
| `update_state` during interrupt | `test_update_state.py` | `scripts/test_update_state.py` |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LANGGRAPH_INTEGRATION_URL` | `http://localhost:2024` | API base URL the tests / scripts dial |
| `LANGSMITH_API_KEY` | (required) | Real key for the licensed image; passed into the container |
| `LANGGRAPH_CLOUD_LICENSE_KEY` | (optional) | Alternative license path for the image |

### Feature flags on the API

The v3 thread-centric streaming protocol is gated behind
`FF_OPTIMIZED_STREAMING=true` in the API server. The compose file sets
it for the `api` service. If you point the tests at a remotely-deployed
langgraph-api, confirm the flag is enabled there too, otherwise the v3
endpoints (`POST /threads/{id}/stream/events`, `POST /threads/{id}/commands`,
etc.) won't exist and every test will fail at the first request.

## The three example graphs

### `agent` (`graph/streaming_graph.py`)

A plain `StateGraph` with four nodes:

```
__start__ → stream_message → call_tool → ask_human → run_subgraph → __end__
```

- `stream_message` writes progress events and returns a final `AIMessage`.
- `call_tool` invokes the `search` tool and emits a `ToolMessage`.
- `ask_human` calls `interrupt("Are we good?")` to test
  `thread.interrupted` and `thread.run.respond(...)`.
- `run_subgraph` invokes a nested `StateGraph` once. Plain nested
  invokes do not produce a `thread.subgraphs` child handle in the v3
  scoped-handle sense.

Hermetic — no network or API keys required. Most tests target this graph.

### `tools_agent` (`graph/tools_agent.py`)

A `create_agent` graph with a real `search` tool. Used by `test_tools`
to verify the v3 `tools` channel fires through the canonical
langchain-agent surface (rather than the synthetic `streaming_graph`'s
hand-rolled tool message).

### `deep_agent` (`graph/deep_agent.py`)

A deep agent built with `create_deep_agent` from the `deepagents`
package. Has one configured `SubAgent` (`researcher`). Both the
supervisor and the researcher use `FakeMessagesListChatModel` (via a
small `bind_tools`-aware subclass) with pre-scripted responses, so the
graph is hermetic and deterministic. The supervisor's first response
issues `task(subagent_type="researcher", description=...)`, which is
the canonical path that produces a `thread.subgraphs` child handle.

## Tearing down

```bash
docker compose down -v
# `-v` drops the postgres/redis volumes too (we use tmpfs but this is
# the convention).
```

## Troubleshooting

**`Cannot reach the integration API at http://localhost:2024`**
The api container isn't up or hasn't passed its healthcheck yet. Check
`docker compose logs api`. The base image will fail to start if
`DATABASE_URI` or `REDIS_URI` is missing.

**`License verification failed` on startup**
The licensed base image requires `LANGSMITH_API_KEY` (or
`LANGGRAPH_CLOUD_LICENSE_KEY`) in the environment. Export one before
`docker compose up`.

**`thread.subgraphs` is empty for `deep_agent`**
The supervisor's fake model's first scripted response should include a
`task` tool call. If you've edited `graph/deep_agent.py`, restart the
api container (`docker compose restart api`) so it reloads the mounted
graph.

## What's intentionally **not** here

- **TLS / `wss://`.** The WebSocket test uses `ws://` (fine locally;
  production would require TLS).
- **True reconnect-under-failure.** `test_reconnect` exercises a
  graceful client close, which does NOT trigger the controller's
  reconnect path (that fires only on a non-cancelled `shared.done`
  error). True transport-error reconnect/dedup is covered in unit
  tests with a mock transport, where the failure mode is deterministic.
