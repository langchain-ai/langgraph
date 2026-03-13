# Threat Model: LangGraph

> Generated: 2026-03-04 | Commit: a3823395 | Scope: Full monorepo (all libs/) | Mode: Open Source

## Scope

### In Scope

- `libs/langgraph` — Core graph execution engine (Pregel, StateGraph, channels, functional API with `@entrypoint`/`@task`)
- `libs/prebuilt` — High-level agent APIs (ToolNode, create_react_agent, ValidationNode)
- `libs/checkpoint` — Checkpoint serialization/deserialization (JsonPlusSerializer, EncryptedSerializer, BaseCache, stores)
- `libs/checkpoint-postgres` — PostgreSQL checkpoint and store implementation
- `libs/checkpoint-sqlite` — SQLite checkpoint and store implementation
- `libs/cli` — CLI for Docker-based deployment (`langgraph up/build/dev/new`)
- `libs/sdk-py` — Python SDK client for LangGraph Server API

### Out of Scope

- `libs/sdk-js` — Moved to external `langchain-ai/langgraphjs` repository; no source in this repo
- LangGraph Server / `langgraph-api` — Closed-source server runtime; not in this repo
- LangChain Core (`langchain-core`) — Upstream dependency; separate threat model
- User application code — Tools, prompts, model selection, deployment infrastructure
- LLM provider behavior — Model output content and safety
- LangSmith platform — Observability/tracing backend
- Tests, benchmarks, documentation — Not shipped code

### Assumptions

1. The project is used as a library/framework — users control their own application code, model selection, and deployment.
2. Checkpoint storage backends (databases) are deployed with proper access controls by the user.
3. LLM providers return well-formed responses per their documented API contracts.
4. The `langgraph.json` configuration file is developer-controlled and not user-supplied at runtime.
5. The CLI runs in a developer environment with Docker access.

---

## System Overview

LangGraph is an open-source Python framework for building stateful, multi-actor AI agent applications. It provides a graph-based execution model (Bulk Synchronous Parallel via the Pregel engine) where user-defined nodes process shared state through typed channels. The framework supports two authoring APIs: the declarative StateGraph API and the functional API (`@entrypoint`/`@task` decorators). It includes checkpointing (persistence of graph state to databases), tool execution (dispatching LLM-generated tool calls), remote graph composition (calling LangGraph Server APIs), and Docker-based deployment via a CLI.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Application                                 │
│  ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌───────────────────┐  │
│  │User Code │──>│ StateGraph │──>│ ToolNode │──>│ User-Registered   │  │
│  │(nodes,   │   │ / Pregel   │   │(prebuilt)│   │ Tools (BaseTool)  │  │
│  │ tools)   │   │  (core)    │   └──────────┘   └───────────────────┘  │
│  └──────────┘   └─────┬──────┘                                         │
│       │               │                                                 │
│  @entrypoint    ┌─────┘                                                │
│  @task ─────────┘                                                      │
│                       │                                                 │
│ - - - - - - - - - - - │ - - - - TB1: User/Framework API - - - - - - -  │
│                       │                                                 │
│                 ┌─────▼──────┐   ┌──────────────┐                      │
│                 │ Checkpoint │   │ RemoteGraph   │                      │
│                 │ Serializer │   │ (SDK client)  │                      │
│                 │(jsonplus)  │   └──────┬────────┘                      │
│                 └─────┬──────┘          │                               │
│                       │                 │                               │
│ - - - - - - - - - - - │ - - - - - - - -│- - TB2: Storage/Network - - - │
│                       ▼                 ▼                               │
│               ┌──────────────┐  ┌──────────────┐                       │
│               │  PostgreSQL  │  │  LangGraph   │                       │
│               │  / SQLite    │  │  Server API  │                       │
│               └──────────────┘  └──────────────┘                       │
│                                                                         │
│  ┌──────────┐                   ┌──────────────┐                       │
│  │   CLI    │──────────────────>│   Docker     │                       │
│  │(langgraph│   TB4: Config    │   Engine     │                       │
│  │ up/build)│                   └──────────────┘                       │
│  └──────────┘                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Components

| ID | Component | Description | Trust Level | Entry Points |
|----|-----------|-------------|-------------|--------------|
| C1 | StateGraph / Pregel | Core graph builder and execution engine | framework-controlled | `StateGraph.add_node()`, `StateGraph.compile()`, `Pregel.invoke()`, `Pregel.stream()` |
| C2 | JsonPlusSerializer | Checkpoint serialization/deserialization with msgpack, JSON, and pickle codecs | framework-controlled | `loads_typed()`, `dumps_typed()`, msgpack `ext_hook`, JSON `_reviver` |
| C3 | ToolNode | Dispatches LLM-generated tool calls to registered BaseTool instances | framework-controlled | `ToolNode._func()`, `_run_one()`, `_execute_tool_sync()` |
| C4 | RemoteGraph | Client for remote LangGraph Server API; implements PregelProtocol | framework-controlled | `RemoteGraph.stream()`, `RemoteGraph.invoke()`, `RemoteGraph.get_state()` |
| C5 | PostgresSaver / PostgresStore | PostgreSQL checkpoint and key-value store | framework-controlled | `from_conn_string()`, `put()`, `get_tuple()`, `search()` |
| C6 | SqliteSaver / SqliteStore | SQLite checkpoint and key-value store with JSON path filtering | framework-controlled | `from_conn_string()`, `put()`, `get_tuple()`, `search()` |
| C7 | EncryptedSerializer | AES-EAX authenticated encryption wrapper for checkpoint data | framework-controlled | `from_pycryptodome_aes()`, `loads_typed()`, `dumps_typed()` |
| C8 | CLI (langgraph_cli) | Docker-based build and deployment tooling | framework-controlled | `langgraph up`, `langgraph build`, `langgraph dev`, `langgraph new` |
| C9 | SDK Client (langgraph_sdk) | HTTP client for LangGraph Server API | framework-controlled | `get_client()`, `get_sync_client()`, `HttpClient` |
| C10 | User-Registered Tools | BaseTool instances provided by users | user-controlled | Tool `invoke()` / `ainvoke()` methods |
| C11 | User-Registered Nodes | Arbitrary callables added via `add_node()` or `@task`/`@entrypoint` | user-controlled | Node function signatures |
| C12 | Checkpoint Storage | PostgreSQL or SQLite databases storing serialized graph state | external | Database connection interface |
| C13 | Functional API | `@entrypoint`/`@task` decorators for function-based workflow authoring | framework-controlled | `entrypoint.__call__()`, `task()`, `_TaskFunction.__call__()` (`libs/langgraph/langgraph/func/__init__.py`) |
| C14 | BaseCache | Cache layer for task results with JsonPlusSerializer (pickle_fallback=False) | framework-controlled | `get()`, `set()`, `clear()` (`libs/checkpoint/langgraph/cache/base/__init__.py:15`) |

---

## Trust Boundaries

| ID | Boundary | Description | Controls (Inside) | Does NOT Control (Outside) |
|----|----------|-------------|-------------------|---------------------------|
| TB1 | User/Framework API | Where user-provided code and configuration enters the framework | Graph execution logic, channel semantics, default configs, validation of graph structure | User node implementations, tool behavior, model selection, prompt construction, state schema design |
| TB2 | Checkpoint Storage | Where serialized data enters/leaves the persistence layer | Serialization format, allowlists for deserialization, encryption (if configured) | Database access controls, who can write to the checkpoint tables, storage infrastructure security |
| TB3 | Remote API | Where data crosses the network to/from LangGraph Server | Outbound config sanitization (`_sanitize_config`), SDK HTTP transport, API key handling | Remote server behavior, response content integrity, network security (TLS) |
| TB4 | CLI Config/Docker | Where developer config drives container image generation | Dockerfile template structure, config schema validation, list-based subprocess args | `langgraph.json` file content, Docker daemon security, host filesystem |

### Boundary Details

#### TB1: User/Framework API

- **Inside**: Graph compilation validates structure (`libs/langgraph/langgraph/pregel/_validate.py`). Channel types enforce update semantics (`libs/langgraph/langgraph/channels/base.py`). Functional API validates entrypoint has at least one parameter (`libs/langgraph/langgraph/func/__init__.py:494`). Sensitive config keys filtered from metadata propagation (`libs/langgraph/langgraph/_internal/_config.py:319-329`).
- **Outside**: What user nodes do, what tools return, what LLMs generate, how users handle output.
- **Crossing mechanism**: Python function calls — `add_node(callable)`, `add_edge()`, `compile(checkpointer=...)`, `@entrypoint`, `@task`.

#### TB2: Checkpoint Storage

- **Inside**: `JsonPlusSerializer` controls serialization format (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py`). Msgpack type allowlist (`SAFE_MSGPACK_TYPES` in `libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py:14-77`). Msgpack method allowlist (`SAFE_MSGPACK_METHODS` in `libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py:82-86`). JSON module allowlist (`_check_allowed_json_modules`). Serde event hooks for monitoring (`libs/checkpoint/langgraph/checkpoint/serde/event_hooks.py`). Optional `EncryptedSerializer` wrapping (`libs/checkpoint/langgraph/checkpoint/serde/encrypted.py`). SQLite filter key regex validation (`libs/checkpoint-sqlite/langgraph/store/sqlite/base.py:110-127` and `libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/utils.py:14-28`). Parameterized SQL queries in Postgres (`libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py`).
- **Outside**: Database access controls, who can read/write checkpoint tables, storage backend integrity.
- **Crossing mechanism**: Database read/write operations — serialized bytes stored as BYTEA (Postgres) or BLOB (SQLite).

#### TB3: Remote API

- **Inside**: `_sanitize_config()` strips non-primitive values from outbound config (`libs/langgraph/langgraph/pregel/remote.py:369-396`). SDK handles API key from env vars (`libs/sdk-py/langgraph_sdk/_shared/utilities.py:39-41`). `RESERVED_HEADERS` prevents user override of auth headers.
- **Outside**: Remote server response content, network integrity, whether the server is legitimate.
- **Crossing mechanism**: HTTPS requests via `httpx` through `langgraph_sdk`.

#### TB4: CLI Config/Docker

- **Inside**: Config file parsed as JSON (`libs/cli/langgraph_cli/config.py`). Docker subprocess invoked with list-based args via `asyncio.create_subprocess_exec`, not `shell=True` (`libs/cli/langgraph_cli/exec.py:50`). Template downloads from hardcoded GitHub URLs (`libs/cli/langgraph_cli/templates.py`). Config schema validation covers store, auth, encryption, http, webhooks, checkpointer, and ui sections (`libs/cli/langgraph_cli/schemas.py`).
- **Outside**: Content of `langgraph.json`, Docker daemon behavior, filesystem permissions.
- **Crossing mechanism**: JSON file read, subprocess execution, ZIP download/extraction.

---

## Data Flows

| ID | Source | Destination | Data Type | Crosses Boundary | Protocol |
|----|--------|-------------|-----------|------------------|----------|
| DF1 | C12 (Checkpoint Storage) | C2 (JsonPlusSerializer) | Serialized checkpoint bytes (msgpack/JSON/pickle) | TB2 | Database read |
| DF2 | C2 (JsonPlusSerializer) | C1 (Pregel) | Deserialized Python objects (channel state) | TB2 | Function call |
| DF3 | LLM (external) | C3 (ToolNode) | Tool call arguments (JSON strings in AIMessage) | TB1 | Function call (via langchain-core) |
| DF4 | C3 (ToolNode) | C10 (User Tools) | Parsed argument dicts | TB1 | `tool.invoke(call_args)` |
| DF5 | C4 (RemoteGraph) | C1 (Pregel) | Stream chunks (JSON-deserialized dicts) | TB3 | HTTPS / SSE |
| DF6 | `langgraph.json` | C8 (CLI) | Config dict (graphs, env, store, auth, encryption, http, webhooks, checkpointer, ui) | TB4 | `json.load()` |
| DF7 | C8 (CLI) | Docker | Dockerfile content with embedded ENV values | TB4 | `asyncio.create_subprocess_exec` |
| DF8 | C11 (User Nodes) | C1 (Pregel) | State updates (arbitrary Python objects) | TB1 | Channel write |
| DF9 | C9 (SDK Client) | C4 (RemoteGraph) | API responses (JSON) | TB3 | HTTPS |
| DF10 | User config | C7 (EncryptedSerializer) | AES key from LANGGRAPH_AES_KEY env var | TB2 | `os.getenv()` |
| DF11 | C12 (Checkpoint Storage) | C14 (BaseCache) | Cached task results via JsonPlusSerializer | TB2 | Database read |

### Flow Details

#### DF1: Checkpoint Storage -> JsonPlusSerializer

- **Data**: Serialized graph state as `(type_tag, bytes)` tuples. Type tags include `"msgpack"`, `"json"`, `"pickle"`, `"bytes"`, `"null"`.
- **Validation**: Type tag dispatches to codec. Msgpack: `ext_hook` with allowlist check — `SAFE_MSGPACK_TYPES` always checked first, then `allowed_modules` determines behavior for unregistered types (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:514-559`). JSON: `_reviver` with `lc:2` module allowlist. Pickle: **no restrictions** (`pickle.loads(data_)` if `pickle_fallback=True`, `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:258`).
- **Trust assumption**: Checkpoint storage is access-controlled. An attacker with write access to the database can craft malicious checkpoint data.

#### DF3: LLM -> ToolNode

- **Data**: Tool call name and arguments from LLM-generated `AIMessage.tool_calls`.
- **Validation**: Tool name checked against registered `tools_by_name` dict — unknown names return error `ToolMessage` (`libs/prebuilt/langgraph/prebuilt/tool_node.py:1252`). Argument values validated only by the target tool's Pydantic schema.
- **Trust assumption**: LLM output is treated as untrusted for tool name routing but argument values pass through to tools without ToolNode-level sanitization.

#### DF5: RemoteGraph -> Pregel

- **Data**: Stream event chunks containing dicts for `Interrupt`, `Command`, state snapshots.
- **Validation**: **None** on inbound data. `Interrupt(**i)` and `Command(**chunk.data)` use dict-splatting with no schema check (`libs/langgraph/langgraph/pregel/remote.py:755,768,865,878`).
- **Trust assumption**: Remote server is trusted. A compromised or malicious server can inject arbitrary field values.

#### DF6: langgraph.json -> CLI

- **Data**: JSON config including `graphs`, `env`, `store`, `auth`, `encryption`, `http`, `webhooks`, `checkpointer`, `ui`, `ui_config` sections.
- **Validation**: Schema validation in `validate_config_file()` (`libs/cli/langgraph_cli/config.py:278`). Config values embedded in Dockerfile via `json.dumps()` in single-quoted `ENV` lines (`libs/cli/langgraph_cli/config.py:1009-1038`). Encryption config path format validated as `path/to/file.py:attribute_name` (`libs/cli/langgraph_cli/config.py:245-251`).
- **Trust assumption**: `langgraph.json` is developer-authored. Single quotes in config values could break Dockerfile `ENV` syntax.

#### DF11: Checkpoint Storage -> BaseCache

- **Data**: Cached task results stored via `BaseCache.set()` and retrieved via `BaseCache.get()`.
- **Validation**: Uses `JsonPlusSerializer(pickle_fallback=False)` by default (`libs/checkpoint/langgraph/cache/base/__init__.py:18`). Subject to same msgpack deserialization behavior as DF1 (allowed_modules defaults based on `LANGGRAPH_STRICT_MSGPACK`).
- **Trust assumption**: Cache storage has same access controls as checkpoint storage.

---

## Threats

| ID | Data Flow | Threat | Boundary | Severity | Status | Code Reference |
|----|-----------|--------|----------|----------|--------|----------------|
| T1 | DF1, DF11 | Arbitrary code execution via msgpack deserialization when strict mode is OFF (default) | TB2 | High | Unmitigated (default config) | `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:501-598` |
| T2 | DF1 | Arbitrary code execution via `pickle.loads` when `pickle_fallback=True` | TB2 | High | Mitigated (off by default) | `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:258` |
| T3 | DF1 | Arbitrary module import/execution via JSON `lc:2` constructor when `allowed_json_modules=True` | TB2 | High | Mitigated (blocked by default) | `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:142-226` |
| T4 | DF5 | Unvalidated dict-splatting from remote API into `Interrupt`/`Command` objects | TB3 | Medium | Unmitigated | `libs/langgraph/langgraph/pregel/remote.py:755,768,865,878` |
| T5 | DF6, DF7 | Dockerfile ENV injection via single-quote in `langgraph.json` config values | TB4 | Low | Unmitigated | `libs/cli/langgraph_cli/config.py:1009-1038` |
| T6 | DF7 | ZIP slip in `langgraph new` template extraction | TB4 | Low | Mitigated (hardcoded source URLs) | `libs/cli/langgraph_cli/templates.py:10-38` |
| T7 | DF10 | AES key entropy limited to printable characters via env var string encoding | TB2 | Info | Accepted | `libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:51-59` |
| T8 | DF10 | EncryptedSerializer cipher name check uses `assert` (stripped with `python -O`) | TB2 | Low | Unmitigated | `libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:72` |

### Threat Details

#### T1: Msgpack Deserialization RCE (Default Config)

- **Flow**: DF1 (Checkpoint Storage -> JsonPlusSerializer), DF11 (Checkpoint Storage -> BaseCache)
- **Description**: When `LANGGRAPH_STRICT_MSGPACK` is not set (the default), the msgpack `ext_hook` allows **any** `(module, class)` pair stored in checkpoint data to be imported via `importlib.import_module` and instantiated with attacker-controlled arguments. The `SAFE_MSGPACK_TYPES` allowlist is checked first, but unregistered types are logged as warnings and allowed through when `allowed_modules=True` (the default when strict mode is off). The `BaseCache` component uses `JsonPlusSerializer(pickle_fallback=False)` but inherits the same msgpack `allowed_modules` default behavior.
- **Preconditions**: Attacker must have write access to the checkpoint database (PostgreSQL or SQLite). This requires compromised database credentials or a co-located attacker.
- **Mitigations**: Setting `LANGGRAPH_STRICT_MSGPACK=true` enables the allowlist as a hard block. The `SAFE_MSGPACK_TYPES` frozenset restricts to ~40 known-safe types. `SAFE_MSGPACK_METHODS` further restricts method calls to a single allowed triple. Serde event hooks (`emit_serde_event`) allow monitoring of blocked/unregistered types. Deprecation warnings are emitted for unregistered types in default mode.
- **Residual risk**: Default installations are vulnerable. The deprecation-to-enforcement transition is incomplete. Historical advisories: GHSA-mhr3-j7m5-c7c9 (BaseCache deserialization RCE, CWE-502), GHSA-wwqv-p2pp-99h5 (JSON mode RCE, CWE-502). Past advisories also include GHSA-h477-2jr3-c5fc, GHSA-mc5m-mv86-88j6, GHSA-9rjh-j88v-42g9, GHSA-xjxx-5jjp-xg7c, GHSA-2f74-782f-8865 (all CWE-502).

#### T2: Pickle Deserialization RCE

- **Flow**: DF1 (Checkpoint Storage -> JsonPlusSerializer)
- **Description**: When `pickle_fallback=True` is explicitly passed to `JsonPlusSerializer`, checkpoint data with type tag `"pickle"` is deserialized via `pickle.loads()` with zero restrictions (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:258`).
- **Preconditions**: (1) Application or checkpointer explicitly enables `pickle_fallback=True`. (2) Attacker writes `("pickle", <payload>)` to checkpoint storage.
- **Mitigations**: `pickle_fallback` defaults to `False`. `BaseCache` explicitly sets `pickle_fallback=False`. Published advisory: GHSA-73ww-chjr-r8g8.
- **Residual risk**: Users who opt into pickle for backward compatibility are vulnerable if their storage is compromised.

#### T3: JSON lc:2 Constructor RCE

- **Flow**: DF1 (Checkpoint Storage -> JsonPlusSerializer)
- **Description**: The JSON `_reviver` handles `lc:2` type constructors by importing the module path from checkpoint JSON data via `importlib.import_module` (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:162-164`). If `allowed_json_modules=True` (explicit opt-in), any module reachable in the Python environment can be imported and instantiated.
- **Preconditions**: (1) `allowed_json_modules` set to `True` (not the default). (2) Attacker writes crafted JSON to checkpoint storage.
- **Mitigations**: Default is `None`, which blocks all `lc:2` constructors. When set to a frozenset, exact tuple matching is enforced via `_check_allowed_json_modules`. Published advisory: GHSA-wwqv-p2pp-99h5.
- **Residual risk**: Users who pass `allowed_json_modules=True` for convenience are fully exposed.

#### T4: RemoteGraph Unvalidated Inbound Data

- **Flow**: DF5 (RemoteGraph -> Pregel)
- **Description**: Stream events from the remote LangGraph Server are deserialized from JSON and dict-splatted into `Interrupt(**i)` and `Command(**chunk.data)` without schema validation at four callsites (`libs/langgraph/langgraph/pregel/remote.py:755,768,865,878`). A compromised or malicious remote server can inject unexpected fields. `Command.update` can carry arbitrary state modifications.
- **Preconditions**: User connects `RemoteGraph` to a compromised or attacker-controlled server URL.
- **Mitigations**: `_sanitize_config` sanitizes outbound data. Python dataclass constructors reject unexpected kwargs (TypeError). HTTPS transport provides network-layer integrity.
- **Residual risk**: A malicious server could supply valid but malicious field values (e.g., crafted `goto` targets or `update` payloads in `Command`) that alter graph execution flow.

#### T5: Dockerfile ENV Single-Quote Injection

- **Flow**: DF6, DF7 (langgraph.json -> CLI -> Dockerfile)
- **Description**: Config values from `langgraph.json` are serialized via `json.dumps()` and embedded in single-quoted `ENV` directives across multiple config sections (store, auth, encryption, http, webhooks, checkpointer, ui, ui_config, graphs). JSON does not escape single quotes, so a config value containing `'` could break the Dockerfile syntax or inject additional Dockerfile instructions. The pattern is duplicated in two Dockerfile generation functions (`libs/cli/langgraph_cli/config.py:1009-1038` and `libs/cli/langgraph_cli/config.py:1140-1169`).
- **Preconditions**: A `langgraph.json` config value contains a single quote character.
- **Mitigations**: `langgraph.json` is developer-controlled, not user-supplied at runtime. Active advisory: GHSA-22p4-fx53-2pwp.
- **Residual risk**: Minimal in normal use. Risk increases if `langgraph.json` is generated from untrusted input.

#### T6: ZIP Slip in Template Extraction

- **Flow**: DF7 (CLI template download)
- **Description**: `langgraph new` downloads a ZIP from GitHub and calls `zip_file.extractall(path)`. If the archive contains path-traversal entries (e.g., `../../etc/cron.d/exploit`), files could be written outside the target directory.
- **Preconditions**: The GitHub-hosted template archive must contain malicious path entries. This requires compromise of the upstream template repo.
- **Mitigations**: Template URLs are selected from a hardcoded `TEMPLATES` dict pointing to `langchain-ai` GitHub repos. Python 3.12+ `extractall` warns on path traversal; Python 3.14 raises by default.
- **Residual risk**: Very low given the controlled source, but defense-in-depth validation of archive member paths would be prudent.

#### T7: AES Key Entropy via Environment Variable

- **Flow**: DF10 (User config -> EncryptedSerializer)
- **Description**: The AES key is loaded from `LANGGRAPH_AES_KEY` as a UTF-8 string and `.encode()`d to bytes (`libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:57`). This limits key entropy to printable characters (~6.57 bits/byte vs. 8 bits/byte for random bytes), reducing effective key strength for AES-128 from 128 bits to ~105 bits.
- **Preconditions**: User relies on environment variable path for key loading (vs. passing raw bytes directly via `key=` parameter).
- **Mitigations**: Key length validation (16, 24, or 32 bytes required, `encrypted.py:58-59`). Users can pass raw `bytes` via the `key=` keyword argument to bypass the env var path.
- **Residual risk**: Informational. Key management hygiene concern, not exploitable in practice given AES-128 brute-force remains infeasible even at reduced entropy.

#### T8: EncryptedSerializer Assert Bypass

- **Flow**: DF10 (Encrypted checkpoint data)
- **Description**: The cipher name check in `decrypt()` uses `assert ciphername == "aes"` (`libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:72`), which is stripped when Python runs with `-O` (optimize) flag. The `ciphername` value comes from the type tag in checkpoint storage (split from the `type+cipher` format at `encrypted.py:32`).
- **Preconditions**: Python running with `-O` flag AND attacker can write to checkpoint storage.
- **Mitigations**: AES-EAX MAC verification (`decrypt_and_verify`) still validates ciphertext integrity (`encrypted.py:78`). Even if the assert is bypassed, a wrong cipher name would produce garbage that fails MAC verification.
- **Residual risk**: Defense-in-depth — should use `if/raise` instead of `assert` for security checks.

---

## Input Source Coverage

| Input Source | Data Flows | Threats | Validation Points | Responsibility | Gaps |
|-------------|-----------|---------|-------------------|----------------|------|
| User direct input (graph state, config) | DF8 | — | Graph structure validation (`pregel/_validate.py`), channel type enforcement | User | Node implementation safety is user's responsibility |
| LLM output (tool calls) | DF3, DF4 | — | Tool name allowlist (`tool_node.py:1252`), tool Pydantic schemas | Shared (project validates name; user validates args via tool schema) | No ToolNode-level argument sanitization |
| Checkpoint storage data | DF1, DF2, DF11 | T1, T2, T3 | Msgpack allowlist (`_msgpack.py:14-77`), msgpack method allowlist (`_msgpack.py:82-86`), JSON allowlist, pickle gating, serde event hooks | Shared (project owns serializer defaults; user owns DB access controls) | Default msgpack mode allows unregistered types |
| Remote API responses | DF5, DF9 | T4 | Outbound config sanitization; no inbound validation | User (user chooses which server to trust) | No inbound schema validation |
| Configuration (langgraph.json) | DF6, DF7 | T5 | JSON schema validation (`config.py:278`), encryption path format validation (`config.py:245-251`), list-based subprocess args | User (developer-controlled file) | Single-quote not escaped in ENV embedding |
| Configuration (env vars) | DF10 | T7, T8 | AES key length validation, EAX MAC verification | User (deployer controls env) | Key entropy, assert-based check |

---

## Out-of-Scope Threats

Threats that appear valid in isolation but fall outside project responsibility because they depend on conditions the project does not control.

| Pattern | Why Out of Scope | Project Responsibility Ends At |
|---------|-----------------|-------------------------------|
| Prompt injection leading to arbitrary tool execution | Project does not control LLM model behavior, user prompt construction, or which tools are registered. ToolNode routes by name only to user-registered tools. | Providing tool name allowlist routing (`libs/prebuilt/langgraph/prebuilt/tool_node.py:1252`); user owns tool registration and argument handling |
| State poisoning via malicious node output | User-registered nodes (including `@task`-decorated functions) can write arbitrary values to channels. The framework executes nodes as provided. | Enforcing channel type contracts (`libs/langgraph/langgraph/channels/base.py`); user owns node implementation correctness |
| Cross-session state access via thread_id guessing | Checkpoint savers index by `thread_id`. Without application-level auth, any caller with a valid thread_id can access that thread's state. | Providing the `Auth` handler system for access control (`libs/sdk-py/langgraph_sdk/auth/`); user must implement auth handlers. Past advisory: GHSA-65c8-xj34-43q4 (closed) |
| Tool shadowing via duplicate registration | If a user registers two tools with the same name, ToolNode uses the last one. This is user misconfiguration. | Documenting tool registration semantics. Past advisory: GHSA-393p-4cgj-rj9m (closed) |
| Indirect prompt injection via tool output | LLM reads tool output and may follow injected instructions. This is a fundamental LLM limitation, not a framework vulnerability. | Not including tool output in system prompts; user owns output handling |
| Model selecting dangerous tool arguments | An LLM may generate SQL injection, path traversal, or command injection payloads as tool arguments. The risk depends entirely on what the user's tools do with those arguments. | Routing tool calls to registered tools only; user owns tool input validation |
| RCE via user-provided node code | `add_node()` and `@entrypoint`/`@task` accept arbitrary callables. A malicious node can do anything. This is by design — the user controls their own code. | Executing nodes within the graph runtime; user owns node code safety |
| SSRF via RemoteGraph URL | User provides the `url` parameter to `RemoteGraph`. Pointing it at an internal service is the user's decision. | Documenting that `url` should be a trusted endpoint; user owns URL selection |

### Rationale

**Prompt injection and tool execution**: LangGraph's `ToolNode` validates tool names against the registered set but does not inspect or sanitize argument values. This is the correct boundary — the framework cannot know what constitutes a "safe" argument for an arbitrary user-defined tool. The tool's own Pydantic schema and implementation must validate inputs. The framework's responsibility is to not execute unregistered tools and to correctly route registered ones. See `libs/prebuilt/langgraph/prebuilt/tool_node.py:1252`.

**State integrity**: LangGraph channels enforce type contracts (e.g., `LastValue` accepts one value, `BinaryOperatorAggregate` applies a reducer). The framework validates graph structure at compile time (`libs/langgraph/langgraph/pregel/_validate.py`). However, the semantic correctness of state updates is the user's responsibility — the framework cannot know what values are "valid" for a user-defined state schema. The functional API (`@entrypoint`/`@task`) produces Pregel graphs with the same channel enforcement.

**Checkpoint access control**: The framework provides `BaseCheckpointSaver` as an abstract interface and the `Auth` handler system for authorization. It does not enforce authentication by default because it operates as a library, not a server. The `langgraph-api` server layer (out of scope) is responsible for enforcing auth on API endpoints. Users embedding LangGraph directly must implement their own access controls.

---

## External Context

### Published Security Advisories

| GHSA ID | Summary | CWEs | Relevance |
|---------|---------|------|-----------|
| GHSA-mhr3-j7m5-c7c9 | BaseCache Deserialization RCE | CWE-502 | Directly relates to T1 — msgpack deserialization in cache layer |
| GHSA-9rwj-6rc7-p77c | SQL injection via metadata filter key in SQLite checkpointer | CWE-89 | Fixed via `_validate_filter_key()` regex in `libs/checkpoint-sqlite/` |
| GHSA-wwqv-p2pp-99h5 | RCE in JSON mode of JsonPlusSerializer | CWE-502 | Directly relates to T3 — `lc:2` constructor import |
| GHSA-7p73-8jqx-23r8 | SQLite Filter Key SQL Injection in SqliteStore | CWE-89 | Fixed via `_validate_filter_key()` regex in `libs/checkpoint-sqlite/` |

**Pattern**: 3 of 4 published advisories involve CWE-502 (insecure deserialization) in the checkpoint serialization layer. This confirms the checkpoint storage boundary (TB2) as the highest-risk area. The SQLi advisories (CWE-89) in the SQLite layer have been remediated with regex-based filter key validation.

### Dependabot Alerts

No open Dependabot alerts.

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-03-04 | Generated by langster-threat-model | Initial threat model |
| 2026-03-04 | Updated by langster-threat-model | Added C13 (Functional API), C14 (BaseCache), DF11. Updated T1 for BaseCache/serde event hooks. Added GHSA-mhr3-j7m5-c7c9 and GHSA-9rwj-6rc7-p77c. Updated CLI config scope. Added External Context section. |
