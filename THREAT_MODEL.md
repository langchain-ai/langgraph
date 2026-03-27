# Threat Model: LangGraph

> Generated: 2026-03-27 | Commit: 7c40db3e | Scope: Full monorepo (all libs/) | Visibility: Open Source | Mode: Deep

For vulnerability reporting, see the [GitHub Security Advisories](https://github.com/langchain-ai/langgraph/security/advisories) page.

## Scope

### In Scope

- `libs/langgraph` — Core graph execution engine (Pregel, StateGraph, channels, functional API with `@entrypoint`/`@task`)
- `libs/prebuilt` — High-level agent APIs (ToolNode, create_react_agent, ValidationNode, ToolRuntime)
- `libs/checkpoint` — Checkpoint serialization/deserialization (JsonPlusSerializer, EncryptedSerializer, BaseCache, stores, serde event hooks)
- `libs/checkpoint-postgres` — PostgreSQL checkpoint saver, key-value store, and vector search
- `libs/checkpoint-sqlite` — SQLite checkpoint saver, key-value store, and vector search
- `libs/cli` — CLI for Docker-based deployment (`langgraph up/build/dev/new`)
- `libs/sdk-py` — Python SDK client for LangGraph Server API (HttpClient, Auth system)

### Out of Scope

- `libs/sdk-js` — Moved to external `langchain-ai/langgraphjs` repository; no source in this repo
- `libs/checkpoint-conformance` — Conformance test suite only; not shipped code
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
6. The SDK connects to trusted LangGraph Server endpoints chosen by the user.

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

| ID | Component | Description | Trust Level | Default? | Entry Points |
|----|-----------|-------------|-------------|----------|--------------|
| C1 | StateGraph / Pregel | Core graph builder and execution engine | framework-controlled | Yes | `StateGraph.add_node()`, `StateGraph.compile()`, `Pregel.invoke()`, `Pregel.stream()` |
| C2 | JsonPlusSerializer | Checkpoint serialization/deserialization with msgpack, JSON, and pickle codecs | framework-controlled | Yes | `loads_typed()`, `dumps_typed()`, `_create_msgpack_ext_hook()`, `_reviver()` |
| C3 | ToolNode | Dispatches LLM-generated tool calls to registered BaseTool instances | framework-controlled | Yes | `ToolNode._func()`, `_run_one()`, `_execute_tool_sync()`, `_validate_tool_call()` |
| C4 | RemoteGraph | Client for remote LangGraph Server API; implements PregelProtocol | framework-controlled | No (opt-in) | `RemoteGraph.stream()`, `RemoteGraph.invoke()`, `RemoteGraph.get_state()` |
| C5 | PostgresSaver / PostgresStore | PostgreSQL checkpoint saver, key-value store, and vector search | framework-controlled | No (opt-in) | `from_conn_string()`, `put()`, `get_tuple()`, `search()` |
| C6 | SqliteSaver / SqliteStore | SQLite checkpoint saver, key-value store with JSON path filtering | framework-controlled | No (opt-in) | `from_conn_string()`, `put()`, `get_tuple()`, `search()` |
| C7 | EncryptedSerializer | AES-EAX authenticated encryption wrapper for checkpoint data | framework-controlled | No (opt-in) | `from_pycryptodome_aes()`, `loads_typed()`, `dumps_typed()` |
| C8 | CLI (langgraph_cli) | Docker-based build and deployment tooling | framework-controlled | Yes | `langgraph up`, `langgraph build`, `langgraph dev`, `langgraph new` |
| C9 | SDK Client (langgraph_sdk) | HTTP client for LangGraph Server API with SSE streaming and reconnection | framework-controlled | Yes | `get_client()`, `get_sync_client()`, `HttpClient.request_reconnect()`, `HttpClient.stream()` |
| C10 | User-Registered Tools | BaseTool instances provided by users | user-controlled | N/A | Tool `invoke()` / `ainvoke()` methods |
| C11 | User-Registered Nodes | Arbitrary callables added via `add_node()` or `@task`/`@entrypoint` | user-controlled | N/A | Node function signatures |
| C12 | Checkpoint Storage | PostgreSQL or SQLite databases storing serialized graph state | external | N/A | Database connection interface |
| C13 | Functional API | `@entrypoint`/`@task` decorators for function-based workflow authoring | framework-controlled | Yes | `entrypoint.__call__()`, `task()`, `_TaskFunction.__call__()` (`libs/langgraph/langgraph/func/__init__.py`) |
| C14 | BaseCache | Cache layer for task results with JsonPlusSerializer (pickle_fallback=False) | framework-controlled | No (opt-in, requires checkpointer) | `get()`, `set()`, `clear()` (`libs/checkpoint/langgraph/cache/base/__init__.py`) |
| C15 | Serde Event Hooks | Monitoring system for serialization/deserialization events | framework-controlled | Yes | `register_serde_event_listener()`, `emit_serde_event()` (`libs/checkpoint/langgraph/checkpoint/serde/event_hooks.py`) |
| C16 | Auth System (SDK) | Custom authentication/authorization handler framework | framework-controlled | No (opt-in) | `Auth.authenticate()`, `Auth.on()` handler registration (`libs/sdk-py/langgraph_sdk/auth/__init__.py`) |

---

## Data Classification

| ID | PII Category | Specific Fields | Sensitivity | Storage Location(s) | Encrypted at Rest | Retention | Regulatory |
|----|-------------|----------------|-------------|---------------------|-------------------|-----------|------------|
| DC1 | API credentials | `x-api-key` header, `LANGGRAPH_API_KEY`, `LANGSMITH_API_KEY`, `LANGCHAIN_API_KEY` env vars | Critical | Environment variables, HTTP headers in transit | N/A (in-memory) | Session lifetime | All — breach trigger |
| DC2 | Encryption keys | `LANGGRAPH_AES_KEY` env var, `key` parameter to `from_pycryptodome_aes()` | Critical | Environment variable, in-memory | N/A | Application lifetime | All — breach trigger |
| DC3 | Serialized graph state | Checkpoint data in `checkpoints` and `writes` tables (msgpack/JSON/pickle bytes) | High | PostgreSQL (BYTEA), SQLite (BLOB) | Optional via EncryptedSerializer | Unbounded (no default TTL) | GDPR if state contains PII |
| DC4 | Store key-value data | User-stored items in `store` tables via BaseStore | High | PostgreSQL, SQLite | No (plaintext JSON) | Configurable TTL, default unbounded | GDPR if contains PII |
| DC5 | Checkpoint metadata | `thread_id`, `checkpoint_ns`, `run_id`, `step`, `source` | Medium | PostgreSQL, SQLite (metadata JSONB/JSON column) | No | Same as DC3 | Minimal |
| DC6 | Agent conversation history | LangChain messages (HumanMessage, AIMessage, ToolMessage) serialized in checkpoint state | High | PostgreSQL, SQLite (within DC3 checkpoint bytes) | Only if DC3 encrypted | Unbounded | GDPR, CCPA if contains user PII |
| DC7 | Connection strings | PostgreSQL URIs, SQLite file paths passed to `from_conn_string()` | Critical | Application code, environment variables | N/A (in-memory) | Application lifetime | All — may contain credentials |
| DC8 | Vector embeddings | Document embeddings in `store_vectors` table | Low | PostgreSQL (pgvector), SQLite (vec extension) | No | Same as DC4 | Minimal |

### Data Classification Details

#### DC1: API Credentials

- **Fields**: `x-api-key` HTTP header, `LANGGRAPH_API_KEY`/`LANGSMITH_API_KEY`/`LANGCHAIN_API_KEY` environment variables
- **Storage**: Environment variables (loaded at runtime), HTTP request headers (in transit)
- **Access**: SDK client code (`libs/sdk-py/langgraph_sdk/_shared/utilities.py:_get_api_key`), any process with env var access
- **Encryption**: TLS in transit (if HTTPS); no at-rest encryption for env vars
- **Retention**: Session/process lifetime
- **Logging exposure**: API key stripped of quotes but could appear in debug logs if HTTP headers are logged. `RESERVED_HEADERS` prevents user override of `x-api-key` but doesn't prevent logging.
- **Cross-border**: Travels with every HTTP request to the LangGraph Server
- **Gaps**: SDK `request_reconnect()` and `stream()` forward `x-api-key` header to server-controlled `Location` redirect URLs without URL validation (see T9)

#### DC2: Encryption Keys

- **Fields**: `LANGGRAPH_AES_KEY` environment variable, `key` bytes parameter
- **Storage**: Environment variable or direct bytes in application code
- **Access**: `libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:from_pycryptodome_aes`
- **Encryption**: N/A — this IS the encryption key
- **Retention**: Application lifetime
- **Logging exposure**: Not logged by framework code
- **Gaps**: Key loaded from env var as UTF-8 string limits entropy to ~6.57 bits/byte (see T7). Cipher name validated with `assert` which is stripped by `python -O` (see T8).

#### DC3: Serialized Graph State

- **Fields**: All channel values serialized via `JsonPlusSerializer.dumps_typed()` — includes complete agent state, conversation history, tool call results, and any user-defined state
- **Storage**: PostgreSQL `checkpoints.checkpoint` (BYTEA), `writes.blob` (BYTEA); SQLite `checkpoints.checkpoint` (BLOB), `writes.blob` (BLOB)
- **Access**: Any code with database credentials; `BaseCheckpointSaver.get_tuple()` / `put()`
- **Encryption**: Optional via `EncryptedSerializer` wrapping (AES-EAX). Not encrypted by default.
- **Retention**: Unbounded by default. Optional TTL via `CheckpointerConfig.ttl` (server-side config)
- **Logging exposure**: Serde event hooks emit module/class names of deserialized types but not the data itself
- **Gaps**: Default unbounded retention of potentially PII-containing state. Unencrypted by default. EncryptedSerializer has fallback that accepts unencrypted data (see T10).

#### DC6: Agent Conversation History

- **Fields**: `HumanMessage.content`, `AIMessage.content`, `ToolMessage.content`, `AIMessage.tool_calls` — embedded within DC3 checkpoint bytes
- **Storage**: Same as DC3 (within serialized checkpoint data)
- **Access**: Same as DC3
- **Encryption**: Only if DC3 is encrypted via EncryptedSerializer
- **Retention**: Same as DC3 (unbounded default)
- **Gaps**: Conversation content may include user PII, PHI, or sensitive business data. No field-level encryption or redaction. Retention inherits from DC3 with no conversation-specific policy.

---

## Trust Boundaries

| ID | Boundary | Description | Controls (Inside) | Does NOT Control (Outside) |
|----|----------|-------------|-------------------|---------------------------|
| TB1 | User/Framework API | Where user-provided code and configuration enters the framework | Graph execution logic, channel semantics, default configs, validation of graph structure | User node implementations, tool behavior, model selection, prompt construction, state schema design |
| TB2 | Checkpoint Storage | Where serialized data enters/leaves the persistence layer | Serialization format, allowlists for deserialization, encryption (if configured), serde event hooks | Database access controls, who can write to the checkpoint tables, storage infrastructure security |
| TB3 | Remote API | Where data crosses the network to/from LangGraph Server | Outbound config sanitization (`_sanitize_config`), SDK HTTP transport, API key handling, `RESERVED_HEADERS` | Remote server behavior, response content integrity, network security (TLS), server-provided Location redirect targets |
| TB4 | CLI Config/Docker | Where developer config drives container image generation | Dockerfile template structure, config schema validation, list-based subprocess args, build command content validation | `langgraph.json` file content, Docker daemon security, host filesystem |

### Boundary Details

#### TB1: User/Framework API

- **Inside**: Graph compilation validates structure (`libs/langgraph/langgraph/pregel/_validate.py:validate_graph`). Channel types enforce update semantics (`libs/langgraph/langgraph/channels/base.py:BaseChannel.update`). Functional API validates entrypoint has at least one parameter (`libs/langgraph/langgraph/func/__init__.py:entrypoint`). Sensitive config keys filtered from metadata propagation — keys containing "key", "token", "secret", "password", "auth" are excluded (`libs/langgraph/langgraph/_internal/_config.py:_exclude_as_metadata`).
- **Outside**: What user nodes do, what tools return, what LLMs generate, how users handle output.
- **Crossing mechanism**: Python function calls — `add_node(callable)`, `add_edge()`, `compile(checkpointer=...)`, `@entrypoint`, `@task`.

#### TB2: Checkpoint Storage

- **Inside**: `JsonPlusSerializer` controls serialization format (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer`). Msgpack type allowlist (`libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py:SAFE_MSGPACK_TYPES` — 40 safe types). Msgpack method allowlist (`libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py:SAFE_MSGPACK_METHODS` — 1 safe method). JSON module allowlist (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:_check_allowed_json_modules`). Serde event hooks for monitoring (`libs/checkpoint/langgraph/checkpoint/serde/event_hooks.py:emit_serde_event`). Optional `EncryptedSerializer` wrapping (`libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:EncryptedSerializer`). SQLite filter key regex validation (`libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/utils.py:_validate_filter_key`). Parameterized SQL queries in both Postgres and SQLite backends.
- **Outside**: Database access controls, who can read/write checkpoint tables, storage backend integrity.
- **Crossing mechanism**: Database read/write operations — serialized bytes stored as BYTEA (Postgres) or BLOB (SQLite).

#### TB3: Remote API

- **Inside**: `_sanitize_config()` strips non-primitive values and drops checkpoint-internal keys from outbound config (`libs/langgraph/langgraph/pregel/remote.py:_sanitize_config`). SDK handles API key from env vars (`libs/sdk-py/langgraph_sdk/_shared/utilities.py:_get_api_key`). `RESERVED_HEADERS` prevents user override of `x-api-key` (`libs/sdk-py/langgraph_sdk/_shared/utilities.py:RESERVED_HEADERS`).
- **Outside**: Remote server response content, network integrity, whether the server is legitimate, server-provided Location redirect targets.
- **Crossing mechanism**: HTTPS requests via `httpx` through `langgraph_sdk`.

#### TB4: CLI Config/Docker

- **Inside**: Config file parsed as JSON (`libs/cli/langgraph_cli/config.py:validate_config_file`). Docker subprocess invoked with list-based args via `asyncio.create_subprocess_exec`, not `shell=True` (`libs/cli/langgraph_cli/exec.py:subp_exec`). Template downloads from hardcoded GitHub URLs (`libs/cli/langgraph_cli/templates.py`). Config schema validation covers store, auth, encryption, http, webhooks, checkpointer, and ui sections (`libs/cli/langgraph_cli/schemas.py`). Build command content validation blocks shell metacharacters (`libs/cli/langgraph_cli/config.py:has_disallowed_build_command_content`).
- **Outside**: Content of `langgraph.json`, Docker daemon behavior, filesystem permissions.
- **Crossing mechanism**: JSON file read, subprocess execution, ZIP download/extraction.

---

## Data Flows

| ID | Source | Destination | Data Type | Classification | Crosses Boundary | Protocol |
|----|--------|-------------|-----------|----------------|------------------|----------|
| DF1 | C12 (Checkpoint Storage) | C2 (JsonPlusSerializer) | Serialized checkpoint bytes (msgpack/JSON/pickle) | DC3 | TB2 | Database read |
| DF2 | C2 (JsonPlusSerializer) | C1 (Pregel) | Deserialized Python objects (channel state) | DC3, DC6 | TB2 | Function call |
| DF3 | LLM (external) | C3 (ToolNode) | Tool call arguments (JSON strings in AIMessage) | — | TB1 | Function call (via langchain-core) |
| DF4 | C3 (ToolNode) | C10 (User Tools) | Parsed argument dicts | — | TB1 | `tool.invoke(call_args)` |
| DF5 | C4 (RemoteGraph) | C1 (Pregel) | Stream chunks (JSON-deserialized dicts) | — | TB3 | HTTPS / SSE |
| DF6 | `langgraph.json` | C8 (CLI) | Config dict (graphs, env, store, auth, encryption, http, webhooks, checkpointer, ui) | — | TB4 | `json.load()` |
| DF7 | C8 (CLI) | Docker | Dockerfile content with embedded ENV values | — | TB4 | `asyncio.create_subprocess_exec` |
| DF8 | C11 (User Nodes) | C1 (Pregel) | State updates (arbitrary Python objects) | — | TB1 | Channel write |
| DF9 | C9 (SDK Client) | C4 (RemoteGraph) | API responses (JSON) | — | TB3 | HTTPS |
| DF10 | User config | C7 (EncryptedSerializer) | AES key from LANGGRAPH_AES_KEY env var | DC2 | TB2 | `os.getenv()` |
| DF11 | C12 (Checkpoint Storage) | C14 (BaseCache) | Cached task results via JsonPlusSerializer | DC3 | TB2 | Database read |
| DF12 | LangGraph Server | C9 (SDK Client) | HTTP responses with Location header | DC1 | TB3 | HTTP redirect |
| DF13 | C9 (SDK Client) | Redirect target | Request headers including x-api-key | DC1 | TB3 | HTTPS |

### Flow Details

#### DF1: Checkpoint Storage -> JsonPlusSerializer

- **Data**: Serialized graph state as `(type_tag, bytes)` tuples. Type tags include `"msgpack"`, `"json"`, `"pickle"`, `"bytes"`, `"null"`. When encrypted: `"msgpack+aes"`, `"json+aes"`.
- **Validation**: Type tag dispatches to codec. Msgpack: `_create_msgpack_ext_hook` with allowlist check — `SAFE_MSGPACK_TYPES` always checked first, then `allowed_modules` determines behavior for unregistered types (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:_create_msgpack_ext_hook`). JSON: `_reviver` with `lc:2` module allowlist. Pickle: **no restrictions** (`pickle.loads(data_)` if `pickle_fallback=True`, `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer.loads_typed`).
- **Trust assumption**: Checkpoint storage is access-controlled. An attacker with write access to the database can craft malicious checkpoint data.

#### DF3: LLM -> ToolNode

- **Data**: Tool call name and arguments from LLM-generated `AIMessage.tool_calls`.
- **Validation**: Tool name checked against registered `tools_by_name` dict — unknown names return error `ToolMessage` (`libs/prebuilt/langgraph/prebuilt/tool_node.py:ToolNode._validate_tool_call`). Argument values validated only by the target tool's Pydantic schema.
- **Trust assumption**: LLM output is treated as untrusted for tool name routing but argument values pass through to tools without ToolNode-level sanitization.

#### DF5: RemoteGraph -> Pregel

- **Data**: Stream event chunks containing dicts for `Interrupt`, `Command`, state snapshots.
- **Validation**: **None** on inbound data. `Interrupt(**i)` uses dict-splatting with no schema check (`libs/langgraph/langgraph/pregel/remote.py:RemoteGraph.stream`). `Command(**chunk.data)` uses dict-splatting for parent commands.
- **Trust assumption**: Remote server is trusted. A compromised or malicious server can inject arbitrary field values.

#### DF6: langgraph.json -> CLI

- **Data**: JSON config including `graphs`, `env`, `store`, `auth`, `encryption`, `http`, `webhooks`, `checkpointer`, `ui`, `ui_config` sections.
- **Validation**: Schema validation in `validate_config_file()` (`libs/cli/langgraph_cli/config.py:validate_config_file`). Config values embedded in Dockerfile via `json.dumps()` in single-quoted `ENV` lines (`libs/cli/langgraph_cli/config.py:python_config_to_docker`). Build command content validation (`libs/cli/langgraph_cli/config.py:has_disallowed_build_command_content`) blocks shell metacharacters.
- **Trust assumption**: `langgraph.json` is developer-authored. Single quotes in config values could break Dockerfile `ENV` syntax.

#### DF11: Checkpoint Storage -> BaseCache

- **Data**: Cached task results stored via `BaseCache.set()` and retrieved via `BaseCache.get()`.
- **Validation**: Uses `JsonPlusSerializer(pickle_fallback=False)` by default (`libs/checkpoint/langgraph/cache/base/__init__.py:BaseCache`). Subject to same msgpack deserialization behavior as DF1 (allowed_modules defaults based on `LANGGRAPH_STRICT_MSGPACK`).
- **Trust assumption**: Cache storage has same access controls as checkpoint storage.

#### DF12-13: Server -> SDK -> Redirect Target (API Key Leak)

- **Data**: Server provides `Location` header in HTTP response. SDK follows the redirect and sends all original request headers (including `x-api-key`) to the target URL.
- **Validation**: **None** on Location URL. No allowlist, no same-origin check, no header stripping on cross-origin redirect.
- **Trust assumption**: The LangGraph Server is trusted to not redirect to malicious URLs. Violated if server is compromised.

---

## Threats

| ID | Data Flow | Classification | Threat | Boundary | Severity | Status | Validation | Code Reference |
|----|-----------|----------------|--------|----------|----------|--------|------------|----------------|
| T1 | DF1, DF11 | DC3 | Arbitrary code execution via msgpack deserialization when strict mode is OFF (default) | TB2 | High | Unmitigated (default config) | Verified | `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:_create_msgpack_ext_hook` |
| T2 | DF1 | DC3 | Arbitrary code execution via `pickle.loads` when `pickle_fallback=True` | TB2 | High | Mitigated (off by default) | Verified | `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer.loads_typed` |
| T3 | DF1 | DC3 | Arbitrary module import/execution via JSON `lc:2` constructor when `allowed_json_modules=True` | TB2 | High | Mitigated (blocked by default) | Verified | `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer._revive_lc2` |
| T4 | DF5 | — | Unvalidated dict-splatting from remote API into `Interrupt`/`Command` objects | TB3 | Medium | Unmitigated | Likely | `libs/langgraph/langgraph/pregel/remote.py:RemoteGraph.stream` |
| T5 | DF6, DF7 | — | Dockerfile ENV injection via single-quote in `langgraph.json` config values | TB4 | Low | Unmitigated | Likely | `libs/cli/langgraph_cli/config.py:python_config_to_docker` |
| T6 | DF7 | — | ZIP slip in `langgraph new` template extraction | TB4 | Low | Mitigated (hardcoded source URLs) | Unverified | `libs/cli/langgraph_cli/templates.py:_download_repo_with_requests` |
| T7 | DF10 | DC2 | AES key entropy limited to printable characters via env var string encoding | TB2 | Info | Accepted | — | `libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:EncryptedSerializer.from_pycryptodome_aes` |
| T8 | DF10 | DC2 | EncryptedSerializer cipher name check uses `assert` (stripped with `python -O`) | TB2 | Low | Unmitigated | Verified | `libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:PycryptodomeAesCipher.decrypt` |
| T9 | DF12, DF13 | DC1 | SDK API key leak via server-controlled Location redirect to attacker-controlled URL | TB3 | Medium | Unmitigated | Verified | `libs/sdk-py/langgraph_sdk/_async/http.py:HttpClient.request_reconnect`, `libs/sdk-py/langgraph_sdk/_async/http.py:HttpClient.stream` |
| T10 | DF1 | DC3 | EncryptedSerializer silently accepts unencrypted data — attacker bypasses encryption by writing plain type tags | TB2 | Medium | Unmitigated | Verified | `libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:EncryptedSerializer.loads_typed` |
| T11 | DF1, DF11 | DC3, DC6 | Unbounded retention of checkpoint data containing PII/conversation history | TB2 | Medium | Accepted (user responsibility) | — | `libs/checkpoint/langgraph/checkpoint/base/__init__.py:BaseCheckpointSaver` |

### Threat Details

#### T1: Msgpack Deserialization RCE (Default Config)

- **Flow**: DF1 (Checkpoint Storage -> JsonPlusSerializer), DF11 (Checkpoint Storage -> BaseCache)
- **Description**: When `LANGGRAPH_STRICT_MSGPACK` is not set (the default), the msgpack `_create_msgpack_ext_hook` allows **any** `(module, class)` pair stored in checkpoint data to be imported via `importlib.import_module` and instantiated with attacker-controlled arguments. The `SAFE_MSGPACK_TYPES` allowlist is checked first, but unregistered types are logged as warnings and allowed through when `allowed_modules=True` (the default when strict mode is off). Seven EXT codes are processed: `EXT_CONSTRUCTOR_SINGLE_ARG` (0), `EXT_CONSTRUCTOR_POS_ARGS` (1), `EXT_CONSTRUCTOR_KW_ARGS` (2), `EXT_METHOD_SINGLE_ARG` (3), `EXT_PYDANTIC_V1` (4), `EXT_PYDANTIC_V2` (5), `EXT_NUMPY_ARRAY` (6). The `BaseCache` component uses `JsonPlusSerializer(pickle_fallback=False)` but inherits the same msgpack `allowed_modules` default behavior.
- **Preconditions**: Attacker must have write access to the checkpoint database (PostgreSQL or SQLite). This requires compromised database credentials or a co-located attacker.
- **Mitigations**: Setting `LANGGRAPH_STRICT_MSGPACK=true` enables the allowlist as a hard block. The `SAFE_MSGPACK_TYPES` frozenset restricts to 40 known-safe types. `SAFE_MSGPACK_METHODS` restricts method calls to a single allowed triple (`datetime.datetime.fromisoformat`). Serde event hooks (`libs/checkpoint/langgraph/checkpoint/serde/event_hooks.py:emit_serde_event`) allow monitoring of blocked/unregistered types. Deprecation warnings are emitted for unregistered types in default mode.
- **Residual risk**: Default installations are vulnerable. The deprecation-to-enforcement transition is incomplete. Historical advisories: GHSA-mhr3-j7m5-c7c9 (BaseCache deserialization RCE, CWE-502), GHSA-wwqv-p2pp-99h5 (JSON mode RCE, CWE-502), GHSA-g48c-2wqr-h844 (unsafe msgpack deserialization, CWE-502). Extensive closed advisory history (~15 deserialization bypass attempts) confirms this as the primary attack surface.

#### T2: Pickle Deserialization RCE

- **Flow**: DF1 (Checkpoint Storage -> JsonPlusSerializer)
- **Description**: When `pickle_fallback=True` is explicitly passed to `JsonPlusSerializer`, checkpoint data with type tag `"pickle"` is deserialized via `pickle.loads()` with zero restrictions (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer.loads_typed`).
- **Preconditions**: (1) Application or checkpointer explicitly enables `pickle_fallback=True`. (2) Attacker writes `("pickle", <payload>)` to checkpoint storage.
- **Mitigations**: `pickle_fallback` defaults to `False`. `BaseCache` explicitly sets `pickle_fallback=False`. Published advisory: GHSA-73ww-chjr-r8g8.
- **Residual risk**: Users who opt into pickle for backward compatibility are vulnerable if their storage is compromised.

#### T3: JSON lc:2 Constructor RCE

- **Flow**: DF1 (Checkpoint Storage -> JsonPlusSerializer)
- **Description**: The JSON `_reviver` handles `lc:2` type constructors by importing the module path from checkpoint JSON data via `importlib.import_module` (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer._revive_lc2`). If `allowed_json_modules=True` (explicit opt-in), any module reachable in the Python environment can be imported and instantiated. The method also supports method chaining — a `method` key in the JSON can call arbitrary methods on the imported class.
- **Preconditions**: (1) `allowed_json_modules` set to `True` (not the default). (2) Attacker writes crafted JSON to checkpoint storage.
- **Mitigations**: Default is `None`, which blocks all `lc:2` constructors. When set to a frozenset, exact tuple matching is enforced via `_check_allowed_json_modules`. Published advisory: GHSA-wwqv-p2pp-99h5.
- **Residual risk**: Users who pass `allowed_json_modules=True` for convenience are fully exposed.

#### T4: RemoteGraph Unvalidated Inbound Data

- **Flow**: DF5 (RemoteGraph -> Pregel)
- **Description**: Stream events from the remote LangGraph Server are deserialized from JSON and dict-splatted into `Interrupt(**i)` and `Command(**chunk.data)` without schema validation (`libs/langgraph/langgraph/pregel/remote.py:RemoteGraph.stream`). A compromised or malicious remote server can inject unexpected fields. `Command.update` can carry arbitrary state modifications; `Command.goto` can alter graph execution flow.
- **Preconditions**: User connects `RemoteGraph` to a compromised or attacker-controlled server URL.
- **Mitigations**: `_sanitize_config` sanitizes outbound data. Python dataclass constructors reject unexpected kwargs (TypeError). HTTPS transport provides network-layer integrity.
- **Residual risk**: A malicious server could supply valid but malicious field values (e.g., crafted `goto` targets or `update` payloads in `Command`) that alter graph execution flow.

#### T5: Dockerfile ENV Single-Quote Injection

- **Flow**: DF6, DF7 (langgraph.json -> CLI -> Dockerfile)
- **Description**: Config values from `langgraph.json` are serialized via `json.dumps()` and embedded in single-quoted `ENV` directives across multiple config sections (store, auth, encryption, http, webhooks, checkpointer, ui, ui_config, graphs). JSON does not escape single quotes, so a config value containing `'` could break the Dockerfile syntax or inject additional Dockerfile instructions. The pattern is duplicated in two Dockerfile generation functions (`libs/cli/langgraph_cli/config.py:python_config_to_docker` and `libs/cli/langgraph_cli/config.py:node_config_to_docker`).
- **Preconditions**: A `langgraph.json` config value contains a single quote character.
- **Mitigations**: `langgraph.json` is developer-controlled, not user-supplied at runtime. `has_disallowed_build_command_content()` blocks shell metacharacters in `dockerfile_lines` but does NOT validate JSON config values embedded in ENV. Active advisory: GHSA-22p4-fx53-2pwp.
- **Residual risk**: Minimal in normal use. Risk increases if `langgraph.json` is generated from untrusted input.

#### T6: ZIP Slip in Template Extraction

- **Flow**: DF7 (CLI template download)
- **Description**: `langgraph new` downloads a ZIP from GitHub and uses `zip_file.extractall(path)`. If the archive contains path-traversal entries (e.g., `../../etc/cron.d/exploit`), files could be written outside the target directory.
- **Preconditions**: The GitHub-hosted template archive must contain malicious path entries. This requires compromise of the upstream template repo.
- **Mitigations**: Template URLs are selected from a hardcoded `TEMPLATES` dict pointing to `langchain-ai` GitHub repos (`libs/cli/langgraph_cli/templates.py`). Python 3.12+ `extractall` warns on path traversal; Python 3.14 raises by default.
- **Residual risk**: Very low given the controlled source, but defense-in-depth validation of archive member paths would be prudent.

#### T7: AES Key Entropy via Environment Variable

- **Flow**: DF10 (User config -> EncryptedSerializer)
- **Description**: The AES key is loaded from `LANGGRAPH_AES_KEY` as a UTF-8 string and `.encode()`d to bytes (`libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:EncryptedSerializer.from_pycryptodome_aes`). This limits key entropy to printable characters (~6.57 bits/byte vs. 8 bits/byte for random bytes), reducing effective key strength for AES-128 from 128 bits to ~105 bits.
- **Preconditions**: User relies on environment variable path for key loading (vs. passing raw bytes directly via `key=` parameter).
- **Mitigations**: Key length validation (16, 24, or 32 bytes required). Users can pass raw `bytes` via the `key=` keyword argument to bypass the env var path.
- **Residual risk**: Informational. Key management hygiene concern, not exploitable in practice given AES-128 brute-force remains infeasible even at reduced entropy.

#### T8: EncryptedSerializer Assert Bypass

- **Flow**: DF10 (Encrypted checkpoint data)
- **Description**: The cipher name check in `decrypt()` uses `assert ciphername == "aes"` (`libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:PycryptodomeAesCipher.decrypt`), which is stripped when Python runs with `-O` (optimize) flag. The `ciphername` value comes from the type tag in checkpoint storage (split from the `type+cipher` format).
- **Preconditions**: Python running with `-O` flag AND attacker can write to checkpoint storage.
- **Mitigations**: AES-EAX MAC verification (`decrypt_and_verify`) still validates ciphertext integrity. Even if the assert is bypassed, a wrong cipher name would produce garbage that fails MAC verification.
- **Residual risk**: Defense-in-depth — should use `if/raise` instead of `assert` for security checks.

#### T9: SDK API Key Leak via Server-Controlled Location Redirect

- **Flow**: DF12 (Server -> SDK), DF13 (SDK -> Redirect target)
- **Description**: The SDK's `HttpClient.request_reconnect()` (`libs/sdk-py/langgraph_sdk/_async/http.py:HttpClient.request_reconnect`) follows server-provided `Location` headers and forwards the full `request_headers` dict (including the `x-api-key` authentication header) to the redirected URL. The `HttpClient.stream()` method (`libs/sdk-py/langgraph_sdk/_async/http.py:HttpClient.stream`) also follows `Location` headers for SSE reconnection and forwards `reconnect_headers` (which include `x-api-key`) to the server-controlled URL. No URL validation, same-origin check, or sensitive header stripping is performed before following the redirect. The same pattern exists in the sync client (`libs/sdk-py/langgraph_sdk/_sync/http.py`).
- **Preconditions**: (1) User connects SDK to a LangGraph Server that is compromised or attacker-controlled. (2) The server returns a response with a `Location` header pointing to an attacker-controlled URL.
- **Mitigations**: `RESERVED_HEADERS` prevents user override of `x-api-key` header but does not prevent it from being forwarded to redirect targets. HTTPS provides transport integrity for the initial connection.
- **Residual risk**: API key exfiltration if the server is compromised. The leaked key could be used for impersonation or access to LangSmith/LangGraph services.

#### T10: EncryptedSerializer Encryption Bypass via Unencrypted Data Injection

- **Flow**: DF1 (Checkpoint Storage -> EncryptedSerializer)
- **Description**: `EncryptedSerializer.loads_typed()` (`libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:EncryptedSerializer.loads_typed`) checks if the type tag contains a `+` delimiter. If it does not (e.g., type tag is `"msgpack"` instead of `"msgpack+aes"`), the data is passed directly to the inner serde's `loads_typed()` **without any decryption or MAC verification**. An attacker with write access to checkpoint storage can bypass the encryption layer entirely by writing data with a plain type tag.
- **Preconditions**: (1) Application uses `EncryptedSerializer` for checkpoint protection. (2) Attacker has write access to checkpoint storage.
- **Mitigations**: The inner serde's deserialization allowlists (SAFE_MSGPACK_TYPES, etc.) still apply to the unencrypted data. The encryption bypass does not directly enable RCE unless msgpack strict mode is also off (combining T10 + T1).
- **Residual risk**: Defeats the purpose of encryption for checkpoint data confidentiality and integrity. An attacker who can write to the database can inject unencrypted payloads that bypass MAC verification. If `LANGGRAPH_STRICT_MSGPACK` is off (default), this chains with T1 for full RCE.

#### T11: Unbounded Checkpoint Data Retention

- **Flow**: DF1, DF11 (Checkpoint Storage lifecycle)
- **Description**: Checkpoint data (DC3, DC6) is retained indefinitely by default. No built-in TTL, pruning, or data lifecycle management in the library-level checkpoint savers. Conversation history containing user PII may accumulate without bounds.
- **Preconditions**: Application uses checkpointing (the primary use case). No explicit cleanup configured.
- **Mitigations**: Optional TTL configuration available via `CheckpointerConfig.ttl` in server-side configuration. `BaseCheckpointSaver.prune()` method available but not called automatically. Users must implement their own cleanup.
- **Residual risk**: Compliance risk (GDPR right to erasure, CCPA deletion requests) for applications storing PII in agent state. Accepted as user responsibility since LangGraph is a library.

---

## Input Source Coverage

| Input Source | Data Flows | Threats | Validation Points | Responsibility | Gaps |
|-------------|-----------|---------|-------------------|----------------|------|
| User direct input (graph state, config) | DF8 | — | Graph structure validation (`libs/langgraph/langgraph/pregel/_validate.py:validate_graph`), channel type enforcement (`libs/langgraph/langgraph/channels/base.py:BaseChannel`), sensitive key filtering (`libs/langgraph/langgraph/_internal/_config.py:_exclude_as_metadata`) | User | Node implementation safety is user's responsibility |
| LLM output (tool calls) | DF3, DF4 | — | Tool name allowlist (`libs/prebuilt/langgraph/prebuilt/tool_node.py:ToolNode._validate_tool_call`), tool Pydantic schemas | Shared (project validates name; user validates args via tool schema) | No ToolNode-level argument sanitization |
| Checkpoint storage data | DF1, DF2, DF11 | T1, T2, T3, T10 | Msgpack allowlist (`libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py:SAFE_MSGPACK_TYPES`), msgpack method allowlist (`libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py:SAFE_MSGPACK_METHODS`), JSON allowlist (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:_check_allowed_json_modules`), pickle gating, serde event hooks, optional encryption | Shared (project owns serializer defaults; user owns DB access controls) | Default msgpack mode allows unregistered types; EncryptedSerializer accepts unencrypted data |
| Remote API responses | DF5, DF9, DF12, DF13 | T4, T9 | Outbound config sanitization (`libs/langgraph/langgraph/pregel/remote.py:_sanitize_config`); no inbound validation; no redirect URL validation | User (user chooses which server to trust) | No inbound schema validation; API key forwarded on redirects |
| Configuration (langgraph.json) | DF6, DF7 | T5 | JSON schema validation (`libs/cli/langgraph_cli/config.py:validate_config_file`), build command content validation (`libs/cli/langgraph_cli/config.py:has_disallowed_build_command_content`), list-based subprocess args | User (developer-controlled file) | Single-quote not escaped in ENV embedding |
| Configuration (env vars) | DF10 | T7, T8 | AES key length validation, EAX MAC verification | User (deployer controls env) | Key entropy, assert-based check |

---

## Out-of-Scope Threats

Threats that appear valid in isolation but fall outside project responsibility because they depend on conditions the project does not control.

| Pattern | Why Out of Scope | Project Responsibility Ends At |
|---------|-----------------|-------------------------------|
| Prompt injection leading to arbitrary tool execution | Project does not control LLM model behavior, user prompt construction, or which tools are registered. ToolNode routes by name only to user-registered tools. | Providing tool name allowlist routing (`libs/prebuilt/langgraph/prebuilt/tool_node.py:ToolNode._validate_tool_call`); user owns tool registration and argument handling |
| State poisoning via malicious node output | User-registered nodes (including `@task`-decorated functions) can write arbitrary values to channels. The framework executes nodes as provided. | Enforcing channel type contracts (`libs/langgraph/langgraph/channels/base.py:BaseChannel.update`); user owns node implementation correctness |
| Cross-session state access via thread_id guessing | Checkpoint savers index by `thread_id`. Without application-level auth, any caller with a valid thread_id can access that thread's state. | Providing the `Auth` handler system for access control (`libs/sdk-py/langgraph_sdk/auth/__init__.py:Auth`); user must implement auth handlers. Past advisory: GHSA-65c8-xj34-43q4 (closed) |
| Tool shadowing via duplicate registration | If a user registers two tools with the same name, ToolNode uses the last one. This is user misconfiguration. | Documenting tool registration semantics. Past advisory: GHSA-393p-4cgj-rj9m (closed) |
| Indirect prompt injection via tool output | LLM reads tool output and may follow injected instructions. This is a fundamental LLM limitation, not a framework vulnerability. | Not including tool output in system prompts; user owns output handling |
| Model selecting dangerous tool arguments | An LLM may generate SQL injection, path traversal, or command injection payloads as tool arguments. The risk depends entirely on what the user's tools do with those arguments. | Routing tool calls to registered tools only; user owns tool input validation |
| RCE via user-provided node code | `add_node()` and `@entrypoint`/`@task` accept arbitrary callables. A malicious node can do anything. This is by design — the user controls their own code. | Executing nodes within the graph runtime; user owns node code safety |
| SSRF via RemoteGraph URL | User provides the `url` parameter to `RemoteGraph`. Pointing it at an internal service is the user's decision. | Documenting that `url` should be a trusted endpoint; user owns URL selection |

### Rationale

**Prompt injection and tool execution**: LangGraph's `ToolNode` validates tool names against the registered set but does not inspect or sanitize argument values. This is the correct boundary — the framework cannot know what constitutes a "safe" argument for an arbitrary user-defined tool. The tool's own Pydantic schema and implementation must validate inputs. The framework's responsibility is to not execute unregistered tools and to correctly route registered ones.

**State integrity**: LangGraph channels enforce type contracts (e.g., `LastValue` accepts one value per step, `BinaryOperatorAggregate` applies a reducer). The framework validates graph structure at compile time (`libs/langgraph/langgraph/pregel/_validate.py:validate_graph`). However, the semantic correctness of state updates is the user's responsibility — the framework cannot know what values are "valid" for a user-defined state schema.

**Checkpoint access control**: The framework provides `BaseCheckpointSaver` as an abstract interface and the `Auth` handler system for authorization (`libs/sdk-py/langgraph_sdk/auth/__init__.py:Auth`). It does not enforce authentication by default because it operates as a library, not a server. The `langgraph-api` server layer (out of scope) is responsible for enforcing auth on API endpoints. Users embedding LangGraph directly must implement their own access controls.

---

## Investigated and Dismissed

| ID | Original Threat | Investigation | Evidence | Conclusion |
|----|----------------|---------------|----------|------------|
| D1 | SQL injection via filter keys in PostgreSQL store | Traced filter key handling through `libs/checkpoint-postgres/langgraph/store/postgres/base.py:_get_filter_condition`. All filter operations use parameterized queries with `%s` placeholders. Keys map to `json_extract` path operators with type-safe wrappers. | `libs/checkpoint-postgres/langgraph/store/postgres/base.py:_get_filter_condition` — parameterized `%s` for all value bindings; key names used in `value->%s` path expressions are also parameterized | Disproven: All SQL operations in PostgreSQL store are fully parameterized. No injection vector. |
| D2 | SQL injection via filter keys in SQLite store (post-fix) | Traced current filter handling through `libs/checkpoint-sqlite/langgraph/store/sqlite/base.py` and `libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/utils.py:_validate_filter_key`. Regex `^[a-zA-Z0-9_.-]+$` applied to all filter keys before use in `json_extract()` expressions. | `libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/utils.py:_validate_filter_key` — regex validation blocks injection characters. Published advisories GHSA-9rwj-6rc7-p77c and GHSA-7p73-8jqx-23r8 confirmed fixed. | Disproven: SQL injection in SQLite store filter keys is remediated by regex validation. |
| D3 | Command injection via CLI subprocess execution | Traced CLI subprocess invocation path. `libs/cli/langgraph_cli/exec.py:subp_exec` uses `asyncio.create_subprocess_exec` with list-based arguments (not `shell=True`). `has_disallowed_build_command_content` blocks shell metacharacters in user-provided Dockerfile lines. | `libs/cli/langgraph_cli/exec.py:subp_exec` — explicit exec-style invocation; `libs/cli/langgraph_cli/config.py:has_disallowed_build_command_content` — regex blocks `|`, `;`, `$`, `>`, `<`, backtick, `\`, single `&` | Disproven: CLI uses exec-style subprocess and validates build command content. No shell injection vector. |

---

## External Context

### Published Security Advisories

| GHSA ID | Severity | Summary | CWEs | Relevance |
|---------|----------|---------|------|-----------|
| GHSA-g48c-2wqr-h844 | Medium | Unsafe msgpack deserialization in LangGraph checkpoint loading | CWE-502 | Directly relates to T1 — patched in 1.0.10, confirms attack path via crafted msgpack payloads |
| GHSA-mhr3-j7m5-c7c9 | Medium | BaseCache Deserialization RCE | CWE-502 | Directly relates to T1 — msgpack deserialization in cache layer |
| GHSA-9rwj-6rc7-p77c | High | SQL injection via metadata filter key in SQLite checkpointer | CWE-89 | Fixed via `_validate_filter_key()` regex — see D2 |
| GHSA-wwqv-p2pp-99h5 | High | RCE in JSON mode of JsonPlusSerializer | CWE-502 | Directly relates to T3 — `lc:2` constructor import |
| GHSA-7p73-8jqx-23r8 | High | SQLite Filter Key SQL Injection in SqliteStore | CWE-89 | Fixed via `_validate_filter_key()` regex — see D2 |

**Pattern**: 3 of 5 published advisories involve CWE-502 (insecure deserialization) in the checkpoint serialization layer, with the newest (GHSA-g48c-2wqr-h844) published after the original threat model. This confirms the checkpoint storage boundary (TB2) as the highest-risk area. The 25+ closed advisories show an extensive history of deserialization bypass attempts, further validating this assessment.

**Closed advisory patterns**: ~15 deserialization RCE attempts (CWE-502), 2 SQLite SQL injection variants (CWE-89, fixed), 2 state poisoning reports (user responsibility), 1 IDOR via thread_id (user responsibility, Auth system provided), 1 Docker Compose injection (active advisory GHSA-22p4), 1 SSRF in Mermaid rendering.

### Dependabot Alerts

3 open alerts on JavaScript dependencies in CLI examples (handlebars prototype pollution, picomatch method injection). These affect `libs/cli/js-examples/` and `libs/cli/js-monorepo-example/` — example/documentation code, not core Python libraries. Low relevance to the primary threat model.

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-03-04 | Generated | Initial threat model |
| 2026-03-04 | Updated | Added C13 (Functional API), C14 (BaseCache), DF11. Updated T1 for BaseCache/serde event hooks. Added GHSA-mhr3-j7m5-c7c9 and GHSA-9rwj-6rc7-p77c. Updated CLI config scope. Added External Context section. |
| 2026-03-27 | Deep refinement | **Mode upgraded to Deep.** Added: Data Classification section (DC1-DC8 with detailed analysis for Critical/High entries). Added: C15 (Serde Event Hooks), C16 (Auth System). Added: Default? column to Components. Added: Classification column to Data Flows. Added: DF12-DF13 (SDK redirect flows). Added: T9 (SDK API key leak via Location redirect), T10 (EncryptedSerializer encryption bypass), T11 (unbounded checkpoint retention). Added: Validation column to Threats with flaw validation for High/Critical. Added: Investigated and Dismissed section (D1-D3: SQL injection and CLI command injection disproven). Added: Input Source Coverage section. Updated external context with GHSA-g48c-2wqr-h844 (new published advisory). Updated all code references to file:SymbolName notation. Expanded trust boundary details. |
