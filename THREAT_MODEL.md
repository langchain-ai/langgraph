# Threat Model: LangGraph

> Generated: 2026-03-28 | Commit: 0ba22143 | Scope: Full monorepo (all libs/)

> **Disclaimer:** This threat model is automatically generated to help developers and security researchers understand where trust is placed in this system and where boundaries exist. It is experimental, subject to change, and not an authoritative security reference — findings should be validated before acting on them. The analysis may be incomplete or contain inaccuracies. We welcome suggestions and corrections to improve this document.

For vulnerability reporting, see the [GitHub Security Advisories](https://github.com/langchain-ai/langgraph/security/advisories) page.

## Scope

### In Scope

- `libs/langgraph` — Core graph execution engine (Pregel, StateGraph, channels, functional API with `@entrypoint`/`@task`)
- `libs/prebuilt` — High-level agent APIs (ToolNode, create_react_agent, ValidationNode, InjectedState/InjectedStore/ToolRuntime injection)
- `libs/checkpoint` — Checkpoint serialization/deserialization (JsonPlusSerializer, EncryptedSerializer, BaseCache, stores, serde event hooks, SAFE_MSGPACK_TYPES allowlist)
- `libs/checkpoint-postgres` — PostgreSQL checkpoint saver, key-value store, and vector search
- `libs/checkpoint-sqlite` — SQLite checkpoint saver, key-value store, and vector search
- `libs/cli` — CLI for Docker-based deployment (`langgraph up/build/dev/new`), WebhookUrlPolicy
- `libs/sdk-py` — Python SDK client for LangGraph Server API (HttpClient, Auth system, Encryption handlers)

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
7. SDK Encryption handlers are developer-authored server-side code with application-level trust.

---

## System Overview

LangGraph is an open-source Python framework for building stateful, multi-actor AI agent applications. It provides a graph-based execution model (Bulk Synchronous Parallel via the Pregel engine) where user-defined nodes process shared state through typed channels. The framework supports two authoring APIs: the declarative StateGraph API and the functional API (`@entrypoint`/`@task` decorators). It includes checkpointing (persistence of graph state to databases), tool execution (dispatching LLM-generated tool calls with runtime injection of state/store/context), remote graph composition (calling LangGraph Server APIs), Docker-based deployment via a CLI, and a beta SDK encryption framework for custom at-rest encryption handlers.

### Architecture Diagram

```
+---------------------------------------------------------------------------+
|                        User Application                                    |
|                                                                            |
|  +-----------------------------------------------+                        |
|  |  User Application Code                         |                       |
|  |  (graph nodes, tools; StateGraph builder API   |                       |
|  |   and functional API @entrypoint/@task both    |                       |
|  |   compile to the same Pregel execution engine) |                       |
|  +------------------------+----------------------+                        |
|                           |                                                |
|              +------------v-----------+                                   |
|              |  StateGraph / Pregel    |                                   |
|              |  (core execution engine)|                                   |
|              +------+----------+------+                                   |
|                     |          |                                           |
|              InjectedState  +--v---------+                                |
|              InjectedStore  |  ToolNode  |                                |
|              ToolRuntime    |  (opt-in)  |                                |
|                             +------------+                                |
|                       |                                                    |
| - - - - - - - - - - - | - - - - TB1: User/Framework API - - - - - - - -  |
|                       |                                                    |
|                 +-----v------+   +--------------+                         |
|                 | Checkpoint |   | RemoteGraph   |                         |
|                 | Serializer |   | (SDK client)  |                         |
|                 |(jsonplus)  |   +------+--------+                         |
|                 +-----+------+          |                                  |
|                       |                 |                                   |
| - - - - - - - - - - - | - - - - - - - -|- - TB2: Storage/Network - - - -  |
|                       v                 v                                   |
|               +--------------+  +--------------+                          |
|               |  PostgreSQL  |  |  LangGraph   |                          |
|               |  / SQLite    |  |  Server API  |                          |
|               +--------------+  +--------------+                          |
|                                                                            |
|  +----------+                   +--------------+                          |
|  |   CLI    |------------------>|   Docker     |                          |
|  |(langgraph|   TB4: Config    |   Engine     |                          |
|  | up/build)|                   +--------------+                          |
|  +----------+                                                             |
|                                                                            |
|  +--------------------+                                                   |
|  | SDK Encryption     |  TB5: Developer-authored handlers                 |
|  | Handlers (beta)    |  (server-side execution in langgraph-api)         |
|  +--------------------+                                                   |
+---------------------------------------------------------------------------+
```

---

## Components

| ID | Component | Description | Trust Level | Default? | Entry Points |
|----|-----------|-------------|-------------|----------|--------------|
| C1 | StateGraph / Pregel | Core graph builder and execution engine with v1/v2 output, durability modes (sync/async/exit), interrupt_before/interrupt_after | framework-controlled | Yes | `StateGraph.add_node()`, `StateGraph.compile()`, `Pregel.invoke()`, `Pregel.stream()` |
| C2 | JsonPlusSerializer | Checkpoint serialization/deserialization with msgpack, JSON, and pickle codecs; 47-entry SAFE_MSGPACK_TYPES allowlist | framework-controlled | Yes | `loads_typed()`, `dumps_typed()`, `_create_msgpack_ext_hook()`, `_reviver()` |
| C3 | ToolNode | Dispatches LLM-generated tool calls to registered BaseTool instances; supports InjectedState/InjectedStore/ToolRuntime injection into tools | framework-controlled | No (explicit opt-in required) | `ToolNode._func()`, `_run_one()`, `_execute_tool_sync()`, `_validate_tool_call()`, `_inject_tool_args()` |
| C4 | RemoteGraph | Client for remote LangGraph Server API; implements PregelProtocol | framework-controlled | No (opt-in) | `RemoteGraph.stream()`, `RemoteGraph.invoke()`, `RemoteGraph.get_state()` |
| C5 | PostgresSaver / PostgresStore | PostgreSQL checkpoint saver, key-value store, and vector search | framework-controlled | No (opt-in) | `from_conn_string()`, `put()`, `get_tuple()`, `search()` |
| C6 | SqliteSaver / SqliteStore | SQLite checkpoint saver, key-value store with JSON path filtering | framework-controlled | No (opt-in) | `from_conn_string()`, `put()`, `get_tuple()`, `search()` |
| C7 | EncryptedSerializer | AES-EAX authenticated encryption wrapper for checkpoint data | framework-controlled | No (opt-in) | `from_pycryptodome_aes()`, `loads_typed()`, `dumps_typed()` |
| C8 | CLI (langgraph_cli) | Docker-based build and deployment tooling; config schema includes WebhookUrlPolicy for SSRF protection | framework-controlled | No (separate install) | `langgraph up`, `langgraph build`, `langgraph dev`, `langgraph new` |
| C9 | SDK Client (langgraph_sdk) | HTTP client for LangGraph Server API with SSE streaming and reconnection | framework-controlled | Yes | `get_client()`, `get_sync_client()`, `HttpClient.request_reconnect()`, `HttpClient.stream()` |
| C10 | User-Registered Tools | BaseTool instances provided by users; may use InjectedState/InjectedStore/ToolRuntime annotations | user-controlled | N/A | Tool `invoke()` / `ainvoke()` methods |
| C11 | User-Registered Nodes | Arbitrary callables added via `add_node()` or `@task`/`@entrypoint` | user-controlled | N/A | Node function signatures |
| C12 | Checkpoint Storage | PostgreSQL or SQLite databases storing serialized graph state | external | N/A | Database connection interface |
| C13 | Functional API | `@entrypoint`/`@task` decorators for function-based workflow authoring with retry/cache policies | framework-controlled | Yes | `entrypoint.__call__()`, `task()`, `_TaskFunction.__call__()` (`libs/langgraph/langgraph/func/__init__.py`) |
| C14 | BaseCache | Cache layer for task results with JsonPlusSerializer (pickle_fallback=False) | framework-controlled | No (opt-in, requires checkpointer) | `get()`, `set()`, `clear()` (`libs/checkpoint/langgraph/cache/base/__init__.py`) |
| C15 | Serde Event Hooks | Monitoring system for serialization/deserialization events (msgpack_blocked, msgpack_unregistered_allowed, msgpack_method_blocked) | framework-controlled | Yes | `register_serde_event_listener()`, `emit_serde_event()` (`libs/checkpoint/langgraph/checkpoint/serde/event_hooks.py`) |
| C16 | Auth System (SDK) | Custom authentication/authorization handler framework | framework-controlled | No (opt-in) | `Auth.authenticate()`, `Auth.on()` handler registration (`libs/sdk-py/langgraph_sdk/auth/__init__.py`) |
| C17 | SDK Encryption Handlers (beta) | Custom at-rest encryption/decryption framework; supports blob and JSON handlers with per-model/field context; server-side execution | framework-controlled | No (opt-in, beta) | `Encryption.encrypt.blob()`, `Encryption.encrypt.json()`, `Encryption.decrypt.blob()`, `Encryption.decrypt.json()`, `Encryption.context()` (`libs/sdk-py/langgraph_sdk/encryption/__init__.py`) |

---

## Data Classification

| ID | PII Category | Specific Fields | Sensitivity | Storage Location(s) | Encrypted at Rest | Retention | Regulatory |
|----|-------------|----------------|-------------|---------------------|-------------------|-----------|------------|
| DC1 | API credentials | `x-api-key` header, `LANGGRAPH_API_KEY`, `LANGSMITH_API_KEY`, `LANGCHAIN_API_KEY` env vars | Critical | Environment variables, HTTP headers in transit | N/A (in-memory) | Session lifetime | All — breach trigger |
| DC2 | Encryption keys | `LANGGRAPH_AES_KEY` env var, `key` parameter to `from_pycryptodome_aes()` | Critical | Environment variable, in-memory | N/A | Application lifetime | All — breach trigger |
| DC3 | Serialized graph state | Checkpoint data in `checkpoints` and `writes` tables (msgpack/JSON/pickle bytes) | High | PostgreSQL (BYTEA), SQLite (BLOB) | Optional via EncryptedSerializer or SDK Encryption Handlers | Unbounded (no default TTL) | GDPR if state contains PII |
| DC4 | Store key-value data | User-stored items in `store` tables via BaseStore | High | PostgreSQL, SQLite | No (plaintext JSON); optional via SDK Encryption Handlers | Configurable TTL, default unbounded | GDPR if contains PII |
| DC5 | Checkpoint metadata | `thread_id`, `checkpoint_ns`, `run_id`, `step`, `source` | Medium | PostgreSQL, SQLite (metadata JSONB/JSON column) | No | Same as DC3 | Minimal |
| DC6 | Agent conversation history | LangChain messages (HumanMessage, AIMessage, ToolMessage) serialized in checkpoint state | High | PostgreSQL, SQLite (within DC3 checkpoint bytes) | Only if DC3 encrypted | Unbounded | GDPR, CCPA if contains user PII |
| DC7 | Connection strings | PostgreSQL URIs, SQLite file paths passed to `from_conn_string()` | Critical | Application code, environment variables | N/A (in-memory) | Application lifetime | All — may contain credentials |
| DC8 | Vector embeddings | Document embeddings in `store_vectors` table | Low | PostgreSQL (pgvector), SQLite (vec extension) | No | Same as DC4 | Minimal |
| DC9 | SDK Encryption context metadata | `EncryptionContext.metadata` dict passed to encryption handlers | Medium | In-memory per request; persisted with encrypted data | N/A (context, not payload) | Request lifetime + persistence alongside encrypted data | Depends on content |

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
- **Encryption**: Optional via `EncryptedSerializer` wrapping (AES-EAX) or SDK Encryption Handlers (beta, server-side). Not encrypted by default.
- **Retention**: Unbounded by default. Optional TTL via `CheckpointerConfig.ttl` (server-side config)
- **Logging exposure**: Serde event hooks emit module/class names of deserialized types but not the data itself
- **Gaps**: Default unbounded retention of potentially PII-containing state. Unencrypted by default. EncryptedSerializer has fallback that accepts unencrypted data (see T10).

#### DC6: Agent Conversation History

- **Fields**: `HumanMessage.content`, `AIMessage.content`, `ToolMessage.content`, `AIMessage.tool_calls` — embedded within DC3 checkpoint bytes
- **Storage**: Same as DC3 (within serialized checkpoint data)
- **Access**: Same as DC3
- **Encryption**: Only if DC3 is encrypted via EncryptedSerializer or SDK Encryption Handlers
- **Retention**: Same as DC3 (unbounded default)
- **Gaps**: Conversation content may include user PII, PHI, or sensitive business data. No field-level encryption or redaction. Retention inherits from DC3 with no conversation-specific policy.

#### DC9: SDK Encryption Context Metadata

- **Fields**: `EncryptionContext.model` (str), `EncryptionContext.field` (str), `EncryptionContext.metadata` (dict)
- **Storage**: In-memory during request processing; persisted alongside encrypted data for later decryption
- **Access**: Encryption/decryption handlers (developer-authored), ContextHandler (receives authenticated BaseUser)
- **Encryption**: N/A — this is context for encryption, not encrypted data itself
- **Retention**: Persisted with encrypted data indefinitely
- **Logging exposure**: Not logged by SDK code
- **Gaps**: `metadata` is a mutable dict — whether cross-request isolation is enforced depends on server-side implementation (langgraph-api, out of scope). ContextHandler registration at `libs/sdk-py/langgraph_sdk/encryption/__init__.py:Encryption.context` does not call `_validate_handler` (missing async/param-count validation, unlike all other handler types).

---

## Trust Boundaries

| ID | Boundary | Description | Controls (Inside) | Does NOT Control (Outside) |
|----|----------|-------------|-------------------|---------------------------|
| TB1 | User/Framework API | Where user-provided code and configuration enters the framework | Graph execution logic, channel semantics, default configs, validation of graph structure, tool injection merge order (system values overwrite LLM values) | User node implementations, tool behavior, model selection, prompt construction, state schema design |
| TB2 | Checkpoint Storage | Where serialized data enters/leaves the persistence layer | Serialization format, allowlists for deserialization (47 safe types, 1 safe method), encryption (if configured), serde event hooks | Database access controls, who can write to the checkpoint tables, storage infrastructure security |
| TB3 | Remote API | Where data crosses the network to/from LangGraph Server | Outbound config sanitization (`_sanitize_config`), SDK HTTP transport, API key handling, `RESERVED_HEADERS` | Remote server behavior, response content integrity, network security (TLS), server-provided Location redirect targets |
| TB4 | CLI Config/Docker | Where developer config drives container image generation | Dockerfile template structure, config schema validation (including WebhookUrlPolicy), list-based subprocess args, build command content validation | `langgraph.json` file content, Docker daemon security, host filesystem |
| TB5 | SDK Encryption Handlers | Where developer-authored encryption handlers process sensitive data | Handler signature validation (async, 2-param for encrypt/decrypt), duplicate registration prevention, EncryptionContext construction | Handler implementation correctness, key management, actual encrypt/decrypt behavior, server-side execution environment |

### Boundary Details

#### TB1: User/Framework API

- **Inside**: Graph compilation validates structure (`libs/langgraph/langgraph/pregel/_validate.py:validate_graph`). Channel types enforce update semantics (`libs/langgraph/langgraph/channels/base.py:BaseChannel.update`). Functional API validates entrypoint has at least one parameter (`libs/langgraph/langgraph/func/__init__.py:entrypoint`). Sensitive config keys filtered from metadata propagation — keys containing "key", "token", "secret", "password", "auth" are excluded (`libs/langgraph/langgraph/_internal/_config.py:_exclude_as_metadata`). Tool injection merge order ensures system-injected values (InjectedState/InjectedStore/ToolRuntime) overwrite any LLM-supplied collisions (`libs/prebuilt/langgraph/prebuilt/tool_node.py:ToolNode._inject_tool_args` line 1380). Injected parameter names hidden from LLM tool schema via `tool_call_schema` filtering.
- **Outside**: What user nodes do, what tools return, what LLMs generate, how users handle output.
- **Crossing mechanism**: Python function calls — `add_node(callable)`, `add_edge()`, `compile(checkpointer=...)`, `@entrypoint`, `@task`.

#### TB2: Checkpoint Storage

- **Inside**: `JsonPlusSerializer` controls serialization format (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer`). Msgpack type allowlist (`libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py:SAFE_MSGPACK_TYPES` — 47 safe types including stdlib, langchain_core messages, and langgraph types). Msgpack method allowlist (`libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py:SAFE_MSGPACK_METHODS` — 1 safe method: `datetime.datetime.fromisoformat`). JSON module allowlist (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:_check_allowed_json_modules`). Serde event hooks for monitoring (`libs/checkpoint/langgraph/checkpoint/serde/event_hooks.py:emit_serde_event`). Optional `EncryptedSerializer` wrapping (`libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:EncryptedSerializer`). SQLite filter key regex validation (`libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/utils.py:_validate_filter_key`). Parameterized SQL queries in both Postgres and SQLite backends.
- **Outside**: Database access controls, who can read/write checkpoint tables, storage backend integrity.
- **Crossing mechanism**: Database read/write operations — serialized bytes stored as BYTEA (Postgres) or BLOB (SQLite).

#### TB3: Remote API

- **Inside**: `_sanitize_config()` strips non-primitive values and drops checkpoint-internal keys from outbound config (`libs/langgraph/langgraph/pregel/remote.py:_sanitize_config`). SDK handles API key from env vars (`libs/sdk-py/langgraph_sdk/_shared/utilities.py:_get_api_key`). `RESERVED_HEADERS` prevents user override of `x-api-key` (`libs/sdk-py/langgraph_sdk/_shared/utilities.py:RESERVED_HEADERS`).
- **Outside**: Remote server response content, network integrity, whether the server is legitimate, server-provided Location redirect targets.
- **Crossing mechanism**: HTTPS requests via `httpx` through `langgraph_sdk`.

#### TB4: CLI Config/Docker

- **Inside**: Config file parsed as JSON (`libs/cli/langgraph_cli/config.py:validate_config_file`). Docker subprocess invoked with list-based args via `asyncio.create_subprocess_exec`, not `shell=True` (`libs/cli/langgraph_cli/exec.py:subp_exec`). Template downloads from hardcoded GitHub URLs (`libs/cli/langgraph_cli/templates.py`). Config schema validation covers store, auth, encryption, http, webhooks, checkpointer, and ui sections (`libs/cli/langgraph_cli/schemas.py`). Build command content validation blocks shell metacharacters (`libs/cli/langgraph_cli/config.py:has_disallowed_build_command_content`). WebhookUrlPolicy (`libs/cli/langgraph_cli/schemas.py:WebhookUrlPolicy`) supports `require_https`, `allowed_domains`, `allowed_ports`, `max_url_length`, `disable_loopback` for SSRF protection.
- **Outside**: Content of `langgraph.json`, Docker daemon behavior, filesystem permissions.
- **Crossing mechanism**: JSON file read, subprocess execution, ZIP download/extraction.

#### TB5: SDK Encryption Handlers

- **Inside**: Handler signature validation — must be async, must accept exactly 2 positional params (`libs/sdk-py/langgraph_sdk/encryption/__init__.py:_validate_handler`). Duplicate handler registration prevention (`DuplicateHandlerError`). `EncryptionContext` construction with model/field/metadata (`libs/sdk-py/langgraph_sdk/encryption/types.py:EncryptionContext`). JSON key preservation constraint documented (enforced server-side).
- **Outside**: Handler implementation correctness, key management strategy, actual encryption/decryption logic, server-side execution in langgraph-api.
- **Crossing mechanism**: Python decorator registration at import time; server-side invocation at runtime.

---

## Data Flows

| ID | Source | Destination | Data Type | Classification | Crosses Boundary | Protocol |
|----|--------|-------------|-----------|----------------|------------------|----------|
| DF1 | C12 (Checkpoint Storage) | C2 (JsonPlusSerializer) | Serialized checkpoint bytes (msgpack/JSON/pickle) | DC3 | TB2 | Database read |
| DF2 | C2 (JsonPlusSerializer) | C1 (Pregel) | Deserialized Python objects (channel state) | DC3, DC6 | TB2 | Function call |
| DF3 | LLM (external) | C3 (ToolNode) | Tool call arguments (JSON strings in AIMessage) | — | TB1 | Function call (via langchain-core) |
| DF4 | C3 (ToolNode) | C10 (User Tools) | Parsed argument dicts merged with injected state/store/runtime | — | TB1 | `tool.invoke(call_args)` |
| DF5 | C4 (RemoteGraph) | C1 (Pregel) | Stream chunks (JSON-deserialized dicts) | — | TB3 | HTTPS / SSE |
| DF6 | `langgraph.json` | C8 (CLI) | Config dict (graphs, env, store, auth, encryption, http, webhooks, checkpointer, ui) | — | TB4 | `json.load()` |
| DF7 | C8 (CLI) | Docker | Dockerfile content with embedded ENV values | — | TB4 | `asyncio.create_subprocess_exec` |
| DF8 | C11 (User Nodes) | C1 (Pregel) | State updates (arbitrary Python objects) | — | TB1 | Channel write |
| DF9 | C9 (SDK Client) | C4 (RemoteGraph) | API responses (JSON) | — | TB3 | HTTPS |
| DF10 | User config | C7 (EncryptedSerializer) | AES key from LANGGRAPH_AES_KEY env var | DC2 | TB2 | `os.getenv()` |
| DF11 | C12 (Checkpoint Storage) | C14 (BaseCache) | Cached task results via JsonPlusSerializer | DC3 | TB2 | Database read |
| DF12 | LangGraph Server | C9 (SDK Client) | HTTP responses with Location header | DC1 | TB3 | HTTP redirect |
| DF13 | C9 (SDK Client) | Redirect target | Request headers including x-api-key | DC1 | TB3 | HTTPS |
| DF14 | C1 (Pregel state) | C3 (ToolNode) | InjectedState/InjectedStore/ToolRuntime values for tool injection | DC3, DC4 | TB1 | Function call (dict merge) |
| DF15 | Developer code | C17 (SDK Encryption Handlers) | Encryption/decryption handler functions and context handler | — | TB5 | Python decorator registration |

### Flow Details

#### DF1: Checkpoint Storage -> JsonPlusSerializer

- **Data**: Serialized graph state as `(type_tag, bytes)` tuples. Type tags include `"msgpack"`, `"json"`, `"pickle"`, `"bytes"`, `"null"`. When encrypted: `"msgpack+aes"`, `"json+aes"`.
- **Validation**: Type tag dispatches to codec. Msgpack: `_create_msgpack_ext_hook` with allowlist check — `SAFE_MSGPACK_TYPES` (47 entries) always checked first, then `allowed_modules` determines behavior for unregistered types (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:_create_msgpack_ext_hook`). JSON: `_reviver` with `lc:2` module allowlist. Pickle: **no restrictions** (`pickle.loads(data_)` if `pickle_fallback=True`, `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer.loads_typed`). The proposed `secure_pickle.py` with `RestrictedUnpickler` was documented in `SECURITY_FIX_SUMMARY.md` but never merged.
- **Trust assumption**: Checkpoint storage is access-controlled. An attacker with write access to the database can craft malicious checkpoint data.

#### DF3: LLM -> ToolNode

- **Data**: Tool call name and arguments from LLM-generated `AIMessage.tool_calls`.
- **Validation**: Tool name checked against registered `tools_by_name` dict — unknown names return error `ToolMessage` (`libs/prebuilt/langgraph/prebuilt/tool_node.py:ToolNode._validate_tool_call`). Argument values validated only by the target tool's Pydantic schema.
- **Trust assumption**: LLM output is treated as untrusted for tool name routing but argument values pass through to tools without ToolNode-level sanitization.

#### DF4: ToolNode -> User Tools (with Injection)

- **Data**: Parsed argument dicts from LLM, merged with system-injected InjectedState/InjectedStore/ToolRuntime values.
- **Validation**: Four-layer defense: (1) Injected parameter names hidden from LLM via `tool_call_schema` filtering. (2) Dict merge `{**llm_args, **injected_args}` places system values last — system always wins on collision (`libs/prebuilt/langgraph/prebuilt/tool_node.py:ToolNode._inject_tool_args` line 1380). (3) Pydantic `model_validate` with default `extra="ignore"` drops unknown keys. (4) Output construction only includes declared model fields.
- **Trust assumption**: LLM-provided arguments cannot override system-injected values due to merge order.

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

#### DF14: Pregel State -> ToolNode (Runtime Injection)

- **Data**: Graph state dict (InjectedState), BaseStore instance (InjectedStore), ToolRuntime object (containing state, config, store, context, stream_writer, tool_call_id).
- **Validation**: Injection targets determined by tool type annotations at compile time. Injected values overwrite any LLM-provided values with matching keys (safe merge order). Pydantic validation on tool input drops extra keys not in the tool's declared schema.
- **Trust assumption**: System-injected values are trusted; LLM-provided values cannot interfere due to merge order guarantees.

#### DF15: Developer Code -> SDK Encryption Handlers

- **Data**: Async Python callables registered via decorators for blob/JSON encryption/decryption and context derivation.
- **Validation**: `_validate_handler` checks async-ness and 2-param signature for encrypt/decrypt handlers. `DuplicateHandlerError` prevents double registration. **Gap**: `Encryption.context()` method does NOT call `_validate_handler` — a sync function or wrong param count passes registration and fails only at server-side invocation (`libs/sdk-py/langgraph_sdk/encryption/__init__.py:Encryption.context`).
- **Trust assumption**: Handler authors are application developers with server-level trust.

---

## Threats

| ID | Data Flow | Classification | Threat | Boundary | Severity | Validation | Code Reference |
|----|-----------|----------------|--------|----------|----------|------------|----------------|
| T1 | DF1, DF11 | DC3 | Arbitrary code execution via msgpack deserialization when strict mode is OFF (default) | TB2 | High | Verified | `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:_create_msgpack_ext_hook` |
| T2 | DF1 | DC3 | Arbitrary code execution via `pickle.loads` when `pickle_fallback=True` | TB2 | High | Verified | `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer.loads_typed` |
| T3 | DF1 | DC3 | Arbitrary module import/execution via JSON `lc:2` constructor when `allowed_json_modules=True` | TB2 | High | Verified | `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer._revive_lc2` |
| T4 | DF5 | — | Unvalidated dict-splatting from remote API into `Interrupt`/`Command` objects | TB3 | Medium | Likely | `libs/langgraph/langgraph/pregel/remote.py:RemoteGraph.stream` |
| T5 | DF6, DF7 | — | Dockerfile ENV injection via single-quote in `langgraph.json` config values | TB4 | Low | Likely | `libs/cli/langgraph_cli/config.py:python_config_to_docker` |
| T6 | DF7 | — | ZIP slip in `langgraph new` template extraction | TB4 | Low | Unverified | `libs/cli/langgraph_cli/templates.py:_download_repo_with_requests` |
| T7 | DF10 | DC2 | AES key entropy limited to printable characters via env var string encoding | TB2 | Info | — | `libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:EncryptedSerializer.from_pycryptodome_aes` |
| T8 | DF10 | DC2 | EncryptedSerializer cipher name check uses `assert` (stripped with `python -O`) | TB2 | Low | Verified | `libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:PycryptodomeAesCipher.decrypt` |
| T9 | DF12, DF13 | DC1 | SDK API key leak via server-controlled Location redirect to attacker-controlled URL | TB3 | Medium | Verified | `libs/sdk-py/langgraph_sdk/_async/http.py:HttpClient.request_reconnect`, `libs/sdk-py/langgraph_sdk/_async/http.py:HttpClient.stream` |
| T10 | DF1 | DC3 | EncryptedSerializer silently accepts unencrypted data — attacker bypasses encryption by writing plain type tags | TB2 | Medium | Verified | `libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:EncryptedSerializer.loads_typed` |
| T11 | DF1, DF11 | DC3, DC6 | Unbounded retention of checkpoint data containing PII/conversation history | TB2 | Medium | — | `libs/checkpoint/langgraph/checkpoint/base/__init__.py:BaseCheckpointSaver` |

### Threat Details

#### T1: Msgpack Deserialization RCE (Default Config)

- **Flow**: DF1 (Checkpoint Storage -> JsonPlusSerializer), DF11 (Checkpoint Storage -> BaseCache)
- **Description**: When `LANGGRAPH_STRICT_MSGPACK` is not set (the default), the msgpack `_create_msgpack_ext_hook` allows **any** `(module, class)` pair stored in checkpoint data to be imported via `importlib.import_module` and instantiated with attacker-controlled arguments. The `SAFE_MSGPACK_TYPES` allowlist (47 entries) is checked first, but unregistered types are logged as warnings and allowed through when `allowed_modules=True` (the default when strict mode is off). Seven EXT codes are processed: `EXT_CONSTRUCTOR_SINGLE_ARG` (0), `EXT_CONSTRUCTOR_POS_ARGS` (1), `EXT_CONSTRUCTOR_KW_ARGS` (2), `EXT_METHOD_SINGLE_ARG` (3), `EXT_PYDANTIC_V1` (4), `EXT_PYDANTIC_V2` (5), `EXT_NUMPY_ARRAY` (6). The `BaseCache` component uses `JsonPlusSerializer(pickle_fallback=False)` but inherits the same msgpack `allowed_modules` default behavior. The proposed `RestrictedUnpickler` (`secure_pickle.py`) documented in `SECURITY_FIX_SUMMARY.md` was never merged — pickle remains unrestricted when enabled.
- **Preconditions**: Attacker must have write access to the checkpoint database (PostgreSQL or SQLite). This requires compromised database credentials or a co-located attacker.

#### T2: Pickle Deserialization RCE

- **Flow**: DF1 (Checkpoint Storage -> JsonPlusSerializer)
- **Description**: When `pickle_fallback=True` is explicitly passed to `JsonPlusSerializer`, checkpoint data with type tag `"pickle"` is deserialized via `pickle.loads()` with zero restrictions (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer.loads_typed`).
- **Preconditions**: (1) Application or checkpointer explicitly enables `pickle_fallback=True`. (2) Attacker writes `("pickle", <payload>)` to checkpoint storage.

#### T3: JSON lc:2 Constructor RCE

- **Flow**: DF1 (Checkpoint Storage -> JsonPlusSerializer)
- **Description**: The JSON `_reviver` handles `lc:2` type constructors by importing the module path from checkpoint JSON data via `importlib.import_module` (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:JsonPlusSerializer._revive_lc2`). If `allowed_json_modules=True` (explicit opt-in), any module reachable in the Python environment can be imported and instantiated. The method also supports method chaining — a `method` key in the JSON can call arbitrary methods on the imported class.
- **Preconditions**: (1) `allowed_json_modules` set to `True` (not the default). (2) Attacker writes crafted JSON to checkpoint storage.

#### T4: RemoteGraph Unvalidated Inbound Data

- **Flow**: DF5 (RemoteGraph -> Pregel)
- **Description**: Stream events from the remote LangGraph Server are deserialized from JSON and dict-splatted into `Interrupt(**i)` and `Command(**chunk.data)` without schema validation (`libs/langgraph/langgraph/pregel/remote.py:RemoteGraph.stream`). A compromised or malicious remote server can inject unexpected fields. `Command.update` can carry arbitrary state modifications; `Command.goto` can alter graph execution flow. `Interrupt` accepts `**deprecated_kwargs` which includes a `ns` parameter that can override interrupt ID generation via `xxh3_128_hexdigest`.
- **Preconditions**: User connects `RemoteGraph` to a compromised or attacker-controlled server URL.

#### T5: Dockerfile ENV Single-Quote Injection

- **Flow**: DF6, DF7 (langgraph.json -> CLI -> Dockerfile)
- **Description**: Config values from `langgraph.json` are serialized via `json.dumps()` and embedded in single-quoted `ENV` directives across multiple config sections (store, auth, encryption, http, webhooks, checkpointer, ui, ui_config, graphs). JSON does not escape single quotes, so a config value containing `'` could break the Dockerfile syntax or inject additional Dockerfile instructions. The pattern is duplicated in two Dockerfile generation functions (`libs/cli/langgraph_cli/config.py:python_config_to_docker` and `libs/cli/langgraph_cli/config.py:node_config_to_docker`).
- **Preconditions**: A `langgraph.json` config value contains a single quote character.

#### T6: ZIP Slip in Template Extraction

- **Flow**: DF7 (CLI template download)
- **Description**: `langgraph new` downloads a ZIP from GitHub and uses `zip_file.extractall(path)`. If the archive contains path-traversal entries (e.g., `../../etc/cron.d/exploit`), files could be written outside the target directory.
- **Preconditions**: The GitHub-hosted template archive must contain malicious path entries. This requires compromise of the upstream template repo.

#### T7: AES Key Entropy via Environment Variable

- **Flow**: DF10 (User config -> EncryptedSerializer)
- **Description**: The AES key is loaded from `LANGGRAPH_AES_KEY` as a UTF-8 string and `.encode()`d to bytes (`libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:EncryptedSerializer.from_pycryptodome_aes`). This limits key entropy to printable characters (~6.57 bits/byte vs. 8 bits/byte for random bytes), reducing effective key strength for AES-128 from 128 bits to ~105 bits.
- **Preconditions**: User relies on environment variable path for key loading (vs. passing raw bytes directly via `key=` parameter).

#### T8: EncryptedSerializer Assert Bypass

- **Flow**: DF10 (Encrypted checkpoint data)
- **Description**: The cipher name check in `decrypt()` uses `assert ciphername == "aes"` (`libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:PycryptodomeAesCipher.decrypt`), which is stripped when Python runs with `-O` (optimize) flag. The `ciphername` value comes from the type tag in checkpoint storage (split from the `type+cipher` format).
- **Preconditions**: Python running with `-O` flag AND attacker can write to checkpoint storage.

#### T9: SDK API Key Leak via Server-Controlled Location Redirect

- **Flow**: DF12 (Server -> SDK), DF13 (SDK -> Redirect target)
- **Description**: The SDK's `HttpClient.request_reconnect()` (`libs/sdk-py/langgraph_sdk/_async/http.py:HttpClient.request_reconnect`) follows server-provided `Location` headers and forwards the full `request_headers` dict (including the `x-api-key` authentication header) to the redirected URL. The `HttpClient.stream()` method (`libs/sdk-py/langgraph_sdk/_async/http.py:HttpClient.stream`) also follows `Location` headers for SSE reconnection and forwards `reconnect_headers` (which include `x-api-key`) to the server-controlled URL. No URL validation, same-origin check, or sensitive header stripping is performed before following the redirect. The same pattern exists in the sync client (`libs/sdk-py/langgraph_sdk/_sync/http.py`).
- **Preconditions**: (1) User connects SDK to a LangGraph Server that is compromised or attacker-controlled. (2) The server returns a response with a `Location` header pointing to an attacker-controlled URL.

#### T10: EncryptedSerializer Encryption Bypass via Unencrypted Data Injection

- **Flow**: DF1 (Checkpoint Storage -> EncryptedSerializer)
- **Description**: `EncryptedSerializer.loads_typed()` (`libs/checkpoint/langgraph/checkpoint/serde/encrypted.py:EncryptedSerializer.loads_typed`) checks if the type tag contains a `+` delimiter. If it does not (e.g., type tag is `"msgpack"` instead of `"msgpack+aes"`), the data is passed directly to the inner serde's `loads_typed()` **without any decryption or MAC verification**. An attacker with write access to checkpoint storage can bypass the encryption layer entirely by writing data with a plain type tag.
- **Preconditions**: (1) Application uses `EncryptedSerializer` for checkpoint protection. (2) Attacker has write access to checkpoint storage.

#### T11: Unbounded Checkpoint Data Retention

- **Flow**: DF1, DF11 (Checkpoint Storage lifecycle)
- **Description**: Checkpoint data (DC3, DC6) is retained indefinitely by default. No built-in TTL, pruning, or data lifecycle management in the library-level checkpoint savers. Conversation history containing user PII may accumulate without bounds.
- **Preconditions**: Application uses checkpointing (the primary use case). No explicit cleanup configured.

---

## Input Source Coverage

| Input Source | Data Flows | Threats | Validation Points | Responsibility | Gaps |
|-------------|-----------|---------|-------------------|----------------|------|
| User direct input (graph state, config) | DF8 | — | Graph structure validation (`libs/langgraph/langgraph/pregel/_validate.py:validate_graph`), channel type enforcement (`libs/langgraph/langgraph/channels/base.py:BaseChannel`), sensitive key filtering (`libs/langgraph/langgraph/_internal/_config.py:_exclude_as_metadata`) | User | Node implementation safety is user's responsibility |
| LLM output (tool calls) | DF3, DF4, DF14 | — | Tool name allowlist (`libs/prebuilt/langgraph/prebuilt/tool_node.py:ToolNode._validate_tool_call`), tool Pydantic schemas, injection merge order (`libs/prebuilt/langgraph/prebuilt/tool_node.py:ToolNode._inject_tool_args`), `tool_call_schema` filtering of injected params | Shared (project validates name and injection safety; user validates args via tool schema) | No ToolNode-level argument sanitization beyond injection overwrite |
| Checkpoint storage data | DF1, DF2, DF11 | T1, T2, T3, T10 | Msgpack allowlist (`libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py:SAFE_MSGPACK_TYPES` — 47 entries), msgpack method allowlist (`libs/checkpoint/langgraph/checkpoint/serde/_msgpack.py:SAFE_MSGPACK_METHODS`), JSON allowlist (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py:_check_allowed_json_modules`), pickle gating, serde event hooks, optional encryption | Shared (project owns serializer defaults; user owns DB access controls) | Default msgpack mode allows unregistered types; EncryptedSerializer accepts unencrypted data; proposed secure_pickle.py never merged |
| Remote API responses | DF5, DF9, DF12, DF13 | T4, T9 | Outbound config sanitization (`libs/langgraph/langgraph/pregel/remote.py:_sanitize_config`); no inbound validation; no redirect URL validation | User (user chooses which server to trust) | No inbound schema validation; API key forwarded on redirects |
| Configuration (langgraph.json) | DF6, DF7 | T5 | JSON schema validation (`libs/cli/langgraph_cli/config.py:validate_config_file`), build command content validation (`libs/cli/langgraph_cli/config.py:has_disallowed_build_command_content`), list-based subprocess args, WebhookUrlPolicy (`libs/cli/langgraph_cli/schemas.py:WebhookUrlPolicy`) | User (developer-controlled file) | Single-quote not escaped in ENV embedding |
| Configuration (env vars) | DF10 | T7, T8 | AES key length validation, EAX MAC verification | User (deployer controls env) | Key entropy, assert-based check |
| Developer encryption handlers | DF15 | — | Handler signature validation (`libs/sdk-py/langgraph_sdk/encryption/__init__.py:_validate_handler`), duplicate prevention | User (developer-authored code) | `context()` handler missing `_validate_handler` call |

---

## Out-of-Scope Threats

Threats that appear valid in isolation but fall outside project responsibility because they depend on conditions the project does not control.

| Pattern | Why Out of Scope | Project Responsibility Ends At |
|---------|-----------------|-------------------------------|
| Prompt injection leading to arbitrary tool execution | Project does not control LLM model behavior, user prompt construction, or which tools are registered. ToolNode routes by name only to user-registered tools. | Providing tool name allowlist routing (`libs/prebuilt/langgraph/prebuilt/tool_node.py:ToolNode._validate_tool_call`); user owns tool registration and argument handling |
| State poisoning via malicious node output | User-registered nodes (including `@task`-decorated functions) can write arbitrary values to channels. The framework executes nodes as provided. | Enforcing channel type contracts (`libs/langgraph/langgraph/channels/base.py:BaseChannel.update`); user owns node implementation correctness |
| Cross-session state access via thread_id guessing | Checkpoint savers index by `thread_id`. Without application-level auth, any caller with a valid thread_id can access that thread's state. | Providing the `Auth` handler system for access control (`libs/sdk-py/langgraph_sdk/auth/__init__.py:Auth`); user must implement auth handlers |
| Tool shadowing via duplicate registration | If a user registers two tools with the same name, ToolNode uses the last one. This is user misconfiguration. | Documenting tool registration semantics |
| Indirect prompt injection via tool output | LLM reads tool output and may follow injected instructions. This is a fundamental LLM limitation, not a framework vulnerability. | Not including tool output in system prompts; user owns output handling |
| Model selecting dangerous tool arguments | An LLM may generate SQL injection, path traversal, or command injection payloads as tool arguments. The risk depends entirely on what the user's tools do with those arguments. | Routing tool calls to registered tools only; user owns tool input validation |
| RCE via user-provided node code | `add_node()` and `@entrypoint`/`@task` accept arbitrary callables. A malicious node can do anything. This is by design — the user controls their own code. | Executing nodes within the graph runtime; user owns node code safety |
| SSRF via RemoteGraph URL | User provides the `url` parameter to `RemoteGraph`. Pointing it at an internal service is the user's decision. | Documenting that `url` should be a trusted endpoint; user owns URL selection |
| Malicious SDK Encryption handler | Encryption handlers are developer-authored server-side code. A malicious handler has full process access, equivalent to any application code. | Validating handler signature (async, param count); handler behavior is the developer's responsibility |

### Rationale

**Prompt injection and tool execution**: LangGraph's `ToolNode` validates tool names against the registered set but does not inspect or sanitize argument values. This is the correct boundary — the framework cannot know what constitutes a "safe" argument for an arbitrary user-defined tool. The tool's own Pydantic schema and implementation must validate inputs. The framework's responsibility is to not execute unregistered tools and to correctly route registered ones. The injection system (InjectedState/InjectedStore/ToolRuntime) is safe because system-injected values always overwrite LLM-supplied collisions via dict merge order, and injected parameter names are hidden from the LLM's tool schema.

**State integrity**: LangGraph channels enforce type contracts (e.g., `LastValue` accepts one value per step, `BinaryOperatorAggregate` applies a reducer). The framework validates graph structure at compile time (`libs/langgraph/langgraph/pregel/_validate.py:validate_graph`). However, the semantic correctness of state updates is the user's responsibility — the framework cannot know what values are "valid" for a user-defined state schema.

**Checkpoint access control**: The framework provides `BaseCheckpointSaver` as an abstract interface and the `Auth` handler system for authorization (`libs/sdk-py/langgraph_sdk/auth/__init__.py:Auth`). It does not enforce authentication by default because it operates as a library, not a server. The `langgraph-api` server layer (out of scope) is responsible for enforcing auth on API endpoints. Users embedding LangGraph directly must implement their own access controls.

**Encryption handler safety**: The SDK Encryption module (`libs/sdk-py/langgraph_sdk/encryption/`) provides a registration framework for developer-authored encryption handlers. These handlers run server-side with full process access, identical to any application code. A buggy or malicious handler could return crafted data, but this is the same trust model as any developer-written code. The SDK validates handler shape (async, 2-param) but not handler behavior — this is the correct boundary for developer-trust-level code.

---

## Investigated and Dismissed

| ID | Original Threat | Investigation | Evidence | Conclusion |
|----|----------------|---------------|----------|------------|
| D1 | SQL injection via filter keys in PostgreSQL store | Traced filter key handling through `libs/checkpoint-postgres/langgraph/store/postgres/base.py:_get_filter_condition`. All filter operations use parameterized queries with `%s` placeholders. Keys map to `json_extract` path operators with type-safe wrappers. | `libs/checkpoint-postgres/langgraph/store/postgres/base.py:_get_filter_condition` — parameterized `%s` for all value bindings; key names used in `value->%s` path expressions are also parameterized | Disproven: All SQL operations in PostgreSQL store are fully parameterized. No injection vector. |
| D2 | SQL injection via filter keys in SQLite store (post-fix) | Traced current filter handling through `libs/checkpoint-sqlite/langgraph/store/sqlite/base.py` and `libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/utils.py:_validate_filter_key`. Regex `^[a-zA-Z0-9_.-]+$` applied to all filter keys before use in `json_extract()` expressions. | `libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/utils.py:_validate_filter_key` — regex validation blocks injection characters. Published advisories GHSA-9rwj-6rc7-p77c and GHSA-7p73-8jqx-23r8 confirmed fixed. | Disproven: SQL injection in SQLite store filter keys is remediated by regex validation. |
| D3 | Command injection via CLI subprocess execution | Traced CLI subprocess invocation path. `libs/cli/langgraph_cli/exec.py:subp_exec` uses `asyncio.create_subprocess_exec` with list-based arguments (not `shell=True`). `has_disallowed_build_command_content` blocks shell metacharacters in user-provided Dockerfile lines. | `libs/cli/langgraph_cli/exec.py:subp_exec` — explicit exec-style invocation; `libs/cli/langgraph_cli/config.py:has_disallowed_build_command_content` — regex blocks `\|`, `;`, `$`, `>`, `<`, backtick, `\`, single `&` | Disproven: CLI uses exec-style subprocess and validates build command content. No shell injection vector. |
| D4 | Tool argument injection via InjectedState/InjectedStore dict-splatting | Investigated whether LLM-generated tool call arguments could override system-injected values (InjectedState, InjectedStore, ToolRuntime) via key collision in the dict merge at `libs/prebuilt/langgraph/prebuilt/tool_node.py:ToolNode._inject_tool_args` line 1380. Traced four independent defense layers. | (1) `tool_call_schema` at langchain-core `base.py` filters injected params from LLM schema. (2) `{**llm_args, **injected_args}` merge puts system values last — system wins on collision. (3) Pydantic `model_validate` with `extra="ignore"` drops unknown keys. (4) Output construction at `base.py` only includes declared model fields. | Disproven: Four-layer defense prevents LLM arguments from overriding system-injected values. Merge order guarantees system values win. No adversarial collision path exists. |

---

## External Context

### Published Security Advisories

| GHSA ID | Severity | Summary | CWEs | Relevance |
|---------|----------|---------|------|-----------|
| GHSA-g48c-2wqr-h844 | Medium | Unsafe msgpack deserialization in LangGraph checkpoint loading | — | Directly relates to T1 — patched in 1.0.10, confirms attack path via crafted msgpack payloads |
| GHSA-mhr3-j7m5-c7c9 | Medium | BaseCache Deserialization RCE | CWE-502 | Directly relates to T1 — msgpack deserialization in cache layer |
| GHSA-9rwj-6rc7-p77c | High | SQL injection via metadata filter key in SQLite checkpointer | CWE-89 | Fixed via `_validate_filter_key()` regex — see D2 |
| GHSA-wwqv-p2pp-99h5 | High | RCE in JSON mode of JsonPlusSerializer | CWE-502 | Directly relates to T3 — `lc:2` constructor import |
| GHSA-7p73-8jqx-23r8 | High | SQLite Filter Key SQL Injection in SqliteStore | CWE-89 | Fixed via `_validate_filter_key()` regex — see D2 |

**Pattern**: 3 of 5 published advisories involve CWE-502 (insecure deserialization) in the checkpoint serialization layer. This confirms the checkpoint storage boundary (TB2) as the highest-risk area. The extensive closed advisory history (~15 deserialization bypass attempts) further validates this assessment. No new published advisories since the prior assessment (2026-03-27).


---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-03-04 | Generated | Initial threat model |
| 2026-03-04 | Updated | Added C13 (Functional API), C14 (BaseCache), DF11. Updated T1 for BaseCache/serde event hooks. Added GHSA-mhr3-j7m5-c7c9 and GHSA-9rwj-6rc7-p77c. Updated CLI config scope. Added External Context section. |
| 2026-03-27 | Deep refinement | **Mode upgraded to Deep.** Added: Data Classification section (DC1-DC8 with detailed analysis for Critical/High entries). Added: C15 (Serde Event Hooks), C16 (Auth System). Added: Default? column to Components. Added: Classification column to Data Flows. Added: DF12-DF13 (SDK redirect flows). Added: T9 (SDK API key leak via Location redirect), T10 (EncryptedSerializer encryption bypass), T11 (unbounded checkpoint retention). Added: Validation column to Threats with flaw validation for High/Critical. Added: Investigated and Dismissed section (D1-D3: SQL injection and CLI command injection disproven). Added: Input Source Coverage section. Updated external context with GHSA-g48c-2wqr-h844 (new published advisory). Updated all code references to file:SymbolName notation. Expanded trust boundary details. |
| 2026-03-30 | Diagram and Default? corrections | Fixed architecture diagram: merged "User Code" and "User-Registered Tools" into single "User Application Code" boundary; removed @entrypoint/@task as separate diagram elements (both compile to Pregel — authoring style, not separate component). Fixed Default? column: C3 ToolNode → No (explicit opt-in required); C8 CLI → No (separate install). |
| 2026-03-28 | Deep update | **Added:** C17 (SDK Encryption Handlers — beta at-rest encryption framework). DC9 (SDK Encryption context metadata). TB5 (SDK Encryption Handler boundary). DF14 (ToolRuntime injection flow), DF15 (Encryption handler registration flow). D4 (Tool argument injection via InjectedState dict-splatting — disproven with 4-layer defense evidence). **Updated:** C1 description (v1/v2 output, durability modes, interrupt_before/after). C2 description (SAFE_MSGPACK_TYPES now 47 entries including langchain_core messages, Document, GetOp). C3 description (InjectedState/InjectedStore/ToolRuntime injection support, _inject_tool_args entry point). C8 description (WebhookUrlPolicy for SSRF protection). TB1 details (tool injection merge order guarantees). TB2 details (47 safe types, updated allowlist composition). TB4 details (WebhookUrlPolicy). DF4 description (injection merge semantics). T1 details (noted secure_pickle.py proposed but never merged). T4 details (Interrupt deprecated_kwargs ns parameter). Input Source Coverage (LLM output row updated with injection validation points, encryption handler row added). Out-of-Scope Threats (malicious encryption handler pattern added). Commit updated to 0ba22143. External context confirmed no new published advisories. |
