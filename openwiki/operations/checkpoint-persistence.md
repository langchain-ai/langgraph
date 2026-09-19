---
type: Persistence & Durability
title: Checkpoint Persistence
description: Setup and operation of checkpoint savers for durable execution and state management across graph invocations and resumptions.
tags: [checkpointing, persistence, durability, state-snapshot, resumption, thread-id, setup, maintenance]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-81563e23823dc9fcbda0c3a7
    resource: repo://libs/checkpoint-postgres/langgraph/checkpoint/postgres/__init__.py
  - id: openwiki-source-4b5bfe9ada53f6d0bb86d81c
    resource: repo://libs/checkpoint-postgres/README.md
  - id: openwiki-source-6b78a7cf5b2c23036826b38f
    resource: repo://libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/__init__.py
  - id: openwiki-source-9bc37d8be5ff3801f852d013
    resource: repo://libs/checkpoint-sqlite/README.md
  - id: openwiki-source-6c95109f667df245389a281a
    resource: repo://libs/checkpoint/langgraph/checkpoint/base/__init__.py
  - id: openwiki-source-0bb72cb2f8e9b84c7909edba
    resource: repo://libs/checkpoint/langgraph/checkpoint/memory/__init__.py
  - id: openwiki-source-16071c666268b16a8eb57a30
    resource: repo://libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py
  - id: openwiki-source-d938b485005732564a37e4a1
    resource: repo://libs/checkpoint/README.md
  - id: openwiki-source-36058e684ab11833a28af83b
    resource: repo://libs/langgraph/langgraph/graph/state.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

Checkpoint persistence in LangGraph enables graphs to maintain durable state across invocations, pause and resume execution at any point, and support time-travel debugging by replaying from historical checkpoints. Checkpoint savers provide a pluggable interface for storing and retrieving graph snapshots, with built-in implementations for in-memory (development), SQLite (lightweight), and PostgreSQL (production) backends.

To enable persistence, a checkpoint saver instance must be passed to the graph's `compile()` method, and invocations must include a `thread_id` in the configuration's `configurable` dict. The thread ID is the primary key for isolating execution contexts—use unique thread IDs for independent runs or reuse the same ID to accumulate state across multiple invocations (e.g., conversational memory).

---

## Core Concepts

### Checkpoint ID and Thread ID

- **Thread ID** (`thread_id`): Primary key isolating execution contexts. Passed in config as `{"configurable": {"thread_id": "user-123"}}`. Reusing the same thread ID accumulates state across invocations; unique IDs per run yield independent traces.
- **Checkpoint ID** (`checkpoint_id`): Unique, monotonically increasing identifier (UUID6 by default) for a snapshot. Enables time-travel: invoke with a previous checkpoint ID to resume from that historical point rather than the latest state.
- **Checkpoint Namespace** (`checkpoint_ns`): Isolates state for subgraphs and nested contexts (formatted as pipe-separated hierarchy, e.g., `"outer|inner"`). Root graph uses empty string `""`. Allows multiple independent checkpoint chains on a single thread.

### Configuration and State

Checkpoints are identified and retrieved by a **RunnableConfig** (dict) containing:

```python
config = {
    "configurable": {
        "thread_id": "user-123",           # Required for persistence
        "checkpoint_ns": "",                # Optional; root uses ""
        "checkpoint_id": "uuid6-...",       # Optional; specify historical checkpoint for time-travel
    }
}
```

The graph's `invoke()`, `stream()`, or `batch()` methods accept this config to determine which checkpoint thread and starting point to use.

---

## Setup Flow

### 1. Create a Checkpointer Instance

Choose based on your deployment:

```python
# In-memory (development/testing only)
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# SQLite (lightweight, file-based)
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    # Use checkpointer

# PostgreSQL (production-grade)
from langgraph.checkpoint.postgres import PostgresSaver
DB_URI = "postgresql://user:password@localhost/dbname"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Use checkpointer
```

### 2. Call `.setup()` (For SQL-backed Savers)

Before first use, initialize the database schema. InMemorySaver requires no setup.

```python
# SQLite
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    # setup() is called automatically on first use, but can be called explicitly
    checkpointer.setup()
    # ... use checkpointer

# PostgreSQL (must be called explicitly)
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # Creates tables and runs migrations
    # ... use checkpointer
```

**Important PostgreSQL Details:**
- **Connection Requirements**: When creating a `PostgresSaver` manually, connection **must** have `autocommit=True` and `row_factory=dict_row` (from `psycopg.rows`):
  ```python
  from psycopg import Connection
  from psycopg.rows import dict_row
  
  conn = Connection.connect(
      DB_URI,
      autocommit=True,        # Required for .setup() to persist table creation
      row_factory=dict_row    # Required for dict-style row access
  )
  checkpointer = PostgresSaver(conn)
  checkpointer.setup()
  ```
- Without these parameters, `.setup()` may not persist tables, and read operations will fail with `TypeError` when accessing columns by name.

### 3. Compile the Graph

Pass the checkpointer to `compile()`:

```python
from langgraph.graph import StateGraph

builder = StateGraph(MyState)
builder.add_node("my_node", my_function)
builder.set_entry_point("my_node")
builder.set_finish_point("my_node")

checkpointer = InMemorySaver()  # or SQLiteSaver, PostgresSaver
graph = builder.compile(checkpointer=checkpointer)
```

### 4. Invoke with Thread ID

Always pass a `thread_id` when checkpointer is enabled:

```python
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"input": "data"}, config)

# State persists; subsequent invocations on same thread resume from latest checkpoint
result = graph.invoke({"input": "more data"}, config)
```

---

## Checkpoint Implementations

### InMemorySaver

**Location**: `langgraph.checkpoint.memory.InMemorySaver`

**Characteristics**:
- Stores checkpoints in process memory using nested dicts.
- No setup required; `setup()` is a no-op.
- Lost on process restart.
- Suitable for development, testing, and prototyping only.

**Use Cases**:
- Unit tests and integration tests.
- Local debugging and experimentation.
- Single-machine prototypes.

**Limitations**:
- Not suitable for production.
- No persistence across restarts.
- Unbounded memory growth without cleanup.

**Example**:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
graph.invoke({"input": "test"}, {"configurable": {"thread_id": "test-1"}})
```

### SQLiteSaver

**Location**: `langgraph.checkpoint.sqlite.SqliteSaver` (sync) / `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver` (async)

**Characteristics**:
- File-based persistence using SQLite.
- Lighter than PostgreSQL; simpler setup.
- Suitable for small to medium deployments.
- Supports both sync and async execution.

**Setup**:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# File-based
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    checkpointer.setup()  # Called automatically, but explicit call OK
    graph = builder.compile(checkpointer=checkpointer)
    # Use graph

# In-memory (testing)
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

**Database Schema**:
- `checkpoints`: Stores checkpoint snapshots with thread ID, namespace, checkpoint ID, parent reference, and serialized checkpoint data.
- `checkpoint_writes`: Stores pending writes (intermediate state changes from tasks that succeeded but whose checkpoint save failed).

**Async Support**:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    await checkpointer.aput(config, checkpoint, {}, {})
    result = await checkpointer.aget_tuple(config)
```

### PostgresSaver

**Location**: `langgraph.checkpoint.postgres.PostgresSaver` (sync) / `langgraph.checkpoint.postgres.aio.AsyncPostgresSaver` (async)

**Characteristics**:
- Enterprise-grade persistence using PostgreSQL.
- Enables multi-tenant deployments and high-availability architectures.
- Optimized for large-scale checkpoint storage and retrieval.
- Supports concurrent async access via `AsyncPostgresSaver`.

**Installation**:

```bash
pip install langgraph-checkpoint-postgres
```

**Setup**:

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/dbname"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # Creates checkpoints, checkpoint_writes, and checkpoint_migrations tables
    graph = builder.compile(checkpointer=checkpointer)
    # Use graph
```

**Database Schema**:
- `checkpoints`: Stores checkpoint snapshots (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint blob, metadata).
- `checkpoint_writes`: Stores pending writes for recovery of partially completed supersteps.
- `checkpoint_migrations`: Tracks applied migrations for schema versioning.

**Migrations**:
The `setup()` method is idempotent and handles automatic schema migrations. It checks the migration version and applies any pending migrations:

```python
def setup(self) -> None:
    """Initializes the database and applies any pending migrations."""
    with self._cursor() as cur:
        # Create migration tracking table
        cur.execute(self.MIGRATIONS[0])
        # Fetch current version
        row = cur.fetchone()
        version = row["v"] if row else -1
        # Apply new migrations
        for v, migration in zip(range(version + 1, len(self.MIGRATIONS)), ...):
            cur.execute(migration)
            cur.execute("INSERT INTO checkpoint_migrations (v) VALUES (%s)", (v,))
```

**Async Support**:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()
    await checkpointer.aput(config, checkpoint, {}, {})
    result = await checkpointer.aget_tuple(config)
```

---

## Core Operations

### Storing Checkpoints

Checkpoints are stored automatically during graph execution. The `.put()` method is called by the execution loop at each superstep (depending on `durability` mode):

```python
checkpointer.put(
    config,              # RunnableConfig with thread_id, checkpoint_ns, checkpoint_id
    checkpoint,          # Checkpoint dict with v, ts, id, channel_values, channel_versions, ...
    metadata,            # CheckpointMetadata with source, step, parents, ...
    new_versions,        # ChannelVersions dict for version tracking
) -> RunnableConfig     # Returns config with updated checkpoint_id
```

Manual checkpoint storage is rarely needed; the execution loop handles it.

### Retrieving Checkpoints

Fetch the latest checkpoint on a thread:

```python
config = {"configurable": {"thread_id": "user-123"}}
checkpoint_tuple = checkpointer.get_tuple(config)
if checkpoint_tuple:
    checkpoint = checkpoint_tuple.checkpoint          # Full snapshot
    metadata = checkpoint_tuple.metadata              # Metadata (source, step, ...)
    parent_config = checkpoint_tuple.parent_config    # Parent checkpoint config
    pending_writes = checkpoint_tuple.pending_writes  # Writes to replay on resume
```

Retrieve a specific historical checkpoint:

```python
config = {
    "configurable": {
        "thread_id": "user-123",
        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875"
    }
}
checkpoint_tuple = checkpointer.get_tuple(config)  # Fetches exact checkpoint
```

### Listing Checkpoints

Enumerate all checkpoints on a thread:

```python
config = {"configurable": {"thread_id": "user-123"}}
for checkpoint_tuple in checkpointer.list(config, limit=10):
    print(f"Step {checkpoint_tuple.metadata['step']}: {checkpoint_tuple.checkpoint['id']}")
```

Filter by metadata or retrieve checkpoints before a specific point:

```python
# List before a specific checkpoint (for navigation)
before_config = {
    "configurable": {
        "checkpoint_id": "some-checkpoint-id"
    }
}
earlier_checkpoints = list(checkpointer.list(
    config,
    before=before_config,
    limit=5
))
```

### Thread Management

Delete all checkpoints and writes for a thread:

```python
checkpointer.delete_thread("user-123")  # Cleanup after user deletes account, etc.
```

Async variant:

```python
await checkpointer.adelete_thread("user-123")
```

---

## Pending Writes and Fault Recovery

When a node task completes successfully but the checkpoint save fails, the node's writes (state updates) are stored as **pending writes**. On resumption, these writes are replayed before scheduling the next tasks, ensuring side effects are not lost due to downstream failures.

### PendingWrite Structure

A pending write is a tuple: `(task_id, channel_name, value)`.

Example: Node "fetch_data" completes and updates channel "messages" with a new message. If the checkpoint save fails, this write is stored. On resumption:

```python
pending_writes = checkpoint_tuple.pending_writes
# Each is (task_id, channel_name, value)
# Execution loop replays: messages.put(value)
```

### Storage

Pending writes are stored via `.put_writes()`:

```python
checkpointer.put_writes(
    config,              # RunnableConfig
    writes=[
        ("task-1", "messages", new_message_value),
        ("task-1", "state", updated_state),
    ],
    task_id="task-1",    # ID of the task that generated these writes
    task_path="fetch_data"
)
```

The execution loop handles this automatically; manual calls are rarely needed.

---

## Serialization and Security

### Serializer Configuration

Checkpoints are serialized using `JsonPlusSerializer` by default, which encodes to msgpack with fallback support for LangChain types (messages, tools, dates, etc.).

```python
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

serde = JsonPlusSerializer(
    pickle_fallback=False,  # Disable pickle for security
    allowed_msgpack_modules=[
        ("my_app", "custom_type"),
        ("langchain_core.messages", "HumanMessage"),
    ]
)
checkpointer = InMemorySaver(serde=serde)
```

### Security: Strict Msgpack Mode

**Critical**: Deserialization of untrusted checkpoint data can execute arbitrary code. Enable **strict mode** to restrict deserialization to safe types:

```python
import os
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"  # Restrict to allowlist

# Or pass explicit allowlist
serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("langchain_core.messages", "BaseMessage"),
        ("langchain_core.messages", "HumanMessage"),
        ("langchain_core.messages", "AIMessage"),
    ]
)
```

When `LANGGRAPH_STRICT_MSGPACK=true`, only types in the built-in allowlist (`SAFE_MSGPACK_TYPES`) can be deserialized. Custom types must be explicitly allowed via `allowed_msgpack_modules`.

### Encrypted Storage

For additional security, wrap the serializer with `EncryptedSerializer`:

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

serde = JsonPlusSerializer()
encrypted_serde = EncryptedSerializer(
    serde,
    encryption_key=b"your-secret-key-32-bytes"  # Must be 32 bytes
)
checkpointer = PostgresSaver(conn, serde=encrypted_serde)
```

---

## Maintenance and Operations

### Listing Threads and Checkpoints

Query all checkpoints on a thread to understand execution history:

```python
config = {"configurable": {"thread_id": "user-123"}}
checkpoints = list(checkpointer.list(config))
for cp in checkpoints:
    print(f"Checkpoint {cp.checkpoint['id']}: step {cp.metadata['step']}")
    print(f"  Channels updated: {cp.checkpoint['updated_channels']}")
    print(f"  Source: {cp.metadata['source']}")  # 'input', 'loop', 'update', 'fork'
```

### Pruning and Cleanup

Remove old checkpoints to reclaim storage. Implementations provide `.prune()` with strategies:

```python
# Keep only the latest checkpoint per namespace
checkpointer.prune(
    thread_ids=["user-123", "user-456"],
    strategy="keep_latest"
)

# Delete all checkpoints for cleanup
checkpointer.prune(
    thread_ids=["user-123"],
    strategy="delete"
)
```

**Delta Channel Caveat**: If the graph uses `DeltaChannel` (beta feature for efficient incremental state updates), naive pruning of intermediate checkpoints can break reconstruction because delta channels store only incremental deltas and require ancestor checkpoints containing snapshot blobs. Safe pruning strategies:

1. Walk back from kept checkpoints and preserve all ancestors up to the nearest `_DeltaSnapshot` ancestor for each delta channel.
2. Force a fresh snapshot on the kept checkpoint before pruning (rewrite `channel_values[k] = _DeltaSnapshot(value)`).
3. Skip pruning threads with delta channels until one of the above is implemented.

### Vacuuming and Archival

For PostgreSQL, periodically reclaim space from deleted checkpoints:

```python
# Manual VACUUM
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    with checkpointer._cursor() as cur:
        cur.execute("VACUUM ANALYZE")
```

For archival of old checkpoints:

```python
# Export old checkpoints to external storage, then delete
for checkpoint_tuple in checkpointer.list(config, limit=1000):
    if checkpoint_tuple.metadata['step'] < 100:  # Old checkpoint
        # Archive to S3, GCS, etc.
        archive.put(checkpoint_tuple)
```

SQLite handles vacuum automatically but can be triggered explicitly:

```python
conn.execute("VACUUM")
```

### Monitoring Checkpoint Size

Monitor storage usage:

```python
import sqlite3

# SQLite
conn = sqlite3.connect("checkpoints.db")
size = conn.execute("SELECT page_count * page_size / 1024 / 1024 FROM pragma_page_count(), pragma_page_size()").fetchone()[0]
print(f"Database size: {size} MB")

# PostgreSQL
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    with checkpointer._cursor() as cur:
        cur.execute("SELECT pg_size_pretty(pg_total_relation_size('checkpoints'))")
        size = cur.fetchone()[0]
        print(f"Checkpoints table size: {size}")
```

---

## Configuration Examples

### Development: In-Memory Saver

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

checkpointer = InMemorySaver()
graph = StateGraph(MyState).compile(checkpointer=checkpointer)

# Use without explicit setup
config = {"configurable": {"thread_id": "test-1"}}
result = graph.invoke({"input": "hello"}, config)
```

### Local Development: SQLite

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

with SqliteSaver.from_conn_string("local_checkpoints.db") as checkpointer:
    checkpointer.setup()
    graph = StateGraph(MyState).compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "dev-session"}}
    result = graph.invoke({"input": "test"}, config)
```

### Production: PostgreSQL with Async Support

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph

DB_URI = "postgresql://user:password@localhost:5432/langgraph_db"

async def setup():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        print("PostgreSQL checkpoint tables created")

async def run_graph():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        graph = StateGraph(MyState).compile(checkpointer=checkpointer)
        
        config = {"configurable": {"thread_id": "user-12345"}}
        result = await graph.ainvoke({"input": "data"}, config)
        return result

# First time
await setup()

# Then run
result = await run_graph()
```

### Production: PostgreSQL with Strict Serialization

```python
import os
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Enable strict mode
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

# Or pass explicit allowlist
serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("langchain_core.messages", "BaseMessage"),
        ("langchain_core.messages", "HumanMessage"),
        ("my_app.types", "AgentState"),
    ]
)

DB_URI = "postgresql://user:password@localhost:5432/langgraph_db"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    # serde is JsonPlusSerializer by default; override if needed
    graph = StateGraph(MyState).compile(checkpointer=checkpointer)
```

---

## Resumption and Time-Travel

### Resume from Interrupt

When a graph is interrupted (e.g., awaiting human input), resume with the same `thread_id`:

```python
from langgraph.errors import GraphInterrupt

config = {"configurable": {"thread_id": "user-123"}}

# First invocation hits an interrupt
try:
    result = graph.invoke({"input": "query"}, config)
except GraphInterrupt:
    # State is checkpointed; can resume later
    pass

# Later, resume with human input
resume_value = {"human_input": "approved"}
result = graph.invoke(resume_value, config)  # Resumes from last checkpoint
```

### Time-Travel: Resume from Historical Checkpoint

List checkpoints and replay from any point:

```python
# List all checkpoints
config = {"configurable": {"thread_id": "user-123"}}
checkpoints = list(checkpointer.list(config))

# Choose a historical checkpoint
for cp in checkpoints:
    if cp.metadata['step'] == 5:  # Resume from step 5
        # Invoke with that checkpoint ID
        resume_config = {
            "configurable": {
                "thread_id": "user-123",
                "checkpoint_id": cp.checkpoint['id']
            }
        }
        result = graph.invoke({"debug_input": "test"}, resume_config)
        break
```

---

## Advanced Patterns

### Multi-Tenant State Isolation

Use `thread_id` to isolate per-user or per-session state:

```python
# Each user has their own thread
for user_id in ["user-1", "user-2", "user-3"]:
    config = {"configurable": {"thread_id": f"user-{user_id}"}}
    graph.invoke({"input": "hello"}, config)
    # Each user's state is isolated in checkpoints
```

### Subgraph Persistence

Subgraphs can have independent checkpoints via `checkpoint_ns`:

```python
# Root graph
root_graph = StateGraph(State).compile(checkpointer=root_checkpointer)

# Subgraph with its own namespace
subgraph = StateGraph(SubState).compile(checkpointer=sub_checkpointer)

# When root calls subgraph, subgraph uses nested namespace (e.g., "root|sub")
# Subgraph state is isolated in checkpoint_writes per namespace
```

### Forking and Copying Threads

Copy an entire checkpoint chain to a new thread for branching:

```python
source_thread_id = "user-123"
target_thread_id = "user-123-backup"

checkpointer.copy_thread(source_thread_id, target_thread_id)

# New thread has identical checkpoint history
config = {"configurable": {"thread_id": target_thread_id}}
graph.invoke({"input": "diverge here"}, config)
```

Async:

```python
await checkpointer.acopy_thread(source_thread_id, target_thread_id)
```

---

## Troubleshooting

### PostgreSQL Connection Errors

**Issue**: `TypeError: tuple indices must be integers or slices, not str`

**Cause**: Connection missing `row_factory=dict_row` or `autocommit=True`.

**Solution**:

```python
from psycopg import Connection
from psycopg.rows import dict_row

conn = Connection.connect(
    DB_URI,
    autocommit=True,
    row_factory=dict_row
)
checkpointer = PostgresSaver(conn)
```

### Tables Not Created

**Issue**: `.setup()` appears to run but tables are missing.

**Cause**: Connection missing `autocommit=True`.

**Solution**: Ensure connection has `autocommit=True`:

```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # Uses autocommit=True internally
```

### Serialization Errors

**Issue**: `ValueError: Type not allowed in msgpack: <custom type>`

**Cause**: Custom types not in allowlist when `LANGGRAPH_STRICT_MSGPACK=true`.

**Solution**: Add types to allowlist:

```python
serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("my_app.types", "CustomState"),
    ]
)
checkpointer = InMemorySaver(serde=serde)
```

### SQLite Locking

**Issue**: `database is locked` under concurrent access.

**Cause**: SQLite has limited concurrent write support.

**Solution**: For concurrent access, use PostgreSQL, or ensure single-writer pattern with SQLite.

---

## Related Concepts

- **Durability Modes**: `sync`, `async`, `exit` determine checkpoint frequency and blocking behavior during graph execution (see `checkpointing-and-memory.md`).
- **Delta Channels**: Efficient incremental state for frequently-updated channels; requires careful pruning to maintain ancestor chain.
- **Interrupts**: Pausing graph execution to await human input; checkpoints enable seamless resumption.
- **Thread and Checkpoint Namespacing**: Enables multi-tenant and nested subgraph scenarios.
