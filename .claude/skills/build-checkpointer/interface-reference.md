# Checkpointer Interface Reference

## Imports

```python
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig
```

## Data Structures

```python
# RunnableConfig["configurable"] keys:
#   thread_id: str        — identifies the conversation thread
#   checkpoint_ns: str    — namespace (empty string for root, dotted path for subgraphs)
#   checkpoint_id: str    — unique monotonically-increasing ID (UUID-like, sortable)

# Checkpoint (TypedDict):
#   v: int                — format version (currently 1)
#   id: str               — unique checkpoint ID
#   ts: str               — ISO 8601 timestamp
#   channel_values: dict[str, Any]   — serialized state per channel
#   channel_versions: ChannelVersions — version number per channel
#   versions_seen: dict[str, ChannelVersions] — per-node version tracking

# CheckpointMetadata (TypedDict):
#   source: str           — "input" | "loop" | "update" | "fork"
#   step: int             — -1 for input, 0+ for loop steps
#   parents: dict[str, str] — parent checkpoint IDs
#   (plus any custom keys the caller adds)

# CheckpointTuple (NamedTuple):
#   config: RunnableConfig
#   checkpoint: Checkpoint
#   metadata: CheckpointMetadata
#   parent_config: RunnableConfig | None
#   pending_writes: list[tuple[str, str, Any]] | None
#     Each write is (task_id, channel, value)

# ChannelVersions = dict[str, Any]  (typically str or int version numbers)

# WRITES_IDX_MAP = {"__error__": -1, "__scheduled__": -2, "__interrupt__": -3, "__resume__": -4}
```

## Method Signatures

### Required Methods

```python
async def aput(
    self,
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    new_versions: ChannelVersions,
) -> RunnableConfig:
    """Store a checkpoint. Return config with checkpoint_id set to checkpoint["id"].

    The incoming config["configurable"]["checkpoint_id"] is the PARENT checkpoint ID.
    new_versions contains only the channels that changed — but checkpoint["channel_values"]
    has ALL channels. Store the full checkpoint.
    """

async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
    """Retrieve a checkpoint.

    If config has checkpoint_id: return that exact checkpoint.
    If no checkpoint_id: return the LATEST checkpoint for the thread+namespace.
    Return None if not found.
    Include pending_writes as list of (task_id, channel, value) ordered by (task_id, idx).
    """

async def alist(
    self,
    config: RunnableConfig | None,
    *,
    filter: dict[str, Any] | None = None,
    before: RunnableConfig | None = None,
    limit: int | None = None,
) -> AsyncIterator[CheckpointTuple]:
    """List checkpoints, newest first (descending checkpoint_id).

    If config is None: search ALL threads (global search).
    If config has thread_id: filter to that thread.
    filter: dict of metadata key-value pairs (AND logic).
    before: only return checkpoints before this checkpoint_id.
    limit: max number to return.
    Each yielded tuple must include pending_writes.
    """

async def aput_writes(
    self,
    config: RunnableConfig,
    writes: Sequence[tuple[str, Any]],
    task_id: str,
    task_path: str = "",
) -> None:
    """Store pending writes for a checkpoint.

    Each write is (channel, value). Use WRITES_IDX_MAP.get(channel, idx) for the index.
    Special channels (in WRITES_IDX_MAP) should UPSERT (replace on conflict).
    Regular channels should be idempotent (ignore on conflict).
    """

async def adelete_thread(self, thread_id: str) -> None:
    """Delete ALL checkpoints and writes for a thread (all namespaces)."""
```

### Extended Methods

```python
async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
    """Delete checkpoints+writes where metadata.run_id is in run_ids."""

async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
    """Copy all checkpoints+writes from source thread to target thread."""

async def aprune(
    self,
    thread_ids: Sequence[str],
    *,
    strategy: str = "keep_latest",
) -> None:
    """Prune checkpoints for given threads.
    strategy="keep_latest": keep only the latest checkpoint per thread+namespace.
    strategy="delete_all": delete everything for those threads.
    """
```

## Conformance Test Harness Template

Create `tests/test_conformance.py`:

```python
"""Conformance tests for <Backend>Saver."""
from __future__ import annotations

import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.report import ProgressCallbacks

# Import your checkpointer
from langgraph.checkpoint.<backend> import <Backend>Saver


# Optional: lifespan for one-time setup/teardown (database creation, etc.)
# async def backend_lifespan():
#     # setup
#     yield
#     # teardown


@checkpointer_test(name="<Backend>Saver")  # add lifespan=backend_lifespan if needed
async def backend_checkpointer():
    # Create and yield a fresh checkpointer instance.
    # Use async with if your saver needs connection management.
    saver = <Backend>Saver(...)
    yield saver
    # cleanup (close connections, etc.)


@pytest.mark.asyncio
async def test_full_conformance():
    """<Backend>Saver passes ALL conformance tests."""
    report = await validate(
        backend_checkpointer,
        progress=ProgressCallbacks.verbose(),
    )
    report.print_report()
    assert report.passed_all(), f"Conformance failed: {report.to_dict()}"
```

## Serialization Pattern

```python
# In __init__:
super().__init__(serde=serde)

# Storing metadata (use JSON, not serde):
merged = get_checkpoint_metadata(config, metadata)
serialized_md = json.dumps(merged).encode("utf-8")

# Loading metadata:
metadata = json.loads(serialized_md_bytes)

# Storing/loading blob values and write values (use serde):
type_, serialized = self.serde.dumps_typed(value)
value = self.serde.loads_typed((type_, serialized_bytes))

# For CPU-bound serde in async context, offload to thread:
type_, serialized = await asyncio.to_thread(self.serde.dumps_typed, value)
value = await asyncio.to_thread(self.serde.loads_typed, (type_, serialized_bytes))
```

## Schema Design by Backend Type

All backends must store checkpoints keyed by `(thread_id, checkpoint_ns, checkpoint_id)` and writes keyed by `(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)`.

### SQL backends (Postgres, MySQL, SQLite)

Use 3 tables: **checkpoints** (checkpoint JSON with primitive channel_values inlined + channel_versions for blob lookup, metadata JSON), **checkpoint_blobs** (non-primitive channel values keyed by `(thread_id, checkpoint_ns, channel, version)`), and **checkpoint_writes** (pending writes). The blobs table avoids re-serializing unchanged large values — only write blobs for channels in `new_versions`. On read, JOIN blobs via `channel_versions` to reconstruct all channel values. PKs on all three tables handle most access patterns; add an index on the metadata `run_id` field for `adelete_for_runs`. Use subqueries/JOINs to fetch writes alongside checkpoints in a single round-trip.

### Document stores (MongoDB, Firestore, DynamoDB)

Use 2 collections: **checkpoints** (full checkpoint with all channel_values embedded, metadata as top-level fields) and **writes**. Serialize the full checkpoint including all channel values on every `aput`. Use composite `_id` or PK/SK from the key parts. Required indexes:
- `(thread_id, checkpoint_ns, checkpoint_id DESC)` — for `alist` ordering and `aget_tuple` latest-lookup
- `(thread_id)` — for `adelete_thread`
- `(metadata.run_id)` — for `adelete_for_runs`
- Use native query operators (e.g. MongoDB `$match`, DynamoDB filter expressions) for metadata filtering in `alist(filter=...)`

### Key-value stores (Redis, etcd)

Use composite keys like `cp:{thread_id}:{ns}:{id}`. Use sorted sets or equivalent for descending-order listing. **Requires manual secondary indexes** maintained on every write:
- Thread index (`thread:{thread_id}` → set of `{ns}:{checkpoint_id}`) — for `adelete_thread` and `alist`
- Run ID index (`run:{run_id}` → set of checkpoint keys) — for `adelete_for_runs`
- Write index (`writes:{thread_id}:{ns}:{checkpoint_id}` → set of `{task_id}:{idx}`) — for pending writes lookup
- Metadata filtering for `alist(filter=...)` is the hardest: either scan+deserialize, or maintain per-field indexes. For small datasets scanning is acceptable; for large ones consider a search module.
