# DiffChannel: Incremental Checkpoint Storage for Append-Style Reducers

**Date:** 2026-04-17  
**Status:** Approved for implementation  
**Scope:** `libs/checkpoint`, `libs/langgraph`, `libs/checkpoint-postgres`

---

## Motivation

LangGraph checkpoints today store the **full accumulated value** of every channel on every step. For a `messages` channel backed by `add_messages`, this means each checkpoint blob contains the entire conversation history. Storage cost grows O(N²) in the number of turns: step 1 stores 1 message, step 100 stores 100 messages, step 1000 stores 1000 messages. For long-running agentic conversations with high-token messages this is untenable.

The fix is to store only the **delta** (new writes) per step, reconstructing the full accumulated value at load time by replaying the chain. This is an opt-in mechanism — existing graphs are unaffected.

---

## Non-Goals

- **Compaction / materialized snapshots**: deferred. Load cost stays O(N) blob fetches but those fetches are batched into a single query — acceptable for now.
- **SQLite saver support**: SQLite stores all channel values inline in one row (no per-channel blob table). Deferred to a follow-up.
- **Automatic migration** of existing `BinaryOperatorAggregate` channels: users opt in explicitly. Old checkpoints load correctly via the backwards-compatibility path in `from_checkpoint`.

---

## Architecture Overview

```
User state definition
  └── Annotated[list[AnyMessage], DiffChannel(add_messages)]

Write path (per superstep)
  DiffChannel.update()       — apply operator, accumulate writes in _pending
  DiffChannel.checkpoint()   — return DiffDelta(delta=_pending, prev_version=_base_version)
  serde.dumps_typed()        — serialize DiffDelta as ("diff", msgpack_bytes)
  saver.put()                — store blob at (thread_id, ns, "messages", version_N)
  DiffChannel.after_checkpoint(version_N) — advance _base_version, clear _pending

Read path (on graph load or time-travel)
  saver.get_tuple()          — fetch current-version blob per channel
  saver._load_blobs()        — detect "diff" type → follow chain to reconstruct DiffChainValue
  DiffChannel.from_checkpoint(DiffChainValue) — replay deltas with operator → full list
  DiffChannel.after_checkpoint(version_N)     — set _base_version for next write
```

The pregel layer (`_checkpoint.py`, `_loop.py`) is unchanged except for two small additions to call the new `after_checkpoint` hook. The saver public interface (`BaseCheckpointSaver`) gains no new methods. All chain-following logic lives inside each saver's private `_load_blobs`.

---

## New Protocol Types

**Location:** `libs/checkpoint/langgraph/checkpoint/base/__init__.py`

Two dataclasses form the contract between `DiffChannel` and savers:

```python
@dataclass
class DiffDelta:
    """Returned by DiffChannel.checkpoint(). Written to the blob store."""
    delta: list[Any]           # raw writes passed to update() this step
    prev_version: str | None   # version of the previous diff blob; None = chain root
```

```python
@dataclass
class DiffChainValue:
    """Passed to DiffChannel.from_checkpoint(). Assembled by _load_blobs()."""
    base: list[Any] | None     # starting accumulated value (None = empty start)
    deltas: list[list[Any]]    # write-sets ordered oldest → newest
```

`DiffDelta` lives in the checkpoint base package (not the channel module) so savers can import it without creating a circular dependency. `DiffChainValue` is there for the same reason.

---

## `BaseChannel.after_checkpoint()` Hook

**Location:** `libs/langgraph/langgraph/channels/base.py`

```python
def after_checkpoint(self, version: Any) -> None:
    """Called after checkpoint() (with the new version) and after from_checkpoint()
    (with the current version). No-op by default; DiffChannel overrides."""
    pass
```

This is a **non-abstract, no-op default** — fully backwards compatible. All existing channels inherit it silently. It is NOT in the abstract interface.

---

## `DiffChannel[V]`

**Location:** `libs/langgraph/langgraph/channels/diff.py` (new file)

### Internal state

| Attribute | Type | Description |
|---|---|---|
| `value` | `list[V]` | Full accumulated value (the reconstructed list) |
| `operator` | `Callable` | The binary reducer (e.g. `add_messages`) |
| `_pending` | `list[Any]` | Raw writes accumulated since last `after_checkpoint` call |
| `_base_version` | `str \| None` | Version this channel was last checkpointed at (= `prev_version` for next delta) |
| `_overwritten` | `bool` | True if an `Overwrite` was applied since last `after_checkpoint`; makes next blob a chain root |

### `update(values)`

Mirrors `BinaryOperatorAggregate.update()` with two additions:

1. For each non-Overwrite value: apply `self.operator(self.value, value)` as before; **also append the raw incoming value to `self._pending`**.
2. For an `Overwrite(v)` value: set `self.value = v`; set `self._pending = list(v)` (full value becomes the new delta); set `self._overwritten = True`.

The key: `_pending` stores the **incoming writes** (what was passed to `update()`), not the diff of `self.value`. This is important because `add_messages` handles removal and update-by-ID — replaying the writes with `operator` during reconstruction applies that logic correctly.

### `checkpoint()`

```python
def checkpoint(self) -> DiffDelta:
    return DiffDelta(
        delta=self._pending[:],
        prev_version=None if self._overwritten else self._base_version,
    )
```

- Normal step: `prev_version = self._base_version` → chain link
- After Overwrite: `prev_version = None` → chain root (reconstruction stops here and uses `delta` as the full base value)

Returns `DiffDelta`, never the raw accumulated list. The serde handles serialization.

### `from_checkpoint(checkpoint)`

```python
def from_checkpoint(self, checkpoint) -> Self:
    new = DiffChannel(self.typ, self.operator)
    new.key = self.key
    if checkpoint is MISSING:
        new.value = []
    elif isinstance(checkpoint, DiffChainValue):
        accumulated = checkpoint.base or []
        for step_writes in checkpoint.deltas:
            # Mirror update() exactly: apply each write individually so operator
            # semantics (e.g. add_messages ID-based removal) are respected.
            for write in step_writes:
                accumulated = new.operator(accumulated, write)
        new.value = accumulated
    elif isinstance(checkpoint, DiffDelta):
        # Unsupported saver: _load_blobs returned a raw DiffDelta instead of
        # assembling a DiffChainValue. Raise rather than silently losing history.
        raise ValueError(
            "DiffChannel received a raw DiffDelta from the checkpoint saver. "
            "Your saver does not support incremental channel storage. "
            "Use InMemorySaver or PostgresSaver."
        )
    else:
        # Backwards compat: plain list from old BinaryOperatorAggregate checkpoint.
        new.value = checkpoint
    new._pending = []
    new._base_version = None  # set by the subsequent after_checkpoint() call
    return new
```

The operator is available on `self` (the channel spec) so reconstruction is correct for any reducer — the saver never needs to know about `add_messages`.

`_pending` stores **individual writes** (each `value` from `update()`'s `values` sequence), so each `step_writes` list in `DiffChainValue.deltas` is replayed write-by-write — identical to the `update()` loop.

### `after_checkpoint(version)`

```python
def after_checkpoint(self, version: Any) -> None:
    if version != self._base_version:
        self._base_version = version
        self._pending = []
        self._overwritten = False
```

No-op when `version == self._base_version` (channel wasn't updated this step — blob was not written). Clears `_pending` and advances `_base_version` when the channel was actually checkpointed.

### Opt-in API

```python
from langgraph.channels.diff import DiffChannel

class State(TypedDict):
    messages: Annotated[list[AnyMessage], DiffChannel(add_messages)]
```

`StateGraph` already handles `BaseChannel` instances as annotation metadata — `DiffChannel` inherits this without any changes to `StateGraph`.

---

## Serde Extension

**Location:** `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py`

Add one branch to `dumps_typed` (before the `else` msgpack fallback), using the existing module-level `_msgpack_enc` so message ext-types (Pydantic v2, etc.) are handled correctly:

```python
elif isinstance(obj, DiffDelta):
    return "diff", _msgpack_enc({"d": obj.delta, "p": obj.prev_version})
```

Add one branch to `loads_typed` so savers can decode diff blobs without importing `ormsgpack` directly:

```python
elif type_ == "diff":
    return ormsgpack.unpackb(
        data_, ext_hook=self._unpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
    )
    # returns {"d": [writes...], "p": prev_version_str_or_none}
```

Savers call `serde.loads_typed(("diff", raw_bytes))` to decode a diff blob into `{"d": ..., "p": ...}`, then check `type_tag == "diff"` to trigger chain traversal. The serde layer is the only place that knows about `ormsgpack`.

---

## Saver Changes

### InMemorySaver

**`put()` — `libs/checkpoint/langgraph/checkpoint/memory/__init__.py`**

No change needed. The existing `self.serde.dumps_typed(values[k])` call already handles `DiffDelta` via the new serde branch above, storing it as `("diff", bytes)`.

**`_load_blobs()` — same file**

After checking `vv[0] != "empty"`, add a branch for `"diff"` before calling `serde.loads_typed`:

```python
def _load_blobs(self, thread_id, checkpoint_ns, versions):
    channel_values = {}
    diff_channels = {}  # channel_name -> current_version for diff channels

    for k, v in versions.items():
        kk = (thread_id, checkpoint_ns, k, v)
        if kk not in self.blobs:
            continue
        type_tag, blob_bytes = self.blobs[kk]
        if type_tag == "diff":
            diff_channels[k] = v  # handle below
        elif type_tag != "empty":
            channel_values[k] = self.serde.loads_typed((type_tag, blob_bytes))

    for k, current_version in diff_channels.items():
        # Follow chain: newest → oldest, then reverse
        chain_deltas = []
        base = None
        version = current_version
        while version is not None:
            kk = (thread_id, checkpoint_ns, k, version)
            if kk not in self.blobs:
                break
            type_tag, blob_bytes = self.blobs[kk]
            if type_tag == "diff":
                # Use serde so we don't need to import ormsgpack directly
                payload = self.serde.loads_typed((type_tag, blob_bytes))
                chain_deltas.append(payload["d"])
                version = payload["p"]  # prev_version; None = root
            else:
                # Old non-diff blob encountered: treat as base accumulated value
                base = self.serde.loads_typed((type_tag, blob_bytes))
                break
        chain_deltas.reverse()
        channel_values[k] = DiffChainValue(base=base, deltas=chain_deltas)

    return channel_values
```

Each blob lookup is O(1) on the dict. Total: N dict lookups for a chain of depth N. Memory usage is identical to loading a single full-list blob (same total bytes, split across N entries).

### PostgresSaver

**`_load_blobs()` — `libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py`**

The existing `SELECT_SQL` fetches one blob per channel via a JOIN. After running that query, detect any `"diff"` channels in the result and issue one additional range query:

```python
def _load_blobs(self, blob_values):
    if not blob_values:
        return {}

    result = {}
    diff_channels = {}  # channel_name -> current_version (as str)

    for k, t, v in blob_values:
        channel = k.decode()
        type_tag = t.decode()
        if type_tag == "diff":
            # Decode via serde — no direct ormsgpack import needed
            payload = self.serde.loads_typed((type_tag, v))
            diff_channels[channel] = payload  # store for chain fetch
        elif type_tag != "empty":
            result[channel] = self.serde.loads_typed((type_tag, v))

    if diff_channels:
        result.update(self._load_diff_chains(diff_channels))

    return result
```

`_load_diff_chains` issues one SQL query per diff channel (typically just `messages`):

```sql
SELECT version, type, blob
FROM checkpoint_blobs
WHERE thread_id = %s
  AND checkpoint_ns = %s
  AND channel = %s
  AND version <= %s
ORDER BY version ASC
```

In Python, iterate rows in ascending version order: if `type = "diff"`, accumulate the delta; if any other type is encountered, treat it as the base accumulated value and stop. Return `DiffChainValue(base=..., deltas=[...])`.

This results in **at most 2 queries total** for a graph with one `DiffChannel` — existing behaviour for all other channels is unchanged.

**`put()` / `_dump_blobs()`**

No change needed. `_dump_blobs` calls `self.serde.dumps_typed(v)` for each channel value in `new_versions`. When `v` is a `DiffDelta`, the serde produces `("diff", bytes)` which is stored as `type = "diff"` in `checkpoint_blobs`. The `ON CONFLICT DO NOTHING` semantics are preserved.

### SQLite

Deferred. `SqliteSaver` stores the entire checkpoint as a single serialized row — it has no per-channel blob table. Supporting `DiffChannel` on SQLite would require adding a new blobs table, which is a separate migration tracked separately.

---

## Pregel Layer Changes

### `channels_from_checkpoint` — `libs/langgraph/langgraph/pregel/_checkpoint.py`

After constructing each channel from its checkpoint value, call `after_checkpoint` so the channel records its current version:

```python
channels = {}
for k, v in channel_specs.items():
    ch = v.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
    ch.after_checkpoint(checkpoint["channel_versions"].get(k))
    channels[k] = ch
return channels, managed_specs
```

Existing channels get the no-op `after_checkpoint`. `DiffChannel` uses it to set `_base_version`.

### `PregelLoop._put_checkpoint` — `libs/langgraph/langgraph/pregel/_loop.py`

After `create_checkpoint(self.checkpoint, self.channels, self.step, ...)` returns and `do_checkpoint is True` and `self.channels is not None`, iterate channels and notify:

```python
if do_checkpoint and self.channels:
    for k, ch in self.channels.items():
        ch.after_checkpoint(self.checkpoint["channel_versions"].get(k))
```

This is called after `create_checkpoint` updates `self.checkpoint["channel_versions"]`, so `get(k)` returns the new version for updated channels and the old version for unchanged ones. `DiffChannel.after_checkpoint` only clears `_pending` when `version != _base_version`, so unchanged channels are no-ops.

---

## Backwards Compatibility

| Scenario | Behaviour |
|---|---|
| Existing graph using `add_messages` (BinaryOperatorAggregate) | Unaffected — no code changes, no data migration |
| New graph with `DiffChannel`, loading old checkpoint blobs | `from_checkpoint` receives a plain `list` → used directly as accumulated value |
| `DiffChannel` with `InMemorySaver` or `PostgresSaver` | Fully supported |
| `DiffChannel` with `SqliteSaver` | `from_checkpoint` receives a raw `DiffDelta` (SqliteSaver stores channel_values inline), raises `ValueError` with a clear message pointing to supported savers |
| Time-travel / fork to past checkpoint | Chain traversal uses the version at that checkpoint → reconstruction is correct |
| `update_state` | Treated as a normal step: writes are deltas chained to history |
| `Overwrite` value | Resets chain: next blob has `prev_version=None`; reconstruction starts fresh |

---

## Testing Strategy

1. **Unit tests for `DiffChannel`** (`libs/langgraph/tests/`):
   - `update` → `checkpoint` → `after_checkpoint` → `checkpoint` lifecycle (2 steps, verify delta isolation)
   - `from_checkpoint(DiffChainValue)` correctly replays multi-step chains using the operator
   - `from_checkpoint(plain_list)` backwards-compat path
   - `Overwrite` creates a root blob (`prev_version=None`) and reconstruction ignores prior chain
   - `after_checkpoint` no-ops when version is unchanged

2. **Integration tests with `InMemorySaver`** (`libs/langgraph/tests/`):
   - 10-step conversation: verify final loaded state equals full accumulated messages
   - Time-travel: fork to step 5, verify only messages 1–5 are present
   - Mixed graph: some channels `BinaryOperatorAggregate`, one `DiffChannel` — both reconstruct correctly

3. **Serde tests** (`libs/checkpoint/tests/`):
   - `DiffDelta` round-trips through `dumps_typed` / saver storage
   - Old `"msgpack"` blob for a channel → `DiffChannel.from_checkpoint` handles it

4. **Postgres integration tests** (`libs/checkpoint-postgres/tests/`):
   - Range query reconstructs correct full list after N steps
   - Time-travel to checkpoint M reconstructs correct list of M messages

---

## Files Changed

| File | Change |
|---|---|
| `libs/checkpoint/langgraph/checkpoint/base/__init__.py` | Add `DiffDelta`, `DiffChainValue` dataclasses |
| `libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py` | Add `"diff"` branch in `dumps_typed` |
| `libs/checkpoint/langgraph/checkpoint/memory/__init__.py` | Chain traversal in `_load_blobs` |
| `libs/langgraph/langgraph/channels/base.py` | Add no-op `after_checkpoint` method |
| `libs/langgraph/langgraph/channels/diff.py` | **New file** — `DiffChannel` implementation |
| `libs/langgraph/langgraph/channels/__init__.py` | Export `DiffChannel` |
| `libs/langgraph/langgraph/pregel/_checkpoint.py` | Call `after_checkpoint` in `channels_from_checkpoint` |
| `libs/langgraph/langgraph/pregel/_loop.py` | Call `after_checkpoint` after `create_checkpoint` |
| `libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py` | Range-query chain reconstruction in `_load_blobs` |
