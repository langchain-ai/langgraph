# DeltaChannel: sentinel-based checkpoint blobs + write-replay reconstruction

## TL;DR

Introduces `DeltaChannel`, a new fold-reducer channel that stores only a
zero-byte sentinel in checkpoint blobs instead of the full accumulated value.
On load, the runtime replays the channel's ancestor writes through the reducer
to reconstruct state. For long-running threads with large accumulating state
(e.g. message histories), this trades read-time work for dramatically smaller
checkpoint blobs and avoids redundant duplication of the full value at every
step.

An optional `snapshot_frequency=N` parameter bounds replay depth by writing a
full `_DeltaSnapshot` blob every N steps — a configurable storage vs. latency
tradeoff.

---

## DeltaChannel

### New channel type (`libs/langgraph/langgraph/channels/delta.py`)

`DeltaChannel(typ, operator, *, snapshot_frequency=None)` is a fold-reducer
channel (same semantics as `BinaryOperatorAggregate`) with a different
checkpoint strategy:

- **`checkpoint()`** always returns `DELTA_SENTINEL` (a zero-byte sentinel),
  never the accumulated value.
- **`from_checkpoint()`** reconstructs value from a seed blob + replayed
  writes returned by `_get_channel_writes_history`.
- **`snapshot_frequency=N`**: `create_checkpoint` writes a `_DeltaSnapshot`
  blob every N steps, bounding the ancestor walk to at most N checkpoints.
  Snapshots are written eagerly — even if the channel had no write that step,
  a version bump forces `put()` to store the blob.

The constructor signature mirrors `BinaryOperatorAggregate`: `typ` is required
as the first positional argument and is normalized to its concrete counterpart
(e.g. `Sequence → list`, `Mapping → dict`) in `__init__`. `_is_field_channel`
in `graph/state.py` was updated to reconstruct the channel with the correct
`typ` from the `Annotated` outer type, instead of patching `item.typ` after
construction.

Both `__init__` and the clone path in `copy()` / `from_checkpoint()` start
with `value = MISSING` — no inconsistency between fresh construction and
checkpoint-loaded clones.

### New sentinel and snapshot types (`libs/checkpoint/langgraph/checkpoint/serde/types.py`)

- **`_DeltaSentinel` / `DELTA_SENTINEL`**: singleton marker. Identity-compared
  (`is DELTA_SENTINEL`) throughout; `loads_typed` always returns the same
  instance.
- **`_DeltaSnapshot(NamedTuple)`**: wraps the full accumulated value at a
  snapshot step. Serialized via a dedicated msgpack ext code
  (`EXT_DELTA_SNAPSHOT = 7`), so it round-trips through the standard serde
  path without a separate type tag.

### Serializer support (`libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py`)

- `dumps_typed`: `DELTA_SENTINEL → ("delta", b"")` (zero bytes, no payload).
- `loads_typed`: `"delta"` tag → `DELTA_SENTINEL`.
- `_DeltaSnapshot` → msgpack ext code 7 (value packed as nested msgpack).
- `loads_typed` ext hook: ext 7 → `_DeltaSnapshot(unpacked_value)`.

### Ancestor-walk API on `BaseCheckpointSaver` (`libs/checkpoint/langgraph/checkpoint/base/__init__.py`)

Three new methods on `BaseCheckpointSaver` (all experimental / underscore-prefixed):

- **`_ChannelWritesHistory(NamedTuple)`** — return type carrying `seed: Any`
  (nearest non-sentinel ancestor blob, or `DELTA_SENTINEL` if none found) and
  `writes: list[PendingWrite]` (oldest→newest on-path deltas).
- **`_get_tuple_raw(config)`** / **`_aget_tuple_raw(config)`** — pure storage
  reads used by the base walk implementation. Default delegates to `get_tuple`;
  savers whose `get_tuple` performs channel hydration (which calls
  `channels_from_checkpoint` which calls `_get_channel_writes_history`)
  override this to break the re-entrancy cycle.
- **`_get_channel_writes_history(config, channel)`** /
  **`_aget_channel_writes_history(config, channel)`** — reference
  implementation walks the parent chain via `_get_tuple_raw`, collecting
  `pending_writes` for the target channel and stopping at the first
  non-sentinel blob.

### `InMemorySaver` optimized override (`libs/checkpoint/langgraph/checkpoint/memory/__init__.py`)

`InMemorySaver._get_channel_writes_history` reads directly from
`self.storage`, `self.blobs`, and `self.writes` in one pass without repeated
`get_tuple` calls. Handles two distinct blob-termination cases:

- **Pre-delta blob** (plain value from `BinaryOperatorAggregate` era): the
  blob IS the full state at that ancestor — do not replay its pending_writes
  (they are already baked in). Return immediately.
- **`_DeltaSnapshot` blob**: the snapshot IS the state, but pending_writes at
  that ancestor encode the NEXT step's transition and must be collected before
  returning the snapshot as `seed`.

Also adds **`InMemorySaver.prune(thread_ids, *, strategy)`** — delta-aware
pruning that retains the minimal ancestor chain needed to reconstruct sentinel
channels for the latest checkpoint.

### PostgresSaver / AsyncPostgresSaver optimized override (`libs/checkpoint-postgres/`)

Both sync and async postgres savers override
`_get_channel_writes_history` / `_aget_channel_writes_history` with a
**single-roundtrip UNION ALL query** (`SELECT_DELTA_COMBINED_SQL`) that
fetches rows from `checkpoints`, `checkpoint_writes`, and `checkpoint_blobs`
in one query. Rows are assembled by the shared pure helper
`_build_delta_channel_writes_history` on `BasePostgresSaver`.

- **`_DeltaCombinedRow(TypedDict, total=False)`** — typed view of the nine
  columns emitted by the UNION ALL (kind-tagged `"p"` / `"w"` / `"b"`).
- **`_load_blobs`** parameter typed from `Any` to
  `Sequence[tuple[bytes, bytes, bytes]]`.

A benchmark (`notes/delta_channel_query_bench.md`) shows the prior recursive
CTE carried a hidden O(ancestors × blobs_in_thread) join; the plain UNION ALL
is 3–100× faster in the realistic depth range.

### Pregel integration (`libs/langgraph/langgraph/pregel/`)

**`_checkpoint.py`**:
- `channels_from_checkpoint` gains `saver` and `config` keyword args; for
  each channel where `_needs_replay` is true (DeltaChannel with a sentinel
  blob), it calls `saver._get_channel_writes_history` and replays writes.
- `achannels_from_checkpoint` added — async counterpart, used by
  `AsyncPregelLoop`.
- `create_checkpoint` gains `get_next_version` kwarg; on snapshot steps it
  writes `_DeltaSnapshot` blobs and bumps channel versions eagerly.
- `_needs_replay(spec, stored)` helper — True iff spec is a `DeltaChannel`
  and stored blob is `MISSING` or `DELTA_SENTINEL`.

**`_loop.py`**:
- `SyncPregelLoop.__enter__` passes `saver` + `config` to
  `channels_from_checkpoint`.
- `AsyncPregelLoop.__aenter__` calls `achannels_from_checkpoint`.
- **Write-ordering safety**: `_pending_write_futs` tracks in-flight
  `put_writes` futures. Before committing a checkpoint that contains any
  `DELTA_SENTINEL` blob, the loop flushes all pending write futures
  synchronously. This ensures checkpoint_writes are durable before the
  sentinel blob is stored — a sentinel backed by missing writes would cause
  silent data loss on replay.

### `binop.py` refactoring

- `_strip_extras`: fixed ordering — `Required`/`NotRequired` check must come
  before the generic `__origin__` recurse to avoid swallowing the inner arg.
- `_operators_equal` extracted as a shared helper (used by both
  `BinaryOperatorAggregate.__eq__` and `DeltaChannel.__eq__`): lambdas are
  treated as always-equal since they all share `__name__ == "<lambda>"`.
- `_get_overwrite`: `set(value.keys()) == {OVERWRITE}` → `len(value) == 1 and
  OVERWRITE in value` (avoids throwaway set allocation on every call).

---

## Follow-ups

- **Batch reconstruction across multiple DeltaChannels**: each channel
  currently issues its own `_get_channel_writes_history` call. A single
  ancestor walk collecting writes for all sentinel channels at once would
  reduce roundtrips proportionally to the number of DeltaChannel fields in a
  state schema.
