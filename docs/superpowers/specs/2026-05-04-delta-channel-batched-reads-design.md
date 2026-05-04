# DeltaChannel batched reads: public saver API + cadence rework

**Branch:** `sr/delta-chan-combine-offfshoot`
**Date:** 2026-05-04
**Status:** Design approved; implementation pending

## Goal

Promote the private, batched delta-channel reconstruction path on this branch
to a stable, hardened public API on `BaseCheckpointSaver`, and rework
snapshot cadence so it's driven by per-channel update count rather than
superstep count.

The branch already lands the K-channel batching internally
(`_get_all_delta_channels_writes_history`). What's missing for production
use:

- The method is private and the contract is delta-internal (returns
  channel-aware `seed`/`writes` shaped by sentinel semantics).
- Stage-1 in postgres has no paging — scans every checkpoint in
  `(thread_id, ns)`, breaking on long threads. The optimized override
  needs internal pagination (not an iterator-shaped public API).
- Stage-2 in postgres uses union-of-chains, over-fetching writes when
  channels have very different snapshot cadences.
- Snapshot cadence triggers on superstep count, snapshotting channels
  even when they had no updates this step.
- The saver layer carries `DELTA_SENTINEL` knowledge at write time
  (filter in `put`) and read time (the private method) — a layering
  smell flagged in `notes/model_b_exploration.md`.

## Non-goals

- Compaction or row-level checkpoint delta encoding (separate work,
  scoped in `notes/delta_checkpoint_rows.md`).
- Optimizing the default `BaseCheckpointSaver.get_writes_history` impl —
  it stays simple/correct; in-process savers can override.
- Configurable `page_size` — left as an internal constant for now,
  exposed as a kwarg in a follow-up.
- Step fallback for snapshot cadence (force snapshot after M supersteps
  without one) — deferred entirely; not implemented in this PR.
- Custom sqlite override — sqlite uses the default impl and serves as
  validation that the default works end-to-end.

## Public API

### `BaseCheckpointSaver.get_writes_history`

```python
class WritesHistory(TypedDict):
    writes: list[PendingWrite]            # always present, possibly empty; oldest→newest
    seed: NotRequired[Any]                # absent unless walk found a seed for this channel

def get_writes_history(
    self,
    config: RunnableConfig,
    channels: Sequence[str],
) -> Mapping[str, WritesHistory]: ...

async def aget_writes_history(
    self,
    config: RunnableConfig,
    channels: Sequence[str],
) -> Mapping[str, WritesHistory]: ...
```

**Why TypedDict with `NotRequired[seed]` rather than a sentinel:** the
checkpoint package has no public absence sentinel today (`MISSING` lives in
`langgraph._internal._typing`, which checkpoint can't import). The existing
checkpoint convention is "dict-key absence = absent" — the same pattern
`CheckpointMetadata: TypedDict(total=False)` already uses for its optional
fields. Pregel translates `"seed" not in hist` to `MISSING` at consume
time, where `MISSING` is already in scope.

**Contract:**

- Walks the parent chain (NOT `list()` flat scan) starting from the
  target's `parent_config`. Forks contribute only on-path ancestors.
- For each requested channel, terminates at the nearest ancestor with a
  value present in `channel_values[ch]`. That value goes in `seed`. If
  the walk reaches root with no seed found, `seed` is omitted — the
  consumer treats absence as "start empty."
- `writes` is the full per-channel chain accumulated across the walk,
  oldest to newest. Already filtered to this channel; the consumer does
  no per-channel filtering.
- The method returns once the walk has either seeded every channel or
  exhausted the chain. Saver implementations may page internally and
  stop early as soon as all channels are seeded — pagination is an
  internal optimization, not part of the public contract.
- Empty `channels` → empty mapping.
- Saver layer has no dependency on `DELTA_SENTINEL` or
  `_DeltaSnapshot`. The terminator rule is purely "is there a value at
  `channel_values[ch]` here?"

**Default impl** (in `BaseCheckpointSaver`):

- Calls `get_tuple()` + follows `parent_config` in a loop.
- Walks one ancestor at a time, accumulates per-channel writes, marks
  channels as seeded when their `channel_values[ch]` entry is present,
  stops when all channels are seeded or the chain ends.
- Slow on long chains — that's the explicit tradeoff for keeping the
  default simple. Savers that care override.

**Postgres override:**

- Two-stage query, **paged internally**:
  - **Stage 1 (paged, fixed size = 1024 internal const):** dynamic
    SELECT over `checkpoints` with K parallel JSONB key lookups
    (`ver_i`/`hs_i` per channel), `ORDER BY checkpoint_id DESC LIMIT 1024`,
    cursor on subsequent pages (`AND checkpoint_id < ?`). Walks
    metadata in Python; records per-channel seeds when `hs_i` is true.
    Stops paging as soon as every channel has a seed.
  - **Stage 2 (per-channel UNION ALL):** one `WHERE channel='X' AND
    checkpoint_id = ANY(chain_X)` branch per channel, plus one seed-blob
    branch per channel that has a seed. Single round-trip; no over-fetch
    when channels have different chain depths.
- Returns the assembled `Mapping[str, WritesHistory]`. Pagination is
  an internal optimization; callers see one return value.

**InMemorySaver override:**

- Direct dict access into `self.storage` / `self.blobs` / `self.writes`.
- Single-pass walk; no paging needed in-process.
- ~50 lines, equivalent to today's `_get_all_delta_channels_writes_history`
  override fitted to the new shape.

**SqliteSaver:** uses the default impl unchanged. Smoke-tested for
correctness; not optimized in this PR.

## Sentinel removal

`DELTA_SENTINEL` has not shipped in any stable release (DeltaChannel
exists only on this branch + the `1.2.0a*` series), so we remove it
entirely.

**Write path:**

- `DeltaChannel.checkpoint()` returns `MISSING` (no snapshot path).
- `pregel.create_checkpoint` writes `_DeltaSnapshot(ch.get())` directly
  into `channel_values[ch]` for snapshot steps.
- Delta channels with no snapshot don't appear in `channel_values` at
  all; their `channel_versions` entry still bumps so they're tracked.
- Saver `put` no longer filters sentinels — `channel_values` is stored
  verbatim.

**Read path:**

- `_needs_replay` becomes `stored is MISSING`.
- `DeltaChannel.from_checkpoint` accepts three input shapes:
  - `MISSING` → start empty
  - `_DeltaSnapshot(value)` → use snapshot value
  - any other plain value → use directly (pre-migration legacy)

**Removed code:**

- `_DeltaSentinel` class
- `DELTA_SENTINEL` singleton and re-exports
- `_DeltaSentinel` serde extension code in `jsonplus.py`

## Snapshot cadence

Triggered by **per-channel update count** rather than superstep count.

```python
def should_snapshot(ch_name, ch):
    if force_delta_snapshot:                                       # durability="exit"
        return True
    last_v = prior_metadata.get("delta_snapshot_versions", {}).get(ch_name, 0)
    return current_version[ch_name] - last_v >= ch.snapshot_frequency
```

**`DeltaChannel.snapshot_frequency`:** required positive int, default
`1000`. The previous `None` opt-out is removed; users who want
no-snapshot pass a large int.

**Per-channel state lives in `CheckpointMetadata`:**

```python
class CheckpointMetadata(TypedDict, total=False):
    delta_snapshot_versions: dict[str, int]   # ch → version at last snapshot
```

`total=False`; absence treated as `{}`. Only present on threads using
delta channels. New keys appear when first snapshot fires.

**Bookkeeping in `_put_checkpoint` / `create_checkpoint`:**

1. Read prior `delta_snapshot_versions` from `self.checkpoint_metadata`
   (loaded at resume from `aget_tuple` or carried across steps).
2. For each delta channel, decide via `should_snapshot()`. Snapshotted
   channels get `_DeltaSnapshot(ch.get())` written into `channel_values`
   with the eager-version-bump preserved.
3. Update `delta_snapshot_versions[ch] = current_version[ch]` for each
   channel that snapshotted this step. Channels that didn't snapshot
   keep prior entries.
4. Persist new metadata with the checkpoint.

**Removed code:**

- `DeltaChannel.is_snapshot_step(step)` — no callers after this change.

**Step fallback (deferred):** the originally-discussed "force snapshot
after M supersteps without one" is left as a follow-up. Walk depth for
slow-changing channels in long-running threads can grow; if that bites
in practice we add a second metadata key
(`delta_snapshot_steps: dict[str, int]`) and a second branch in
`should_snapshot`. Purely additive — no migration needed when added.

## Postgres optimization details

### Stage-1 paging

Current: scans all checkpoints in `(thread_id, ns)`. Pathological at
high thread depths.

New: paged with cursor.

```sql
SELECT checkpoint_id, parent_checkpoint_id,
       checkpoint -> 'channel_versions' ->> ? AS ver_0,
       (checkpoint -> 'channel_values' -> ?) IS NOT NULL AS hs_0,
       ...
FROM checkpoints
WHERE thread_id = ? AND checkpoint_ns = ?
  AND (checkpoint_id < ? OR ? IS NULL)
ORDER BY checkpoint_id DESC
LIMIT 1024
```

Cursor (`checkpoint_id < ?`) is the oldest cid from the previous page.
First page passes NULL. Page size is a private constant (`1024`).

### Stage-2 per-channel UNION ALL

Current: single SQL with `channel = ANY(channels) AND checkpoint_id =
ANY(union_chain_cids)`. Over-fetches when chain depths differ across
channels.

New:

```sql
SELECT 'w', cid, channel, type, blob, task_id, idx, NULL::text AS version
FROM checkpoint_writes
WHERE thread_id=? AND checkpoint_ns=? AND channel='A'
  AND checkpoint_id = ANY(chain_A)
UNION ALL
SELECT 'w', cid, channel, type, blob, task_id, idx, NULL::text
FROM checkpoint_writes
WHERE thread_id=? AND checkpoint_ns=? AND channel='B'
  AND checkpoint_id = ANY(chain_B)
UNION ALL
SELECT 'b', NULL, channel, type, blob, NULL, NULL, version
FROM checkpoint_blobs
WHERE thread_id=? AND checkpoint_ns=? AND channel='A' AND version=?
UNION ALL ...
```

For K channels, 2K branches (writes + seed blob per channel). Same
single round-trip per page; writes fetched = `Σ chain_lengths` instead
of `K × max(chain_lengths)`.

### Internal pagination loop

`PostgresSaver.get_writes_history` walks pages internally:

1. Issue stage-1 with the current cursor (initially NULL) and `LIMIT 1024`.
2. Update per-channel seed-found state from the page metadata.
3. If every channel has a seed (or the page returned `< 1024` rows, i.e.
   chain exhausted), break and proceed to stage-2.
4. Otherwise advance cursor to the oldest cid in this page and loop.

Stage-2 then runs once over the assembled per-channel chains and seed
versions. Caller gets a single `Mapping[str, WritesHistory]`; the
multi-page round trips are an implementation detail.

## Migration

| Thread origin | Behavior with new code |
|---|---|
| Pre-DeltaChannel (plain values in `channel_values`) | Unchanged — walk terminates at plain value, `from_checkpoint` accepts it as seed |
| Alpha-only DeltaChannel (`1.2.0a*`) with serialized DELTA_SENTINEL blobs | **Hard break.** Document in alpha changelog. Alpha users restart threads. |

The alpha-only break is the right tradeoff: the alternative is a
read-side shim that maps deserialized `DELTA_SENTINEL` → `MISSING` for
one more alpha, which forces us to keep `_DeltaSentinel` alive purely
for compat with an unstable release. Not worth it.

## Test coverage

| Area | Where it lives | What it covers |
|---|---|---|
| Default impl correctness | `libs/checkpoint/tests/test_memory.py` | Walk semantics, multi-channel termination, empty `channels`, parent-chain (forks) |
| Conformance | `libs/checkpoint-conformance/` | A `validate_get_writes_history(saver)` helper any third-party saver can run |
| Postgres paging | `libs/checkpoint-postgres/tests/` | Seed in page 1 / page 2 / last page; thread shorter than page size; iterator early-stop |
| Per-channel chain bounds | `libs/checkpoint-postgres/tests/` | Two channels with different snapshot cadences; verify per-channel stage-2 returns expected counts |
| Snapshot cadence | `libs/langgraph/tests/test_delta_channel_*` | Update-count trigger fires correctly; force-snapshot for `durability="exit"`; metadata round-trips through resume |
| Migration | `test_delta_channel_migration.py` | Pre-delta thread continues working through new code path |
| Sqlite smoke test | `libs/checkpoint-sqlite/tests/` | One end-to-end test with a delta channel — validates the default impl works on a real saver |

## File-by-file changes

```
libs/checkpoint/langgraph/checkpoint/serde/types.py
  - delete _DeltaSentinel, DELTA_SENTINEL
  - keep _DeltaSnapshot

libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py
  - remove _DeltaSentinel encode/decode
  - keep _DeltaSnapshot encode/decode

libs/checkpoint/langgraph/checkpoint/base/__init__.py
  + class WritesHistory(TypedDict): writes, seed (NotRequired)
  + def get_writes_history(...) -> Mapping[str, WritesHistory]
  + async equivalent
  - _get_all_delta_channels_writes_history + async (delete)
  - _get_tuple_raw + async (delete; only existed for the deleted method)
  - _ChannelWritesHistory (delete)
  - DELTA_SENTINEL re-export (delete)

libs/checkpoint/langgraph/checkpoint/memory/__init__.py
  - drop sentinel filter in put logic
  + get_writes_history override (direct dict access; ~50 lines)
  - drop old _get_all_delta_channels_writes_history override

libs/checkpoint-postgres/langgraph/checkpoint/postgres/base.py
  - update SQL helpers: paged stage-1 with cursor; per-channel UNION-ALL stage-2
  - refactor _build_delta_channels_writes_history into per-page helpers
    consumed by get_writes_history generator

libs/checkpoint-postgres/langgraph/checkpoint/postgres/__init__.py
  - rewrite _get_all_delta_channels_writes_history → get_writes_history with internal pagination loop
  - keep dynamic stage-1 column construction (unchanged win)

libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py
  - async equivalent

libs/langgraph/langgraph/channels/delta.py
  - snapshot_frequency: int (no None), default 1000
  - checkpoint() returns MISSING
  - delete is_snapshot_step
  - from_checkpoint: three branches (MISSING / _DeltaSnapshot / plain)

libs/langgraph/langgraph/pregel/_checkpoint.py
  - create_checkpoint: read prior delta_snapshot_versions, apply update-count trigger,
    write _DeltaSnapshot directly, update metadata bookkeeping
  - _needs_replay: just `stored is MISSING`
  - channels_from_checkpoint: consume single Mapping return, hand each
    history to the corresponding DeltaChannel; "seed" key absence → MISSING

libs/langgraph/langgraph/pregel/_loop.py
  - _put_checkpoint: thread delta_snapshot_versions through metadata read/write

libs/langgraph/tests/test_delta_channel_migration.py
  - update to get_writes_history API
  - add resume-with-metadata test

libs/langgraph/tests/test_pregel.py (or appropriate)
  - add update-count cadence test
  - add durability="exit" force-snapshot test

libs/checkpoint-conformance/...
  + new conformance helper for get_writes_history

libs/checkpoint-sqlite/tests/...
  + new smoke test exercising default get_writes_history impl
```

## Implementation order (each step keeps tests green)

1. **Sentinel removal (write-path):** `DeltaChannel.checkpoint()` returns
   `MISSING`; drop saver filters; delete `_DeltaSentinel`. Small,
   isolated.
2. **Public `get_writes_history` API + default impl + InMemorySaver
   override:** introduce `WritesHistory`, default impl, in-memory
   override. Switch pregel to consume the new API. Delete the private
   method.
3. **Snapshot cadence rewrite:** add `delta_snapshot_versions` metadata,
   replace `is_snapshot_step` with the version-delta check, default
   `snapshot_frequency=1000`.
4. **Postgres optimization:** paged stage-1 with cursor; per-channel
   UNION-ALL stage-2; generator-shaped override.
5. **Conformance + sqlite smoke test:** validate the default impl works
   end-to-end on a real second saver.

Whole thing can ship as one PR or split across these five steps.

## Open questions

None at design lock-in. Knobs we explicitly defer to follow-ups:

- Configurable `page_size` (currently an internal constant).
- Step fallback for snapshot cadence (force snapshot after M supersteps
  without one) — purely additive when added.
- `Checkpoint` schema bump to v5 to host snapshot bookkeeping
  natively (currently lives in `CheckpointMetadata` per S1 vs S2).
- Custom sqlite override (use default for now).
