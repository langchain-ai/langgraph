# AggregateChannel — unified fold-reducer channel with configurable snapshot cadence

**Status:** MVP scope approved. Implementation starting 2026-04-24.
**Supersedes:** `langgraph.channels._delta.DeltaChannel` (experimental, private).
**Branch:** `sr/even-better-writes-idea` (forked from `delta-channel-writes-based`).

## Problem

The experimental `DeltaChannel` (introduced earlier on this branch) stores a
sentinel in every checkpoint and reconstructs state by walking ancestor
writes. It eliminates O(N²) blob growth for append-style reducers on long
threads, but read cost now scales O(N) with thread depth — every load
replays every write since the start of the thread.

`BinaryOperatorAggregate` is the opposite extreme: always snapshots the full
value every step. Zero replay cost at read time, but O(N²) storage on
append-heavy workloads.

These are endpoints of the same axis. A single channel class parameterised
on snapshot cadence covers both — plus every intermediate point.

The concrete pain that surfaced this design: deep-agent workloads at
200+ turns pay O(200) replay per read under `DeltaChannel`. A
`snapshot_frequency` knob bounds that to O(snapshot_frequency) regardless
of thread depth.

## MVP scope (this PR)

1. **New class `AggregateChannel`** at `libs/langgraph/langgraph/channels/aggregate.py`:

   ```python
   class AggregateChannel(Generic[Value], BaseChannel[Any, Any, Any]):
       def __init__(
           self,
           operator: Callable[[Value, Value], Value],
           *,
           snapshot_frequency: int | float = 1,
           typ: type[Value] | None = None,
       ): ...
   ```

   - `snapshot_frequency=1` (default): full snapshot every step. Equivalent
     to today's `BinaryOperatorAggregate`.
   - `snapshot_frequency=N` (integer > 1): full snapshot every Nth step;
     sentinel on other steps.
   - `snapshot_frequency=math.inf`: never snapshot. Equivalent to today's
     `DeltaChannel`.
   - `typ` inferred from `Annotated[...]` via `_strip_extras` when used in
     a `TypedDict` state schema; kwarg is the escape hatch for imperative
     constructions.

2. **Rewire `BinaryOperatorAggregate` as a subclass** of `AggregateChannel`
   with `snapshot_frequency=1` hard-coded. Preserves
   `isinstance(x, BinaryOperatorAggregate)` for any existing user code and
   `_is_field_binop` detection in `graph/state.py`.

   ```python
   class BinaryOperatorAggregate(AggregateChannel):
       def __init__(self, typ, operator):
           super().__init__(operator, typ=typ, snapshot_frequency=1)
   ```

3. **Delete `langgraph/channels/_delta.py`** (`DeltaChannel`). It was
   experimental, private, underscored, and not re-exported — clean
   removal. Users migrate to `AggregateChannel(op, snapshot_frequency=math.inf)`.

4. **Step-aware `create_checkpoint`** at `libs/langgraph/langgraph/pregel/_checkpoint.py`:

   `AggregateChannel` exposes a helper method:

   ```python
   def is_snapshot_step(self, step: int) -> bool:
       if self.snapshot_frequency == 1:
           return True
       if self.snapshot_frequency == math.inf:
           return False
       return step % self.snapshot_frequency == 0
   ```

   `create_checkpoint` calls this per channel. When it returns `False`,
   the row stores `DELTA_SENTINEL` for that channel; when `True`, it
   stores `ch.checkpoint()` as today. `step` is already an argument to
   `create_checkpoint` — no new threading. The explicit branches on
   `1` and `math.inf` avoid relying on `step % math.inf` NaN arithmetic
   and make the common cases (always snapshot / never snapshot) free of
   a modulo.

5. **Generalise `channels_from_checkpoint`**. The existing DeltaChannel-specific
   branch keys off `isinstance(spec, DeltaChannel)`; change to
   `isinstance(spec, AggregateChannel) and spec.snapshot_frequency != 1`.
   The existing "pre-delta seed terminator" walk already treats any
   non-sentinel ancestor blob as the base value and stops — no change
   needed to saver-side logic. A `snapshot_frequency=10` blob at step 50
   serves as a natural terminator for a walk starting at step 57.

6. **Saver API stays as-is.** `_get_channel_writes_history` (private,
   underscored) continues to be the reconstruction hook. The broader
   refactor (`walk_writes` + `put_channel_snapshot`, batched multi-channel
   walks) is deferred to a follow-up PR that can benchmark its own win
   independently.

## Explicitly deferred (documented, not implemented here)

Each of the below lands as its own PR on top of this MVP.

- **`coalesce=` kwarg on `AggregateChannel`.** Batch-shape reducer for users
  who need to see all of a step's writes at once (non-binary-foldable
  reducers: median, priority-pick, dedup-across-writes). Additive to the
  existing `operator` kwarg; exactly one of the two must be provided.
- **Saver API boundary refactor.** Rename `_get_channel_writes_history` →
  `walk_writes(config, *, channels=None)`. Move the `DELTA_SENTINEL`
  terminator check from saver-side to pregel-side. Saver becomes a pure
  storage primitive (aligns with every event-sourced system surveyed:
  Akka Persistence, EventStoreDB, Kafka Streams, Postgres logical
  replication, Firestore). Research memo captured in brainstorming session.
- **Batched multi-channel walks.** One ancestor-walk query per read
  regardless of how many `AggregateChannel` channels need hydration.
  Today each channel triggers its own walk; deep-agent graphs with 7 state
  channels pay 7× the latency.
- **`put_channel_snapshot` saver hook.** Opportunistic/manual compaction of
  a pure-delta (`snapshot_frequency=math.inf`) thread by retroactively
  promoting a sentinel row to a full blob. Separate from the write hot path.
- **ShallowPostgresSaver compat.** `snapshot_frequency > 1` is fundamentally
  incompatible with shallow savers (no parent chain → nowhere to walk).
  Detect at attach time and error loudly. The existing `DeltaChannel` has
  the same silent incompatibility today; make it explicit in the same
  pass.
- **Option A — channel_versions / versions_seen delta-encoding.** Documented
  in `notes/delta_checkpoint_rows.md`. 60% win on the checkpoint row table,
  reuses the same parent-walk machinery. Requires `Checkpoint.v` bump
  (4 → 5), so wants to land on top of the saver API refactor, not stacked
  with this MVP.

## Key design decisions and why

- **`operator`-only MVP, `coalesce` deferred.** The deepagents workload
  uses `add_messages`-shape reducers (binary-foldable). Shipping
  `coalesce=` now expands the API surface before we've validated that
  the snapshot-cadence half works on a real workload. `coalesce=` is
  additive and can land without breaking anyone.
- **Subclass, not alias.** `BinaryOperatorAggregate` is imported and
  instantiated directly in at least `libs/langgraph/langgraph/graph/state.py:1711`
  (`_is_field_binop`). A factory function breaks `isinstance`; a subclass
  doesn't.
- **`snapshot_frequency`, not `snapshot_every`.** Chosen by user preference;
  semantically identical (integer period, default 1).
- **No saver API change in MVP.** The saver's existing
  `_get_channel_writes_history` contract is sufficient for the cadence
  knob to work. Deferring the saver refactor lets this PR land
  independently and the refactor benchmark against a stable baseline.
- **No benchmark harness in this PR.** Benchmarking happens externally
  against deepagents.
- **DELTA_SENTINEL keeps its name.** Even though the class renames to
  `AggregateChannel`, the sentinel itself is still "this row represents a
  delta from ancestors" — the name is accurate. Rename could happen in a
  later cleanup if desired but isn't in scope.

## Migration semantics

- **Existing threads with `BinaryOperatorAggregate`** continue to work
  unchanged — they're now instances of `AggregateChannel` with
  `snapshot_frequency=1`, and the runtime code paths are identical.
- **Switching `snapshot_frequency` mid-thread** (e.g. user bumps
  `snapshot_frequency=1` → `10` on an existing thread): pre-change
  checkpoints have full blobs; they act as natural walk terminators for
  post-change reads. No explicit migration step. No data loss.
- **Reverse migration** (`snapshot_frequency=10` → `1`): next write produces
  a full blob. Reads at ancestors still find the right terminator. Safe.
- **Switching `snapshot_frequency=math.inf` → any finite value:**
  next snapshot-step writes a full blob that closes all prior sentinel-only
  ancestry. Walks from that point forward stop at the new base rather
  than walking to the root.
- **External users who imported `langgraph.channels._delta.DeltaChannel`**:
  `ImportError` at upgrade time. Underscored + experimental + docstring
  says "subject to change or removal without notice" — documented
  breakage; migration is a one-line swap.

## File-level change list

**New:**
- `libs/langgraph/langgraph/channels/aggregate.py` — `AggregateChannel` class

**Modified:**
- `libs/langgraph/langgraph/channels/binop.py` — `BinaryOperatorAggregate`
  becomes subclass of `AggregateChannel`; reducer logic moves to base.
- `libs/langgraph/langgraph/channels/__init__.py` — export
  `AggregateChannel`.
- `libs/langgraph/langgraph/pregel/_checkpoint.py`:
  - `create_checkpoint`: step-aware sentinel vs blob decision.
  - `channels_from_checkpoint` / `achannels_from_checkpoint`: key off
    `AggregateChannel` instead of `DeltaChannel`.
  - `DeltaChannel` import removed.
- `libs/langgraph/langgraph/graph/state.py` — `_is_field_binop` continues
  to work unchanged (subclass relationship preserves detection).

**Deleted:**
- `libs/langgraph/langgraph/channels/_delta.py`

**Tests updated:**
- Any test importing `DeltaChannel` from `langgraph.channels._delta` —
  switch to `AggregateChannel(op, snapshot_frequency=math.inf)`.
- Add: parity test for `snapshot_frequency=1` vs today's `BinaryOperatorAggregate`
  on the same workload.
- Add: cadence test at `snapshot_frequency=10` on a 50-step thread —
  verify blobs land on steps 0/10/20/30/40/50, sentinels elsewhere, and
  reads at every step produce the same value as the all-snapshot baseline.
- Add: mid-thread `snapshot_frequency` change — verify no data loss.

## Out of scope

- Performance benchmarking (user runs externally on deepagents).
- Documentation/tutorial updates for the new knob (until MVP validates).
- Any saver-side changes.
- Any `Checkpoint.v` bump.
- Public API promotion — `AggregateChannel` replaces a private
  experimental class; it's immediately public by virtue of living in
  `langgraph.channels`, but the `snapshot_frequency > 1` path inherits
  DeltaChannel's "experimental, validate on real workloads first"
  caveat until benchmark confirms it.
