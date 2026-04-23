# Delta-channel reconstruction: query strategy benchmark

**Branch:** `delta-channel-writes-based`
**Question (Nuno):** Is the recursive CTE the right query shape for reconstructing a delta channel inside `get_tuple`, or would a plain `SELECT WHERE` be cheaper even though it returns more rows?
**Answer:** Plain `SELECT WHERE` wins at every realistic depth. The recursion isn't the problem — the JSON-expression join inside the CTE is.

## Setup

- Postgres 16 on `localhost:5441` (the `compose-postgres.yml` instance, run directly without docker for this round)
- Single delta channel `messages`, one write per checkpoint, `DELTA_SENTINEL` blob per checkpoint
- Linear chain (`branch=1`) and 5-way branching at every step (`branch=5`) — branching is the case where plain over-fetches sibling rows
- Median of 20 timed runs after 3 warmups, fresh psycopg cursor per strategy
- Bench script: `bench_get_tuple_strategies.py` at repo root

Three strategies compared:

| name | roundtrips | shape |
|------|-----------|-------|
| `cte` | 1 | Current prod: recursive CTE walks ancestors, LEFT JOINs writes + blobs |
| `plain` | 3 | Nuno's suggestion: thread-wide `SELECT WHERE` per table, Python walks parent chain and filters |
| `cte+narrow` | 2 | CTE returns ancestor IDs only, then one `UNION ALL` of writes + blobs filtered by `ANY(ids)` |

## Results (ms per get_tuple, median of 20)

```
 depth  branch       cte       plain    cte+narrow   rows_cte   rows_plain   plain/cte
    10       1     0.14ms      0.26ms       0.24ms          9           30      1.89x
    10       5     0.21ms      0.23ms       0.17ms          9          110      1.11x
    50       1     0.89ms      0.27ms       0.33ms         49          150      0.31x
    50       5     2.35ms      0.66ms       0.51ms         49          550      0.28x
   200       1    11.61ms      0.78ms       1.30ms        199          600      0.07x
   200       5    34.79ms      2.33ms       3.07ms        199         2200      0.07x
  1000       1   274.60ms      2.59ms      13.29ms        999         3000      0.01x
  1000       5   856.01ms     10.14ms      15.31ms        999        11000      0.01x
```

Lower is better. `plain/cte < 1` means plain is faster.

### Headline numbers

- depth 50: plain is **3x** faster
- depth 200: plain is **15x** faster
- depth 1000: plain is **~100x** faster
- Branching makes plain over-fetch (3000 rows → 11000 rows at d=1000), but it remains ~85x faster than the CTE

## Why the CTE collapses

`EXPLAIN (ANALYZE, BUFFERS)` of the CTE at depth 1000 (linear). Excerpt with the load-bearing nodes:

```
Sort  ... actual time=137.798..137.827 rows=999
  CTE ancestors
    ->  Recursive Union  ... actual time=0.005..2.443 rows=999
                                                 ^^^^^^
                                                 recursion is 2.4 ms — fine
  ->  Nested Loop Left Join  ... actual time=2.676..137.529 rows=999
        Join Filter: (cw.checkpoint_id = a.cid)
        Rows Removed by Join Filter: 998001
                                     ^^^^^^^
                                     999 ancestors x ~1000 writes
        ->  Nested Loop Left Join  ... actual time=2.669..85.061 rows=999
              Join Filter: (bl.version = ((c.checkpoint -> 'channel_versions'::text) ->> bl.channel))
              Rows Removed by Join Filter: 998001
                                           ^^^^^^^
                                           same quadratic blow-up on the blob join
```

Two pathological things are happening:

1. **The blob join filter is on a JSON expression**: `bl.version = (c.checkpoint -> 'channel_versions' ->> bl.channel)`. The planner cannot push this into an index lookup, so it materializes `checkpoint_blobs` for the thread and does a nested-loop comparison against every ancestor — a Cartesian product that grows as `O(ancestors × blobs_in_thread)`.
2. **The writes join is similar**: writes for the thread are materialized once, then nested-loop joined against ancestors with a `Join Filter` rather than a hash/merge join over the indexed `checkpoint_id`.

At depth 1000 that's **~2 million rows evaluated, 99.9% of them discarded**. The recursion itself is a rounding error.

For comparison, the plain Q1 (`SELECT … FROM checkpoints WHERE thread_id=? AND checkpoint_ns=?`) at depth 1000:

```
Seq Scan on checkpoints  ... actual time=0.012..0.121 rows=1000
Execution Time: 0.140 ms
```

A simple seq scan over 57 buffers. Q2 and Q3 follow the same shape and complete in well under 1 ms each.

## Crossover and remote-DB reasoning

- Pure local Postgres: plain wins from depth ~30 onward; CTE wins by fractions of a ms below that
- Remote Postgres at ~5 ms RTT adds ~10 ms to plain (3 roundtrips vs 1). Crossover shifts to ~depth 30. Above that, the CTE's quadratic SQL cost still dominates the RTT savings.

There is no realistic conversation depth where the CTE wins on a remote DB. At depth 200+ (anything resembling a real multi-turn agent run) plain is faster regardless of network.

## Recommendation

**Switch to plain SELECT WHERE, one delta channel at a time.**

Three indexed queries per delta channel:

```sql
-- Q1: parent chain + per-checkpoint version of this channel
SELECT checkpoint_id,
       parent_checkpoint_id,
       checkpoint -> 'channel_versions' ->> 'channel_name' AS ver
FROM checkpoints
WHERE thread_id = ? AND checkpoint_ns = ?;

-- Q2: writes for this channel, anywhere in the thread
SELECT checkpoint_id, type, blob, task_id, idx
FROM checkpoint_writes
WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ?;

-- Q3: blobs for this channel, anywhere in the thread
SELECT version, type, blob
FROM checkpoint_blobs
WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ?;
```

Python then:
- Builds `parent_of: dict[cid, parent_cid]` from Q1
- Walks from target's parent newest → oldest
- Filters Q2 rows by `ancestor_set`, processes oldest → newest, applies overwrite-terminator
- Picks seed blob via the per-ancestor `ver` map, terminates at first non-sentinel blob

All O(n) on n = thread checkpoints, with tight constants (dict lookups). No recursion, no JSON-expression joins, no quadratic plans.

If the 3-roundtrip cost ever shows up on remote-DB benchmarks, fold Q2 + Q3 into one `UNION ALL` to get back to 2 roundtrips. Bench says it isn't worth the SQL complexity right now.

## Bonus: code simplification from single-channel scope

Multi-channel reconstruction in the current `_reconstruct_delta_channels_cur` carries:

- `rows_by_cid` nested dicts, keyed by cid then channel
- `seen_blob: set[(cid, channel)]` and `seen_write: set[(cid, channel, task_id, idx)]` dedup
- `collected: dict[channel, list]`, `done: set[channel]`, `seeds: dict[channel, value]`
- Inner `for ch in channels_list` loops and an early-exit `if len(done) == len(channels_list)`

Single-channel collapses these to a single list, a single bool, and one `Optional[Any]`. Roughly half the Python in that function, plus an obvious shape for splitting pure post-processing into `base.py` so sync and async stop duplicating it.

If multi-channel coalescing turns out to matter later, it can come back as a SQL-level optimization without re-introducing the bookkeeping in Python.
