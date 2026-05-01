# DeltaChannel Benchmark Suite

Two benchmark scripts measure DeltaChannel checkpoint read/write performance.

## Scripts

### `test_delta_channel_benchmark.py` — Storage and latency sweep

Compares `DeltaChannel` vs `add_messages` (BinaryOperatorAggregate) across
turn counts and `snapshot_frequency` values.

```bash
cd libs/langgraph
uv run python tests/test_delta_channel_benchmark.py
```

### `test_delta_channel_two_stage_benchmark.py` — Two-stage query optimization

Compares the default one-stage `SELECT_DELTA_COMBINED_SQL` read path against
the two-stage path that avoids fetching unused snapshot blobs.

```bash
cd libs/langgraph
LANGGRAPH_BENCH_POSTGRES_URI="postgres://postgres:postgres@localhost:5442/postgres" \
  BENCH_N_TURNS=500 \
  uv run python tests/test_delta_channel_two_stage_benchmark.py
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `LANGGRAPH_BENCH_POSTGRES_URI` | `postgres://postgres@localhost:5432/postgres?sslmode=disable` | Postgres connection string |
| `BENCH_N_TURNS` | `100` | Number of conversation turns to write |
| `LG_DELTA_TWO_STAGE_QUERY` | `""` (off) | Set to `1` to enable the two-stage read path |

Start Postgres for benchmarks:

```bash
cd libs/langgraph
docker compose -f tests/compose-postgres.yml up -V --force-recreate --wait
```

## Two-stage query optimization

### Problem

The default `_get_channel_writes_history` fetches **all** checkpoint metadata,
writes, and blobs for a `(thread_id, channel)` in one UNION ALL query.  With
`snapshot_frequency=N`, this pulls back K full-size snapshot blobs even though
only the nearest one is needed as the reconstruction seed.  At scale this means
hundreds of MB transferred over the wire for a single `get_state` call.

### Solution

Set `LG_DELTA_TWO_STAGE_QUERY=1` to enable a two-stage read path:

1. **Stage 1** scans `checkpoints` metadata (no blob bytes) to walk the parent
   chain and locate the nearest snapshot marker in the JSONB `channel_values`.
2. **Stage 2** fetches only the chain-limited `checkpoint_writes` rows and the
   single seed snapshot blob.

Write path changes:
- `DELTA_SENTINEL` values are no longer written to `checkpoint_blobs` (row
  absence encodes the sentinel state).
- `_DeltaSnapshot` values leave a `true` marker in the checkpoint JSONB
  `channel_values` while the real blob goes to `checkpoint_blobs` as before.

### Benchmark results

Postgres 16 (Docker, local), ~10 KB per message, `snapshot_frequency=10`.

#### 100 turns (200 messages)

```
  Storage:
    checkpoint rows:  400
    blob rows:        39   (38.1 MB)
    write rows:       200  (1.9 MB)

  Read latency (avg of 5 get_state calls):
  mode                                 latency    msgs
  ----------------------------------------------------
  1-stage (default)                    36.4ms     200
  2-stage (optimized)                   4.9ms     200

  Speedup: 7.46x
```

#### 500 turns (1000 messages)

```
  Storage:
    checkpoint rows:  2000
    blob rows:        199  (964.6 MB)
    write rows:       1000 (9.7 MB)

  Read latency (avg of 5 get_state calls):
  mode                                 latency    msgs
  ----------------------------------------------------
  1-stage (default)                   682.1ms    1000
  2-stage (optimized)                  24.3ms    1000

  Speedup: 28.04x
```

The speedup grows with thread length because the one-stage path transfers all
snapshot blobs (cumulative size ~ O(N^2 / freq)) while the two-stage path
transfers only the single nearest snapshot (O(N)).

### When to enable

Enable `LG_DELTA_TWO_STAGE_QUERY=1` when:
- Using `DeltaChannel` with `snapshot_frequency` set
- Threads are long (100+ steps)
- Channel values are large (1 KB+ per message)
- Database is remote (network transfer is a bottleneck)

The flag is safe to enable for all workloads — it adds one extra DB roundtrip
but dramatically reduces data transfer for long threads with snapshots.
