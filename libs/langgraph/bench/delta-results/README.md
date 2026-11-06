# DeltaChannel snapshot-frequency sweep

## Status and recommendation

The sequential Postgres grid was stopped in favor of the distributed coordinator.
`postgres.csv` contains four completed prefixes from frequency 25 / mock700, not a
complete grid. No default change is justified by these partial measurements alone.
Until the long-thread grid finishes, retain existing defaults rather than selecting a
frequency from short threads. Frequency 300 remains a candidate for both LangGraph
and deepagents based on the supplied fleet evidence, not a conclusion of this run.
The capped grid does not benchmark LangGraph's current 1000 setting.

`pilot.csv` contains the initial 25/300/500-frequency, 50/200-update exploratory run.
Its first read includes initialization overhead; do not use that maximum as a steady
state result. `scaling-pilot.csv` contains 1000-update runs at frequency 300 with
warmup. These are explicitly partial grids. `postgres.csv` contains completed points
from the requested full grid as it progresses; missing rows are not measurements.

## Reproduce

From `libs/langgraph`, sync declared dependencies with `uv sync --frozen --group dev`.
Use a **dedicated disposable Postgres database**, not a production database. For
example, start `postgres:14.7` with a loopback-only published port and configure
`LANGGRAPH_BENCH_POSTGRES_URI` in the environment. The benchmark creates checkpointer
tables and deletes only its UUID-scoped threads when each scenario finishes.

```bash
uv run python tests/test_delta_channel_benchmark.py --snapshot-sweep \
  --backend postgres --output bench/delta-results/postgres.csv
uv run python tests/test_delta_channel_benchmark.py --snapshot-sweep \
  --backend memory --output bench/delta-results/memory.csv
```

Default grid: frequencies 25, 50, ..., 500; lengths 50, 200, 1000, 5000,
10000; profiles `mock700` and `tool_calls`. Select a smoke grid with
`--frequencies 25 300 500 --lengths 50 200`. The original multi-channel benchmark
remains available by omitting `--snapshot-sweep`.

Every frequency/profile builds one growing thread; shorter lengths are prefixes.
Thus each backend performs 20 × 2 × 10000 = 400000 graph invocations and the
same number of measured reads, plus warmups and separate heap measurements.
There are 200 output rows per complete backend grid. Resume with identical
arguments, source, revision and environment metadata: completed rows are skipped.
An interrupted scenario is rebuilt from its beginning up to remaining lengths;
resuming does not restore a partially written graph. Abrupt process termination
may leave the current thread in the disposable database.

## Workload and measurement semantics

- Each invocation runs a deterministic `GenericFakeChatModel` script. There are
  **no provider requests**, sleeps or stochastic outputs. Message and tool-call
  IDs derive from the update index. Checkpoint IDs and timing vary between runs.
- `mock700`: 350 ASCII bytes of human content plus 350 of assistant content.
  This is **700 content bytes**, not 700 serialized bytes: message envelopes,
  checkpoint metadata and a file update add overhead.
- `tool_calls`: human request, scripted assistant `read_file` call, approximately
  3.7 KB tool result, then an approximately 600-byte scripted response. It is a
  synthetic deepagents-style workload, not captured production traffic.
- Both `messages` and `files` receive one update in the same node per invocation.
  Files use deterministic content lines and timestamps, adding a new report path
  each time. This deliberately models growing file history, not overwriting a
  fixed set of paths. The reducer merges batches of file dictionaries.
- An update is **one graph invocation**, not one individual message. Each
  invocation has input/start/node checkpoint supersteps; only the node writes
  the two delta channels. The 5000-superstep bound is retained and recorded via
  `LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT`; the grid does not vary it.
- Read statistics describe one warm-cache `get_state` after **every** invocation
  from update 1 through the reported prefix. They are not repeated final-state
  reads or a fixed-thread-length phase distribution. This captures pre-snapshot
  phases as well as snapshots; the maximum is the **observed** slowest read and
  may reflect system noise, not necessarily the deepest replay chain. Its update
  and checkpoint ID are recorded. The measured checkpoint is later deleted.
- Write statistics time end-to-end `graph.invoke`, including hydration, scripted
  model CPU work and checkpoint writes; they are **not saver-only SQL latency**.
- Heap peak comes from a separate traced read of the observed slowest checkpoint.
  `tracemalloc` is off for all latency samples. This is Python heap, not RSS or
  database memory. Reads and writes are sequential, with no concurrent clients.
- Memory storage counts serialized checkpoint, metadata, blob and pending-write
  payload bytes, not Python-object/key overhead. Postgres storage is the sum of
  `pg_column_size(row)` across checkpoints, blobs and writes scoped to the thread.
  It is an in-row/TOAST-sensitive estimate, **not total relation/index/WAL/disk
  usage**, and is not directly interchangeable with the memory payload count.
  Repeated mock text is highly compressible, especially in Postgres: absolute
  storage numbers should not be compared with production chat without matching
  entropy, serialization and storage accounting.
- Metadata records interpreter, platform, git revision, script SHA-256, grid and
  sampling semantics. Initial pilot artifacts predate the final script hash and
  warmup additions; they are preserved as exploratory evidence, not pooled with
  the full-grid results. Timings need repeat runs to assess variance.

## Charts and selecting a frequency

Each CSV produces an SVG per backend/profile/length. The left panel plots storage,
p50 and maximum read latency against frequency, independently divided by each
metric's maximum (0–1). These normalized curves **do not have shared units**;
their crossing is not a physical optimum. The right panel plots storage bytes
against actual p50/max milliseconds.

A marked compromise point is selected separately for p50 and max: discard
storage/latency-dominated points, min-max normalize each objective over that group,
and choose the nondominated point nearest the ideal origin in Euclidean distance.
This is an equal-weight heuristic, not a universal knee or recommended default.
Changing the candidate range, latency statistic or storage/latency weighting can
change it. Partial groups display their actual number of frequencies and must not
be mistaken for completed sweeps. Frequency points above a short thread's update
count can have identical snapshot behavior, making any apparent short-thread
optimum particularly noisy.

For a final recommendation, compare the compromise points across all five thread
lengths and both workloads, apply an explicit latency budget, inspect tail noise,
and repeat promising frequencies. Keep separate choices for growing message and
file workloads if their access/update patterns differ. This suite currently varies
both cadences together and cannot identify their independent optimum.

## Fleet sanity check

The supplied 10000-turn fleet simulation reports frequency 50: 2.9 GB / 44 ms;
300: 488 MB / 244 ms; 500: 307 MB / 404 ms; 1000: 164 MB / 804 ms. Its delta DB time
was about 35 ms and about 1% of fleet DB time. These are externally supplied
context, **not measurements reproduced here**. Our local Postgres, full graph
hydration, files, compressible scripted content and row-size estimate differ.
Do not calibrate the results to force a match or equate local end-to-end latency
with fleet database time. Storage reduction with less frequent snapshots is a
useful directional sanity check, not evidence that absolute fleet values match.
