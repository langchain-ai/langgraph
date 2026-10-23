# Distributed DeltaChannel benchmark

The coordinator (`delta_distributed.py`) creates one disposable LangSmith sandbox
per `(frequency, payload)` pair: 40 shards for the default 25–500 step-25 grid.
Each shard owns a local Postgres process and grows one graph to 10000 updates,
recording the five prefix lengths. `GenericFakeChatModel` scripts are deterministic;
no provider API is used. Parallelism changes wall time, not total benchmark work.

## Snapshot contract

Supply an existing, immutable LangSmith snapshot with:

- Linux, Python 3.14, git, PostgreSQL 14 binaries (`initdb`, `pg_ctl`) on PATH;
- a checkout at `/workspace/langgraph`, pinned to the intended revision;
- dependencies installed using `uv sync --frozen --group dev` in
  `/workspace/langgraph/libs/langgraph`;
- `/workspace/langgraph/libs/langgraph/.venv/bin/python` available;
- a writable `/tmp`; a non-root user able to run Postgres, or root plus `runuser`
  and the `postgres` OS account;
- no Postgres listener already occupying loopback port 5432;
- **no secrets baked into the snapshot**. Credentials are needed only by the
  coordinator, not the shard. Use a dedicated snapshot, not a personal workspace
  image with unrelated processes or credentials.

The coordinator uploads only the benchmark source and shard worker. Libraries,
interpreter and dependency versions come from the supplied snapshot. Every job
uses that same snapshot ID and explicit CPU/memory/disk configuration. Snapshot
construction/provisioning is intentionally separate; no image or paid fleet has
been created by this change. This is a normal standalone benchmark project using
the LangSmith SDK, not a change to the Open SWE workspace image.

## Plan first

From `libs/langgraph`:

```bash
uv run python bench/delta_distributed.py \
  --snapshot-id YOUR_SNAPSHOT_ID --concurrency 4 --timeout 14400 \
  --vcpus 2 --memory-gib 8 --disk-gib 30 --output /tmp/delta-fleet
```

Without `--execute`, this only writes `plan.json`; it does not instantiate an SDK
client, require credentials or provision resources. Values above are **examples,
not selected budgets**. Agree on them before executing. With 40 shards and a
4-hour benchmark timeout, the benchmark allowance alone is 160 sandbox-hours;
readiness and result collection add overhead. The printed allowance is an
estimate, not a monetary billing cap. Concurrency is capped at 40 and each shard
benchmark timeout at 24 hours. There is no automatic retry storm.

Start with a small pilot (new output directory):

```bash
uv run python bench/delta_distributed.py \
  --snapshot-id YOUR_SNAPSHOT_ID --concurrency 2 --timeout 900 \
  --vcpus 2 --memory-gib 8 --disk-gib 30 --output /tmp/delta-pilot \
  --frequencies 25 300 --lengths 50 200 --profiles mock700
```

Review the generated plan, then repeat the exact command with `--execute` to
create billable resources. The SDK reads `LANGSMITH_API_KEY` and
`LANGSMITH_ENDPOINT` from the coordinator's environment. Never pass keys in CLI
arguments, commit them, or bake them into the snapshot.

## Lifecycle, recovery and outputs

- Bounded thread-pool concurrency; each worker has its own SDK client.
- Readiness timeout 300 seconds, benchmark subprocess timeout from the plan,
  and remote execution timeout 120 seconds longer for startup/cleanup.
- Completed CSV and metadata are fetched before sandbox deletion. A shard is
  marked successful only after its sandbox has been deleted.
- A shard's `attempt.json` records its sandbox name **before** creation. On a
  process crash, ambiguous create or failed deletion, inspect that named resource
  in LangSmith and delete it if it exists. Do not bulk-delete other sandboxes.
  After verifying it is absent, set this ledger's state to `deleted` to allow
  retry. An unresolved ledger blocks retry rather than leaking more resources.
- Idle stop is 10 minutes and deletion after stop is 10 minutes, providing a
  fallback for coordinator crashes. This is **not** a strict hard lifetime cap:
  verify sandbox cleanup after interruption. Active exec/SDK lease behavior can
  affect idle detection.
- A shard failure stops admission of further jobs; already-running jobs finish
  and clean up. Exceptions shown by workers use only their type, avoiding
  inadvertent credential exposure. Failed command stdout/stderr are not copied
  into shared artifacts. Inspect the sandbox before cleanup separately if deep
  diagnostics are needed; automatic cleanup prioritizes preventing spend.
- Rerun the identical plan/output directory to skip successful shards. Failed
  shards restart from zero; partial remote CSVs are not recovered. Each completed
  shard is an independent resume unit.
- Launch **only one coordinator per output directory**; no cross-process lock is
  implemented. Do not modify benchmark/worker source during a run. Source hashes
  and plan equality reject changed-source or changed-budget resume.
- Merge requires every expected frequency/profile/length exactly once. Missing,
  duplicate, foreign and nonfinite rows fail validation. `combined.csv` and SVG
  plots are emitted only for a complete plan. Per-shard CSV/metadata stay available.

## Interpretation and verification limits

See [measurement semantics](delta-results/README.md) before interpreting results.
The normalized Pareto compromise is a heuristic, not a universal default. Full
state hydration and mock payload compression can dominate the apparent tradeoff.
Keep the end-to-end graph as the reference; direct checkpointer seeding is not
implemented because equivalence has not been demonstrated. Read sampling remains
one read per update to avoid treating assumed snapshot phase as measured truth.

The coordinator lifecycle and failure paths are locally tested with a fake SDK;
the benchmark itself has run against local Postgres. A real remote pilot still
needs a prepared snapshot and an agreed concurrency/runtime budget. Do not treat
mocked orchestration tests as evidence that a remote fleet has run successfully.
