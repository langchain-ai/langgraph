---
name: build-checkpointer
description: Build a LangGraph checkpoint saver implementation that passes all conformance tests. Use when creating a new checkpointer for any storage backend (Redis, DynamoDB, MongoDB, etc.) or wrapping an existing storage client.
disable-model-invocation: true
user-invocable: true
argument-hint: [storage-backend]
---

# Build a Conformant LangGraph Checkpointer

You are building a LangGraph checkpoint saver for the **$ARGUMENTS** storage backend. Your goal is FULL conformance: all 82 tests across 8 capabilities must pass.

Read [interface-reference.md](interface-reference.md) for method signatures, data structures, and the conformance test harness template.
Read [critical-contracts.md](critical-contracts.md) for the 8 most common failure points.
Read [sqlite-reference.md](sqlite-reference.md) for patterns from a working implementation.

## Ground Rules

**You are not done until ALL conformance tests pass.** Do not stop after writing code — you must run the tests, read failures, fix, and re-run in a loop until you see FULL conformance. If you hit a wall, try a different approach rather than giving up.

**No hacks or shortcuts.** Specifically:
- Do NOT skip or xfail tests to make the suite "pass"
- Do NOT weaken assertions or modify the conformance test suite itself
- Do NOT use `# type: ignore` to paper over real type mismatches
- Do NOT store data in global/module-level dicts to fake persistence — use the actual storage backend
- Do NOT disable serialization or store raw Python objects — use `self.serde.dumps_typed` / `loads_typed`
- Do NOT catch and swallow exceptions to hide failures

**Flag security concerns.** As you implement:
- Ensure all queries use parameterized statements — never interpolate user-provided values (thread_id, checkpoint_id, etc.) into SQL or query strings
- Check for injection risks in metadata filtering (JSON path queries, NoSQL operators, etc.)
- Ensure connection credentials are not hardcoded in the implementation — accept them as constructor args
- Flag any backend client library that has known CVEs or security advisories
- If the backend requires TLS/auth, note it prominently in the constructor docstring

**Ask the user for help when you need it.** Don't guess or assume — ask when:
- You need database connection details, credentials, or access
- You're unsure which client library or driver to use
- You need the user to start/stop a database service
- You're stuck on a test failure after multiple attempts
- You're unsure about a design decision (e.g., schema layout, indexing strategy)

**Safety checks — ask the user to confirm:**
- "Is this database safe to use for testing? Please confirm it is NOT a production database." (before running any tests that create/delete tables)
- "I'm about to create tables and run destructive test operations (INSERT, DELETE, DROP). Is this OK?" (before first test run)
- "What connection string / credentials should I use?" (never assume defaults for non-local databases)

## Step 1: Understand the target

Determine the storage backend from the arguments. If no arguments were provided, ask the user:
- What storage backend? (Redis, DynamoDB, MongoDB, Cassandra, etc.)
- From scratch, or wrapping an existing client/library?
- Any connection/authentication requirements?
- How do I connect to a test instance? (Docker compose, local install, cloud sandbox, etc.)

Install the backend's Python client library if needed.

## Step 2: Scaffold the package

Create `libs/checkpoint-<backend>/` with this structure:

```
libs/checkpoint-<backend>/
  pyproject.toml
  Makefile
  langgraph/
    checkpoint/
      <backend>/
        __init__.py      # Main implementation
  tests/
    test_conformance.py  # Conformance harness
```

The `pyproject.toml` should depend on:
- `langgraph-checkpoint` (the base interfaces)
- The backend's client library
- `langgraph-checkpoint-conformance` as a test dependency

Model the `Makefile` after `libs/checkpoint-sqlite/Makefile`.

## Step 3: Implement the checkpointer

Subclass `BaseCheckpointSaver` and implement ALL 8 async methods:

**Required (5):** `aput`, `aget_tuple`, `alist`, `aput_writes`, `adelete_thread`
**Extended (3):** `adelete_for_runs`, `acopy_thread`, `aprune`

Key implementation guidance:

1. **Storage layout depends on your backend.** Choose the layout that fits your backend's strengths:
   - **SQL databases (Postgres, MySQL, SQLite):** Use 3 tables — checkpoints, checkpoint_blobs (channel values keyed by version), checkpoint_writes. The blobs table avoids re-serializing unchanged large values on every checkpoint write. Inline primitive channel values (str, int, float, bool, None) in the checkpoint JSON; store non-primitives as blobs keyed by `(thread_id, checkpoint_ns, channel, version)`.
   - **Document stores (MongoDB, DynamoDB, Firestore):** Use 2 collections — checkpoints (with channel values embedded) and writes. Serialize the full checkpoint including all channel values. The blob optimization adds complexity without much benefit in document stores.
   - **Key-value stores (Redis, etcd):** Use composite keys to namespace checkpoints and writes. Store serialized checkpoint + writes as values.

   See `critical-contracts.md` for composite key design.

3. **Serialize blobs and writes with `self.serde`** — use `self.serde.dumps_typed(value)` which returns `(type_str, bytes)` and `self.serde.loads_typed((type_str, bytes))` for deserialization. For CPU-bound serialization, use `asyncio.to_thread()` to avoid blocking the event loop.

4. **Serialize metadata as JSON** — use `get_checkpoint_metadata(config, metadata)` to merge config metadata before storing, then `json.dumps()`. Deserialize with `json.loads()`. Metadata is small enough to store inline (no blob table needed).

5. **Handle `new_versions` correctly** — this is the #1 source of failures. The checkpoint's `channel_values` contains ALL channels, but `new_versions` only lists CHANGED channels. If using a blob table (SQL pattern), only write blobs for channels in `new_versions` and reference all versions in the checkpoint JSON. If storing the full checkpoint (document/KV pattern), just serialize all of `checkpoint["channel_values"]` — simpler and correct.

6. **Handle `WRITES_IDX_MAP`** — special channels (ERROR, INTERRUPT, SCHEDULED, RESUME) use fixed negative indices. Regular writes use their positional index. Special channel writes should UPSERT (replace on conflict); regular writes should be idempotent (ignore on conflict).

7. **Return correct `parent_config`** — the `checkpoint_id` in the incoming config to `aput` is the parent. When returning `CheckpointTuple`, set `parent_config` to a config with that parent checkpoint_id, or None if there was no parent.

### Production-quality patterns

Go beyond "just passing tests" — build something that performs well at scale:

- **Connection pooling.** Accept both a single connection and a connection pool in the constructor. Use a pool for production workloads. For Postgres, use `psycopg_pool.AsyncConnectionPool`. For Redis, use the client's built-in pool. Document which to use.
- **Use native backend features.** Don't treat the backend as a dumb key-value store. Examples:
  - Postgres: use JSONB containment (`@>`) for metadata filtering, `COPY FROM STDIN` for bulk inserts, `DISTINCT ON` for pruning, pipeline mode for batching
  - Redis: use Lua scripts for atomic operations, sorted sets for ordering, hash fields for channel blobs
  - DynamoDB: use query vs scan appropriately, batch write items, GSIs for metadata filtering
  - MongoDB: use `$match` aggregation stages, bulk write operations, compound indexes
- **Batch writes where possible.** In `aput`, group the checkpoint insert and blob upserts into a single round-trip (pipeline, transaction, or batch write). Don't make N separate calls for N blobs.
- **Fetch writes alongside checkpoints in a single query.** Use subqueries, JOINs, or array aggregation to avoid N+1 patterns where you fetch N checkpoints then query writes for each one separately.
- **Use `asyncio.to_thread()` for CPU-bound serialization** — `serde.dumps_typed` and `serde.loads_typed` can be expensive for large values. Offload to a thread to keep the event loop responsive.
- **Add appropriate indexes.** At minimum: primary/unique keys on all collections, and an index on `thread_id` for `adelete_thread`. For `adelete_for_runs`, consider an index on the metadata `run_id` field if the backend supports it.

## Step 4: Run conformance and iterate — DO NOT STOP UNTIL GREEN

```bash
cd libs/checkpoint-<backend>
pip install -e ".[test]"
python -m pytest tests/test_conformance.py -x -v
```

Or run via `make test` if your Makefile is set up.

**This is the core of the task.** You MUST loop:

1. Run the conformance tests
2. Read the failure output carefully — it tells you exactly which contract was violated
3. Understand WHY it failed — read the test source in `libs/checkpoint-conformance/langgraph/checkpoint/conformance/spec/` if the error message isn't clear
4. Fix the implementation with a proper solution (not a hack — see Ground Rules)
5. Re-run. Go back to step 1.

**Do not stop until `report.passed_all()` returns True.** If you've been through 5+ iterations and are still failing, step back and re-read the critical-contracts.md and the failing test source code. The answer is always in the test — it specifies exactly what the contract requires.

**If you're blocked, ask the user.** Common things to ask about:
- "The database isn't reachable — can you check the connection / start the service?"
- "I'm stuck on this test failure after N attempts — here's what I've tried, can you help?"
- "I need to install this package / run this command — is that OK?"

Common failure patterns:
- `test_put_incremental_channel_update` fails → you're not storing all channel values, only the ones in `new_versions`
- `test_put_writes_idempotent` fails → your write upsert logic is wrong, check `WRITES_IDX_MAP` handling
- `test_list_global_search` fails → you're requiring a thread_id when config is None
- `test_get_tuple_pending_writes` fails → writes not ordered by `(task_id, idx)` or missing `task_id` in tuple
- `test_list_metadata_filter_*` fails → metadata filtering not checking all keys, or not handling custom keys

## Step 5: Final verification and review

Run the full suite one more time with verbose output:

```bash
python -m pytest tests/test_conformance.py -v
```

Confirm the output shows FULL conformance (all 82 tests pass). The report should show:
- PUT: all pass
- PUT_WRITES: all pass
- GET_TUPLE: all pass
- LIST: all pass
- DELETE_THREAD: all pass
- DELETE_FOR_RUNS: all pass
- COPY_THREAD: all pass
- PRUNE: all pass

Then do a final review of your implementation:

1. **Run `make lint` and `make format`** to clean up the code
2. **Security review** — check for SQL/query injection, hardcoded credentials, unvalidated inputs
3. **Performance review** — check for N+1 query patterns (fetching writes per checkpoint in a loop), missing indexes on frequently-filtered columns, unnecessary full-table scans
4. **Report findings** — tell the user about any security concerns, performance considerations, or caveats about the implementation
