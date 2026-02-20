# Critical Contracts — Common Failure Points

These are the 8 requirements most likely to cause test failures. Get these right and you'll pass.

## 1. Store the FULL checkpoint, not just the diff

`aput` receives `new_versions` which lists only CHANGED channels. But `checkpoint["channel_values"]` contains ALL channels. You must store all of them. The `new_versions` parameter is informational — some implementations use it to optimize blob storage by only writing changed blobs, but the simplest correct approach is to serialize and store the entire checkpoint.

**Failing test:** `test_put_incremental_channel_update`, `test_put_new_channel_added`, `test_put_channel_removed`

## 2. Write idempotency with WRITES_IDX_MAP

The unique key for a write is `(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)`.

The `idx` comes from `WRITES_IDX_MAP.get(channel, positional_index)`:
- Special channels: `__error__` → -1, `__interrupt__` → -3, `__scheduled__` → -2, `__resume__` → -4
- Regular channels: use their positional index in the writes list (0, 1, 2, ...)

For **special channels** (all writes are in WRITES_IDX_MAP): use UPSERT (replace on conflict) because these channels get updated in place.

For **regular channels**: use INSERT-ignore-on-conflict to be idempotent — calling `aput_writes` twice with the same `(task_id, idx)` must not create duplicates.

```python
if all(w[0] in WRITES_IDX_MAP for w in writes):
    # UPSERT — replace existing
else:
    # INSERT OR IGNORE — idempotent
```

**Failing test:** `test_put_writes_idempotent`, `test_put_writes_special_channels`

## 3. Namespace isolation

`checkpoint_ns` (from `config["configurable"].get("checkpoint_ns", "")`) is part of the composite key for BOTH checkpoints and writes. Default to empty string `""` if not present.

Two checkpoints with the same `thread_id` and `checkpoint_id` but different `checkpoint_ns` are DIFFERENT checkpoints.

**Failing test:** `test_put_child_namespace`, `test_put_writes_across_namespaces`, `test_get_tuple_respects_namespace`

## 4. Metadata round-trip

Before storing metadata, call `get_checkpoint_metadata(config, metadata)` which merges additional keys from config. Store the result as JSON. When loading, deserialize back to dict.

ALL keys must survive — standard ones (`source`, `step`, `parents`, `run_id`) AND custom keys the caller added.

**Failing test:** `test_put_preserves_metadata`, `test_list_metadata_custom_keys`

## 5. Global search: `alist(None, filter=...)`

When `config` is `None`, `alist` must search across ALL threads. Don't require `thread_id`. Filter by metadata keys if `filter` is provided.

**Failing test:** `test_list_global_search`

## 6. parent_config in CheckpointTuple

When `aput(config, checkpoint, ...)` is called, `config["configurable"].get("checkpoint_id")` is the PARENT checkpoint ID. Store this as `parent_checkpoint_id`.

When returning `CheckpointTuple`:
- If `parent_checkpoint_id` exists: set `parent_config = {"configurable": {"thread_id": ..., "checkpoint_ns": ..., "checkpoint_id": parent_checkpoint_id}}`
- If no parent: set `parent_config = None`

**Failing test:** `test_put_parent_config`, `test_get_tuple_parent_config`

## 7. Pending writes in CheckpointTuple

`pending_writes` must be a list of `(task_id, channel, deserialized_value)` tuples, ordered by `(task_id, idx)`.

Every `aget_tuple` and every tuple yielded by `alist` must include pending writes. Don't forget to query the writes table/collection.

**Failing test:** `test_get_tuple_pending_writes`, `test_list_includes_pending_writes`

## 8. Checkpoint ordering in alist

`alist` must return checkpoints in descending order by `checkpoint_id` (newest first). Checkpoint IDs are UUID-like strings that sort chronologically. Use `ORDER BY checkpoint_id DESC` or equivalent.

The `before` parameter means: only return checkpoints with `checkpoint_id < before_checkpoint_id`.

**Failing test:** `test_list_ordering`, `test_list_before`, `test_list_limit_plus_before`

## 9. Storage design principles

All backends must key checkpoints by `(thread_id, checkpoint_ns, checkpoint_id)` and writes by `(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)`.

- **SQL backends:** Consider a 3rd blobs table keyed by `(thread_id, checkpoint_ns, channel, version)` to avoid re-serializing unchanged large channel values. Only write blobs for channels in `new_versions`; reconstruct all values on read via `channel_versions`.
- **Document/KV backends:** Embed all channel values directly in the checkpoint document/value. Serialize the full checkpoint on every `aput` — simpler and correct.
- **All backends need:** descending `checkpoint_id` ordering for `alist`, metadata field filtering for `alist(filter=...)`, delete by `thread_id` for `adelete_thread`, delete by `metadata.run_id` for `adelete_for_runs`.
