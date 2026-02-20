# SQLite Implementation Reference

Working patterns from `libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/aio.py`.

## Schema

```sql
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
```

## aput pattern

```python
async def aput(self, config, checkpoint, metadata, new_versions):
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
    parent_checkpoint_id = config["configurable"].get("checkpoint_id")

    type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
    serialized_metadata = json.dumps(
        get_checkpoint_metadata(config, metadata), ensure_ascii=False
    ).encode("utf-8", "ignore")

    # UPSERT checkpoint row
    await db.execute(
        "INSERT OR REPLACE INTO checkpoints (...) VALUES (...)",
        (thread_id, checkpoint_ns, checkpoint["id"], parent_checkpoint_id,
         type_, serialized_checkpoint, serialized_metadata),
    )

    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint["id"],
        }
    }
```

## aget_tuple pattern

```python
async def aget_tuple(self, config):
    checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

    if checkpoint_id := get_checkpoint_id(config):
        # Fetch specific checkpoint
        query = "... WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?"
    else:
        # Fetch latest
        query = "... WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1"

    row = await fetch_one(query, ...)
    if not row:
        return None

    # Fetch pending writes for this checkpoint
    writes = await fetch_all(
        "SELECT task_id, channel, type, value FROM writes "
        "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
        "ORDER BY task_id, idx",
        ...
    )

    return CheckpointTuple(
        config={"configurable": {"thread_id": ..., "checkpoint_ns": ..., "checkpoint_id": ...}},
        checkpoint=self.serde.loads_typed((type_, blob)),
        metadata=json.loads(metadata_blob),
        parent_config=(
            {"configurable": {"thread_id": ..., "checkpoint_ns": ..., "checkpoint_id": parent_id}}
            if parent_id else None
        ),
        pending_writes=[
            (task_id, channel, self.serde.loads_typed((type_, value)))
            for task_id, channel, type_, value in writes
        ],
    )
```

## alist pattern

```python
async def alist(self, config, *, filter=None, before=None, limit=None):
    # Build WHERE clause dynamically
    where_clauses = []
    params = []

    if config is not None:
        where_clauses.append("thread_id = ?")
        params.append(config["configurable"]["thread_id"])
        if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
            where_clauses.append("checkpoint_ns = ?")
            params.append(checkpoint_ns)

    if filter:
        # Filter on metadata JSON — for each key-value pair:
        for key, value in filter.items():
            where_clauses.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(json.dumps(value) if not isinstance(value, (str, int, float)) else value)

    if before:
        before_id = before["configurable"]["checkpoint_id"]
        where_clauses.append("checkpoint_id < ?")
        params.append(before_id)

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    query = f"SELECT ... FROM checkpoints {where} ORDER BY checkpoint_id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    # For each checkpoint row, also fetch its writes (same as aget_tuple)
```

## aput_writes pattern

```python
async def aput_writes(self, config, writes, task_id, task_path=""):
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
    checkpoint_id = config["configurable"]["checkpoint_id"]

    # Choose UPSERT vs INSERT-ignore based on channel types
    if all(w[0] in WRITES_IDX_MAP for w in writes):
        query = "INSERT OR REPLACE INTO writes (...) VALUES (...)"
    else:
        query = "INSERT OR IGNORE INTO writes (...) VALUES (...)"

    rows = [
        (thread_id, checkpoint_ns, checkpoint_id, task_id,
         WRITES_IDX_MAP.get(channel, idx), channel,
         *self.serde.dumps_typed(value))
        for idx, (channel, value) in enumerate(writes)
    ]
    await executemany(query, rows)
```

## adelete_thread pattern

```python
async def adelete_thread(self, thread_id):
    await execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
    await execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
```

## Key takeaway

The SQLite implementation is ~300 lines and is the simplest correct reference. It uses 2 tables and serializes the full checkpoint as a single blob.

- SQLite patterns show the simplest correct implementation of every contract
- SQL backends can add a 3rd blobs table for performance (see `interface-reference.md` Schema Design section)
- NoSQL backends should adapt the contracts to native idioms — focus on `critical-contracts.md`
- Don't port SQL patterns to NoSQL; use your backend's native features (document embedding, sorted sets, composite keys, etc.)
