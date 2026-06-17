# delta-channel-dump

Recover messages (and other channels) from a Postgres-backed LangGraph thread
written by **langgraph >= 1.2** (DeltaChannel format) — including **LangGraph
Server / langgraph-api** deployments on the Postgres runtime, **deepagents
0.6.x**, or any OSS app using `PostgresSaver` — before rolling back to an older
runtime such as **deepagents 0.5.x / langgraph < 1.2**.

langgraph-api uses the same `checkpoints` / `checkpoint_blobs` / `checkpoint_writes`
schema as OSS `checkpoint-postgres`; this script reads those tables directly.

On the older runtime, `add_messages` does not understand the `EXT_DELTA_SNAPSHOT`
msgpack ext code and silently returns an empty list for affected channels. This
tool reads the raw checkpoint blobs from Postgres and emits JSON you can inspect
and re-apply via `update_state` (LangGraph Server SDK) or `graph.update_state`
(OSS).

## Install

```bash
pip install "psycopg[binary]" ormsgpack
```

## Run

```bash
export DATABASE_URI=postgres://user:pass@host:5432/dbname
python3 dump.py \
    --thread-id <uuid> \
    --channel messages \
    [--channel files ...] \
    [--checkpoint-id <uuid>] \
    [--checkpoint-ns ""] \
    --output recovery.json
```

- `--thread-id` (required): thread UUID
- `--channel` (required, repeatable): channel names to recover
- `--checkpoint-id` (optional): target checkpoint; defaults to latest
- `--checkpoint-ns` (optional): namespace; defaults to `""`
- `--database-uri` (optional): Postgres URI; defaults to `DATABASE_URI` env var
- `--output` (optional): output file; defaults to stdout

## Output

```json
{
  "thread_id": "...",
  "checkpoint_ns": "",
  "target_checkpoint_id": "...",
  "parent_checkpoint_id": "...",
  "channels": {
    "messages": {
      "delta_kind": "snapshot",
      "seed_checkpoint_id": "...",
      "seed_version": "...",
      "seed": [{ "type": "ai", "content": "...", "id": "ai-0" }],
      "writes": [
        {
          "checkpoint_id": "...",
          "task_id": "...",
          "idx": 0,
          "value": [{ "type": "ai", "content": "...", "id": "ai-10" }]
        }
      ]
    }
  }
}
```

`delta_kind` is one of:

- `snapshot` — DeltaChannel snapshot blob (`channel_values[ch] == true`)
- `legacy_plain` — pre-DeltaChannel inline or blob value
- `no_seed` — walked to root without finding a populated ancestor

`writes` are ordered oldest-to-newest (the order a reducer would replay them).

## Reducing back to a single list

For deepagents-style messages, combine seed and writes, then deduplicate:

```python
import json

data = json.load(open("recovery.json"))
ch = data["channels"]["messages"]
messages = list(ch["seed"] or [])
for w in ch["writes"]:
    messages.extend(w["value"] or [])

# Dedup by id, keep last; drop RemoveMessage tombstones
by_id = {}
for m in messages:
    if isinstance(m, dict) and m.get("type") == "remove":
        by_id.pop(m.get("id"), None)
    elif isinstance(m, dict) and m.get("id"):
        by_id[m["id"]] = m
    else:
        by_id[id(m)] = m
reduced = list(by_id.values())
```

This approximates `_messages_delta_reducer` semantics; adjust for your graph.

## Re-applying

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:8123")
await client.threads.update_state(
    thread_id,
    values={"messages": reduced},
)
```

Review the recovered JSON before calling `update_state`. This tool is
read-only and intentionally does not mutate the database.

## Scope / non-goals (v1)

- **Postgres only** — OSS `PostgresSaver` or langgraph-api Postgres runtime; not
  inmem, gRPC core, Mongo, or Redis checkpointer backends
- **No AES decryption** (`LANGGRAPH_AES_KEY`) or custom encryption
- **No reducer** — raw seed + writes only
- **No automatic `update_state`** — operator applies manually

## Copying

`dump.py` is self-contained. Copy it anywhere; only `psycopg[binary]` and
`ormsgpack` are required at runtime. No langgraph imports.
