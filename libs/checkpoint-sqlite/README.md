# LangGraph SQLite Checkpoint

Implementation of LangGraph CheckpointSaver that uses SQLite DB (both sync and async, via `aiosqlite`)

## Usage

```python

from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string(":memory:")
checkpoint = {
  "v": 1,
  "ts": "2024-07-31T20:14:19.804150+00:00",
  "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
  "channel_values": {
    "my_key": "meow",
    "node": "node"
  },
  "channel_versions": {
    "__start__": 2,
    "my_key": 3,
    "start:node": 3,
    "node": 3
  },
  "versions_seen": {
    "__input__": {},
    "__start__": {
      "__start__": 1
    },
    "node": {
      "start:node": 2
    }
  },
  "pending_sends": [],
  "current_tasks": {}
}

# store checkpoint
checkpointer.put(thread_config, checkpoint, {})

# load checkpoint
checkpointer.get(thread_config)

# list checkpoints
list(checkpointer.list(thread_config))
```

### Async

```python

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

checkpointer = AsyncSqliteSaver.from_conn_string(":memory:")
checkpoint = {
  "v": 1,
  "ts": "2024-07-31T20:14:19.804150+00:00",
  "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
  "channel_values": {
    "my_key": "meow",
    "node": "node"
  },
  "channel_versions": {
    "__start__": 2,
    "my_key": 3,
    "start:node": 3,
    "node": 3
  },
  "versions_seen": {
    "__input__": {},
    "__start__": {
      "__start__": 1
    },
    "node": {
      "start:node": 2
    }
  },
  "pending_sends": [],
  "current_tasks": {}
}

# store checkpoint
await checkpointer.aput(thread_config, checkpoint, {})

# load checkpoint
await checkpointer.aget(thread_config)

# list checkpoints
[c async for c in checkpointer.alist(thread_config)]
```