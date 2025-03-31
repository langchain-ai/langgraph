# LangGraph Checkpoint Postgres

Implementation of LangGraph CheckpointSaver that uses Postgres.

## Dependencies

By default `langgraph-checkpoint-postgres` installs `psycopg` (Psycopg 3) without any extras. However, you can choose a specific installation that best suits your needs [here](https://www.psycopg.org/psycopg3/docs/basic/install.html) (for example, `psycopg[binary]`).

## Usage

> [!IMPORTANT]
> When using Postgres checkpointers for the first time, make sure to call `.setup()` method on them to create required tables. See example below.

> [!IMPORTANT]
> When manually creating Postgres connections and passing them to `PostgresSaver` or `AsyncPostgresSaver`, make sure to include `autocommit=True` and `row_factory=dict_row` (`from psycopg.rows import dict_row`). See a full example in this [how-to guide](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/).

```python
from langgraph.checkpoint.postgres import PostgresSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

DB_URI = "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # call .setup() the first time you're using the checkpointer
    checkpointer.setup()
    checkpoint = {
        "v": 2,
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
    }

    # store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # load checkpoint
    checkpointer.get(read_config)

    # list checkpoints
    list(checkpointer.list(read_config))
```

### Async

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpoint = {
        "v": 2,
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
    }

    # store checkpoint
    await checkpointer.aput(write_config, checkpoint, {}, {})

    # load checkpoint
    await checkpointer.aget(read_config)

    # list checkpoints
    [c async for c in checkpointer.alist(read_config)]
```
