# LangGraph Checkpoint SQLServer

Implementation of LangGraph CheckpointSaver that uses Microsoft SQL Server.

## Dependencies

By default `langgraph-checkpoint-sqlserver` installs `pyodbc` without any extras.

## Usage

> [!IMPORTANT]
> When using SQL Server checkpointers for the first time, make sure to call `.setup()` method on them to create required tables. See example below.

> [!IMPORTANT]
> When manually creating SQL Server connections and passing them to `SQLServerSaver`, make sure to include `autocommit=True`. It is required to properly commit the checkpoint tables to the database. Without this, table creation may not be persisted.
>
> **Example of incorrect usage:**
> ```python
> # ❌ This will fail with TypeError during checkpointer operations
> with pyodbc.connect(CONN_STR) as conn:  # Missing autocommit=True
>     checkpointer = SQLServerSaver(conn)
>     checkpointer.setup()  # May not persist tables properly
> ```

```python
from langgraph.checkpoint.sqlserver import SQLServerSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

SQLSERVER_CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=localhost;"
    "PORT=1433;"
    "UID=sa;"
    "PWD=sqlserver123!;"
    "TrustServerCertificate=yes;"
)

with SQLServerSaver.from_conn_string(DB_URI) as checkpointer:
    # call .setup() the first time you're using the checkpointer
    checkpointer.setup()
    checkpoint = {
        "v": 4,
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
    }

    # store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # load checkpoint
    checkpointer.get(read_config)

    # list checkpoints
    list(checkpointer.list(read_config))
```
