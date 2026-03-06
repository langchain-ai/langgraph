# LangGraph Checkpoint CosmosDB

Implementation of LangGraph CheckpointSaver that uses Azure CosmosDB.

## Usage

> [!IMPORTANT]
> To use `CosmosDBSaver` or `AsyncCosmosDBSaver`, you need to provide Azure CosmosDB credentials. You can either:
> - Set `COSMOSDB_ENDPOINT` and `COSMOSDB_KEY` environment variables for key-based authentication
> - Use default Azure credentials (e.g., `az login`, managed identity) if the database and container already exist

### Synchronous

```python
import os
from langgraph.checkpoint.cosmosdb import CosmosDBSaver

# Set environment variables
os.environ["COSMOSDB_ENDPOINT"] = "your_cosmosdb_endpoint"
os.environ["COSMOSDB_KEY"] = "your_cosmosdb_key"  # Optional if using RBAC

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

# Create saver (database and container will be created if they don't exist when using key-based auth)
checkpointer = CosmosDBSaver(
    database_name="your_database",
    container_name="your_container"
)

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

# Store checkpoint
checkpointer.put(write_config, checkpoint, {}, {})

# Load checkpoint
checkpointer.get(read_config)

# List checkpoints
list(checkpointer.list(read_config))
```

### Async

```python
import asyncio
from langgraph.checkpoint.cosmosdb import AsyncCosmosDBSaver

async def main():
    write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
    read_config = {"configurable": {"thread_id": "1"}}

    # Use the async context manager
    async with AsyncCosmosDBSaver.from_conn_info(
        endpoint="your_cosmosdb_endpoint",
        key="your_cosmosdb_key",  # Optional if using RBAC
        database_name="your_database",
        container_name="your_container"
    ) as checkpointer:
        checkpoint = {
            "v": 4,
            "ts": "2024-07-31T20:14:19.804150+00:00",
            "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            "channel_values": {"my_key": "meow", "node": "node"},
            "channel_versions": {
                "__start__": 2,
                "my_key": 3,
                "start:node": 3,
                "node": 3
            },
            "versions_seen": {
                "__input__": {},
                "__start__": {"__start__": 1},
                "node": {"start:node": 2}
            },
        }

        # Store checkpoint
        await checkpointer.aput(write_config, checkpoint, {}, {})

        # Load checkpoint
        await checkpointer.aget(read_config)

        # List checkpoints
        async for checkpoint_tuple in checkpointer.alist(read_config):
            print(checkpoint_tuple)

asyncio.run(main())
```

> [!NOTE]
> `AsyncCosmosDBSaver` uses the native async Azure Cosmos client (`azure.cosmos.aio`) for non-blocking I/O operations. For synchronous code, use `CosmosDBSaver` instead.

### Authentication Options

**Synchronous with key-based authentication:**
```python
import os
from langgraph.checkpoint.cosmosdb import CosmosDBSaver

os.environ["COSMOSDB_ENDPOINT"] = "https://your-account.documents.azure.com:443/"
os.environ["COSMOSDB_KEY"] = "your_primary_key"

checkpointer = CosmosDBSaver(database_name="db", container_name="container")
```

**Synchronous with RBAC / Managed Identity:**
```python
import os
from langgraph.checkpoint.cosmosdb import CosmosDBSaver

# Only endpoint needed; uses DefaultAzureCredential
os.environ["COSMOSDB_ENDPOINT"] = "https://your-account.documents.azure.com:443/"

# For user-assigned managed identity, set AZURE_CLIENT_ID
# os.environ["AZURE_CLIENT_ID"] = "your_client_id"

# Database and container must already exist for RBAC
checkpointer = CosmosDBSaver(database_name="existing_db", container_name="existing_container")
```

**Async with explicit credentials:**
```python
from langgraph.checkpoint.cosmosdb import AsyncCosmosDBSaver

async with AsyncCosmosDBSaver.from_conn_info(
    endpoint="https://your-account.documents.azure.com:443/",
    key="your_primary_key",  # Omit for RBAC/Managed Identity
    database_name="db",
    container_name="container"
) as checkpointer:
    # Use checkpointer
    pass
```

> [!NOTE]
> When using RBAC credentials without a key, the database and container must already exist. An error will be thrown if they don't exist. Key-based authentication will automatically create the database and container if they don't exist.

## Acknowledgments

This library was originally created as a community project by [Kamal (@skamalj)](https://github.com/skamalj) at [skamalj/langgraph_checkpoint_cosmosdb](https://github.com/skamalj/langgraph_checkpoint_cosmosdb). It has been adopted into the official LangGraph repository with the contributions preserved. Thank you for your work!
