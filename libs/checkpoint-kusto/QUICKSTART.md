# Quick Start Guide - LangGraph Kusto Checkpointer

Get up and running with the Kusto checkpointer in 5 minutes.

## Prerequisites

- Python 3.10+
- Azure Data Explorer cluster
- Azure credentials configured

## Step 1: Provision Kusto Tables (One-time)

In Azure Data Explorer Web UI, run:

```kql
// Create tables
.create-merge table Checkpoints (
    thread_id: string, checkpoint_ns: string, checkpoint_id: string,
    parent_checkpoint_id: string, type: string, checkpoint_json: string,
    metadata_json: string, created_at: datetime
)

.create-merge table CheckpointWrites (
    thread_id: string, checkpoint_ns: string, checkpoint_id: string,
    task_id: string, task_path: string, idx: int, channel: string,
    type: string, value_json: string, created_at: datetime
)

.create-merge table CheckpointBlobs (
    thread_id: string, checkpoint_ns: string, channel: string,
    version: string, type: string, blob: string, created_at: datetime
)

// Add policies
.alter table Checkpoints policy caching hot = 7d
.alter table CheckpointWrites policy caching hot = 7d
.alter table CheckpointBlobs policy caching hot = 7d
.alter-merge table Checkpoints policy retention softdelete = 90d
.alter-merge table CheckpointWrites policy retention softdelete = 90d
.alter-merge table CheckpointBlobs policy retention softdelete = 90d
```

Or run the full `provision.kql` file.

## Step 2: Grant Permissions

```kql
.add database YourDatabase viewers ('aaduser=your@email.com')
.add database YourDatabase ingestors ('aaduser=your@email.com')
```

## Step 3: Install Package

```bash
pip install langgraph-checkpoint-kusto
```

## Step 4: Write Code

### Async Example (Recommended)

```python
import asyncio
from langgraph.checkpoint.kusto.aio import AsyncKustoSaver
from langgraph.graph import StateGraph

async def main():
    # Create checkpointer
    async with AsyncKustoSaver.from_connection_string(
        cluster_uri="https://your-cluster.region.kusto.windows.net",
        database="your-database",
    ) as checkpointer:
        # Validate schema
        await checkpointer.setup()
        
        # Create and compile graph
        graph = StateGraph(YourState)
        # ... add nodes and edges ...
        app = graph.compile(checkpointer=checkpointer)
        
        # Run with checkpointing
        config = {"configurable": {"thread_id": "user-123"}}
        result = await app.ainvoke({"input": "Hello"}, config)
        
        # Flush pending writes
        await checkpointer.flush()

asyncio.run(main())
```

### Sync Example

```python
from langgraph.checkpoint.kusto.aio import AsyncKustoSaver
from langgraph.graph import StateGraph

# Note: Sync methods are wrappers around async, use from background threads
with AsyncKustoSaver.from_connection_string(
    cluster_uri="https://your-cluster.region.kusto.windows.net",
    database="your-database",
) as checkpointer:
    checkpointer.setup()  # Calls async setup internally
    
    graph = StateGraph(YourState)
    # ... add nodes and edges ...
    app = graph.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "user-123"}}
    result = app.invoke({"input": "Hello"}, config)  # Uses sync wrapper
```

## Step 5: Monitor

Enable logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Configuration Options

```python
async with AsyncKustoSaver.from_connection_string(
    cluster_uri="https://...",
    database="mydb",
    
    # Ingestion mode: "queued" (default, reliable) or "streaming" (low latency)
    ingest_mode="queued",
    
    # Batch size: higher = better throughput, higher latency
    batch_size=100,
    
    # Flush interval: seconds between auto-flushes
    flush_interval=30.0,
    
    # Custom serializer (optional)
    serde=MySerializer(),
) as checkpointer:
    # ...
```

## Common Patterns

### Pattern 1: Reliable Background Processing

```python
async with AsyncKustoSaver.from_connection_string(
    cluster_uri="...",
    database="...",
    ingest_mode="queued",     # Reliable
    batch_size=500,           # High throughput
    flush_interval=60.0,      # Flush every minute
) as checkpointer:
    # Process many items
    for item in items:
        await app.ainvoke(item, config)
    
    await checkpointer.flush()  # Ensure all written
```

### Pattern 2: Interactive/Real-time

```python
async with AsyncKustoSaver.from_connection_string(
    cluster_uri="...",
    database="...",
    ingest_mode="streaming",  # Low latency
    batch_size=1,             # Immediate flush
    flush_interval=0.1,       # Quick
) as checkpointer:
    result = await app.ainvoke(user_input, config)
    await checkpointer.flush()
```

### Pattern 3: Resuming from Checkpoint

```python
# Get latest checkpoint
config = {"configurable": {"thread_id": "user-123"}}
checkpoint = await checkpointer.aget_tuple(config)

if checkpoint:
    # Resume from checkpoint
    result = await app.ainvoke(
        checkpoint.checkpoint["channel_values"],
        config
    )
else:
    # Start fresh
    result = await app.ainvoke(initial_state, config)
```

### Pattern 4: Listing Conversation History

```python
config = {"configurable": {"thread_id": "user-123"}}

async for checkpoint in checkpointer.alist(config, limit=10):
    print(f"Checkpoint: {checkpoint.checkpoint['id']}")
    print(f"  Created: {checkpoint.metadata.get('created_at')}")
    print(f"  Step: {checkpoint.metadata.get('step')}")
```

## Troubleshooting

### Issue: "Table not found"
**Solution**: Run `provision.kql` to create tables.

### Issue: "Authorization failed"
**Solution**: Grant Database Viewer + Database Ingestor permissions.

### Issue: Data not appearing after write
**Solution**: 
- For queued mode: Wait 2-5 minutes (normal)
- For streaming mode: Wait <1 second
- Always call `await checkpointer.flush()` after writes

### Issue: Import errors
**Solution**: Ensure package installed: `pip install langgraph-checkpoint-kusto`

## Next Steps

- Read the [full README](README.md) for detailed configuration
- Check [examples/](examples/) for more patterns
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Review [API documentation](langgraph/checkpoint/kusto/) for advanced usage

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/langchain-ai/langgraph/issues)
- **Questions**: [GitHub Discussions](https://github.com/langchain-ai/langgraph/discussions)
- **Docs**: [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

**Ready to go!** 🚀
