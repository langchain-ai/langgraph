# LangGraph Checkpoint Kusto

[![PyPI version](https://badge.fury.io/py/langgraph-checkpoint-kusto.svg)](https://badge.fury.io/py/langgraph-checkpoint-kusto)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Kusto implementation of LangGraph checkpoint saver.

## Overview

This library provides a production-ready checkpointer for [LangGraph](https://github.com/langchain-ai/langgraph) that persists checkpoints to Kusto. It replicates the behavior and contracts of the official Postgres checkpointer while leveraging Kusto's scalability and analytics capabilities.

### Key Features

- ✅ **Full LangGraph Compatibility**: Implements all required `BaseCheckpointSaver` methods
- ✅ **Async/Sync Support**: Complete async implementation with sync wrappers
- ✅ **Scalable**: Leverages Kusto's distributed architecture for high throughput
- ✅ **Observable**: Built-in structured logging and metrics
- ✅ **Production-Ready**: Comprehensive error handling, retries, and type safety
- ✅ **Streaming Ingestion**: Low-latency data ingestion (<1 second availability)
- ✅ **Optimized Schema**: Uses Kusto's columnar storage with `dynamic` type for efficient blob storage (v2.0+)
- ✅ **Materialized Views**: Pre-computed `arg_max()` for fast "latest checkpoint" queries

## Architecture

```
┌─────────────────┐
│   LangGraph     │
│   Application   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  AsyncKustoSaver /          │
│  KustoSaver                 │
├─────────────────────────────┤
│  - aget_tuple / get_tuple   │
│  - alist / list             │
│  - aput / put               │
│  - aput_writes / put_writes │
└────────┬────────────────────┘
         │
         ├──────────────┬───────────────┐
         ▼              ▼               ▼
    ┌─────────┐   ┌──────────┐   ┌──────────┐
    │ Kusto   │   │  Kusto   │   │  Kusto   │
    │ Query   │   │ Streaming│   │ Streaming│
    │ Client  │   │ Ingest   │   │ Ingest   │
    │         │   │ Client   │   │  Client  │
    └────┬────┘   └────┬─────┘   └────┬─────┘
         │             │               │
         └──────┬──────┴───────┬───────┘
                │              │
                ▼              ▼
           ┌────────────────────┐
           │  Azure Data        │
           │  Explorer Cluster  │
           └────────────────────┘
```

## Installation

```bash
pip install langgraph-checkpoint-kusto
```

Or with uv:

```bash
uv add langgraph-checkpoint-kusto
```

## Getting a Kusto Endpoint

You have several options to get access to a Kusto cluster:

### Option 1: Azure Data Explorer (ADX) Cluster

Deploy a dedicated cluster in Azure. Ideal for production workloads.

- [Create an Azure Data Explorer cluster](https://learn.microsoft.com/en-us/azure/data-explorer/create-cluster-database-portal)
- Free tier available for development/testing

### Option 2: Microsoft Fabric Eventhouse

Use Kusto as part of Microsoft Fabric's Real-Time Intelligence capabilities.

- [Create an Eventhouse in Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/create-eventhouse)
- Integrated with Fabric workspace and OneLake

### Option 3: Free Cluster

Get started immediately with a free cluster for learning and experimentation.

- [Access free cluster at dataexplorer.azure.com](https://dataexplorer.azure.com/freecluster)
- No Azure subscription required
- Perfect for tutorials and testing

## Quick Start

### 1. Provision Kusto Tables

Run the provided `provision.kql` script in your Kusto cluster:

```bash
# Via Azure CLI
az kusto script create \
  --cluster-name <your-cluster> \
  --database-name <your-database> \
  --script-content @provision.kql

# Or via Kusto Web UI
# Copy and run the contents of provision.kql
```

### 2. Basic Usage (Async)

```python
import asyncio
from langgraph.checkpoint.kusto.aio import AsyncKustoSaver
from langgraph.graph import StateGraph

async def main():
    # Initialize the checkpointer
    async with AsyncKustoSaver.from_connection_string(
        cluster_uri="https://your-cluster.region.kusto.windows.net",
        database="your-database",
        # Uses DefaultAzureCredential (Managed Identity, Azure CLI, etc.)
    ) as checkpointer:
        # Optional: Validate setup
        await checkpointer.setup()
        
        # Use with LangGraph
        graph = StateGraph(...)
        graph = graph.compile(checkpointer=checkpointer)
        
        # Run with checkpointing
        config = {"configurable": {"thread_id": "user-123"}}
        result = await graph.ainvoke({"input": "Hello"}, config)

asyncio.run(main())
```

### 3. Basic Usage (Sync)

```python
from langgraph.checkpoint.kusto import KustoSaver
from langgraph.graph import StateGraph

# Initialize the checkpointer
with KustoSaver.from_connection_string(
    cluster_uri="https://your-cluster.region.kusto.windows.net",
    database="your-database",
) as checkpointer:
    checkpointer.setup()
    
    # Use with LangGraph
    graph = StateGraph(...)
    graph = graph.compile(checkpointer=checkpointer)
    
    # Run with checkpointing
    config = {"configurable": {"thread_id": "user-123"}}
    result = graph.invoke({"input": "Hello"}, config)
```

## Configuration

### Connection Options

```python
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

# Option 1: Default Azure Credential (recommended for production)
checkpointer = AsyncKustoSaver.from_connection_string(
    cluster_uri="https://cluster.region.kusto.windows.net",
    database="mydb",
)

# Option 2: Managed Identity
checkpointer = AsyncKustoSaver.from_connection_string(
    cluster_uri="https://cluster.region.kusto.windows.net",
    database="mydb",
    credential=ManagedIdentityCredential(),
)

# Option 3: Connection String (not recommended for production)
checkpointer = AsyncKustoSaver.from_connection_string(
    cluster_uri="https://cluster.region.kusto.windows.net",
    database="mydb",
    credential="your-app-id;your-app-key;your-tenant-id",
)
```

### Performance Tuning

```python
checkpointer = AsyncKustoSaver(
    # Batch size for writes (higher = better throughput, slightly higher latency)
    batch_size=100,  # Default: 100
    
    # Auto-flush interval in seconds
    flush_interval=30.0,  # Default: 30.0
    
    # Query timeout in seconds
    query_timeout=30,  # Default: 30
    
    # Retry configuration
    max_retries=3,  # Default: 3
    retry_backoff=1.0,  # Default: 1.0 (exponential)
)
```

## Advanced Usage

### Custom Serialization

```python
from langgraph.checkpoint.serde.base import SerializerProtocol

class CustomSerializer(SerializerProtocol):
    # Implement custom serialization logic
    pass

checkpointer = AsyncKustoSaver.from_connection_string(
    cluster_uri="...",
    database="...",
    serde=CustomSerializer(),
)
```

### Manual Flushing

```python
# Manually flush pending writes (optional - auto-flush every 30s by default)
await checkpointer.flush()

# Or use as a context manager for automatic flushing
async with checkpointer:
    await checkpointer.aput(config, checkpoint, metadata, versions)
    # Flushed automatically on exit
```

### Monitoring & Observability

```python
import logging

# Enable structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Access metrics (example with custom handler)
checkpointer = AsyncKustoSaver.from_connection_string(...)

# Metrics are logged automatically:
# - checkpoint.put.duration_seconds (histogram)
# - checkpoint.get_tuple.duration_seconds (histogram)
# - checkpoint.list.duration_seconds (histogram)
# - checkpoint.batch_queue_size (gauge)
```

## Performance Characteristics

### Query Optimization with Materialized Views

This checkpointer uses **Kusto materialized views** with `arg_max()` aggregation for optimal "latest checkpoint" query performance:

- **Materialized View**: `LatestCheckpoints` pre-computes `arg_max(checkpoint_id, *)` by `thread_id` and `checkpoint_ns`
- **Performance Gain**: ~10-100x faster than `ORDER BY checkpoint_id DESC | TAKE 1` on large datasets
- **Update Latency**: Near real-time (seconds to minutes depending on cluster load)
- **Use Case**: Automatically used when calling `aget_tuple()` without a specific `checkpoint_id`

**Query Pattern Comparison**:

| Pattern | Approach | Scan Cost | Typical Latency |
|---------|----------|-----------|-----------------|
| **Materialized View** | Pre-computed `arg_max()` | O(1) - index lookup | 10-50ms |
| **ORDER BY + TAKE** | Full table scan + sort | O(n log n) | 100-1000ms+ |

The materialized view is created automatically by the `provision.kql` script.

### Streaming Ingestion

This library uses **streaming ingestion** which provides:

- **Low Latency**: Data available in <1 second after flushing
- **High Throughput**: Suitable for medium to high volume workloads  
- **Reliable**: Built-in retry logic and error handling

### Recommended Configurations

**High Throughput (Batch Processing)**:
```python
batch_size=500
flush_interval=60.0
```

**Low Latency (Interactive)**:
```python
batch_size=1
flush_interval=0.1
```

**Balanced (General Purpose)**:
```python
batch_size=100
flush_interval=30.0
```

## Troubleshooting

### Issue: "Table not found"
**Solution**: Run `provision.kql` to create required tables.

### Issue: "Authorization failed"
**Solution**: Ensure your credential has these permissions:
- `Database Viewer` (for queries)
- `Database Ingestor` (for writes)

### Issue: "Data not appearing immediately"
**Solution**: 
- Streaming ingestion typically shows data within <1 second after flushing
- Ensure `flush()` is called after writes
- Check ingestion status via Kusto `.show ingestion failures` command

### Issue: "High latency on list() queries"
**Solution**: 
- Ensure caching policy is configured (see `provision.kql`)
- The materialized view optimizes "latest checkpoint" queries
- Reduce page size with `limit` parameter

## Security Best Practices

### 1. Use Managed Identity (Production)

```python
from azure.identity import ManagedIdentityCredential

checkpointer = AsyncKustoSaver.from_connection_string(
    cluster_uri="...",
    database="...",
    credential=ManagedIdentityCredential(),
)
```

### 2. Least Privilege Permissions

```kusto
// Grant minimal permissions
.add database <database> viewers ('aadapp=<app-id>;<tenant-id>')
.add database <database> ingestors ('aadapp=<app-id>;<tenant-id>')
```

### 3. Connection String Security

- **Never** commit connection strings to source control
- Use Azure Key Vault or environment variables
- Rotate credentials regularly

### 4. Network Security

- Enable VNet integration for Kusto cluster
- Use Private Endpoints
- Configure firewall rules

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development setup and guidelines.

### Running Tests

```bash
# Unit tests only
make test-unit

# Integration tests (requires live Kusto cluster)
export KUSTO_CLUSTER_URI="https://..."
export KUSTO_DATABASE="test_db"
make test-integration

# All tests
make test
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type checking
mypy langgraph
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/langchain-ai/langgraph/issues)
- **Discussions**: [GitHub Discussions](https://github.com/langchain-ai/langgraph/discussions)
- **Documentation**: [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- **Tutorials**: See [examples/](examples/) directory for hands-on guides

