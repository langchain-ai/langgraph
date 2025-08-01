# LangGraph Checkpoint Elasticsearch

This library provides an Elasticsearch implementation of the LangGraph checkpoint saver. It allows you to persist LangGraph state in Elasticsearch, enabling features like memory, human-in-the-loop workflows, and time-travel debugging.

## Installation

```bash
pip install langgraph-checkpoint-elasticsearch
```

## Requirements

- Python >=3.9
- Elasticsearch >=8.17,<9.0
- An Elasticsearch cluster with API key authentication

## Quick Start

### Environment Setup

Set your Elasticsearch connection details as environment variables:

```bash
export ES_URL="https://your-elasticsearch-cluster:9200"
export ES_API_KEY="your-api-key"
```

### Basic Usage

```python
from langgraph.checkpoint.elasticsearch import ElasticsearchSaver
from langgraph.graph import StateGraph

# Create the checkpointer
checkpointer = ElasticsearchSaver()

# Use with a graph
builder = StateGraph(dict)
builder.add_node("my_node", lambda x: {"value": x["value"] + 1})
builder.set_entry_point("my_node")
builder.set_finish_point("my_node")

graph = builder.compile(checkpointer=checkpointer)

# Run with memory
config = {"configurable": {"thread_id": "my-thread"}}
result = graph.invoke({"value": 1}, config)
print(result)  # {"value": 2}

# Continue from where we left off
result = graph.invoke({"value": 1}, config)
print(result)  # {"value": 3}
```

### Async Usage

```python
from langgraph.checkpoint.elasticsearch import AsyncElasticsearchSaver

# Create the async checkpointer
checkpointer = AsyncElasticsearchSaver()

# Use with async graph operations
config = {"configurable": {"thread_id": "my-thread"}}
async for chunk in graph.astream({"value": 1}, config):
    print(chunk)

# Clean up
await checkpointer.aclose()
```

## Configuration

### Constructor Parameters

Both `ElasticsearchSaver` and `AsyncElasticsearchSaver` accept the following parameters:

- `es_url` (str, optional): Elasticsearch cluster URL. If not provided, uses `ES_URL` environment variable.
- `api_key` (str, optional): Elasticsearch API key. If not provided, uses `ES_API_KEY` environment variable.
- `index_prefix` (str, optional): Prefix for Elasticsearch indices. Defaults to "langgraph".
- `serde` (SerializerProtocol, optional): Custom serializer for checkpoints. Defaults to JsonPlusSerializer.

### Example with Custom Configuration

```python
from langgraph.checkpoint.elasticsearch import ElasticsearchSaver

checkpointer = ElasticsearchSaver(
    es_url="https://localhost:9200",
    api_key="your-api-key",
    index_prefix="my_app"
)
```

## Authentication

This library **only** supports API key authentication. Other authentication methods (username/password, certificates, etc.) are not supported.

### Creating an API Key

You can create an API key in Elasticsearch using:

```bash
# Using curl
curl -X POST "localhost:9200/_security/api_key" \
-H "Content-Type: application/json" \
-u elastic:password \
-d '{
  "name": "langgraph-key",
  "role_descriptors": {
    "langgraph_role": {
      "cluster": ["monitor"],
      "index": [
        {
          "names": ["langgraph_*"],
          "privileges": ["all"]
        }
      ]
    }
  }
}'
```

Or through Kibana's Security > API Keys interface.

## Index Structure

The library creates two indices in Elasticsearch:

### Checkpoints Index (`{prefix}_checkpoints`)

Stores the main checkpoint data:
- `thread_id`: Thread identifier
- `checkpoint_ns`: Checkpoint namespace
- `checkpoint_id`: Unique checkpoint identifier
- `parent_checkpoint_id`: Parent checkpoint reference
- `checkpoint_data`: Serialized checkpoint (base64 encoded)
- `checkpoint_type`: Serialization type
- `metadata`: Additional metadata
- `timestamp`: Checkpoint creation time
- `channel_versions`: Channel version information

### Writes Index (`{prefix}_writes`)

Stores intermediate writes:
- `thread_id`: Thread identifier
- `checkpoint_ns`: Checkpoint namespace
- `checkpoint_id`: Associated checkpoint
- `task_id`: Task identifier
- `task_path`: Task path
- `idx`: Write index
- `channel`: Channel name
- `value_type`: Value serialization type
- `value_data`: Serialized value (base64 encoded)

## Features

### Memory and Persistence

```python
config = {"configurable": {"thread_id": "conversation-1"}}

# Each invocation remembers the previous state
result1 = graph.invoke({"message": "Hello"}, config)
result2 = graph.invoke({"message": "How are you?"}, config)
```

### Human-in-the-Loop

```python
from langgraph.checkpoint.elasticsearch import ElasticsearchSaver

checkpointer = ElasticsearchSaver()

# Create a graph with interrupts
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_review"])

# Run until interrupt
config = {"configurable": {"thread_id": "review-session"}}
graph.invoke(input, config)

# Resume after human review
graph.invoke(None, config)
```

### Time Travel

```python
# Get checkpoint history
config = {"configurable": {"thread_id": "my-thread"}}
history = list(checkpointer.list(config))

# Go back to a specific checkpoint
old_config = {"configurable": {"thread_id": "my-thread", "checkpoint_id": history[2].config["configurable"]["checkpoint_id"]}}
past_state = graph.get_state(old_config)
```

## Error Handling

The library includes proper error handling for common Elasticsearch issues:

```python
from elasticsearch.exceptions import ConnectionError, AuthenticationException

try:
    checkpointer = ElasticsearchSaver()
    result = graph.invoke(input, config)
except ConnectionError:
    print("Could not connect to Elasticsearch")
except AuthenticationException:
    print("Authentication failed - check your API key")
```

## Performance Considerations

### Bulk Operations

The library uses Elasticsearch's bulk API for efficient writes:

```python
# Multiple writes are automatically batched
writes = [("channel1", data1), ("channel2", data2), ("channel3", data3)]
checkpointer.put_writes(config, writes, task_id)
```

### Index Settings

Indices are created with performance-optimized settings:
- Single shard for small to medium workloads
- No replicas for development
- 1-second refresh interval for near real-time search

### Connection Pooling

The Elasticsearch client automatically handles connection pooling and retries.

## Development

### Running Tests

```bash
# Install dependencies
pip install -e ".[dev]"

# Run unit tests (with mocked Elasticsearch)
pytest tests/

# Run integration tests (requires running Elasticsearch)
docker-compose -f tests/docker-compose.yml up -d
ES_URL=http://localhost:9200 ES_API_KEY=test pytest tests/
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
mypy .
```

## Comparison with Other Checkpointers

| Feature | Elasticsearch | PostgreSQL | SQLite |
|---------|---------------|------------|--------|
| Async Support | ✅ | ✅ | ✅ |
| Horizontal Scaling | ✅ | ✅ | ❌ |
| Full-text Search | ✅ | ✅ | ❌ |
| JSON Queries | ✅ | ✅ | Limited |
| Setup Complexity | Medium | Medium | Low |
| Performance | High | High | Medium |

## License

This library is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
- [GitHub Issues](https://github.com/langchain-ai/langgraph/issues)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## Contributing

Contributions are welcome! Please see the [Contributing Guide](https://github.com/langchain-ai/langgraph/blob/main/CONTRIBUTING.md) for details.