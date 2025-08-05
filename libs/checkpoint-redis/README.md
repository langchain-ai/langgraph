# LangGraph Redis Cache

Implementation of LangGraph node-level cache that uses Redis (both sync and async)

## Usage

```python
import redis
from langgraph.cache.redis import RedisCache

# Create Redis client
client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

# Create Redis cache
cache = RedisCache(client, prefix="my_app:")

# Use with LangGraph
from langgraph import StateGraph

builder = StateGraph(...)
graph = builder.compile(cache=cache)

# Cache will automatically store and retrieve node execution results
result = graph.invoke({"input": "data"})
```

### Async

```python
import redis.asyncio as aioredis
from langgraph.cache.redis import RedisCache

# Create async Redis client
async def main():
    client = aioredis.Redis(host="localhost", port=6379, db=0, decode_responses=False)
    
    # Create Redis cache with async client
    cache = RedisCache(client, prefix="my_app:")
    
    # Use with LangGraph
    from langgraph import StateGraph
    
    builder = StateGraph(...)
    graph = builder.compile(cache=cache)
    
    # Cache will automatically store and retrieve node execution results
    result = await graph.ainvoke({"input": "data"})
    
    await client.aclose()
```

## Features

- **Node-level caching**: Automatically caches the results of individual nodes in your LangGraph
- **TTL support**: Set time-to-live for cached values
- **Namespace isolation**: Different graphs/nodes are automatically isolated
- **Batch operations**: Efficient bulk get/set operations
- **Async support**: Works with both sync and async Redis clients
- **Error resilience**: Gracefully handles Redis connection issues