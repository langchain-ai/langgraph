# Use Pydantic models as state with caching

This guide shows how to use Pydantic models as state objects with LangGraph's caching system for improved performance.

## Overview

LangGraph's caching system works with both regular Python dictionaries and Pydantic models as state objects. When using caching, ensure your state objects are deterministic - identical data should always produce identical cache keys.

> **Note**: LangGraph automatically handles deterministic serialization for Pydantic models, so you can use them directly without additional configuration.

## Usage

Simply use Pydantic models as your state schema with caching enabled:

```python
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
import time

class MyState(BaseModel):
    messages: list[str]
    counter: int = 0

def expensive_node(state: MyState) -> dict:
    # Expensive operation that benefits from caching
    print(f"Executing expensive_node with counter={state.counter}")
    time.sleep(1)  # Simulate expensive work
    return {"counter": state.counter + 1}

# Create graph with caching
builder = StateGraph(MyState)
builder.add_node(
    "expensive_node", 
    expensive_node,
    cache_policy=CachePolicy(ttl=300)  # 5 minute TTL
)
builder.add_edge(START, "expensive_node")
builder.add_edge("expensive_node", END)

graph = builder.compile(cache=InMemoryCache())

# Test with identical inputs
state = MyState(messages=["hello"], counter=0)

print("First invocation (should cache miss):")
start_time = time.time()
result1 = graph.invoke(state)
first_duration = time.time() - start_time
print(f"Result 1: {result1}")
print(f"Duration: {first_duration:.2f}s")

print("\nSecond invocation with identical state (should cache hit):")
start_time = time.time()
result2 = graph.invoke(state)
second_duration = time.time() - start_time
print(f"Result 2: {result2}")
print(f"Duration: {second_duration:.2f}s")

# Verify caching worked
if second_duration < first_duration * 0.5:  # Should be much faster
    print("✅ Caching worked! Second call was significantly faster.")
else:
    print("❌ Caching may not be working. Second call wasn't much faster.")

# Test with different state (should cache miss)
different_state = MyState(messages=["hello"], counter=1)  # Different counter
print(f"\nThird invocation with different state (should cache miss):")
start_time = time.time()
result3 = graph.invoke(different_state)
third_duration = time.time() - start_time
print(f"Result 3: {result3}")
print(f"Duration: {third_duration:.2f}s")
```

### Example Output

When you run the above code, you will see caching in action:

```
Testing Pydantic caching example...
First invocation (should cache miss):
Executing expensive_node with counter=0
Result 1: {'messages': ['hello'], 'counter': 1}
Duration: 1.01s

Second invocation with identical state (should cache hit):
Result 2: {'messages': ['hello'], 'counter': 1}
Duration: 0.00s
✅ Caching worked! Second call was significantly faster.

Third invocation with different state (should cache miss):
Executing expensive_node with counter=1
Result 3: {'messages': ['hello'], 'counter': 2}
Duration: 1.00s
```

Notice how:
- **First call**: Takes ~1 second and executes the function
- **Second call**: Returns instantly (0.00s) without executing the function - cache hit!
- **Third call**: Takes ~1 second again because the state is different - new cache entry

## Benefits

- **Consistent cache behavior**: Identical Pydantic models always produce cache hits
- **Better performance**: Avoids redundant expensive operations  
- **Type safety**: Keep Pydantic's validation and type checking benefits
- **Backward compatibility**: Existing code works without changes

## Best Practices

1. **Use immutable data**: Prefer immutable field values for predictable caching
2. **Consistent instantiation**: Create models with identical data for cache hits
3. **Monitor cache performance**: Use cache hit rates to validate effectiveness

## Related

- [Node caching documentation](graph-api.md#add-node-caching)
- [Pydantic state models guide](../concepts/low_level.md#state)
- [Cache types reference](../reference/cache.md)
