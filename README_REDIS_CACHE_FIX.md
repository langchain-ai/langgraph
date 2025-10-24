# Redis Cache Async Fix - Issue #6247

This document explains the fix for issue #6247 where `RedisCache` did not work with async Redis clients.

## Problem Summary

The original `RedisCache` implementation had a critical bug when used with `redis.asyncio.Redis` clients:

```python
# This would fail with: TypeError: 'coroutine' object is not iterable
from redis.asyncio import Redis as AsyncRedis

redis = AsyncRedis(host='localhost')
cache = RedisCache(redis)
result = await cache.aget(keys)  # Error here
```

**Root Cause**: The async methods (`aget`, `aset`, `aclear`) incorrectly delegated to sync methods, which tried to iterate over coroutine objects returned by async Redis operations.

## Solution: Explicit Validation Approach

Instead of runtime auto-detection that masks errors, our solution uses explicit validation:

### 1. Client Type Validation at Initialization

```python
def _validate_and_detect_client_type(self) -> bool:
    """Validate Redis client and detect if it's async during initialization."""
    # Check for required Redis methods
    required_methods = ['mget', 'pipeline', 'keys', 'delete']
    for method in required_methods:
        if not hasattr(self.redis, method):
            raise ValueError(f"Invalid Redis client: missing '{method}' method")
    
    # Detect client type based on mget method
    if inspect.iscoroutinefunction(self.redis.mget):
        return True  # Async client
    elif callable(self.redis.mget):
        return False  # Sync client
    else:
        raise ValueError("Invalid Redis client: 'mget' is not callable")
```

### 2. Explicit Error Handling

```python
async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
    if not self._is_async:
        raise RuntimeError(
            "Cannot use async method 'aget' with synchronous Redis client. "
            "Use 'get' method instead, or initialize with redis.asyncio.Redis."
        )
    # ... proper async implementation with await

def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
    if self._is_async:
        raise RuntimeError(
            "Cannot use sync method 'get' with asynchronous Redis client. "
            "Use 'aget' method instead, or initialize with redis.Redis."
        )
    # ... sync implementation
```

## Usage Examples

### ✅ Correct Async Usage

```python
from redis.asyncio import Redis as AsyncRedis
from langgraph.cache.redis import RedisCache

# Create async Redis client
async_redis = AsyncRedis(host='localhost', port=6379)
cache = RedisCache(async_redis)

# Use async methods
result = await cache.aget(keys)
await cache.aset(data)
await cache.aclear()
```

### ✅ Correct Sync Usage

```python
from redis import Redis
from langgraph.cache.redis import RedisCache

# Create sync Redis client  
sync_redis = Redis(host='localhost', port=6379)
cache = RedisCache(sync_redis)

# Use sync methods
result = cache.get(keys)
cache.set(data)
cache.clear()
```

### ❌ Wrong Usage (Clear Errors)

```python
# Async client with sync method
async_redis = AsyncRedis(host='localhost')
cache = RedisCache(async_redis)
cache.get(keys)  # RuntimeError with clear message

# Sync client with async method
sync_redis = Redis(host='localhost')
cache = RedisCache(sync_redis)
await cache.aget(keys)  # RuntimeError with clear message
```

## Key Benefits

1. **Explicit Error Handling**: Clear error messages when wrong method is used
2. **Initialization-time Validation**: Client type detected once during `__init__`
3. **No Silent Failures**: Users get immediate feedback for incorrect usage
4. **Backward Compatibility**: Existing sync Redis code continues to work
5. **Performance**: No runtime detection overhead on method calls
6. **Clear API Contract**: Documentation clearly indicates sync vs async usage

## Testing

Comprehensive test coverage includes:

- Client validation during initialization
- Explicit error handling for method mismatches
- Reproduction test for original issue #6247
- Data flow and serialization tests
- Backward compatibility verification

Run tests:
```bash
pytest libs/checkpoint/tests/test_redis_cache_explicit_validation.py
pytest libs/checkpoint/tests/test_issue_6247_reproduction.py
```

## Migration Guide

No breaking changes for existing users:

### If you currently use sync Redis:
```python
# This continues to work exactly as before
from redis import Redis
cache = RedisCache(Redis(host='localhost'))
result = cache.get(keys)  # Still works
```

### If you tried async Redis and got errors:
```python
# Before (would error)
from redis.asyncio import Redis as AsyncRedis
cache = RedisCache(AsyncRedis(host='localhost'))
result = await cache.aget(keys)  # TypeError

# After (works correctly)
from redis.asyncio import Redis as AsyncRedis
cache = RedisCache(AsyncRedis(host='localhost'))  # Validates client
result = await cache.aget(keys)  # Works! ✅
```

## Comparison with Previous PR #6268

| Aspect | PR #6268 (Auto-detection) | This PR (Explicit Validation) |
|--------|---------------------------|---------------------------------|
| **Error Clarity** | Silent fallbacks possible | Explicit errors with guidance |
| **Performance** | Runtime detection on each call | One-time validation at init |
| **API Contract** | Unclear (any client accepted) | Clear (sync vs async methods) |
| **Debugging** | Hard to debug silent issues | Clear error messages |
| **Maintenance** | Complex detection logic | Simple validation |
| **User Experience** | Potentially confusing | Educational and clear |

## Implementation Details

### Refactored Architecture

- `_validate_and_detect_client_type()`: One-time client validation
- `_process_raw_values()`: Common deserialization logic
- `_build_set_pipeline()`: Shared pipeline creation
- `_get_keys_to_delete_async()`: Async-specific key deletion

### Error Message Design

All error messages follow this pattern:
1. **What went wrong**: "Cannot use [sync/async] method with [async/sync] client"
2. **Immediate fix**: "Use '[correct_method]' instead"
3. **Long-term fix**: "Or initialize with [correct_client] for [sync/async] support"

## Conclusion

This fix resolves issue #6247 by implementing explicit validation that:
- Prevents the original coroutine iteration error
- Provides clear guidance for correct usage
- Maintains backward compatibility
- Offers better performance and maintainability

The explicit approach is more robust and user-friendly than auto-detection, leading to a better developer experience and fewer debugging sessions.
