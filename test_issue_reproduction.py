"""Test to reproduce and verify fix for issue #6247."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from langgraph.cache.redis import RedisCache
from langgraph.cache.base import FullKey


async def test_original_issue_reproduction():
    """Reproduce the original issue from #6247."""
    print("Testing original issue reproduction...")
    
    # Create an async Redis mock that behaves like redis.asyncio.Redis
    async_redis = AsyncMock()
    async_redis.mget = AsyncMock(return_value=[b"json:{\"test\": true}"])
    
    # This would fail with the old implementation
    cache = RedisCache(async_redis, prefix="test:")
    keys: list[FullKey] = [(('graph', 'node'), 'key1')]
    
    try:
        result = await cache.aget(keys)
        print(f"SUCCESS: aget() returned {result}")
        assert len(result) == 1
        assert result[keys[0]] == {"test": True}
        print("Async Redis client is working correctly!")
    except TypeError as e:
        print(f"FAILED: {e}")
        raise


async def test_sync_redis_still_works():
    """Verify sync Redis client still works (backward compatibility)."""
    print("\nTesting sync Redis compatibility...")
    
    sync_redis = MagicMock()
    sync_redis.mget = MagicMock(return_value=[b"json:{\"test\": true}"])
    
    cache = RedisCache(sync_redis, prefix="test:")
    keys: list[FullKey] = [(('graph', 'node'), 'key1')]
    
    result = await cache.aget(keys)
    print(f"SUCCESS: aget() with sync Redis returned {result}")
    assert len(result) == 1
    assert result[keys[0]] == {"test": True}
    print("Sync Redis client compatibility maintained!")


async def test_async_detection():
    """Test the async Redis detection mechanism."""
    print("\nTesting async detection mechanism...")
    
    # Test async Redis detection
    async_redis = AsyncMock()
    async_redis.mget = AsyncMock(return_value=[])
    cache_async = RedisCache(async_redis, prefix="test:")
    print(f"Async Redis detected: {cache_async._is_async_redis()}")
    assert cache_async._is_async_redis() == True
    
    # Test sync Redis detection
    sync_redis = MagicMock()
    sync_redis.mget = MagicMock(return_value=[])
    cache_sync = RedisCache(sync_redis, prefix="test:")
    print(f"Sync Redis detected: {cache_sync._is_async_redis()}")
    assert cache_sync._is_async_redis() == False
    
    print("Async detection working correctly!")


async def main():
    """Run all tests."""
    print("Running issue #6247 reproduction and fix verification...\n")
    
    await test_original_issue_reproduction()
    await test_sync_redis_still_works()
    await test_async_detection()
    
    print("\nAll tests passed! Issue #6247 has been successfully fixed.")


if __name__ == "__main__":
    asyncio.run(main())
