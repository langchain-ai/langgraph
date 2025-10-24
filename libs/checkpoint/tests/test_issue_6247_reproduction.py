"""Reproduction test for issue #6247.

This script reproduces the exact error scenario from issue #6247
and verifies that our explicit validation fix resolves the problem.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy
from langgraph.cache.redis import RedisCache
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class State(TypedDict):
    test: bool


class TestIssue6247Reproduction:
    """Test class to reproduce and verify the fix for issue #6247."""
    
    def create_mock_async_redis(self):
        """Create a mock async Redis client that mimics redis.asyncio.Redis behavior."""
        mock_redis = Mock()
        
        # The key issue: async methods return coroutines
        mock_redis.mget = AsyncMock(return_value=[])
        mock_redis.pipeline = Mock()
        mock_redis.keys = AsyncMock(return_value=[])
        mock_redis.delete = AsyncMock()
        
        # Mock pipeline for async operations
        pipeline_mock = Mock()
        pipeline_mock.setex = Mock()
        pipeline_mock.set = Mock()
        pipeline_mock.execute = AsyncMock()
        mock_redis.pipeline.return_value = pipeline_mock
        
        return mock_redis
    
    def create_mock_sync_redis(self):
        """Create a mock sync Redis client that mimics redis.Redis behavior."""
        mock_redis = Mock()
        
        # Sync methods return values directly
        mock_redis.mget = Mock(return_value=[])
        mock_redis.pipeline = Mock()
        mock_redis.keys = Mock(return_value=[])
        mock_redis.delete = Mock()
        
        # Mock pipeline for sync operations
        pipeline_mock = Mock()
        pipeline_mock.setex = Mock()
        pipeline_mock.set = Mock()
        pipeline_mock.execute = Mock()
        mock_redis.pipeline.return_value = pipeline_mock
        
        return mock_redis
    
    async def test_original_issue_fixed_with_async_redis(self):
        """Test that the original issue #6247 is now fixed with async Redis client."""
        # Create async Redis mock (this would cause the original error)
        redis = self.create_mock_async_redis()
        
        # Create RedisCache - this should NOT raise any errors now
        redis_cache = RedisCache(
            redis=redis,
            prefix="foo",
            serde=JsonPlusSerializer()
        )
        
        # Verify it detected async client correctly
        assert redis_cache._is_async is True
        
        # Create the graph as in the original issue
        graph_builder = StateGraph(State)
        
        graph_builder.add_node(
            "foo",
            lambda state: state,
            cache_policy=CachePolicy(ttl=86400),
        )
        
        graph_builder.add_edge(START, "foo")
        graph_builder.add_edge("foo", END)
        
        graph = graph_builder.compile(cache=redis_cache)
        
        # This should NOT raise "TypeError: 'coroutine' object is not iterable" anymore
        result = await graph.ainvoke({"test": True})
        
        # Verify the result
        assert result == {"test": True}
        
        # Verify async methods were properly awaited
        redis.mget.assert_awaited()
    
    async def test_sync_redis_continues_to_work(self):
        """Test that sync Redis clients continue to work as before."""
        # Create sync Redis mock
        redis = self.create_mock_sync_redis()
        
        # Create RedisCache
        redis_cache = RedisCache(
            redis=redis,
            prefix="foo",
            serde=JsonPlusSerializer()
        )
        
        # Verify it detected sync client correctly
        assert redis_cache._is_async is False
        
        # Create the graph
        graph_builder = StateGraph(State)
        
        graph_builder.add_node(
            "foo",
            lambda state: state,
            cache_policy=CachePolicy(ttl=86400),
        )
        
        graph_builder.add_edge(START, "foo")
        graph_builder.add_edge("foo", END)
        
        graph = graph_builder.compile(cache=redis_cache)
        
        # This should work as before
        result = await graph.ainvoke({"test": True})
        
        # Verify the result
        assert result == {"test": True}
        
        # Verify sync methods were called normally
        redis.mget.assert_called()
    
    def test_clear_error_when_mixing_sync_async(self):
        """Test that clear errors are provided when mixing sync/async incorrectly."""
        # Test async Redis with sync cache method
        async_redis = self.create_mock_async_redis()
        cache = RedisCache(async_redis, serde=JsonPlusSerializer())
        
        # This should raise a clear RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            cache.get([(('ns',), 'key')])
        
        error_msg = str(exc_info.value)
        assert "Cannot use sync method 'get' with asynchronous Redis client" in error_msg
        assert "Use 'aget' method instead" in error_msg
        assert "redis.asyncio.Redis for async support" in error_msg
    
    async def test_clear_error_when_mixing_async_sync(self):
        """Test that clear errors are provided when mixing async/sync incorrectly."""
        # Test sync Redis with async cache method
        sync_redis = self.create_mock_sync_redis()
        cache = RedisCache(sync_redis, serde=JsonPlusSerializer())
        
        # This should raise a clear RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await cache.aget([(('ns',), 'key')])
        
        error_msg = str(exc_info.value)
        assert "Cannot use async method 'aget' with synchronous Redis client" in error_msg
        assert "Use 'get' method instead" in error_msg
        assert "redis.asyncio.Redis for async support" in error_msg
    
    def test_invalid_redis_client_rejected(self):
        """Test that invalid Redis clients are rejected during initialization."""
        # Test completely invalid client
        with pytest.raises(ValueError) as exc_info:
            RedisCache("not_a_redis_client")
        
        error_msg = str(exc_info.value)
        assert "Invalid Redis client" in error_msg
        assert "Missing required method" in error_msg
    
    async def test_comprehensive_async_workflow(self):
        """Test a comprehensive async workflow to ensure everything works together."""
        redis = self.create_mock_async_redis()
        cache = RedisCache(redis, prefix="test:", serde=JsonPlusSerializer())
        
        # Test aget
        result = await cache.aget([(('ns',), 'key')])
        assert isinstance(result, dict)
        
        # Test aset
        await cache.aset({(('ns',), 'key'): ('value', 300)})
        
        # Test aclear
        await cache.aclear([(('ns',)])
        
        # Verify all async methods were awaited
        redis.mget.assert_awaited()
        redis.keys.assert_awaited()
        redis.delete.assert_awaited()
    
    def test_comprehensive_sync_workflow(self):
        """Test a comprehensive sync workflow to ensure backward compatibility."""
        redis = self.create_mock_sync_redis()
        cache = RedisCache(redis, prefix="test:", serde=JsonPlusSerializer())
        
        # Test get
        result = cache.get([(('ns',), 'key')])
        assert isinstance(result, dict)
        
        # Test set
        cache.set({(('ns',), 'key'): ('value', 300)})
        
        # Test clear
        cache.clear([(('ns',)])
        
        # Verify all sync methods were called
        redis.mget.assert_called()
        redis.keys.assert_called()
        redis.delete.assert_called()


if __name__ == "__main__":
    # Run the reproduction test as a script
    async def main():
        test_instance = TestIssue6247Reproduction()
        
        print("Testing original issue #6247 reproduction with async Redis...")
        await test_instance.test_original_issue_fixed_with_async_redis()
        print("✓ Async Redis client works correctly now")
        
        print("Testing backward compatibility with sync Redis...")
        await test_instance.test_sync_redis_continues_to_work()
        print("✓ Sync Redis client continues to work as before")
        
        print("Testing error handling for mixed sync/async usage...")
        test_instance.test_clear_error_when_mixing_sync_async()
        await test_instance.test_clear_error_when_mixing_async_sync()
        print("✓ Clear error messages provided for incorrect usage")
        
        print("Testing invalid client rejection...")
        test_instance.test_invalid_redis_client_rejected()
        print("✓ Invalid Redis clients properly rejected")
        
        print("\nAll tests passed! Issue #6247 has been resolved.")
    
    asyncio.run(main())
