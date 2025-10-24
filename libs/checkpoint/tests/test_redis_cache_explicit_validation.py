"""Tests for explicit Redis client validation in RedisCache.

This module tests the improved RedisCache implementation that provides
explicit validation and error handling for sync/async Redis clients.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Any

from langgraph.cache.redis import RedisCache
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class TestRedisCacheValidation:
    """Test Redis client validation and initialization."""
    
    def test_sync_redis_client_detection(self):
        """Test that sync Redis clients are correctly detected."""
        # Mock sync Redis client
        sync_redis = Mock()
        sync_redis.mget = Mock(return_value=[None])
        sync_redis.pipeline = Mock()
        sync_redis.keys = Mock(return_value=[])
        sync_redis.delete = Mock()
        
        cache = RedisCache(sync_redis, serde=JsonPlusSerializer())
        assert not cache._is_async
    
    def test_async_redis_client_detection(self):
        """Test that async Redis clients are correctly detected."""
        # Mock async Redis client
        async_redis = Mock()
        async_redis.mget = AsyncMock(return_value=[None])
        async_redis.pipeline = Mock()
        async_redis.keys = AsyncMock(return_value=[])
        async_redis.delete = AsyncMock()
        
        cache = RedisCache(async_redis, serde=JsonPlusSerializer())
        assert cache._is_async
    
    def test_invalid_redis_client_missing_method(self):
        """Test that clients missing required methods are rejected."""
        invalid_client = Mock()
        # Missing mget method
        invalid_client.pipeline = Mock()
        invalid_client.keys = Mock()
        invalid_client.delete = Mock()
        
        with pytest.raises(ValueError, match="Missing required method 'mget'"):
            RedisCache(invalid_client)
    
    def test_invalid_redis_client_non_callable_mget(self):
        """Test that clients with non-callable mget are rejected."""
        invalid_client = Mock()
        invalid_client.mget = "not_callable"
        invalid_client.pipeline = Mock()
        invalid_client.keys = Mock()
        invalid_client.delete = Mock()
        
        with pytest.raises(ValueError, match="The 'mget' attribute is not callable"):
            RedisCache(invalid_client)
    
    def test_completely_invalid_client(self):
        """Test that completely invalid objects are rejected."""
        with pytest.raises(ValueError, match="Invalid Redis client"):
            RedisCache("not_a_redis_client")


class TestSyncRedisOperations:
    """Test sync Redis operations and error handling."""
    
    @pytest.fixture
    def sync_cache(self):
        """Create a RedisCache with mocked sync Redis client."""
        sync_redis = Mock()
        sync_redis.mget = Mock(return_value=[])
        sync_redis.pipeline = Mock()
        sync_redis.keys = Mock(return_value=[])
        sync_redis.delete = Mock()
        
        # Mock pipeline
        pipeline_mock = Mock()
        pipeline_mock.setex = Mock()
        pipeline_mock.set = Mock()
        pipeline_mock.execute = Mock()
        sync_redis.pipeline.return_value = pipeline_mock
        
        return RedisCache(sync_redis, serde=JsonPlusSerializer())
    
    def test_sync_get_works(self, sync_cache):
        """Test that sync get method works with sync client."""
        result = sync_cache.get([])
        assert result == {}
        
        # Test with actual keys
        sync_cache.redis.mget.return_value = [None]
        result = sync_cache.get([(('ns',), 'key')])
        assert result == {}
    
    def test_sync_set_works(self, sync_cache):
        """Test that sync set method works with sync client."""
        # Should not raise any exceptions
        sync_cache.set({})
        
        # Test with actual data
        mapping = {(('ns',), 'key'): ('value', 60)}
        sync_cache.set(mapping)
        
        # Verify pipeline methods were called
        sync_cache.redis.pipeline.assert_called()
    
    def test_sync_clear_works(self, sync_cache):
        """Test that sync clear method works with sync client."""
        # Should not raise any exceptions
        sync_cache.clear()
        
        # Test with namespaces
        sync_cache.clear([(('ns',),)])
    
    def test_async_methods_fail_with_sync_client(self, sync_cache):
        """Test that async methods raise RuntimeError with sync client."""
        with pytest.raises(RuntimeError, match="Cannot use async method 'aget' with synchronous Redis client"):
            sync_cache.aget([])
        
        with pytest.raises(RuntimeError, match="Cannot use async method 'aset' with synchronous Redis client"):
            sync_cache.aset({})
        
        with pytest.raises(RuntimeError, match="Cannot use async method 'aclear' with synchronous Redis client"):
            sync_cache.aclear()


class TestAsyncRedisOperations:
    """Test async Redis operations and error handling."""
    
    @pytest.fixture
    def async_cache(self):
        """Create a RedisCache with mocked async Redis client."""
        async_redis = Mock()
        async_redis.mget = AsyncMock(return_value=[])
        async_redis.pipeline = Mock()
        async_redis.keys = AsyncMock(return_value=[])
        async_redis.delete = AsyncMock()
        
        # Mock async pipeline
        pipeline_mock = Mock()
        pipeline_mock.setex = Mock()
        pipeline_mock.set = Mock()
        pipeline_mock.execute = AsyncMock()
        async_redis.pipeline.return_value = pipeline_mock
        
        return RedisCache(async_redis, serde=JsonPlusSerializer())
    
    async def test_async_aget_works(self, async_cache):
        """Test that async aget method works with async client."""
        result = await async_cache.aget([])
        assert result == {}
        
        # Test with actual keys
        async_cache.redis.mget.return_value = [None]
        result = await async_cache.aget([(('ns',), 'key')])
        assert result == {}
        
        # Verify async method was awaited
        async_cache.redis.mget.assert_awaited()
    
    async def test_async_aset_works(self, async_cache):
        """Test that async aset method works with async client."""
        # Should not raise any exceptions
        await async_cache.aset({})
        
        # Test with actual data
        mapping = {(('ns',), 'key'): ('value', 60)}
        await async_cache.aset(mapping)
        
        # Verify pipeline methods were called
        async_cache.redis.pipeline.assert_called()
    
    async def test_async_aclear_works(self, async_cache):
        """Test that async aclear method works with async client."""
        # Should not raise any exceptions
        await async_cache.aclear()
        
        # Test with namespaces
        await async_cache.aclear([(('ns',),)])
        
        # Verify async methods were awaited
        async_cache.redis.keys.assert_awaited()
    
    def test_sync_methods_fail_with_async_client(self, async_cache):
        """Test that sync methods raise RuntimeError with async client."""
        with pytest.raises(RuntimeError, match="Cannot use sync method 'get' with asynchronous Redis client"):
            async_cache.get([])
        
        with pytest.raises(RuntimeError, match="Cannot use sync method 'set' with asynchronous Redis client"):
            async_cache.set({})
        
        with pytest.raises(RuntimeError, match="Cannot use sync method 'clear' with asynchronous Redis client"):
            async_cache.clear()


class TestIssue6247Reproduction:
    """Test cases that reproduce and verify the fix for issue #6247."""
    
    async def test_original_issue_reproduction_fixed(self):
        """Test that the original issue from #6247 is now fixed."""
        # Mock async Redis client similar to the original issue
        async_redis = Mock()
        
        # The key fix: mget now properly returns a coroutine that gets awaited
        async_redis.mget = AsyncMock(return_value=[None])  # This would cause the original error
        async_redis.pipeline = Mock()
        async_redis.keys = AsyncMock(return_value=[])
        async_redis.delete = AsyncMock()
        
        # This should not raise TypeError anymore
        cache = RedisCache(async_redis, serde=JsonPlusSerializer())
        
        # The original failing operation should now work
        result = await cache.aget([(('ns',), 'key')])
        assert isinstance(result, dict)
        
        # Verify the async mget was properly awaited
        async_redis.mget.assert_awaited_once()
    
    def test_sync_client_still_works_as_before(self):
        """Test that sync clients continue to work exactly as before."""
        # Mock sync Redis client
        sync_redis = Mock()
        sync_redis.mget = Mock(return_value=[None])  # Returns directly, not a coroutine
        sync_redis.pipeline = Mock()
        sync_redis.keys = Mock(return_value=[])
        sync_redis.delete = Mock()
        
        # Should work exactly as before
        cache = RedisCache(sync_redis, serde=JsonPlusSerializer())
        result = cache.get([(('ns',), 'key')])
        assert isinstance(result, dict)
        
        # Verify sync method was called normally
        sync_redis.mget.assert_called_once()
    
    def test_error_when_mixing_client_types(self):
        """Test that clear errors are raised when mixing sync/async incorrectly."""
        # Test async client with sync method
        async_redis = Mock()
        async_redis.mget = AsyncMock(return_value=[])
        async_redis.pipeline = Mock()
        async_redis.keys = AsyncMock(return_value=[])
        async_redis.delete = AsyncMock()
        
        async_cache = RedisCache(async_redis, serde=JsonPlusSerializer())
        
        with pytest.raises(RuntimeError) as exc_info:
            async_cache.get([])
        
        assert "Cannot use sync method 'get' with asynchronous Redis client" in str(exc_info.value)
        assert "Use 'aget' method instead" in str(exc_info.value)


class TestRedisCacheDataFlow:
    """Test actual data flow and serialization."""
    
    @pytest.fixture
    def mock_sync_redis_with_data(self):
        """Create a mock sync Redis client that returns serialized data."""
        redis_mock = Mock()
        
        # Mock serialized data as it would appear in Redis
        serialized_data = b"json:value"
        redis_mock.mget = Mock(return_value=[serialized_data])
        
        redis_mock.pipeline = Mock()
        redis_mock.keys = Mock(return_value=[])
        redis_mock.delete = Mock()
        
        # Mock pipeline
        pipeline_mock = Mock()
        pipeline_mock.setex = Mock()
        pipeline_mock.set = Mock()
        pipeline_mock.execute = Mock()
        redis_mock.pipeline.return_value = pipeline_mock
        
        return redis_mock
    
    def test_data_serialization_deserialization(self, mock_sync_redis_with_data):
        """Test that data is properly serialized and deserialized."""
        cache = RedisCache(mock_sync_redis_with_data, serde=JsonPlusSerializer())
        
        # Test getting data
        result = cache.get([(('test',), 'key')])
        
        # Should have deserialized the value
        assert len(result) == 1
        assert (('test',), 'key') in result
        assert result[(('test',), 'key')] == "value"
        
        # Test setting data
        cache.set({(('test',), 'new_key'): ('new_value', 300)})
        
        # Verify pipeline was used for setting
        mock_sync_redis_with_data.pipeline.assert_called()
        pipeline = mock_sync_redis_with_data.pipeline.return_value
        pipeline.execute.assert_called_once()
