"""Unit tests for Redis cache async implementation."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.cache.base import FullKey
from langgraph.cache.redis import RedisCache


class TestRedisCacheAsync:
    """Test Redis cache async functionality with mocked Redis clients."""

    def test_is_async_redis_detection_sync(self) -> None:
        """Test async Redis client detection with sync client."""
        sync_redis = MagicMock()
        sync_redis.mget = MagicMock(return_value=[b"test:data"])
        
        cache = RedisCache(sync_redis, prefix="test:cache:")
        assert not cache._is_async_redis()

    def test_is_async_redis_detection_async(self) -> None:
        """Test async Redis client detection with async client."""
        async_redis = AsyncMock()
        async_redis.mget = AsyncMock(return_value=[b"test:data"])
        
        cache = RedisCache(async_redis, prefix="test:cache:")
        assert cache._is_async_redis()

    @pytest.mark.asyncio
    async def test_aget_with_async_redis(self) -> None:
        """Test aget with async Redis client."""
        async_redis = AsyncMock()
        # Mock mget to return serialized data
        test_data = b"json:{\"result\": 42}"
        async_redis.mget = AsyncMock(return_value=[test_data])
        
        cache = RedisCache(async_redis, prefix="test:cache:")
        keys: list[FullKey] = [(('graph', 'node'), 'key1')]
        
        result = await cache.aget(keys)
        
        # Verify async Redis client was used
        async_redis.mget.assert_called_once()
        assert len(result) == 1
        assert result[keys[0]] == {"result": 42}

    @pytest.mark.asyncio
    async def test_aget_with_sync_redis(self) -> None:
        """Test aget with sync Redis client fallback."""
        sync_redis = MagicMock()
        # Mock mget to return serialized data
        test_data = b"json:{\"result\": 42}"
        sync_redis.mget = MagicMock(return_value=[test_data])
        
        cache = RedisCache(sync_redis, prefix="test:cache:")
        keys: list[FullKey] = [(('graph', 'node'), 'key1')]
        
        result = await cache.aget(keys)
        
        # Verify sync Redis client was used
        sync_redis.mget.assert_called_once()
        assert len(result) == 1
        assert result[keys[0]] == {"result": 42}

    @pytest.mark.asyncio
    async def test_aset_with_async_redis(self) -> None:
        """Test aset with async Redis client."""
        async_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock()
        async_redis.pipeline = MagicMock(return_value=mock_pipeline)
        
        cache = RedisCache(async_redis, prefix="test:cache:")
        values = {(('graph', 'node'), 'key1'): ({"result": 42}, None)}
        
        await cache.aset(values)
        
        # Verify pipeline was created and executed asynchronously
        async_redis.pipeline.assert_called_once()
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_aset_with_sync_redis(self) -> None:
        """Test aset with sync Redis client fallback."""
        sync_redis = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.execute = MagicMock()
        sync_redis.pipeline = MagicMock(return_value=mock_pipeline)
        
        cache = RedisCache(sync_redis, prefix="test:cache:")
        values = {(('graph', 'node'), 'key1'): ({"result": 42}, None)}
        
        await cache.aset(values)
        
        # Verify pipeline was created and executed synchronously
        sync_redis.pipeline.assert_called_once()
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclear_with_async_redis(self) -> None:
        """Test aclear with async Redis client."""
        async_redis = AsyncMock()
        async_redis.keys = AsyncMock(return_value=["test:cache:key1", "test:cache:key2"])
        async_redis.delete = AsyncMock()
        
        cache = RedisCache(async_redis, prefix="test:cache:")
        
        await cache.aclear()
        
        # Verify async Redis client was used
        async_redis.keys.assert_called_once_with("test:cache:*")
        async_redis.delete.assert_called_once_with("test:cache:key1", "test:cache:key2")

    @pytest.mark.asyncio
    async def test_aclear_with_sync_redis(self) -> None:
        """Test aclear with sync Redis client fallback."""
        sync_redis = MagicMock()
        sync_redis.keys = MagicMock(return_value=["test:cache:key1", "test:cache:key2"])
        sync_redis.delete = MagicMock()
        
        cache = RedisCache(sync_redis, prefix="test:cache:")
        
        await cache.aclear()
        
        # Verify sync Redis client was used
        sync_redis.keys.assert_called_once_with("test:cache:*")
        sync_redis.delete.assert_called_once_with("test:cache:key1", "test:cache:key2")

    @pytest.mark.asyncio
    async def test_aclear_with_namespaces_async_redis(self) -> None:
        """Test aclear with namespaces using async Redis client."""
        async_redis = AsyncMock()
        async_redis.keys = AsyncMock(return_value=["test:cache:graph:node:key1"])
        async_redis.delete = AsyncMock()
        
        cache = RedisCache(async_redis, prefix="test:cache:")
        namespaces = [("graph", "node")]
        
        await cache.aclear(namespaces)
        
        # Verify correct pattern was used
        async_redis.keys.assert_called_once_with("test:cache:graph:node:*")
        async_redis.delete.assert_called_once_with("test:cache:graph:node:key1")

    @pytest.mark.asyncio
    async def test_async_error_handling(self) -> None:
        """Test error handling in async operations."""
        async_redis = AsyncMock()
        async_redis.mget = AsyncMock(side_effect=Exception("Redis unavailable"))
        
        cache = RedisCache(async_redis, prefix="test:cache:")
        keys: list[FullKey] = [(('graph', 'node'), 'key1')]
        
        # Should return empty dict on error
        result = await cache.aget(keys)
        assert result == {}

    @pytest.mark.asyncio
    async def test_empty_operations_async(self) -> None:
        """Test async operations with empty inputs."""
        async_redis = AsyncMock()
        cache = RedisCache(async_redis, prefix="test:cache:")
        
        # Empty get
        result = await cache.aget([])
        assert result == {}
        
        # Empty set
        await cache.aset({})
        
        # These should not call Redis at all
        async_redis.mget.assert_not_called()
        async_redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_async_operations(self) -> None:
        """Test batch operations with async Redis client."""
        async_redis = AsyncMock()
        
        # Mock batch get operation
        test_data = [
            b"json:{\"result\": 1}",
            b"json:{\"result\": 2}",
            b"json:{\"result\": 3}"
        ]
        async_redis.mget = AsyncMock(return_value=test_data)
        
        cache = RedisCache(async_redis, prefix="test:cache:")
        keys: list[FullKey] = [
            (('graph', 'node1'), 'key1'),
            (('graph', 'node2'), 'key2'),
            (('other', 'node'), 'key3')
        ]
        
        result = await cache.aget(keys)
        
        assert len(result) == 3
        assert result[keys[0]] == {"result": 1}
        assert result[keys[1]] == {"result": 2}
        assert result[keys[2]] == {"result": 3}
        
        # Verify correct Redis keys were requested
        expected_redis_keys = [
            "test:cache:graph:node1:key1",
            "test:cache:graph:node2:key2",
            "test:cache:other:node:key3"
        ]
        async_redis.mget.assert_called_once_with(expected_redis_keys)

    @pytest.mark.asyncio
    async def test_ttl_handling_async(self) -> None:
        """Test TTL handling in async set operations."""
        async_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_pipeline.setex = MagicMock()
        mock_pipeline.set = MagicMock()
        mock_pipeline.execute = AsyncMock()
        async_redis.pipeline = MagicMock(return_value=mock_pipeline)
        
        cache = RedisCache(async_redis, prefix="test:cache:")
        values = {
            (('graph', 'node1'), 'key1'): ({"result": 1}, None),    # No TTL
            (('graph', 'node2'), 'key2'): ({"result": 2}, 60),     # With TTL
        }
        
        await cache.aset(values)
        
        # Verify both setex (with TTL) and set (without TTL) were called
        assert mock_pipeline.setex.call_count == 1
        assert mock_pipeline.set.call_count == 1
        mock_pipeline.execute.assert_called_once()

    def test_process_raw_values_with_corrupted_data(self) -> None:
        """Test _process_raw_values handles corrupted data gracefully."""
        sync_redis = MagicMock()
        cache = RedisCache(sync_redis, prefix="test:cache:")
        
        keys: list[FullKey] = [
            (('graph', 'node1'), 'key1'),
            (('graph', 'node2'), 'key2'),
            (('graph', 'node3'), 'key3')
        ]
        
        raw_values = [
            b"json:{\"result\": 1}",    # Valid data
            b"invalid_format",          # Corrupted - no colon separator
            b"json:{\"result\": 3}"     # Valid data
        ]
        
        result = cache._process_raw_values(keys, raw_values)
        
        # Should only return valid entries, skipping corrupted one
        assert len(result) == 2
        assert result[keys[0]] == {"result": 1}
        assert result[keys[2]] == {"result": 3}
        # keys[1] should be skipped due to corruption
        assert keys[1] not in result

    @pytest.mark.asyncio
    async def test_integration_async_workflow(self) -> None:
        """Test complete async workflow: set, get, clear."""
        async_redis = AsyncMock()
        
        # Mock pipeline for set operation
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock()
        async_redis.pipeline = MagicMock(return_value=mock_pipeline)
        
        # Mock get operation
        test_data = b"json:{\"workflow\": \"complete\"}"
        async_redis.mget = AsyncMock(return_value=[test_data])
        
        # Mock clear operation
        async_redis.keys = AsyncMock(return_value=["test:cache:workflow:key"])
        async_redis.delete = AsyncMock()
        
        cache = RedisCache(async_redis, prefix="test:cache:")
        key: FullKey = (('workflow',), 'key')
        
        # Set data
        await cache.aset({key: ({"workflow": "complete"}, 60)})
        
        # Get data
        result = await cache.aget([key])
        assert result[key] == {"workflow": "complete"}
        
        # Clear data
        await cache.aclear()
        
        # Verify all operations were called correctly
        async_redis.pipeline.assert_called_once()
        mock_pipeline.execute.assert_called_once()
        async_redis.mget.assert_called_once()
        async_redis.keys.assert_called_once()
        async_redis.delete.assert_called_once()
