"""Unit tests for Redis cache implementation."""

import time

import pytest
import redis

from langgraph.cache.base import FullKey
from langgraph.cache.redis import RedisCache


class TestRedisCache:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test Redis client and cache."""
        self.client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=False
        )
        try:
            self.client.ping()
        except redis.ConnectionError:
            pytest.skip("Redis server not available")

        self.cache: RedisCache = RedisCache(self.client, prefix="test:cache:")

        # Clean up before each test
        self.client.flushdb()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        try:
            self.client.flushdb()
        except Exception:
            pass

    def test_basic_set_and_get(self) -> None:
        """Test basic set and get operations."""
        keys: list[FullKey] = [(("graph", "node"), "key1")]
        values = {keys[0]: ({"result": 42}, None)}

        # Set value
        self.cache.set(values)

        # Get value
        result = self.cache.get(keys)
        assert len(result) == 1
        assert result[keys[0]] == {"result": 42}

    def test_batch_operations(self) -> None:
        """Test batch set and get operations."""
        keys: list[FullKey] = [
            (("graph", "node1"), "key1"),
            (("graph", "node2"), "key2"),
            (("other", "node"), "key3"),
        ]
        values = {
            keys[0]: ({"result": 1}, None),
            keys[1]: ({"result": 2}, 60),  # With TTL
            keys[2]: ({"result": 3}, None),
        }

        # Set values
        self.cache.set(values)

        # Get all values
        result = self.cache.get(keys)
        assert len(result) == 3
        assert result[keys[0]] == {"result": 1}
        assert result[keys[1]] == {"result": 2}
        assert result[keys[2]] == {"result": 3}

    def test_ttl_behavior(self) -> None:
        """Test TTL (time-to-live) functionality."""
        key: FullKey = (("graph", "node"), "ttl_key")
        values = {key: ({"data": "expires_soon"}, 1)}  # 1 second TTL

        # Set with TTL
        self.cache.set(values)

        # Should be available immediately
        result = self.cache.get([key])
        assert len(result) == 1
        assert result[key] == {"data": "expires_soon"}

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        result = self.cache.get([key])
        assert len(result) == 0

    def test_namespace_isolation(self) -> None:
        """Test that different namespaces are isolated."""
        key1: FullKey = (("graph1", "node"), "same_key")
        key2: FullKey = (("graph2", "node"), "same_key")

        values = {key1: ({"graph": 1}, None), key2: ({"graph": 2}, None)}

        self.cache.set(values)

        result = self.cache.get([key1, key2])
        assert result[key1] == {"graph": 1}
        assert result[key2] == {"graph": 2}

    def test_clear_all(self) -> None:
        """Test clearing all cached values."""
        keys: list[FullKey] = [
            (("graph", "node1"), "key1"),
            (("graph", "node2"), "key2"),
        ]
        values = {keys[0]: ({"result": 1}, None), keys[1]: ({"result": 2}, None)}

        self.cache.set(values)

        # Verify data exists
        result = self.cache.get(keys)
        assert len(result) == 2

        # Clear all
        self.cache.clear()

        # Verify data is gone
        result = self.cache.get(keys)
        assert len(result) == 0

    def test_clear_by_namespace(self) -> None:
        """Test clearing cached values by namespace."""
        keys: list[FullKey] = [
            (("graph1", "node"), "key1"),
            (("graph2", "node"), "key2"),
            (("graph1", "other"), "key3"),
        ]
        values = {
            keys[0]: ({"result": 1}, None),
            keys[1]: ({"result": 2}, None),
            keys[2]: ({"result": 3}, None),
        }

        self.cache.set(values)

        # Clear only graph1 namespace
        self.cache.clear([("graph1", "node"), ("graph1", "other")])

        # graph1 should be cleared, graph2 should remain
        result = self.cache.get(keys)
        assert len(result) == 1
        assert result[keys[1]] == {"result": 2}

    def test_empty_operations(self) -> None:
        """Test behavior with empty keys/values."""
        # Empty get
        result = self.cache.get([])
        assert result == {}

        # Empty set
        self.cache.set({})  # Should not raise error

    def test_nonexistent_keys(self) -> None:
        """Test getting keys that don't exist."""
        keys: list[FullKey] = [(("graph", "node"), "nonexistent")]
        result = self.cache.get(keys)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_async_operations(self) -> None:
        """Test async set and get operations with sync Redis client."""
        # Create sync Redis client and cache (like main integration tests)
        client = redis.Redis(host="localhost", port=6379, db=1, decode_responses=False)
        try:
            client.ping()
        except Exception:
            pytest.skip("Redis not available")

        cache: RedisCache = RedisCache(client, prefix="test:async:")

        keys: list[FullKey] = [(("graph", "node"), "async_key")]
        values = {keys[0]: ({"async": True}, None)}

        # Async set (delegates to sync)
        await cache.aset(values)

        # Async get (delegates to sync)
        result = await cache.aget(keys)
        assert len(result) == 1
        assert result[keys[0]] == {"async": True}

        # Cleanup
        client.flushdb()

    @pytest.mark.asyncio
    async def test_async_clear(self) -> None:
        """Test async clear operations with sync Redis client."""
        # Create sync Redis client and cache (like main integration tests)
        client = redis.Redis(host="localhost", port=6379, db=1, decode_responses=False)
        try:
            client.ping()
        except Exception:
            pytest.skip("Redis not available")

        cache: RedisCache = RedisCache(client, prefix="test:async:")

        keys: list[FullKey] = [(("graph", "node"), "key")]
        values = {keys[0]: ({"data": "test"}, None)}

        await cache.aset(values)

        # Verify data exists
        result = await cache.aget(keys)
        assert len(result) == 1

        # Clear all (delegates to sync)
        await cache.aclear()

        # Verify data is gone
        result = await cache.aget(keys)
        assert len(result) == 0

        # Cleanup
        client.flushdb()

    def test_redis_unavailable_get(self) -> None:
        """Test behavior when Redis is unavailable during get operations."""
        # Create cache with non-existent Redis server
        bad_client = redis.Redis(
            host="nonexistent", port=9999, socket_connect_timeout=0.1
        )
        cache: RedisCache = RedisCache(bad_client, prefix="test:cache:")

        keys: list[FullKey] = [(("graph", "node"), "key")]
        result = cache.get(keys)

        # Should return empty dict when Redis unavailable
        assert result == {}

    def test_redis_unavailable_set(self) -> None:
        """Test behavior when Redis is unavailable during set operations."""
        # Create cache with non-existent Redis server
        bad_client = redis.Redis(
            host="nonexistent", port=9999, socket_connect_timeout=0.1
        )
        cache: RedisCache = RedisCache(bad_client, prefix="test:cache:")

        keys: list[FullKey] = [(("graph", "node"), "key")]
        values = {keys[0]: ({"data": "test"}, None)}

        # Should not raise exception when Redis unavailable
        cache.set(values)  # Should silently fail

    @pytest.mark.asyncio
    async def test_redis_unavailable_async(self) -> None:
        """Test async behavior when Redis is unavailable."""
        # Create sync cache with non-existent Redis server (like main integration tests)
        bad_client = redis.Redis(
            host="nonexistent", port=9999, socket_connect_timeout=0.1
        )
        cache: RedisCache = RedisCache(bad_client, prefix="test:cache:")

        keys: list[FullKey] = [(("graph", "node"), "key")]
        values = {keys[0]: ({"data": "test"}, None)}

        # Should return empty dict for get (delegates to sync)
        result = await cache.aget(keys)
        assert result == {}

        # Should not raise exception for set (delegates to sync)
        await cache.aset(values)  # Should silently fail

    def test_corrupted_data_handling(self) -> None:
        """Test handling of corrupted data in Redis."""
        # Set some valid data first
        keys: list[FullKey] = [(("graph", "node"), "valid_key")]
        values = {keys[0]: ({"data": "valid"}, None)}
        self.cache.set(values)

        # Manually insert corrupted data
        corrupted_key = self.cache._make_key(("graph", "node"), "corrupted_key")
        self.client.set(corrupted_key, b"invalid:data:format:too:many:colons")

        # Should skip corrupted entry and return only valid ones
        all_keys: list[FullKey] = [keys[0], (("graph", "node"), "corrupted_key")]
        result = self.cache.get(all_keys)

        assert len(result) == 1
        assert result[keys[0]] == {"data": "valid"}

    def test_key_parsing_edge_cases(self) -> None:
        """Test key parsing with edge cases."""
        # Test empty namespace
        key1: FullKey = ((), "empty_ns")
        values = {key1: ({"data": "empty_ns"}, None)}
        self.cache.set(values)
        result = self.cache.get([key1])
        assert result[key1] == {"data": "empty_ns"}

        # Test namespace with special characters
        key2: FullKey = (
            ("graph:with:colons", "node-with-dashes"),
            "key_with_underscores",
        )
        values = {key2: ({"data": "special_chars"}, None)}
        self.cache.set(values)
        result = self.cache.get([key2])
        assert result[key2] == {"data": "special_chars"}

    def test_large_data_serialization(self) -> None:
        """Test handling of large data objects."""
        # Create a large data structure
        large_data = {"large_list": list(range(1000)), "nested": {"data": "x" * 1000}}
        key: FullKey = (("graph", "node"), "large_key")
        values = {key: (large_data, None)}

        self.cache.set(values)
        result = self.cache.get([key])

        assert len(result) == 1
        assert result[key] == large_data
