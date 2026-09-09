"""Unit tests for Redis cache implementation."""

import fnmatch
import itertools
import time
from typing import Any

import pytest
import redis

from langgraph.cache.base import FullKey
from langgraph.cache.redis import RedisCache


class FakePipeline:
    """Minimal in-memory stand-in for a Redis pipeline."""

    def __init__(self, store: dict[str, bytes]) -> None:
        self.store = store

    def set(self, key: str, value: bytes) -> None:
        self.store[key] = value

    def setex(self, key: str, ttl: int, value: bytes) -> None:
        self.store[key] = value

    def execute(self) -> list[Any]:
        return []


class FakeRedis:
    """Minimal in-memory stand-in for a Redis client (no server needed)."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def mget(self, keys: list[str]) -> list[bytes | None]:
        return [self.store.get(key) for key in keys]

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self.store)

    def keys(self, pattern: str) -> list[str]:
        return [key for key in self.store if fnmatch.fnmatchcase(key, pattern)]

    def delete(self, *keys: str) -> None:
        for key in keys:
            self.store.pop(key, None)


class TestRedisCacheKeyEncoding:
    """Regression tests for issue #8851: non-injective key encoding.

    These tests exercise the key encoding only and do not require a live
    Redis server.
    """

    def make_cache(self) -> tuple[RedisCache, FakeRedis]:
        client = FakeRedis()
        return RedisCache(client, prefix="test:cache:"), client

    def test_make_key_is_injective(self) -> None:
        """Distinct FullKeys must map to distinct Redis keys."""
        cache, _ = self.make_cache()
        full_keys: list[FullKey] = [
            (("a", "b"), "k"),
            (("a:b",), "k"),
            (("a",), "b:k"),
            ((), "a:b"),
            (("a",), "b"),
            ((), "a"),
            (("a", "b", "c"), "k"),
            (("a:b:c",), "k"),
        ]
        redis_keys = [cache._make_key(ns, key) for ns, key in full_keys]
        for (left_full, left), (right_full, right) in itertools.combinations(
            zip(full_keys, redis_keys, strict=True), 2
        ):
            assert left != right, f"{left_full} and {right_full} both encode to {left}"

    def test_parse_key_round_trip(self) -> None:
        """_parse_key must invert _make_key exactly."""
        cache, _ = self.make_cache()
        full_keys: list[FullKey] = [
            (("a", "b"), "k"),
            (("a:b",), "k"),
            (("a",), "b:k"),
            ((), "a:b"),
            (("graph:with:colons", "node"), "key"),
            (("with space", "with%percent"), "with*glob"),
            ((), "plain"),
        ]
        for ns, key in full_keys:
            assert cache._parse_key(cache._make_key(ns, key)) == (ns, key)

    def test_plain_segments_keep_legacy_encoding(self) -> None:
        """Keys without reserved characters keep their pre-fix Redis keys."""
        cache, _ = self.make_cache()
        assert (
            cache._make_key(("graph", "node"), "key1") == "test:cache:graph:node:key1"
        )
        assert cache._make_key((), "key1") == "test:cache:key1"

    def test_colliding_logical_keys_store_independent_values(self) -> None:
        """Logical keys that previously collided must not overwrite each other."""
        cache, _ = self.make_cache()
        key1: FullKey = (("a", "b"), "k")
        key2: FullKey = (("a:b",), "k")
        key3: FullKey = (("a",), "b:k")

        cache.set(
            {
                key1: ({"value": 1}, None),
                key2: ({"value": 2}, None),
                key3: ({"value": 3}, None),
            }
        )

        result = cache.get([key1, key2, key3])
        assert result[key1] == {"value": 1}
        assert result[key2] == {"value": 2}
        assert result[key3] == {"value": 3}

    def test_clear_namespace_with_colon_is_unambiguous(self) -> None:
        """clear() must not delete a sibling namespace whose encoded prefix
        would overlap under the old colon-joined encoding."""
        cache, client = self.make_cache()
        nested: FullKey = (("a", "b"), "k")
        colon: FullKey = (("a:b",), "k")
        cache.set({nested: ({"value": 1}, None), colon: ({"value": 2}, None)})

        # Clearing ("a:b",) must leave ("a", "b") untouched.
        cache.clear([("a:b",)])
        result = cache.get([nested, colon])
        assert result == {nested: {"value": 1}}

        # And clearing ("a",) must clear the nested namespace, not ("a:b",).
        cache.set({colon: ({"value": 2}, None)})
        cache.clear([("a",)])
        result = cache.get([nested, colon])
        assert result == {colon: {"value": 2}}

    def test_clear_all_with_fake_client(self) -> None:
        cache, client = self.make_cache()
        cache.set({(("a:b",), "k"): ({"value": 1}, None)})
        assert client.store
        cache.clear()
        assert not client.store


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
