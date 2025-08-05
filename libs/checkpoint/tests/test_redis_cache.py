"""Unit tests for Redis cache implementation."""
import pytest
import redis

from langgraph.cache.redis import RedisCache


class TestRedisCache:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test Redis client and cache."""
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)
        try:
            self.redis_client.ping()
        except redis.ConnectionError:
            pytest.skip("Redis server not available")
        
        self.cache = RedisCache(self.redis_client, prefix="test:cache:")
        
        # Clean up before each test
        self.redis_client.flushdb()

    def teardown_method(self):
        """Clean up after each test."""
        try:
            self.redis_client.flushdb()
        except Exception:
            pass

    def test_basic_set_and_get(self):
        """Test basic set and get operations."""
        keys = [(("graph", "node"), "key1")]
        values = {keys[0]: ({"result": 42}, None)}
        
        # Set value
        self.cache.set(values)
        
        # Get value
        result = self.cache.get(keys)
        assert len(result) == 1
        assert result[keys[0]] == {"result": 42}

    def test_batch_operations(self):
        """Test batch set and get operations."""
        keys = [
            (("graph", "node1"), "key1"),
            (("graph", "node2"), "key2"),
            (("other", "node"), "key3")
        ]
        values = {
            keys[0]: ({"result": 1}, None),
            keys[1]: ({"result": 2}, 60),  # With TTL
            keys[2]: ({"result": 3}, None)
        }
        
        # Set values
        self.cache.set(values)
        
        # Get all values
        result = self.cache.get(keys)
        assert len(result) == 3
        assert result[keys[0]] == {"result": 1}
        assert result[keys[1]] == {"result": 2}
        assert result[keys[2]] == {"result": 3}

    def test_ttl_behavior(self):
        """Test TTL (time-to-live) functionality."""
        import time
        
        key = (("graph", "node"), "ttl_key")
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

    def test_namespace_isolation(self):
        """Test that different namespaces are isolated."""
        key1 = (("graph1", "node"), "same_key")
        key2 = (("graph2", "node"), "same_key")
        
        values = {
            key1: ({"graph": 1}, None),
            key2: ({"graph": 2}, None)
        }
        
        self.cache.set(values)
        
        result = self.cache.get([key1, key2])
        assert result[key1] == {"graph": 1}
        assert result[key2] == {"graph": 2}

    def test_clear_all(self):
        """Test clearing all cached values."""
        keys = [
            (("graph", "node1"), "key1"),
            (("graph", "node2"), "key2")
        ]
        values = {
            keys[0]: ({"result": 1}, None),
            keys[1]: ({"result": 2}, None)
        }
        
        self.cache.set(values)
        
        # Verify data exists
        result = self.cache.get(keys)
        assert len(result) == 2
        
        # Clear all
        self.cache.clear()
        
        # Verify data is gone
        result = self.cache.get(keys)
        assert len(result) == 0

    def test_clear_by_namespace(self):
        """Test clearing cached values by namespace."""
        keys = [
            (("graph1", "node"), "key1"),
            (("graph2", "node"), "key2"),
            (("graph1", "other"), "key3")
        ]
        values = {
            keys[0]: ({"result": 1}, None),
            keys[1]: ({"result": 2}, None),
            keys[2]: ({"result": 3}, None)
        }
        
        self.cache.set(values)
        
        # Clear only graph1 namespace
        self.cache.clear([("graph1", "node"), ("graph1", "other")])
        
        # graph1 should be cleared, graph2 should remain
        result = self.cache.get(keys)
        assert len(result) == 1
        assert result[keys[1]] == {"result": 2}

    def test_empty_operations(self):
        """Test behavior with empty keys/values."""
        # Empty get
        result = self.cache.get([])
        assert result == {}
        
        # Empty set
        self.cache.set({})  # Should not raise error

    def test_nonexistent_keys(self):
        """Test getting keys that don't exist."""
        keys = [(("graph", "node"), "nonexistent")]
        result = self.cache.get(keys)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test async set and get operations."""
        keys = [(("graph", "node"), "async_key")]
        values = {keys[0]: ({"async": True}, None)}
        
        # Async set
        await self.cache.aset(values)
        
        # Async get
        result = await self.cache.aget(keys)
        assert len(result) == 1
        assert result[keys[0]] == {"async": True}

    @pytest.mark.asyncio
    async def test_async_clear(self):
        """Test async clear operations."""
        keys = [(("graph", "node"), "key")]
        values = {keys[0]: ({"data": "test"}, None)}
        
        await self.cache.aset(values)
        
        # Verify data exists
        result = await self.cache.aget(keys)
        assert len(result) == 1
        
        # Clear all
        await self.cache.aclear()
        
        # Verify data is gone
        result = await self.cache.aget(keys)
        assert len(result) == 0