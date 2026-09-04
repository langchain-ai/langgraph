from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol


class RedisCache(BaseCache[ValueT]):
    """Redis-based cache implementation with TTL support.
    
    This class supports both synchronous and asynchronous Redis clients,
    but enforces explicit usage to prevent runtime errors and confusion.
    """

    def __init__(
        self,
        redis: Any,
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "langgraph:cache:",
    ) -> None:
        """Initialize the cache with a Redis client.

        Args:
            redis: Redis client instance (sync or async)
            serde: Serializer to use for values
            prefix: Key prefix for all cached values
            
        Raises:
            ValueError: If redis client is invalid or doesn't have required methods
        """
        super().__init__(serde=serde)
        self.redis = redis
        self.prefix = prefix
        
        # Validate client type during initialization and store result
        self._is_async = self._validate_and_detect_client_type()

    def _validate_and_detect_client_type(self) -> bool:
        """Validate Redis client and detect if it's async during initialization.
        
        Returns:
            bool: True if async client, False if sync client
            
        Raises:
            ValueError: If client is invalid or lacks required methods
        """
        # Check for required Redis methods
        required_methods = ['mget', 'pipeline', 'keys', 'delete']
        for method in required_methods:
            if not hasattr(self.redis, method):
                raise ValueError(
                    f"Invalid Redis client: missing '{method}' method"
                )
        
        # Detect client type based on mget method
        if inspect.iscoroutinefunction(self.redis.mget):
            return True
        elif callable(self.redis.mget):
            return False
        else:
            raise ValueError("Invalid Redis client: 'mget' not callable")

    def _make_key(self, ns: Namespace, key: str) -> str:
        """Create a Redis key from namespace and key."""
        ns_str = ":".join(ns) if ns else ""
        return f"{self.prefix}{ns_str}:{key}" if ns_str else f"{self.prefix}{key}"

    def _parse_key(self, redis_key: str) -> tuple[Namespace, str]:
        """Parse a Redis key back to namespace and key."""
        if not redis_key.startswith(self.prefix):
            raise ValueError(
                f"Key {redis_key} does not start with prefix {self.prefix}"
            )

        remaining = redis_key[len(self.prefix) :]
        if ":" in remaining:
            parts = remaining.split(":")
            key = parts[-1]
            ns_parts = parts[:-1]
            return (tuple(ns_parts), key)
        else:
            return (tuple(), remaining)

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""
        if not keys:
            return {}

        # Build Redis keys
        redis_keys = [self._make_key(ns, key) for ns, key in keys]

        # Get values from Redis using MGET
        try:
            raw_values = self.redis.mget(redis_keys)
        except Exception:
            # If Redis is unavailable, return empty dict
            return {}

        values: dict[FullKey, ValueT] = {}
        for i, raw_value in enumerate(raw_values):
            if raw_value is not None:
                try:
                    # Deserialize the value
                    encoding, data = raw_value.split(b":", 1)
                    values[keys[i]] = self.serde.loads_typed((encoding.decode(), data))
                except Exception:
                    # Skip corrupted entries
                    continue

        return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""
        if not self._is_async:
            raise RuntimeError(
                "Cannot use async method 'aget' with synchronous Redis client. "
                "Use 'get' method instead, or initialize with redis.asyncio.Redis."
            )
        
        if not keys:
            return {}

        # Build Redis keys
        redis_keys = [self._make_key(ns, key) for ns, key in keys]

        # Get values from Redis using MGET (properly await)
        try:
            raw_values = await self.redis.mget(redis_keys)
        except Exception:
            # If Redis is unavailable, return empty dict
            return {}

        values: dict[FullKey, ValueT] = {}
        for i, raw_value in enumerate(raw_values):
            if raw_value is not None:
                try:
                    # Deserialize the value
                    encoding, data = raw_value.split(b":", 1)
                    values[keys[i]] = self.serde.loads_typed((encoding.decode(), data))
                except Exception:
                    # Skip corrupted entries
                    continue

        return values

    def set(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""
        if not mapping:
            return

        # Use pipeline for efficient batch operations
        pipe = self.redis.pipeline()

        for (ns, key), (value, ttl) in mapping.items():
            redis_key = self._make_key(ns, key)
            encoding, data = self.serde.dumps_typed(value)

            # Store as "encoding:data" format
            serialized_value = f"{encoding}:".encode() + data

            if ttl is not None:
                pipe.setex(redis_key, ttl, serialized_value)
            else:
                pipe.set(redis_key, serialized_value)

        try:
            pipe.execute()
        except Exception:
            # Silently fail if Redis is unavailable
            pass

    async def aset(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""
        if not self._is_async:
            raise RuntimeError(
                "Cannot use async method 'aset' with synchronous Redis client. "
                "Use 'set' method instead, or initialize with redis.asyncio.Redis."
            )
        
        if not mapping:
            return

        # Use pipeline for efficient batch operations
        pipe = self.redis.pipeline()

        for (ns, key), (value, ttl) in mapping.items():
            redis_key = self._make_key(ns, key)
            encoding, data = self.serde.dumps_typed(value)

            # Store as "encoding:data" format
            serialized_value = f"{encoding}:".encode() + data

            if ttl is not None:
                pipe.setex(redis_key, ttl, serialized_value)
            else:
                pipe.set(redis_key, serialized_value)

        try:
            await pipe.execute()
        except Exception:
            # Silently fail if Redis is unavailable
            pass

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        try:
            if namespaces is None:
                # Clear all keys with our prefix
                pattern = f"{self.prefix}*"
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
            else:
                # Clear specific namespaces
                keys_to_delete = []
                for ns in namespaces:
                    ns_str = ":".join(ns) if ns else ""
                    pattern = (
                        f"{self.prefix}{ns_str}:*" if ns_str else f"{self.prefix}*"
                    )
                    keys = self.redis.keys(pattern)
                    keys_to_delete.extend(keys)

                if keys_to_delete:
                    self.redis.delete(*keys_to_delete)
        except Exception:
            # Silently fail if Redis is unavailable
            pass

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        # For backward compatibility, delegate to sync clear
        self.clear(namespaces)
