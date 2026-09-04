from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol


class RedisCache(BaseCache[ValueT]):
    """Redis-based cache implementation with TTL support.
    
    Supports both synchronous and asynchronous Redis clients by detecting
    the client type and calling appropriate methods.
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
        """
        super().__init__(serde=serde)
        self.redis = redis
        self.prefix = prefix

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

    def _is_async_redis(self) -> bool:
        """Check if the Redis client is async by checking for async methods."""
        return hasattr(self.redis, "mget") and inspect.iscoroutinefunction(
            self.redis.mget
        )

    def _process_raw_values(
        self, keys: Sequence[FullKey], raw_values: Any
    ) -> dict[FullKey, ValueT]:
        """Process raw Redis values into typed dictionary."""
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

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""
        if not keys:
            return {}

        # Build Redis keys
        redis_keys = [self._make_key(ns, key) for ns, key in keys]

        # Get values from Redis using MGET
        try:
            raw_values = self.redis.mget(redis_keys)
            return self._process_raw_values(keys, raw_values)
        except Exception:
            # If Redis is unavailable, return empty dict
            return {}

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""
        if not keys:
            return {}

        # Build Redis keys
        redis_keys = [self._make_key(ns, key) for ns, key in keys]

        try:
            if self._is_async_redis():
                # Use async Redis client
                raw_values = await self.redis.mget(redis_keys)
            else:
                # Fall back to sync Redis client
                raw_values = self.redis.mget(redis_keys)
            
            return self._process_raw_values(keys, raw_values)
        except Exception:
            # If Redis is unavailable, return empty dict
            return {}

    def _build_set_pipeline(
        self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]
    ) -> Any:
        """Build Redis pipeline for set operations."""
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
        
        return pipe

    def set(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""
        if not mapping:
            return

        try:
            pipe = self._build_set_pipeline(mapping)
            pipe.execute()
        except Exception:
            # Silently fail if Redis is unavailable
            pass

    async def aset(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""
        if not mapping:
            return

        try:
            pipe = self._build_set_pipeline(mapping)
            
            if self._is_async_redis():
                # Use async Redis client
                await pipe.execute()
            else:
                # Fall back to sync Redis client
                pipe.execute()
        except Exception:
            # Silently fail if Redis is unavailable
            pass

    def _get_keys_to_delete(
        self, namespaces: Sequence[Namespace] | None = None
    ) -> list[str]:
        """Get list of Redis keys to delete based on namespaces."""
        keys_to_delete = []
        
        if namespaces is None:
            # Clear all keys with our prefix
            pattern = f"{self.prefix}*"
            keys = self.redis.keys(pattern)
            keys_to_delete.extend(keys)
        else:
            # Clear specific namespaces
            for ns in namespaces:
                ns_str = ":".join(ns) if ns else ""
                pattern = (
                    f"{self.prefix}{ns_str}:*" if ns_str else f"{self.prefix}*"
                )
                keys = self.redis.keys(pattern)
                keys_to_delete.extend(keys)
                
        return keys_to_delete

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        try:
            keys_to_delete = self._get_keys_to_delete(namespaces)
            
            if keys_to_delete:
                self.redis.delete(*keys_to_delete)
        except Exception:
            # Silently fail if Redis is unavailable
            pass

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        try:
            if self._is_async_redis():
                # Use async Redis client for key lookup and deletion
                keys_to_delete = []
                
                if namespaces is None:
                    # Clear all keys with our prefix
                    pattern = f"{self.prefix}*"
                    keys = await self.redis.keys(pattern)
                    keys_to_delete.extend(keys)
                else:
                    # Clear specific namespaces
                    for ns in namespaces:
                        ns_str = ":".join(ns) if ns else ""
                        pattern = (
                            f"{self.prefix}{ns_str}:*" if ns_str else f"{self.prefix}*"
                        )
                        keys = await self.redis.keys(pattern)
                        keys_to_delete.extend(keys)
                
                if keys_to_delete:
                    await self.redis.delete(*keys_to_delete)
            else:
                # Fall back to sync Redis client
                keys_to_delete = self._get_keys_to_delete(namespaces)
                
                if keys_to_delete:
                    self.redis.delete(*keys_to_delete)
        except Exception:
            # Silently fail if Redis is unavailable
            pass
