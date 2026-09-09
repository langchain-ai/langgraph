from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, unquote

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol


def _encode_segment(segment: str) -> str:
    """Percent-encode a namespace segment or cache key.

    Escaping every character outside `[A-Za-z0-9_.~-]` guarantees that an
    encoded segment can never contain the `:` separator (or the Redis glob
    metacharacters `*`, `?`, `[` and `]`), which makes the joined key
    injective: distinct `(namespace, key)` pairs always map to distinct
    Redis keys. Segments without reserved characters are unchanged, so keys
    produced by earlier releases remain valid for them.
    """
    return quote(segment, safe="")


class RedisCache(BaseCache[ValueT]):
    """Redis-based cache implementation with TTL support."""

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
        """Create a Redis key from namespace and key.

        Each namespace segment and the key are percent-encoded before being
        joined with `:` so that separators occurring inside segments cannot
        collide with the segment boundaries (e.g. `(("a", "b"), "k")` versus
        `(("a:b",), "k")`).
        """
        parts = [_encode_segment(segment) for segment in (*ns, key)]
        return f"{self.prefix}{':'.join(parts)}"

    def _parse_key(self, redis_key: str) -> tuple[Namespace, str]:
        """Parse a Redis key back to namespace and key."""
        if not redis_key.startswith(self.prefix):
            raise ValueError(
                f"Key {redis_key} does not start with prefix {self.prefix}"
            )

        remaining = redis_key[len(self.prefix) :]
        *ns_parts, key = remaining.split(":")
        return (tuple(unquote(part) for part in ns_parts), unquote(key))

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
        return self.get(keys)

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
        self.set(mapping)

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
                    # Encoded segments contain no glob metacharacters, so the
                    # pattern matches exactly the requested namespace.
                    ns_str = ":".join(_encode_segment(segment) for segment in ns)
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
        self.clear(namespaces)
