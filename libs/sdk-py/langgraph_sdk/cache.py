"""Key/value cache for use inside LangGraph deployments.

Thin wrapper around ``langgraph_api.cache``.
Values must be JSON-serializable (dicts, lists, strings, numbers, booleans,
``None``).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

try:
    from langgraph_api.cache import (  # type: ignore[unresolved-import]
        cache_get as _cache_get,
    )
    from langgraph_api.cache import (  # type: ignore[unresolved-import]
        cache_set as _cache_set,
    )
except ImportError:
    _cache_get = None
    _cache_set = None


__all__ = [
    "cache_get",
    "cache_set",
]


async def cache_get(key: str) -> Any | None:
    """Get a value from the cache.

    Returns the deserialized value, or ``None`` if the key is missing or expired.

    Requires Agent Server runtime version 0.7.29 or later.
    """
    if _cache_get is None:
        raise RuntimeError(
            "Cache is only available server-side within the LangGraph Agent Server "
            "(https://docs.langchain.com/langsmith/deployments)."
        )
    return await _cache_get(key)


async def cache_set(key: str, value: Any, *, ttl: timedelta | None = None) -> None:
    """Set a value in the cache.

    Args:
        key: The cache key.
        value: The value to cache (must be JSON-serializable).
        ttl: Optional time-to-live. Capped at 1 day; ``None`` or zero
            defaults to 1 day.

    Requires Agent Server runtime version 0.7.29 or later.
    """
    if _cache_set is None:
        raise RuntimeError(
            "Cache is only available server-side within the LangGraph Agent Server "
            "(https://docs.langchain.com/langsmith/deployments)."
        )
    await _cache_set(key, value, ttl)
