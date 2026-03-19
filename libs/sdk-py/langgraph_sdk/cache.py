"""Key/value cache for use inside LangGraph deployments.

Thin wrapper around ``langgraph_api.cache``.
Values must be JSON-serializable (dicts, lists, strings, numbers, booleans,
``None``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T")

CacheStatus = Literal["miss", "fresh", "stale", "expired"]

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


try:
    from langgraph_api.cache import SWRResult  # type: ignore[unresolved-import]
    from langgraph_api.cache import swr as _api_swr  # type: ignore[unresolved-import]

except ImportError:
    _api_swr = None

    class SWRResult(Generic[T]):
        """Result wrapper returned by :func:`swr`."""

        value: T
        status: CacheStatus

        async def mutate(self, value: T = ...) -> T:  # type: ignore[assignment]
            """Update or revalidate the cached value."""
            ...


__all__ = [
    "SWRResult",
    "cache_get",
    "cache_set",
    "swr",
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


async def swr(
    key: str,
    loader: Callable[[], Awaitable[T]],
    *,
    fresh_for: timedelta | None = None,
    max_age: timedelta | None = None,
    model: type[T] | None = None,
) -> SWRResult[T]:
    """Load a cached value using stale-while-revalidate semantics.

    This helper is server-side only and is intended for caching internal async
    dependencies such as auth or metadata lookups.

    Args:
        key: Cache key.
        loader: Async callable that fetches the value on miss/revalidation.
        fresh_for: How long a cached value is considered fresh (no revalidation).
            Defaults to ``timedelta(0)`` so every access triggers a background
            revalidate while still returning the cached value instantly. Values
            above :data:`MAX_CACHE_TTL` are clamped to the backend maximum.
        max_age: Total lifetime of a cached entry. After this, the next access
            blocks on the loader. Defaults to :data:`MAX_CACHE_TTL` (24 h by
            default). Values above :data:`MAX_CACHE_TTL` are clamped to the
            backend maximum.
        model: Optional Pydantic model class. When provided, values are
            serialized via ``model_dump(mode="json")`` before storage and
            deserialized via ``model.model_validate()`` on read.

    Returns:
        An :class:`SWRResult` with ``.value``, ``.status``, and an async
        ``.mutate()`` method.

    Semantics:
    - cache miss: await ``loader()``, store the value, return it
    - fresh hit (age < fresh_for): return the cached value
    - stale hit (fresh_for <= age < max_age): return the cached value
      immediately and trigger a best-effort background refresh
    - expired (age >= max_age): await ``loader()``, store the value, return it
    """
    if _api_swr is None:
        raise RuntimeError(
            "Cache is only available server-side within the LangGraph Agent Server "
            "(https://docs.langchain.com/langsmith/deployments)."
        )
    if fresh_for is None:
        fresh_for = timedelta(0)
    if max_age is None:
        max_age = timedelta(days=1)
    return await _api_swr(
        key, loader, fresh_for=fresh_for, max_age=max_age, model=model
    )
