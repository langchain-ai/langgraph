"""Key/value cache for use inside LangGraph deployments.

Thin wrapper around ``langgraph_api.cache``.
Values must be JSON-serializable (dicts, lists, strings, numbers, booleans,
``None``).
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import inspect
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import PurePath
from typing import Any, Generic, Literal, TypeVar, get_type_hints, overload
from uuid import UUID

import orjson

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
    "swr_cached",
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


def _build_cache_key(
    module: str,
    qualname: str,
    sig: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    payload = {
        "module": module,
        "qualname": qualname,
        "args": [
            {
                "name": name,
                "value": _normalize_cache_key_value(value, name=name, path=name),
            }
            for name, value in bound.arguments.items()
        ],
    }
    return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode()


def _type_identifier(tp: type[Any]) -> str:
    return f"{tp.__module__}.{tp.__qualname__}"


def _stable_key_dump(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def _normalize_cache_key_value(
    value: Any,
    *,
    name: str | None = None,
    path: str = "value",
) -> Any:
    if name in {"self", "cls"}:
        cls = value if isinstance(value, type) else type(value)
        return {"class": _type_identifier(cls), "kind": name}

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return {"hex": value.hex(), "kind": "bytes"}

    if isinstance(value, (datetime, date, time)):
        return {"kind": type(value).__name__, "value": value.isoformat()}

    if isinstance(value, timedelta):
        return {"kind": "timedelta", "value": value.total_seconds()}

    if isinstance(value, Decimal):
        return {"kind": "decimal", "value": str(value)}

    if isinstance(value, UUID):
        return {"kind": "uuid", "value": str(value)}

    if isinstance(value, PurePath):
        return {"kind": "path", "value": str(value)}

    if isinstance(value, enum.Enum):
        return {
            "kind": "enum",
            "type": _type_identifier(type(value)),
            "value": _normalize_cache_key_value(value.value, path=f"{path}.value"),
        }

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "fields": {
                field.name: _normalize_cache_key_value(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}",
                )
                for field in dataclasses.fields(value)
            },
            "kind": "dataclass",
            "type": _type_identifier(type(value)),
        }

    if hasattr(value, "model_dump") and callable(value.model_dump):
        if isinstance(value, type):
            raise TypeError(
                f"Cannot auto-generate a stable cache key for `{path}` from the "
                f"type object {value!r}. Pass `key=` to `swr_cached` instead."
            )
        return {
            "kind": "model",
            "type": _type_identifier(type(value)),
            "value": _normalize_cache_key_value(
                value.model_dump(mode="json"),
                path=path,
            ),
        }

    if hasattr(value, "dict") and callable(value.dict):
        if isinstance(value, type):
            raise TypeError(
                f"Cannot auto-generate a stable cache key for `{path}` from the "
                f"type object {value!r}. Pass `key=` to `swr_cached` instead."
            )
        return {
            "kind": "model",
            "type": _type_identifier(type(value)),
            "value": _normalize_cache_key_value(value.dict(), path=path),
        }

    if isinstance(value, Mapping):
        items = [
            {
                "key": _normalize_cache_key_value(key, path=f"{path}.<key>"),
                "value": _normalize_cache_key_value(
                    item,
                    path=f"{path}[{key!r}]",
                ),
            }
            for key, item in value.items()
        ]
        items.sort(key=lambda item: _stable_key_dump(item["key"]))
        return {"items": items, "kind": "mapping"}

    if isinstance(value, tuple):
        return {
            "items": [
                _normalize_cache_key_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
            "kind": "tuple",
        }

    if isinstance(value, list):
        return {
            "items": [
                _normalize_cache_key_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
            "kind": "list",
        }

    if isinstance(value, (set, frozenset)):
        items = [_normalize_cache_key_value(item, path=f"{path}[]") for item in value]
        items.sort(key=_stable_key_dump)
        return {"items": items, "kind": type(value).__name__}

    if isinstance(value, type):
        return {"kind": "type", "type": _type_identifier(value)}

    if type(value).__repr__ is object.__repr__:
        raise TypeError(
            f"Cannot auto-generate a stable cache key for `{path}` of type "
            f"`{_type_identifier(type(value))}`. Pass `key=` to `swr_cached` "
            "instead."
        )

    return {
        "kind": "repr",
        "repr": repr(value),
        "type": _type_identifier(type(value)),
    }


def _get_model_from_hints(func: Callable[..., Any]) -> type | None:
    try:
        hints = get_type_hints(func)
    except Exception:
        return None
    ret = hints.get("return")
    if ret is None:
        return None
    try:
        from pydantic import BaseModel
    except ImportError:
        return None
    if isinstance(ret, type) and issubclass(ret, BaseModel):
        return ret
    return None


@overload
def swr_cached(
    fn: Callable[..., Awaitable[T]],
    /,
) -> Callable[..., Awaitable[SWRResult[T]]]: ...


@overload
def swr_cached(
    *,
    key: str | Callable[..., str] | None = ...,
    fresh_for: timedelta | None = ...,
    max_age: timedelta | None = ...,
    model: type[T] | None = ...,
) -> Callable[
    [Callable[..., Awaitable[T]]], Callable[..., Awaitable[SWRResult[T]]]
]: ...


def swr_cached(
    fn=None,
    *,
    key=None,
    fresh_for=None,
    max_age=None,
    model=None,
):
    """Decorator that wraps an async function with :func:`swr` caching.

    Can be used with or without parentheses::

        @swr_cached
        async def fetch_config():
            ...

        @swr_cached(fresh_for=timedelta(minutes=5))
        async def fetch_profile(user_id: str) -> Profile:
            ...

    The cache key is auto-derived from the function's module, qualified name,
    and a structured serialization of the bound call arguments. For methods,
    ``self`` and ``cls`` are keyed by class identity rather than object
    instance identity. Override with ``key=`` (a static string or a callable
    that receives the same arguments as the decorated function) when method
    state matters or arguments are not stably serializable.

    If the return annotation is a Pydantic `BaseModel` subclass and
    ``model`` is not provided, the model is inferred automatically.
    """

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[SWRResult[T]]]:
        sig = inspect.signature(func)
        resolved_model = model
        if resolved_model is None:
            resolved_model = _get_model_from_hints(func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> SWRResult[T]:
            if key is None:
                module = getattr(func, "__module__", None) or "unknown"
                qualname = getattr(func, "__qualname__", None) or getattr(
                    func, "__name__", "unknown"
                )
                cache_key = _build_cache_key(module, qualname, sig, args, kwargs)
            elif callable(key):
                cache_key = key(*args, **kwargs)
            else:
                cache_key = key
            return await swr(
                cache_key,
                lambda: func(*args, **kwargs),
                fresh_for=fresh_for,
                max_age=max_age,
                model=resolved_model,
            )

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
