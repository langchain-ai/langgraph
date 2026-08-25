import inspect
from typing import Any

import pytest

from langgraph.cache.base import BaseCache, FullKey
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.redis import RedisCache


def test_base_cache_signatures() -> None:
    for cache_cls in (BaseCache, InMemoryCache, RedisCache):
        set_params = list(inspect.signature(cache_cls.set).parameters.keys())
        assert "pairs" in set_params, (
            f"{cache_cls.__name__}.set must have 'pairs' parameter"
        )

        aset_params = list(inspect.signature(cache_cls.aset).parameters.keys())
        assert "pairs" in aset_params, (
            f"{cache_cls.__name__}.aset must have 'pairs' parameter"
        )


def test_in_memory_cache_keyword_args() -> None:
    cache = InMemoryCache()
    key: FullKey = (("ns",), "k1")
    value: tuple[Any, int | None] = ({"data": 123}, 60)

    # Calling with keyword argument 'pairs'
    cache.set(pairs={key: value})
    res = cache.get([key])
    assert res == {key: {"data": 123}}

    # Calling with positional argument
    key2: FullKey = (("ns",), "k2")
    cache.set({key2: ({"data": 456}, None)})
    assert cache.get([key2]) == {key2: {"data": 456}}


@pytest.mark.asyncio
async def test_in_memory_cache_async_keyword_args() -> None:
    cache = InMemoryCache()
    key: FullKey = (("ns",), "k1")
    value: tuple[Any, int | None] = ({"data": 789}, 60)

    # Calling with keyword argument 'pairs'
    await cache.aset(pairs={key: value})
    res = await cache.aget([key])
    assert res == {key: {"data": 789}}

    # Calling with positional argument
    key2: FullKey = (("ns",), "k2")
    await cache.aset({key2: ({"data": 999}, None)})
    assert await cache.aget([key2]) == {key2: {"data": 999}}
