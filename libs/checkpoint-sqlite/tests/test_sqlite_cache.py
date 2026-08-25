import inspect
from pathlib import Path
from typing import Any

import pytest
from langgraph.cache.base import FullKey

from langgraph.cache.sqlite import SqliteCache


def test_sqlite_cache_signature() -> None:
    set_params = list(inspect.signature(SqliteCache.set).parameters.keys())
    assert "pairs" in set_params, "SqliteCache.set must have 'pairs' parameter"

    aset_params = list(inspect.signature(SqliteCache.aset).parameters.keys())
    assert "pairs" in aset_params, "SqliteCache.aset must have 'pairs' parameter"


def test_sqlite_cache_keyword_args(tmp_path: Path) -> None:
    db_path = str(tmp_path / "cache_sync.db")
    cache = SqliteCache(path=db_path)
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
async def test_sqlite_cache_async_keyword_args(tmp_path: Path) -> None:
    db_path = str(tmp_path / "cache_async.db")
    cache = SqliteCache(path=db_path)
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
