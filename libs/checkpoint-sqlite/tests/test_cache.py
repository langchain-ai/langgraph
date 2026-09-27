import sqlite3
import sys
from collections.abc import Iterator

import pytest
from langgraph.cache.base import FullKey

from langgraph.cache.sqlite import SqliteCache


@pytest.fixture
def cache() -> Iterator[SqliteCache[str]]:
    cache = SqliteCache[str](path=":memory:")
    try:
        yield cache
    finally:
        cache._conn.close()


@pytest.fixture
def limited_cache(cache: SqliteCache[str]) -> SqliteCache[str]:
    if sys.version_info < (3, 11):
        pytest.skip("Connection.setlimit requires Python 3.11 or later")
    cache._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 7)
    return cache


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("count", [0, 1, 3, 4, 10])
async def test_get_parameter_limit(
    limited_cache: SqliteCache[str], count: int, asynchronous: bool
) -> None:
    keys: list[FullKey] = [(("ns",), f"key-{i}") for i in range(count)]
    expected = {key: f"value-{i}" for i, key in enumerate(keys)}
    limited_cache.set({key: (value, None) for key, value in expected.items()})

    result = await limited_cache.aget(keys) if asynchronous else limited_cache.get(keys)

    assert result == expected


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_get_batches_preserve_cache_semantics(
    limited_cache: SqliteCache[str], asynchronous: bool
) -> None:
    first: FullKey = (("ns", "nested"), "first")
    second: FullKey = (("ns",), "second")
    third: FullKey = (("other",), "third")
    expired: FullKey = (("ns",), "expired")
    missing: FullKey = (("ns",), "missing")
    unrequested: FullKey = (("ns",), "unrequested")
    expected = {first: "first", second: "second", third: "third"}
    limited_cache.set(
        {
            **{key: (value, None) for key, value in expected.items()},
            expired: ("expired", -1),
            unrequested: ("keep", None),
        }
    )
    keys = [first, missing, expired, second, first, expired, missing, third]

    result = await limited_cache.aget(keys) if asynchronous else limited_cache.get(keys)

    assert result == expected
    assert (
        limited_cache._conn.execute(
            "SELECT key FROM cache WHERE key = 'expired'"
        ).fetchall()
        == []
    )
    assert limited_cache.get([unrequested]) == {unrequested: "keep"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("count", [0, 1, 7, 8, 20])
async def test_clear_parameter_limit(
    limited_cache: SqliteCache[str], count: int, asynchronous: bool
) -> None:
    keys: list[FullKey] = [((f"ns-{i}",), "key") for i in range(count)]
    retained: FullKey = (("retain",), "key")
    limited_cache.set({key: ("value", None) for key in [*keys, retained]})
    namespaces = [key[0] for key in keys]

    if asynchronous:
        await limited_cache.aclear(namespaces)
    else:
        limited_cache.clear(namespaces)

    assert limited_cache._conn.execute("SELECT ns, key FROM cache").fetchall() == [
        ("retain", "key")
    ]
    assert limited_cache.get([retained]) == {retained: "value"}


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_clear_all(limited_cache: SqliteCache[str], asynchronous: bool) -> None:
    keys: list[FullKey] = [((f"ns-{i}",), "key") for i in range(20)]
    limited_cache.set({key: ("value", None) for key in keys})

    if asynchronous:
        await limited_cache.aclear()
    else:
        limited_cache.clear()

    assert limited_cache._conn.execute("SELECT * FROM cache").fetchall() == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_clear_batches_roll_back_on_later_failure(
    limited_cache: SqliteCache[str], asynchronous: bool
) -> None:
    namespaces = [(f"ns-{i}",) for i in range(8)]
    limited_cache.set({(ns, "key"): ("value", None) for ns in namespaces})
    deleted: list[str] = []
    limited_cache._conn.create_function("track_delete", 1, deleted.append)
    limited_cache._conn.executescript(
        """
        CREATE TRIGGER record_delete AFTER DELETE ON cache
        BEGIN
            SELECT track_delete(OLD.ns);
        END;
        CREATE TRIGGER fail_later BEFORE DELETE ON cache
        WHEN OLD.ns = 'ns-7'
        BEGIN
            SELECT RAISE(ABORT, 'later batch failed');
        END;
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="later batch failed"):
        if asynchronous:
            await limited_cache.aclear(namespaces)
        else:
            limited_cache.clear(namespaces)

    assert set(deleted) == {f"ns-{i}" for i in range(7)}
    assert limited_cache._conn.execute(
        "SELECT ns FROM cache ORDER BY ns"
    ).fetchall() == [(ns[0],) for ns in namespaces]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_large_cache_operations(
    cache: SqliteCache[str], asynchronous: bool
) -> None:
    keys: list[FullKey] = [((f"ns-{i}",), "key") for i in range(1001)]
    expected = {key: f"value-{i}" for i, key in enumerate(keys)}
    cache.set({key: (value, None) for key, value in expected.items()})
    statements: list[str] = []
    cache._conn.set_trace_callback(statements.append)

    result = await cache.aget(keys) if asynchronous else cache.get(keys)
    assert result == expected
    if sys.version_info < (3, 11):
        assert sum(sql.startswith("SELECT") for sql in statements) == 3

    namespaces = [key[0] for key in keys[:-1]]
    statements.clear()
    if asynchronous:
        await cache.aclear(namespaces)
    else:
        cache.clear(namespaces)
    if sys.version_info < (3, 11):
        assert sum(sql.startswith("DELETE") for sql in statements) == 2

    result = await cache.aget(keys) if asynchronous else cache.get(keys)
    assert result == {keys[-1]: expected[keys[-1]]}
