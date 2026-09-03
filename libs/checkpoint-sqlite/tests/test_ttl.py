"""Test SQLite store Time-To-Live (TTL) functionality."""

import asyncio
import os
import tempfile
import time
from collections.abc import Generator

import pytest
from langgraph.store.base import GetOp, TTLConfig

from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.aio import AsyncSqliteStore


@pytest.fixture
def temp_db_file() -> Generator[str, None, None]:
    """Create a temporary database file for testing."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield path
    os.unlink(path)


def test_ttl_basic(temp_db_file: str) -> None:
    """Test basic TTL functionality with synchronous API."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    with SqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes}
    ) as store:
        store.setup()

        store.put(("test",), "item1", {"value": "test"})

        item = store.get(("test",), "item1")
        assert item is not None
        assert item.value["value"] == "test"

        time.sleep(ttl_seconds + 1.0)

        store.sweep_ttl()

        item = store.get(("test",), "item1")
        assert item is None


@pytest.mark.flaky(retries=3)
def test_ttl_refresh(temp_db_file: str) -> None:
    """Test TTL refresh on read."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    with SqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes, "refresh_on_read": True}
    ) as store:
        store.setup()

        # Store an item with TTL
        store.put(("test",), "item1", {"value": "test"})

        # Sleep almost to expiration
        time.sleep(ttl_seconds - 0.5)
        swept = store.sweep_ttl()
        assert swept == 0

        # Get the item and refresh TTL
        item = store.get(("test",), "item1", refresh_ttl=True)
        assert item is not None

        time.sleep(ttl_seconds - 0.5)
        swept = store.sweep_ttl()
        assert swept == 0

        # Get the item, should still be there
        item = store.get(("test",), "item1")
        assert item is not None
        assert item.value["value"] == "test"

        # Sleep again but don't refresh this time
        time.sleep(ttl_seconds + 0.75)

        swept = store.sweep_ttl()
        assert swept == 1

        # Item should be gone now
        item = store.get(("test",), "item1")
        assert item is None


def test_ttl_sweeper(temp_db_file: str) -> None:
    """Test TTL sweeper thread."""
    ttl_seconds = 2
    ttl_minutes = ttl_seconds / 60

    ttl_config: TTLConfig = {
        "default_ttl": ttl_minutes,
        "sweep_interval_minutes": ttl_minutes / 2,
    }
    with SqliteStore.from_conn_string(
        temp_db_file,
        ttl=ttl_config,
    ) as store:
        store.setup()

        # Start the TTL sweeper
        store.start_ttl_sweeper()

        # Store an item with TTL
        store.put(("test",), "item1", {"value": "test"})

        # Item should be there initially
        item = store.get(("test",), "item1")
        assert item is not None

        # Wait for TTL to expire and the sweeper to run
        time.sleep(ttl_seconds + (ttl_seconds / 2) + 0.5)

        # Item should be gone now (swept automatically)
        item = store.get(("test",), "item1")
        assert item is None

        # Stop the sweeper
        store.stop_ttl_sweeper()


@pytest.mark.flaky(retries=3)
def test_ttl_custom_value(temp_db_file: str) -> None:
    """Test TTL with custom value per item."""
    with SqliteStore.from_conn_string(temp_db_file) as store:
        store.setup()

        # Store items with different TTLs
        store.put(("test",), "item1", {"value": "short"}, ttl=1 / 60)  # 1 second
        store.put(("test",), "item2", {"value": "long"}, ttl=3 / 60)  # 3 seconds

        # Item with short TTL
        time.sleep(2)  # Wait for short TTL
        store.sweep_ttl()

        # Probe reads must not extend the items' TTLs: refresh_on_read defaults to
        # True, so a default read of item2 here would push its expiry out and make
        # the final "should be gone" assertion timing-dependent.
        item1 = store.get(("test",), "item1", refresh_ttl=False)
        item2 = store.get(("test",), "item2", refresh_ttl=False)
        assert item1 is None
        assert item2 is not None

        # Wait for the second item's TTL
        time.sleep(4)
        store.sweep_ttl()

        # Now both should be gone
        item2 = store.get(("test",), "item2", refresh_ttl=False)
        assert item2 is None


@pytest.mark.flaky(retries=3)
def test_ttl_override_default(temp_db_file: str) -> None:
    """Test overriding default TTL at the item level."""
    with SqliteStore.from_conn_string(
        temp_db_file,
        ttl={"default_ttl": 5 / 60},  # 5 seconds default
    ) as store:
        store.setup()

        # Store an item with shorter than default TTL
        store.put(("test",), "item1", {"value": "override"}, ttl=1 / 60)  # 1 second

        # Store an item with default TTL
        store.put(("test",), "item2", {"value": "default"})  # Uses default 5 seconds

        # Store an item with no TTL
        store.put(("test",), "item3", {"value": "permanent"}, ttl=None)

        # Wait for the override TTL to expire
        time.sleep(2)
        store.sweep_ttl()

        # Probe reads must not extend the items' TTLs: refresh_on_read defaults to
        # True, so default reads would push item2's expiry out past the final sweep.
        item1 = store.get(("test",), "item1", refresh_ttl=False)
        item2 = store.get(("test",), "item2", refresh_ttl=False)
        item3 = store.get(("test",), "item3", refresh_ttl=False)

        assert item1 is None  # Should be expired
        assert item2 is not None  # Default TTL, should still be there
        assert item3 is not None  # No TTL, should still be there

        # Wait for default TTL to expire
        time.sleep(4)
        store.sweep_ttl()

        # Check results again
        item2 = store.get(("test",), "item2", refresh_ttl=False)
        item3 = store.get(("test",), "item3", refresh_ttl=False)

        assert item2 is None  # Default TTL item should be gone
        assert item3 is not None  # No TTL item should still be there


@pytest.mark.flaky(retries=3)
def test_search_with_ttl(temp_db_file: str) -> None:
    """Test TTL with search operations."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    with SqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes}
    ) as store:
        store.setup()

        # Store items
        store.put(("test",), "item1", {"value": "apple"})
        store.put(("test",), "item2", {"value": "banana"})

        # Search before expiration
        results = store.search(("test",), filter={"value": "apple"})
        assert len(results) == 1
        assert results[0].key == "item1"

        # Wait for TTL to expire
        time.sleep(ttl_seconds + 1)
        store.sweep_ttl()

        # Search after expiration
        results = store.search(("test",), filter={"value": "apple"})
        assert len(results) == 0


@pytest.mark.asyncio
async def test_async_ttl_basic(temp_db_file: str) -> None:
    """Test basic TTL functionality with asynchronous API."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes}
    ) as store:
        await store.setup()

        # Store an item with TTL
        await store.aput(("test",), "item1", {"value": "test"})

        # Get the item before expiration
        item = await store.aget(("test",), "item1")
        assert item is not None
        assert item.value["value"] == "test"

        # Wait for TTL to expire
        await asyncio.sleep(ttl_seconds + 1.0)

        # Manual sweep needed without the sweeper thread
        await store.sweep_ttl()

        # Item should be gone now
        item = await store.aget(("test",), "item1")
        assert item is None


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3)
async def test_async_ttl_refresh(temp_db_file: str) -> None:
    """Test TTL refresh on read with async API."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes, "refresh_on_read": True}
    ) as store:
        await store.setup()

        # Store an item with TTL
        await store.aput(("test",), "item1", {"value": "test"})

        # Sleep almost to expiration
        await asyncio.sleep(ttl_seconds - 0.5)

        # Get the item and refresh TTL
        item = await store.aget(("test",), "item1", refresh_ttl=True)
        assert item is not None

        # Sleep again - without refresh, would have expired by now
        await asyncio.sleep(ttl_seconds - 0.5)

        # Get the item, should still be there
        item = await store.aget(("test",), "item1")
        assert item is not None
        assert item.value["value"] == "test"

        # Sleep again but don't refresh this time
        await asyncio.sleep(ttl_seconds + 1.0)

        # Manual sweep
        await store.sweep_ttl()

        # Item should be gone now
        item = await store.aget(("test",), "item1")
        assert item is None


@pytest.mark.asyncio
async def test_async_ttl_sweeper(temp_db_file: str) -> None:
    """Test TTL sweeper thread with async API."""
    ttl_seconds = 2
    ttl_minutes = ttl_seconds / 60

    ttl_config: TTLConfig = {
        "default_ttl": ttl_minutes,
        "sweep_interval_minutes": ttl_minutes / 2,
    }

    async with AsyncSqliteStore.from_conn_string(
        temp_db_file,
        ttl=ttl_config,
    ) as store:
        await store.setup()

        # Start the TTL sweeper
        await store.start_ttl_sweeper()

        # Store an item with TTL
        await store.aput(("test",), "item1", {"value": "test"})

        # Item should be there initially
        item = await store.aget(("test",), "item1")
        assert item is not None

        # Wait for TTL to expire and the sweeper to run
        await asyncio.sleep(ttl_seconds + (ttl_seconds / 2) + 0.5)

        # Item should be gone now (swept automatically)
        item = await store.aget(("test",), "item1")
        assert item is None

        # Stop the sweeper
        await store.stop_ttl_sweeper()


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3)
async def test_async_search_with_ttl(temp_db_file: str) -> None:
    """Test TTL with search operations using async API."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes}
    ) as store:
        await store.setup()

        # Store items
        await store.aput(("test",), "item1", {"value": "apple"})
        await store.aput(("test",), "item2", {"value": "banana"})

        # Search before expiration
        results = await store.asearch(("test",), filter={"value": "apple"})
        assert len(results) == 1
        assert results[0].key == "item1"

        # Wait for TTL to expire
        await asyncio.sleep(ttl_seconds + 1)
        await store.sweep_ttl()

        # Search after expiration
        results = await store.asearch(("test",), filter={"value": "apple"})
        assert len(results) == 0


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3)
async def test_async_asearch_refresh_ttl(temp_db_file: str) -> None:
    """Test TTL refresh on asearch with async API."""
    ttl_seconds = 4.0  # Increased TTL for less sensitivity to timing
    ttl_minutes = ttl_seconds / 60.0

    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes, "refresh_on_read": True}
    ) as store:
        await store.setup()

        namespace = ("docs", "user1")
        # t=0: items put, expire at t=4.0s
        await store.aput(namespace, "item1", {"text": "content1", "id": 1})
        await store.aput(namespace, "item2", {"text": "content2", "id": 2})

        # t=3.0s: (after sleep ttl_seconds * 0.75 = 3s)
        await asyncio.sleep(ttl_seconds * 0.75)

        # Perform asearch with refresh_ttl=True for item1.
        # item1's TTL should be refreshed. New expiry: t=3.0s + 4.0s = t=7.0s.
        # item2's TTL is not affected. Expires at t=4.0s.
        searched_items = await store.asearch(
            namespace, filter={"id": 1}, refresh_ttl=True
        )
        assert len(searched_items) == 1
        assert searched_items[0].key == "item1"

        # t=5.0s: (after sleep ttl_seconds * 0.5 = 2s more. Total elapsed: 3s + 2s = 5s)
        await asyncio.sleep(ttl_seconds * 0.5)
        # At this point:
        # - item1 (refreshed by asearch) should expire at t=7.0s. Should be ALIVE.
        # - item2 (original TTL) should have expired at t=4.0s. Should be GONE after sweep.

        await store.sweep_ttl()

        # Check item1 (should exist due to asearch refresh)
        item1_check1 = await store.aget(namespace, "item1", refresh_ttl=False)
        assert item1_check1 is not None, (
            "Item1 should exist after asearch refresh and first sweep"
        )
        assert item1_check1.value["text"] == "content1"

        # Check item2 (should be gone)
        item2_check1 = await store.aget(namespace, "item2", refresh_ttl=False)
        assert item2_check1 is None, (
            "Item2 should be gone after its original TTL expired"
        )

        # t=7.5s: (after sleep ttl_seconds * 0.625 = 2.5s more. Total elapsed: 5s + 2.5s = 7.5s)
        await asyncio.sleep(ttl_seconds * 0.625)
        # At this point:
        # - item1 (refreshed by asearch, expired at t=7.0s) should be GONE after sweep.

        await store.sweep_ttl()

        # Check item1 again (should be gone now)
        item1_final_check = await store.aget(namespace, "item1", refresh_ttl=False)
        assert item1_final_check is None, (
            "Item1 should be gone after its refreshed TTL expired"
        )


def _expires_map(store: SqliteStore) -> dict[str, str]:
    """Return a {key: expires_at} map for direct TTL comparisons."""
    return dict(store.conn.execute("SELECT key, expires_at FROM store"))


async def _aexpires_map(store: AsyncSqliteStore) -> dict[str, str]:
    """Return a {key: expires_at} map for direct TTL comparisons."""
    async with store.conn.execute("SELECT key, expires_at FROM store") as cur:
        return dict(await cur.fetchall())


def test_batch_get_respects_per_op_refresh_ttl(temp_db_file: str) -> None:
    """A GetOp with refresh_ttl=False must not refresh siblings in the same batch."""
    with SqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": 60, "refresh_on_read": True}
    ) as store:
        store.setup()
        store.put(("docs",), "refresh", {"k": 1})
        store.put(("docs",), "keep", {"k": 2})

        # Far-future baseline so a refresh (now + ttl) is clearly distinguishable.
        store.conn.execute(
            "UPDATE store SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+1 day')"
        )
        before = _expires_map(store)

        store.batch(
            [
                GetOp(("docs",), "refresh", refresh_ttl=True),
                GetOp(("docs",), "keep", refresh_ttl=False),
            ]
        )

        after = _expires_map(store)
        assert after["refresh"] != before["refresh"], "refresh=True record must refresh"
        assert after["keep"] == before["keep"], "refresh=False record must not refresh"


def test_get_refreshes_by_default_without_refresh_on_read(
    temp_db_file: str,
) -> None:
    """refresh_on_read defaults to True, so reads refresh by default."""
    with SqliteStore.from_conn_string(temp_db_file, ttl={"default_ttl": 60}) as store:
        store.setup()
        store.put(("docs",), "key1", {"k": 1})
        store.conn.execute(
            "UPDATE store SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+1 day')"
        )
        before = _expires_map(store)

        assert store.get(("docs",), "key1") is not None

        after = _expires_map(store)
        assert after["key1"] != before["key1"], "default read should refresh the TTL"


def test_get_explicit_refresh_ttl_overrides_store_default(
    temp_db_file: str,
) -> None:
    """An explicit refresh_ttl=True overrides a store-level refresh_on_read=False."""
    with SqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": 60, "refresh_on_read": False}
    ) as store:
        store.setup()
        store.put(("docs",), "key1", {"k": 1})
        store.conn.execute(
            "UPDATE store SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+1 day')"
        )
        baseline = _expires_map(store)

        # Store default is False: no flag means no refresh.
        assert store.get(("docs",), "key1") is not None
        assert _expires_map(store)["key1"] == baseline["key1"]

        # Per-operation override opts back in.
        assert store.get(("docs",), "key1", refresh_ttl=True) is not None
        after = _expires_map(store)
        assert after["key1"] != baseline["key1"]


def test_search_refreshes_by_default_without_refresh_on_read(
    temp_db_file: str,
) -> None:
    """Search refreshes returned items by default (refresh_on_read defaults to True)."""
    with SqliteStore.from_conn_string(temp_db_file, ttl={"default_ttl": 60}) as store:
        store.setup()
        store.put(("docs",), "k1", {"group": "a"})
        store.put(("docs",), "k2", {"group": "b"})
        store.conn.execute(
            "UPDATE store SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+1 day')"
        )
        baseline = _expires_map(store)

        results = store.search(("docs",), filter={"group": "a"}, limit=10)
        assert [item.key for item in results] == ["k1"]

        after = _expires_map(store)
        assert after["k1"] != baseline["k1"], "matched item should refresh"
        assert after["k2"] == baseline["k2"], "unmatched item must not refresh"


def test_search_refresh_ttl_overrides_store_default(temp_db_file: str) -> None:
    """Search honors an explicit refresh_ttl=True over a False store default."""
    with SqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": 60, "refresh_on_read": False}
    ) as store:
        store.setup()
        store.put(("docs",), "k1", {"group": "a"})
        store.put(("docs",), "k2", {"group": "b"})
        store.conn.execute(
            "UPDATE store SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+1 day')"
        )
        baseline = _expires_map(store)

        # Store default is False: no flag means no refresh.
        assert [i.key for i in store.search(("docs",), filter={"group": "a"})] == ["k1"]
        assert _expires_map(store)["k1"] == baseline["k1"]

        # Per-operation override opts back in.
        assert [
            i.key
            for i in store.search(("docs",), filter={"group": "a"}, refresh_ttl=True)
        ] == ["k1"]
        after = _expires_map(store)
        assert after["k1"] != baseline["k1"]
        assert after["k2"] == baseline["k2"]


@pytest.mark.asyncio
async def test_async_batch_get_respects_per_op_refresh_ttl(
    temp_db_file: str,
) -> None:
    """Async batch GET must not refresh records whose op opted out."""
    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": 60, "refresh_on_read": True}
    ) as store:
        await store.setup()
        await store.aput(("docs",), "refresh", {"k": 1})
        await store.aput(("docs",), "keep", {"k": 2})
        await store.conn.execute(
            "UPDATE store SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+1 day')"
        )
        before = await _aexpires_map(store)

        await store.abatch(
            [
                GetOp(("docs",), "refresh", refresh_ttl=True),
                GetOp(("docs",), "keep", refresh_ttl=False),
            ]
        )

        after = await _aexpires_map(store)
        assert after["refresh"] != before["refresh"]
        assert after["keep"] == before["keep"]


@pytest.mark.asyncio
async def test_async_aget_defaults_to_refresh_without_refresh_on_read(
    temp_db_file: str,
) -> None:
    """Async reads refresh by default when refresh_on_read is unset."""
    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": 60}
    ) as store:
        await store.setup()
        await store.aput(("docs",), "key1", {"k": 1})
        await store.conn.execute(
            "UPDATE store SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+1 day')"
        )
        before = await _aexpires_map(store)

        assert await store.aget(("docs",), "key1") is not None

        after = await _aexpires_map(store)
        assert after["key1"] != before["key1"]


@pytest.mark.asyncio
async def test_async_aget_explicit_refresh_ttl_overrides_store_default(
    temp_db_file: str,
) -> None:
    """Explicit refresh_ttl=True overrides a False store default in async reads."""
    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": 60, "refresh_on_read": False}
    ) as store:
        await store.setup()
        await store.aput(("docs",), "key1", {"k": 1})
        await store.conn.execute(
            "UPDATE store SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+1 day')"
        )
        baseline = await _aexpires_map(store)

        assert await store.aget(("docs",), "key1") is not None
        assert (await _aexpires_map(store))["key1"] == baseline["key1"]

        assert await store.aget(("docs",), "key1", refresh_ttl=True) is not None
        after = await _aexpires_map(store)
        assert after["key1"] != baseline["key1"]


@pytest.mark.asyncio
async def test_async_search_refresh_ttl_overrides_store_default(
    temp_db_file: str,
) -> None:
    """Async search honors an explicit refresh_ttl=True over a False store default."""
    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": 60, "refresh_on_read": False}
    ) as store:
        await store.setup()
        await store.aput(("docs",), "k1", {"group": "a"})
        await store.aput(("docs",), "k2", {"group": "b"})
        await store.conn.execute(
            "UPDATE store SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+1 day')"
        )
        baseline = await _aexpires_map(store)

        assert [
            i.key for i in await store.asearch(("docs",), filter={"group": "a"})
        ] == ["k1"]
        assert (await _aexpires_map(store))["k1"] == baseline["k1"]

        assert [
            i.key
            for i in await store.asearch(
                ("docs",), filter={"group": "a"}, refresh_ttl=True
            )
        ] == ["k1"]
        after = await _aexpires_map(store)
        assert after["k1"] != baseline["k1"]
        assert after["k2"] == baseline["k2"]
