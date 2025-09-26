"""Test SQLite store Time-To-Live (TTL) functionality."""

import asyncio
import os
import tempfile
import time
from collections.abc import Generator

import pytest
from langgraph.store.base import TTLConfig

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

        # Short TTL item should be gone, long TTL item should remain
        item1 = store.get(("test",), "item1")
        item2 = store.get(("test",), "item2")
        assert item1 is None
        assert item2 is not None

        # Wait for the second item's TTL
        time.sleep(4)
        store.sweep_ttl()

        # Now both should be gone
        item2 = store.get(("test",), "item2")
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

        # Check results
        item1 = store.get(("test",), "item1")
        item2 = store.get(("test",), "item2")
        item3 = store.get(("test",), "item3")

        assert item1 is None  # Should be expired
        assert item2 is not None  # Default TTL, should still be there
        assert item3 is not None  # No TTL, should still be there

        # Wait for default TTL to expire
        time.sleep(4)
        store.sweep_ttl()

        # Check results again
        item2 = store.get(("test",), "item2")
        item3 = store.get(("test",), "item3")

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
