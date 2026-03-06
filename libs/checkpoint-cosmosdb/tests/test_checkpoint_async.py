# type: ignore
"""Async tests for CosmosDB checkpoint implementation."""

import os
from collections.abc import AsyncIterator

import pytest

from langgraph.checkpoint.cosmosdb import AsyncCosmosDBSaver

# Skip tests if CosmosDB credentials are not available
pytestmark = pytest.mark.skipif(
    not os.getenv("COSMOSDB_ENDPOINT"),
    reason="COSMOSDB_ENDPOINT environment variable not set",
)


@pytest.fixture
async def async_cosmosdb_saver() -> AsyncIterator[AsyncCosmosDBSaver]:
    """Create an AsyncCosmosDBSaver instance for testing."""
    endpoint = os.getenv("COSMOSDB_ENDPOINT")
    key = os.getenv("COSMOSDB_KEY")
    database_name = os.getenv("COSMOSDB_TEST_DATABASE", "test_langgraph")
    container_name = os.getenv("COSMOSDB_TEST_CONTAINER", "test_checkpoints")

    async with AsyncCosmosDBSaver.from_conn_info(
        endpoint=endpoint,
        key=key,
        database_name=database_name,
        container_name=container_name,
    ) as saver:
        yield saver


@pytest.mark.asyncio
async def test_async_put_and_get(async_cosmosdb_saver: AsyncCosmosDBSaver) -> None:
    """Test async put and get operations."""
    config = {
        "configurable": {
            "thread_id": "test_thread_async",
            "checkpoint_ns": "",
            "checkpoint_id": "test_checkpoint_async",
        }
    }

    checkpoint = {
        "v": 1,
        "id": "test_checkpoint_async",
        "ts": "2024-01-01T00:00:00.000000+00:00",
        "channel_values": {"async_key": "async_value"},
        "channel_versions": {"async_key": 1},
        "versions_seen": {},
        "pending_sends": [],
    }

    metadata = {"source": "async_test"}

    # Async put
    result_config = await async_cosmosdb_saver.aput(config, checkpoint, metadata, {})
    assert result_config["configurable"]["thread_id"] == "test_thread_async"

    # Async get
    retrieved = await async_cosmosdb_saver.aget_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["id"] == "test_checkpoint_async"


@pytest.mark.asyncio
async def test_async_list(async_cosmosdb_saver: AsyncCosmosDBSaver) -> None:
    """Test async list operation."""
    thread_id = "test_thread_async_list"

    # Create checkpoints
    for i in range(2):
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": f"async_checkpoint_{i}",
            }
        }
        checkpoint = {
            "v": 1,
            "id": f"async_checkpoint_{i}",
            "ts": f"2024-01-01T00:00:0{i}.000000+00:00",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        await async_cosmosdb_saver.aput(config, checkpoint, {}, {})

    # Async list
    list_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoints = []
    async for checkpoint_tuple in async_cosmosdb_saver.alist(list_config):
        checkpoints.append(checkpoint_tuple)

    assert len(checkpoints) >= 2
