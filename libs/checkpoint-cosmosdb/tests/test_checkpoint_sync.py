# type: ignore
"""Sync tests for CosmosDB checkpoint implementation."""

import os
from collections.abc import Iterator

import pytest

from langgraph.checkpoint.cosmosdb import CosmosDBSaver

# Skip tests if CosmosDB credentials are not available
pytestmark = pytest.mark.skipif(
    not os.getenv("COSMOSDB_ENDPOINT"),
    reason="COSMOSDB_ENDPOINT environment variable not set",
)


@pytest.fixture
def cosmosdb_saver() -> Iterator[CosmosDBSaver]:
    """Create a CosmosDBSaver instance for testing."""
    database_name = os.getenv("COSMOSDB_TEST_DATABASE", "test_langgraph")
    container_name = os.getenv("COSMOSDB_TEST_CONTAINER", "test_checkpoints")

    saver = CosmosDBSaver(
        database_name=database_name,
        container_name=container_name,
    )

    yield saver


def test_cosmosdb_saver_init(cosmosdb_saver: CosmosDBSaver) -> None:
    """Test that CosmosDBSaver initializes correctly."""
    assert cosmosdb_saver is not None
    assert cosmosdb_saver.container is not None


def test_put_and_get_checkpoint(cosmosdb_saver: CosmosDBSaver) -> None:
    """Test basic put and get operations."""
    config = {
        "configurable": {
            "thread_id": "test_thread_1",
            "checkpoint_ns": "",
            "checkpoint_id": "test_checkpoint_1",
        }
    }

    checkpoint = {
        "v": 1,
        "id": "test_checkpoint_1",
        "ts": "2024-01-01T00:00:00.000000+00:00",
        "channel_values": {"test_key": "test_value"},
        "channel_versions": {"test_key": 1},
        "versions_seen": {},
        "pending_sends": [],
    }

    metadata = {"source": "test", "step": 1}

    # Put checkpoint
    result_config = cosmosdb_saver.put(config, checkpoint, metadata, {})
    assert result_config["configurable"]["thread_id"] == "test_thread_1"

    # Get checkpoint
    retrieved = cosmosdb_saver.get_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["id"] == "test_checkpoint_1"
    assert retrieved.metadata["source"] == "test"


def test_list_checkpoints(cosmosdb_saver: CosmosDBSaver) -> None:
    """Test listing checkpoints."""
    thread_id = "test_thread_list"

    # Create multiple checkpoints
    for i in range(3):
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": f"checkpoint_{i}",
            }
        }
        checkpoint = {
            "v": 1,
            "id": f"checkpoint_{i}",
            "ts": f"2024-01-01T00:00:0{i}.000000+00:00",
            "channel_values": {"step": i},
            "channel_versions": {"step": i},
            "versions_seen": {},
            "pending_sends": [],
        }
        cosmosdb_saver.put(config, checkpoint, {"step": i}, {})

    # List checkpoints
    list_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoints = list(cosmosdb_saver.list(list_config))

    assert len(checkpoints) >= 3
