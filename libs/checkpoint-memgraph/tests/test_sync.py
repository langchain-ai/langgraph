import socket
from collections.abc import Iterator
from typing import Any
from urllib.parse import unquote, urlparse

import pytest
from langchain_core.runnables import RunnableConfig
from neo4j import GraphDatabase

from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.memgraph import MemgraphSaver
from langgraph.checkpoint.serde.types import TASKS
from tests.conftest import DEFAULT_MEMGRAPH_URI


def is_memgraph_unavailable() -> bool:
    """
    Check if a Memgraph instance is unavailable.

    Returns:
        bool: True if a Memgraph instance is not available, False otherwise.
    """
    try:
        # Try to create a connection to the default Memgraph port.
        parsed_uri = urlparse(DEFAULT_MEMGRAPH_URI)
        if parsed_uri.port is None:
            return True
        with socket.create_connection(
            (parsed_uri.hostname, parsed_uri.port), timeout=1
        ):
            return False
    except (socket.timeout, ConnectionRefusedError):
        return True


# Skip all tests in this module if Memgraph is not available.
pytestmark = pytest.mark.skipif(
    is_memgraph_unavailable(), reason="Memgraph instance not available"
)


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Exclude private keys from config before comparing with metadata."""
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


@pytest.fixture
def saver() -> Iterator[MemgraphSaver]:
    """
    Fixture for MemgraphSaver testing.
    Connects to Memgraph, sets up the checkpointer, yields it to the test,
    and cleans up the database afterwards.
    """
    # Parse the connection string to separate URI from credentials
    parsed = urlparse(DEFAULT_MEMGRAPH_URI)
    uri = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 7687}"
    auth = (unquote(parsed.username or ""), unquote(parsed.password or ""))
    # Create the driver with URI and auth details separated
    driver = GraphDatabase.driver(uri, auth=auth)
    # Run setup migrations in auto-commit transactions
    with driver.session() as session:
        for migration in MemgraphSaver.MIGRATIONS:
            if not migration.startswith("//"):
                session.run(migration)
    try:
        # Pass the driver to the checkpointer
        checkpointer = MemgraphSaver(driver)
        # The setup is now done outside, so we don't call checkpointer.setup() here
        yield checkpointer
    finally:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()


@pytest.fixture
def test_data() -> dict[str, Any]:
    """Fixture providing common test data for checkpoint tests."""
    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_id": "1",
            "checkpoint_ns": "",
        }
    }
    config_2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    config_3: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2-inner",
            "checkpoint_ns": "inner",
        }
    }
    chkpnt_1: Checkpoint = empty_checkpoint()
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()
    metadata_1: CheckpointMetadata = {
        "source": "input",
        "step": 2,
        "writes": {},
        "score": 1,
    }
    metadata_2: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "writes": {"foo": "bar"},
        "score": None,
    }
    metadata_3: CheckpointMetadata = {}
    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


def test_combined_metadata(saver: MemgraphSaver, test_data: dict[str, Any]) -> None:
    """
    Tests that metadata from the config and the put call are correctly combined.
    """
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_ns": "",
            "__super_private_key": "super_private_value",
        },
        "metadata": {"run_id": "my_run_id"},
    }
    chkpnt: Checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
    metadata: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "writes": {"foo": "bar"},
        "score": None,
    }
    saved_config = saver.put(config, chkpnt, metadata, {})
    checkpoint = saver.get_tuple(saved_config)
    assert checkpoint is not None
    assert checkpoint.metadata == {
        **metadata,
        "run_id": "my_run_id",
    }


def test_search(saver: MemgraphSaver, test_data: dict[str, Any]) -> None:
    """
    Tests searching for checkpoints using metadata and config filters.
    """
    configs = test_data["configs"]
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]
    saver.put(configs[0], checkpoints[0], metadata[0], {})
    saver.put(configs[1], checkpoints[1], metadata[1], {})
    saver.put(configs[2], checkpoints[2], metadata[2], {})
    # Define metadata filter queries
    query_1 = {"source": "input"}  # search by 1 key
    query_2 = {"step": 1, "writes": {"foo": "bar"}}  # search by multiple keys
    query_3: dict[str, Any] = {}  # empty filter, returns all
    query_4 = {"source": "update", "step": 1}  # no match
    # Test metadata filters
    search_results_1 = list(saver.list(None, filter=query_1))
    assert len(search_results_1) == 1
    assert search_results_1[0].metadata == metadata[0]
    search_results_2 = list(saver.list(None, filter=query_2))
    assert len(search_results_2) == 1
    assert search_results_2[0].metadata == metadata[1]
    search_results_3 = list(saver.list(None, filter=query_3))
    assert len(search_results_3) == 3
    search_results_4 = list(saver.list(None, filter=query_4))
    assert len(search_results_4) == 0
    # Test search by config (thread_id)
    search_results_5 = list(saver.list({"configurable": {"thread_id": "thread-2"}}))
    assert len(search_results_5) == 2
    checkpoint_ns_set = {
        res.config["configurable"]["checkpoint_ns"] for res in search_results_5
    }
    assert checkpoint_ns_set == {"", "inner"}


def test_nonnull_migrations() -> None:
    """
    Ensures that all defined migration strings are valid statements.
    """
    for migration in MemgraphSaver.MIGRATIONS:
        # no-op migrations are commented out, skip them
        if not migration.startswith("//"):
            assert migration.strip()


def test_pending_sends_migration(saver: MemgraphSaver) -> None:
    """
    Tests backward compatibility for migrating pending sends.
    Checks that writes to the TASKS channel are correctly associated
    with the *next* checkpoint.
    """
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
        }
    }
    # Create the first checkpoint
    checkpoint_0 = empty_checkpoint()
    config = saver.put(config, checkpoint_0, {}, {})
    # Put some pending sends linked to the first checkpoint
    saver.put_writes(config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1")
    saver.put_writes(config, [(TASKS, "send-3")], task_id="task-2")
    # Check that fetching checkpoint_0 directly does not include pending sends
    tuple_0 = saver.get_tuple(config)
    assert tuple_0 is not None
    assert tuple_0.checkpoint["channel_values"] == {}
    assert tuple_0.checkpoint["channel_versions"] == {}
    # Create the second checkpoint
    checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
    config = saver.put(config, checkpoint_1, {}, {})
    # Check that pending sends are now attached to checkpoint_1
    tuple_1 = saver.get_tuple(config)
    assert tuple_1 is not None
    assert tuple_1.checkpoint["channel_values"] == {
        TASKS: ["send-1", "send-2", "send-3"]
    }
    assert TASKS in tuple_1.checkpoint["channel_versions"]
    # Check that listing checkpoints also applies the migration correctly
    search_results = list(saver.list({"configurable": {"thread_id": "thread-1"}}))
    assert len(search_results) == 2
    # The newer checkpoint (checkpoint_1) should have the migrated sends
    assert (
        search_results[0].config["configurable"]["checkpoint_id"] == checkpoint_1["id"]
    )
    assert search_results[0].checkpoint["channel_values"] == {
        TASKS: ["send-1", "send-2", "send-3"]
    }
    assert TASKS in search_results[0].checkpoint["channel_versions"]
    # The older checkpoint (checkpoint_0) should not have them
    assert (
        search_results[1].config["configurable"]["checkpoint_id"] == checkpoint_0["id"]
    )
    assert search_results[1].checkpoint["channel_values"] == {}
    assert search_results[1].checkpoint["channel_versions"] == {}
