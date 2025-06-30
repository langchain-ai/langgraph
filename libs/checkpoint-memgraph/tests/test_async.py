import socket
from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import unquote, urlparse

import pytest
from langchain_core.runnables import RunnableConfig
from neo4j import AsyncGraphDatabase

from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.memgraph.aio import AsyncMemgraphSaver
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
async def saver() -> AsyncGenerator[AsyncMemgraphSaver, None]:
    """Fixture for AsyncMemgraphSaver testing."""
    # Parse the connection string to separate URI from credentials
    parsed = urlparse(DEFAULT_MEMGRAPH_URI)
    uri = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 7687}"
    auth = (unquote(parsed.username or ""), unquote(parsed.password or ""))
    driver = AsyncGraphDatabase.driver(uri, auth=auth)
    checkpointer = AsyncMemgraphSaver(driver)
    await checkpointer.setup()
    try:
        yield checkpointer
    finally:
        # Clean up the database after the test
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        await driver.close()


@pytest.fixture
def test_data() -> dict[str, Any]:
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


@pytest.mark.asyncio
async def test_combined_metadata(
    saver: AsyncMemgraphSaver, test_data: dict[str, Any]
) -> None:
    config: Any = {
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
    saved_config = await saver.aput(config, chkpnt, metadata, {})
    checkpoint = await saver.aget_tuple(saved_config)
    assert checkpoint is not None
    # The get_checkpoint_metadata helper correctly excludes dunder keys.
    # The assertion should only check for the combination of public metadata.
    assert checkpoint.metadata == {
        **config.get("metadata", {}),
        **metadata,
    }


@pytest.mark.asyncio
async def test_alist(saver: AsyncMemgraphSaver, test_data: dict[str, Any]) -> None:
    configs = test_data["configs"]
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]
    await saver.aput(configs[0], checkpoints[0], metadata[0], {})
    await saver.aput(configs[1], checkpoints[1], metadata[1], {})
    await saver.aput(configs[2], checkpoints[2], metadata[2], {})
    query_1 = {"source": "input"}
    query_2 = {"step": 1, "writes": {"foo": "bar"}}
    query_3: dict[str, Any] = {}
    query_4 = {"source": "update", "step": 1}
    search_results_1 = [c async for c in saver.alist(None, filter=query_1)]
    assert len(search_results_1) == 1
    assert search_results_1[0].metadata == metadata[0]
    search_results_2 = [c async for c in saver.alist(None, filter=query_2)]
    assert len(search_results_2) == 1
    assert search_results_2[0].metadata == metadata[1]
    search_results_3 = [c async for c in saver.alist(None, filter=query_3)]
    assert len(search_results_3) == 3
    search_results_4 = [c async for c in saver.alist(None, filter=query_4)]
    assert len(search_results_4) == 0
    search_results_5 = [
        c async for c in saver.alist({"configurable": {"thread_id": "thread-2"}})
    ]
    assert len(search_results_5) == 2
    checkpoint_ns_set = {
        res.config["configurable"]["checkpoint_ns"] for res in search_results_5
    }
    assert checkpoint_ns_set == {"", "inner"}


@pytest.mark.asyncio
async def test_pending_sends_migration(saver: AsyncMemgraphSaver) -> None:
    config: Any = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
        }
    }
    checkpoint_0 = empty_checkpoint()
    config = await saver.aput(config, checkpoint_0, {}, {})
    await saver.aput_writes(
        config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1"
    )
    await saver.aput_writes(config, [(TASKS, "send-3")], task_id="task-2")
    tuple_0 = await saver.aget_tuple(config)
    assert tuple_0 is not None
    assert tuple_0.checkpoint["channel_values"] == {}
    assert tuple_0.checkpoint["channel_versions"] == {}
    checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
    config = await saver.aput(config, checkpoint_1, {}, {})
    tuple_1 = await saver.aget_tuple(config)
    assert tuple_1 is not None
    assert tuple_1.checkpoint["channel_values"] == {
        TASKS: ["send-1", "send-2", "send-3"]
    }
    assert TASKS in tuple_1.checkpoint["channel_versions"]
    search_results = [
        c async for c in saver.alist({"configurable": {"thread_id": "thread-1"}})
    ]
    assert len(search_results) == 2
    assert (
        search_results[0].config["configurable"]["checkpoint_id"] == checkpoint_1["id"]
    )
    assert search_results[0].checkpoint["channel_values"] == {
        TASKS: ["send-1", "send-2", "send-3"]
    }
    assert TASKS in search_results[0].checkpoint["channel_versions"]
    assert (
        search_results[1].config["configurable"]["checkpoint_id"] == checkpoint_0["id"]
    )
    assert search_results[1].checkpoint["channel_values"] == {}
    assert search_results[1].checkpoint["channel_versions"] == {}
