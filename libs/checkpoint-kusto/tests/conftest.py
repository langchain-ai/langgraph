"""Test configuration and fixtures for Kusto checkpointer tests."""

import os
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

# Test configuration
KUSTO_CLUSTER_URI = os.getenv(
    "KUSTO_CLUSTER_URI",
    "https://test-cluster.eastus.kusto.windows.net",
)
KUSTO_DATABASE = os.getenv("KUSTO_DATABASE", "test_langgraph")
KUSTO_RUN_INTEGRATION = os.getenv("KUSTO_RUN_INTEGRATION", "false").lower() == "true"


@pytest.fixture
def test_data() -> dict[str, Any]:
    """Fixture providing test data for checkpoint tests."""
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
        "score": 1,
    }
    metadata_2: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "score": None,
    }
    metadata_3: CheckpointMetadata = {}

    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


@pytest.fixture
def unique_database() -> str:
    """Generate a unique database name for isolated testing."""
    return f"test_{uuid4().hex[:16]}"


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requires live Kusto)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (no external dependencies)"
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Skip integration tests unless KUSTO_RUN_INTEGRATION is set."""
    if KUSTO_RUN_INTEGRATION:
        return
    
    skip_integration = pytest.mark.skip(
        reason="Integration tests disabled. Set KUSTO_RUN_INTEGRATION=true to run."
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
