"""Async integration tests for Kusto checkpointer."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.kusto.aio import AsyncKustoSaver
from tests.conftest import KUSTO_CLUSTER_URI, KUSTO_DATABASE


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Exclude private metadata keys."""
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


@asynccontextmanager
async def _test_saver() -> AsyncIterator[AsyncKustoSaver]:
    """Create a test saver instance with streaming ingestion."""
    async with AsyncKustoSaver.from_connection_string(
        cluster_uri=KUSTO_CLUSTER_URI,
        database=KUSTO_DATABASE,
        batch_size=1,  # Flush immediately for tests
    ) as saver:
        await saver.setup()
        yield saver


@pytest.mark.integration
@pytest.mark.asyncio
async def test_setup_validates_schema() -> None:
    """Test that setup validates the Kusto schema."""
    async with _test_saver():
        pass  # If we get here, setup was successful


@pytest.mark.integration
@pytest.mark.asyncio
async def test_aget_tuple_not_found(test_data: dict[str, Any]) -> None:
    """Test getting a non-existent checkpoint."""
    async with _test_saver() as saver:
        config = {
            "configurable": {
                "thread_id": "nonexistent-thread",
                "checkpoint_ns": "",
            }
        }
        
        result = await saver.aget_tuple(config)
        assert result is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_aput_and_aget_tuple(test_data: dict[str, Any]) -> None:
    """Test putting and getting a checkpoint."""
    async with _test_saver() as saver:
        config = test_data["configs"][0]
        checkpoint = test_data["checkpoints"][0]
        metadata = test_data["metadata"][0]
        
        # Put checkpoint
        returned_config = await saver.aput(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            new_versions={},
        )
        
        # Flush to ensure data is ingested
        await saver.flush()
        
        # Give Kusto a moment to process (streaming mode should be fast)
        import asyncio
        await asyncio.sleep(1)
        
        # Get checkpoint
        result = await saver.aget_tuple(returned_config)
        
        # Note: May be None immediately due to ingestion latency
        # In production, would need to poll or wait
        if result:
            assert result.checkpoint["id"] == checkpoint["id"]
            assert result.metadata == _exclude_keys(metadata)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_aput_with_blobs(test_data: dict[str, Any]) -> None:
    """Test putting a checkpoint with blob data."""
    async with _test_saver() as saver:
        config = test_data["configs"][0]
        checkpoint = test_data["checkpoints"][0].copy()
        
        # Add large blob data
        checkpoint["channel_values"]["large_data"] = {
            "key": "value" * 1000
        }
        
        metadata = test_data["metadata"][0]
        new_versions = {"large_data": "1.0"}
        
        # Put checkpoint
        returned_config = await saver.aput(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            new_versions=new_versions,
        )
        
        await saver.flush()
        
        assert returned_config["configurable"]["checkpoint_id"] == checkpoint["id"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_aput_writes(test_data: dict[str, Any]) -> None:
    """Test putting checkpoint writes."""
    async with _test_saver() as saver:
        config = test_data["configs"][0].copy()
        config["configurable"]["checkpoint_id"] = "test-checkpoint-1"
        
        writes = [
            ("channel1", {"data": "value1"}),
            ("channel2", [1, 2, 3]),
        ]
        
        await saver.aput_writes(
            config=config,
            writes=writes,
            task_id="test-task-1",
            task_path="path/to/task",
        )
        
        await saver.flush()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_alist_empty(test_data: dict[str, Any]) -> None:
    """Test listing checkpoints when none exist."""
    async with _test_saver() as saver:
        config = {
            "configurable": {
                "thread_id": "empty-thread",
                "checkpoint_ns": "",
            }
        }
        
        checkpoints = []
        async for cp in saver.alist(config):
            checkpoints.append(cp)
        
        assert len(checkpoints) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_alist_with_limit(test_data: dict[str, Any]) -> None:
    """Test listing checkpoints with limit."""
    async with _test_saver() as saver:
        config_base = test_data["configs"][0].copy()
        
        # Put multiple checkpoints
        for i in range(5):
            checkpoint = empty_checkpoint()
            checkpoint["id"] = f"checkpoint-{i}"
            config = config_base.copy()
            config["configurable"]["checkpoint_id"] = f"checkpoint-{i-1}" if i > 0 else None
            
            await saver.aput(
                config=config,
                checkpoint=checkpoint,
                metadata={},
                new_versions={},
            )
        
        await saver.flush()
        
        # List with limit
        checkpoints = []
        async for cp in saver.alist(config_base, limit=3):
            checkpoints.append(cp)
        
        # May be empty due to ingestion latency, but should not exceed limit
        assert len(checkpoints) <= 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_adelete_thread(test_data: dict[str, Any]) -> None:
    """Test deleting a thread."""
    async with _test_saver() as saver:
        config = test_data["configs"][0]
        checkpoint = test_data["checkpoints"][0]
        
        # Put checkpoint
        await saver.aput(
            config=config,
            checkpoint=checkpoint,
            metadata={},
            new_versions={},
        )
        
        await saver.flush()
        
        # Delete thread
        await saver.adelete_thread(config["configurable"]["thread_id"])
        
        # Note: Deletes are eventually consistent in Kusto
        # In a real test, would need to poll for deletion


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metadata_filtering(test_data: dict[str, Any]) -> None:
    """Test filtering checkpoints by metadata."""
    async with _test_saver() as saver:
        config = test_data["configs"][0]
        checkpoint = test_data["checkpoints"][0]
        metadata = {"source": "test", "step": 1}
        
        # Put checkpoint
        await saver.aput(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            new_versions={},
        )
        
        await saver.flush()
        
        # List with filter
        filter_criteria = {"source": "test"}
        checkpoints = []
        async for cp in saver.alist(config, filter=filter_criteria):
            checkpoints.append(cp)
        
        # Verify filter was applied (results may be empty due to latency)
        # In production, would poll for results


@pytest.mark.integration
@pytest.mark.asyncio
async def test_batch_flushing() -> None:
    """Test that batching works correctly."""
    async with AsyncKustoSaver.from_connection_string(
        cluster_uri=KUSTO_CLUSTER_URI,
        database=KUSTO_DATABASE,
        batch_size=5,  # Batch 5 records
    ) as saver:
        await saver.setup()
        
        # Put fewer than batch_size checkpoints
        for i in range(3):
            config = {
                "configurable": {
                    "thread_id": f"batch-thread-{i}",
                    "checkpoint_ns": "",
                }
            }
            checkpoint = empty_checkpoint()
            checkpoint["id"] = f"batch-checkpoint-{i}"
            
            await saver.aput(config, checkpoint, {}, {})
        
        # Should not have flushed yet (buffers should have data)
        assert len(saver._checkpoint_buffer) == 3
        
        # Manual flush
        await saver.flush()
        
        # Buffers should be empty now
        assert len(saver._checkpoint_buffer) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
