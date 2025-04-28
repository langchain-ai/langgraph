"""
Tests for the async database operations of the PersonaMemory class.

This module contains tests for asynchronously saving and loading PersonaMemory state to/from a database.
"""

import os
import tempfile

import pytest
import pytest_asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.experimental.memory.persona_memory import PersonaMemory


@pytest.fixture
def db_file():
    """Create a temporary SQLite database file."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        return tmp.name


@pytest_asyncio.fixture
async def saver(db_file):
    """Create an AsyncSqliteSaver instance with a temporary database."""
    async with AsyncSqliteSaver.from_conn_string(db_file) as saver:
        # Setup the database tables
        await saver.setup()
        yield saver


@pytest.fixture
def config():
    """Create a test configuration."""
    return {
        "configurable": {
            "thread_id": "test-thread-1",
            "checkpoint_ns": "",  # Optional field for checkpoint namespace; blank for tests
        }
    }


@pytest.mark.asyncio
async def test_save_and_load_persona_memory_async(saver, config):
    """Test async saving and loading PersonaMemory to/from database."""
    # Create initial memory and add some data
    memory = PersonaMemory()
    memory.update_memory("John is happy and excited about Paris.")

    # Save to database asynchronously
    await memory.save_to_db_async(saver, config)

    # Load from database asynchronously
    loaded_memory = await PersonaMemory.load_from_db_async(saver, config)

    # Verify loaded data matches original
    assert loaded_memory.traits == memory.traits
    assert len(loaded_memory.associations) == len(memory.associations)
    assert "John" in loaded_memory.associations
    assert "Paris" in loaded_memory.associations
    assert loaded_memory.associations["John"].entity_type == "person"
    assert loaded_memory.associations["Paris"].entity_type == "place"
    assert "positive" in loaded_memory.associations["John"].traits


@pytest.mark.asyncio
async def test_save_and_load_empty_memory_async(saver, config):
    """Test async saving and loading an empty PersonaMemory."""
    # Create empty memory
    memory = PersonaMemory()

    # Save to database asynchronously
    await memory.save_to_db_async(saver, config)

    # Load from database asynchronously
    loaded_memory = await PersonaMemory.load_from_db_async(saver, config)

    # Verify loaded memory is empty
    assert loaded_memory.traits == []
    assert loaded_memory.associations == {}


@pytest.mark.asyncio
async def test_save_and_load_complex_memory_async(saver, config):
    """Test async saving and loading a complex PersonaMemory with multiple entities and traits."""
    # Create memory with complex data
    memory = PersonaMemory()
    memory.update_memory("John is happy and excited about Paris.")
    memory.update_memory("Mary is analytical and logical in London.")
    memory.update_memory("John and Mary are friendly with each other.")

    # Save to database asynchronously
    await memory.save_to_db_async(saver, config)

    # Load from database asynchronously
    loaded_memory = await PersonaMemory.load_from_db_async(saver, config)

    # Verify all data was preserved
    assert set(loaded_memory.traits) == {"positive", "analytical", "friendly"}
    assert len(loaded_memory.associations) == 4  # John, Mary, Paris, London

    # Verify specific associations
    assert "positive" in loaded_memory.associations["John"].traits
    assert "analytical" in loaded_memory.associations["Mary"].traits
    assert "friendly" in loaded_memory.associations["John"].traits
    assert "friendly" in loaded_memory.associations["Mary"].traits


@pytest.mark.asyncio
async def test_concurrent_operations(saver, config):
    """Test concurrent async operations on PersonaMemory."""
    # Create two different memories
    memory1 = PersonaMemory()
    memory1.update_memory("John is happy in Paris.")

    memory2 = PersonaMemory()
    memory2.update_memory("Mary is analytical in London.")

    # Create separate configs for each memory
    config1 = {
        "configurable": {
            "thread_id": "test-thread-1",
            "checkpoint_ns": "",
        }
    }
    config2 = {
        "configurable": {
            "thread_id": "test-thread-2",
            "checkpoint_ns": "",
        }
    }

    # Save both memories concurrently with different configs
    await memory1.save_to_db_async(saver, config1)
    await memory2.save_to_db_async(saver, config2)

    # Load both memories concurrently with their respective configs
    loaded_memory1 = await PersonaMemory.load_from_db_async(saver, config1)
    loaded_memory2 = await PersonaMemory.load_from_db_async(saver, config2)

    # Verify both memories were saved and loaded correctly
    assert "John" in loaded_memory1.associations
    assert "Paris" in loaded_memory1.associations
    assert "Mary" in loaded_memory2.associations
    assert "London" in loaded_memory2.associations


def test_cleanup(db_file):
    """Clean up the temporary database file."""
    if os.path.exists(db_file):
        os.unlink(db_file)
