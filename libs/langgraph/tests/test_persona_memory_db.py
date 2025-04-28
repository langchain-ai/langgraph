"""
Tests for the PersonaMemory class database operations.

This module contains tests for saving and loading PersonaMemory state to/from a database.
"""

import os
import tempfile

import pytest

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.experimental.memory.persona_memory import PersonaMemory


@pytest.fixture
def db_file():
    """Create a temporary SQLite database file."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        return tmp.name


@pytest.fixture
def saver(db_file):
    """Create a SqliteSaver instance with a temporary database."""
    with SqliteSaver.from_conn_string(db_file) as saver:
        # Setup the database tables
        saver.setup()
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


def test_save_and_load_persona_memory(saver, config):
    """Test saving and loading PersonaMemory to/from database."""
    # Create initial memory and add some data
    memory = PersonaMemory()
    memory.update_memory("John is happy and excited about Paris.")

    # Save to database
    memory.save_to_db(saver, config)

    # Load from database
    loaded_memory = PersonaMemory.load_from_db(saver, config)

    # Verify loaded data matches original
    assert loaded_memory.traits == memory.traits
    assert len(loaded_memory.associations) == len(memory.associations)
    assert "John" in loaded_memory.associations
    assert "Paris" in loaded_memory.associations
    assert loaded_memory.associations["John"].entity_type == "person"
    assert loaded_memory.associations["Paris"].entity_type == "place"
    assert "positive" in loaded_memory.associations["John"].traits


def test_save_and_load_empty_memory(saver, config):
    """Test saving and loading an empty PersonaMemory."""
    # Create empty memory
    memory = PersonaMemory()

    # Save to database
    memory.save_to_db(saver, config)

    # Load from database
    loaded_memory = PersonaMemory.load_from_db(saver, config)

    # Verify loaded memory is empty
    assert loaded_memory.traits == []
    assert loaded_memory.associations == {}


def test_save_and_load_complex_memory(saver, config):
    """Test saving and loading a complex PersonaMemory with multiple entities and traits."""
    # Create memory with complex data
    memory = PersonaMemory()
    memory.update_memory("John is happy and excited about Paris.")
    memory.update_memory("Mary is analytical and logical in London.")
    memory.update_memory("John and Mary are friendly with each other.")

    # Save to database
    memory.save_to_db(saver, config)

    # Load from database
    loaded_memory = PersonaMemory.load_from_db(saver, config)

    # Verify all data was preserved
    assert set(loaded_memory.traits) == {"positive", "analytical", "friendly"}
    assert len(loaded_memory.associations) == 4  # John, Mary, Paris, London

    # Verify specific associations
    assert "positive" in loaded_memory.associations["John"].traits
    assert "analytical" in loaded_memory.associations["Mary"].traits
    assert "friendly" in loaded_memory.associations["John"].traits
    assert "friendly" in loaded_memory.associations["Mary"].traits


def test_cleanup(db_file):
    """Clean up the temporary database file."""
    if os.path.exists(db_file):
        os.unlink(db_file)
