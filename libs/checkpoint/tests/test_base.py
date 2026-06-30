"""Tests for langgraph.checkpoint.base utility functions and types."""

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    WRITES_IDX_MAP,
    CheckpointMetadata,
    CheckpointTuple,
    EmptyChannelError,
    copy_checkpoint,
    create_checkpoint,
    empty_checkpoint,
    get_checkpoint_id,
    get_checkpoint_metadata,
    get_serializable_checkpoint_metadata,
)

# --- empty_checkpoint tests ---


def test_empty_checkpoint_structure() -> None:
    """Test empty_checkpoint returns a valid Checkpoint dict."""
    cp = empty_checkpoint()
    assert cp["v"] == 2
    assert isinstance(cp["id"], str)
    assert len(cp["id"]) > 0
    assert isinstance(cp["ts"], str)
    assert cp["channel_values"] == {}
    assert cp["channel_versions"] == {}
    assert cp["versions_seen"] == {}
    assert cp["updated_channels"] is None


def test_empty_checkpoint_unique_ids() -> None:
    """Test each call to empty_checkpoint generates a unique ID."""
    cp1 = empty_checkpoint()
    cp2 = empty_checkpoint()
    assert cp1["id"] != cp2["id"]


def test_empty_checkpoint_has_iso_timestamp() -> None:
    """Test the timestamp is a valid ISO 8601 string."""
    cp = empty_checkpoint()
    # ISO 8601 timestamps contain 'T' separator
    assert "T" in cp["ts"]


# --- copy_checkpoint tests ---


def test_copy_checkpoint_returns_equal_values() -> None:
    """Test copy_checkpoint returns a checkpoint with equal values."""
    original = empty_checkpoint()
    original["channel_values"] = {"messages": ["hello"]}
    original["channel_versions"] = {"messages": "1"}
    original["versions_seen"] = {"node_a": {"messages": "1"}}

    copied = copy_checkpoint(original)

    assert copied["v"] == original["v"]
    assert copied["id"] == original["id"]
    assert copied["ts"] == original["ts"]
    assert copied["channel_values"] == original["channel_values"]
    assert copied["channel_versions"] == original["channel_versions"]
    assert copied["versions_seen"] == original["versions_seen"]


def test_copy_checkpoint_is_deep_copy() -> None:
    """Test that modifying the copy does not affect the original."""
    original = empty_checkpoint()
    original["channel_values"] = {"key": [1, 2, 3]}
    original["channel_versions"] = {"key": "1"}
    original["versions_seen"] = {"node": {"key": "1"}}

    copied = copy_checkpoint(original)

    # Modify the copy
    copied["channel_values"]["new_key"] = "new"
    copied["channel_versions"]["new_key"] = "2"
    copied["versions_seen"]["other_node"] = {"key": "2"}

    # Original should be unchanged
    assert "new_key" not in original["channel_values"]
    assert "new_key" not in original["channel_versions"]
    assert "other_node" not in original["versions_seen"]


def test_copy_checkpoint_copies_inner_dicts() -> None:
    """Test that inner version dicts in versions_seen are also copied."""
    original = empty_checkpoint()
    original["versions_seen"] = {"node_a": {"ch1": "1", "ch2": "2"}}

    copied = copy_checkpoint(original)
    copied["versions_seen"]["node_a"]["ch3"] = "3"

    assert "ch3" not in original["versions_seen"]["node_a"]


# --- create_checkpoint tests ---


def test_create_checkpoint_from_empty() -> None:
    """Test creating a checkpoint from an empty checkpoint."""
    base = empty_checkpoint()
    new_cp = create_checkpoint(base, None, step=1)

    assert new_cp["v"] == 2
    assert isinstance(new_cp["id"], str)
    assert new_cp["id"] != base["id"]
    assert isinstance(new_cp["ts"], str)
    assert new_cp["updated_channels"] is None


def test_create_checkpoint_with_custom_id() -> None:
    """Test creating a checkpoint with a custom ID."""
    base = empty_checkpoint()
    new_cp = create_checkpoint(base, None, step=1, id="custom-id-123")
    assert new_cp["id"] == "custom-id-123"


def test_create_checkpoint_preserves_channel_versions() -> None:
    """Test that create_checkpoint preserves channel versions from base."""
    base = empty_checkpoint()
    base["channel_versions"] = {"messages": "v1", "state": "v2"}
    base["versions_seen"] = {"node_a": {"messages": "v1"}}

    new_cp = create_checkpoint(base, None, step=1)
    assert new_cp["channel_versions"] == {"messages": "v1", "state": "v2"}
    assert new_cp["versions_seen"] == {"node_a": {"messages": "v1"}}


def test_create_checkpoint_with_none_channels_uses_base_values() -> None:
    """Test that None channels preserves existing channel_values."""
    base = empty_checkpoint()
    base["channel_values"] = {"key": "value"}

    new_cp = create_checkpoint(base, None, step=0)
    assert new_cp["channel_values"] == {"key": "value"}


# --- get_checkpoint_id tests ---


def test_get_checkpoint_id_present() -> None:
    """Test extracting checkpoint_id when present."""
    config: RunnableConfig = {
        "configurable": {"checkpoint_id": "abc-123"},
    }
    assert get_checkpoint_id(config) == "abc-123"


def test_get_checkpoint_id_missing() -> None:
    """Test extracting checkpoint_id when not present."""
    config: RunnableConfig = {
        "configurable": {"thread_id": "t1"},
    }
    assert get_checkpoint_id(config) is None


def test_get_checkpoint_id_none_value() -> None:
    """Test extracting checkpoint_id when explicitly set to None."""
    config: RunnableConfig = {
        "configurable": {"checkpoint_id": None},
    }
    assert get_checkpoint_id(config) is None


# --- get_checkpoint_metadata tests ---


def test_get_checkpoint_metadata_basic() -> None:
    """Test basic metadata extraction."""
    config: RunnableConfig = {"configurable": {}}
    metadata: CheckpointMetadata = {"source": "input", "step": 1}

    result = get_checkpoint_metadata(config, metadata)
    assert result["source"] == "input"
    assert result["step"] == 1


def test_get_checkpoint_metadata_strips_null_bytes() -> None:
    """Test that null bytes are stripped from string metadata values."""
    config: RunnableConfig = {"configurable": {}}
    metadata: CheckpointMetadata = {"source": "inp\x00ut", "step": 1}

    result = get_checkpoint_metadata(config, metadata)
    assert result["source"] == "input"


def test_get_checkpoint_metadata_excludes_reserved_keys() -> None:
    """Test that reserved keys from config are not added to metadata."""
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "t1",
            "checkpoint_id": "cp1",
            "checkpoint_ns": "",
        },
        "metadata": {
            "langgraph_step": 5,
            "langgraph_node": "my_node",
        },
    }
    metadata: CheckpointMetadata = {"source": "loop", "step": 0}

    result = get_checkpoint_metadata(config, metadata)
    # Reserved keys should not be added
    assert "thread_id" not in result
    assert "checkpoint_id" not in result
    assert "langgraph_step" not in result
    assert "langgraph_node" not in result


def test_get_checkpoint_metadata_merges_custom_from_config() -> None:
    """Test that custom keys from config metadata are merged."""
    config: RunnableConfig = {
        "configurable": {},
        "metadata": {
            "custom_key": "custom_value",
            "score": 42,
        },
    }
    metadata: CheckpointMetadata = {"source": "input", "step": 0}

    result = get_checkpoint_metadata(config, metadata)
    assert result["custom_key"] == "custom_value"
    assert result["score"] == 42


def test_get_checkpoint_metadata_existing_keys_not_overwritten() -> None:
    """Test that existing metadata keys are not overwritten by config."""
    config: RunnableConfig = {
        "configurable": {},
        "metadata": {
            "source": "should_not_overwrite",
        },
    }
    metadata: CheckpointMetadata = {"source": "input", "step": 0}

    result = get_checkpoint_metadata(config, metadata)
    assert result["source"] == "input"  # Original preserved


def test_get_checkpoint_metadata_ignores_dunder_keys() -> None:
    """Test that keys starting with __ are ignored from config."""
    config: RunnableConfig = {
        "configurable": {
            "__pregel_durability": "async",
            "__pregel_something": True,
        },
    }
    metadata: CheckpointMetadata = {"source": "loop", "step": 1}

    result = get_checkpoint_metadata(config, metadata)
    assert "__pregel_durability" not in result
    assert "__pregel_something" not in result


# --- get_serializable_checkpoint_metadata tests ---


def test_serializable_metadata_strips_writes() -> None:
    """Test that 'writes' key is removed from serializable metadata."""
    config: RunnableConfig = {"configurable": {}}
    metadata: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "writes": {"node_a": {"key": "value"}},
    }

    result = get_serializable_checkpoint_metadata(config, metadata)
    assert "writes" not in result
    assert result["source"] == "loop"
    assert result["step"] == 1


def test_serializable_metadata_without_writes() -> None:
    """Test serializable metadata when there are no writes."""
    config: RunnableConfig = {"configurable": {}}
    metadata: CheckpointMetadata = {"source": "input", "step": 0}

    result = get_serializable_checkpoint_metadata(config, metadata)
    assert result["source"] == "input"


# --- CheckpointTuple tests ---


def test_checkpoint_tuple_creation() -> None:
    """Test CheckpointTuple can be created with minimal args."""
    config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
    cp = empty_checkpoint()
    metadata: CheckpointMetadata = {"source": "input", "step": 0}

    ct = CheckpointTuple(config=config, checkpoint=cp, metadata=metadata)
    assert ct.config == config
    assert ct.checkpoint == cp
    assert ct.metadata == metadata
    assert ct.parent_config is None
    assert ct.pending_writes is None


def test_checkpoint_tuple_with_all_fields() -> None:
    """Test CheckpointTuple with all fields populated."""
    config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
    parent_config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
    cp = empty_checkpoint()
    metadata: CheckpointMetadata = {"source": "loop", "step": 1}
    writes = [("task1", "channel1", "value1")]

    ct = CheckpointTuple(
        config=config,
        checkpoint=cp,
        metadata=metadata,
        parent_config=parent_config,
        pending_writes=writes,
    )
    assert ct.parent_config == parent_config
    assert ct.pending_writes == writes


# --- WRITES_IDX_MAP tests ---


def test_writes_idx_map_values() -> None:
    """Test WRITES_IDX_MAP has the expected negative indices."""
    assert WRITES_IDX_MAP["__error__"] == -1
    assert WRITES_IDX_MAP["__scheduled__"] == -2
    assert WRITES_IDX_MAP["__interrupt__"] == -3
    assert WRITES_IDX_MAP["__resume__"] == -4


def test_writes_idx_map_all_negative() -> None:
    """Test all special write indices are negative."""
    for key, idx in WRITES_IDX_MAP.items():
        assert idx < 0, f"Expected negative index for {key}, got {idx}"


# --- EXCLUDED_METADATA_KEYS tests ---


def test_excluded_metadata_keys_content() -> None:
    """Test EXCLUDED_METADATA_KEYS contains expected keys."""
    assert "thread_id" in EXCLUDED_METADATA_KEYS
    assert "checkpoint_id" in EXCLUDED_METADATA_KEYS
    assert "checkpoint_ns" in EXCLUDED_METADATA_KEYS
    assert "langgraph_step" in EXCLUDED_METADATA_KEYS
    assert "langgraph_node" in EXCLUDED_METADATA_KEYS


# --- EmptyChannelError tests ---


def test_empty_channel_error() -> None:
    """Test EmptyChannelError can be raised and caught."""
    with pytest.raises(EmptyChannelError):
        raise EmptyChannelError()
