"""Unit tests for Kusto checkpointer."""

import pytest
from langgraph.checkpoint.base import empty_checkpoint, create_checkpoint
from langgraph.checkpoint.kusto.base import BaseKustoSaver


@pytest.mark.unit
class TestBaseKustoSaver:
    """Test the base Kusto saver functionality."""

    def test_get_next_version_from_none(self) -> None:
        """Test version generation from None."""
        saver = BaseKustoSaver()
        version = saver.get_next_version(None, None)
        assert version.startswith("1.")
        parts = version.split(".")
        assert len(parts) == 2
        assert int(parts[0]) == 1
        assert 0 <= float(parts[1]) < 1

    def test_get_next_version_increment(self) -> None:
        """Test version increment."""
        saver = BaseKustoSaver()
        v1 = saver.get_next_version(None, None)
        v2 = saver.get_next_version(v1, None)
        
        v1_major = int(v1.split(".")[0])
        v2_major = int(v2.split(".")[0])
        assert v2_major == v1_major + 1

    def test_get_next_version_from_int(self) -> None:
        """Test version generation from int."""
        saver = BaseKustoSaver()
        version = saver.get_next_version(5, None)
        assert version.startswith("6.")

    def test_load_blobs_empty(self) -> None:
        """Test loading empty blob list."""
        saver = BaseKustoSaver()
        result = saver._load_blobs([])
        assert result == {}

    def test_dump_blobs_empty(self) -> None:
        """Test dumping empty versions."""
        saver = BaseKustoSaver()
        result = saver._dump_blobs("thread-1", "", {}, {})
        assert result == []

    def test_dump_blobs_with_values(self) -> None:
        """Test dumping blobs with values."""
        saver = BaseKustoSaver()
        values = {"channel1": {"key": "value"}}
        versions = {"channel1": "1.0"}
        
        result = saver._dump_blobs("thread-1", "ns1", values, versions)
        
        assert len(result) == 1
        assert result[0]["thread_id"] == "thread-1"
        assert result[0]["checkpoint_ns"] == "ns1"
        assert result[0]["channel"] == "channel1"
        assert result[0]["version"] == "1.0"
        assert result[0]["type"] != "empty"

    def test_dump_blobs_with_empty_channel(self) -> None:
        """Test dumping blobs with missing channel."""
        saver = BaseKustoSaver()
        values = {}
        versions = {"channel1": "1.0"}
        
        result = saver._dump_blobs("thread-1", "", values, versions)
        
        assert len(result) == 1
        assert result[0]["type"] == "empty"
        assert result[0]["blob"] == ""

    def test_load_writes_empty(self) -> None:
        """Test loading empty writes list."""
        saver = BaseKustoSaver()
        result = saver._load_writes([])
        assert result == []

    def test_dump_writes(self) -> None:
        """Test dumping writes."""
        saver = BaseKustoSaver()
        writes = [("channel1", {"data": "value"})]
        
        result = saver._dump_writes(
            "thread-1",
            "ns1",
            "ckpt-1",
            "task-1",
            "path/to/task",
            writes,
        )
        
        assert len(result) == 1
        record = result[0]
        assert record["thread_id"] == "thread-1"
        assert record["checkpoint_ns"] == "ns1"
        assert record["checkpoint_id"] == "ckpt-1"
        assert record["task_id"] == "task-1"
        assert record["task_path"] == "path/to/task"
        assert record["channel"] == "channel1"
        assert record["idx"] >= 0

    def test_build_kql_filter_basic(self) -> None:
        """Test building basic KQL filter."""
        saver = BaseKustoSaver()
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "ns1",
            }
        }
        
        result = saver._build_kql_filter(config, None, None)
        
        assert "params" in result
        assert result["params"]["thread_id_param"] == "thread-1"
        assert result["params"]["checkpoint_ns_param"] == "ns1"

    def test_build_kql_filter_with_checkpoint_id(self) -> None:
        """Test building filter with checkpoint_id."""
        saver = BaseKustoSaver()
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "ckpt-1",
            }
        }
        
        result = saver._build_kql_filter(config, None, None)
        
        assert result["params"]["checkpoint_id_param"] == "ckpt-1"
        assert any("checkpoint_id" in f for f in result["filters"])

    def test_build_kql_filter_with_metadata(self) -> None:
        """Test building filter with metadata."""
        saver = BaseKustoSaver()
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }
        metadata_filter = {"source": "user", "step": 1}
        
        result = saver._build_kql_filter(config, metadata_filter, None)
        
        assert "metadata_filter" in result["params"]
        assert len(result["filters"]) > 0

    def test_format_kql_record_simple(self) -> None:
        """Test formatting a simple record."""
        saver = BaseKustoSaver()
        record = {
            "col1": "value1",
            "col2": 123,
            "col3": None,
        }
        
        result = saver._format_kql_record(record)
        
        # Basic validation - should be CSV format
        assert '"value1"' in result
        assert "123" in result

    def test_format_kql_record_with_json(self) -> None:
        """Test formatting record with JSON data."""
        saver = BaseKustoSaver()
        record = {
            "data": {"nested": "value"},
            "list": [1, 2, 3],
        }
        
        result = saver._format_kql_record(record)
        
        # Should contain JSON-encoded strings
        assert "{" in result or "[" in result

    def test_format_kql_records_multiple(self) -> None:
        """Test formatting multiple records."""
        saver = BaseKustoSaver()
        records = [
            {"col1": "a", "col2": 1},
            {"col1": "b", "col2": 2},
        ]
        
        result = saver._format_kql_records(records)
        
        # Should have multiple lines
        lines = result.split("\n")
        assert len(lines) == 2


@pytest.mark.unit
class TestSerialization:
    """Test serialization and deserialization."""

    def test_checkpoint_roundtrip(self) -> None:
        """Test checkpoint serialization roundtrip."""
        saver = BaseKustoSaver()
        checkpoint = empty_checkpoint()
        
        # Add some data
        checkpoint["channel_values"]["test_channel"] = {"data": "value"}
        
        # Serialize
        versions = {"test_channel": "1.0"}
        blob_records = saver._dump_blobs("thread-1", "", checkpoint["channel_values"], versions)
        
        # Deserialize
        loaded = saver._load_blobs(blob_records)
        
        # Verify
        assert "test_channel" in loaded
        assert loaded["test_channel"]["data"] == "value"

    def test_writes_roundtrip(self) -> None:
        """Test writes serialization roundtrip."""
        saver = BaseKustoSaver()
        writes = [
            ("channel1", {"key1": "value1"}),
            ("channel2", [1, 2, 3]),
        ]
        
        # Serialize
        records = saver._dump_writes("thread-1", "", "ckpt-1", "task-1", "", writes)
        
        # Deserialize
        loaded = saver._load_writes(records)
        
        # Verify
        assert len(loaded) == 2
        assert loaded[0][0] == "task-1"
        assert loaded[0][1] == "channel1"
        assert loaded[0][2]["key1"] == "value1"
        assert loaded[1][2] == [1, 2, 3]

    def test_metadata_serialization(self) -> None:
        """Test metadata can be serialized."""
        import orjson
        
        metadata = {
            "source": "user",
            "step": 5,
            "score": 0.95,
            "tags": ["tag1", "tag2"],
        }
        
        # Should be able to serialize
        json_str = orjson.dumps(metadata).decode()
        assert "user" in json_str
        
        # Should be able to deserialize
        loaded = orjson.loads(json_str)
        assert loaded == metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
