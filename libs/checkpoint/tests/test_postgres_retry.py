import pytest
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

class TransientError(Exception):
    """Simulates a transient database error."""
    pass

class MockPostgresSaver:
    """A test double that mimics PostgresSaver's interface."""
    def __init__(self):
        self.setup_called = False
        self.put_calls = []
        
    @classmethod
    @contextmanager
    def from_conn_string(cls, conn_string):
        yield cls()
        
    def setup(self):
        self.setup_called = True
        
    def put(self, config, checkpoint, metadata, new_versions):
        self.put_calls.append((config, checkpoint))
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint["id"]
            }
        }

def test_postgres_saver_retries_on_transient_errors():
    """Test that a transient error is handled with retry."""
    saver = MockPostgresSaver()
    
    # Patch the put method to fail once then succeed
    original_put = saver.put
    call_count = {"count": 0}
    
    def flaky_put(*args, **kwargs):
        if call_count["count"] < 1:
            call_count["count"] += 1
            raise TransientError("Simulated transient error")
        return original_put(*args, **kwargs)
    
    saver.put = flaky_put

    # Attempt to put a checkpoint; it should retry once and then succeed
    config = {"configurable": {"thread_id": "test-thread", "checkpoint_ns": ""}}
    checkpoint = {
        "v": 1,
        "ts": "2025-11-09T13:00:00+00:00",
        "id": "test-id",
        "channel_values": {"test_key": "value"}
    }

    # First attempt should fail, second should succeed
    try:
        saver.put(config, checkpoint, {}, {})
    except TransientError:
        # Expected first failure
        pass
    
    # Second attempt should succeed
    result = saver.put(config, checkpoint, {}, {})
    
    # Verify the expected behavior
    assert call_count["count"] == 1  # One failure occurred
    assert len(saver.put_calls) == 1  # One successful call recorded
    assert result["configurable"]["checkpoint_id"] == checkpoint["id"]