import pytest
from langgraph.pregel.checkpoint import create_checkpoint, empty_checkpoint


class MockLastValue:
    """Mock compatible with LangGraph's LastValue channel interface."""
    def __init__(self, initial_value):
        self._value = initial_value
        self._version = 1

    def update(self, new_values):
        self._value = new_values[-1]
        self._version += 1

    def snapshot(self):
        return self._value

    @property
    def version(self):
        return self._version

    def checkpoint(self):
        return self._value


@pytest.mark.parametrize("explicit_id", ["test-id-123", "another-id"])
def test_checkpoint_id_generation(explicit_id):
    initial = empty_checkpoint()
    checkpoint1 = create_checkpoint(initial, None, 0, id=explicit_id)
    assert checkpoint1["id"] == explicit_id

    # Fork with explicit ID should ignore the ID and generate new one
    checkpoint2 = create_checkpoint(initial, None, 0, id=explicit_id, is_fork=True)
    assert checkpoint2["id"] != explicit_id
    assert checkpoint2["id"] != initial["id"]


def test_checkpoint_with_counter_only():
    counter_channel = MockLastValue(42)
    channels = {"counter": counter_channel}

    initial = empty_checkpoint()
    initial["channel_versions"] = {"counter": 1}
    checkpoint1 = create_checkpoint(initial, channels, 0)
    print("Checkpoint 1 Channel Values:", checkpoint1["channel_values"])

    counter_channel.update([99])

    fork = create_checkpoint(checkpoint1, channels, 1, is_fork=True)
    print("Fork Channel Values:", fork["channel_values"])

    assert fork["channel_values"]["counter"] == 99
    assert fork["id"] != checkpoint1["id"]


def test_checkpoint_metadata_preservation():
    initial = empty_checkpoint()
    initial["versions_seen"] = {"node1": {"channel1": 1}}
    initial["channel_versions"] = {"channel1": 1}

    fork = create_checkpoint(initial, None, 0, is_fork=True)

    assert fork["versions_seen"] == initial["versions_seen"]
    assert fork["channel_versions"] == initial["channel_versions"]
    assert fork["v"] == initial["v"]
    assert fork["id"] != initial["id"]
