from typing import Iterator

import pytest

from langgraph.pregel.io import (
    AddableUpdatesDict,
    AddableValuesDict,
    map_input,
    map_output_updates,
    map_output_values,
    single,
)


def test_single() -> None:
    closed = False

    def myiter() -> Iterator[int]:
        try:
            yield 1
            yield 2
        finally:
            nonlocal closed
            closed = True

    assert single(myiter()) == 1
    assert closed


def test_addable_values_dict_operations() -> None:
    d1 = AddableValuesDict({"a": 1})
    d2 = {"b": 2}

    # Test __add__
    result = d1 + d2
    assert isinstance(result, dict)
    assert result == {"a": 1, "b": 2}

    # Test __radd__
    result = d2 + d1
    assert isinstance(result, dict)
    assert result == {"a": 1, "b": 2}


def test_addable_updates_dict() -> None:
    d1 = AddableUpdatesDict({"a": 1})
    d2 = {"b": 2}

    # Test __add__
    result = d1 + d2
    assert result == [{"a": 1}, {"b": 2}]

    # Test __radd__ raises TypeError
    with pytest.raises(
        TypeError, match="AddableUpdatesDict does not support right-side addition"
    ):
        d2 + d1


def test_map_output_values_sequence() -> None:
    class MockChannel:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    channels = {"ch1": MockChannel("value1"), "ch2": MockChannel("value2")}

    # Test with sequence output channels and True pending writes
    result = list(map_output_values(["ch1", "ch2"], True, channels))
    assert len(result) == 1
    assert isinstance(result[0], AddableValuesDict)
    assert result[0] == {"ch1": "value1", "ch2": "value2"}


def test_map_output_updates_basic() -> None:
    from dataclasses import dataclass

    @dataclass
    class MockTask:
        name: str
        config: dict

    task1 = MockTask("node1", {})
    task2 = MockTask("node2", {"tags": ["hidden"]})

    tasks = [(task1, [("channel1", "value1")]), (task2, [("channel2", "value2")])]

    # Test with string output channel
    result = list(map_output_updates("channel1", tasks))
    assert len(result) == 1
    assert result[0]["node1"] == "value1"

    # Test with sequence output channels
    result = list(map_output_updates(["channel1", "channel2"], tasks))
    assert len(result) == 1
    assert result[0]["node1"] == {"channel1": "value1"}

    # Test with cached=True
    result = list(map_output_updates(["channel1"], tasks, cached=True))
    assert len(result) == 1
    assert result[0]["__metadata__"] == {"cached": True}


def test_map_input() -> None:
    # Test with None input
    result = list(map_input("test_channel", None))
    assert len(result) == 0

    # Test with string input channel and direct value
    result = list(map_input("test_channel", "test_value"))
    assert result == [("test_channel", "test_value")]

    # Test with sequence input channels and dict value
    input_channels = ["ch1", "ch2"]
    chunk = {"ch1": "value1", "ch2": "value2", "ch3": "value3"}
    result = list(map_input(input_channels, chunk))
    assert set(result) == {("ch1", "value1"), ("ch2", "value2")}

    # Test with invalid input type
    with pytest.raises(TypeError, match="Expected chunk to be a dict"):
        list(map_input(["ch1", "ch2"], "invalid"))
