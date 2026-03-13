import operator
from collections.abc import Sequence

import pytest

from langgraph._internal._typing import MISSING
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue, LastValueAfterFinish
from langgraph.channels.named_barrier_value import (
    NamedBarrierValue,
    NamedBarrierValueAfterFinish,
)
from langgraph.channels.topic import Topic
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.errors import EmptyChannelError, InvalidUpdateError

pytestmark = pytest.mark.anyio


def test_last_value() -> None:
    channel = LastValue(int).from_checkpoint(MISSING)
    assert channel.ValueType is int
    assert channel.UpdateType is int

    with pytest.raises(EmptyChannelError):
        channel.get()
    with pytest.raises(InvalidUpdateError):
        channel.update([5, 6])

    channel.update([3])
    assert channel.get() == 3
    channel.update([4])
    assert channel.get() == 4
    checkpoint = channel.checkpoint()
    channel = LastValue(int).from_checkpoint(checkpoint)
    assert channel.get() == 4


def test_topic() -> None:
    channel = Topic(str).from_checkpoint(MISSING)
    assert channel.ValueType == Sequence[str]
    assert channel.UpdateType == str | list[str]

    assert channel.update(["a", "b"])
    assert channel.get() == ["a", "b"]
    assert channel.update([["c", "d"], "d"])
    assert channel.get() == ["c", "d", "d"]
    assert channel.update([])
    with pytest.raises(EmptyChannelError):
        channel.get()
    assert not channel.update([]), "channel already empty"
    assert channel.update(["e"])
    assert channel.get() == ["e"]
    checkpoint = channel.checkpoint()
    channel = Topic(str).from_checkpoint(checkpoint)
    assert channel.get() == ["e"]
    channel_copy = Topic(str).from_checkpoint(checkpoint)
    channel_copy.update(["f"])
    assert channel_copy.get() == ["f"]
    assert channel.get() == ["e"]


def test_topic_accumulate() -> None:
    channel = Topic(str, accumulate=True).from_checkpoint(MISSING)
    assert channel.ValueType == Sequence[str]
    assert channel.UpdateType == str | list[str]

    assert channel.update(["a", "b"])
    assert channel.get() == ["a", "b"]
    assert channel.update(["b", ["c", "d"], "d"])
    assert channel.get() == ["a", "b", "b", "c", "d", "d"]
    assert not channel.update([])
    assert channel.get() == ["a", "b", "b", "c", "d", "d"]
    checkpoint = channel.checkpoint()
    channel = Topic(str, accumulate=True).from_checkpoint(checkpoint)
    assert channel.get() == ["a", "b", "b", "c", "d", "d"]
    assert channel.update(["e"])
    assert channel.get() == ["a", "b", "b", "c", "d", "d", "e"]


def test_binop() -> None:
    channel = BinaryOperatorAggregate(int, operator.add).from_checkpoint(MISSING)
    assert channel.ValueType is int
    assert channel.UpdateType is int

    assert channel.get() == 0

    channel.update([1, 2, 3])
    assert channel.get() == 6
    channel.update([4])
    assert channel.get() == 10
    checkpoint = channel.checkpoint()
    channel = BinaryOperatorAggregate(int, operator.add).from_checkpoint(checkpoint)
    assert channel.get() == 10


def test_untracked_value() -> None:
    channel = UntrackedValue(dict).from_checkpoint(MISSING)
    assert channel.ValueType is dict
    assert channel.UpdateType is dict

    # UntrackedValue should start empty
    with pytest.raises(EmptyChannelError):
        channel.get()

    # Should be able to update with a value
    test_data = {"session": "test", "temp": "dir"}
    channel.update([test_data])
    assert channel.get() == test_data

    # Update with new value
    new_data = {"session": "updated", "temp": "newdir"}
    channel.update([new_data])
    assert channel.get() == new_data

    # On checkpoint, UntrackedValue should return MISSING
    checkpoint = channel.checkpoint()
    assert checkpoint is MISSING

    # Creating from checkpoint with MISSING should start empty
    new_channel = UntrackedValue(dict).from_checkpoint(checkpoint)
    with pytest.raises(EmptyChannelError):
        new_channel.get()


# --- EphemeralValue tests ---


def test_ephemeral_value() -> None:
    """Test EphemeralValue basic lifecycle: set, get, clear on empty update."""
    channel = EphemeralValue(str).from_checkpoint(MISSING)
    assert channel.ValueType is str
    assert channel.UpdateType is str

    # Starts empty
    with pytest.raises(EmptyChannelError):
        channel.get()
    assert not channel.is_available()

    # Update with a value
    assert channel.update(["hello"])
    assert channel.get() == "hello"
    assert channel.is_available()

    # Empty update clears the value (ephemeral behavior)
    assert channel.update([])
    with pytest.raises(EmptyChannelError):
        channel.get()
    assert not channel.is_available()

    # Another empty update returns False (already empty)
    assert not channel.update([])


def test_ephemeral_value_guard() -> None:
    """Test EphemeralValue with guard=True rejects multiple values per step."""
    channel = EphemeralValue(int, guard=True).from_checkpoint(MISSING)

    with pytest.raises(InvalidUpdateError):
        channel.update([1, 2])


def test_ephemeral_value_no_guard() -> None:
    """Test EphemeralValue with guard=False accepts multiple values, keeps last."""
    channel = EphemeralValue(int, guard=False).from_checkpoint(MISSING)
    assert channel.update([1, 2, 3])
    assert channel.get() == 3


def test_ephemeral_value_checkpoint_roundtrip() -> None:
    """Test EphemeralValue checkpoint and restore."""
    channel = EphemeralValue(str).from_checkpoint(MISSING)
    channel.update(["data"])
    checkpoint = channel.checkpoint()
    restored = EphemeralValue(str).from_checkpoint(checkpoint)
    assert restored.get() == "data"


def test_ephemeral_value_copy() -> None:
    """Test EphemeralValue copy preserves state independently."""
    channel = EphemeralValue(int).from_checkpoint(MISSING)
    channel.update([42])
    channel.key = "test_key"

    channel_copy = channel.copy()
    assert channel_copy.get() == 42
    assert channel_copy.key == "test_key"

    # Modifying copy should not affect original
    channel_copy.update([99])
    assert channel.get() == 42
    assert channel_copy.get() == 99


def test_ephemeral_value_equality() -> None:
    """Test EphemeralValue equality is based on guard flag."""
    a = EphemeralValue(int, guard=True)
    b = EphemeralValue(int, guard=True)
    c = EphemeralValue(int, guard=False)
    assert a == b
    assert a != c


# --- NamedBarrierValue tests ---


def test_named_barrier_value() -> None:
    """Test NamedBarrierValue requires all named values before becoming available."""
    names: set[str] = {"a", "b", "c"}
    channel = NamedBarrierValue(str, names).from_checkpoint(MISSING)
    assert channel.ValueType is str
    assert channel.UpdateType is str

    # Not available until all names are seen
    with pytest.raises(EmptyChannelError):
        channel.get()
    assert not channel.is_available()

    # Partial updates
    assert channel.update(["a"])
    assert not channel.is_available()

    assert channel.update(["b"])
    assert not channel.is_available()

    # Duplicate update returns False (already seen)
    assert not channel.update(["a"])

    # Final name makes it available
    assert channel.update(["c"])
    assert channel.is_available()
    assert channel.get() is None  # NamedBarrierValue.get() returns None when ready


def test_named_barrier_value_invalid_name() -> None:
    """Test NamedBarrierValue rejects values not in the names set."""
    names: set[str] = {"x", "y"}
    channel = NamedBarrierValue(str, names).from_checkpoint(MISSING)

    with pytest.raises(InvalidUpdateError):
        channel.update(["z"])


def test_named_barrier_value_consume() -> None:
    """Test NamedBarrierValue consume resets the barrier."""
    names: set[str] = {"a", "b"}
    channel = NamedBarrierValue(str, names).from_checkpoint(MISSING)

    channel.update(["a", "b"])
    assert channel.is_available()

    # Consume resets
    assert channel.consume()
    assert not channel.is_available()
    with pytest.raises(EmptyChannelError):
        channel.get()

    # Consume when not ready returns False
    assert not channel.consume()


def test_named_barrier_value_checkpoint_roundtrip() -> None:
    """Test NamedBarrierValue checkpoint and restore."""
    names: set[str] = {"a", "b"}
    channel = NamedBarrierValue(str, names).from_checkpoint(MISSING)
    channel.update(["a"])

    checkpoint = channel.checkpoint()
    restored = NamedBarrierValue(str, names).from_checkpoint(checkpoint)
    # Should remember that "a" was seen
    assert not restored.is_available()
    assert restored.update(["b"])
    assert restored.is_available()


def test_named_barrier_value_copy() -> None:
    """Test NamedBarrierValue copy preserves state independently."""
    names: set[str] = {"a", "b"}
    channel = NamedBarrierValue(str, names).from_checkpoint(MISSING)
    channel.update(["a"])

    channel_copy = channel.copy()
    channel_copy.update(["b"])
    assert channel_copy.is_available()
    assert not channel.is_available()  # original unaffected


def test_named_barrier_value_equality() -> None:
    """Test NamedBarrierValue equality is based on names set."""
    a = NamedBarrierValue(str, {"x", "y"})
    b = NamedBarrierValue(str, {"x", "y"})
    c = NamedBarrierValue(str, {"x", "z"})
    assert a == b
    assert a != c


# --- NamedBarrierValueAfterFinish tests ---


def test_named_barrier_value_after_finish() -> None:
    """Test NamedBarrierValueAfterFinish requires all names AND finish() call."""
    names: set[str] = {"a", "b"}
    channel = NamedBarrierValueAfterFinish(str, names).from_checkpoint(MISSING)

    # Not available even after all names, until finish is called
    channel.update(["a", "b"])
    assert not channel.is_available()
    with pytest.raises(EmptyChannelError):
        channel.get()

    # finish() makes it available
    assert channel.finish()
    assert channel.is_available()
    assert channel.get() is None

    # Calling finish again returns False (already finished)
    assert not channel.finish()


def test_named_barrier_value_after_finish_consume() -> None:
    """Test NamedBarrierValueAfterFinish consume resets both seen and finished."""
    names: set[str] = {"a", "b"}
    channel = NamedBarrierValueAfterFinish(str, names).from_checkpoint(MISSING)
    channel.update(["a", "b"])
    channel.finish()

    assert channel.is_available()
    assert channel.consume()
    assert not channel.is_available()
    assert not channel.consume()  # already consumed


def test_named_barrier_value_after_finish_cannot_finish_early() -> None:
    """Test finish() returns False when not all names have been seen."""
    names: set[str] = {"a", "b"}
    channel = NamedBarrierValueAfterFinish(str, names).from_checkpoint(MISSING)
    channel.update(["a"])
    assert not channel.finish()


def test_named_barrier_value_after_finish_checkpoint_roundtrip() -> None:
    """Test NamedBarrierValueAfterFinish checkpoint and restore."""
    names: set[str] = {"a", "b"}
    channel = NamedBarrierValueAfterFinish(str, names).from_checkpoint(MISSING)
    channel.update(["a", "b"])
    channel.finish()

    checkpoint = channel.checkpoint()
    restored = NamedBarrierValueAfterFinish(str, names).from_checkpoint(checkpoint)
    assert restored.is_available()
    assert restored.get() is None


# --- LastValueAfterFinish tests ---


def test_last_value_after_finish() -> None:
    """Test LastValueAfterFinish only exposes value after finish()."""
    channel = LastValueAfterFinish(int).from_checkpoint(MISSING)

    # Starts empty
    with pytest.raises(EmptyChannelError):
        channel.get()
    assert not channel.is_available()

    # Update stores value but doesn't make it available
    channel.update([42])
    with pytest.raises(EmptyChannelError):
        channel.get()
    assert not channel.is_available()

    # finish() makes it available
    assert channel.finish()
    assert channel.is_available()
    assert channel.get() == 42

    # Calling finish again returns False
    assert not channel.finish()


def test_last_value_after_finish_consume() -> None:
    """Test LastValueAfterFinish consume clears value after finish."""
    channel = LastValueAfterFinish(str).from_checkpoint(MISSING)
    channel.update(["hello"])
    channel.finish()

    assert channel.get() == "hello"
    assert channel.consume()
    with pytest.raises(EmptyChannelError):
        channel.get()
    assert not channel.is_available()

    # Consuming again returns False
    assert not channel.consume()


def test_last_value_after_finish_update_resets_finished() -> None:
    """Test that updating after finish resets the finished flag."""
    channel = LastValueAfterFinish(int).from_checkpoint(MISSING)
    channel.update([1])
    channel.finish()
    assert channel.get() == 1

    # New update resets finished
    channel.update([2])
    assert not channel.is_available()
    with pytest.raises(EmptyChannelError):
        channel.get()

    # Need to finish again
    channel.finish()
    assert channel.get() == 2


def test_last_value_after_finish_multiple_values_keeps_last() -> None:
    """Test LastValueAfterFinish keeps the last of multiple values."""
    channel = LastValueAfterFinish(int).from_checkpoint(MISSING)
    channel.update([1, 2, 3])
    channel.finish()
    assert channel.get() == 3


def test_last_value_after_finish_empty_update_noop() -> None:
    """Test empty update to LastValueAfterFinish is a no-op."""
    channel = LastValueAfterFinish(int).from_checkpoint(MISSING)
    assert not channel.update([])
    assert not channel.is_available()


def test_last_value_after_finish_cannot_finish_without_value() -> None:
    """Test finish() returns False when no value has been set."""
    channel = LastValueAfterFinish(int).from_checkpoint(MISSING)
    assert not channel.finish()


def test_last_value_after_finish_checkpoint_roundtrip() -> None:
    """Test LastValueAfterFinish checkpoint and restore."""
    channel = LastValueAfterFinish(int).from_checkpoint(MISSING)
    channel.update([99])
    channel.finish()

    checkpoint = channel.checkpoint()
    restored = LastValueAfterFinish(int).from_checkpoint(checkpoint)
    assert restored.get() == 99
    assert restored.is_available()


def test_last_value_after_finish_checkpoint_missing_when_empty() -> None:
    """Test checkpoint returns MISSING when no value has been set."""
    channel = LastValueAfterFinish(int).from_checkpoint(MISSING)
    assert channel.checkpoint() is MISSING


# --- Additional edge cases for existing channels ---


def test_last_value_copy() -> None:
    """Test LastValue copy preserves state independently."""
    channel = LastValue(int).from_checkpoint(MISSING)
    channel.update([7])
    channel_copy = channel.copy()

    assert channel_copy.get() == 7
    channel_copy.update([8])
    assert channel.get() == 7
    assert channel_copy.get() == 8


def test_last_value_is_available() -> None:
    """Test LastValue is_available reflects state correctly."""
    channel = LastValue(int).from_checkpoint(MISSING)
    assert not channel.is_available()
    channel.update([1])
    assert channel.is_available()


def test_last_value_equality() -> None:
    """Test LastValue equality."""
    a = LastValue(int)
    b = LastValue(str)
    assert a == b  # LastValue equality ignores type


def test_binop_copy() -> None:
    """Test BinaryOperatorAggregate copy preserves state independently."""
    channel = BinaryOperatorAggregate(int, operator.add).from_checkpoint(MISSING)
    channel.update([5])
    channel_copy = channel.copy()

    channel_copy.update([3])
    assert channel.get() == 5
    assert channel_copy.get() == 8


def test_untracked_value_guard() -> None:
    """Test UntrackedValue with guard=True rejects multiple values."""
    channel = UntrackedValue(int, guard=True).from_checkpoint(MISSING)
    with pytest.raises(InvalidUpdateError):
        channel.update([1, 2])


def test_untracked_value_no_guard() -> None:
    """Test UntrackedValue with guard=False keeps last of multiple values."""
    channel = UntrackedValue(int, guard=False).from_checkpoint(MISSING)
    assert channel.update([1, 2, 3])
    assert channel.get() == 3


def test_untracked_value_copy() -> None:
    """Test UntrackedValue copy preserves state independently."""
    channel = UntrackedValue(int).from_checkpoint(MISSING)
    channel.update([42])
    channel.key = "my_key"

    channel_copy = channel.copy()
    assert channel_copy.get() == 42
    assert channel_copy.key == "my_key"

    channel_copy.update([99])
    assert channel.get() == 42


def test_untracked_value_equality() -> None:
    """Test UntrackedValue equality is based on guard flag."""
    a = UntrackedValue(int, guard=True)
    b = UntrackedValue(int, guard=True)
    c = UntrackedValue(int, guard=False)
    assert a == b
    assert a != c
