import operator
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from langgraph._internal._typing import MISSING
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.errors import EmptyChannelError, InvalidUpdateError
from langgraph.types import Overwrite

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


@dataclass
class _Metrics:
    """Helper dataclass with no zero-arg constructor for MISSING path tests."""

    count: int

    def __add__(self, other: "_Metrics") -> "_Metrics":
        return _Metrics(self.count + other.count)


def test_binop_overwrite_on_missing_unwraps_value() -> None:
    """Overwrite() on a MISSING channel must store the unwrapped payload, not the wrapper.

    Regression test for https://github.com/langchain-ai/langgraph/issues/6909.
    _Metrics has no zero-arg constructor so the channel starts as MISSING.
    """
    channel = BinaryOperatorAggregate(_Metrics, operator.add)
    assert channel.value is MISSING

    channel.update([Overwrite(_Metrics(count=42))])

    result = channel.get()
    assert isinstance(result, _Metrics), f"Expected _Metrics, got {type(result)}"
    assert result.count == 42


def test_binop_overwrite_on_missing_raises_on_duplicate() -> None:
    """Two Overwrite() values in the same super-step must raise InvalidUpdateError
    even when the channel starts as MISSING.

    Regression test for https://github.com/langchain-ai/langgraph/issues/6909.
    """
    channel = BinaryOperatorAggregate(_Metrics, operator.add)
    assert channel.value is MISSING

    with pytest.raises(InvalidUpdateError, match="Can receive only one Overwrite"):
        channel.update([Overwrite(_Metrics(1)), Overwrite(_Metrics(2))])


def test_binop_overwrite_dict_form_on_missing_unwraps_value() -> None:
    """Dict-form overwrite {OVERWRITE_KEY: x} on a MISSING channel must also unwrap.

    Regression test for https://github.com/langchain-ai/langgraph/issues/6909.
    """
    channel = BinaryOperatorAggregate(_Metrics, operator.add).from_checkpoint(MISSING)
    assert channel.value is MISSING

    # Dict form: {"__overwrite__": value}
    channel.update([{"__overwrite__": _Metrics(count=7)}])

    result = channel.get()
    assert isinstance(result, _Metrics), f"Expected _Metrics, got {type(result)}"
    assert result.count == 7


def test_binop_overwrite_on_missing_followed_by_normal_value() -> None:
    """Overwrite() as first value on MISSING channel, followed by a normal value,
    must discard the normal value (existing seen_overwrite semantics: normal updates
    after an Overwrite in the same super-step are silently dropped).

    Regression test for https://github.com/langchain-ai/langgraph/issues/6909.
    """
    channel = BinaryOperatorAggregate(_Metrics, operator.add).from_checkpoint(MISSING)
    assert channel.value is MISSING

    # Overwrite sets value to _Metrics(10); subsequent normal value is dropped
    # because seen_overwrite=True causes the `if not seen_overwrite` guard to skip it.
    channel.update([Overwrite(_Metrics(count=10)), _Metrics(count=5)])

    result = channel.get()
    assert isinstance(result, _Metrics), f"Expected _Metrics, got {type(result)}"
    assert result.count == 10


def test_binop_overwrite_from_checkpoint_on_missing() -> None:
    """from_checkpoint(MISSING) + Overwrite() must behave identically to a fresh
    MISSING channel: unwrap the value and set the duplicate guard.

    Regression test for https://github.com/langchain-ai/langgraph/issues/6909.
    """
    channel = BinaryOperatorAggregate(_Metrics, operator.add).from_checkpoint(MISSING)
    assert channel.value is MISSING

    channel.update([Overwrite(_Metrics(count=99))])

    assert channel.get() == _Metrics(count=99)

    channel2 = BinaryOperatorAggregate(_Metrics, operator.add).from_checkpoint(MISSING)
    with pytest.raises(InvalidUpdateError, match="Can receive only one Overwrite"):
        channel2.update([Overwrite(_Metrics(1)), Overwrite(_Metrics(2))])


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
