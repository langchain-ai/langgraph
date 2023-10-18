import operator
from typing import FrozenSet, Sequence

import pytest

import permchain.channels as channels


def test_last_value() -> None:
    with channels.LastValue(int) as channel:
        assert channel.ValueType is int
        assert channel.UpdateType is int

        with pytest.raises(channels.EmptyChannelError):
            channel._get()
        with pytest.raises(channels.InvalidUpdateError):
            channel._update([5, 6])

        channel._update([3])
        assert channel._get() == 3
        channel._update([4])
        assert channel._get() == 4


def test_inbox() -> None:
    with channels.Inbox(str) as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is str

        with pytest.raises(channels.EmptyChannelError):
            channel._get()

        channel._update(["a", "b"])
        assert channel._get() == ("a", "b")
        channel._update(["c"])
        assert channel._get() == ("c",)


def test_set() -> None:
    with channels.Set(str) as channel:
        assert channel.ValueType is FrozenSet[str]
        assert channel.UpdateType is str

        with pytest.raises(channels.EmptyChannelError):
            channel._get()

        channel._update(["a", "b"])
        assert channel._get() == frozenset(("a", "b"))
        channel._update(["b", "c"])
        assert channel._get() == frozenset(("a", "b", "c"))


def test_binop() -> None:
    with channels.BinaryOperatorAggregate(int, operator.add) as channel:
        assert channel.ValueType is int
        assert channel.UpdateType is int

        with pytest.raises(channels.EmptyChannelError):
            channel._get()

        channel._update([1, 2, 3])
        assert channel._get() == 6
        channel._update([4])
        assert channel._get() == 10
