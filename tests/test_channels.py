import operator
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, FrozenSet, Generator, Sequence

import pytest
from pytest_mock import MockerFixture

import permchain.channels as channels


def test_last_value() -> None:
    with channels.LastValue(int)._empty() as channel:
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


async def test_last_value_async() -> None:
    async with channels.LastValue(int)._aempty() as channel:
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
    with channels.Inbox(str)._empty() as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is str

        with pytest.raises(channels.EmptyChannelError):
            channel._get()

        channel._update(["a", "b"])
        assert channel._get() == ("a", "b")
        channel._update(["c"])
        assert channel._get() == ("c",)


async def test_inbox_async() -> None:
    async with channels.Inbox(str)._aempty() as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is str

        with pytest.raises(channels.EmptyChannelError):
            channel._get()

        channel._update(["a", "b"])
        assert channel._get() == ("a", "b")
        channel._update(["c"])
        assert channel._get() == ("c",)


def test_set() -> None:
    with channels.Set(str)._empty() as channel:
        assert channel.ValueType is FrozenSet[str]
        assert channel.UpdateType is str

        with pytest.raises(channels.EmptyChannelError):
            channel._get()

        channel._update(["a", "b"])
        assert channel._get() == frozenset(("a", "b"))
        channel._update(["b", "c"])
        assert channel._get() == frozenset(("a", "b", "c"))


async def test_set_async() -> None:
    async with channels.Set(str)._aempty() as channel:
        assert channel.ValueType is FrozenSet[str]
        assert channel.UpdateType is str

        with pytest.raises(channels.EmptyChannelError):
            channel._get()

        channel._update(["a", "b"])
        assert channel._get() == frozenset(("a", "b"))
        channel._update(["b", "c"])
        assert channel._get() == frozenset(("a", "b", "c"))


def test_binop() -> None:
    with channels.BinaryOperatorAggregate(int, operator.add)._empty() as channel:
        assert channel.ValueType is int
        assert channel.UpdateType is int

        with pytest.raises(channels.EmptyChannelError):
            channel._get()

        channel._update([1, 2, 3])
        assert channel._get() == 6
        channel._update([4])
        assert channel._get() == 10


async def test_binop_async() -> None:
    async with channels.BinaryOperatorAggregate(int, operator.add)._aempty() as channel:
        assert channel.ValueType is int
        assert channel.UpdateType is int

        with pytest.raises(channels.EmptyChannelError):
            channel._get()

        channel._update([1, 2, 3])
        assert channel._get() == 6
        channel._update([4])
        assert channel._get() == 10


def test_ctx_manager(mocker: MockerFixture) -> None:
    setup = mocker.Mock()
    cleanup = mocker.Mock()

    @contextmanager
    def an_int() -> Generator[int, None, None]:
        setup()
        try:
            yield 5
        finally:
            cleanup()

    with channels.ContextManager(int, an_int)._empty() as channel:
        assert setup.call_count == 1
        assert cleanup.call_count == 0

        assert channel.ValueType is int
        with pytest.raises(channels.InvalidUpdateError):
            assert channel.UpdateType is None

        assert channel._get() == 5

        with pytest.raises(channels.InvalidUpdateError):
            channel._update([5])

    assert setup.call_count == 1
    assert cleanup.call_count == 1


async def test_ctx_manager_async(mocker: MockerFixture) -> None:
    setup = mocker.Mock()
    cleanup = mocker.Mock()

    @contextmanager
    def an_int_sync() -> Generator[int, None, None]:
        try:
            yield 5
        finally:
            pass

    @asynccontextmanager
    async def an_int() -> AsyncGenerator[int, None]:
        setup()
        try:
            yield 5
        finally:
            cleanup()

    async with channels.ContextManager(int, an_int_sync, an_int)._aempty() as channel:
        assert setup.call_count == 1
        assert cleanup.call_count == 0

        assert channel.ValueType is int
        with pytest.raises(channels.InvalidUpdateError):
            assert channel.UpdateType is None

        assert channel._get() == 5

        with pytest.raises(channels.InvalidUpdateError):
            channel._update([5])

    assert setup.call_count == 1
    assert cleanup.call_count == 1
