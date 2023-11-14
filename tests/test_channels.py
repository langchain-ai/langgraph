import operator
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Sequence, Union

import httpx
import pytest
from pytest_mock import MockerFixture

from permchain.channels.base import EmptyChannelError, InvalidUpdateError
from permchain.channels.binop import BinaryOperatorAggregate
from permchain.channels.context import Context
from permchain.channels.last_value import LastValue
from permchain.channels.topic import Topic


def test_last_value() -> None:
    with LastValue(int).empty() as channel:
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
    with LastValue(int).empty(checkpoint) as channel:
        assert channel.get() == 4


async def test_last_value_async() -> None:
    async with LastValue(int).aempty() as channel:
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
    async with LastValue(int).aempty(checkpoint) as channel:
        assert channel.get() == 4


def test_topic() -> None:
    with Topic(str).empty() as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        channel.update([["c", "d"], "d"])
        assert channel.get() == ["c", "d", "d"]
        channel.update([])
        assert channel.get() == []
        channel.update(["e"])
        assert channel.get() == ["e"]
        checkpoint = channel.checkpoint()
    with Topic(str).empty(checkpoint) as channel:
        assert channel.get() == ["e"]


async def test_topic_async() -> None:
    async with Topic(str).aempty() as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["b", "c", "d", "d"]
        channel.update([])
        assert channel.get() == []
        channel.update(["e"])
        assert channel.get() == ["e"]
        checkpoint = channel.checkpoint()
    async with Topic(str).aempty(checkpoint) as channel:
        assert channel.get() == ["e"]


def test_topic_unique() -> None:
    with Topic(str, unique=True).empty() as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["c", "d"], "de-dupes from current and previous steps"
        channel.update([])
        assert channel.get() == []
        channel.update(["e"])
        assert channel.get() == ["e"]
        checkpoint = channel.checkpoint()
    with Topic(str, unique=True).empty(checkpoint) as channel:
        assert channel.get() == ["e"]
        channel.update(["d", "f"])
        assert channel.get() == ["f"], "de-dupes from checkpoint"


async def test_topic_unique_async() -> None:
    async with Topic(str, unique=True).aempty() as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["c", "d"], "de-dupes from current and previous steps"
        channel.update([])
        assert channel.get() == []
        channel.update(["e"])
        assert channel.get() == ["e"]
        checkpoint = channel.checkpoint()
    async with Topic(str, unique=True).aempty(checkpoint) as channel:
        assert channel.get() == ["e"]
        channel.update(["d", "f"])
        assert channel.get() == ["f"], "de-dupes from checkpoint"


def test_topic_accumulate() -> None:
    with Topic(str, accumulate=True).empty() as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        channel.update([])
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        checkpoint = channel.checkpoint()
    with Topic(str, accumulate=True).empty(checkpoint) as channel:
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        channel.update(["e"])
        assert channel.get() == ["a", "b", "b", "c", "d", "d", "e"]


async def test_topic_accumulate_async() -> None:
    async with Topic(str, accumulate=True).aempty() as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        channel.update([])
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        checkpoint = channel.checkpoint()
    async with Topic(str, accumulate=True).aempty(checkpoint) as channel:
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        channel.update(["e"])
        assert channel.get() == ["a", "b", "b", "c", "d", "d", "e"]


def test_topic_unique_accumulate() -> None:
    with Topic(str, unique=True, accumulate=True).empty() as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["a", "b", "c", "d"]
        channel.update([])
        assert channel.get() == ["a", "b", "c", "d"]
        checkpoint = channel.checkpoint()
    with Topic(str, unique=True, accumulate=True).empty(checkpoint) as channel:
        assert channel.get() == ["a", "b", "c", "d"]
        channel.update(["d", "e"])
        assert channel.get() == ["a", "b", "c", "d", "e"]


async def test_topic_unique_accumulate_async() -> None:
    async with Topic(str, unique=True, accumulate=True).aempty() as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["a", "b", "c", "d"]
        channel.update([])
        assert channel.get() == ["a", "b", "c", "d"]
        checkpoint = channel.checkpoint()
    async with Topic(str, unique=True, accumulate=True).aempty(checkpoint) as channel:
        assert channel.get() == ["a", "b", "c", "d"]
        channel.update(["d", "e"])
        assert channel.get() == ["a", "b", "c", "d", "e"]


def test_binop() -> None:
    with BinaryOperatorAggregate(int, operator.add).empty() as channel:
        assert channel.ValueType is int
        assert channel.UpdateType is int

        with pytest.raises(EmptyChannelError):
            channel.get()

        channel.update([1, 2, 3])
        assert channel.get() == 6
        channel.update([4])
        assert channel.get() == 10
        checkpoint = channel.checkpoint()
    with BinaryOperatorAggregate(int, operator.add).empty(checkpoint) as channel:
        assert channel.get() == 10


async def test_binop_async() -> None:
    async with BinaryOperatorAggregate(int, operator.add).aempty() as channel:
        assert channel.ValueType is int
        assert channel.UpdateType is int

        with pytest.raises(EmptyChannelError):
            channel.get()

        channel.update([1, 2, 3])
        assert channel.get() == 6
        channel.update([4])
        assert channel.get() == 10
        checkpoint = channel.checkpoint()
    async with BinaryOperatorAggregate(int, operator.add).aempty(checkpoint) as channel:
        assert channel.get() == 10


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

    with Context(an_int, None, int).empty() as channel:
        assert setup.call_count == 1
        assert cleanup.call_count == 0

        assert channel.ValueType is int
        with pytest.raises(InvalidUpdateError):
            assert channel.UpdateType is None

        assert channel.get() == 5

        with pytest.raises(InvalidUpdateError):
            channel.update([5])  # type: ignore

    assert setup.call_count == 1
    assert cleanup.call_count == 1


def test_ctx_manager_ctx(mocker: MockerFixture) -> None:
    with Context(httpx.Client).empty() as channel:
        assert channel.ValueType is httpx.Client
        with pytest.raises(InvalidUpdateError):
            assert channel.UpdateType is None

        assert isinstance(channel.get(), httpx.Client)

        with pytest.raises(InvalidUpdateError):
            channel.update([5])  # type: ignore

        checkpoint = channel.checkpoint()
        assert checkpoint is None
    with Context(httpx.Client).empty(checkpoint) as channel:
        assert channel.ValueType is httpx.Client
        with pytest.raises(InvalidUpdateError):
            assert channel.UpdateType is None

        assert isinstance(channel.get(), httpx.Client)

        with pytest.raises(InvalidUpdateError):
            channel.update([5])


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

    async with Context(an_int_sync, an_int, int).aempty() as channel:
        assert setup.call_count == 1
        assert cleanup.call_count == 0

        assert channel.ValueType is int
        with pytest.raises(InvalidUpdateError):
            assert channel.UpdateType is None

        assert channel.get() == 5

        with pytest.raises(InvalidUpdateError):
            channel.update([5])  # type: ignore

    assert setup.call_count == 1
    assert cleanup.call_count == 1
