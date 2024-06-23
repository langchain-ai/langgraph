import operator
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Sequence, Union

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from pytest_mock import MockerFixture

from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.context import Context
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.errors import EmptyChannelError, InvalidUpdateError


def test_last_value() -> None:
    with LastValue(int).from_checkpoint(None, {}) as channel:
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
    with LastValue(int).from_checkpoint(checkpoint, {}) as channel:
        assert channel.get() == 4


async def test_last_value_async() -> None:
    async with LastValue(int).afrom_checkpoint(None, {}) as channel:
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
    async with LastValue(int).afrom_checkpoint(checkpoint, {}) as channel:
        assert channel.get() == 4


def test_topic() -> None:
    with Topic(str).from_checkpoint(None, {}) as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

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
    with Topic(str).from_checkpoint(checkpoint, {}) as channel:
        assert channel.get() == ["e"]
        with Topic(str).from_checkpoint(checkpoint, {}) as channel_copy:
            channel_copy.update(["f"])
            assert channel_copy.get() == ["f"]
            assert channel.get() == ["e"]


async def test_topic_async() -> None:
    async with Topic(str).afrom_checkpoint(None, {}) as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        assert channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        assert channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["b", "c", "d", "d"]
        assert channel.update([])
        with pytest.raises(EmptyChannelError):
            channel.get()
        assert not channel.update([]), "channel already empty"
        assert channel.update(["e"])
        assert channel.get() == ["e"]
        checkpoint = channel.checkpoint()
    async with Topic(str).afrom_checkpoint(checkpoint, {}) as channel:
        assert channel.get() == ["e"]


def test_topic_unique() -> None:
    with Topic(str, unique=True).from_checkpoint(None, {}) as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        assert channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        assert channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["c", "d"], "de-dupes from current and previous steps"
        assert channel.update([])
        with pytest.raises(EmptyChannelError):
            channel.get()
        assert not channel.update([]), "channel already empty"
        assert channel.update(["e"])
        assert channel.get() == ["e"]
        checkpoint = channel.checkpoint()
    with Topic(str, unique=True).from_checkpoint(checkpoint, {}) as channel:
        assert channel.get() == ["e"]
        assert channel.update(["d", "f"])
        assert channel.get() == ["f"], "de-dupes from checkpoint"


async def test_topic_unique_async() -> None:
    async with Topic(str, unique=True).afrom_checkpoint(None, {}) as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        assert channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        assert channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["c", "d"], "de-dupes from current and previous steps"
        assert channel.update([])
        with pytest.raises(EmptyChannelError):
            channel.get()
        assert not channel.update([]), "channel already empty"
        assert channel.update(["e"])
        assert channel.get() == ["e"]
        checkpoint = channel.checkpoint()
    async with Topic(str, unique=True).afrom_checkpoint(checkpoint, {}) as channel:
        assert channel.get() == ["e"]
        assert channel.update(["d", "f"])
        assert channel.get() == ["f"], "de-dupes from checkpoint"


def test_topic_accumulate() -> None:
    with Topic(str, accumulate=True).from_checkpoint(None, {}) as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        assert channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        assert channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        assert not channel.update([])
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        checkpoint = channel.checkpoint()
    with Topic(str, accumulate=True).from_checkpoint(checkpoint, {}) as channel:
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        assert channel.update(["e"])
        assert channel.get() == ["a", "b", "b", "c", "d", "d", "e"]


async def test_topic_accumulate_async() -> None:
    async with Topic(str, accumulate=True).afrom_checkpoint(None, {}) as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        assert channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        assert channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        assert not channel.update([])
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        checkpoint = channel.checkpoint()
    async with Topic(str, accumulate=True).afrom_checkpoint(checkpoint, {}) as channel:
        assert channel.get() == ["a", "b", "b", "c", "d", "d"]
        assert channel.update(["e"])
        assert channel.get() == ["a", "b", "b", "c", "d", "d", "e"]


def test_topic_unique_accumulate() -> None:
    with Topic(str, unique=True, accumulate=True).from_checkpoint(None, {}) as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        assert channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        assert channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["a", "b", "c", "d"]
        assert not channel.update(["c"]), "no new values"
        assert not channel.update([])
        assert channel.get() == ["a", "b", "c", "d"]
        checkpoint = channel.checkpoint()
    with Topic(str, unique=True, accumulate=True).from_checkpoint(
        checkpoint, {}
    ) as channel:
        assert channel.get() == ["a", "b", "c", "d"]
        assert channel.update(["d", "e"])
        assert channel.get() == ["a", "b", "c", "d", "e"]


async def test_topic_unique_accumulate_async() -> None:
    async with Topic(str, unique=True, accumulate=True).afrom_checkpoint(
        None, {}
    ) as channel:
        assert channel.ValueType is Sequence[str]
        assert channel.UpdateType is Union[str, list[str]]

        channel.update(["a", "b"])
        assert channel.get() == ["a", "b"]
        channel.update(["b", ["c", "d"], "d"])
        assert channel.get() == ["a", "b", "c", "d"]
        channel.update([])
        assert channel.get() == ["a", "b", "c", "d"]
        checkpoint = channel.checkpoint()
    async with Topic(str, unique=True, accumulate=True).afrom_checkpoint(
        checkpoint, {}
    ) as channel:
        assert channel.get() == ["a", "b", "c", "d"]
        channel.update(["d", "e"])
        assert channel.get() == ["a", "b", "c", "d", "e"]


def test_binop() -> None:
    with BinaryOperatorAggregate(int, operator.add).from_checkpoint(
        None, {}
    ) as channel:
        assert channel.ValueType is int
        assert channel.UpdateType is int

        assert channel.get() == 0

        channel.update([1, 2, 3])
        assert channel.get() == 6
        channel.update([4])
        assert channel.get() == 10
        checkpoint = channel.checkpoint()
    with BinaryOperatorAggregate(int, operator.add).from_checkpoint(
        checkpoint, {}
    ) as channel:
        assert channel.get() == 10


async def test_binop_async() -> None:
    async with BinaryOperatorAggregate(int, operator.add).afrom_checkpoint(
        None, {}
    ) as channel:
        assert channel.ValueType is int
        assert channel.UpdateType is int

        assert channel.get() == 0

        channel.update([1, 2, 3])
        assert channel.get() == 6
        channel.update([4])
        assert channel.get() == 10
        checkpoint = channel.checkpoint()
    async with BinaryOperatorAggregate(int, operator.add).afrom_checkpoint(
        checkpoint, {}
    ) as channel:
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

    with Context(an_int, None).from_checkpoint(None, {}) as channel:
        assert setup.call_count == 1
        assert cleanup.call_count == 0

        assert channel.ValueType is None
        assert channel.UpdateType is None

        assert channel.get() == 5

        with pytest.raises(InvalidUpdateError):
            channel.update([5])  # type: ignore

    assert setup.call_count == 1
    assert cleanup.call_count == 1


def test_ctx_manager_ctx(mocker: MockerFixture) -> None:
    with Context(httpx.Client).from_checkpoint(None, {}) as channel:
        assert channel.ValueType is None
        assert channel.UpdateType is None

        assert isinstance(channel.get(), httpx.Client)

        with pytest.raises(InvalidUpdateError):
            channel.update([5])  # type: ignore

        with pytest.raises(EmptyChannelError):
            channel.checkpoint()


async def test_ctx_manager_async(mocker: MockerFixture) -> None:
    setup = mocker.Mock()
    cleanup = mocker.Mock()

    @contextmanager
    def an_int_sync(config: RunnableConfig) -> Generator[int, None, None]:
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

    async with Context(an_int_sync, an_int).afrom_checkpoint(None, {}) as channel:
        assert setup.call_count == 1
        assert cleanup.call_count == 0

        assert channel.ValueType is None
        assert channel.UpdateType is None

        assert channel.get() == 5

        with pytest.raises(InvalidUpdateError):
            channel.update([5])  # type: ignore

    assert setup.call_count == 1
    assert cleanup.call_count == 1
