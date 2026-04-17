import operator
from collections.abc import Sequence

import pytest

from langgraph._internal._typing import MISSING
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.last_value import LastValue
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


def test_diff_channel_basic_two_steps() -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.base import DiffDelta

    from langgraph.channels.diff import DiffChannel
    from langgraph.graph.message import add_messages

    ch = DiffChannel(add_messages).from_checkpoint(MISSING)
    ch.after_checkpoint(None)

    # Step 1: one message added
    ch.update([HumanMessage(content="hi", id="h1")])
    d1 = ch.checkpoint()
    assert isinstance(d1, DiffDelta)
    assert len(d1.delta) == 1
    assert d1.prev_version is None  # first ever step
    ch.after_checkpoint("v1")

    # Step 2: another message
    ch.update([AIMessage(content="hello", id="a1")])
    d2 = ch.checkpoint()
    assert d2.prev_version == "v1"
    assert len(d2.delta) == 1
    ch.after_checkpoint("v2")

    # Full accumulated value is preserved in memory
    assert len(ch.get()) == 2
    assert ch.get()[0].content == "hi"
    assert ch.get()[1].content == "hello"


def test_diff_channel_after_checkpoint_no_op_when_unchanged() -> None:
    from langchain_core.messages import HumanMessage

    from langgraph.channels.diff import DiffChannel
    from langgraph.graph.message import add_messages

    ch = DiffChannel(add_messages).from_checkpoint(MISSING)
    ch.after_checkpoint(None)
    ch.update([HumanMessage(content="hi", id="h1")])
    ch.after_checkpoint("v1")

    # Same version: no-op
    ch.after_checkpoint("v1")
    assert ch._base_version == "v1"
    assert ch._pending == []


def test_diff_channel_from_checkpoint_chain() -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.base import DiffChainValue

    from langgraph.channels.diff import DiffChannel
    from langgraph.graph.message import add_messages

    spec = DiffChannel(add_messages)
    chain = DiffChainValue(
        base=None,
        deltas=[
            [HumanMessage(content="hi", id="h1")],
            [AIMessage(content="hello", id="a1")],
            [HumanMessage(content="bye", id="h2")],
        ],
    )
    ch = spec.from_checkpoint(chain)
    msgs = ch.get()
    assert len(msgs) == 3
    assert msgs[0].content == "hi"
    assert msgs[1].content == "hello"
    assert msgs[2].content == "bye"


def test_diff_channel_from_checkpoint_backwards_compat() -> None:
    from langchain_core.messages import HumanMessage

    from langgraph.channels.diff import DiffChannel
    from langgraph.graph.message import add_messages

    # Old BinaryOperatorAggregate checkpoint: plain list
    spec = DiffChannel(add_messages)
    old_value = [HumanMessage(content="old", id="h1")]
    ch = spec.from_checkpoint(old_value)
    assert ch.get() == old_value


def test_diff_channel_overwrite_resets_chain() -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.base import DiffDelta

    from langgraph.channels.diff import DiffChannel
    from langgraph.graph.message import add_messages
    from langgraph.types import Overwrite

    ch = DiffChannel(add_messages).from_checkpoint(MISSING)
    ch.after_checkpoint(None)
    ch.update([HumanMessage(content="old", id="h1")])
    ch.after_checkpoint("v1")

    # Overwrite should create a root blob (prev_version=None)
    ch.update([Overwrite([HumanMessage(content="new", id="h2")])])
    d = ch.checkpoint()
    assert isinstance(d, DiffDelta)
    assert d.prev_version is None  # chain root
    assert len(d.delta) == 1
    assert d.delta[0].content == "new"


def test_diff_channel_unsupported_saver_raises() -> None:
    from langgraph.checkpoint.base import DiffDelta

    from langgraph.channels.diff import DiffChannel
    from langgraph.graph.message import add_messages

    # If a saver returns a raw DiffDelta (unsupported), from_checkpoint raises
    spec = DiffChannel(add_messages)
    raw_delta = DiffDelta(delta=[], prev_version=None)
    with pytest.raises(ValueError, match="DiffChannel received a raw DiffDelta"):
        spec.from_checkpoint(raw_delta)
