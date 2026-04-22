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


def test_delta_channel_basic_two_steps() -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.base import DELTA_SENTINEL

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    ch = DeltaChannel(add_messages).from_checkpoint(MISSING)

    # Step 1: one message added
    ch.update([HumanMessage(content="hi", id="h1")])
    d1 = ch.checkpoint()
    assert d1 is DELTA_SENTINEL

    # Step 2: another message
    ch.update([AIMessage(content="hello", id="a1")])
    d2 = ch.checkpoint()
    assert d2 is DELTA_SENTINEL

    # Full accumulated value is preserved in memory
    assert len(ch.get()) == 2
    assert ch.get()[0].content == "hi"
    assert ch.get()[1].content == "hello"


def test_delta_channel_from_checkpoint_writes_list() -> None:
    """from_checkpoint given DeltaChannelWrites replays through the operator."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.base import DeltaChannelWrites

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    spec = DeltaChannel(add_messages)
    writes = DeltaChannelWrites(
        [
            HumanMessage(content="hi", id="h1"),
            AIMessage(content="hello", id="a1"),
            HumanMessage(content="bye", id="h2"),
        ]
    )
    ch = spec.from_checkpoint(writes)
    msgs = ch.get()
    assert len(msgs) == 3
    assert msgs[0].content == "hi"
    assert msgs[1].content == "hello"
    assert msgs[2].content == "bye"


def test_delta_channel_from_checkpoint_backwards_compat() -> None:
    from langchain_core.messages import HumanMessage

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    # Old BinaryOperatorAggregate checkpoint: plain list treated as backward compat
    spec = DeltaChannel(add_messages)
    old_value = [HumanMessage(content="old", id="h1")]
    ch = spec.from_checkpoint(old_value)
    assert ch.get() == old_value


def test_delta_channel_overwrite() -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.base import DELTA_SENTINEL

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages
    from langgraph.types import Overwrite

    ch = DeltaChannel(add_messages).from_checkpoint(MISSING)
    ch.update([HumanMessage(content="old", id="h1")])

    ch.update([Overwrite([HumanMessage(content="new", id="h2")])])
    d = ch.checkpoint()
    assert d is DELTA_SENTINEL
    # After overwrite, value is reset to only the new message
    assert len(ch.get()) == 1
    assert ch.get()[0].content == "new"


def test_delta_channel_remove_message_and_replay() -> None:
    """RemoveMessage must round-trip correctly when writes are replayed."""
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    spec = DeltaChannel(add_messages)
    ch = spec.from_checkpoint(MISSING)

    # Step 1: add two messages
    ch.update([HumanMessage(content="hi", id="h1")])
    ch.update([AIMessage(content="hello", id="a1")])
    assert ch.get() == [
        HumanMessage(content="hi", id="h1"),
        AIMessage(content="hello", id="a1"),
    ]

    # Step 2: remove the AI message
    ch.update([RemoveMessage(id="a1")])
    assert ch.get() == [HumanMessage(content="hi", id="h1")]

    # Replay the writes list from scratch — must reproduce the post-remove state
    from langgraph.checkpoint.base import DeltaChannelWrites

    writes = DeltaChannelWrites(
        [
            HumanMessage(content="hi", id="h1"),
            AIMessage(content="hello", id="a1"),
            RemoveMessage(id="a1"),
        ]
    )
    ch2 = spec.from_checkpoint(writes)
    assert ch2.get() == [HumanMessage(content="hi", id="h1")]


def test_delta_channel_update_by_id_and_replay() -> None:
    """Updating a message by ID must round-trip correctly through writes replay."""
    from langchain_core.messages import HumanMessage

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    spec = DeltaChannel(add_messages)
    ch = spec.from_checkpoint(MISSING)

    # Step 1: add a message
    ch.update([HumanMessage(content="original", id="h1")])

    # Step 2: update the same message by ID
    ch.update([HumanMessage(content="updated", id="h1")])
    assert ch.get() == [HumanMessage(content="updated", id="h1")]

    # Replay writes — must produce the updated message, not the original
    from langgraph.checkpoint.base import DeltaChannelWrites

    writes = DeltaChannelWrites(
        [
            HumanMessage(content="original", id="h1"),
            HumanMessage(content="updated", id="h1"),
        ]
    )
    ch2 = spec.from_checkpoint(writes)
    assert len(ch2.get()) == 1
    assert ch2.get()[0].content == "updated"


def test_delta_channel_checkpoint_returns_sentinel() -> None:
    """checkpoint() always returns DELTA_SENTINEL regardless of state."""
    from langgraph.checkpoint.base import DELTA_SENTINEL

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    ch = DeltaChannel(add_messages).from_checkpoint(MISSING)
    assert ch.checkpoint() is DELTA_SENTINEL

    from langchain_core.messages import HumanMessage

    ch.update([HumanMessage(content="hi", id="h1")])
    assert ch.checkpoint() is DELTA_SENTINEL


def test_delta_channel_inmemory_saver_assembles_writes() -> None:
    """InMemorySaver assembles writes from checkpoint_writes inside get_tuple."""
    from typing import Annotated

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from typing_extensions import TypedDict

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph import START, StateGraph
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, DeltaChannel(add_messages)]

    n = {"v": 0}

    def respond(state: State) -> dict:
        n["v"] += 1
        return {"messages": [AIMessage(content=f"ok{n['v']}", id=f"ai{n['v']}")]}

    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "t1"}}

    graph.invoke({"messages": [HumanMessage(content="hi", id="h1")]}, config)
    graph.invoke({"messages": [HumanMessage(content="bye", id="h2")]}, config)

    # get_tuple must return a DeltaChannelWrites wrapper (not the raw sentinel)
    from langgraph.checkpoint.base import DELTA_SENTINEL, DeltaChannelWrites

    saved = saver.get_tuple(config)
    assert saved is not None
    assert "messages" in saved.checkpoint["channel_values"]
    assert saved.checkpoint["channel_values"]["messages"] is not DELTA_SENTINEL
    assert isinstance(
        saved.checkpoint["channel_values"]["messages"], DeltaChannelWrites
    )

    state = graph.get_state(config)
    assert len(state.values["messages"]) == 4  # 2 human + 2 AI


def _delta_channel_with_type(operator, typ):
    """Build a DeltaChannel with an explicit type via the Annotated injection path."""
    from typing import Annotated

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.state import _get_channel

    return _get_channel("_test", Annotated[typ, DeltaChannel(operator)])


def test_delta_channel_dict_reducer_fresh_channel() -> None:
    """DeltaChannel with a dict reducer starts as empty dict on MISSING checkpoint."""

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    ch = _delta_channel_with_type(merge_dicts, dict).from_checkpoint(MISSING)
    # Should be available (not raise EmptyChannelError) and start empty
    assert ch.is_available()
    assert ch.get() == {}


def test_delta_channel_dict_reducer_basic_updates() -> None:
    """DeltaChannel with a dict reducer accumulates key/value pairs across steps."""
    from langgraph.checkpoint.base import DELTA_SENTINEL

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    ch = _delta_channel_with_type(merge_dicts, dict).from_checkpoint(MISSING)

    ch.update([{"a": 1}])
    d1 = ch.checkpoint()
    assert d1 is DELTA_SENTINEL

    ch.update([{"b": 2}])
    d2 = ch.checkpoint()
    assert d2 is DELTA_SENTINEL

    assert ch.get() == {"a": 1, "b": 2}


def test_delta_channel_dict_reducer_writes_reconstruction() -> None:
    """from_checkpoint given DeltaChannelWrites replays through a dict merge reducer."""
    from langgraph.checkpoint.base import DeltaChannelWrites

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    spec = _delta_channel_with_type(merge_dicts, dict)
    writes = DeltaChannelWrites([{"a": 1}, {"b": 2}, {"c": 3}])
    ch = spec.from_checkpoint(writes)
    assert ch.get() == {"a": 1, "b": 2, "c": 3}


def test_delta_channel_dict_reducer_with_deletions() -> None:
    """Dict reducer that treats None values as deletions works end-to-end (deepagents pattern)."""

    def merge_files(left: dict | None, right: dict) -> dict:
        if left is None:
            return {k: v for k, v in right.items() if v is not None}
        result = {**left}
        for k, v in right.items():
            if v is None:
                result.pop(k, None)
            else:
                result[k] = v
        return result

    ch = _delta_channel_with_type(merge_files, dict).from_checkpoint(MISSING)

    ch.update([{"file1.py": "content1", "file2.py": "content2"}])

    # Delete file1, add file3
    ch.update([{"file1.py": None, "file3.py": "content3"}])

    assert ch.get() == {"file2.py": "content2", "file3.py": "content3"}

    # Confirm writes reconstruction produces the same result
    from langgraph.checkpoint.base import DeltaChannelWrites

    writes = DeltaChannelWrites(
        [
            {"file1.py": "content1", "file2.py": "content2"},
            {"file1.py": None, "file3.py": "content3"},
        ]
    )
    spec = _delta_channel_with_type(merge_files, dict)
    ch2 = spec.from_checkpoint(writes)
    assert ch2.get() == {"file2.py": "content2", "file3.py": "content3"}
