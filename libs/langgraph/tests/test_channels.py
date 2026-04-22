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
    from langgraph.checkpoint.base import DeltaValue

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    ch = DeltaChannel(add_messages).from_checkpoint(MISSING)
    ch.after_checkpoint(None)

    # Step 1: one message added
    ch.update([HumanMessage(content="hi", id="h1")])
    d1 = ch.checkpoint()
    assert isinstance(d1, DeltaValue)
    assert len(d1.delta) == 1
    ch.after_checkpoint("v1", checkpoint_id="cid1")

    # Step 2: another message
    ch.update([AIMessage(content="hello", id="a1")])
    d2 = ch.checkpoint()
    assert len(d2.delta) == 1
    ch.after_checkpoint("v2")

    # Full accumulated value is preserved in memory
    assert len(ch.get()) == 2
    assert ch.get()[0].content == "hi"
    assert ch.get()[1].content == "hello"


def test_delta_channel_after_checkpoint_no_op_when_unchanged() -> None:
    from langchain_core.messages import HumanMessage

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    ch = DeltaChannel(add_messages).from_checkpoint(MISSING)
    ch.after_checkpoint(None)
    ch.update([HumanMessage(content="hi", id="h1")])
    ch.after_checkpoint("v1")

    # Same version: no-op
    ch.after_checkpoint("v1")
    assert ch._base_version == "v1"
    assert ch._pending == []


def test_delta_channel_from_checkpoint_chain() -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.base import DeltaChainValue

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    spec = DeltaChannel(add_messages)
    chain = DeltaChainValue(
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


def test_delta_channel_from_checkpoint_backwards_compat() -> None:
    from langchain_core.messages import HumanMessage

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    # Old BinaryOperatorAggregate checkpoint: plain list
    spec = DeltaChannel(add_messages)
    old_value = [HumanMessage(content="old", id="h1")]
    ch = spec.from_checkpoint(old_value)
    assert ch.get() == old_value


def test_delta_channel_overwrite_resets_chain() -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.base import DeltaValue

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages
    from langgraph.types import Overwrite

    ch = DeltaChannel(add_messages).from_checkpoint(MISSING)
    ch.after_checkpoint(None)
    ch.update([HumanMessage(content="old", id="h1")])
    ch.after_checkpoint("v1")

    ch.update([Overwrite([HumanMessage(content="new", id="h2")])])
    d = ch.checkpoint()
    assert isinstance(d, DeltaValue)
    assert len(d.delta) == 1
    assert d.delta[0].content == "new"
    # _overwritten flag must be set so next checkpoint acts as a chain root
    assert ch._overwritten is True


def test_delta_channel_unsupported_saver_raises() -> None:
    """from_checkpoint raises ValueError when the saver returns a raw DeltaValue."""
    from langgraph.checkpoint.base import DeltaValue

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    spec = DeltaChannel(add_messages)
    raw = DeltaValue(delta=[{"type": "human", "content": "hello"}])
    with pytest.raises(
        ValueError, match="does not support incremental channel storage"
    ):
        spec.from_checkpoint(raw)


def test_delta_channel_remove_message_delta_and_replay() -> None:
    """RemoveMessage stored in a delta must round-trip correctly through the chain."""
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
    from langgraph.checkpoint.base import DeltaChainValue, DeltaValue

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    spec = DeltaChannel(add_messages)
    ch = spec.from_checkpoint(MISSING)
    ch.after_checkpoint(None)

    # Step 1: add two messages
    ch.update([HumanMessage(content="hi", id="h1")])
    ch.update([AIMessage(content="hello", id="a1")])
    d1 = ch.checkpoint()
    assert isinstance(d1, DeltaValue)
    ch.after_checkpoint("v1", checkpoint_id="cid1")
    assert ch.get() == [
        HumanMessage(content="hi", id="h1"),
        AIMessage(content="hello", id="a1"),
    ]

    # Step 2: remove the AI message
    ch.update([RemoveMessage(id="a1")])
    d2 = ch.checkpoint()
    assert isinstance(d2, DeltaValue)
    assert any(isinstance(w, RemoveMessage) for w in d2.delta)
    ch.after_checkpoint("v2", checkpoint_id="cid2")
    assert ch.get() == [HumanMessage(content="hi", id="h1")]

    # Replay the full chain from scratch — must reproduce the post-remove state
    chain = DeltaChainValue(base=None, deltas=[d1.delta, d2.delta])
    ch2 = spec.from_checkpoint(chain)
    assert ch2.get() == [HumanMessage(content="hi", id="h1")]


def test_delta_channel_update_by_id_delta_and_replay() -> None:
    """Updating a message by ID stored in a delta must round-trip correctly."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.base import DeltaChainValue, DeltaValue

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    spec = DeltaChannel(add_messages)
    ch = spec.from_checkpoint(MISSING)
    ch.after_checkpoint(None)

    # Step 1: add a message
    ch.update([HumanMessage(content="original", id="h1")])
    d1 = ch.checkpoint()
    assert isinstance(d1, DeltaValue)
    ch.after_checkpoint("v1", checkpoint_id="cid1")

    # Step 2: update the same message by ID
    ch.update([HumanMessage(content="updated", id="h1")])
    d2 = ch.checkpoint()
    assert isinstance(d2, DeltaValue)
    ch.after_checkpoint("v2", checkpoint_id="cid2")
    assert ch.get() == [HumanMessage(content="updated", id="h1")]

    # Replay the full chain — must produce the updated message, not the original
    chain = DeltaChainValue(base=None, deltas=[d1.delta, d2.delta])
    ch2 = spec.from_checkpoint(chain)
    assert len(ch2.get()) == 1
    assert ch2.get()[0].content == "updated"


def test_delta_channel_snapshot_every_emits_plain_list() -> None:
    """snapshot_every=N causes a plain-list snapshot after N steps; next deltas chain to it."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.base import DeltaValue

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    SNAP = 3
    spec = DeltaChannel(add_messages, snapshot_every=SNAP)
    ch = spec.from_checkpoint(MISSING)
    # First after_checkpoint anchors _base_version without counting a step.
    ch.after_checkpoint("v0", checkpoint_id="cid0")

    # Steps 1..SNAP: each should stay as DeltaValue; counter increments each step.
    for i in range(1, SNAP + 1):
        ch.update([HumanMessage(content=f"m{i}", id=f"h{i}")])
        ckpt = ch.checkpoint()
        assert isinstance(ckpt, DeltaValue), f"expected DeltaValue at step {i}"
        ch.after_checkpoint(f"v{i}", checkpoint_id=f"cid{i}")

    # Step SNAP+1: _steps_since_snapshot == SNAP → snapshot fires
    ch.update([HumanMessage(content="snap", id="hsnap")])
    snap = ch.checkpoint()
    assert isinstance(snap, list), "expected plain-list snapshot at snapshot_every step"
    assert len(snap) == SNAP + 1

    # After snapshot, counter resets — next step is DeltaValue again
    ch.after_checkpoint("vsnap", checkpoint_id="cidsnap")
    ch.update([HumanMessage(content="post", id="hpost")])
    post = ch.checkpoint()
    assert isinstance(post, DeltaValue)


def test_delta_channel_snapshot_every_end_to_end() -> None:
    """Graph with snapshot_every: get_state returns correct accumulated value after snapshot."""
    from typing import Annotated

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from typing_extensions import TypedDict

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph import START, StateGraph
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, DeltaChannel(add_messages, snapshot_every=2)]

    counter = {"n": 0}

    def respond(state: State) -> dict:
        counter["n"] += 1
        return {
            "messages": [
                AIMessage(content=f"ai-{counter['n']}", id=f"ai-{counter['n']}")
            ]
        }

    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "snap-test"}}

    # Run 5 turns — snapshot fires after 2 steps, then again after 2 more
    for i in range(5):
        graph.invoke({"messages": [HumanMessage(content=f"h{i}", id=f"h{i}")]}, config)

    state = graph.get_state(config)
    msgs = state.values["messages"]
    # 5 human + 5 AI = 10 total
    assert len(msgs) == 10, f"expected 10 messages, got {len(msgs)}: {msgs}"


def test_delta_channel_inmemory_saver_assembles_chain() -> None:
    """InMemorySaver assembles the delta chain inside get_tuple (no pregel involvement)."""
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

    # get_tuple must return a fully assembled DeltaChainValue, not raw DeltaValue
    from langgraph.checkpoint.base import DeltaChainValue, DeltaValue

    saved = saver.get_tuple(config)
    assert saved is not None
    assert "messages" in saved.checkpoint["channel_values"]
    assert not isinstance(saved.checkpoint["channel_values"]["messages"], DeltaValue)
    assert isinstance(saved.checkpoint["channel_values"]["messages"], DeltaChainValue)

    state = graph.get_state(config)
    assert len(state.values["messages"]) == 4  # 2 human + 2 AI


def test_delta_channel_dict_reducer_fresh_channel() -> None:
    """DeltaChannel with a dict reducer starts as empty dict on MISSING checkpoint."""
    from langgraph.channels.delta import DeltaChannel

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    ch = DeltaChannel(merge_dicts, dict).from_checkpoint(MISSING)
    # Should be available (not raise EmptyChannelError) and start empty
    assert ch.is_available()
    assert ch.get() == {}


def test_delta_channel_dict_reducer_basic_updates() -> None:
    """DeltaChannel with a dict reducer accumulates key/value pairs across steps."""
    from langgraph.checkpoint.base import DeltaValue

    from langgraph.channels.delta import DeltaChannel

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    ch = DeltaChannel(merge_dicts, dict).from_checkpoint(MISSING)
    ch.after_checkpoint(None)

    ch.update([{"a": 1}])
    d1 = ch.checkpoint()
    assert isinstance(d1, DeltaValue)
    assert d1.delta == [{"a": 1}]
    ch.after_checkpoint("v1", checkpoint_id="cid1")

    ch.update([{"b": 2}])
    d2 = ch.checkpoint()
    assert d2.delta == [{"b": 2}]
    ch.after_checkpoint("v2")

    assert ch.get() == {"a": 1, "b": 2}


def test_delta_channel_dict_reducer_chain_reconstruction() -> None:
    """DeltaChainValue replays correctly through a dict merge reducer."""
    from langgraph.checkpoint.base import DeltaChainValue

    from langgraph.channels.delta import DeltaChannel

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    spec = DeltaChannel(merge_dicts, dict)
    chain = DeltaChainValue(
        base={"a": 1},
        deltas=[[{"b": 2}], [{"c": 3}]],
    )
    ch = spec.from_checkpoint(chain)
    assert ch.get() == {"a": 1, "b": 2, "c": 3}
    assert ch._steps_since_snapshot == 2


def test_delta_channel_dict_reducer_with_deletions() -> None:
    """Dict reducer that treats None values as deletions works end-to-end (deepagents pattern)."""
    from langgraph.checkpoint.base import DeltaChainValue

    from langgraph.channels.delta import DeltaChannel

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

    ch = DeltaChannel(merge_files, dict).from_checkpoint(MISSING)
    ch.after_checkpoint(None)

    ch.update([{"file1.py": "content1", "file2.py": "content2"}])
    ch.after_checkpoint("v1", checkpoint_id="cid1")

    # Delete file1, add file3
    ch.update([{"file1.py": None, "file3.py": "content3"}])
    ch.after_checkpoint("v2", checkpoint_id="cid2")

    assert ch.get() == {"file2.py": "content2", "file3.py": "content3"}

    # Confirm chain reconstruction produces the same result
    chain = DeltaChainValue(
        base={},
        deltas=[
            [{"file1.py": "content1", "file2.py": "content2"}],
            [{"file1.py": None, "file3.py": "content3"}],
        ],
    )
    spec = DeltaChannel(merge_files, dict)
    ch2 = spec.from_checkpoint(chain)
    assert ch2.get() == {"file2.py": "content2", "file3.py": "content3"}
