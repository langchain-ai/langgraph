import operator
from collections.abc import Sequence

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import DELTA_SENTINEL

from langgraph._internal._typing import MISSING
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.delta import DeltaChannel
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.errors import EmptyChannelError, InvalidUpdateError
from langgraph.graph.message import add_messages

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
    """replay_writes on a fresh channel replays through the operator."""
    from langchain_core.messages import AIMessage, HumanMessage

    from langgraph.graph.message import add_messages

    spec = DeltaChannel(add_messages)
    ch = spec.from_checkpoint(DELTA_SENTINEL)
    ch.replay_writes(
        [
            ("t0", "messages", HumanMessage(content="hi", id="h1")),
            ("t1", "messages", AIMessage(content="hello", id="a1")),
            ("t2", "messages", HumanMessage(content="bye", id="h2")),
        ]
    )
    msgs = ch.get()
    assert len(msgs) == 3
    assert msgs[0].content == "hi"
    assert msgs[1].content == "hello"
    assert msgs[2].content == "bye"


def test_delta_channel_from_checkpoint_backwards_compat() -> None:
    from langchain_core.messages import HumanMessage

    from langgraph.graph.message import add_messages

    # Old BinaryOperatorAggregate checkpoint: plain list treated as backward compat
    spec = DeltaChannel(add_messages)
    old_value = [HumanMessage(content="old", id="h1")]
    ch = spec.from_checkpoint(old_value)
    assert ch.get() == old_value


def test_delta_channel_overwrite() -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.base import DELTA_SENTINEL

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
    ch2 = spec.from_checkpoint(DELTA_SENTINEL)
    ch2.replay_writes(
        [
            ("t0", "messages", HumanMessage(content="hi", id="h1")),
            ("t1", "messages", AIMessage(content="hello", id="a1")),
            ("t2", "messages", RemoveMessage(id="a1")),
        ]
    )
    assert ch2.get() == [HumanMessage(content="hi", id="h1")]


def test_delta_channel_update_by_id_and_replay() -> None:
    """Updating a message by ID must round-trip correctly through writes replay."""
    from langchain_core.messages import HumanMessage

    from langgraph.graph.message import add_messages

    spec = DeltaChannel(add_messages)
    ch = spec.from_checkpoint(MISSING)

    # Step 1: add a message
    ch.update([HumanMessage(content="original", id="h1")])

    # Step 2: update the same message by ID
    ch.update([HumanMessage(content="updated", id="h1")])
    assert ch.get() == [HumanMessage(content="updated", id="h1")]

    # Replay writes — must produce the updated message, not the original
    ch2 = spec.from_checkpoint(DELTA_SENTINEL)
    ch2.replay_writes(
        [
            ("t0", "messages", HumanMessage(content="original", id="h1")),
            ("t1", "messages", HumanMessage(content="updated", id="h1")),
        ]
    )
    assert len(ch2.get()) == 1
    assert ch2.get()[0].content == "updated"


def test_delta_channel_checkpoint_returns_sentinel() -> None:
    """checkpoint() always returns DELTA_SENTINEL regardless of state."""
    from langgraph.checkpoint.base import DELTA_SENTINEL

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

    # get_tuple returns raw storage shape — channel_values stores DELTA_SENTINEL
    # for delta channels; the reconstructed writes flow separately via
    # saver._get_channel_writes_history.
    saved = saver.get_tuple(config)
    assert saved is not None
    assert "messages" in saved.checkpoint["channel_values"]
    assert saved.checkpoint["channel_values"]["messages"] is DELTA_SENTINEL

    state = graph.get_state(config)
    assert len(state.values["messages"]) == 4  # 2 human + 2 AI


# ---------------------------------------------------------------------------
# Dict-reducer tests
# ---------------------------------------------------------------------------


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
    assert ch.is_available()
    assert ch.get() == {}


def test_delta_channel_dict_reducer_basic_updates() -> None:
    """DeltaChannel with a dict reducer accumulates key/value pairs across steps."""

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
    """replay_writes on a fresh channel replays through a dict merge reducer."""

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    spec = _delta_channel_with_type(merge_dicts, dict)
    ch = spec.from_checkpoint(DELTA_SENTINEL)
    ch.replay_writes(
        [
            ("t0", "files", {"a": 1}),
            ("t1", "files", {"b": 2}),
            ("t2", "files", {"c": 3}),
        ]
    )
    assert ch.get() == {"a": 1, "b": 2, "c": 3}


def test_delta_channel_dict_reducer_with_deletions() -> None:
    """Dict reducer that treats None values as deletions works end-to-end."""

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
    ch.update([{"file1.py": None, "file3.py": "content3"}])
    assert ch.get() == {"file2.py": "content2", "file3.py": "content3"}

    spec = _delta_channel_with_type(merge_files, dict)
    ch2 = spec.from_checkpoint(DELTA_SENTINEL)
    ch2.replay_writes(
        [
            ("t0", "files", {"file1.py": "content1", "file2.py": "content2"}),
            ("t1", "files", {"file1.py": None, "file3.py": "content3"}),
        ]
    )
    assert ch2.get() == {"file2.py": "content2", "file3.py": "content3"}


def test_delta_channel_dict_reducer_overwrite_in_update() -> None:
    """Overwrite(dict) in update() must preserve dict shape, not coerce to list."""
    from langgraph.types import Overwrite

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    ch = _delta_channel_with_type(merge_dicts, dict).from_checkpoint(MISSING)
    ch.update([{"a": 1}])
    ch.update([Overwrite({"b": 2, "c": 3})])
    assert ch.get() == {"b": 2, "c": 3}


def test_delta_channel_dict_reducer_overwrite_in_writes_replay() -> None:
    """Overwrite(dict) embedded in replayed writes must reconstruct as dict."""
    from langgraph.types import Overwrite

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    spec = _delta_channel_with_type(merge_dicts, dict)
    ch = spec.from_checkpoint(DELTA_SENTINEL)
    ch.replay_writes(
        [
            ("t0", "files", {"a": 1}),
            ("t1", "files", Overwrite({"x": 10, "y": 20})),
            ("t2", "files", {"z": 30}),
        ]
    )
    assert ch.get() == {"x": 10, "y": 20, "z": 30}


def test_delta_channel_dict_reducer_with_notrequired_annotation() -> None:
    """DeltaChannel infers dict type through `Annotated[NotRequired[dict[...]], ch]`."""
    from typing import Annotated

    from typing_extensions import NotRequired

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.state import _get_channel

    def merge_dicts(left: dict | None, right: dict) -> dict:
        if left is None:
            return dict(right)
        return {**left, **right}

    annotation = Annotated[NotRequired[dict[str, int]], DeltaChannel(merge_dicts)]
    ch = _get_channel("files", annotation).from_checkpoint(MISSING)
    assert ch.get() == {}
    ch.update([{"a": 1}])
    ch.update([{"b": 2}])
    assert ch.get() == {"a": 1, "b": 2}


def test_delta_channel_dict_reducer_end_to_end_filesystem() -> None:
    """End-to-end: graph with dict-reducer (filesystem-style) channel wrapped in DeltaChannel."""
    from typing import Annotated

    from langgraph.checkpoint.memory import InMemorySaver
    from typing_extensions import TypedDict

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph import START, StateGraph

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

    class State(TypedDict):
        files: Annotated[dict[str, str], DeltaChannel(merge_files)]

    turn = {"v": 0}

    def write_file(state: State) -> dict:
        turn["v"] += 1
        n = turn["v"]
        return {"files": {f"/doc_{n}.txt": f"content for turn {n}"}}

    builder = StateGraph(State)
    builder.add_node("write_file", write_file)
    builder.add_edge(START, "write_file")
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "fs"}}

    for _ in range(3):
        graph.invoke({"files": {}}, config)

    saved = saver.get_tuple(config)
    assert saved is not None
    assert saved.checkpoint["channel_values"]["files"] is DELTA_SENTINEL
    state = graph.get_state(config)
    assert state.values["files"] == {
        "/doc_1.txt": "content for turn 1",
        "/doc_2.txt": "content for turn 2",
        "/doc_3.txt": "content for turn 3",
    }

    def delete_file(state: State) -> dict:
        return {"files": {"/doc_1.txt": None}}

    builder2 = StateGraph(State)
    builder2.add_node("write_file", write_file)
    builder2.add_node("delete_file", delete_file)
    builder2.add_edge(START, "write_file")
    builder2.add_edge("write_file", "delete_file")
    turn["v"] = 0
    saver2 = InMemorySaver()
    graph2 = builder2.compile(checkpointer=saver2)
    config2 = {"configurable": {"thread_id": "fs2"}}
    graph2.invoke({"files": {}}, config2)
    state2 = graph2.get_state(config2)
    assert state2.values["files"] == {}


def test_delta_channel_dict_reducer_backwards_compat() -> None:
    """A pre-DeltaChannel dict checkpoint must load as a dict, not be listified."""

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    spec = _delta_channel_with_type(merge_dicts, dict)
    old_value = {"a": 1, "b": 2}
    ch = spec.from_checkpoint(old_value)
    assert ch.get() == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# seed / pre-delta migration
# ---------------------------------------------------------------------------


def test_delta_channel_from_checkpoint_honors_seed() -> None:
    """A non-sentinel value to from_checkpoint is used as the pre-delta seed.

    Guards the pre-delta migration path: when the saver's ancestor walk hits
    a pre-DeltaChannel blob it passes it as `seed` so replay reconstructs
    the post-migration state correctly rather than replaying from empty.
    """
    spec = DeltaChannel(add_messages)
    seed = [HumanMessage(content="pre-delta", id="p1")]
    ch = spec.from_checkpoint(seed)
    ch.replay_writes(
        [
            ("t0", "messages", AIMessage(content="delta-1", id="d1")),
            ("t1", "messages", HumanMessage(content="delta-2", id="d2")),
        ]
    )
    msgs = ch.get()
    assert [m.content for m in msgs] == ["pre-delta", "delta-1", "delta-2"]


def test_delta_channel_from_checkpoint_seed_without_writes() -> None:
    """Reconstruction at a pre-delta ancestor with no newer deltas returns
    just the seed — the saver's terminator fired immediately."""
    spec = DeltaChannel(add_messages)
    seed = [HumanMessage(content="only-snap", id="s1")]
    ch = spec.from_checkpoint(seed)
    ch.replay_writes([])
    assert ch.get() == seed


def test_delta_channel_from_checkpoint_seed_none_is_distinct_from_sentinel() -> None:
    """`seed=None` must start replay from None, not from an empty channel.

    The DELTA_SENTINEL / MISSING sentinels mean 'no seed'; passing `None`
    explicitly should feed None to the reducer as the left operand.
    """

    def replace(left, right):
        return right

    spec = DeltaChannel(replace)
    ch = spec.from_checkpoint(None)
    ch.replay_writes([("t0", "x", "after")])
    # Reducer replaces; seed=None → first write produces "after".
    assert ch.get() == "after"
