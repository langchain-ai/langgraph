import operator
from collections.abc import Sequence

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import SEED_UNSET, DeltaChannelWrites

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
    """Overwrite(dict) embedded in DeltaChannelWrites must reconstruct as dict."""
    from langgraph.checkpoint.base import DeltaChannelWrites

    from langgraph.types import Overwrite

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    spec = _delta_channel_with_type(merge_dicts, dict)
    writes = DeltaChannelWrites(
        [
            {"a": 1},
            Overwrite({"x": 10, "y": 20}),
            {"z": 30},
        ]
    )
    ch = spec.from_checkpoint(writes)
    assert ch.get() == {"x": 10, "y": 20, "z": 30}


def test_delta_channel_dict_reducer_snapshot_write_preserves_shape() -> None:
    """snapshot_write() on a dict channel must emit Overwrite(dict), not Overwrite(list)."""
    from langgraph.channels.delta import DeltaChannel
    from langgraph.types import Overwrite

    def merge_dicts(left: dict, right: dict) -> dict:
        return {**left, **right}

    spec = _delta_channel_with_type(merge_dicts, dict)
    # Force snapshot_every by reaching into the spec instance.
    assert isinstance(spec, DeltaChannel)
    spec.snapshot_every = 2

    ch = spec.from_checkpoint(MISSING)
    ch.update([{"a": 1}])
    ch.update([{"b": 2}])
    assert ch.should_snapshot()

    w = ch.snapshot_write()
    assert isinstance(w, Overwrite)
    assert w.value == {"a": 1, "b": 2}
    assert isinstance(w.value, dict)


def test_delta_channel_dict_reducer_with_notrequired_annotation() -> None:
    """DeltaChannel infers dict type through `Annotated[NotRequired[dict[...]], ch]`.

    This is the shape the deepagents filesystem middleware uses for its
    `files` field; without unwrapping NotRequired we'd fall through to `list`
    and blow up on the first dict operator call.
    """
    from typing import Annotated

    from typing_extensions import NotRequired

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.state import _get_channel

    def merge_dicts(left: dict | None, right: dict) -> dict:
        if left is None:
            return dict(right)
        return {**left, **right}

    annotation = Annotated[
        NotRequired[dict[str, int]],
        DeltaChannel(merge_dicts),
    ]
    ch = _get_channel("files", annotation).from_checkpoint(MISSING)
    assert ch.get() == {}
    ch.update([{"a": 1}])
    ch.update([{"b": 2}])
    assert ch.get() == {"a": 1, "b": 2}


def test_delta_channel_dict_reducer_end_to_end_filesystem() -> None:
    """End-to-end: graph with dict-reducer (filesystem-style) channel wrapped in DeltaChannel.

    Mirrors the deepagents filesystem pattern: `files: Annotated[dict, reducer]`
    where the reducer merges dicts and treats None values as deletions.
    """
    from typing import Annotated

    from langgraph.checkpoint.base import DELTA_SENTINEL, DeltaChannelWrites
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

    # Checkpoint stores only the sentinel — per-step writes live in checkpoint_writes.
    saved = saver.get_tuple(config)
    assert saved is not None
    cv = saved.checkpoint["channel_values"]["files"]
    assert cv is not DELTA_SENTINEL
    assert isinstance(cv, DeltaChannelWrites)

    state = graph.get_state(config)
    assert state.values["files"] == {
        "/doc_1.txt": "content for turn 1",
        "/doc_2.txt": "content for turn 2",
        "/doc_3.txt": "content for turn 3",
    }

    # Deletion path must round-trip through writes replay.
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
# snapshot_every
# ---------------------------------------------------------------------------


def test_delta_channel_snapshot_counter_triggers() -> None:
    """Counter hits threshold → should_snapshot() true; snapshot_write() resets it."""
    from langchain_core.messages import HumanMessage

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages
    from langgraph.types import Overwrite

    ch = DeltaChannel(add_messages, snapshot_every=3).from_checkpoint(MISSING)
    assert not ch.should_snapshot()

    ch.update([HumanMessage(content="a", id="1")])
    assert not ch.should_snapshot()
    ch.update([HumanMessage(content="b", id="2")])
    assert not ch.should_snapshot()
    ch.update([HumanMessage(content="c", id="3")])
    assert ch.should_snapshot()

    w = ch.snapshot_write()
    assert isinstance(w, Overwrite)
    assert len(w.value) == 3
    # counter reset
    assert not ch.should_snapshot()


def test_delta_channel_snapshot_default_disabled() -> None:
    """No snapshot_every → should_snapshot() is never true."""
    from langchain_core.messages import HumanMessage

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages

    ch = DeltaChannel(add_messages).from_checkpoint(MISSING)
    for i in range(50):
        ch.update([HumanMessage(content=str(i), id=str(i))])
    assert not ch.should_snapshot()


def test_delta_channel_user_overwrite_resets_counter() -> None:
    """An Overwrite (user or framework) resets the snapshot counter."""
    from langchain_core.messages import HumanMessage

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages
    from langgraph.types import Overwrite

    ch = DeltaChannel(add_messages, snapshot_every=3).from_checkpoint(MISSING)
    ch.update([HumanMessage(content="a", id="1")])
    ch.update([HumanMessage(content="b", id="2")])
    # Right before threshold — user Overwrite should reset.
    ch.update([Overwrite([HumanMessage(content="new", id="new")])])
    assert not ch.should_snapshot()
    # Need 3 more writes to trigger.
    ch.update([HumanMessage(content="c", id="c")])
    ch.update([HumanMessage(content="d", id="d")])
    ch.update([HumanMessage(content="e", id="e")])
    assert ch.should_snapshot()


def test_delta_channel_from_checkpoint_honors_seed() -> None:
    """DeltaChannelWrites(seed=...) starts replay from that snapshot.

    Guards the pre-delta migration path: when the saver's ancestor walk hits
    a pre-DeltaChannel blob it passes it as `seed` so replay reconstructs
    the post-migration state correctly rather than replaying from empty.
    """
    spec = DeltaChannel(add_messages)
    seed = [HumanMessage(content="pre-delta", id="p1")]
    writes = DeltaChannelWrites(
        writes=[
            AIMessage(content="delta-1", id="d1"),
            HumanMessage(content="delta-2", id="d2"),
        ],
        seed=seed,
    )
    ch = spec.from_checkpoint(writes)
    msgs = ch.get()
    assert [m.content for m in msgs] == ["pre-delta", "delta-1", "delta-2"]


def test_delta_channel_from_checkpoint_seed_without_writes() -> None:
    """Reconstruction at a pre-delta ancestor with no newer deltas returns
    just the seed — the saver's terminator fired immediately."""
    spec = DeltaChannel(add_messages)
    seed = [HumanMessage(content="only-snap", id="s1")]
    ch = spec.from_checkpoint(DeltaChannelWrites(writes=[], seed=seed))
    assert ch.get() == seed


def test_delta_channel_from_checkpoint_seed_none_is_distinct_from_unset() -> None:
    """`seed=None` must start replay from None, not from the channel's empty
    value. `SEED_UNSET` is the sentinel meaning 'no seed'."""

    def replace(left, right):
        return right

    spec = DeltaChannel(replace)
    ch = spec.from_checkpoint(DeltaChannelWrites(writes=["after"], seed=None))
    # Reducer replaces; seed=None → first write produces "after".
    assert ch.get() == "after"
    # And the default (unset) is distinct.
    unset = DeltaChannelWrites(writes=["after"])
    assert unset.seed is SEED_UNSET


def test_delta_channel_replay_tracks_counter_across_overwrite() -> None:
    """Counter reloaded from writes reflects writes-since-last-Overwrite."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.base import DeltaChannelWrites

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph.message import add_messages
    from langgraph.types import Overwrite

    spec = DeltaChannel(add_messages, snapshot_every=3)
    writes = DeltaChannelWrites(
        [
            HumanMessage(content="a", id="1"),
            HumanMessage(content="b", id="2"),
            HumanMessage(content="c", id="3"),
            Overwrite([HumanMessage(content="reset", id="r")]),
            HumanMessage(content="d", id="4"),
        ]
    )
    ch = spec.from_checkpoint(writes)
    # Post-snapshot: one write since reset.
    assert ch._writes_since_snapshot == 1
    assert not ch.should_snapshot()
    assert len(ch.get()) == 2  # reset → [r], +d → [r, d]


def test_delta_channel_snapshot_end_to_end_inmemory() -> None:
    """Full graph: snapshot is injected, descendants short-circuit at it."""
    from typing import Annotated

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from typing_extensions import TypedDict

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph import START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.types import Overwrite

    class State(TypedDict):
        messages: Annotated[list, DeltaChannel(add_messages, snapshot_every=2)]

    n = {"v": 0}

    def respond(state: State) -> dict:
        n["v"] += 1
        return {"messages": [AIMessage(content=f"r{n['v']}", id=f"ai{n['v']}")]}

    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "snap"}}

    # 3 turns × (HumanMessage input + AIMessage reply) — plenty to trigger
    # multiple snapshots at snapshot_every=2.
    for i in range(3):
        graph.invoke({"messages": [HumanMessage(content=f"q{i}", id=f"h{i}")]}, config)

    # Final state has 6 messages (3 H + 3 AI) regardless of snapshots.
    state = graph.get_state(config)
    assert len(state.values["messages"]) == 6

    # At least one Overwrite snapshot write was injected into checkpoint_writes.
    all_writes = [
        v
        for wdict in saver.writes.values()
        for (_task, _idx), (_, ch, serialized, _) in wdict.items()
        if ch == "messages"
        for v in [saver.serde.loads_typed(serialized)]
    ]
    overwrites = [w for w in all_writes if isinstance(w, Overwrite)]
    assert len(overwrites) >= 1, (
        f"expected at least one snapshot Overwrite, got writes: {all_writes}"
    )


def test_delta_channel_snapshot_preserves_time_travel() -> None:
    """Time-travel to a checkpoint created before a snapshot still replays correctly."""
    from typing import Annotated

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from typing_extensions import TypedDict

    from langgraph.channels.delta import DeltaChannel
    from langgraph.graph import START, StateGraph
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, DeltaChannel(add_messages, snapshot_every=2)]

    def respond(state: State) -> dict:
        n = len(state["messages"])
        return {"messages": [AIMessage(content=f"r{n}", id=f"ai{n}")]}

    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "snap-tt"}}

    for i in range(4):
        graph.invoke({"messages": [HumanMessage(content=f"q{i}", id=f"h{i}")]}, config)

    # Walk history; each snapshot has the expected message count at that point.
    history = list(graph.get_state_history(config))
    # Reverse to chronological order.
    history = list(reversed(history))
    # At each point the visible message count should monotonically grow.
    counts = [len(h.values.get("messages", [])) for h in history]
    assert counts == sorted(counts), f"message counts not monotonic: {counts}"
