"""Tests for the `Private` channel marker (GitHub issue #8644).

Covers:
  a) Compile-time: `Private`-annotated channels present in `compiled.channels`
     but absent from `compiled.stream_channels`.
  b) `get_state()` hides the private key from `.values`.
  c) `get_state_history()` hides the private key from every snapshot's `.values`.
  d) Restore/replay: internal checkpointing still works across turns.
  e) `NotRequired[Annotated[..., Private]]` unwrapping.
  f) Regression guard: schemas without `Private` compile identically to before.
"""

import operator
from typing import Annotated, Any
from typing_extensions import NotRequired, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.channels.last_value import LastValue
from langgraph.graph import END, START, Private, StateGraph



def _simple_reducer(a: list, b: list) -> list:
    return (a or []) + (b or [])

def test_private_channel_in_channels_not_stream_channels():
    """A `Private`-annotated field must be registered as a channel (checkpointable)
    but must be absent from stream_channels (hidden from public state reads)."""

    class State(TypedDict):
        public: str
        secret: Annotated[str, Private]

    def node(state: State) -> dict:
        return {"public": "hello", "secret": "hidden"}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    compiled = builder.compile()

    assert "secret" in compiled.channels
    assert isinstance(compiled.channels["secret"], LastValue)

    stream_ch = compiled.stream_channels_asis
    if isinstance(stream_ch, str):
        assert stream_ch != "secret"
    else:
        assert "secret" not in stream_ch

    assert "public" in stream_ch


def test_get_state_hides_private_key():
    """After invoking a graph, `get_state()` must not include the private
    channel in `StateSnapshot.values`."""

    class State(TypedDict):
        count: int
        _internal: Annotated[str, Private]

    def node(state: State) -> dict:
        return {"count": (state.get("count") or 0) + 1, "_internal": "secret-value"}

    saver = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    compiled = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "test-b"}}
    compiled.invoke({"count": 0}, config)

    snapshot = compiled.get_state(config)
    assert "count" in snapshot.values
    assert "_internal" not in snapshot.values


def test_get_state_history_hides_private_key():
    """Across multiple turns, every snapshot from `get_state_history()` must
    not contain the private channel key in `.values`."""

    class State(TypedDict):
        count: int
        _cache: Annotated[str, Private]

    def node(state: State) -> dict:
        n = (state.get("count") or 0) + 1
        return {"count": n, "_cache": f"cached-{n}"}

    saver = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    compiled = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "test-c"}}

    for _ in range(3):
        compiled.invoke({"count": 0}, config)

    history = list(compiled.get_state_history(config))
    assert len(history) > 0, "Expected at least one history snapshot"
    for snapshot in history:
        assert "_cache" not in snapshot.values, (
            f"Private key '_cache' must not appear in history snapshot: {snapshot.values}"
        )


def test_private_channel_checkpointed_and_restored():
    """The private channel must be checkpointed and restored correctly across
    turns, even though it is hidden from get_state(). We verify this by having
    a node read the private channel and use its value to produce a public output."""

    class State(TypedDict):
        public_sum: int
        _accumulator: Annotated[list[int], operator.add, Private]

    def node(state: State) -> dict:
        # Read the private accumulator (restored from checkpoint)
        acc = state.get("_accumulator") or []
        new_acc = acc + [1]
        return {"_accumulator": [1], "public_sum": sum(new_acc)}

    saver = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    compiled = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "test-d"}}

    result1 = compiled.invoke({"public_sum": 0, "_accumulator": []}, config)
    assert result1["public_sum"] == 1

    result2 = compiled.invoke({"public_sum": 0, "_accumulator": []}, config)
    assert result2["public_sum"] == 2

    result3 = compiled.invoke({"public_sum": 0, "_accumulator": []}, config)
    assert result3["public_sum"] == 3

    snapshot = compiled.get_state(config)
    assert "_accumulator" not in snapshot.values


def test_private_with_not_required_unwrapping():
    """`NotRequired[Annotated[bytes, Private]]` must still be detected correctly
    so the private marker unwrapping path from _get_channels works."""

    class State(TypedDict):
        public: str
        optional_secret: NotRequired[Annotated[str, Private]]

    def node(state: State) -> dict:
        return {"public": "hello", "optional_secret": "should-be-private"}

    saver = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    compiled = builder.compile(checkpointer=saver)

    assert "optional_secret" in compiled.channels

    stream_ch = compiled.stream_channels_asis
    if isinstance(stream_ch, str):
        assert stream_ch != "optional_secret"
    else:
        assert "optional_secret" not in stream_ch

    config = {"configurable": {"thread_id": "test-e"}}
    compiled.invoke({"public": "start"}, config)
    snapshot = compiled.get_state(config)
    assert "optional_secret" not in snapshot.values
    assert "public" in snapshot.values


def test_no_private_schema_unchanged():
    """A schema with no `Private` fields must produce `stream_channels` equal
    to all non-managed channels — identical to the pre-change behaviour."""

    class State(TypedDict):
        x: int
        y: str

    builder = StateGraph(State)
    builder.add_node("node", lambda s: s)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    compiled = builder.compile()

    stream_ch = compiled.stream_channels_asis
    assert isinstance(stream_ch, list), "Expected a list of channel names"

    # Both fields must be present — no leakage of Private logic to clean schemas
    assert "x" in stream_ch
    assert "y" in stream_ch

    assert compiled.builder.private_channels == set()


def test_private_bare_class_and_instance_both_work():
    """Both `Private` (bare class) and `Private()` (instance) should mark a
    channel as private."""

    class StateBare(TypedDict):
        pub: str
        priv: Annotated[str, Private]  # bare class

    class StateInstance(TypedDict):
        pub: str
        priv: Annotated[str, Private()]  # instance

    for State in [StateBare, StateInstance]:
        builder = StateGraph(State)
        builder.add_node("n", lambda s: s)
        builder.add_edge(START, "n")
        builder.add_edge("n", END)
        compiled = builder.compile()

        stream_ch = compiled.stream_channels_asis
        assert "priv" not in stream_ch, (
            f"'priv' should be excluded from stream_channels for {State.__name__}"
        )
        assert "pub" in stream_ch


def test_multiple_private_channels():
    """Multiple `Private`-annotated fields must all be excluded from
    stream_channels but all still be present in channels."""

    class State(TypedDict):
        visible: str
        secret_a: Annotated[str, Private]
        secret_b: Annotated[int, Private]

    builder = StateGraph(State)
    builder.add_node("n", lambda s: {"visible": "v", "secret_a": "a", "secret_b": 42})
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    saver = InMemorySaver()
    compiled = builder.compile(checkpointer=saver)

    assert "secret_a" in compiled.channels
    assert "secret_b" in compiled.channels

    stream_ch = compiled.stream_channels_asis
    assert "secret_a" not in stream_ch
    assert "secret_b" not in stream_ch
    assert "visible" in stream_ch

    config = {"configurable": {"thread_id": "test-h"}}
    compiled.invoke({"visible": "start", "secret_a": "", "secret_b": 0}, config)
    snapshot = compiled.get_state(config)
    assert "secret_a" not in snapshot.values
    assert "secret_b" not in snapshot.values
    assert "visible" in snapshot.values