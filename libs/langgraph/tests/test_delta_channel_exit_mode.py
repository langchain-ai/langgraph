"""Tests for exit-mode delta channel persistence redesign.

Validates that `durability="exit"` correctly persists delta-channel writes
using count-based snapshot decisions (rather than force-snapshotting every
channel), lazy stub creation when no parent exists, and proper read-path
reconstruction via ancestor walks.
"""

import uuid
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import START, StateGraph
from langgraph.graph.message import _messages_delta_reducer
from langgraph.pregel._checkpoint import exit_delta_task_id

pytestmark = pytest.mark.anyio


def test_exit_delta_task_id_is_valid_uuid_and_ordered() -> None:
    """Exit-mode synthetic task ids must parse as UUID and sort by superstep."""
    tid = "4f7226e4-0270-bf16-1ef8-fb321bef9f3d"
    id1 = exit_delta_task_id(1, tid)
    id7 = exit_delta_task_id(7, tid)

    uuid.UUID(id1)
    uuid.UUID(id7)
    assert id1 < id7
    assert id1.split("-")[0] == "00000001"
    assert id7.split("-")[0] == "00000007"
    assert id1.endswith("-0270-bf16-1ef8-fb321bef9f3d")

    with pytest.raises(ValueError):
        uuid.UUID(f"00000001-{tid}")


def _build_graph(
    checkpointer: InMemorySaver,
    *,
    freq: int = 1000,
) -> Any:
    channel = DeltaChannel(_messages_delta_reducer, snapshot_frequency=freq)
    # Functional TypedDict form: class form can't reference `channel` (a
    # local variable) inside Annotated due to forward-ref evaluation rules.
    State = TypedDict("State", {"messages": Annotated[list, channel]})  # type: ignore[call-overload]  # noqa: UP013

    def respond(state: dict) -> dict:
        i = len(state["messages"])
        return {"messages": [AIMessage(content=f"reply-{i}", id=f"ai{i}")]}

    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 8a. Write-path / structural tests
# ---------------------------------------------------------------------------


async def test_exit_first_run_no_delta_writes() -> None:
    """Graph with delta channel invoked with input that doesn't touch it.
    Only one checkpoint row, no stub."""
    State = TypedDict(  # noqa: UP013
        "State",
        {
            "messages": Annotated[list, DeltaChannel(_messages_delta_reducer)],
            "value": str,
        },
    )  # type: ignore[call-overload]

    def noop(state: dict) -> dict:
        return {"value": "done"}

    saver = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node("noop", noop)
    builder.add_edge(START, "noop")
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "no-delta-writes"}}

    graph.invoke({"value": "start"}, config, durability="exit")

    checkpoints = list(saver.list(config))
    assert len(checkpoints) == 1
    stubs = [t for t in checkpoints if t.metadata.get("step") == -2]
    assert len(stubs) == 0


async def test_exit_first_run_all_snapshot() -> None:
    """snapshot_frequency=1 forces every channel to snapshot.
    No stub needed; final_checkpoint has _DeltaSnapshot."""
    saver = InMemorySaver()
    graph = _build_graph(saver, freq=1)
    config = {"configurable": {"thread_id": "all-snapshot"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="hi", id="h1")]},
        config,
        durability="exit",
    )
    assert len(result["messages"]) == 2

    checkpoints = list(saver.list(config))
    stubs = [t for t in checkpoints if t.metadata.get("step") == -2]
    assert len(stubs) == 0

    head = saver.get_tuple(config)
    assert head is not None
    assert isinstance(head.checkpoint["channel_values"].get("messages"), _DeltaSnapshot)

    state = graph.get_state(config)
    assert [m.content for m in state.values["messages"]] == ["hi", "reply-1"]


async def test_exit_first_run_sub_freq_with_writes() -> None:
    """First run with default snapshot_frequency (1000), writes below threshold.
    A stub is created; writes are anchored under it; get_state reconstructs."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "sub-freq-first"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="hello", id="h1")]},
        config,
        durability="exit",
    )
    assert [m.content for m in result["messages"]] == ["hello", "reply-1"]

    checkpoints = list(saver.list(config))
    stubs = [t for t in checkpoints if t.metadata.get("step") == -2]
    assert len(stubs) == 1, f"Expected 1 stub, got {len(stubs)}"

    head = saver.get_tuple(config)
    assert head is not None
    assert "messages" not in head.checkpoint["channel_values"]
    assert "messages" in head.checkpoint["channel_versions"]

    state = graph.get_state(config)
    assert [m.content for m in state.values["messages"]] == ["hello", "reply-1"]


async def test_exit_resumed_run_sub_freq() -> None:
    """Two consecutive exit runs. Second run anchors on the first's
    final_checkpoint (no new stub). Ordering preserved."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "resumed-sub-freq"}}

    graph.invoke(
        {"messages": [HumanMessage(content="msg1", id="h1")]},
        config,
        durability="exit",
    )

    graph.invoke(
        {"messages": [HumanMessage(content="msg2", id="h2")]},
        config,
        durability="exit",
    )

    checkpoints = list(saver.list(config))
    stubs = [t for t in checkpoints if t.metadata.get("step") == -2]
    assert len(stubs) == 1

    state = graph.get_state(config)
    contents = [m.content for m in state.values["messages"]]
    assert len(contents) == 4
    assert contents[0] == "msg1"
    assert contents[2] == "msg2"
    assert contents[0:4:2] == ["msg1", "msg2"]


async def test_exit_count_parity_sync_vs_exit() -> None:
    """Sync and exit durability produce the same update count in
    counters_since_delta_snapshot after an equivalent run."""
    for durability in ("sync", "exit"):
        saver = InMemorySaver()
        graph = _build_graph(saver)
        config = {"configurable": {"thread_id": f"parity-{durability}"}}

        graph.invoke(
            {"messages": [HumanMessage(content="hi", id="h1")]},
            config,
            durability=durability,
        )

        head = saver.get_tuple(config)
        assert head is not None
        counters = head.metadata.get("counters_since_delta_snapshot", {})
        updates, supersteps = counters.get("messages", (0, 0))
        assert updates == 2, (
            f"durability={durability}: expected updates=2, got {updates}"
        )
        assert supersteps >= 2, (
            f"durability={durability}: expected supersteps>=2, got {supersteps}"
        )


async def test_exit_snapshot_fires_at_frequency() -> None:
    """With snapshot_frequency=3, after 3 exit runs (each incrementing count
    by 2: input + superstep), the 2nd run hits count=4>=3, triggering snapshot.
    After that run, count resets to 0 and channel_values has _DeltaSnapshot."""
    saver = InMemorySaver()
    graph = _build_graph(saver, freq=3)
    config = {"configurable": {"thread_id": "snapshot-at-freq"}}

    graph.invoke(
        {"messages": [HumanMessage(content="m1", id="h1")]},
        config,
        durability="exit",
    )
    head = saver.get_tuple(config)
    assert head is not None
    counters1 = head.metadata.get("counters_since_delta_snapshot", {})
    updates1 = counters1.get("messages", (0, 0))[0]
    assert updates1 == 2

    graph.invoke(
        {"messages": [HumanMessage(content="m2", id="h2")]},
        config,
        durability="exit",
    )
    head = saver.get_tuple(config)
    assert head is not None
    counters2 = head.metadata.get("counters_since_delta_snapshot", {})
    updates2 = counters2.get("messages", (0, 0))[0]
    assert updates2 == 0, f"Expected reset to 0 after snapshot, got {updates2}"
    assert isinstance(head.checkpoint["channel_values"].get("messages"), _DeltaSnapshot)


async def test_exit_mixed_snapshot_and_non_snapshot() -> None:
    """One delta channel at freq=1 (always snapshot) and one at freq=1000
    (never snapshot within this test). Verify correct behavior for both."""

    fast_ch = DeltaChannel(_messages_delta_reducer, snapshot_frequency=1)
    slow_ch = DeltaChannel(_messages_delta_reducer, snapshot_frequency=1000)
    State = TypedDict(  # noqa: UP013
        "State",
        {"fast": Annotated[list, fast_ch], "slow": Annotated[list, slow_ch]},
    )  # type: ignore[call-overload]

    def respond(state: dict) -> dict:
        return {
            "fast": [AIMessage(content="fast-reply", id="f1")],
            "slow": [AIMessage(content="slow-reply", id="s1")],
        }

    saver = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "mixed-freq"}}

    graph.invoke(
        {
            "fast": [HumanMessage(content="fast-in", id="fi")],
            "slow": [HumanMessage(content="slow-in", id="si")],
        },
        config,
        durability="exit",
    )

    head = saver.get_tuple(config)
    assert head is not None
    assert isinstance(head.checkpoint["channel_values"].get("fast"), _DeltaSnapshot)
    assert "slow" not in head.checkpoint["channel_values"]

    state = graph.get_state(config)
    assert [m.content for m in state.values["fast"]] == ["fast-in", "fast-reply"]
    assert [m.content for m in state.values["slow"]] == ["slow-in", "slow-reply"]


# ---------------------------------------------------------------------------
# 8b. Read-path tests
# ---------------------------------------------------------------------------


async def test_exit_multi_run_replay_chain() -> None:
    """K=4 consecutive exit runs, each adding a message. After each run,
    get_state returns all messages in chronological order."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "replay-chain"}}

    for i in range(4):
        graph.invoke(
            {"messages": [HumanMessage(content=f"user-{i}", id=f"h{i}")]},
            config,
            durability="exit",
        )

        state = graph.get_state(config)
        contents = [m.content for m in state.values["messages"]]
        user_msgs = [c for c in contents if c.startswith("user-")]
        assert user_msgs == [f"user-{j}" for j in range(i + 1)], (
            f"After run {i}: user messages out of order: {user_msgs}"
        )
        assert len(contents) == (i + 1) * 2


async def test_exit_metadata_round_trip() -> None:
    """K=5 consecutive exit runs with snapshot_frequency=5. Verify metadata
    counters_since_delta_snapshot increments correctly across runs."""
    freq = 5
    saver = InMemorySaver()
    graph = _build_graph(saver, freq=freq)
    config = {"configurable": {"thread_id": "metadata-rt"}}

    for i in range(1, 6):
        graph.invoke(
            {"messages": [HumanMessage(content=f"m{i}", id=f"h{i}")]},
            config,
            durability="exit",
        )
        head = saver.get_tuple(config)
        assert head is not None
        counters = head.metadata.get("counters_since_delta_snapshot", {})
        updates = counters.get("messages", (0, 0))[0]
        cumulative = i * 2
        if cumulative >= freq:
            assert updates == 0 or updates == cumulative % freq or updates < freq, (
                f"After run {i}: updates={updates} should have reset or be partial"
            )
        else:
            assert updates == cumulative, (
                f"After run {i}: expected {cumulative}, got {updates}"
            )


async def test_exit_mixed_durability_round_trip() -> None:
    """Alternate sync and exit durability; verify counts stay monotonic
    and state accumulates correctly."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "mixed-durability"}}

    for i, dur in enumerate(["sync", "exit", "sync", "exit"]):
        graph.invoke(
            {"messages": [HumanMessage(content=f"msg-{i}", id=f"h{i}")]},
            config,
            durability=dur,
        )

        state = graph.get_state(config)
        contents = [m.content for m in state.values["messages"]]
        user_msgs = [c for c in contents if c.startswith("msg-")]
        assert user_msgs == [f"msg-{j}" for j in range(i + 1)], (
            f"After run {i} (durability={dur}): {user_msgs}"
        )
        assert len(contents) == (i + 1) * 2


async def test_exit_snapshot_then_tail_deltas() -> None:
    """Run 1 forces snapshot (freq=1). Run 2 at freq=1000 adds more writes
    that don't snapshot. Reading after run 2 must combine the snapshot seed
    with the tail deltas."""
    saver = InMemorySaver()

    graph1 = _build_graph(saver, freq=1)
    config = {"configurable": {"thread_id": "snapshot-then-tail"}}
    graph1.invoke(
        {"messages": [HumanMessage(content="seed-msg", id="h1")]},
        config,
        durability="exit",
    )

    head = saver.get_tuple(config)
    assert head is not None
    assert isinstance(head.checkpoint["channel_values"].get("messages"), _DeltaSnapshot)

    graph2 = _build_graph(saver, freq=1000)
    graph2.invoke(
        {"messages": [HumanMessage(content="tail-msg", id="h2")]},
        config,
        durability="exit",
    )

    state = graph2.get_state(config)
    contents = [m.content for m in state.values["messages"]]
    assert "seed-msg" in contents
    assert "tail-msg" in contents
    assert contents.index("seed-msg") < contents.index("tail-msg")
