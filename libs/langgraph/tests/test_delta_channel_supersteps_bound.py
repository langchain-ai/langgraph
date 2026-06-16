"""Tests for the supersteps-since-last-snapshot bound on DeltaChannel.

Validates that a delta channel which stops receiving writes is still
force-snapshotted after DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT supersteps,
preventing unbounded ancestor walks.
"""

from typing import Annotated, Any
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph
from langgraph.pregel._checkpoint import delta_channels_to_snapshot

pytestmark = pytest.mark.anyio


def _simple_reducer(current: list, updates: list) -> list:
    """Flatten updates into current list (each update is itself a list)."""
    result = list(current)
    for u in updates:
        if isinstance(u, list):
            result.extend(u)
        else:
            result.append(u)
    return result


def _build_two_channel_graph(
    checkpointer: InMemorySaver,
    *,
    freq_a: int = 10_000,
    freq_b: int = 10_000,
    n_loops: int = 1,
) -> Any:
    """Graph with two delta channels A and B.

    The node only writes to channel A; B is never written by the node.
    `n_loops` controls how many supersteps the graph runs (via chained nodes).
    """
    ch_a = DeltaChannel(_simple_reducer, list, snapshot_frequency=freq_a)
    ch_b = DeltaChannel(_simple_reducer, list, snapshot_frequency=freq_b)
    State = TypedDict(  # noqa: UP013
        "State",
        {"a": Annotated[list, ch_a], "b": Annotated[list, ch_b]},
    )  # type: ignore[call-overload]

    builder = StateGraph(State)

    for i in range(n_loops):
        name = f"step_{i}"

        def node_fn(state: dict, _i: int = i) -> dict:
            return {"a": [f"a-val-{_i}"]}

        builder.add_node(name, node_fn)
        if i == 0:
            builder.add_edge(START, name)
        else:
            builder.add_edge(f"step_{i - 1}", name)
        if i == n_loops - 1:
            builder.add_edge(name, END)

    return builder.compile(checkpointer=checkpointer)


async def test_forced_snapshot_single_run() -> None:
    """A single invoke with enough supersteps triggers snapshot on the
    unwritten channel B via the supersteps bound."""
    max_ss = 3
    with patch(
        "langgraph.pregel._checkpoint.DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT",
        max_ss,
    ):
        saver = InMemorySaver()
        graph = _build_two_channel_graph(saver, n_loops=4)
        config = {"configurable": {"thread_id": "single-run-ss"}}

        graph.invoke({"a": ["seed-a"], "b": ["seed-b"]}, config)

        head = saver.get_tuple(config)
        assert head is not None
        assert isinstance(head.checkpoint["channel_values"].get("b"), _DeltaSnapshot), (
            "Channel B should have been force-snapshotted via supersteps bound"
        )

        state = graph.get_state(config)
        assert state.values["b"] == ["seed-b"]
        assert "seed-a" in state.values["a"]


async def test_forced_snapshot_accumulates_across_runs() -> None:
    """Supersteps counter for an unwritten channel persists across separate
    invoke() calls. After enough runs, the channel is force-snapshotted."""
    max_ss = 5
    with patch(
        "langgraph.pregel._checkpoint.DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT",
        max_ss,
    ):
        saver = InMemorySaver()
        graph = _build_two_channel_graph(saver, n_loops=1)
        config = {"configurable": {"thread_id": "multi-run-ss"}}

        graph.invoke({"a": ["init-a"], "b": ["init-b"]}, config)

        for i in range(1, 6):
            graph.invoke({"a": [f"run-{i}"]}, config)

            head = saver.get_tuple(config)
            assert head is not None
            counters = head.metadata.get("counters_since_delta_snapshot", {})
            b_counters = counters.get("b", (0, 0))

            if b_counters == (0, 0):
                assert isinstance(
                    head.checkpoint["channel_values"].get("b"), _DeltaSnapshot
                ), f"Run {i}: counter reset but no snapshot blob for B"
                break
        else:
            pytest.fail("Channel B was never force-snapshotted after multiple runs")

        state = graph.get_state(config)
        assert state.values["b"] == ["init-b"]
        assert "init-a" in state.values["a"]


async def test_predicate_fires_on_supersteps_overflow() -> None:
    """Unit test: delta_channels_to_snapshot fires when supersteps >= MAX
    even when updates == 0."""
    ch = DeltaChannel(_simple_reducer, list, snapshot_frequency=10_000)
    ch.key = "x"
    ch_instance = ch.from_checkpoint(None)

    channels = {"x": ch_instance}
    counters: dict[str, tuple[int, int]] = {"x": (0, 5000)}

    result = delta_channels_to_snapshot(channels, counters)
    assert "x" in result

    counters_below: dict[str, tuple[int, int]] = {"x": (0, 4999)}
    result2 = delta_channels_to_snapshot(channels, counters_below)
    assert "x" not in result2


async def test_counter_reset_after_supersteps_snapshot() -> None:
    """After the supersteps bound triggers a snapshot, the counters for
    that channel reset. Verify by using a bound higher than one run's
    supersteps so we can see the counter in an intermediate state."""
    max_ss = 15
    with patch(
        "langgraph.pregel._checkpoint.DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT",
        max_ss,
    ):
        saver = InMemorySaver()
        graph = _build_two_channel_graph(saver, n_loops=4)
        config = {"configurable": {"thread_id": "counter-reset"}}

        graph.invoke({"a": ["seed-a"], "b": ["seed-b"]}, config)

        head = saver.get_tuple(config)
        assert head is not None
        counters = head.metadata.get("counters_since_delta_snapshot", {})
        b_counters = counters.get("b", (0, 0))
        run1_supersteps = b_counters[1]
        assert run1_supersteps > 0, "Should have some supersteps"
        assert b_counters[0] == 1, "B written once (input step)"

        graph.invoke({"a": ["more-a"]}, config)
        head2 = saver.get_tuple(config)
        assert head2 is not None
        counters2 = head2.metadata.get("counters_since_delta_snapshot", {})
        b_counters2 = counters2.get("b", (0, 0))
        run2_supersteps = b_counters2[1]
        assert run2_supersteps > run1_supersteps, "Supersteps should accumulate"
        assert b_counters2[0] == 1, "B written once total (only original input)"

        graph.invoke({"a": ["even-more"]}, config)
        head3 = saver.get_tuple(config)
        assert head3 is not None
        assert isinstance(
            head3.checkpoint["channel_values"].get("b"), _DeltaSnapshot
        ), "B should have snapshotted at supersteps >= max_ss"
        counters3 = head3.metadata.get("counters_since_delta_snapshot", {})
        b_counters3 = counters3.get("b", (0, 0))
        assert b_counters3[1] < max_ss, (
            f"After snapshot, supersteps should have reset, got {b_counters3}"
        )

        state = graph.get_state(config)
        assert state.values["b"] == ["seed-b"]
