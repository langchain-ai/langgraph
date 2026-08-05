"""Tests that `DeltaChannel` replay preserves live parallel-write order.

Regression suite for #8382.

`apply_writes` sorts a super-step's tasks by `task_path_str(task.path[:3])`
before handing their values to `channel.update`, so the order a reducer sees is
deterministic and independent of which parallel task finishes first. Replay has
to recover that same order. Ordering a checkpoint's writes by `(task_id, idx)`
does not: `task_id` is a hash of the path, so for two or more tasks writing one
`DeltaChannel` in a single super-step it yields an effectively arbitrary
permutation. Reducers are required to be batching-invariant, not
order-invariant, so the permutation changes the reconstructed value.

Every test runs against the full `async_checkpointer` matrix — memory, sqlite,
and postgres in three pool modes — because each saver reconstructs delta
channels through its own `aget_delta_channel_history` override rather than a
shared code path, and the three stored `task_path` differently before this fix.
"""

from itertools import pairwise
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph

pytestmark = pytest.mark.anyio

# Node names double as the values written. They are listed in sorted order,
# which is also the order live execution applies them: each node is a PULL task
# whose path is `("__pregel_pull", name)`, so sorting paths sorts by name.
FAN_OUT_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]


def _append_reducer(current: list, updates: list) -> list:
    """Order-sensitive list accumulation, as in the `DeltaChannel` docstring."""
    result = list(current)
    for update in updates:
        if isinstance(update, list):
            result.extend(update)
        else:
            result.append(update)
    return result


def _build_graph(checkpointer: BaseCheckpointSaver, *, sequential: bool = False) -> Any:
    """Compile a `DeltaChannel`-backed `items` graph over `FAN_OUT_NAMES`.

    By default every node is wired off `START`, so they all write `items` in one
    super-step — the shape #8382 is about. `sequential=True` chains them
    instead, giving one writer per super-step as a control.

    `snapshot_frequency` is far above the number of updates these tests make, so
    no snapshot is ever written and the value has to come from replaying
    ancestor writes — the path under test.
    """

    class State(TypedDict):
        items: Annotated[
            list, DeltaChannel(_append_reducer, list, snapshot_frequency=10_000)
        ]

    def make_node(label: str) -> Any:
        def node(state: State) -> dict:
            return {"items": [label]}

        return node

    builder = StateGraph(State)
    for name in FAN_OUT_NAMES:
        builder.add_node(name, make_node(name))
    if sequential:
        for source, target in pairwise([START, *FAN_OUT_NAMES, END]):
            builder.add_edge(source, target)
    else:
        for name in FAN_OUT_NAMES:
            builder.add_edge(START, name)
            builder.add_edge(name, END)
    return builder.compile(checkpointer=checkpointer)


async def test_get_state_matches_live_invoke_order(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """A cold read reports the same order `invoke` returned."""
    graph = _build_graph(async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    live = (await graph.ainvoke({"items": []}, config))["items"]
    replayed = (await graph.aget_state(config)).values["items"]

    assert live == FAN_OUT_NAMES
    assert replayed == live


async def test_continuing_thread_preserves_committed_prefix(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """A second run appends without reordering the first run's items.

    The more serious half of #8382: the reordered replay becomes the base that
    later writes build on, so the corruption is persisted rather than confined
    to a read.
    """
    graph = _build_graph(async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    first = (await graph.ainvoke({"items": []}, config))["items"]
    second = (await graph.ainvoke({"items": []}, config))["items"]

    assert second == first + first
    assert (await graph.aget_state(config)).values["items"] == second


async def test_order_stable_across_many_supersteps(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Order holds over a chain spanning several supersteps with no snapshot."""
    runs = 5
    graph = _build_graph(async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    for _ in range(runs):
        live = (await graph.ainvoke({"items": []}, config))["items"]

    assert live == FAN_OUT_NAMES * runs
    assert (await graph.aget_state(config)).values["items"] == live


async def test_state_history_reports_live_order_at_every_step(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Every checkpoint in the history replays in live order.

    Guards the walk at intermediate depths, not just from the head. Entries are
    checked against the order live execution produced rather than against the
    replayed head — comparing replayed values only to each other passes even
    when every one of them is permuted the same wrong way.
    """
    runs = 3
    graph = _build_graph(async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    for _ in range(runs):
        await graph.ainvoke({"items": []}, config)
    live_expected = FAN_OUT_NAMES * runs

    seen = [
        snapshot.values["items"]
        async for snapshot in graph.aget_state_history(config)
        if "items" in snapshot.values
    ]

    assert seen, "expected at least one snapshot carrying `items`"
    # The deepest entry is the head, so the matrix below covers the full value
    # as well as every partial prefix.
    assert max(len(values) for values in seen) == len(live_expected)
    for values in seen:
        assert values == live_expected[: len(values)], (
            f"history entry {values} is not the live order "
            f"{live_expected[: len(values)]}"
        )


async def test_sequential_graph_unaffected(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """One writer per super-step replays correctly with or without the fix.

    Control: it localises #8382 to multiple tasks writing one channel in a
    single super-step, rather than to delta replay in general. This is the one
    test here that passes on main.
    """
    graph = _build_graph(async_checkpointer, sequential=True)
    config = {"configurable": {"thread_id": "1"}}

    live = (await graph.ainvoke({"items": []}, config))["items"]
    replayed = (await graph.aget_state(config)).values["items"]

    assert live == FAN_OUT_NAMES
    assert replayed == live
