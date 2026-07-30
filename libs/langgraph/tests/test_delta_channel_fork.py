"""Forking a thread must not replay the abandoned branch into the fork.

Regression suite for #8443. Addressing an older checkpoint creates a fork: the
shared base ends up with two children, and it keeps the `checkpoint_writes` of
the branch the fork abandons. Nothing in the stored data records which child
consumed which write, so the `DeltaChannel` ancestor walk used to collect the
abandoned branch's writes as well.

Every graph here carries a `DeltaChannel` and a plain reducer channel fed the
same values. `full` channels store complete `channel_values` and need no
replay, so the plain channel is the oracle: after a fork the two must agree.

Coverage: fork by `invoke` with new input (sync/async, all durabilities) and
fork by `update_state` (sync/async).
"""

from collections.abc import Sequence
from operator import add
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import START, StateGraph
from langgraph.types import Durability

pytestmark = pytest.mark.anyio


def _append(current: list | None, writes: Sequence[Any]) -> list:
    out = list(current or [])
    for write in writes:
        out.extend(write if isinstance(write, list) else [write])
    return out


class State(TypedDict):
    delta: Annotated[list, DeltaChannel(_append, snapshot_frequency=1000)]
    plain: Annotated[list, add]


def _build(checkpointer: BaseCheckpointSaver, tag: str) -> Any:
    """Graph whose single node appends `f"{tag}-out"` to both channels."""

    def node(state: State) -> dict:
        return {"delta": [f"{tag}-out"], "plain": [f"{tag}-out"]}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.set_finish_point("node")
    return builder.compile(checkpointer=checkpointer)


def _both(value: str) -> dict:
    return {"delta": [value], "plain": [value]}


def _fork_config(config: dict, snapshot: Any) -> dict:
    return {
        "configurable": {
            **config["configurable"],
            "checkpoint_ns": "",
            "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
        }
    }


def test_fork_with_new_input_ignores_abandoned_writes(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = {"configurable": {"thread_id": "1"}}
    _build(sync_checkpointer, "first").invoke(
        _both("in-1"), config, durability=durability
    )
    graph = _build(sync_checkpointer, "second")
    graph.invoke(_both("in-2"), config, durability=durability)

    # deepest checkpoint of the second run that predates its input: the base a
    # fork branches off, and the only one still holding "in-2" as a write.
    base = next(
        snapshot
        for snapshot in graph.get_state_history(config)
        if "in-2" not in snapshot.values["plain"]
    )
    fork_config = _fork_config(config, base)
    _build(sync_checkpointer, "third").invoke(
        _both("in-3"), fork_config, durability=durability
    )

    values = graph.get_state(config).values
    assert values["plain"] == ["in-1", "first-out", "in-3", "third-out"]
    assert values["delta"] == values["plain"]


async def test_afork_with_new_input_ignores_abandoned_writes(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = {"configurable": {"thread_id": "1"}}
    await _build(async_checkpointer, "first").ainvoke(
        _both("in-1"), config, durability=durability
    )
    graph = _build(async_checkpointer, "second")
    await graph.ainvoke(_both("in-2"), config, durability=durability)

    base = None
    async for snapshot in graph.aget_state_history(config):
        if "in-2" not in snapshot.values["plain"]:
            base = snapshot
            break
    assert base is not None
    await _build(async_checkpointer, "third").ainvoke(
        _both("in-3"), _fork_config(config, base), durability=durability
    )

    values = (await graph.aget_state(config)).values
    assert values["plain"] == ["in-1", "first-out", "in-3", "third-out"]
    assert values["delta"] == values["plain"]


def test_fork_from_thread_root_ignores_abandoned_writes(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    """Fork off the checkpoint that predates the thread's first input.

    The delta channel has no version there, so the fork's own first checkpoint
    cannot carry a snapshot blob — the walk has to be terminated by the first
    checkpoint that can.
    """
    config = {"configurable": {"thread_id": "1"}}
    graph = _build(sync_checkpointer, "first")
    graph.invoke(_both("in-1"), config, durability=durability)

    root = list(graph.get_state_history(config))[-1]
    assert root.values["plain"] == []
    _build(sync_checkpointer, "second").invoke(
        _both("in-2"), _fork_config(config, root), durability=durability
    )

    values = graph.get_state(config).values
    assert values["plain"] == ["in-2", "second-out"]
    assert values["delta"] == values["plain"]


def test_update_state_fork_ignores_abandoned_writes(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    config = {"configurable": {"thread_id": "1"}}
    _build(sync_checkpointer, "first").invoke(_both("in-1"), config)
    graph = _build(sync_checkpointer, "second")
    graph.invoke(_both("in-2"), config)

    base = next(
        snapshot
        for snapshot in graph.get_state_history(config)
        if "in-2" not in snapshot.values["plain"]
    )
    forked = graph.update_state(_fork_config(config, base), _both("patched"))

    values = graph.get_state(forked).values
    assert values["plain"] == ["in-1", "first-out", "patched"]
    assert values["delta"] == values["plain"]


async def test_aupdate_state_fork_ignores_abandoned_writes(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    config = {"configurable": {"thread_id": "1"}}
    await _build(async_checkpointer, "first").ainvoke(_both("in-1"), config)
    graph = _build(async_checkpointer, "second")
    await graph.ainvoke(_both("in-2"), config)

    base = None
    async for snapshot in graph.aget_state_history(config):
        if "in-2" not in snapshot.values["plain"]:
            base = snapshot
            break
    assert base is not None
    forked = await graph.aupdate_state(_fork_config(config, base), _both("patched"))

    values = (await graph.aget_state(forked)).values
    assert values["plain"] == ["in-1", "first-out", "patched"]
    assert values["delta"] == values["plain"]


def test_no_fork_leaves_snapshot_cadence_untouched(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """A run without an addressed checkpoint must not force extra snapshots."""
    config = {"configurable": {"thread_id": "1"}}
    graph = _build(sync_checkpointer, "first")
    graph.invoke(_both("in-1"), config)
    graph.invoke(_both("in-2"), config)

    for snapshot in graph.get_state_history(config):
        stored = sync_checkpointer.get_tuple(snapshot.config)
        assert stored is not None
        assert "delta" not in stored.checkpoint["channel_values"]
