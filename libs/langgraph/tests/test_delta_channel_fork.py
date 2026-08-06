"""Forking a thread must not replay the abandoned branch into the fork.

Regression suite for #8443. Addressing an older checkpoint creates a fork: the
shared base ends up with two children, and it keeps the ``checkpoint_writes``
of the branch the fork abandons. Nothing in the stored data records which child
consumed which write, so the ``DeltaChannel`` ancestor walk used to collect the
abandoned branch's writes as well.

Every graph here carries a ``DeltaChannel`` and a plain reducer channel fed the
same values. ``full`` channels store complete ``channel_values`` and need no
replay, so the plain channel is the oracle: after a fork the two must agree.

Coverage: fork by ``invoke`` with new input (sync/async, all durabilities),
fork off the checkpoint that predates the thread's first input (sync/async),
fork by ``update_state`` / ``aupdate_state``, and guards that neither an
unaddressed run nor an unaddressed multi-superstep ``bulk_update_state``
departs from the normal ``snapshot_frequency`` cadence.
"""

from collections.abc import Sequence
from operator import add
from typing import Annotated, Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from typing_extensions import TypedDict

from langgraph._internal._constants import INPUT
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, StateGraph
from langgraph.types import Durability, StateSnapshot, StateUpdate

pytestmark = pytest.mark.anyio


def _append(current: list | None, writes: Sequence[Any]) -> list:
    """DeltaChannel reducer: extend the list with every batched write."""
    out = list(current or [])
    for write in writes:
        out.extend(write if isinstance(write, list) else [write])
    return out


class _State(TypedDict):
    log: Annotated[list, DeltaChannel(_append, snapshot_frequency=1000)]
    plain: Annotated[list, add]


def _build(checkpointer: BaseCheckpointSaver, tag: str) -> Any:
    """Compile a one-node graph whose node appends ``{tag}-out`` to both channels.

    ``snapshot_frequency=1000`` keeps the cadence from masking the bug: without
    a forced snapshot the fork's ancestor walk always runs past the fork base.
    """

    def node(state: _State) -> dict:
        return {"log": [f"{tag}-out"], "plain": [f"{tag}-out"]}

    builder = StateGraph(_State)
    builder.add_node("n", node)
    builder.set_entry_point("n")
    builder.set_finish_point("n")
    return builder.compile(checkpointer=checkpointer)


def _thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _at(config: RunnableConfig, snapshot: StateSnapshot) -> RunnableConfig:
    """Config addressing one specific checkpoint of ``config``'s thread."""
    return {
        "configurable": {
            **config["configurable"],
            "checkpoint_ns": "",
            "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
        }
    }


def _input(marker: str) -> dict:
    return {"log": [marker], "plain": [marker]}


def _snapshotted_checkpoints(
    checkpointer: BaseCheckpointSaver, config: RunnableConfig
) -> list[str]:
    """Ids of this thread's checkpoints carrying a ``log`` snapshot blob."""
    return [
        tuple_.config["configurable"]["checkpoint_id"]
        for tuple_ in checkpointer.list(config)
        if isinstance(tuple_.checkpoint["channel_values"].get("log"), _DeltaSnapshot)
    ]


def _assert_fork_is_clean(state: StateSnapshot, abandoned: str) -> None:
    """The delta channel must match the plain channel and drop ``abandoned``."""
    assert state.values["log"] == state.values["plain"], (
        f"delta channel diverged from the plain channel: "
        f"{state.values['log']} != {state.values['plain']}"
    )
    assert abandoned not in state.values["log"], (
        f"{abandoned!r} belongs to the branch the fork replaced, "
        f"but was replayed into {state.values['log']}"
    )


def test_fork_by_invoke(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    _build(sync_checkpointer, "first").invoke(
        _input("in-1"), config, durability=durability
    )
    graph = _build(sync_checkpointer, "second")
    graph.invoke(_input("in-2"), config, durability=durability)

    # The last checkpoint that predates "in-2" entering state: forking here
    # abandons the "in-2" branch, whose writes still hang off this checkpoint.
    base = next(
        snapshot
        for snapshot in graph.get_state_history(config)
        if "in-2" not in snapshot.values["log"]
    )
    _build(sync_checkpointer, "third").invoke(
        _input("in-3"), _at(config, base), durability=durability
    )

    state = graph.get_state(config)
    _assert_fork_is_clean(state, "in-2")
    assert state.values["log"] == [*base.values["log"], "in-3", "third-out"]


async def test_afork_by_invoke(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    await _build(async_checkpointer, "first").ainvoke(
        _input("in-1"), config, durability=durability
    )
    graph = _build(async_checkpointer, "second")
    await graph.ainvoke(_input("in-2"), config, durability=durability)

    base = await anext(
        snapshot
        async for snapshot in graph.aget_state_history(config)
        if "in-2" not in snapshot.values["log"]
    )
    await _build(async_checkpointer, "third").ainvoke(
        _input("in-3"), _at(config, base), durability=durability
    )

    state = await graph.aget_state(config)
    _assert_fork_is_clean(state, "in-2")
    assert state.values["log"] == [*base.values["log"], "in-3", "third-out"]


def test_fork_off_checkpoint_before_first_input(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    """Fork off the root checkpoint, which predates any value for ``log``.

    ``create_checkpoint`` drops a requested snapshot for a channel absent from
    ``channel_versions``, so the fork's own first checkpoint cannot carry the
    blob. The request has to stay queued until a superstep gives the channel a
    value, otherwise the root's ``in-1`` write still leaks into the fork.
    """
    config = _thread("t")
    graph = _build(sync_checkpointer, "first")
    graph.invoke(_input("in-1"), config, durability=durability)

    root = list(graph.get_state_history(config))[-1]
    assert root.values["log"] == []

    _build(sync_checkpointer, "third").invoke(
        _input("in-9"), _at(config, root), durability=durability
    )

    state = graph.get_state(config)
    _assert_fork_is_clean(state, "in-1")
    assert state.values["log"] == ["in-9", "third-out"]


async def test_afork_off_checkpoint_before_first_input(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    """Async twin of ``test_fork_off_checkpoint_before_first_input``."""
    config = _thread("t")
    graph = _build(async_checkpointer, "first")
    await graph.ainvoke(_input("in-1"), config, durability=durability)

    root = [snapshot async for snapshot in graph.aget_state_history(config)][-1]
    assert root.values["log"] == []

    await _build(async_checkpointer, "third").ainvoke(
        _input("in-9"), _at(config, root), durability=durability
    )

    state = await graph.aget_state(config)
    _assert_fork_is_clean(state, "in-1")
    assert state.values["log"] == ["in-9", "third-out"]


def test_fork_by_update_state(sync_checkpointer: BaseCheckpointSaver) -> None:
    config = _thread("t")
    _build(sync_checkpointer, "first").invoke(_input("in-1"), config)
    graph = _build(sync_checkpointer, "second")
    graph.invoke(_input("in-2"), config)

    base = next(
        snapshot
        for snapshot in graph.get_state_history(config)
        if "in-2" not in snapshot.values["log"]
    )
    forked = graph.update_state(_at(config, base), _input("patched"))

    state = graph.get_state(forked)
    _assert_fork_is_clean(state, "in-2")
    assert state.values["log"] == [*base.values["log"], "patched"]


async def test_afork_by_update_state(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    config = _thread("t")
    await _build(async_checkpointer, "first").ainvoke(_input("in-1"), config)
    graph = _build(async_checkpointer, "second")
    await graph.ainvoke(_input("in-2"), config)

    base = await anext(
        snapshot
        async for snapshot in graph.aget_state_history(config)
        if "in-2" not in snapshot.values["log"]
    )
    forked = await graph.aupdate_state(_at(config, base), _input("patched"))

    state = await graph.aget_state(forked)
    _assert_fork_is_clean(state, "in-2")
    assert state.values["log"] == [*base.values["log"], "patched"]


def test_unaddressed_run_keeps_snapshot_cadence(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    """A run with no explicitly addressed checkpoint writes no snapshot blob.

    Guards the cost of the fix: the forced snapshot is one per addressed run,
    not a change to the normal ``snapshot_frequency`` cadence.
    """
    config = _thread("t")
    graph = _build(sync_checkpointer, "first")
    graph.invoke(_input("in-1"), config, durability=durability)
    graph.invoke(_input("in-2"), config, durability=durability)

    assert not _snapshotted_checkpoints(sync_checkpointer, config)


@pytest.mark.parametrize("first_as_node", [INPUT, END, "__copy__"])
def test_fork_by_bulk_update_whose_first_superstep_skips_the_plan(
    sync_checkpointer: BaseCheckpointSaver, first_as_node: str
) -> None:
    """The fork must be sealed by whichever checkpoint the fork writes first.

    ``as_node`` of INPUT, END or ``__copy__`` writes a checkpoint and returns
    before ``create_checkpoint_plan_for_update_state_api`` runs. If that
    checkpoint carries no snapshot the branch is still unsealed, so the next
    superstep reconstructs through the shared base, picks up the abandoned
    writes, and bakes them into whatever it snapshots. Sealing later is too
    late: by then the in-memory value is already wrong.
    """
    config = _thread("t")
    _build(sync_checkpointer, "first").invoke(_input("in-1"), config)
    graph = _build(sync_checkpointer, "second")
    graph.invoke(_input("in-2"), config)

    base = next(
        snapshot
        for snapshot in graph.get_state_history(config)
        if "in-2" not in snapshot.values["log"]
    )
    first = (
        StateUpdate(_input("first-step"), first_as_node)
        if first_as_node == INPUT
        else StateUpdate(None, first_as_node)
    )
    forked = graph.bulk_update_state(
        _at(config, base),
        [[first], [StateUpdate(_input("second-step"), "n")]],
    )

    state = graph.get_state(forked)
    # END legitimately absorbs the base's already-run task writes, so "in-2"
    # belongs there; the plain channel is the oracle for which is which.
    assert state.values["log"] == state.values["plain"], (
        f"delta channel diverged from the plain channel: "
        f"{state.values['log']} != {state.values['plain']}"
    )


def test_unaddressed_bulk_update_keeps_snapshot_cadence(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """A multi-superstep ``bulk_update_state`` forks at most once, at the head.

    ``perform_superstep`` returns the config of the checkpoint it just wrote
    and the driver feeds that back in, so every superstep after the first
    receives a config naming a checkpoint even when the caller addressed none.
    Deriving the fork flag from that config snapshots the whole growing value
    once per superstep.
    """
    config = _thread("t")
    graph = _build(sync_checkpointer, "first")
    graph.invoke(_input("in-1"), config)

    graph.bulk_update_state(
        config,
        [[StateUpdate(_input(f"u{i}"), "n")] for i in range(4)],
    )

    assert not _snapshotted_checkpoints(sync_checkpointer, config)
