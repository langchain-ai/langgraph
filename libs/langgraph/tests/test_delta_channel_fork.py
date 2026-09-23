"""Forking a thread must not replay the abandoned branch into the fork.

Every graph carries a ``DeltaChannel`` and a plain reducer channel fed the same
values; the plain channel needs no replay, so it is the oracle.
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
    out = list(current or [])
    for write in writes:
        out.extend(write if isinstance(write, list) else [write])
    return out


class _State(TypedDict):
    log: Annotated[list, DeltaChannel(_append, snapshot_frequency=1000)]
    plain: Annotated[list, add]
    other: Annotated[list, add]


def _build(checkpointer: BaseCheckpointSaver, tag: str) -> Any:
    def node(state: _State) -> dict:
        return {"log": [f"{tag}-out"], "plain": [f"{tag}-out"]}

    builder = StateGraph(_State)
    builder.add_node("n", node)
    builder.set_entry_point("n")
    builder.set_finish_point("n")
    return builder.compile(checkpointer=checkpointer)


def _build_without_delta_writes(checkpointer: BaseCheckpointSaver, tag: str) -> Any:
    def node(state: _State) -> dict:
        return {"other": [f"{tag}-other"]}

    builder = StateGraph(_State)
    builder.add_node("n", node)
    builder.set_entry_point("n")
    builder.set_finish_point("n")
    return builder.compile(checkpointer=checkpointer)


def _thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _at(config: RunnableConfig, snapshot: StateSnapshot) -> RunnableConfig:
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
    return [
        tuple_.config["configurable"]["checkpoint_id"]
        for tuple_ in checkpointer.list(config)
        if isinstance(tuple_.checkpoint["channel_values"].get("log"), _DeltaSnapshot)
    ]


def _assert_fork_is_clean(state: StateSnapshot, abandoned: str) -> None:
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
    abandoned_head = graph.get_state(config)

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

    abandoned = graph.get_state(abandoned_head.config).values
    assert abandoned["log"] == abandoned["plain"] == abandoned_head.values["log"]


async def test_afork_by_invoke(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    await _build(async_checkpointer, "first").ainvoke(
        _input("in-1"), config, durability=durability
    )
    graph = _build(async_checkpointer, "second")
    await graph.ainvoke(_input("in-2"), config, durability=durability)
    abandoned_head = await graph.aget_state(config)

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

    abandoned = (await graph.aget_state(abandoned_head.config)).values
    assert abandoned["log"] == abandoned["plain"] == abandoned_head.values["log"]


def test_fork_off_checkpoint_before_first_input(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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
    config = _thread("t")
    graph = _build(sync_checkpointer, "first")
    graph.invoke(_input("in-1"), config, durability=durability)
    graph.invoke(_input("in-2"), config, durability=durability)

    assert not _snapshotted_checkpoints(sync_checkpointer, config)


def test_fork_before_first_value_when_fork_never_writes_the_channel(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    graph = _build(sync_checkpointer, "first")
    graph.invoke(_input("in-1"), config, durability=durability)

    root = list(graph.get_state_history(config))[-1]
    assert root.values["log"] == []

    _build_without_delta_writes(sync_checkpointer, "third").invoke(
        {"other": ["in-9"]}, _at(config, root), durability=durability
    )

    state = graph.get_state(config)
    _assert_fork_is_clean(state, "in-1")
    assert state.values["log"] == []


async def test_afork_before_first_value_when_fork_never_writes_the_channel(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    graph = _build(async_checkpointer, "first")
    await graph.ainvoke(_input("in-1"), config, durability=durability)

    root = [snapshot async for snapshot in graph.aget_state_history(config)][-1]
    assert root.values["log"] == []

    await _build_without_delta_writes(async_checkpointer, "third").ainvoke(
        {"other": ["in-9"]}, _at(config, root), durability=durability
    )

    state = await graph.aget_state(config)
    _assert_fork_is_clean(state, "in-1")
    assert state.values["log"] == []


def test_fork_before_first_value_by_bulk_update(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    config = _thread("t")
    graph = _build(sync_checkpointer, "first")
    graph.invoke(_input("in-1"), config)

    root = list(graph.get_state_history(config))[-1]
    assert root.values["log"] == []

    forked = graph.bulk_update_state(
        _at(config, root),
        [
            [StateUpdate({"other": ["s1"]}, "n")],
            [StateUpdate(_input("s2"), "n")],
        ],
    )

    state = graph.get_state(forked)
    _assert_fork_is_clean(state, "in-1")
    assert state.values["log"] == ["s2"]


@pytest.mark.parametrize("first_as_node", [INPUT, END, "__copy__"])
def test_fork_by_bulk_update_whose_first_superstep_skips_the_plan(
    sync_checkpointer: BaseCheckpointSaver, first_as_node: str
) -> None:
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
    assert state.values["log"] == state.values["plain"], (
        f"delta channel diverged from the plain channel: "
        f"{state.values['log']} != {state.values['plain']}"
    )


def test_unaddressed_bulk_update_keeps_snapshot_cadence(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    config = _thread("t")
    graph = _build(sync_checkpointer, "first")
    graph.invoke(_input("in-1"), config)

    graph.bulk_update_state(
        config,
        [[StateUpdate(_input(f"u{i}"), "n")] for i in range(4)],
    )

    assert not _snapshotted_checkpoints(sync_checkpointer, config)
