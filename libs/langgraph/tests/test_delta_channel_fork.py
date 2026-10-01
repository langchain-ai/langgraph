"""Forking a thread must not replay the abandoned branch into the fork.

Every graph carries a `DeltaChannel` and a plain reducer channel fed the same
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
from langgraph.graph import END, START, StateGraph
from langgraph.types import (
    Command,
    Durability,
    Send,
    StateSnapshot,
    StateUpdate,
    interrupt,
)

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


def _both(marker: str) -> dict:
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
        _both("in-1"), config, durability=durability
    )
    graph = _build(sync_checkpointer, "second")
    graph.invoke(_both("in-2"), config, durability=durability)
    abandoned_head = graph.get_state(config)

    base = next(
        snapshot
        for snapshot in graph.get_state_history(config)
        if "in-2" not in snapshot.values["log"]
    )
    _build(sync_checkpointer, "third").invoke(
        _both("in-3"), _at(config, base), durability=durability
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
        _both("in-1"), config, durability=durability
    )
    graph = _build(async_checkpointer, "second")
    await graph.ainvoke(_both("in-2"), config, durability=durability)
    abandoned_head = await graph.aget_state(config)

    base = await anext(
        snapshot
        async for snapshot in graph.aget_state_history(config)
        if "in-2" not in snapshot.values["log"]
    )
    await _build(async_checkpointer, "third").ainvoke(
        _both("in-3"), _at(config, base), durability=durability
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
    graph.invoke(_both("in-1"), config, durability=durability)

    root = list(graph.get_state_history(config))[-1]
    assert root.values["log"] == []

    _build(sync_checkpointer, "third").invoke(
        _both("in-9"), _at(config, root), durability=durability
    )

    state = graph.get_state(config)
    _assert_fork_is_clean(state, "in-1")
    assert state.values["log"] == ["in-9", "third-out"]


async def test_afork_off_checkpoint_before_first_input(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    graph = _build(async_checkpointer, "first")
    await graph.ainvoke(_both("in-1"), config, durability=durability)

    root = [snapshot async for snapshot in graph.aget_state_history(config)][-1]
    assert root.values["log"] == []

    await _build(async_checkpointer, "third").ainvoke(
        _both("in-9"), _at(config, root), durability=durability
    )

    state = await graph.aget_state(config)
    _assert_fork_is_clean(state, "in-1")
    assert state.values["log"] == ["in-9", "third-out"]


def test_fork_by_update_state(sync_checkpointer: BaseCheckpointSaver) -> None:
    config = _thread("t")
    _build(sync_checkpointer, "first").invoke(_both("in-1"), config)
    graph = _build(sync_checkpointer, "second")
    graph.invoke(_both("in-2"), config)

    base = next(
        snapshot
        for snapshot in graph.get_state_history(config)
        if "in-2" not in snapshot.values["log"]
    )
    forked = graph.update_state(_at(config, base), _both("patched"))

    state = graph.get_state(forked)
    _assert_fork_is_clean(state, "in-2")
    assert state.values["log"] == [*base.values["log"], "patched"]


async def test_afork_by_update_state(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    config = _thread("t")
    await _build(async_checkpointer, "first").ainvoke(_both("in-1"), config)
    graph = _build(async_checkpointer, "second")
    await graph.ainvoke(_both("in-2"), config)

    base = await anext(
        snapshot
        async for snapshot in graph.aget_state_history(config)
        if "in-2" not in snapshot.values["log"]
    )
    forked = await graph.aupdate_state(_at(config, base), _both("patched"))

    state = await graph.aget_state(forked)
    _assert_fork_is_clean(state, "in-2")
    assert state.values["log"] == [*base.values["log"], "patched"]


def test_unaddressed_run_keeps_snapshot_cadence(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    graph = _build(sync_checkpointer, "first")
    graph.invoke(_both("in-1"), config, durability=durability)
    graph.invoke(_both("in-2"), config, durability=durability)

    assert not _snapshotted_checkpoints(sync_checkpointer, config)


def test_fork_before_first_value_when_fork_never_writes_the_channel(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    graph = _build(sync_checkpointer, "first")
    graph.invoke(_both("in-1"), config, durability=durability)

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
    await graph.ainvoke(_both("in-1"), config, durability=durability)

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
    graph.invoke(_both("in-1"), config)

    root = list(graph.get_state_history(config))[-1]
    assert root.values["log"] == []

    forked = graph.bulk_update_state(
        _at(config, root),
        [
            [StateUpdate({"other": ["s1"]}, "n")],
            [StateUpdate(_both("s2"), "n")],
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
    _build(sync_checkpointer, "first").invoke(_both("in-1"), config)
    graph = _build(sync_checkpointer, "second")
    graph.invoke(_both("in-2"), config)

    base = next(
        snapshot
        for snapshot in graph.get_state_history(config)
        if "in-2" not in snapshot.values["log"]
    )
    first = (
        StateUpdate(_both("first-step"), first_as_node)
        if first_as_node == INPUT
        else StateUpdate(None, first_as_node)
    )
    forked = graph.bulk_update_state(
        _at(config, base),
        [[first], [StateUpdate(_both("second-step"), "n")]],
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
    graph.invoke(_both("in-1"), config)

    graph.bulk_update_state(
        config,
        [[StateUpdate(_both(f"u{i}"), "n")] for i in range(4)],
    )

    assert not _snapshotted_checkpoints(sync_checkpointer, config)


def _build_paused_before_b(checkpointer: BaseCheckpointSaver) -> Any:
    builder = StateGraph(_State)
    builder.add_node("a", lambda state: _both("a"))
    builder.add_node("b", lambda state: _both("b"))
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    return builder.compile(checkpointer=checkpointer, interrupt_before=["b"])


def _build_parallel_interrupt(checkpointer: BaseCheckpointSaver) -> Any:
    def ask(state: _State) -> dict:
        interrupt("approve?")
        return {"other": ["q"]}

    builder = StateGraph(_State)
    builder.add_node("p", lambda state: _both("p"))
    builder.add_node("q", ask)
    builder.add_edge(START, "p")
    builder.add_edge(START, "q")
    return builder.compile(checkpointer=checkpointer)


def test_resume_at_interrupt_before_with_the_head_checkpoint_id_runs_the_node(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    graph = _build_paused_before_b(sync_checkpointer)
    graph.invoke(_both("in"), config, durability=durability)

    graph.invoke(None, graph.get_state(config).config, durability=durability)

    state = graph.get_state(config)
    assert state.next == (), f"resume paused again before {state.next}"
    assert state.values["log"] == state.values["plain"] == ["in", "a", "b"]


async def test_aresume_at_interrupt_before_with_the_head_checkpoint_id_runs_the_node(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    graph = _build_paused_before_b(async_checkpointer)
    await graph.ainvoke(_both("in"), config, durability=durability)

    await graph.ainvoke(
        None, (await graph.aget_state(config)).config, durability=durability
    )

    state = await graph.aget_state(config)
    assert state.next == (), f"resume paused again before {state.next}"
    assert state.values["log"] == state.values["plain"] == ["in", "a", "b"]


def test_replay_from_a_paused_checkpoint_runs_the_node_once(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    config = _thread("t")
    graph = _build_paused_before_b(sync_checkpointer)
    graph.invoke(_both("in"), config)
    paused = graph.get_state(config).config
    graph.invoke(None, config)

    graph.invoke(None, paused)

    state = graph.get_state(config)
    assert state.next == (), f"replay paused again before {state.next}"
    assert state.values["log"] == state.values["plain"] == ["in", "a", "b"]


@pytest.mark.parametrize("addressed", [False, True])
def test_new_input_on_an_interrupted_head_does_not_replay_its_pending_writes(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability, addressed: bool
) -> None:
    config = _thread("t")
    graph = _build_parallel_interrupt(sync_checkpointer)
    graph.invoke(_both("in-1"), config, durability=durability)
    head = graph.get_state(config).config

    graph.invoke(_both("in-2"), head if addressed else config, durability=durability)

    state = graph.get_state(config)
    assert state.values["log"] == state.values["plain"] == ["in-1", "in-2", "p"]


def _build_deferred_after_interrupt(checkpointer: BaseCheckpointSaver) -> Any:
    builder = StateGraph(_State)
    builder.add_node("a", lambda state: _both("a"))
    builder.add_node("b", lambda state: _both("b"), defer=True)
    builder.add_node("c", lambda state: {})
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    return builder.compile(checkpointer=checkpointer, interrupt_after=["a"])


@pytest.mark.parametrize(
    "durability",
    [
        "sync",
        "async",
        pytest.param(
            "exit",
            marks=pytest.mark.xfail(
                reason="exit durability stores a resumed run's loaded writes twice",
                strict=True,
            ),
        ),
    ],
)
def test_resume_on_an_interrupted_head_consumes_its_writes_without_a_snapshot(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    graph = _build_parallel_interrupt(sync_checkpointer)
    graph.invoke(_both("in-1"), config, durability=durability)

    graph.invoke(Command(resume="yes"), config, durability=durability)

    state = graph.get_state(config)
    assert state.next == ()
    assert state.values["log"] == state.values["plain"] == ["in-1", "p"]
    assert not _snapshotted_checkpoints(sync_checkpointer, config)


def _build_send_fan_out(checkpointer: BaseCheckpointSaver) -> Any:
    def q(state: _State) -> dict:
        interrupt("continue?")
        return _both("q")

    builder = StateGraph(_State)
    builder.add_node("p", lambda state: _both("p"))
    builder.add_node("q", q)
    builder.add_conditional_edges(
        START, lambda state: [Send("p", state), Send("q", state)], ["p", "q"]
    )
    return builder.compile(checkpointer=checkpointer)


def test_resume_that_replaces_the_pending_sends_drops_the_finished_task(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph = _build_send_fan_out(sync_checkpointer)
    config = _thread("t")
    graph.invoke(_both("in"), config, durability=durability)

    live = graph.invoke(
        Command(resume="yes", goto=[Send("q", _both("unused"))]),
        config,
        durability=durability,
    )

    state = graph.get_state(config)
    assert state.values["log"] == state.values["plain"] == live["log"], (
        f"'p' ran in the fan-out the resume replaced, but the reload reads "
        f"{state.values['log']} against the live {live['log']}"
    )


async def test_aresume_that_replaces_the_pending_sends_drops_the_finished_task(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph = _build_send_fan_out(async_checkpointer)
    config = _thread("t")
    await graph.ainvoke(_both("in"), config, durability=durability)

    live = await graph.ainvoke(
        Command(resume="yes", goto=[Send("q", _both("unused"))]),
        config,
        durability=durability,
    )

    state = await graph.aget_state(config)
    assert state.values["log"] == state.values["plain"] == live["log"], (
        f"'p' ran in the fan-out the resume replaced, but the reload reads "
        f"{state.values['log']} against the live {live['log']}"
    )


def test_replay_interrupted_in_its_first_step_still_seals_the_fork(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph = _build_send_fan_out(sync_checkpointer)
    config = _thread("t")
    graph.invoke(_both("in"), config, durability=durability)

    graph.invoke(None, graph.get_state(config).config, durability=durability)

    state = graph.get_state(config)
    assert state.values["log"] == state.values["plain"] == ["in", "p"], (
        f"the replay reran p, so the fork must not also replay the first p, "
        f"but it reads {state.values['log']}"
    )


async def test_areplay_interrupted_in_its_first_step_still_seals_the_fork(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph = _build_send_fan_out(async_checkpointer)
    config = _thread("t")
    await graph.ainvoke(_both("in"), config, durability=durability)

    await graph.ainvoke(
        None, (await graph.aget_state(config)).config, durability=durability
    )

    state = await graph.aget_state(config)
    assert state.values["log"] == state.values["plain"] == ["in", "p"], (
        f"the replay reran p, so the fork must not also replay the first p, "
        f"but it reads {state.values['log']}"
    )


def test_resume_addressed_at_an_interrupted_head_reruns_its_tasks_once(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    config = _thread("t")
    graph = _build_parallel_interrupt(sync_checkpointer)
    graph.invoke(_both("in-1"), config, durability=durability)

    graph.invoke(
        Command(resume="yes"), graph.get_state(config).config, durability=durability
    )

    state = graph.get_state(config)
    assert state.next == ()
    assert state.values["log"] == state.values["plain"] == ["in-1", "p"]


def test_update_state_with_the_head_checkpoint_id_keeps_a_deferred_node(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _build_deferred_after_interrupt(sync_checkpointer)
    config = _thread("t")
    graph.invoke(_both("in"), config)

    graph.update_state(graph.get_state(config).config, _both("u"), as_node="c")
    graph.invoke(None, config)

    state = graph.get_state(config)
    assert state.next == (), f"deferred node never ran, still pending: {state.next}"
    assert state.values["log"] == state.values["plain"] == ["in", "a", "u", "b"]


async def test_aupdate_state_with_the_head_checkpoint_id_keeps_a_deferred_node(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _build_deferred_after_interrupt(async_checkpointer)
    config = _thread("t")
    await graph.ainvoke(_both("in"), config)

    await graph.aupdate_state(
        (await graph.aget_state(config)).config, _both("u"), as_node="c"
    )
    await graph.ainvoke(None, config)

    state = await graph.aget_state(config)
    assert state.next == (), f"deferred node never ran, still pending: {state.next}"
    assert state.values["log"] == state.values["plain"] == ["in", "a", "u", "b"]


def test_turns_addressed_at_the_head_store_no_snapshot(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    config = _thread("t")
    graph = _build(sync_checkpointer, "turn")
    graph.invoke(_both("in-1"), config)
    for turn in range(2, 5):
        graph.invoke(_both(f"in-{turn}"), graph.get_state(config).config)

    assert not _snapshotted_checkpoints(sync_checkpointer, config)
    assert (
        graph.get_state(config).values["log"] == graph.get_state(config).values["plain"]
    )
