"""Tests for `update_state` / `aupdate_state` against `DeltaChannel`.

Regression suite for deepagents#3774 and Postgres read-path compatibility.

Fresh-thread ``update_state`` force-snapshots DeltaChannels (1.2.8). Non-fresh
``update_state`` persists ``checkpoint_writes`` on the parent, advances
``counters_since_delta_snapshot`` on the new head, and snapshots when a
channel reaches ``snapshot_frequency`` (mirroring normal run cadence).

Coverage:

* fresh-thread regression: single ``update_state`` writes a message and reads back
* non-fresh thread: ``update_state`` after ``invoke``, after another ``update_state``,
  and ``bulk_update_state`` with multiple per-superstep updates
* update-by-id end-to-end via ``update_state`` (DeltaChannel reducer semantics)
* fresh-thread head is snapshotted; non-fresh heads carry delta replay counters
"""

from typing import Annotated, Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import START, StateGraph
from langgraph.graph.message import _messages_delta_reducer

pytestmark = pytest.mark.anyio


def _build_graph(
    checkpointer: InMemorySaver,
    *,
    two_nodes: bool = False,
    snapshot_frequency: int = 1000,
) -> Any:
    """Compile a minimal DeltaChannel-backed `messages` graph.

    `two_nodes=True` adds a second writer node so `bulk_update_state` can route
    distinct updates to different `as_node` values within a single superstep.
    """
    channel = DeltaChannel(
        _messages_delta_reducer, snapshot_frequency=snapshot_frequency
    )
    State = TypedDict("State", {"messages": Annotated[list, channel]})  # type: ignore[call-overload]  # noqa: UP013

    def model(state: dict) -> dict:
        return {}

    def assistant(state: dict) -> dict:
        return {}

    builder = StateGraph(State)
    builder.add_node("model", model)
    builder.add_edge(START, "model")
    if two_nodes:
        builder.add_node("assistant", assistant)
        builder.add_edge("model", "assistant")
        builder.set_finish_point("assistant")
    else:
        builder.set_finish_point("model")
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Fresh-thread regression (deepagents#3774)
# ---------------------------------------------------------------------------


def test_update_state_fresh_thread_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "fresh-sync"}}
    message = HumanMessage(content="hello", id="m1")

    graph.update_state(config, {"messages": [message]}, as_node="model")

    state = graph.get_state(config)
    assert [m.content for m in state.values["messages"]] == ["hello"]


async def test_aupdate_state_fresh_thread_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "fresh-async"}}
    message = HumanMessage(content="hello", id="m1")

    await graph.aupdate_state(config, {"messages": [message]}, as_node="model")

    state = await graph.aget_state(config)
    assert [m.content for m in state.values["messages"]] == ["hello"]


def test_fresh_update_state_head_snapshots_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "fresh-head-snapshot"}}

    graph.update_state(
        config,
        {"messages": [HumanMessage(content="hello", id="m1")]},
        as_node="model",
    )

    head = saver.get_tuple(config)
    assert head is not None
    assert isinstance(head.checkpoint["channel_values"].get("messages"), _DeltaSnapshot)
    assert head.metadata is not None
    assert "counters_since_delta_snapshot" not in head.metadata


# ---------------------------------------------------------------------------
# Non-fresh thread: update_state after invoke
# ---------------------------------------------------------------------------


def test_update_state_after_invoke_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "after-invoke-sync"}}

    graph.invoke({"messages": [HumanMessage(content="seed", id="m1")]}, config)
    graph.update_state(
        config,
        {"messages": [HumanMessage(content="appended", id="m2")]},
        as_node="model",
    )

    state = graph.get_state(config)
    assert [m.content for m in state.values["messages"]] == ["seed", "appended"]
    assert [m.id for m in state.values["messages"]] == ["m1", "m2"]

    head = saver.get_tuple(config)
    assert head is not None
    assert "messages" not in head.checkpoint["channel_values"]
    assert head.metadata is not None
    assert head.metadata["counters_since_delta_snapshot"]["messages"] == [2, 4]


async def test_aupdate_state_after_invoke_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "after-invoke-async"}}

    await graph.ainvoke({"messages": [HumanMessage(content="seed", id="m1")]}, config)
    await graph.aupdate_state(
        config,
        {"messages": [HumanMessage(content="appended", id="m2")]},
        as_node="model",
    )

    state = await graph.aget_state(config)
    assert [m.content for m in state.values["messages"]] == ["seed", "appended"]


# ---------------------------------------------------------------------------
# Non-fresh thread: consecutive update_state calls
# ---------------------------------------------------------------------------


def test_consecutive_update_states_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "consecutive-sync"}}

    graph.update_state(
        config,
        {"messages": [HumanMessage(content="first", id="m1")]},
        as_node="model",
    )
    graph.update_state(
        config,
        {"messages": [HumanMessage(content="second", id="m2")]},
        as_node="model",
    )

    state = graph.get_state(config)
    assert [m.content for m in state.values["messages"]] == ["first", "second"]
    assert [m.id for m in state.values["messages"]] == ["m1", "m2"]

    head = saver.get_tuple(config)
    assert head is not None
    assert "messages" not in head.checkpoint["channel_values"]
    assert head.metadata is not None
    assert head.metadata["counters_since_delta_snapshot"]["messages"] == [1, 1]


def test_update_state_snapshots_at_frequency() -> None:
    """Non-fresh update_state snapshots when counters reach snapshot_frequency."""
    saver = InMemorySaver()
    graph = _build_graph(saver, snapshot_frequency=1)
    config = {"configurable": {"thread_id": "snapshot-at-freq"}}

    graph.update_state(
        config,
        {"messages": [HumanMessage(content="first", id="m1")]},
        as_node="model",
    )
    graph.update_state(
        config,
        {"messages": [HumanMessage(content="second", id="m2")]},
        as_node="model",
    )

    state = graph.get_state(config)
    assert [m.content for m in state.values["messages"]] == ["first", "second"]

    head = saver.get_tuple(config)
    assert head is not None
    assert isinstance(head.checkpoint["channel_values"].get("messages"), _DeltaSnapshot)
    assert head.metadata is not None
    assert "counters_since_delta_snapshot" not in head.metadata


async def test_aconsecutive_update_states_delta_channel() -> None:
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "consecutive-async"}}

    await graph.aupdate_state(
        config,
        {"messages": [HumanMessage(content="first", id="m1")]},
        as_node="model",
    )
    await graph.aupdate_state(
        config,
        {"messages": [HumanMessage(content="second", id="m2")]},
        as_node="model",
    )

    state = await graph.aget_state(config)
    assert [m.content for m in state.values["messages"]] == ["first", "second"]


# ---------------------------------------------------------------------------
# Update-by-id semantics through the update_state path
# ---------------------------------------------------------------------------


def test_update_state_replaces_message_by_id_delta_channel() -> None:
    """`_messages_delta_reducer` dedups by `id` — re-issuing a write with the
    same id replaces the existing entry rather than appending. Verify this
    works through the `update_state` path (not just `invoke`)."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "update-by-id"}}

    graph.invoke({"messages": [HumanMessage(content="original", id="h1")]}, config)
    graph.update_state(
        config,
        {"messages": [HumanMessage(content="updated", id="h1")]},
        as_node="model",
    )

    state = graph.get_state(config)
    msgs = state.values["messages"]
    assert len(msgs) == 1
    assert msgs[0].id == "h1"
    assert msgs[0].content == "updated"


# ---------------------------------------------------------------------------
# bulk_update_state with multiple updates per superstep
# ---------------------------------------------------------------------------


def test_bulk_update_state_multi_task_per_superstep_delta_channel() -> None:
    """`bulk_update_state` with N updates in one superstep produces N tasks
    that each call `put_writes`. Guards the regression where moving
    `put_writes` outside the per-task loop would persist only the last
    task's writes.

    Explicit `task_id`s are required to disambiguate writes belonging to
    different `StateUpdate`s targeting the same node — otherwise both share
    the deterministic interrupt-derived id and collide in the saver.
    """
    from langgraph.types import StateUpdate

    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "bulk-multi-task"}}

    graph.bulk_update_state(
        config,
        [
            [
                StateUpdate(
                    values={"messages": [HumanMessage(content="first", id="m1")]},
                    as_node="model",
                    task_id="task-1",
                ),
                StateUpdate(
                    values={"messages": [HumanMessage(content="second", id="m2")]},
                    as_node="model",
                    task_id="task-2",
                ),
            ]
        ],
    )

    state = graph.get_state(config)
    contents = [m.content for m in state.values["messages"]]
    ids = [m.id for m in state.values["messages"]]
    assert sorted(contents) == ["first", "second"], (
        f"both updates' writes must persist; got {contents}"
    )
    assert sorted(ids) == ["m1", "m2"]


# ---------------------------------------------------------------------------
# Public-API observation of fresh-thread checkpoint shape
# ---------------------------------------------------------------------------


def test_state_history_chain_after_fresh_update_state_delta_channel() -> None:
    """A fresh-thread `update_state` should produce a single self-contained
    checkpoint visible via `get_state_history`: step=0, `source='update'`,
    no parent, with the DeltaChannel value snapshotted inline."""
    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "history-chain"}}

    graph.update_state(
        config,
        {"messages": [HumanMessage(content="hello", id="m1")]},
        as_node="model",
    )

    history = list(graph.get_state_history(config))
    assert len(history) == 1

    (update_snapshot,) = history
    assert update_snapshot.metadata is not None
    assert update_snapshot.metadata["source"] == "update"
    assert update_snapshot.metadata["step"] == 0
    assert update_snapshot.parent_config is None
    assert [m.content for m in update_snapshot.values["messages"]] == ["hello"]
