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
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from typing_extensions import TypedDict

from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.delta import DeltaChannel
from langgraph.channels.last_value import LastValue
from langgraph.graph import START, StateGraph
from langgraph.graph.message import _messages_delta_reducer
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.types import StateSnapshot, StateUpdate

pytestmark = pytest.mark.anyio


def _build_graph(
    checkpointer: BaseCheckpointSaver,
    *,
    two_nodes: bool = False,
    snapshot_frequency: int = 1000,
    interrupt_before: list[str] | None = None,
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
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


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


def test_fresh_update_state_stores_nothing_for_a_delta_channel_it_did_not_write() -> (
    None
):
    saver = InMemorySaver()
    State = TypedDict(  # type: ignore[call-overload]  # noqa: UP013
        "State",
        {
            "messages": Annotated[list, DeltaChannel(_messages_delta_reducer)],
            "notes": Annotated[list, DeltaChannel(_messages_delta_reducer)],
        },
    )
    graph = (
        StateGraph(State)
        .add_node("model", lambda state: {})
        .add_edge(START, "model")
        .compile(checkpointer=saver)
    )
    config = {"configurable": {"thread_id": "fresh-unwritten"}}

    graph.update_state(
        config,
        {"messages": [HumanMessage(content="hello", id="m1")]},
        as_node="model",
    )

    head = saver.get_tuple(config)
    assert head is not None
    assert "notes" not in head.checkpoint["channel_versions"]
    assert graph.get_state(config).values["notes"] == []


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
    """

    saver = InMemorySaver()
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "bulk-multi-task"}}
    graph.invoke({"messages": [HumanMessage(content="hi", id="hi")]}, config)
    base = saver.get_tuple(config)
    assert base is not None

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

    stored = saver.get_tuple(base.config)
    assert stored is not None
    assert {task_id for task_id, _, _ in stored.pending_writes or []} == {
        "task-1",
        "task-2",
    }, "explicit task ids must key the stored writes"
    state = graph.get_state(config)
    contents = [m.content for m in state.values["messages"]]
    ids = [m.id for m in state.values["messages"]]
    assert sorted(contents) == ["first", "hi", "second"], (
        f"both updates' writes must persist; got {contents}"
    )
    assert sorted(ids) == ["hi", "m1", "m2"]


def _update(content: str, as_node: str) -> StateUpdate:
    return StateUpdate(
        values={"messages": [HumanMessage(content=content, id=content)]},
        as_node=as_node,
    )


def _contents(state: StateSnapshot) -> list[str]:
    return [m.content for m in state.values["messages"]]


def test_bulk_update_state_keeps_every_update_without_task_ids(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _build_graph(sync_checkpointer, two_nodes=True)
    config = {"configurable": {"thread_id": "bulk-no-task-ids"}}
    graph.invoke({"messages": [HumanMessage(content="hi", id="hi")]}, config)

    graph.bulk_update_state(
        config,
        [
            [
                _update("first", "model"),
                _update("second", "model"),
                _update("third", "assistant"),
            ]
        ],
    )

    contents = _contents(graph.get_state(config))
    assert sorted(contents) == ["first", "hi", "second", "third"], (
        f"every update's writes must persist; got {contents}"
    )


async def test_abulk_update_state_keeps_every_update_without_task_ids(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _build_graph(async_checkpointer, two_nodes=True)
    config = {"configurable": {"thread_id": "bulk-no-task-ids"}}
    await graph.ainvoke({"messages": [HumanMessage(content="hi", id="hi")]}, config)

    await graph.abulk_update_state(
        config,
        [
            [
                _update("first", "model"),
                _update("second", "model"),
                _update("third", "assistant"),
            ]
        ],
    )

    contents = _contents(await graph.aget_state(config))
    assert sorted(contents) == ["first", "hi", "second", "third"], (
        f"every update's writes must persist; got {contents}"
    )


def test_bulk_update_state_keeps_every_update_next_to_a_pending_task(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _build_graph(
        sync_checkpointer, two_nodes=True, interrupt_before=["assistant"]
    )
    config = {"configurable": {"thread_id": "bulk-pending-task"}}
    graph.invoke({"messages": [HumanMessage(content="hi", id="hi")]}, config)
    assert graph.get_state(config).next == ("assistant",)

    graph.bulk_update_state(
        config,
        [
            [
                _update("first", "assistant"),
                _update("second", "model"),
                _update("third", "model"),
            ]
        ],
    )

    contents = _contents(graph.get_state(config))
    assert sorted(contents) == ["first", "hi", "second", "third"], (
        f"every update's writes must persist; got {contents}"
    )


class _TaskPathOrderSaver(InMemorySaver):
    """Replays each checkpoint's writes by `(task_path, task_id, idx)`."""

    def get_tuple(self, config: Any) -> Any:
        tup = super().get_tuple(config)
        if tup is None or not tup.pending_writes:
            return tup
        conf = tup.config["configurable"]
        stored = self.writes[
            (conf["thread_id"], conf["checkpoint_ns"], conf["checkpoint_id"])
        ]
        rows = sorted(
            zip(stored.items(), tup.pending_writes),
            key=lambda row: (row[0][1][3], *row[0][0]),
        )
        return tup._replace(pending_writes=[write for _, write in rows])

    get_delta_channel_history = BaseCheckpointSaver.get_delta_channel_history
    aget_delta_channel_history = BaseCheckpointSaver.aget_delta_channel_history


GIVEN = ["u1", "u2", "u3", "u4", "u5", "u6"]


def _updates_in_given_order() -> list[list[StateUpdate]]:
    return [
        [_update(c, "assistant" if i % 2 else "model") for i, c in enumerate(GIVEN)]
    ]


def test_bulk_update_state_replays_updates_in_the_order_given() -> None:
    graph = _build_graph(_TaskPathOrderSaver(), two_nodes=True)
    config = {"configurable": {"thread_id": "bulk-order"}}
    graph.invoke({"messages": [HumanMessage(content="hi", id="hi")]}, config)

    graph.bulk_update_state(config, _updates_in_given_order())

    assert _contents(graph.get_state(config)) == ["hi", *GIVEN]


async def test_abulk_update_state_replays_updates_in_the_order_given() -> None:
    graph = _build_graph(_TaskPathOrderSaver(), two_nodes=True)
    config = {"configurable": {"thread_id": "bulk-order"}}
    await graph.ainvoke({"messages": [HumanMessage(content="hi", id="hi")]}, config)

    await graph.abulk_update_state(config, _updates_in_given_order())

    assert _contents(await graph.aget_state(config)) == ["hi", *GIVEN]


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


def test_update_state_that_snapshots_keeps_a_deferred_node_pending() -> None:
    channel = DeltaChannel(_messages_delta_reducer, snapshot_frequency=1)

    class State(TypedDict):
        messages: Annotated[list, channel]

    builder = StateGraph(State)
    builder.add_node("a", lambda state: {"messages": [HumanMessage("a", id="a")]})
    builder.add_node(
        "b", lambda state: {"messages": [HumanMessage("b", id="b")]}, defer=True
    )
    builder.add_node("c", lambda state: {})
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    graph = builder.compile(checkpointer=InMemorySaver(), interrupt_after=["a"])
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage("s", id="s")]}, config)

    graph.update_state(config, {"messages": [HumanMessage("u", id="u")]}, as_node="c")
    final = graph.invoke(None, config)

    assert [m.content for m in final["messages"]] == ["s", "a", "u", "b"]
    assert graph.get_state(config).next == ()


def _sorted_extend(current: list, writes: list) -> list:
    return sorted([*current, *(item for write in writes for item in write)])


def _delta_input_graph(snapshot_frequency: int = 1000) -> Any:
    node = NodeBuilder().subscribe_only("go").do(lambda _: [2]).write_to("log", "plain")
    return Pregel(
        nodes={"n": node},
        channels={
            "log": DeltaChannel(_sorted_extend, snapshot_frequency=snapshot_frequency),
            "plain": BinaryOperatorAggregate(list, lambda a, b: sorted(a + b)),
            "go": LastValue(int),
        },
        input_channels=["log", "plain", "go"],
        output_channels=["log", "plain"],
        checkpointer=InMemorySaver(),
    )


@pytest.mark.parametrize("snapshot_frequency", [1, 2])
def test_update_as_input_reads_back_on_its_checkpoint_and_after_the_next_run(
    snapshot_frequency: int,
) -> None:
    graph = _delta_input_graph(snapshot_frequency)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"log": [0], "plain": [0], "go": 1}, config)

    graph.update_state(config, {"log": [1], "plain": [1], "go": 1}, as_node="__input__")
    after_update = graph.get_state(config).values
    graph.invoke(None, config)
    after_run = graph.get_state(config).values

    assert after_update["log"] == after_update["plain"]
    assert after_run["log"] == after_run["plain"]


@pytest.mark.parametrize("snapshot_frequency", [1, 2])
async def test_aupdate_as_input_reads_back_on_its_checkpoint_and_after_the_next_run(
    snapshot_frequency: int,
) -> None:
    graph = _delta_input_graph(snapshot_frequency)
    config = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke({"log": [0], "plain": [0], "go": 1}, config)

    await graph.aupdate_state(
        config, {"log": [1], "plain": [1], "go": 1}, as_node="__input__"
    )
    after_update = (await graph.aget_state(config)).values
    await graph.ainvoke(None, config)
    after_run = (await graph.aget_state(config)).values

    assert after_update["log"] == after_update["plain"]
    assert after_run["log"] == after_run["plain"]


def test_update_as_input_to_an_older_checkpoint_stays_out_of_its_other_branch() -> None:
    graph = _delta_input_graph()
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"go": 1}, config)
    older = graph.get_state(config).config
    graph.invoke({"go": 1}, config)
    other_branch = graph.get_state(config)

    edited = graph.update_state(
        older, {"log": [1], "plain": [1], "go": 1}, as_node="__input__"
    )

    values = graph.get_state(edited).values
    assert values["log"] == values["plain"]
    assert graph.get_state(other_branch.config).values == other_branch.values


async def test_aupdate_as_input_to_an_older_checkpoint_stays_out_of_its_other_branch() -> (
    None
):
    graph = _delta_input_graph()
    config = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke({"go": 1}, config)
    older = (await graph.aget_state(config)).config
    await graph.ainvoke({"go": 1}, config)
    other_branch = await graph.aget_state(config)

    edited = await graph.aupdate_state(
        older, {"log": [1], "plain": [1], "go": 1}, as_node="__input__"
    )

    values = (await graph.aget_state(edited)).values
    assert values["log"] == values["plain"]
    assert (await graph.aget_state(other_branch.config)).values == other_branch.values
