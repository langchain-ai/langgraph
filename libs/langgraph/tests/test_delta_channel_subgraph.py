import operator
from typing import Annotated, Any, Literal

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph._internal._constants import CONFIG_KEY_CHECKPOINTER
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph
from langgraph.pregel._checkpoint import (
    achannels_from_checkpoint,
    channels_from_checkpoint,
    empty_checkpoint,
)

pytestmark = pytest.mark.anyio


def _extend(state: list | None, writes: list[Any]) -> list:
    out = list(state or [])
    for write in writes:
        out.extend(write if isinstance(write, list) else [write])
    return out


def _state_schema(snapshot_frequency: int = 1000) -> type:
    class State(TypedDict, total=False):
        delta: Annotated[
            list, DeltaChannel(_extend, snapshot_frequency=snapshot_frequency)
        ]
        plain: Annotated[list, operator.add]

    return State


def _both(*items: str) -> dict:
    return {"delta": list(items), "plain": list(items)}


def _child_builder(*, snapshot_frequency: int = 1000) -> StateGraph:
    builder = StateGraph(_state_schema(snapshot_frequency))
    builder.add_node("a", lambda state: _both("a1"))
    builder.add_node("b", lambda state: _both("b1", "b2"))
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    return builder


def _wrap(
    inner: StateGraph,
    *,
    checkpointer: bool | None = None,
    interrupt_before: list[str] | None = None,
) -> StateGraph:
    builder = StateGraph(inner.state_schema)
    builder.add_node(
        "child",
        inner.compile(checkpointer=checkpointer, interrupt_before=interrupt_before),
    )
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    return builder


def _nested_app(
    checkpointer: BaseCheckpointSaver,
    *,
    depth: int = 1,
    snapshot_frequency: int = 1000,
    pause_before_b: bool = False,
    subgraph_checkpointer: bool | None = None,
) -> Any:
    graph = _child_builder(snapshot_frequency=snapshot_frequency)
    for _ in range(depth):
        graph = _wrap(
            graph,
            checkpointer=subgraph_checkpointer,
            interrupt_before=["b"] if pause_before_b else None,
        )
    return graph.compile(checkpointer=checkpointer)


def _scoped(config: dict, namespace: str) -> dict:
    return {"configurable": {**config["configurable"], "checkpoint_ns": namespace}}


def _child_namespace(app: Any, config: dict, *, depth: int = 1) -> str:
    namespace = ""
    for level in range(depth):
        scoped = _scoped(config, namespace) if namespace else config
        namespace = next(
            (
                task.state["configurable"]["checkpoint_ns"]
                for snapshot in app.get_state_history(scoped)
                for task in snapshot.tasks
                if task.name == "child" and isinstance(task.state, dict)
            ),
            "",
        )
        assert namespace, f"no `child` subgraph task at nesting level {level}"
    return namespace


async def _achild_namespace(app: Any, config: dict) -> str:
    async for snapshot in app.aget_state_history(config):
        for task in snapshot.tasks:
            if task.name == "child" and isinstance(task.state, dict):
                return task.state["configurable"]["checkpoint_ns"]
    raise AssertionError("no `child` subgraph task")


HISTORY = [_both("a1", "b1", "b2"), _both("a1"), _both(), _both()]


def test_subgraph_get_state(sync_checkpointer: BaseCheckpointSaver) -> None:
    app = _nested_app(sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    child = _scoped(config, _child_namespace(app, config))

    assert app.get_state(config).values == _both("a1", "b1", "b2")
    assert app.get_state(child).values == _both("a1", "b1", "b2")


async def test_subgraph_aget_state(async_checkpointer: BaseCheckpointSaver) -> None:
    app = _nested_app(async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({}, config)

    child = _scoped(config, await _achild_namespace(app, config))

    assert (await app.aget_state(child)).values == _both("a1", "b1", "b2")


def test_subgraph_get_state_history(sync_checkpointer: BaseCheckpointSaver) -> None:
    app = _nested_app(sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    child = _scoped(config, _child_namespace(app, config))

    assert [s.values for s in app.get_state_history(child)] == HISTORY


async def test_subgraph_aget_state_history(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    app = _nested_app(async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({}, config)

    child = _scoped(config, await _achild_namespace(app, config))

    assert [s.values async for s in app.aget_state_history(child)] == HISTORY


def test_doubly_nested_subgraph_get_state(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    app = _nested_app(sync_checkpointer, depth=2)
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    child = _scoped(config, _child_namespace(app, config, depth=2))

    assert app.get_state(child).values == _both("a1", "b1", "b2")


@pytest.mark.parametrize("persistence", ["per-invocation", "per-thread"])
def test_interrupted_subgraph_task_state(
    sync_checkpointer: BaseCheckpointSaver,
    persistence: Literal["per-invocation", "per-thread"],
) -> None:
    app = _nested_app(
        sync_checkpointer,
        pause_before_b=True,
        subgraph_checkpointer=True if persistence == "per-thread" else None,
    )
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    (task,) = app.get_state(config, subgraphs=True).tasks

    assert task.state.values == _both("a1")


@pytest.mark.parametrize("persistence", ["per-invocation", "per-thread"])
async def test_interrupted_subgraph_task_state_async(
    async_checkpointer: BaseCheckpointSaver,
    persistence: Literal["per-invocation", "per-thread"],
) -> None:
    app = _nested_app(
        async_checkpointer,
        pause_before_b=True,
        subgraph_checkpointer=True if persistence == "per-thread" else None,
    )
    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({}, config)

    (task,) = (await app.aget_state(config, subgraphs=True)).tasks

    assert task.state.values == _both("a1")


@pytest.mark.parametrize("persistence", ["per-invocation", "per-thread"])
def test_interrupted_subgraph_history_from_its_task_config(
    sync_checkpointer: BaseCheckpointSaver,
    persistence: Literal["per-invocation", "per-thread"],
) -> None:
    app = _nested_app(
        sync_checkpointer,
        pause_before_b=True,
        subgraph_checkpointer=True if persistence == "per-thread" else None,
    )
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)
    (task,) = app.get_state(config).tasks

    history = list(app.get_state_history(task.state))

    assert [snapshot.values for snapshot in history][:1] == [_both("a1")]


@pytest.mark.parametrize("persistence", ["per-invocation", "per-thread"])
async def test_interrupted_subgraph_ahistory_from_its_task_config(
    async_checkpointer: BaseCheckpointSaver,
    persistence: Literal["per-invocation", "per-thread"],
) -> None:
    app = _nested_app(
        async_checkpointer,
        pause_before_b=True,
        subgraph_checkpointer=True if persistence == "per-thread" else None,
    )
    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({}, config)
    (task,) = (await app.aget_state(config)).tasks

    history = [snapshot async for snapshot in app.aget_state_history(task.state)]

    assert [snapshot.values for snapshot in history][:1] == [_both("a1")]


@pytest.mark.parametrize("persistence", ["per-invocation", "per-thread"])
def test_interrupted_subgraph_update_state_from_its_task_config(
    sync_checkpointer: BaseCheckpointSaver,
    persistence: Literal["per-invocation", "per-thread"],
) -> None:
    app = _nested_app(
        sync_checkpointer,
        pause_before_b=True,
        subgraph_checkpointer=True if persistence == "per-thread" else None,
    )
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)
    (task,) = app.get_state(config).tasks

    app.update_state(task.state, _both("edit"), as_node="a")

    (task,) = app.get_state(config, subgraphs=True).tasks
    assert task.state.values == _both("a1", "edit")


@pytest.mark.parametrize("persistence", ["per-invocation", "per-thread"])
async def test_interrupted_subgraph_aupdate_state_from_its_task_config(
    async_checkpointer: BaseCheckpointSaver,
    persistence: Literal["per-invocation", "per-thread"],
) -> None:
    app = _nested_app(
        async_checkpointer,
        pause_before_b=True,
        subgraph_checkpointer=True if persistence == "per-thread" else None,
    )
    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({}, config)
    (task,) = (await app.aget_state(config)).tasks

    await app.aupdate_state(task.state, _both("edit"), as_node="a")

    (task,) = (await app.aget_state(config, subgraphs=True)).tasks
    assert task.state.values == _both("a1", "edit")


def test_subgraph_update_state_keeps_history(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    app = _nested_app(sync_checkpointer, snapshot_frequency=2)
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    child = _scoped(config, _child_namespace(app, config))
    app.update_state(child, _both("manual"))

    assert app.get_state(child).values == _both("a1", "b1", "b2", "manual")


async def test_subgraph_aupdate_state_keeps_history(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    app = _nested_app(async_checkpointer, snapshot_frequency=2)
    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({}, config)

    child = _scoped(config, await _achild_namespace(app, config))
    await app.aupdate_state(child, _both("manual"))

    assert (await app.aget_state(child)).values == _both("a1", "b1", "b2", "manual")


def test_stateless_subgraph_persists_nothing(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    app = _nested_app(sync_checkpointer, subgraph_checkpointer=False)
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    child_tasks = [
        task
        for snapshot in app.get_state_history(config)
        for task in snapshot.tasks
        if task.name == "child" and isinstance(task.state, dict)
    ]

    assert child_tasks == []
    assert app.get_state(config).values == _both("a1", "b1", "b2")


def test_completed_subgraph_exposes_no_task_state(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    app = _nested_app(sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    assert app.get_state(config, subgraphs=True).tasks == ()


def _written_delta_checkpoint() -> Any:
    checkpoint = empty_checkpoint()
    checkpoint["channel_versions"]["delta"] = 1
    return checkpoint


def test_hydrating_written_delta_channel_without_saver_raises() -> None:
    with pytest.raises(ValueError, match="no checkpointer"):
        channels_from_checkpoint(
            {"delta": DeltaChannel(_extend)}, _written_delta_checkpoint()
        )


async def test_ahydrating_written_delta_channel_without_saver_raises() -> None:
    with pytest.raises(ValueError, match="no checkpointer"):
        await achannels_from_checkpoint(
            {"delta": DeltaChannel(_extend)}, _written_delta_checkpoint()
        )


def test_hydrating_written_delta_channel_without_config_raises() -> None:
    with pytest.raises(ValueError, match="no checkpointer"):
        channels_from_checkpoint(
            {"delta": DeltaChannel(_extend)},
            _written_delta_checkpoint(),
            saver=InMemorySaver(),
        )


def test_hydrating_unwritten_delta_channel_without_saver_is_empty() -> None:
    channels, _ = channels_from_checkpoint(
        {"delta": DeltaChannel(_extend)}, empty_checkpoint()
    )
    assert channels["delta"].get() == []


def test_root_checkpointer_true_graph_state_read_raises() -> None:
    app = _child_builder().compile(checkpointer=True)
    with pytest.raises(RuntimeError, match="checkpointer=True cannot be used"):
        app.get_state({"configurable": {"thread_id": "1"}})


def test_stateless_graph_update_state_ignores_lent_saver() -> None:
    saver = InMemorySaver()
    app = _child_builder().compile(checkpointer=False)
    config = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "child:1",
            CONFIG_KEY_CHECKPOINTER: saver,
        }
    }
    with pytest.raises(ValueError, match="No checkpointer set"):
        app.update_state(config, _both("x"), as_node="a")
    assert list(saver.list(None)) == []


def test_graph_without_a_checkpointer_reads_through_a_lent_saver(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    app = _child_builder().compile()
    config = {
        "configurable": {"thread_id": "1", CONFIG_KEY_CHECKPOINTER: sync_checkpointer}
    }
    app.invoke(_both("in"), config)

    assert app.get_state(config).values == _both("in", "a1", "b1", "b2")
    app.update_state(config, _both("edit"), as_node="b")
    assert app.get_state(config).values == _both("in", "a1", "b1", "b2", "edit")
    assert next(iter(app.get_state_history(config))).values == _both(
        "in", "a1", "b1", "b2", "edit"
    )


async def test_graph_without_a_checkpointer_areads_through_a_lent_saver(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    app = _child_builder().compile()
    config = {
        "configurable": {"thread_id": "1", CONFIG_KEY_CHECKPOINTER: async_checkpointer}
    }
    await app.ainvoke(_both("in"), config)

    assert (await app.aget_state(config)).values == _both("in", "a1", "b1", "b2")
    await app.aupdate_state(config, _both("edit"), as_node="b")
    assert (await app.aget_state(config)).values == _both(
        "in", "a1", "b1", "b2", "edit"
    )
    history = [s async for s in app.aget_state_history(config)]
    assert history[0].values == _both("in", "a1", "b1", "b2", "edit")


def test_second_call_of_a_checkpointer_true_subgraph_reads_its_own_history(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    child = _child_builder().compile(checkpointer=True)

    def node(state: dict) -> dict:
        child.invoke(_both("first"))
        child.invoke(_both("second"))
        return {}

    builder = StateGraph(_state_schema())
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.compile(checkpointer=sync_checkpointer).invoke(
        _both(), {"configurable": {"thread_id": "1"}}
    )

    def read(namespace: str) -> dict:
        return child.get_state(
            {
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": namespace,
                    CONFIG_KEY_CHECKPOINTER: sync_checkpointer,
                }
            }
        ).values

    assert read("node") == _both("first", "a1", "b1", "b2")
    assert read("node|1") == _both("second", "a1", "b1", "b2")


async def test_second_call_of_a_checkpointer_true_subgraph_areads_its_own_history(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    child = _child_builder().compile(checkpointer=True)

    async def node(state: dict, config: RunnableConfig) -> dict:
        await child.ainvoke(_both("first"), config)
        await child.ainvoke(_both("second"), config)
        return {}

    builder = StateGraph(_state_schema())
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    await builder.compile(checkpointer=async_checkpointer).ainvoke(
        _both(), {"configurable": {"thread_id": "1"}}
    )

    async def read(namespace: str) -> dict:
        snapshot = await child.aget_state(
            {
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": namespace,
                    CONFIG_KEY_CHECKPOINTER: async_checkpointer,
                }
            }
        )
        return snapshot.values

    assert await read("node") == _both("first", "a1", "b1", "b2")
    assert await read("node|1") == _both("second", "a1", "b1", "b2")
