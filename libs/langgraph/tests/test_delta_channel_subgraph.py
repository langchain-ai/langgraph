import operator
from typing import Annotated, Any, Literal

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph

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
