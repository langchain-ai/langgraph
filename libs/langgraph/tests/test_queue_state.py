"""Tests for `Pregel.queue_state`: durable state updates applied by a run at
the superstep boundary they name."""

from __future__ import annotations

import asyncio
import operator
import threading
import time
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.errors import InvalidUpdateError
from langgraph.graph import StateGraph
from langgraph.pregel._queue import QUEUE_NS, list_pending, queue_config
from langgraph.types import Command, Durability, QueuedUpdate, interrupt

pytestmark = pytest.mark.anyio


class State(TypedDict):
    log: Annotated[list[str], operator.add]


class Gate:
    """Lets a test hold a node open until it has queued an update."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def wait(self) -> None:
        self.entered.set()
        assert self.release.wait(10), "gate was never released"


def chain(
    gate: Gate | None = None,
    gated: str = "a",
    delay: float = 0.05,
    *,
    key: str = "log",
    schema: type = State,
    override: dict[str, Any] | None = None,
):
    """START -> a -> b -> c -> d -> END. The gated node blocks until released;
    every node sleeps briefly so in-flight queue reads complete before the
    next boundary, which keeps the `async` and `exit` schedules deterministic.
    """

    def node(name: str):
        def fn(state: Any) -> dict:
            if gate is not None and name == gated:
                gate.wait()
            time.sleep(delay)
            return {key: [name]}

        return fn

    builder = StateGraph(schema)
    for name in "abcd":
        builder.add_node(name, (override or {}).get(name) or node(name))
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", "d")
    builder.add_edge("d", END)
    return builder


def run_in_thread(fn, *args, **kwargs) -> tuple[threading.Thread, dict]:
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # pragma: no cover - surfaced by the test
            result["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    return thread, result


def test_queue_on_idle_thread_is_visible_and_applied_by_next_run(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = chain().compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    item_id = graph.queue_state(config, {"log": ["steer"]}, steer="c")

    # visible, not applied, on a thread that has no checkpoint yet
    snapshot = graph.get_state(config)
    assert snapshot.values == {}
    assert snapshot.queued == (
        QueuedUpdate(item_id, {"log": ["steer"]}, "c", snapshot.queued[0].accepted_at),
    )

    # the next run applies it before c, after its own input
    result = graph.invoke({"log": ["in"]}, config)
    assert result["log"] == ["in", "a", "b", "steer", "c", "d"]
    assert graph.get_state(config).queued == ()
    # acknowledged in the saver, not merely hidden
    assert list_pending(sync_checkpointer, config) == []
    # recorded by the checkpoint that carried it
    assert any(
        item_id in (s.metadata or {}).get("queue_consumed", ())
        for s in graph.get_state_history(config)
    )


@pytest.mark.parametrize("steer", ["c", None])
def test_queue_mid_run(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability, steer: str | None
) -> None:
    gate = Gate()
    graph = chain(gate).compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    thread, result = run_in_thread(
        graph.invoke, {"log": ["in"]}, config, durability=durability
    )
    assert gate.entered.wait(10)
    graph.queue_state(config, {"log": ["steer"]}, steer=steer)
    assert graph.get_state(config).queued[0].steer == steer
    gate.release.set()
    thread.join(20)
    assert "error" not in result, result.get("error")

    if steer == "c" and durability != "exit":
        # "sync" sees it at the next boundary, "async" one boundary later;
        # both are before c
        expected = ["in", "a", "b", "steer", "c", "d"]
    else:
        # a follow-up, or "exit" mode which only reads at the end: applied
        # when the run would finish, then the graph continues from START
        expected = ["in", "a", "b", "c", "d", "steer", "a", "b", "c", "d"]
    assert result["value"]["log"] == expected
    assert graph.get_state(config).queued == ()
    assert list_pending(sync_checkpointer, config) == []


def test_follow_up_writes_an_input_checkpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    gate = Gate()
    graph = chain(gate).compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    thread, result = run_in_thread(graph.invoke, {"log": ["in"]}, config)
    assert gate.entered.wait(10)
    graph.queue_state(config, {"log": ["follow"]})
    gate.release.set()
    thread.join(20)
    assert result["value"]["log"] == [
        "in",
        "a",
        "b",
        "c",
        "d",
        "follow",
        "a",
        "b",
        "c",
        "d",
    ]
    sources = [s.metadata["source"] for s in graph.get_state_history(config)]
    # oldest first: the run's input, its steps, the follow-up's input, more steps
    assert sources[::-1].count("input") == 2
    # the queue namespace never leaks into history
    assert all(
        s.config["configurable"]["checkpoint_ns"] == ""
        for s in graph.get_state_history(config)
    )


def test_resume_does_not_consume(sync_checkpointer: BaseCheckpointSaver) -> None:
    seen_by_b: list[list[str]] = []

    def b(state: State) -> dict:
        seen_by_b.append(list(state["log"]))
        answer = interrupt("go?")
        time.sleep(0.05)  # let the in-flight queue read complete, as chain() does
        return {"log": ["b", answer]}

    graph = chain(override={"b": b}).compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"log": ["in"]}, config)
    assert graph.get_state(config).next == ("b",)

    # queued while paused at b's interrupt
    graph.queue_state(config, {"log": ["steer"]}, steer="c")
    # invoke(None) with nothing to resume completes no step: nothing consumed
    assert len(graph.get_state(config).queued) == 1

    result = graph.invoke(Command(resume="yes"), config)
    # b re-ran against the state it was interrupted on
    assert seen_by_b[-1] == ["in", "a"]
    # the item landed at the next boundary that steers c
    assert result["log"] == ["in", "a", "b", "yes", "steer", "c", "d"]
    assert graph.get_state(config).queued == ()


def test_finished_thread_invoke_none_consumes_nothing(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = chain().compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"log": ["in"]}, config)
    graph.queue_state(config, {"log": ["late"]})
    before = len(list(graph.get_state_history(config)))
    graph.invoke(None, config)
    assert len(list(graph.get_state_history(config))) == before
    assert len(graph.get_state(config).queued) == 1
    # the next run with input takes it at its end and continues
    result = graph.invoke({"log": ["in2"]}, config)
    assert result["log"] == [
        "in",
        "a",
        "b",
        "c",
        "d",
        "in2",
        "a",
        "b",
        "c",
        "d",
        "late",
        "a",
        "b",
        "c",
        "d",
    ]


def test_items_apply_in_accept_order(sync_checkpointer: BaseCheckpointSaver) -> None:
    graph = chain().compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    graph.queue_state(config, {"log": ["one"]}, steer="c")
    graph.queue_state(config, {"log": ["two"]}, steer="c")
    graph.queue_state(config, {"log": ["three"]}, steer="d")
    result = graph.invoke({"log": ["in"]}, config)
    assert result["log"] == ["in", "a", "b", "one", "two", "c", "three", "d"]


def test_validation(sync_checkpointer: BaseCheckpointSaver) -> None:
    class Typed(TypedDict):
        n: Annotated[int, operator.add]
        s: str

    builder = StateGraph(Typed)
    builder.add_node("a", lambda s: {"n": 1})
    builder.add_edge(START, "a")
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    with pytest.raises(InvalidUpdateError, match="Unknown channel"):
        graph.queue_state(config, {"nope": 1})
    with pytest.raises(InvalidUpdateError, match="non-empty"):
        graph.queue_state(config, {})
    with pytest.raises(InvalidUpdateError, match="rejected"):
        graph.queue_state(config, {"n": "not a number"})
    with pytest.raises(ValueError, match="not found"):
        graph.queue_state(config, {"n": 1}, steer="zzz")
    with pytest.raises(ValueError, match="not found"):
        graph.queue_state(config, {"n": 1}, steer=END)
    # nothing was written
    assert graph.get_state(config).queued == ()
    # a valid one is accepted
    graph.queue_state(config, {"n": 1, "s": "x"}, steer="a")
    assert len(graph.get_state(config).queued) == 1

    with pytest.raises(ValueError, match="No checkpointer"):
        builder.compile().queue_state(config, {"n": 1})


def test_stale_item_is_acked_not_reapplied(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """A crash between the checkpoint that records an item as consumed and
    the ack: the resumed loop acks it and does not apply it again."""
    graph = chain().compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    item_id = graph.queue_state(config, {"log": ["steer"]}, steer="d")
    graph.invoke({"log": ["in"]}, config)
    head = graph.get_state(config)
    assert head.metadata["queue_consumed"] == [item_id]

    # undo the ack, as if the process had died before it landed
    qconfig = queue_config(config)
    saved = next(iter(sync_checkpointer.list(qconfig)))
    sync_checkpointer.put(
        qconfig, saved.checkpoint, {**saved.metadata, "consumed": False}, {}
    )
    assert [i.id for i in list_pending(sync_checkpointer, config)] == [item_id]
    # hidden from the snapshot, because the head records it as consumed
    assert graph.get_state(config).queued == ()

    result = graph.invoke({"log": ["in2"]}, config)
    assert result["log"] == [
        "in",
        "a",
        "b",
        "c",
        "steer",
        "d",
        "in2",
        "a",
        "b",
        "c",
        "d",
    ]
    assert list_pending(sync_checkpointer, config) == []


def test_delete_thread_removes_queue(sync_checkpointer: BaseCheckpointSaver) -> None:
    graph = chain().compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    graph.queue_state(config, {"log": ["x"]})
    assert len(list_pending(sync_checkpointer, config)) == 1
    sync_checkpointer.delete_thread("1")
    assert list_pending(sync_checkpointer, config) == []
    assert graph.get_state(config).queued == ()


def test_subgraph_is_addressed_by_namespace(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    class Both(TypedDict):
        log: Annotated[list[str], operator.add]
        clog: Annotated[list[str], operator.add]

    class ChildState(TypedDict):
        clog: Annotated[list[str], operator.add]

    gate = Gate()
    # the child works on its own key, so its output is not added back into
    # the parent's log
    child = chain(gate, gated="a", key="clog", schema=ChildState).compile()

    parent_builder = StateGraph(Both)
    parent_builder.add_node("outer", lambda s: {"log": ["outer"]})
    parent_builder.add_node("child", child)
    parent_builder.add_edge(START, "outer")
    parent_builder.add_edge("outer", "child")
    parent_builder.add_edge("child", END)
    parent = parent_builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    thread, result = run_in_thread(parent.invoke, {"log": ["in"]}, config)
    assert gate.entered.wait(10)
    # the child's namespace, as get_state(subgraphs=True) reports it, once
    # the child's first checkpoint write has landed
    for _ in range(100):
        sub = parent.get_state(config, subgraphs=True).tasks[0].state
        if sub is not None:
            break
        time.sleep(0.02)
    assert sub is not None
    assert sub.config["configurable"]["checkpoint_ns"].startswith("child:")
    item_id = parent.queue_state(sub.config, {"clog": ["steer"]}, steer="c")
    # queued against the child's namespace, visible through the child snapshot
    sub_after = parent.get_state(config, subgraphs=True).tasks[0].state
    assert [q.id for q in sub_after.queued] == [item_id]
    assert parent.get_state(config).queued == ()
    with pytest.raises(ValueError, match="not found"):
        parent.queue_state(sub.config, {"clog": ["x"]}, steer="outer")
    with pytest.raises(InvalidUpdateError, match="Unknown channel"):
        parent.queue_state(sub.config, {"log": ["x"]}, steer="c")
    gate.release.set()
    thread.join(20)
    assert "error" not in result, result.get("error")
    # the child consumed it at its own boundary, before c
    assert result["value"]["log"] == ["in", "outer"]
    assert result["value"]["clog"] == ["a", "b", "steer", "c", "d"]
    assert parent.get_state(config, subgraphs=True).queued == ()
    assert list_pending(sync_checkpointer, sub.config) == []
    # queue namespaces are never part of a thread's history
    assert all(
        QUEUE_NS not in s.config["configurable"]["checkpoint_ns"]
        for s in parent.get_state_history(config)
    )


async def test_async_queue_mid_run(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    gate = Gate()
    graph = chain(gate).compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    task = asyncio.create_task(
        graph.ainvoke({"log": ["in"]}, config, durability=durability)
    )
    await asyncio.to_thread(gate.entered.wait, 10)
    item_id = await graph.aqueue_state(config, {"log": ["steer"]}, steer="c")
    assert [q.id for q in (await graph.aget_state(config)).queued] == [item_id]
    gate.release.set()
    result = await task

    if durability != "exit":
        expected = ["in", "a", "b", "steer", "c", "d"]
    else:
        expected = ["in", "a", "b", "c", "d", "steer", "a", "b", "c", "d"]
    assert result["log"] == expected
    assert (await graph.aget_state(config)).queued == ()


async def test_async_idle_then_follow_up(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = chain().compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    await graph.aqueue_state(config, {"log": ["follow"]})
    result = await graph.ainvoke({"log": ["in"]}, config)
    assert result["log"] == ["in", "a", "b", "c", "d", "follow", "a", "b", "c", "d"]
    assert (await graph.aget_state(config)).queued == ()
