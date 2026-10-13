import operator
import sys
from collections.abc import Callable
from contextlib import contextmanager
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.base import PRUNE_DELTA_THREADS_SQL, PRUNE_HEADS_SQL
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import START, StateGraph
from langgraph.graph.message import _messages_delta_reducer, add_messages
from langgraph.types import Command, StateSnapshot, interrupt
from tests.conftest import NO_DOCKER
from tests.conftest_checkpointer import (
    DEFAULT_POSTGRES_URI,
    _checkpointer_postgres,
    _checkpointer_postgres_aio,
    _checkpointer_postgres_aio_pipe,
    _checkpointer_postgres_aio_pool,
    _checkpointer_postgres_pipe,
    _checkpointer_postgres_pool,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(NO_DOCKER, reason="needs Postgres"),
]

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)

SYNC = [
    _checkpointer_postgres,
    _checkpointer_postgres_pipe,
    _checkpointer_postgres_pool,
]
ASYNC = [
    _checkpointer_postgres_aio,
    _checkpointer_postgres_aio_pipe,
    _checkpointer_postgres_aio_pool,
]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    notes: Annotated[list[str], operator.add]


class DeltaState(TypedDict):
    messages: Annotated[list[AnyMessage], DeltaChannel(_messages_delta_reducer)]


def _reply(ask: bool):
    def reply(state: dict) -> dict:
        n = len(state["messages"])
        answer = interrupt("approve?") if ask else "ok"
        return {"messages": [AIMessage(f"{answer} {n}", id=f"reply-{n}")]}

    return reply


def _graph(
    checkpointer: Any,
    *,
    ask: bool = False,
    fail: set[str] | None = None,
    calls: dict[str, int] | None = None,
) -> Any:
    def note(name: str):
        def node(state: State) -> dict:
            if calls is not None:
                calls[name] = calls.get(name, 0) + 1
            if fail is not None and name in fail:
                fail.discard(name)
                raise ValueError(f"{name} failed")
            return {"notes": [name]}

        return node

    builder = StateGraph(State)
    builder.add_node("left", note("left"))
    builder.add_node("right", note("right"))
    builder.add_node("reply", _reply(ask))
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    builder.add_edge(["left", "right"], "reply")
    return builder.compile(checkpointer=checkpointer)


def _delta_graph(checkpointer: Any) -> Any:
    builder = StateGraph(DeltaState)
    builder.add_node("reply", _reply(ask=False))
    builder.add_edge(START, "reply")
    return builder.compile(checkpointer=checkpointer)


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _say(i: int) -> dict:
    return {"messages": [HumanMessage(f"hi {i}", id=f"hi-{i}")]}


def _checkpoint_ids(checkpointer: Any, thread_id: str) -> list[str]:
    return [c.checkpoint["id"] for c in checkpointer.list(_config(thread_id))]


async def _acheckpoint_ids(checkpointer: Any, thread_id: str) -> list[str]:
    return [c.checkpoint["id"] async for c in checkpointer.alist(_config(thread_id))]


def _parentless(snapshot: StateSnapshot) -> StateSnapshot:
    return snapshot._replace(
        parent_config=None,
        tasks=tuple(
            t._replace(state=_parentless(t.state))
            if isinstance(t.state, StateSnapshot)
            else t
            for t in snapshot.tasks
        ),
    )


def _run_once_after(
    saver: PostgresSaver, sql: str, action: Callable[[], Any], *, nth: int = 1
) -> None:
    seen = 0
    cursor = saver._cursor

    class Hooked:
        def __init__(self, cur: Any) -> None:
            self.cur = cur

        def __getattr__(self, name: str) -> Any:
            return getattr(self.cur, name)

        def execute(self, query: Any, params: Any = None, **kwargs: Any) -> None:
            nonlocal seen
            self.cur.execute(query, params, **kwargs)
            if query is sql:
                seen += 1
                if seen == nth:
                    action()

    @contextmanager
    def hooked(**kwargs: Any):
        with cursor(**kwargs) as cur:
            yield Hooked(cur)

    saver._cursor = hooked


@pytest.mark.parametrize("durability", ["sync", "async", "exit"])
@pytest.mark.parametrize("make_checkpointer", SYNC)
def test_keep_latest_keeps_the_state_and_the_thread_runs_on_like_its_twin(
    make_checkpointer: Any, durability: str
) -> None:
    with make_checkpointer() as checkpointer:
        graph = _graph(checkpointer)
        for i in range(3):
            for thread_id in ("pruned", "twin"):
                graph.invoke(_say(i), _config(thread_id), durability=durability)
        before = graph.get_state(_config("pruned"))

        checkpointer.prune(["pruned"])

        assert _checkpoint_ids(checkpointer, "pruned") == [
            before.config["configurable"]["checkpoint_id"]
        ]
        assert graph.get_state(_config("pruned")) == _parentless(before)
        for thread_id in ("pruned", "twin"):
            graph.invoke(_say(3), _config(thread_id), durability=durability)
        assert (
            graph.get_state(_config("pruned")).values
            == graph.get_state(_config("twin")).values
        )


@pytest.mark.parametrize("make_checkpointer", SYNC)
def test_keep_latest_keeps_a_pending_interrupt_resumable(
    make_checkpointer: Any,
) -> None:
    with make_checkpointer() as checkpointer:
        graph = _graph(checkpointer, ask=True)
        for thread_id in ("pruned", "twin"):
            for i in range(2):
                graph.invoke(_say(i), _config(thread_id))
                graph.invoke(Command(resume=f"yes {i}"), _config(thread_id))
            graph.invoke(_say(2), _config(thread_id))
        before = graph.get_state(_config("pruned"))
        assert before.interrupts

        checkpointer.prune(["pruned"])

        assert graph.get_state(_config("pruned")) == _parentless(before)
        for thread_id in ("pruned", "twin"):
            graph.invoke(Command(resume="yes 2"), _config(thread_id))
        assert (
            graph.get_state(_config("pruned")).values
            == graph.get_state(_config("twin")).values
        )


@pytest.mark.parametrize("make_checkpointer", SYNC)
def test_keep_latest_keeps_writes_of_tasks_that_finished_before_a_failure(
    make_checkpointer: Any,
) -> None:
    with make_checkpointer() as checkpointer:
        calls: dict[str, int] = {}
        fail: set[str] = set()
        graph = _graph(checkpointer, fail=fail, calls=calls)
        for thread_id in ("pruned", "twin"):
            for i in range(2):
                graph.invoke(_say(i), _config(thread_id))
            fail.add("right")
            with pytest.raises(ValueError, match="right failed"):
                graph.invoke(_say(2), _config(thread_id))
        before = graph.get_state(_config("pruned"))

        checkpointer.prune(["pruned"])

        assert graph.get_state(_config("pruned")) == _parentless(before)
        left_runs = calls["left"]
        graph.invoke(None, _config("pruned"))
        assert calls["left"] == left_runs, "a task that had finished ran again"
        graph.invoke(None, _config("twin"))
        assert (
            graph.get_state(_config("pruned")).values
            == graph.get_state(_config("twin")).values
        )


@pytest.mark.parametrize("make_checkpointer", SYNC)
def test_keep_latest_keeps_an_interrupted_subgraph_resumable(
    make_checkpointer: Any,
) -> None:
    with make_checkpointer() as checkpointer:
        sub = StateGraph(State)
        sub.add_node("reply", _reply(ask=True))
        sub.add_edge(START, "reply")
        parent = StateGraph(State)
        parent.add_node("sub", sub.compile())
        parent.add_edge(START, "sub")
        graph = parent.compile(checkpointer=checkpointer)
        for thread_id in ("pruned", "twin"):
            for i in range(2):
                graph.invoke(_say(i), _config(thread_id))
                graph.invoke(Command(resume=f"yes {i}"), _config(thread_id))
            graph.invoke(_say(2), _config(thread_id))
        before = graph.get_state(_config("pruned"), subgraphs=True)
        namespaces = {
            c.config["configurable"]["checkpoint_ns"]
            for c in checkpointer.list(_config("pruned"))
        }

        checkpointer.prune(["pruned"])

        assert len(_checkpoint_ids(checkpointer, "pruned")) == len(namespaces)
        assert graph.get_state(_config("pruned"), subgraphs=True) == _parentless(before)
        for thread_id in ("pruned", "twin"):
            graph.invoke(Command(resume="yes 2"), _config(thread_id))
        assert (
            graph.get_state(_config("pruned")).values
            == graph.get_state(_config("twin")).values
        )


def test_keep_latest_keeps_a_turn_saved_after_it_read_the_heads() -> None:
    with (
        _checkpointer_postgres() as checkpointer,
        PostgresSaver.from_conn_string(
            DEFAULT_POSTGRES_URI + checkpointer.conn.info.dbname
        ) as pruner,
    ):
        graph = _graph(checkpointer)
        for i in range(3):
            for thread_id in ("pruned", "twin"):
                graph.invoke(_say(i), _config(thread_id))
        _run_once_after(
            pruner, PRUNE_HEADS_SQL, lambda: graph.invoke(_say(3), _config("pruned"))
        )

        pruner.prune(["pruned"])

        graph.invoke(_say(3), _config("twin"))
        assert (
            graph.get_state(_config("pruned")).values
            == graph.get_state(_config("twin")).values
        )


def test_keep_latest_refuses_a_thread_that_switches_to_a_delta_channel_mid_batch() -> (
    None
):
    with (
        _checkpointer_postgres() as checkpointer,
        PostgresSaver.from_conn_string(
            DEFAULT_POSTGRES_URI + checkpointer.conn.info.dbname
        ) as pruner,
    ):
        graph, delta_graph = _graph(checkpointer), _delta_graph(checkpointer)
        for i in range(3):
            for thread_id in ("switched", "twin"):
                graph.invoke(_say(i), _config(thread_id))
        _run_once_after(
            pruner,
            PRUNE_DELTA_THREADS_SQL,
            lambda: delta_graph.invoke(_say(3), _config("switched")),
        )

        with pytest.raises(NotImplementedError, match="switched"):
            pruner.prune(["switched"])

        delta_graph.invoke(_say(3), _config("twin"))
        assert (
            delta_graph.get_state(_config("switched")).values
            == delta_graph.get_state(_config("twin")).values
        )


def test_keep_latest_keeps_a_delta_channel_turn_saved_after_the_thread_check() -> None:
    with (
        _checkpointer_postgres() as checkpointer,
        PostgresSaver.from_conn_string(
            DEFAULT_POSTGRES_URI + checkpointer.conn.info.dbname
        ) as pruner,
    ):
        graph, delta_graph = _graph(checkpointer), _delta_graph(checkpointer)
        for i in range(3):
            for thread_id in ("switched", "twin"):
                graph.invoke(_say(i), _config(thread_id))
        head = _checkpoint_ids(checkpointer, "switched")[0]
        _run_once_after(
            pruner,
            PRUNE_DELTA_THREADS_SQL,
            lambda: delta_graph.invoke(_say(3), _config("switched")),
            nth=2,
        )

        pruner.prune(["switched"])

        assert _checkpoint_ids(checkpointer, "switched")[-1] == head
        delta_graph.invoke(_say(3), _config("twin"))
        assert (
            delta_graph.get_state(_config("switched")).values
            == delta_graph.get_state(_config("twin")).values
        )


@pytest.mark.parametrize("make_checkpointer", SYNC)
def test_keep_latest_refuses_a_batch_with_a_delta_channel_thread(
    make_checkpointer: Any,
) -> None:
    with make_checkpointer() as checkpointer:
        graph, delta_graph = _graph(checkpointer), _delta_graph(checkpointer)
        for i in range(3):
            graph.invoke(_say(i), _config("ordinary"))
            delta_graph.invoke(_say(i), _config("delta"))
        before = {t: _checkpoint_ids(checkpointer, t) for t in ("ordinary", "delta")}
        delta_before = delta_graph.get_state(_config("delta"))

        with pytest.raises(NotImplementedError, match="delta"):
            checkpointer.prune(["ordinary", "delta"])

        assert {t: _checkpoint_ids(checkpointer, t) for t in before} == before
        assert delta_graph.get_state(_config("delta")) == delta_before


@NEEDS_CONTEXTVARS
@pytest.mark.parametrize("make_checkpointer", ASYNC)
async def test_akeep_latest_keeps_a_pending_interrupt_resumable(
    make_checkpointer: Any,
) -> None:
    async with make_checkpointer() as checkpointer:
        graph = _graph(checkpointer, ask=True)
        for thread_id in ("pruned", "twin"):
            for i in range(2):
                await graph.ainvoke(_say(i), _config(thread_id))
                await graph.ainvoke(Command(resume=f"yes {i}"), _config(thread_id))
            await graph.ainvoke(_say(2), _config(thread_id))
        before = await graph.aget_state(_config("pruned"))

        await checkpointer.aprune(["pruned"])

        assert await _acheckpoint_ids(checkpointer, "pruned") == [
            before.config["configurable"]["checkpoint_id"]
        ]
        assert await graph.aget_state(_config("pruned")) == _parentless(before)
        for thread_id in ("pruned", "twin"):
            await graph.ainvoke(Command(resume="yes 2"), _config(thread_id))
        assert (await graph.aget_state(_config("pruned"))).values == (
            await graph.aget_state(_config("twin"))
        ).values


@pytest.mark.parametrize("make_checkpointer", ASYNC)
async def test_akeep_latest_refuses_a_batch_with_a_delta_channel_thread(
    make_checkpointer: Any,
) -> None:
    async with make_checkpointer() as checkpointer:
        graph, delta_graph = _graph(checkpointer), _delta_graph(checkpointer)
        for i in range(3):
            await graph.ainvoke(_say(i), _config("ordinary"))
            await delta_graph.ainvoke(_say(i), _config("delta"))
        before = {
            t: await _acheckpoint_ids(checkpointer, t) for t in ("ordinary", "delta")
        }

        with pytest.raises(NotImplementedError, match="delta"):
            await checkpointer.aprune(["ordinary", "delta"])

        assert {t: await _acheckpoint_ids(checkpointer, t) for t in before} == before
