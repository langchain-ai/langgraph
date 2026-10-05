"""State reads while some tasks of a superstep are finished and others are paused.

When parallel tasks each call `interrupt()` and only some of them are resumed,
the superstep stays open. Its recorded writes then contain the old interrupt of
each finished task next to that task's output. These tests check that state
reads, which are rebuilt from the checkpointer, report only the interrupts that
still need an answer.
"""

import operator
import sys
import uuid
from collections import Counter
from typing import Annotated, Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph._internal._constants import (
    ERROR,
    INTERRUPT,
    NO_WRITES,
    NULL_TASK_ID,
    RESUME,
    RETURN,
)
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.pregel._task_status import read_task_statuses
from langgraph.types import Command, Durability, Interrupt, Send, interrupt

pytestmark = pytest.mark.anyio

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


class State(TypedDict, total=False):
    log: Annotated[list[str], operator.add]
    count: int


def _config() -> dict[str, Any]:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _build_parallel(
    checkpointer: BaseCheckpointSaver,
    calls: Counter[str],
    *,
    a_questions: int = 1,
    a_returns: Any = "log",
):
    """Build a graph where nodes `a` and `b` start in parallel and both ask questions.

    `a` asks `a_questions` questions in a row. `a_returns` controls what `a`
    returns after its last answer. The default `"log"` returns the answers in
    `log`. Any other value is returned as-is.
    """

    def a(state: State) -> Any:
        calls["a"] += 1
        answers = [interrupt(f"A{i + 1}") for i in range(a_questions)]
        if a_returns == "log":
            return {"log": [f"a:{answer}" for answer in answers]}
        return a_returns

    def b(state: State) -> State:
        calls["b"] += 1
        return {"log": [f"b:{interrupt('B')}"]}

    builder = StateGraph(State)
    builder.add_node("a", a)
    builder.add_node("b", b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    return builder.compile(checkpointer=checkpointer)


def _interrupt_by_value(snapshot: Any, value: str) -> Interrupt:
    return next(i for i in snapshot.interrupts if i.value == value)


def _task(snapshot: Any, name: str) -> Any:
    return next(t for t in snapshot.tasks if t.name == name)


def _interrupt_values(interrupts: Any) -> list[str]:
    return sorted(i.value for i in interrupts)


# --- Task A answered and finished, task B still paused ---


def test_finished_task_does_not_report_answered_interrupt(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    calls: Counter[str] = Counter()
    graph = _build_parallel(sync_checkpointer, calls)
    config = _config()

    graph.invoke({"log": []}, config, durability=durability)
    snapshot = graph.get_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["A1", "B"]

    graph.invoke(
        Command(resume={_interrupt_by_value(snapshot, "A1").id: "yes"}),
        config,
        durability=durability,
    )

    snapshot = graph.get_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["B"]
    assert snapshot.next == ("b",)
    assert _task(snapshot, "a").interrupts == ()
    assert _task(snapshot, "a").result == {"log": ["a:yes"]}
    assert _interrupt_values(_task(snapshot, "b").interrupts) == ["B"]
    assert _task(snapshot, "b").result is None

    # Reading the same checkpoint by id gives the record of the step: every task
    # in it, and every question asked, including the one A already answered.
    record = graph.get_state(snapshot.config)
    assert sorted(record.next) == ["a", "b"]
    assert _interrupt_values(record.interrupts) == ["A1", "B"]
    assert _interrupt_values(_task(record, "a").interrupts) == ["A1"]
    assert _task(record, "a").result == {"log": ["a:yes"]}

    # B can still be answered, and the graph finishes normally.
    result = graph.invoke(
        Command(resume={_interrupt_by_value(snapshot, "B").id: "ok"}),
        config,
        durability=durability,
    )
    assert sorted(result["log"]) == ["a:yes", "b:ok"]
    assert calls == {"a": 2, "b": 3}
    snapshot = graph.get_state(config)
    assert snapshot.next == ()
    assert snapshot.interrupts == ()

    # History still shows where each question was asked.
    asked = [
        _interrupt_values(s.interrupts)
        for s in graph.get_state_history(config)
        if s.interrupts
    ]
    if durability != "exit":
        assert asked == [["A1", "B"]]


@NEEDS_CONTEXTVARS
async def test_finished_task_does_not_report_answered_interrupt_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    calls: Counter[str] = Counter()
    graph = _build_parallel(async_checkpointer, calls)
    config = _config()

    await graph.ainvoke({"log": []}, config)
    snapshot = await graph.aget_state(config)
    await graph.ainvoke(
        Command(resume={_interrupt_by_value(snapshot, "A1").id: "yes"}), config
    )

    snapshot = await graph.aget_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["B"]
    assert snapshot.next == ("b",)
    assert _task(snapshot, "a").interrupts == ()
    assert _task(snapshot, "a").result == {"log": ["a:yes"]}
    assert _interrupt_values(_task(snapshot, "b").interrupts) == ["B"]

    record = await graph.aget_state(snapshot.config)
    assert _interrupt_values(record.interrupts) == ["A1", "B"]
    assert _interrupt_values(_task(record, "a").interrupts) == ["A1"]

    result = await graph.ainvoke(
        Command(resume={_interrupt_by_value(snapshot, "B").id: "ok"}), config
    )
    assert sorted(result["log"]) == ["a:yes", "b:ok"]
    assert calls == {"a": 2, "b": 3}


# --- Task A answered its first question and asked a second one ---


def test_task_paused_at_second_question_stays_pending(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    calls: Counter[str] = Counter()
    graph = _build_parallel(sync_checkpointer, calls, a_questions=2)
    config = _config()

    graph.invoke({"log": []}, config)
    snapshot = graph.get_state(config)
    graph.invoke(
        Command(resume={_interrupt_by_value(snapshot, "A1").id: "one"}), config
    )

    snapshot = graph.get_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["A2", "B"]
    # A is not finished: it has a saved answer, but no output.
    assert sorted(snapshot.next) == ["a", "b"]
    assert _interrupt_values(_task(snapshot, "a").interrupts) == ["A2"]
    assert _task(snapshot, "a").result is None
    assert _interrupt_values(_task(snapshot, "b").interrupts) == ["B"]

    # Both remaining questions can be answered together.
    result = graph.invoke(
        Command(
            resume={
                _interrupt_by_value(snapshot, "A2").id: "two",
                _interrupt_by_value(snapshot, "B").id: "ok",
            }
        ),
        config,
    )
    assert sorted(result["log"]) == ["a:one", "a:two", "b:ok"]
    snapshot = graph.get_state(config)
    assert snapshot.next == ()
    assert snapshot.interrupts == ()


@NEEDS_CONTEXTVARS
async def test_task_paused_at_second_question_stays_pending_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    calls: Counter[str] = Counter()
    graph = _build_parallel(async_checkpointer, calls, a_questions=2)
    config = _config()

    await graph.ainvoke({"log": []}, config)
    snapshot = await graph.aget_state(config)
    await graph.ainvoke(
        Command(resume={_interrupt_by_value(snapshot, "A1").id: "one"}), config
    )

    snapshot = await graph.aget_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["A2", "B"]
    assert sorted(snapshot.next) == ["a", "b"]
    assert _interrupt_values(_task(snapshot, "a").interrupts) == ["A2"]
    assert _task(snapshot, "a").result is None


def test_task_paused_at_second_question_then_other_task_finishes(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    calls: Counter[str] = Counter()
    graph = _build_parallel(sync_checkpointer, calls, a_questions=2)
    config = _config()

    graph.invoke({"log": []}, config)
    snapshot = graph.get_state(config)
    graph.invoke(
        Command(resume={_interrupt_by_value(snapshot, "A1").id: "one"}), config
    )
    snapshot = graph.get_state(config)
    graph.invoke(Command(resume={_interrupt_by_value(snapshot, "B").id: "ok"}), config)

    snapshot = graph.get_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["A2"]
    assert snapshot.next == ("a",)
    assert _task(snapshot, "b").interrupts == ()
    assert _task(snapshot, "b").result == {"log": ["b:ok"]}

    result = graph.invoke(
        Command(resume={_interrupt_by_value(snapshot, "A2").id: "two"}), config
    )
    assert sorted(result["log"]) == ["a:one", "a:two", "b:ok"]


def test_resume_without_id_rejected_when_second_question_and_other_task_pending(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    calls: Counter[str] = Counter()
    graph = _build_parallel(sync_checkpointer, calls, a_questions=2)
    config = _config()

    graph.invoke({"log": []}, config)
    snapshot = graph.get_state(config)
    graph.invoke(
        Command(resume={_interrupt_by_value(snapshot, "A1").id: "one"}), config
    )

    # A2 and B are both waiting, so a resume value without an id is ambiguous.
    with pytest.raises(RuntimeError, match="multiple pending interrupts"):
        graph.invoke(Command(resume="ambiguous"), config)


def test_resume_without_id_rejected_when_subgraph_has_parallel_interrupts(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    # A subgraph node whose child graph pauses in two parallel nodes records
    # both interrupts under one parent task. Both count as pending, so a resume
    # value without an id is ambiguous. (Before, only the first was counted and
    # the value went to whichever interrupt consumed it first.)
    child_builder = StateGraph(State)
    child_builder.add_node("a", lambda s: {"log": [f"a:{interrupt('A')}"]})
    child_builder.add_node("b", lambda s: {"log": [f"b:{interrupt('B')}"]})
    child_builder.add_edge(START, "a")
    child_builder.add_edge(START, "b")

    builder = StateGraph(State)
    builder.add_node("child", child_builder.compile())
    builder.add_edge(START, "child")
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = _config()

    graph.invoke({"log": []}, config)
    snapshot = graph.get_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["A", "B"]

    with pytest.raises(RuntimeError, match="multiple pending interrupts"):
        graph.invoke(Command(resume="ambiguous"), config)

    result = graph.invoke(
        Command(
            resume={
                _interrupt_by_value(snapshot, "A").id: "x",
                _interrupt_by_value(snapshot, "B").id: "y",
            }
        ),
        config,
    )
    assert sorted(result["log"]) == ["a:x", "b:y"]


# --- Task A finished with an empty or falsy result ---


@pytest.mark.parametrize(
    "a_returns",
    [None, {}, {"count": 0}, {"log": []}],
    ids=["none", "empty_dict", "zero", "empty_list"],
)
def test_task_finished_with_falsy_result(
    sync_checkpointer: BaseCheckpointSaver, a_returns: Any
) -> None:
    calls: Counter[str] = Counter()
    graph = _build_parallel(sync_checkpointer, calls, a_returns=a_returns)
    config = _config()

    graph.invoke({"log": []}, config)
    snapshot = graph.get_state(config)
    graph.invoke(
        Command(resume={_interrupt_by_value(snapshot, "A1").id: "yes"}), config
    )

    snapshot = graph.get_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["B"]
    assert snapshot.next == ("b",)
    assert _task(snapshot, "a").interrupts == ()

    graph.invoke(Command(resume={_interrupt_by_value(snapshot, "B").id: "ok"}), config)
    # A already finished, so resuming B must not run A again.
    assert calls == {"a": 2, "b": 3}
    snapshot = graph.get_state(config)
    assert snapshot.next == ()
    assert snapshot.interrupts == ()


@pytest.mark.parametrize("a_returns", [None, {"count": 0}], ids=["none", "zero"])
@NEEDS_CONTEXTVARS
async def test_task_finished_with_falsy_result_async(
    async_checkpointer: BaseCheckpointSaver, a_returns: Any
) -> None:
    calls: Counter[str] = Counter()
    graph = _build_parallel(async_checkpointer, calls, a_returns=a_returns)
    config = _config()

    await graph.ainvoke({"log": []}, config)
    snapshot = await graph.aget_state(config)
    await graph.ainvoke(
        Command(resume={_interrupt_by_value(snapshot, "A1").id: "yes"}), config
    )

    snapshot = await graph.aget_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["B"]
    assert snapshot.next == ("b",)
    assert _task(snapshot, "a").interrupts == ()

    await graph.ainvoke(
        Command(resume={_interrupt_by_value(snapshot, "B").id: "ok"}), config
    )
    assert calls == {"a": 2, "b": 3}


# --- Subgraphs and the functional API ---


def test_parallel_subgraphs_report_only_pending_interrupts(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    class ChildState(TypedDict):
        prompt: str
        answers: Annotated[list[str], operator.add]

    def ask(state: ChildState) -> dict[str, Any]:
        return {"answers": [interrupt(state["prompt"])]}

    child_builder = StateGraph(ChildState)
    child_builder.add_node("ask", ask)
    child_builder.add_edge(START, "ask")
    child = child_builder.compile()

    class ParentState(TypedDict):
        answers: Annotated[list[str], operator.add]

    builder = StateGraph(ParentState)
    builder.add_node("child", child)
    builder.add_conditional_edges(
        START,
        lambda _: [Send("child", {"prompt": p, "answers": []}) for p in ("a", "b")],
        ["child"],
    )
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = _config()

    graph.invoke({"answers": []}, config)
    snapshot = graph.get_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["a", "b"]
    graph.invoke(Command(resume={_interrupt_by_value(snapshot, "a").id: "x"}), config)

    snapshot = graph.get_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["b"]
    assert snapshot.next == ("child",)
    finished = next(t for t in snapshot.tasks if t.result is not None)
    assert finished.interrupts == ()
    assert finished.result == {"answers": ["x"]}

    result = graph.invoke(
        Command(resume={_interrupt_by_value(snapshot, "b").id: "y"}), config
    )
    assert sorted(result["answers"]) == ["x", "y"]


def test_functional_task_finished_with_none_is_not_rerun(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    calls: Counter[str] = Counter()

    @task
    def ask_a() -> None:
        calls["a"] += 1
        interrupt("A")

    @task
    def ask_b() -> str:
        calls["b"] += 1
        return interrupt("B")

    @entrypoint(checkpointer=sync_checkpointer)
    def workflow(_: Any) -> list[Any]:
        a, b = ask_a(), ask_b()
        return [a.result(), b.result()]

    config = _config()
    workflow.invoke(1, config)
    snapshot = workflow.get_state(config)
    workflow.invoke(
        Command(resume={_interrupt_by_value(snapshot, "A").id: "x"}), config
    )

    snapshot = workflow.get_state(config)
    assert _interrupt_values(snapshot.interrupts) == ["B"]

    result = workflow.invoke(
        Command(resume={_interrupt_by_value(snapshot, "B").id: "y"}), config
    )
    assert result == [None, "y"]
    assert calls == {"a": 2, "b": 3}


# --- Reading task status from recorded writes ---


def test_read_task_statuses() -> None:
    a1 = Interrupt(value="A1", id="a")
    a2 = Interrupt(value="A2", id="a")
    b = Interrupt(value="B", id="b")
    error = ValueError("boom")

    statuses = read_task_statuses(
        [
            # answered and finished: old interrupt stays recorded
            ("finished", INTERRUPT, (a1,)),
            ("finished", RESUME, ["yes"]),
            ("finished", "log", ["a:yes"]),
            # answered once, then paused at a second question
            ("paused", INTERRUPT, (a2,)),
            ("paused", RESUME, ["one"]),
            # finished with no output
            ("no_output", INTERRUPT, (b,)),
            ("no_output", RESUME, ["ok"]),
            ("no_output", NO_WRITES, None),
            # functional task that returned None
            ("returned_none", RETURN, None),
            # failed
            ("failed", ERROR, error),
            # not a task
            (NULL_TASK_ID, RESUME, "global"),
        ]
    )

    assert set(statuses) == {
        "finished",
        "paused",
        "no_output",
        "returned_none",
        "failed",
    }

    assert statuses["finished"].finished
    assert statuses["finished"].interrupts == (a1,)
    assert statuses["finished"].pending_interrupts == ()
    assert statuses["finished"].output == (("log", ["a:yes"]),)

    assert not statuses["paused"].finished
    assert statuses["paused"].interrupts == (a2,)
    assert statuses["paused"].pending_interrupts == (a2,)
    assert statuses["paused"].output == ()

    assert statuses["no_output"].finished
    assert statuses["no_output"].interrupts == (b,)
    assert statuses["no_output"].pending_interrupts == ()

    assert statuses["returned_none"].finished
    assert statuses["returned_none"].output == ((RETURN, None),)

    assert not statuses["failed"].finished
    assert statuses["failed"].error is error
