import operator
import sys
import uuid
from typing import Annotated

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict
from xxhash import xxh3_128_hexdigest

from langgraph._internal._constants import CONFIG_KEY_STRICT_RESUME, INTERRUPT
from langgraph.errors import UnknownInterruptError
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Durability, Interrupt, RetryPolicy, interrupt

pytestmark = pytest.mark.anyio
NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)

UNKNOWN_ID = "0" * 32


class State(TypedDict, total=False):
    log: Annotated[list[str], operator.add]


def _config(*, strict: bool = True) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            CONFIG_KEY_STRICT_RESUME: strict,
        }
    }


def _parallel(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    def a(state: State):
        return {"log": [f"a:{interrupt('a?')}"]}

    def b(state: State):
        return {"log": [f"b:{interrupt('b?')}"]}

    return (
        StateGraph(State)
        .add_node("a", a)
        .add_node("b", b)
        .add_edge(START, "a")
        .add_edge(START, "b")
        .compile(checkpointer=checkpointer)
    )


def _ask_twice(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    def ask_twice(state: State):
        first = interrupt("first?")
        second = interrupt("second?")
        return {"log": [f"{first},{second}"]}

    return (
        StateGraph(State)
        .add_node("ask", ask_twice)
        .add_edge(START, "ask")
        .compile(checkpointer=checkpointer)
    )


def _ask_then_fail_once(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    failures = [ConnectionError("flaky api")]

    def ask(state: State):
        answer = interrupt("delete a.txt?")
        if failures:
            raise failures.pop()
        return {"log": [answer]}

    return (
        StateGraph(State)
        .add_node(
            "ask", ask, retry_policy=RetryPolicy(initial_interval=0, jitter=False)
        )
        .add_edge(START, "ask")
        .compile(checkpointer=checkpointer)
    )


def _pending_ids(graph: CompiledStateGraph, config: RunnableConfig) -> dict[str, str]:
    return {i.value: i.id for i in graph.get_state(config).interrupts}


def test_unknown_id_is_refused_before_anything_is_written(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph = _parallel(sync_checkpointer)
    config = _config()
    graph.invoke({"log": []}, config, durability=durability)
    before = graph.get_state(config)
    checkpoints = len(list(sync_checkpointer.list(config)))
    ids = _pending_ids(graph, config)

    with pytest.raises(
        UnknownInterruptError,
        match=f"Unknown or already-consumed interrupt: {UNKNOWN_ID}",
    ) as refused:
        graph.invoke(Command(resume={UNKNOWN_ID: "x"}), config, durability=durability)

    assert (
        refused.value.reason,
        refused.value.interrupt_ids,
        refused.value.pending_ids,
    ) == ("not_pending", (UNKNOWN_ID,), tuple(sorted(ids.values())))
    after = graph.get_state(config)
    assert len(list(sync_checkpointer.list(config))) == checkpoints
    assert (after.values, after.next, after.interrupts) == (
        before.values,
        before.next,
        before.interrupts,
    )
    result = graph.invoke(
        Command(resume={ids["a?"]: "yes", ids["b?"]: "no"}),
        config,
        durability=durability,
    )
    assert sorted(result["log"]) == ["a:yes", "b:no"]


def test_an_answered_id_is_refused(sync_checkpointer: BaseCheckpointSaver) -> None:
    graph = _parallel(sync_checkpointer)
    config = _config()
    graph.invoke({"log": []}, config)
    ids = _pending_ids(graph, config)
    graph.invoke(Command(resume={ids["a?"]: "yes"}), config)

    with pytest.raises(UnknownInterruptError, match=ids["a?"]) as refused:
        graph.invoke(Command(resume={ids["a?"]: "again"}), config)

    assert (
        refused.value.reason,
        refused.value.interrupt_ids,
        refused.value.pending_ids,
    ) == ("not_pending", (ids["a?"],), (ids["b?"],))

    result = graph.invoke(Command(resume={ids["b?"]: "no"}), config)
    assert sorted(result["log"]) == ["a:yes", "b:no"]


def test_an_empty_map_is_refused(sync_checkpointer: BaseCheckpointSaver) -> None:
    graph = _parallel(sync_checkpointer)
    config = _config()
    graph.invoke({"log": []}, config)

    with pytest.raises(UnknownInterruptError, match="names no interrupt") as refused:
        graph.invoke(Command(resume={}), config)

    assert (
        refused.value.reason,
        refused.value.interrupt_ids,
        refused.value.pending_ids,
    ) == ("empty_map", (), tuple(sorted(_pending_ids(graph, config).values())))


def test_one_value_is_refused_while_several_are_pending(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _parallel(sync_checkpointer)
    config = _config()
    graph.invoke({"log": []}, config)

    with pytest.raises(
        UnknownInterruptError, match="multiple pending interrupts"
    ) as refused:
        graph.invoke(Command(resume="yes"), config)

    ids = _pending_ids(graph, config)
    assert (refused.value.reason, refused.value.pending_ids) == (
        "ambiguous",
        tuple(sorted(ids.values())),
    )
    graph.invoke(Command(resume={ids["a?"]: "yes"}), config)
    result = graph.invoke(Command(resume="no"), config)
    assert sorted(result["log"]) == ["a:yes", "b:no"]


def test_one_value_resumes_a_static_breakpoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = (
        StateGraph(State)
        .add_node("a", lambda state: {"log": ["a"]})
        .add_edge(START, "a")
        .compile(checkpointer=sync_checkpointer, interrupt_before=["a"])
    )
    config = _config()
    graph.invoke({"log": []}, config)

    with pytest.raises(UnknownInterruptError, match=UNKNOWN_ID):
        graph.invoke(Command(resume={UNKNOWN_ID: "x"}), config)

    assert graph.invoke(Command(resume="go"), config) == {"log": ["a"]}


def test_without_strict_mode_nothing_changes(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _parallel(sync_checkpointer)
    config = _config(strict=False)
    graph.invoke({"log": []}, config)

    graph.invoke(Command(resume={UNKNOWN_ID: "x"}), config)
    assert set(_pending_ids(graph, config)) == {"a?", "b?"}
    with pytest.raises(RuntimeError, match="multiple pending interrupts"):
        graph.invoke(Command(resume="yes"), config)


def test_a_subgraph_interrupt_is_answered_by_its_id(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    def ask(state: State):
        return {"log": [f"asked:{interrupt('sub?')}"]}

    subgraph = StateGraph(State).add_node("ask", ask).add_edge(START, "ask").compile()
    graph = (
        StateGraph(State)
        .add_node("sub", subgraph)
        .add_edge(START, "sub")
        .compile(checkpointer=sync_checkpointer)
    )
    config = _config()
    graph.invoke({"log": []}, config)

    result = graph.invoke(
        Command(resume={_pending_ids(graph, config)["sub?"]: "yes"}), config
    )
    assert result["log"] == ["asked:yes"]


def test_a_second_interrupt_in_one_node_gets_its_own_id(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _ask_twice(sync_checkpointer)
    config = _config()
    graph.invoke({"log": []}, config)
    state = graph.get_state(config)
    (card,) = state.interrupts
    assert card.id == xxh3_128_hexdigest(f"ask:{state.tasks[0].id}".encode())
    graph.invoke(Command(resume={card.id: "1"}), config)
    (second,) = graph.get_state(config).interrupts
    assert second.id != card.id

    result = graph.invoke(Command(resume={second.id: "2"}), config)
    assert result["log"] == ["1,2"]


def test_a_retried_answer_in_one_node_is_refused(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _ask_twice(sync_checkpointer)
    config = _config()
    graph.invoke({"log": []}, config)
    (card,) = graph.get_state(config).interrupts
    graph.invoke(Command(resume={card.id: "yes"}), config)
    (second,) = graph.get_state(config).interrupts

    with pytest.raises(UnknownInterruptError) as refused:
        graph.invoke(Command(resume={card.id: "yes"}), config)

    assert (
        refused.value.reason,
        refused.value.interrupt_ids,
        refused.value.pending_ids,
    ) == ("not_pending", (card.id,), (second.id,))
    result = graph.invoke(Command(resume={second.id: "no"}), config)
    assert result["log"] == ["yes,no"]


def test_parallel_interrupts_in_one_node_each_take_an_answer(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    def ask(name: str) -> str:
        return interrupt(f"{name}?")

    def ask_in_parallel(state: State, config: RunnableConfig):
        with get_executor_for_config(config) as pool:
            b = pool.submit(ask, "b")
            b.exception()
            a = pool.submit(ask, "a")
            return {"log": [f"{a.result()},{b.result()}"]}

    graph = (
        StateGraph(State)
        .add_node("ask", ask_in_parallel)
        .add_edge(START, "ask")
        .compile(checkpointer=sync_checkpointer)
    )
    config = _config()
    graph.invoke({"log": []}, config)
    for answer in ("1", "2"):
        (card,) = graph.get_state(config).interrupts
        graph.invoke(Command(resume={card.id: answer}), config)

    assert graph.get_state(config).next == ()


def test_a_later_interrupt_saved_with_the_first_id_is_still_answered(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _ask_twice(sync_checkpointer)
    config = _config()
    graph.invoke({"log": []}, config)
    (card,) = graph.get_state(config).interrupts
    graph.invoke(Command(resume={card.id: "1"}), config)
    state = graph.get_state(config)
    sync_checkpointer.put_writes(
        state.config,
        [(INTERRUPT, (Interrupt(value="second?", id=card.id),))],
        state.tasks[0].id,
    )

    result = graph.invoke(Command(resume={card.id: "2"}), config)
    assert result["log"] == ["1,2"]


def test_a_retried_node_reuses_its_answer(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _ask_then_fail_once(sync_checkpointer)
    config = _config()
    graph.invoke({"log": []}, config)
    (card,) = graph.get_state(config).interrupts

    result = graph.invoke(Command(resume={card.id: "yes"}), config)
    assert result["log"] == ["yes"]


def test_a_card_whose_resumed_task_raised_is_not_refused(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    failures = ["once"]

    def ask_then_fail_once(state: State):
        answer = interrupt("approve?")
        if failures:
            failures.pop()
            raise ValueError("failed after the answer")
        return {"log": [f"answered:{answer}"]}

    graph = (
        StateGraph(State)
        .add_node("ask", ask_then_fail_once)
        .add_edge(START, "ask")
        .compile(checkpointer=sync_checkpointer)
    )
    config = _config()
    graph.invoke({"log": []}, config)
    (card,) = graph.get_state(config).interrupts
    with pytest.raises(ValueError, match="failed after the answer"):
        graph.invoke(Command(resume={card.id: "first"}), config)

    graph.invoke(Command(resume={card.id: "second"}), config)
    assert graph.get_state(config).next == ()


def test_a_pinned_head_checkpoint_is_checked(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _parallel(sync_checkpointer)
    config = _config()
    graph.invoke({"log": []}, config)
    head = graph.get_state(config).config
    pinned = {"configurable": {**config["configurable"], **head["configurable"]}}

    with pytest.raises(UnknownInterruptError):
        graph.invoke(Command(resume={UNKNOWN_ID: "x"}), pinned)

    ids = _pending_ids(graph, config)
    result = graph.invoke(Command(resume={ids["a?"]: "yes", ids["b?"]: "no"}), pinned)
    assert sorted(result["log"]) == ["a:yes", "b:no"]


def test_a_superseded_checkpoint_is_not_checked(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _parallel(sync_checkpointer)
    config = _config()
    graph.invoke({"log": []}, config)
    paused = graph.get_state(config).config
    ids = _pending_ids(graph, config)
    graph.invoke(Command(resume={ids["a?"]: "yes", ids["b?"]: "no"}), config)

    pinned = {"configurable": {**config["configurable"], **paused["configurable"]}}
    graph.invoke(Command(resume={ids["a?"]: "again", ids["b?"]: "again"}), pinned)


@NEEDS_CONTEXTVARS
async def test_unknown_id_is_refused_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _parallel(async_checkpointer)
    config = _config()
    await graph.ainvoke({"log": []}, config)

    with pytest.raises(UnknownInterruptError, match=UNKNOWN_ID):
        await graph.ainvoke(Command(resume={UNKNOWN_ID: "x"}), config)

    state = await graph.aget_state(config)
    assert {i.value for i in state.interrupts} == {"a?", "b?"}


@NEEDS_CONTEXTVARS
async def test_a_retried_node_reuses_its_answer_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    graph = _ask_then_fail_once(async_checkpointer)
    config = _config()
    await graph.ainvoke({"log": []}, config)
    (card,) = (await graph.aget_state(config)).interrupts

    result = await graph.ainvoke(Command(resume={card.id: "yes"}), config)
    assert result["log"] == ["yes"]
