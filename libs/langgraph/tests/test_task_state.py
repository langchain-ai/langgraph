import operator
import sys
from collections.abc import Callable
from typing import Annotated, Any

import pytest
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.config import read_state, write_state
from langgraph.errors import InvalidUpdateError, NotInTaskError
from langgraph.func import entrypoint
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Overwrite, interrupt

pytestmark = pytest.mark.anyio

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


class State(TypedDict, total=False):
    one: str
    many: Annotated[list[int], operator.add]
    ephemeral: Annotated[str, EphemeralValue(str)]
    untracked: Annotated[str, UntrackedValue(str)]


def run(node: Callable[..., Any], input: State | None = None) -> dict[str, Any]:
    graph = StateGraph(State).add_node("node", node).add_edge(START, "node").compile()
    return graph.invoke(input or {"many": [0]})


def test_read_state_applies_the_nodes_own_updates() -> None:
    seen: dict[str, Any] = {}

    def node(state: State) -> None:
        seen["start"] = read_state()
        write_state({"one": "a", "many": [1]})
        seen["all"] = read_state()
        seen["one"] = read_state("one")
        seen["both"] = read_state("one", "many")
        seen["input"] = dict(state)

    assert run(node) == {"one": "a", "many": [0, 1]}
    assert seen == {
        "start": {"many": [0]},
        "all": {"one": "a", "many": [0, 1]},
        "one": {"one": "a"},
        "both": {"one": "a", "many": [0, 1]},
        "input": {"many": [0]},
    }


def test_write_state_adds_to_the_update_the_node_returns() -> None:
    def node(state: State) -> State:
        write_state({"many": [1]})
        write_state({"many": [2]})
        return {"many": [3]}

    assert run(node) == {"many": [0, 1, 2, 3]}


@NEEDS_CONTEXTVARS
async def test_an_async_node_finds_its_task_from_context() -> None:
    async def node(state: State) -> None:
        write_state({"one": "a"})
        assert read_state("one") == {"one": "a"}

    graph = StateGraph(State).add_node("node", node).add_edge(START, "node").compile()
    assert await graph.ainvoke({"many": [0]}) == {"one": "a", "many": [0]}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"one": "a"}, {"one": "b"}),
        ({"many": Overwrite([1])}, {"many": Overwrite([2])}),
        ({"ephemeral": "a"}, {"ephemeral": "b"}),
        ({"untracked": "a"}, {"untracked": "b"}),
    ],
    ids=["last-value", "overwrite", "ephemeral", "untracked"],
)
def test_write_state_rejects_a_second_value_for_a_one_value_key(
    first: dict[str, Any], second: dict[str, Any]
) -> None:
    after: dict[str, Any] = {}

    def node(state: State) -> None:
        write_state(first)
        before = read_state()
        with pytest.raises(InvalidUpdateError):
            write_state(second)
        after.update(unchanged=read_state() == before)

    assert run(node) == run(lambda state: first)
    assert after == {"unchanged": True}


def test_a_rejected_update_queues_none_of_its_keys() -> None:
    def node(state: State) -> None:
        write_state({"one": "a"})
        with pytest.raises(InvalidUpdateError):
            write_state({"many": [1], "one": "b"})

    assert run(node) == {"one": "a", "many": [0]}


@pytest.mark.parametrize(
    "key",
    ["missing", "__interrupt__", "__pregel_tasks", "__start__", "branch:to:other"],
)
def test_write_state_rejects_a_key_that_is_not_state(key: str) -> None:
    ran: list[str] = []

    def node(state: State) -> None:
        with pytest.raises(InvalidUpdateError, match="not a state key"):
            write_state({"many": [1], key: None})

    graph = (
        StateGraph(State)
        .add_node("node", node)
        .add_node("other", lambda state: ran.append("other"))
        .add_edge(START, "node")
        .add_edge("other", END)
        .compile()
    )
    assert graph.invoke({"many": [0]}) == {"many": [0]}
    assert ran == []


def test_read_state_returns_state_keys_only() -> None:
    seen: list[dict[str, Any]] = []

    graph = (
        StateGraph(State)
        .add_node("a", lambda state: {"many": [1]})
        .add_node("b", lambda state: {"many": [2]})
        .add_node("joined", lambda state: seen.append(read_state()))
        .add_edge(START, "a")
        .add_edge(START, "b")
        .add_edge(["a", "b"], "joined")
        .compile()
    )
    graph.invoke({"many": [0]})
    assert [sorted(state) for state in seen] == [["many"]]


def test_outside_a_task() -> None:
    with pytest.raises(NotInTaskError, match="read_state outside"):
        read_state()
    with pytest.raises(NotInTaskError, match="write_state outside"):
        write_state({"one": "a"})
    with pytest.raises(NotInTaskError):
        RunnableLambda(lambda _: read_state()).invoke(None)
    with pytest.raises(NotInTaskError):
        RunnableLambda(lambda _: write_state({"one": "a"})).invoke(None)


def test_an_entrypoint_has_no_state_keys() -> None:
    @entrypoint()
    def workflow(value: int) -> dict[str, Any]:
        with pytest.raises(InvalidUpdateError, match="not a state key"):
            write_state({END: value})
        return read_state()

    assert workflow.invoke(1) == {}


def test_updates_written_before_an_interrupt_apply_once_after_resume() -> None:
    def node(state: State) -> State:
        write_state({"many": [1]})
        answer = interrupt("ok?")
        write_state({"one": answer})
        return {"many": [2]}

    graph = (
        StateGraph(State)
        .add_node("node", node)
        .add_edge(START, "node")
        .compile(checkpointer=InMemorySaver())
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    graph.invoke({"many": [0]}, config)
    assert graph.get_state(config).values == {"many": [0]}
    assert graph.invoke(Command(resume="yes"), config) == {
        "one": "yes",
        "many": [0, 1, 2],
    }


def test_read_state_in_a_conditional_edge_during_update_state() -> None:
    seen: list[dict[str, Any]] = []

    def route(state: State) -> str:
        seen.append(read_state("one"))
        return "end"

    graph = (
        StateGraph(State)
        .add_node("node", lambda state: None)
        .add_edge(START, "node")
        .add_conditional_edges("node", route, {"end": "__end__"})
        .compile(checkpointer=InMemorySaver())
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    graph.invoke({"one": "a"}, config)
    graph.update_state(config, {"one": "b"}, as_node="node")
    assert seen == [{"one": "a"}, {"one": "b"}]
