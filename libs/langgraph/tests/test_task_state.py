import contextvars
import operator
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Annotated, Any

import pytest
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.config import read_state, write_state
from langgraph.errors import InvalidUpdateError, NotInTaskError
from langgraph.graph import START, StateGraph
from langgraph.types import Command, Overwrite, interrupt

pytestmark = pytest.mark.anyio


class State(TypedDict, total=False):
    one: str
    many: Annotated[list[int], operator.add]
    ephemeral: Annotated[str, EphemeralValue(str)]
    untracked: Annotated[str, UntrackedValue(str)]


def run(node: Callable[..., Any], input: State | None = None) -> dict[str, Any]:
    graph = StateGraph(State).add_node("node", node).add_edge(START, "node").compile()
    return graph.invoke(input or {"many": [0]})


def test_state_declares_the_one_value_channels_under_test() -> None:
    graph = StateGraph(State).add_node("node", lambda state: None)
    channels = graph.add_edge(START, "node").compile().channels
    assert isinstance(channels["ephemeral"], EphemeralValue)
    assert isinstance(channels["untracked"], UntrackedValue)
    assert channels["ephemeral"].guard
    assert channels["untracked"].guard


def test_read_state_applies_the_tasks_own_updates() -> None:
    seen: dict[str, Any] = {}

    def node(state: State) -> None:
        seen["start"] = read_state()
        write_state({"one": "a", "many": [1]})
        seen["all"] = read_state()
        seen["one"] = read_state("one")
        seen["list"] = read_state(["many"])
        seen["input"] = dict(state)

    assert run(node) == {"one": "a", "many": [0, 1]}
    assert seen == {
        "start": {"many": [0]},
        "all": {"one": "a", "many": [0, 1]},
        "one": {"one": "a"},
        "list": {"many": [0, 1]},
        "input": {"many": [0]},
    }


def test_read_state_rejects_a_key_that_is_not_state() -> None:
    def node(state: State) -> None:
        read_state("missing")

    with pytest.raises(KeyError, match="missing"):
        run(node)


def test_write_state_adds_to_the_update_the_node_returns() -> None:
    def node(state: State) -> State:
        write_state([("many", [1]), ("many", [2])])
        return {"many": [3]}

    assert run(node) == {"many": [0, 1, 2, 3]}


async def test_write_state_in_an_async_node() -> None:
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
        ({"many": [1]}, [("one", "a"), ("one", "b")]),
    ],
    ids=["last-value", "overwrite", "ephemeral", "untracked", "twice-in-one-update"],
)
def test_write_state_rejects_a_second_value_for_a_one_value_key(
    first: dict[str, Any], second: Any
) -> None:
    after: dict[str, Any] = {}

    def node(state: State) -> None:
        write_state(first)
        before = read_state()
        with pytest.raises(InvalidUpdateError):
            write_state(second)
        after.update(unchanged=read_state() == before)

    expected = run(lambda state: first)
    assert run(node) == expected
    assert after == {"unchanged": True}


def test_rejected_update_queues_none_of_its_keys() -> None:
    def node(state: State) -> None:
        write_state({"one": "a"})
        with pytest.raises(InvalidUpdateError):
            write_state({"many": [1], "one": "b"})

    assert run(node) == {"one": "a", "many": [0]}


@pytest.mark.parametrize(
    "update",
    [
        {"missing": 1},
        {"many": [1], "__interrupt__": "x"},
        [("many", [1]), ("__pregel_tasks", "x")],
        ["many"],
        [("many", [1], "extra")],
    ],
    ids=["unknown", "reserved-write", "tasks-channel", "not-tuples", "long-tuple"],
)
def test_write_state_rejects_keys_that_are_not_state(update: Any) -> None:
    def node(state: State) -> None:
        with pytest.raises(InvalidUpdateError):
            write_state(update)

    assert run(node) == {"many": [0]}


def test_outside_a_task() -> None:
    with pytest.raises(NotInTaskError, match="read_state outside"):
        read_state()
    with pytest.raises(NotInTaskError, match="write_state outside"):
        write_state({"one": "a"})
    with pytest.raises(NotInTaskError):
        RunnableLambda(lambda _: read_state()).invoke(None)
    with pytest.raises(NotInTaskError):
        RunnableLambda(lambda _: write_state({"one": "a"})).invoke(None)


def in_thread(fn: Callable[[], Any]) -> Any:
    """Run `fn` on a thread that starts with an empty context."""
    out: list[Any] = []

    def target() -> None:
        try:
            out.append(fn())
        except Exception as exc:
            out.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    return out[0]


@pytest.mark.skipif(
    getattr(sys.flags, "thread_inherit_context", 0) == 1,
    reason="threads inherit the caller's context on this build",
)
def test_a_thread_reaches_the_task_through_a_copied_context() -> None:
    def node(state: State) -> None:
        assert isinstance(in_thread(read_state), NotInTaskError)
        context = contextvars.copy_context()
        in_thread(lambda: context.run(write_state, {"one": "a"}))
        assert in_thread(lambda: context.run(read_state, "one")) == {"one": "a"}

    assert run(node) == {"one": "a", "many": [0]}


@pytest.fixture
def eager_thread_switches() -> Iterator[None]:
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(interval)


def task_thread(fn: Callable[..., None], *args: Any) -> threading.Thread:
    """A thread that runs `fn` in a copy of the calling task's context."""
    return threading.Thread(target=contextvars.copy_context().run, args=(fn, *args))


def start_all(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


@pytest.mark.usefixtures("eager_thread_switches")
def test_concurrent_writers_to_a_one_value_key_have_one_winner() -> None:
    writers = 16
    outcomes: list[str] = []

    def node(state: State) -> None:
        barrier = threading.Barrier(writers)

        def write(i: int) -> None:
            barrier.wait()
            try:
                write_state({"one": str(i), "many": [i]})
            except InvalidUpdateError:
                outcomes.append("rejected")
            else:
                outcomes.append(str(i))

        start_all([task_thread(write, i) for i in range(writers)])

    for _ in range(50):
        outcomes.clear()
        result = run(node, {"many": []})
        winners = [o for o in outcomes if o != "rejected"]
        assert len(winners) == 1
        assert result == {"one": winners[0], "many": [int(winners[0])]}


@pytest.mark.usefixtures("eager_thread_switches")
def test_concurrent_readers_and_writers_lose_no_update() -> None:
    writers, sends = 8, 50
    errors: list[Exception] = []

    def node(state: State) -> None:
        def work(i: int) -> None:
            try:
                for _ in range(sends):
                    write_state({"many": [i]})
                    read_state("many")
            except Exception as exc:
                errors.append(exc)

        start_all([task_thread(work, i) for i in range(writers)])

    result = run(node, {"many": []})
    assert errors == []
    assert sorted(result["many"]) == sorted(list(range(writers)) * sends)


def test_a_writer_waits_for_another_threads_check_to_finish() -> None:
    checking, release = threading.Event(), threading.Event()

    def slow_add(left: list[int], right: list[int]) -> list[int]:
        if right == [1]:
            checking.set()
            release.wait(5)
        return left + right

    class SlowState(TypedDict, total=False):
        one: str
        slow: Annotated[list[int], slow_add]

    outcomes: dict[str, str] = {}

    def node(state: SlowState) -> None:
        def write(name: str, update: dict[str, Any]) -> None:
            try:
                write_state(update)
            except InvalidUpdateError:
                outcomes[name] = "rejected"
            else:
                outcomes[name] = "queued"

        first = task_thread(write, "first", {"slow": [1], "one": "a"})
        second = task_thread(write, "second", {"one": "b"})
        first.start()
        checking.wait(5)
        second.start()
        second.join(0.1)
        release.set()
        first.join()
        second.join()

    graph = StateGraph(SlowState).add_node("node", node).add_edge(START, "node")
    assert graph.compile().invoke({"slow": [0]}) == {"one": "a", "slow": [0, 1]}
    assert outcomes == {"first": "queued", "second": "rejected"}


def test_updates_sent_before_an_interrupt_apply_once_after_resume() -> None:
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
