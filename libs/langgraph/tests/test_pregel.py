import enum
import logging
import operator
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    get_type_hints,
)

import pytest
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion
from typing_extensions import TypedDict

from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import CONFIG_KEY_NODE_FINISHED, ERROR, PULL, START
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from langgraph.pregel import GraphRecursionError, NodeBuilder, Pregel, StateSnapshot
from langgraph.pregel.retry import RetryPolicy
from langgraph.types import (
    Command,
    Interrupt,
    PregelTask,
    Send,
    interrupt,
)
from langgraph.utils.runnable import RunnableSeq
from tests.agents import AgentAction, AgentFinish
from tests.any_str import AnyStr, AnyVersion, FloatBetween, UnsortedSequence
from tests.conftest import (
    ALL_CHECKPOINTERS_SYNC,
    REGULAR_CHECKPOINTERS_SYNC,
)
from tests.messages import (
    _AnyIdHumanMessage,
)

pytestmark = pytest.mark.anyio

logger = logging.getLogger(__name__)


def test_graph_validation() -> None:
    class State(TypedDict):
        hello: str

    def logic(inp: str) -> str:
        return ""

    workflow = StateGraph(State)
    workflow.add_node("agent", logic)
    workflow.set_entry_point("agent")
    workflow.set_finish_point("agent")
    assert workflow.compile(), "valid graph"

    # Accept a dead-end
    workflow = StateGraph(State)
    workflow.add_node("agent", logic)
    workflow.set_entry_point("agent")
    workflow.compile()

    workflow = StateGraph(State)
    workflow.add_node("agent", logic)
    workflow.set_finish_point("agent")
    with pytest.raises(ValueError, match="must have an entrypoint"):
        workflow.compile()

    workflow = StateGraph(State)
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", logic, {"continue": "tools", "exit": END})
    workflow.add_edge("tools", "agent")
    assert workflow.compile(), "valid graph"

    workflow = StateGraph(State)
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    workflow.set_entry_point("tools")
    workflow.add_conditional_edges("agent", logic, {"continue": "tools", "exit": END})
    workflow.add_edge("tools", "agent")
    assert workflow.compile(), "valid graph"

    workflow = StateGraph(State)
    workflow.set_entry_point("tools")
    workflow.add_conditional_edges("agent", logic, {"continue": "tools", "exit": END})
    workflow.add_edge("tools", "agent")
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    assert workflow.compile(), "valid graph"

    workflow = StateGraph(State)
    workflow.set_entry_point("tools")
    workflow.add_conditional_edges(
        "agent", logic, {"continue": "tools", "exit": END, "hmm": "extra"}
    )
    workflow.add_edge("tools", "agent")
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    with pytest.raises(ValueError, match="unknown"):  # extra is not defined
        workflow.compile()

    workflow = StateGraph(State)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", logic, {"continue": "tools", "exit": END})
    workflow.add_edge("tools", "extra")
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    with pytest.raises(ValueError, match="unknown"):  # extra is not defined
        workflow.compile()

    workflow = StateGraph(State)
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    workflow.add_node("extra", logic)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", logic)
    workflow.add_edge("tools", "agent")
    # Accept, even though extra is dead-end
    workflow.compile()

    graph = StateGraph(State)
    graph.add_node("start", lambda x: x)
    graph.add_edge("__start__", "start")
    graph.add_edge("unknown", "start")
    graph.add_edge("start", "__end__")
    with pytest.raises(ValueError, match="Found edge starting at unknown node "):
        graph.compile()

    def bad_reducer(a): ...

    class BadReducerState(TypedDict):
        hello: Annotated[str, bad_reducer]

    with pytest.raises(ValueError, match="Invalid reducer"):
        StateGraph(BadReducerState)

    def node_b(state: State) -> State:
        return {"hello": "world"}

    builder = StateGraph(State)
    builder.add_node("a", node_b)
    builder.add_node("b", node_b)
    builder.add_node("c", node_b)
    builder.set_entry_point("a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    graph = builder.compile()

    with pytest.raises(InvalidUpdateError, match="At key 'hello'"):
        graph.invoke({"hello": "there"})


def test_graph_validation_with_command() -> None:
    class State(TypedDict):
        foo: str
        bar: str

    def node_a(state: State):
        return Command(goto="b", update={"foo": "bar"})

    def node_b(state: State):
        return Command(goto=END, update={"bar": "baz"})

    builder = StateGraph(State)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    graph = builder.compile()
    assert graph.invoke({"foo": ""}) == {"foo": "bar", "bar": "baz"}


def test_node_schemas_custom_output() -> None:
    class State(TypedDict):
        hello: str
        bye: str
        messages: Annotated[list[str], add_messages]

    class Output(TypedDict):
        messages: list[str]

    class StateForA(TypedDict):
        hello: str
        messages: Annotated[list[str], add_messages]

    def node_a(state: StateForA) -> State:
        assert state == {
            "hello": "there",
            "messages": [_AnyIdHumanMessage(content="hello")],
        }

    class StateForB(TypedDict):
        bye: str
        now: int

    def node_b(state: StateForB):
        assert state == {
            "bye": "world",
        }
        return {
            "now": 123,
            "hello": "again",
        }

    class StateForC(TypedDict):
        hello: str
        now: int

    def node_c(state: StateForC) -> StateForC:
        assert state == {
            "hello": "again",
            "now": 123,
        }

    builder = StateGraph(State, output=Output)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_node("c", node_c)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    graph = builder.compile()

    assert graph.invoke({"hello": "there", "bye": "world", "messages": "hello"}) == {
        "messages": [_AnyIdHumanMessage(content="hello")],
    }

    builder = StateGraph(State, output=Output)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_node("c", node_c)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    graph = builder.compile()

    assert graph.invoke(
        {
            "hello": "there",
            "bye": "world",
            "messages": "hello",
            "now": 345,  # ignored because not in input schema
        }
    ) == {
        "messages": [_AnyIdHumanMessage(content="hello")],
    }

    assert [
        c
        for c in graph.stream(
            {
                "hello": "there",
                "bye": "world",
                "messages": "hello",
                "now": 345,  # ignored because not in input schema
            }
        )
    ] == [
        {"a": None},
        {"b": {"hello": "again", "now": 123}},
        {"c": None},
    ]


def test_reducer_before_first_node() -> None:
    class State(TypedDict):
        hello: str
        messages: Annotated[list[str], add_messages]

    def node_a(state: State) -> State:
        assert state == {
            "hello": "there",
            "messages": [_AnyIdHumanMessage(content="hello")],
        }

    builder = StateGraph(State)
    builder.add_node("a", node_a)
    builder.set_entry_point("a")
    builder.set_finish_point("a")
    graph = builder.compile()
    assert graph.invoke({"hello": "there", "messages": "hello"}) == {
        "hello": "there",
        "messages": [_AnyIdHumanMessage(content="hello")],
    }

    class State(TypedDict):
        hello: str
        messages: Annotated[List[str], add_messages]

    def node_a(state: State) -> State:
        assert state == {
            "hello": "there",
            "messages": [_AnyIdHumanMessage(content="hello")],
        }

    builder = StateGraph(State)
    builder.add_node("a", node_a)
    builder.set_entry_point("a")
    builder.set_finish_point("a")
    graph = builder.compile()
    assert graph.invoke({"hello": "there", "messages": "hello"}) == {
        "hello": "there",
        "messages": [_AnyIdHumanMessage(content="hello")],
    }

    class State(TypedDict):
        hello: str
        messages: Annotated[Sequence[str], add_messages]

    def node_a(state: State) -> State:
        assert state == {
            "hello": "there",
            "messages": [_AnyIdHumanMessage(content="hello")],
        }

    builder = StateGraph(State)
    builder.add_node("a", node_a)
    builder.set_entry_point("a")
    builder.set_finish_point("a")
    graph = builder.compile()
    assert graph.invoke({"hello": "there", "messages": "hello"}) == {
        "hello": "there",
        "messages": [_AnyIdHumanMessage(content="hello")],
    }


def test_invoke_single_process_in_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = (
        NodeBuilder().subscribe_to("input").add_node(add_one).write_to("output").build()
    )

    app = Pregel(
        nodes={
            "one": chain,
        },
        channels={
            "input": LastValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    assert app.invoke(2) == 3
    assert app.invoke(2, output_keys=["output"]) == {"output": 3}
    assert repr(app), "does not raise recursion error"


def test_invoke_single_process_in_write_kwargs(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = (
        NodeBuilder()
        .subscribe_to("input")
        .add_node(add_one)
        .write_to("output", fixed=5, output_plus_one=lambda x: x + 1)
        .build()
    )

    app = Pregel(
        nodes={"one": chain},
        channels={
            "input": LastValue(int),
            "output": LastValue(int),
            "fixed": LastValue(int),
            "output_plus_one": LastValue(int),
        },
        output_channels=["output", "fixed", "output_plus_one"],
        input_channels="input",
    )

    assert app.invoke(2) == {"output": 3, "fixed": 5, "output_plus_one": 4}


def test_invoke_single_process_in_out_dict(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = (
        NodeBuilder().subscribe_to("input").add_node(add_one).write_to("output").build()
    )

    app = Pregel(
        nodes={"one": chain},
        channels={"input": LastValue(int), "output": LastValue(int)},
        input_channels="input",
        output_channels=["output"],
    )

    assert app.invoke(2) == {"output": 3}


def test_invoke_single_process_in_dict_out_dict(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = (
        NodeBuilder().subscribe_to("input").add_node(add_one).write_to("output").build()
    )

    app = Pregel(
        nodes={"one": chain},
        channels={"input": LastValue(int), "output": LastValue(int)},
        input_channels=["input"],
        output_channels=["output"],
    )

    assert app.invoke({"input": 2}) == {"output": 3}


def test_invoke_two_processes_in_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = (
        NodeBuilder().subscribe_to("input").add_node(add_one).write_to("inbox").build()
    )
    two = (
        NodeBuilder().subscribe_to("inbox").add_node(add_one).write_to("output").build()
    )

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "inbox": LastValue(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    assert app.invoke(2) == 4

    with pytest.raises(GraphRecursionError):
        app.invoke(2, {"recursion_limit": 1})


@pytest.mark.parametrize("checkpointer_name", REGULAR_CHECKPOINTERS_SYNC)
def test_run_from_checkpoint_id_retains_previous_writes(
    request: pytest.FixtureRequest, checkpointer_name: str, mocker: MockerFixture
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class MyState(TypedDict):
        myval: Annotated[int, operator.add]
        otherval: bool

    class Anode:
        def __init__(self):
            self.switch = False

        def __call__(self, state: MyState):
            self.switch = not self.switch
            print("node", state)
            return {"myval": 2 if self.switch else 1, "otherval": self.switch}

    builder = StateGraph(MyState)
    thenode = Anode()  # Fun.
    builder.add_node("node_one", thenode)
    builder.add_node("node_two", thenode)
    builder.add_edge(START, "node_one")

    def _getedge(src: str):
        swap = "node_one" if src == "node_two" else "node_two"

        def _edge(st: MyState) -> Literal["__end__", "node_one", "node_two"]:
            print("edge", src, st)
            if st["myval"] > 3:
                return END
            if st["otherval"]:
                return swap
            return src

        return _edge

    builder.add_conditional_edges("node_one", _getedge("node_one"))
    builder.add_conditional_edges("node_two", _getedge("node_two"))
    graph = builder.compile(checkpointer=checkpointer)

    thread_id = uuid.uuid4()
    thread1 = {"configurable": {"thread_id": str(thread_id)}}

    result = graph.invoke({"myval": 1}, thread1, log_mode=["updates", "values"])
    assert result["myval"] == 4
    history = [c for c in graph.get_state_history(thread1)]

    assert len(history) == 4
    assert history[-1].values == {"myval": 0}
    assert history[0].values == {"myval": 4, "otherval": False}

    second_run_config = {
        **thread1,
        "configurable": {
            **thread1["configurable"],
            "checkpoint_id": history[1].config["configurable"]["checkpoint_id"],
        },
    }
    second_result = graph.invoke(None, second_run_config)
    assert second_result == {"myval": 5, "otherval": True}

    new_history = [
        c
        for c in graph.get_state_history(
            {"configurable": {"thread_id": str(thread_id), "checkpoint_ns": ""}}
        )
    ]

    assert len(new_history) == len(history) + 1
    for original, new in zip(history, new_history[1:]):
        assert original.values == new.values
        assert original.next == new.next
        assert original.metadata["step"] == new.metadata["step"]

    def _get_tasks(hist: list, start: int):
        return [h.tasks for h in hist[start:]]

    assert _get_tasks(new_history, 1) == _get_tasks(history, 0)


def test_invoke_many_processes_in_out(mocker: MockerFixture) -> None:
    test_size = 100
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    nodes = {
        "-1": NodeBuilder()
        .subscribe_to("input")
        .add_node(add_one)
        .write_to("-1")
        .build()
    }
    for i in range(test_size - 2):
        nodes[str(i)] = (
            NodeBuilder()
            .subscribe_to(str(i - 1))
            .add_node(add_one)
            .write_to(str(i))
            .build()
        )
    nodes["last"] = (
        NodeBuilder().subscribe_to(str(i)).add_node(add_one).write_to("output").build()
    )

    app = Pregel(
        nodes=nodes,
        channels={str(i): LastValue(int) for i in range(-1, test_size - 2)}
        | {"input": LastValue(int), "output": LastValue(int)},
        input_channels="input",
        output_channels="output",
    )

    for _ in range(10):
        assert app.invoke(2, {"recursion_limit": test_size}) == 2 + test_size

    with ThreadPoolExecutor() as executor:
        assert [
            *executor.map(app.invoke, [2] * 10, [{"recursion_limit": test_size}] * 10)
        ] == [2 + test_size] * 10


def test_invoke_two_processes_two_in_two_out_invalid(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = (
        NodeBuilder().subscribe_to("input").add_node(add_one).write_to("output").build()
    )
    two = (
        NodeBuilder().subscribe_to("input").add_node(add_one).write_to("output").build()
    )

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={"output": LastValue(int), "input": LastValue(int)},
        input_channels="input",
        output_channels="output",
    )

    with pytest.raises(InvalidUpdateError):
        # LastValue channels can only be updated once per iteration
        app.invoke(2)

    class State(TypedDict):
        hello: str

    def my_node(input: State) -> State:
        return {"hello": "world"}

    builder = StateGraph(State)
    builder.add_node("one", my_node)
    builder.add_node("two", my_node)
    builder.set_conditional_entry_point(lambda _: ["one", "two"])

    graph = builder.compile()
    with pytest.raises(InvalidUpdateError, match="At key 'hello'"):
        graph.invoke({"hello": "there"})


def test_invoke_two_processes_two_in_two_out_valid(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = (
        NodeBuilder().subscribe_to("input").add_node(add_one).write_to("output").build()
    )
    two = (
        NodeBuilder().subscribe_to("input").add_node(add_one).write_to("output").build()
    )

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "input": LastValue(int),
            "output": Topic(int),
        },
        input_channels="input",
        output_channels="output",
    )

    # An Inbox channel accumulates updates into a sequence
    assert app.invoke(2) == [3, 3]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_invoke_checkpoint_two(
    mocker: MockerFixture, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )
    add_one = mocker.Mock(side_effect=lambda x: x["total"] + x["input"])
    errored_once = False

    def raise_if_above_10(input: int) -> int:
        nonlocal errored_once
        if input > 4:
            if errored_once:
                pass
            else:
                errored_once = True
                raise ConnectionError("I will be retried")
        if input > 10:
            raise ValueError("Input is too large")
        return input

    one = (
        NodeBuilder()
        .subscribe_to(["input"])
        .read_from("total")
        .add_node(add_one)
        .write_to("output", "total")
        .add_node(raise_if_above_10)
        .build()
    )

    app = Pregel(
        nodes={"one": one},
        channels={
            "total": BinaryOperatorAggregate(int, operator.add),
            "input": LastValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
        checkpointer=checkpointer,
        retry_policy=RetryPolicy(),
    )

    # total starts out as 0, so output is 0+2=2
    assert app.invoke(2, {"configurable": {"thread_id": "1"}}) == 2
    checkpoint = checkpointer.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 2
    # total is now 2, so output is 2+3=5
    assert app.invoke(3, {"configurable": {"thread_id": "1"}}) == 5
    assert errored_once, "errored and retried"
    checkpoint_tup = checkpointer.get_tuple({"configurable": {"thread_id": "1"}})
    assert checkpoint_tup is not None
    assert checkpoint_tup.checkpoint["channel_values"].get("total") == 7
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        app.invoke(4, {"configurable": {"thread_id": "1"}})
    # checkpoint is not updated, error is recorded
    checkpoint_tup = checkpointer.get_tuple({"configurable": {"thread_id": "1"}})
    assert checkpoint_tup is not None
    assert checkpoint_tup.checkpoint["channel_values"].get("total") == 7
    assert checkpoint_tup.pending_writes == [
        (AnyStr(), ERROR, "ValueError('Input is too large')")
    ]
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert app.invoke(5, {"configurable": {"thread_id": "2"}}) == 5
    checkpoint = checkpointer.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    checkpoint = checkpointer.get({"configurable": {"thread_id": "2"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 5


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_pending_writes_resume(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    class State(TypedDict):
        value: Annotated[int, operator.add]

    class AwhileMaker:
        def __init__(self, sleep: float, rtn: Union[Dict, Exception]) -> None:
            self.sleep = sleep
            self.rtn = rtn
            self.reset()

        def __call__(self, input: State) -> Any:
            self.calls += 1
            time.sleep(self.sleep)
            if isinstance(self.rtn, Exception):
                raise self.rtn
            else:
                return self.rtn

        def reset(self):
            self.calls = 0

    one = AwhileMaker(0.1, {"value": 2})
    two = AwhileMaker(0.3, ConnectionError("I'm not good"))
    builder = StateGraph(State)
    builder.add_node("one", one)
    builder.add_node("two", two, retry=RetryPolicy(max_attempts=2))
    builder.add_edge(START, "one")
    builder.add_edge(START, "two")
    graph = builder.compile(checkpointer=checkpointer)

    thread1: RunnableConfig = {"configurable": {"thread_id": "1"}}
    with pytest.raises(ConnectionError, match="I'm not good"):
        graph.invoke({"value": 1}, thread1)

    # both nodes should have been called once
    assert one.calls == 1
    assert two.calls == 2  # two attempts

    # latest checkpoint should be before nodes "one", "two"
    # but we should have applied the write from "one"
    state = graph.get_state(thread1)
    assert state is not None
    assert state.values == {"value": 3}
    assert state.next == ("two",)
    assert state.tasks == (
        PregelTask(AnyStr(), "one", (PULL, "one"), result={"value": 2}),
        PregelTask(AnyStr(), "two", (PULL, "two"), 'ConnectionError("I\'m not good")'),
    )
    assert state.metadata == {
        "parents": {},
        "source": "loop",
        "step": 0,
        "writes": None,
        "thread_id": "1",
    }
    # get_state with checkpoint_id should not apply any pending writes
    state = graph.get_state(state.config)
    assert state is not None
    assert state.values == {"value": 1}
    assert state.next == ("one", "two")
    # should contain pending write of "one"
    checkpoint = checkpointer.get_tuple(thread1)
    assert checkpoint is not None
    # should contain error from "two"
    expected_writes = [
        (AnyStr(), "one", "one"),
        (AnyStr(), "value", 2),
        (AnyStr(), ERROR, 'ConnectionError("I\'m not good")'),
    ]
    assert len(checkpoint.pending_writes) == 3
    assert all(w in expected_writes for w in checkpoint.pending_writes)
    # both non-error pending writes come from same task
    non_error_writes = [w for w in checkpoint.pending_writes if w[1] != ERROR]
    assert non_error_writes[0][0] == non_error_writes[1][0]
    # error write is from the other task
    error_write = next(w for w in checkpoint.pending_writes if w[1] == ERROR)
    assert error_write[0] != non_error_writes[0][0]

    # resume execution
    with pytest.raises(ConnectionError, match="I'm not good"):
        graph.invoke(None, thread1)

    # node "one" succeeded previously, so shouldn't be called again
    assert one.calls == 1
    # node "two" should have been called once again
    assert two.calls == 4  # two attempts before + two attempts now

    # confirm no new checkpoints saved
    state_two = graph.get_state(thread1)
    assert state_two.metadata == state.metadata

    # resume execution, without exception
    two.rtn = {"value": 3}
    # both the pending write and the new write were applied, 1 + 2 + 3 = 6
    assert graph.invoke(None, thread1) == {"value": 6}

    if "shallow" in checkpointer_name:
        assert len(list(checkpointer.list(thread1))) == 1
        return

    # check all final checkpoints
    checkpoints = [c for c in checkpointer.list(thread1)]
    # we should have 3
    assert len(checkpoints) == 3
    # the last one not too interesting for this test
    assert checkpoints[0] == CheckpointTuple(
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        checkpoint={
            "v": 1,
            "id": AnyStr(),
            "ts": AnyStr(),
            "pending_sends": [],
            "versions_seen": {
                "one": {
                    "start:one": AnyVersion(),
                },
                "two": {
                    "start:two": AnyVersion(),
                },
                "__input__": {},
                "__start__": {
                    "__start__": AnyVersion(),
                },
                "__interrupt__": {
                    "value": AnyVersion(),
                    "__start__": AnyVersion(),
                    "start:one": AnyVersion(),
                    "start:two": AnyVersion(),
                },
            },
            "channel_versions": {
                "one": AnyVersion(),
                "two": AnyVersion(),
                "value": AnyVersion(),
                "__start__": AnyVersion(),
                "start:one": AnyVersion(),
                "start:two": AnyVersion(),
            },
            "channel_values": {"one": "one", "two": "two", "value": 6},
        },
        metadata={
            "parents": {},
            "step": 1,
            "source": "loop",
            "writes": {"one": {"value": 2}, "two": {"value": 3}},
            "thread_id": "1",
        },
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoints[1].config["configurable"]["checkpoint_id"],
            }
        },
        pending_writes=[],
    )
    # the previous one we assert that pending writes contains both
    # - original error
    # - successful writes from resuming after preventing error
    assert checkpoints[1] == CheckpointTuple(
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        checkpoint={
            "v": 1,
            "id": AnyStr(),
            "ts": AnyStr(),
            "pending_sends": [],
            "versions_seen": {
                "__input__": {},
                "__start__": {
                    "__start__": AnyVersion(),
                },
            },
            "channel_versions": {
                "value": AnyVersion(),
                "__start__": AnyVersion(),
                "start:one": AnyVersion(),
                "start:two": AnyVersion(),
            },
            "channel_values": {
                "value": 1,
                "start:one": "__start__",
                "start:two": "__start__",
            },
        },
        metadata={
            "parents": {},
            "step": 0,
            "source": "loop",
            "writes": None,
            "thread_id": "1",
        },
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoints[2].config["configurable"]["checkpoint_id"],
            }
        },
        pending_writes=UnsortedSequence(
            (AnyStr(), "one", "one"),
            (AnyStr(), "value", 2),
            (AnyStr(), "__error__", 'ConnectionError("I\'m not good")'),
            (AnyStr(), "two", "two"),
            (AnyStr(), "value", 3),
        ),
    )
    assert checkpoints[2] == CheckpointTuple(
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        checkpoint={
            "v": 1,
            "id": AnyStr(),
            "ts": AnyStr(),
            "pending_sends": [],
            "versions_seen": {"__input__": {}},
            "channel_versions": {
                "__start__": AnyVersion(),
            },
            "channel_values": {"__start__": {"value": 1}},
        },
        metadata={
            "parents": {},
            "step": -1,
            "source": "input",
            "writes": {"__start__": {"value": 1}},
            "thread_id": "1",
        },
        parent_config=None,
        pending_writes=UnsortedSequence(
            (AnyStr(), "value", 1),
            (AnyStr(), "start:one", "__start__"),
            (AnyStr(), "start:two", "__start__"),
        ),
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_invoke_checkpoint_three(
    mocker: MockerFixture, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    adder = mocker.Mock(side_effect=lambda x: x["total"] + x["input"])

    def raise_if_above_10(input: int) -> int:
        if input > 10:
            raise ValueError("Input is too large")
        return input

    one = (
        NodeBuilder()
        .subscribe_to(["input"])
        .read_from("total")
        .add_node(adder)
        .write_to("output", "total")
        .add_node(raise_if_above_10)
        .build()
    )

    app = Pregel(
        nodes={"one": one},
        channels={
            "total": BinaryOperatorAggregate(int, operator.add),
            "input": LastValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
        checkpointer=checkpointer,
    )

    thread_1 = {"configurable": {"thread_id": "1"}}
    # total starts out as 0, so output is 0+2=2
    assert app.invoke(2, thread_1) == 2
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 2
    assert state.next == ()
    assert (
        state.config["configurable"]["checkpoint_id"]
        == checkpointer.get(thread_1)["id"]
    )
    # total is now 2, so output is 2+3=5
    assert app.invoke(3, thread_1) == 5
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert (
        state.config["configurable"]["checkpoint_id"]
        == checkpointer.get(thread_1)["id"]
    )
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        app.invoke(4, thread_1)
    # checkpoint is updated with new input
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert state.next == ("one",)
    """we checkpoint inputs and it failed on "one", so the next node is one"""
    # we can recover from error by sending new inputs
    assert app.invoke(2, thread_1) == 9
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 16, "total is now 7+9=16"
    assert state.next == ()

    thread_2 = {"configurable": {"thread_id": "2"}}
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert app.invoke(5, thread_2) == 5
    state = app.get_state({"configurable": {"thread_id": "1"}})
    assert state is not None
    assert state.values.get("total") == 16
    assert state.next == (), "checkpoint of other thread not touched"
    state = app.get_state(thread_2)
    assert state is not None
    assert state.values.get("total") == 5
    assert state.next == ()

    if "shallow" in checkpointer_name:
        return

    assert len(list(app.get_state_history(thread_1, limit=1))) == 1
    # list all checkpoints for thread 1
    thread_1_history = [c for c in app.get_state_history(thread_1)]
    # there are 7 checkpoints
    assert len(thread_1_history) == 7
    assert Counter(c.metadata["source"] for c in thread_1_history) == {
        "input": 4,
        "loop": 3,
    }
    # sorted descending
    assert (
        thread_1_history[0].config["configurable"]["checkpoint_id"]
        > thread_1_history[1].config["configurable"]["checkpoint_id"]
    )
    # cursor pagination
    cursored = list(
        app.get_state_history(thread_1, limit=1, before=thread_1_history[0].config)
    )
    assert len(cursored) == 1
    assert cursored[0].config == thread_1_history[1].config
    # the last checkpoint
    assert thread_1_history[0].values["total"] == 16
    # the first "loop" checkpoint
    assert thread_1_history[-2].values["total"] == 2
    # can get each checkpoint using aget with config
    assert (
        checkpointer.get(thread_1_history[0].config)["id"]
        == thread_1_history[0].config["configurable"]["checkpoint_id"]
    )
    assert (
        checkpointer.get(thread_1_history[1].config)["id"]
        == thread_1_history[1].config["configurable"]["checkpoint_id"]
    )

    thread_1_next_config = app.update_state(thread_1_history[1].config, 10)
    # update creates a new checkpoint
    assert (
        thread_1_next_config["configurable"]["checkpoint_id"]
        > thread_1_history[0].config["configurable"]["checkpoint_id"]
    )
    # update makes new checkpoint child of the previous one
    assert (
        app.get_state(thread_1_next_config).parent_config == thread_1_history[1].config
    )
    # 1 more checkpoint in history
    assert len(list(app.get_state_history(thread_1))) == 8
    assert Counter(c.metadata["source"] for c in app.get_state_history(thread_1)) == {
        "update": 1,
        "input": 4,
        "loop": 3,
    }
    # the latest checkpoint is the updated one
    assert app.get_state(thread_1) == app.get_state(thread_1_next_config)


def test_invoke_two_processes_two_in_join_two_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: sorted(y + 10 for y in x))

    one = (
        NodeBuilder().subscribe_to("input").add_node(add_one).write_to("inbox").build()
    )
    chain_three = (
        NodeBuilder().subscribe_to("input").add_node(add_one).write_to("inbox").build()
    )
    chain_four = (
        NodeBuilder()
        .subscribe_to("inbox")
        .add_node(add_10_each)
        .write_to("output")
        .build()
    )

    app = Pregel(
        nodes={
            "one": one,
            "chain_three": chain_three,
            "chain_four": chain_four,
        },
        channels={
            "inbox": Topic(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    # Then invoke app
    # We get a single array result as chain_four waits for all publishers to finish
    # before operating on all elements published to topic_two as an array
    for _ in range(100):
        assert app.invoke(2) == [13, 13]

    with ThreadPoolExecutor() as executor:
        assert [*executor.map(app.invoke, [2] * 100)] == [[13, 13]] * 100


def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = (
        NodeBuilder()
        .subscribe_to("input")
        .add_node(add_one)
        .write_to("output", "between")
        .build()
    )
    two = (
        NodeBuilder()
        .subscribe_to("between")
        .add_node(add_one)
        .write_to("output")
        .build()
    )

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "input": LastValue(int),
            "between": LastValue(int),
            "output": LastValue(int),
        },
        stream_channels=["output", "between"],
        input_channels="input",
        output_channels="output",
    )

    assert [c for c in app.stream(2, stream_mode="updates")] == [
        {"one": {"between": 3, "output": 3}},
        {"two": {"output": 4}},
    ]
    assert [c for c in app.stream(2)] == [
        {"between": 3, "output": 3},
        {"between": 3, "output": 4},
    ]


def test_invoke_two_processes_no_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = (
        NodeBuilder()
        .subscribe_to("input")
        .add_node(add_one)
        .write_to("between")
        .build()
    )
    two = NodeBuilder().subscribe_to("between").add_node(add_one).build()

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "input": LastValue(int),
            "between": LastValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    # It finishes executing (once no more messages being published)
    # but returns nothing, as nothing was published to OUT topic
    assert app.invoke(2) is None


def test_invoke_two_processes_no_in(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = (
        NodeBuilder()
        .subscribe_to("between")
        .add_node(add_one)
        .write_to("output")
        .build()
    )
    two = NodeBuilder().subscribe_to("between").add_node(add_one).build()

    with pytest.raises(TypeError):
        Pregel(nodes={"one": one, "two": two})


def test_conditional_entrypoint_to_multiple_state_graph(
    snapshot: SnapshotAssertion,
) -> None:
    class OverallState(TypedDict):
        locations: list[str]
        results: Annotated[list[str], operator.add]

    def get_weather(state: OverallState) -> OverallState:
        location = state["location"]
        weather = "sunny" if len(location) > 2 else "cloudy"
        return {"results": [f"It's {weather} in {location}"]}

    def continue_to_weather(state: OverallState) -> list[Send]:
        return [
            Send("get_weather", {"location": location})
            for location in state["locations"]
        ]

    workflow = StateGraph(OverallState)

    workflow.add_node("get_weather", get_weather)
    workflow.add_edge("get_weather", END)
    workflow.set_conditional_entry_point(continue_to_weather)

    app = workflow.compile()

    assert app.invoke({"locations": ["sf", "nyc"]}) == {
        "locations": ["sf", "nyc"],
        "results": ["It's cloudy in sf", "It's sunny in nyc"],
    }

    assert [*app.stream({"locations": ["sf", "nyc"]}, stream_mode="values")][-1] == {
        "locations": ["sf", "nyc"],
        "results": ["It's cloudy in sf", "It's sunny in nyc"],
    }


def test_conditional_state_graph_with_list_edge_inputs(snapshot: SnapshotAssertion):
    class State(TypedDict):
        foo: Annotated[list[str], operator.add]

    graph_builder = StateGraph(State)
    graph_builder.add_node("A", lambda x: {"foo": ["A"]})
    graph_builder.add_node("B", lambda x: {"foo": ["B"]})
    graph_builder.add_edge(START, "A")
    graph_builder.add_edge(START, "B")
    graph_builder.add_edge(["A", "B"], END)

    app = graph_builder.compile()
    assert app.invoke({"foo": []}) == {"foo": ["A", "B"]}


def test_state_graph_w_config_inherited_state_keys(snapshot: SnapshotAssertion) -> None:
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import tool

    class BaseState(TypedDict):
        input: str
        agent_outcome: Optional[Union[AgentAction, AgentFinish]]

    class AgentState(BaseState, total=False):
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    assert get_type_hints(AgentState).keys() == {
        "input",
        "agent_outcome",
        "intermediate_steps",
    }

    class Config(TypedDict, total=False):
        tools: list[str]

    # Assemble the tools
    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    # Construct the agent
    prompt = PromptTemplate.from_template("Hello!")

    llm = FakeStreamingListLLM(
        responses=[
            "tool:search_api:query",
            "tool:search_api:another",
            "finish:answer",
        ]
    )

    def agent_parser(input: str) -> dict[str, Union[AgentAction, AgentFinish]]:
        if input.startswith("finish"):
            _, answer = input.split(":")
            return {
                "agent_outcome": AgentFinish(
                    return_values={"answer": answer}, log=input
                )
            }
        else:
            _, tool_name, tool_input = input.split(":")
            return {
                "agent_outcome": AgentAction(
                    tool=tool_name, tool_input=tool_input, log=input
                )
            }

    agent = RunnableSeq(prompt, llm, agent_parser)

    # Define tool execution logic
    def execute_tools(data: AgentState) -> dict:
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = {t.name: t for t in tools}[agent_action.tool].invoke(
            agent_action.tool_input
        )
        return {"intermediate_steps": [(agent_action, observation)]}

    # Define decision-making logic
    def should_continue(data: AgentState) -> str:
        # Logic to decide whether to continue in the loop or exit
        if isinstance(data["agent_outcome"], AgentFinish):
            return "exit"
        else:
            return "continue"

    # Define a new graph
    builder = StateGraph(AgentState, Config)

    builder.add_node("agent", agent)
    builder.add_node("tools", execute_tools)

    builder.set_entry_point("agent")

    builder.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    builder.add_edge("tools", "agent")

    app = builder.compile()

    assert builder.channels.keys() == {"input", "agent_outcome", "intermediate_steps"}

    assert app.invoke({"input": "what is weather in sf"}) == {
        "agent_outcome": AgentFinish(
            return_values={"answer": "answer"}, log="finish:answer"
        ),
        "input": "what is weather in sf",
        "intermediate_steps": [
            (
                AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
                "result for query",
            ),
            (
                AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
                "result for another",
            ),
        ],
    }


def test_conditional_entrypoint_graph_state(snapshot: SnapshotAssertion) -> None:
    class AgentState(TypedDict, total=False):
        input: str
        output: str
        steps: Annotated[list[str], operator.add]

    def left(data: AgentState) -> AgentState:
        return {"output": data["input"] + "->left"}

    def right(data: AgentState) -> AgentState:
        return {"output": data["input"] + "->right"}

    def should_start(data: AgentState) -> str:
        assert data["steps"] == [], "Expected input to be read from the state"
        # Logic to decide where to start
        if len(data["input"]) > 10:
            return "go-right"
        else:
            return "go-left"

    # Define a new graph
    workflow = StateGraph(AgentState)

    workflow.add_node("left", left)
    workflow.add_node("right", right)

    workflow.set_conditional_entry_point(
        should_start, {"go-left": "left", "go-right": "right"}
    )

    workflow.add_conditional_edges("left", lambda data: END, {END: END})
    workflow.add_edge("right", END)

    app = workflow.compile()

    assert app.invoke({"input": "what is weather in sf"}) == {
        "input": "what is weather in sf",
        "output": "what is weather in sf->right",
        "steps": [],
    }

    assert [*app.stream({"input": "what is weather in sf"})] == [
        {"right": {"output": "what is weather in sf->right"}},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_in_one_fan_out_state_graph_waiting_edge(
    snapshot: SnapshotAssertion, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    workflow = StateGraph(State)

    @workflow.add_node
    def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)  # to ensure stream order
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    workflow.add_node(analyzer_one)
    workflow.add_node(retriever_one)
    workflow.add_node(retriever_two)
    workflow.add_node(qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"__interrupt__": ()},
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["qa"],
    )
    config = {"configurable": {"thread_id": "2"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"__interrupt__": ()},
    ]

    app_w_interrupt.update_state(config, {"docs": ["doc5"]})
    expected_parent_config = (
        None
        if "shallow" in checkpointer_name
        else list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
    )
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "query": "analyzed: query: what is weather in sf",
            "docs": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        },
        tasks=(PregelTask(AnyStr(), "qa", (PULL, "qa")),),
        next=("qa",),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 4,
            "writes": {"retriever_one": {"docs": ["doc5"]}},
            "thread_id": "2",
        },
        parent_config=expected_parent_config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4,doc5"}},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_in_one_fan_out_state_graph_waiting_edge_via_branch(
    snapshot: SnapshotAssertion, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    def rewrite_query_then(data: State) -> Literal["retriever_two"]:
        return "retriever_two"

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_conditional_edges("rewrite_query", rewrite_query_then)
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"__interrupt__": ()},
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic1(
    snapshot: SnapshotAssertion,
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
    checkpointer_name: str,
) -> None:
    from pydantic.v1 import BaseModel, ValidationError

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class InnerObject(BaseModel):
        yo: int

    class State(BaseModel):
        class Config:
            arbitrary_types_allowed = True

        query: str
        inner: InnerObject
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]

    class Input(BaseModel):
        query: str
        inner: InnerObject

    class Output(BaseModel):
        answer: str
        docs: list[str]

    class StateUpdate(BaseModel):
        query: Optional[str] = None
        answer: Optional[str] = None
        docs: Optional[list[str]] = None

    def rewrite_query(data: State) -> State:
        return {"query": f"query: {data.query}"}

    def analyzer_one(data: State) -> State:
        return StateUpdate(query=f"analyzed: {data.query}")

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data.docs)}

    def decider(data: State) -> str:
        assert isinstance(data, State)
        return "retriever_two"

    workflow = StateGraph(State, input=Input, output=Output)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_conditional_edges(
        "rewrite_query", decider, {"retriever_two": "retriever_two"}
    )
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    with pytest.raises(ValidationError):
        app.invoke({"query": {}})

    assert app.invoke({"query": "what is weather in sf", "inner": {"yo": 1}}) == {
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf", "inner": {"yo": 1}})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        for c in app_w_interrupt.stream(
            {"query": "what is weather in sf", "inner": {"yo": 1}}, config
        )
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"__interrupt__": ()},
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    assert app_w_interrupt.update_state(
        config, {"docs": ["doc5"]}, as_node="rewrite_query"
    ) == {
        "configurable": {
            "thread_id": "1",
            "checkpoint_id": AnyStr(),
            "checkpoint_ns": "",
        }
    }


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_in_one_fan_out_state_graph_waiting_edge_plus_regular(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    def analyzer_one(data: State) -> State:
        time.sleep(0.1)
        return {"query": f"analyzed: {data['query']}"}

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.2)
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    # silly edge, to make sure having been triggered before doesn't break
    # semantics of named barrier (== waiting edges)
    workflow.add_edge("rewrite_query", "qa")

    app = workflow.compile()

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"qa": {"answer": ""}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"qa": {"answer": ""}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"__interrupt__": ()},
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


def test_in_one_fan_out_state_graph_waiting_edge_multiple() -> None:
    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    def decider(data: State) -> None:
        return None

    def decider_cond(data: State) -> str:
        if data["query"].count("analyzed") > 1:
            return "qa"
        else:
            return "rewrite_query"

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("decider", decider)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    workflow.add_edge(["retriever_one", "retriever_two"], "decider")
    workflow.add_conditional_edges("decider", decider_cond)
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: analyzed: query: what is weather in sf",
        "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
        "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"decider": None},
        {"rewrite_query": {"query": "query: analyzed: query: what is weather in sf"}},
        {
            "analyzer_one": {
                "query": "analyzed: query: analyzed: query: what is weather in sf"
            }
        },
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"decider": None},
        {"qa": {"answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4"}},
    ]


def test_callable_in_conditional_edges_with_no_path_map() -> None:
    class State(TypedDict, total=False):
        query: str

    def rewrite(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    def analyze(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

    class ChooseAnalyzer:
        def __call__(self, data: State) -> str:
            return "analyzer"

    workflow = StateGraph(State)
    workflow.add_node("rewriter", rewrite)
    workflow.add_node("analyzer", analyze)
    workflow.add_conditional_edges("rewriter", ChooseAnalyzer())
    workflow.set_entry_point("rewriter")
    app = workflow.compile()

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
    }


def test_function_in_conditional_edges_with_no_path_map() -> None:
    class State(TypedDict, total=False):
        query: str

    def rewrite(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    def analyze(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

    def choose_analyzer(data: State) -> str:
        return "analyzer"

    workflow = StateGraph(State)
    workflow.add_node("rewriter", rewrite)
    workflow.add_node("analyzer", analyze)
    workflow.add_conditional_edges("rewriter", choose_analyzer)
    workflow.set_entry_point("rewriter")
    app = workflow.compile()

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
    }


def test_in_one_fan_out_state_graph_waiting_edge_multiple_cond_edge() -> None:
    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    def retriever_picker(data: State) -> list[str]:
        return ["analyzer_one", "retriever_two"]

    def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    def decider(data: State) -> None:
        return None

    def decider_cond(data: State) -> str:
        if data["query"].count("analyzed") > 1:
            return "qa"
        else:
            return "rewrite_query"

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("decider", decider)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_conditional_edges("rewrite_query", retriever_picker)
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_edge(["retriever_one", "retriever_two"], "decider")
    workflow.add_conditional_edges("decider", decider_cond)
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: analyzed: query: what is weather in sf",
        "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
        "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"decider": None},
        {"rewrite_query": {"query": "query: analyzed: query: what is weather in sf"}},
        {
            "analyzer_one": {
                "query": "analyzed: query: analyzed: query: what is weather in sf"
            }
        },
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"decider": None},
        {"qa": {"answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4"}},
    ]


def test_simple_multi_edge(snapshot: SnapshotAssertion) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def up(state: State):
        pass

    def side(state: State):
        pass

    def other(state: State):
        return {"my_key": "_more"}

    def down(state: State):
        pass

    graph = StateGraph(State)

    graph.add_node("up", up)
    graph.add_node("side", side)
    graph.add_node("other", other)
    graph.add_node("down", down)

    graph.set_entry_point("up")
    graph.add_edge("up", "side")
    graph.add_edge("up", "other")
    graph.add_edge(["up", "side"], "down")
    graph.set_finish_point("down")

    app = graph.compile()
    assert app.invoke({"my_key": "my_value"}) == {"my_key": "my_value_more"}
    assert [*app.stream({"my_key": "my_value"})] in (
        [
            {"up": None},
            {"side": None},
            {"other": {"my_key": "_more"}},
            {"down": None},
        ],
        [
            {"up": None},
            {"other": {"my_key": "_more"}},
            {"side": None},
            {"down": None},
        ],
    )


def test_nested_graph(snapshot: SnapshotAssertion) -> None:
    def never_called_fn(state: Any):
        assert 0, "This function should never be called"

    never_called = RunnableLambda(never_called_fn)

    class InnerState(TypedDict):
        my_key: str
        my_other_key: str

    def up(state: InnerState):
        return {"my_key": state["my_key"] + " there", "my_other_key": state["my_key"]}

    inner = StateGraph(InnerState)
    inner.add_node("up", up)
    inner.set_entry_point("up")
    inner.set_finish_point("up")

    class State(TypedDict):
        my_key: str
        never_called: Any

    def side(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile())
    graph.add_node("side", side)
    graph.set_entry_point("inner")
    graph.add_edge("inner", "side")
    graph.set_finish_point("side")

    app = graph.compile()
    assert app.invoke({"my_key": "my value", "never_called": never_called}) == {
        "my_key": "my value there and back again",
        "never_called": never_called,
    }
    assert [*app.stream({"my_key": "my value", "never_called": never_called})] == [
        {"inner": {"my_key": "my value there"}},
        {"side": {"my_key": "my value there and back again"}},
    ]
    assert [
        *app.stream(
            {"my_key": "my value", "never_called": never_called}, stream_mode="values"
        )
    ] == [
        {
            "my_key": "my value",
            "never_called": never_called,
        },
        {
            "my_key": "my value there",
            "never_called": never_called,
        },
        {
            "my_key": "my value there and back again",
            "never_called": never_called,
        },
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_subgraph_checkpoint_true(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class InnerState(TypedDict):
        my_key: Annotated[str, operator.add]
        my_other_key: str

    def inner_1(state: InnerState):
        return {"my_key": " got here", "my_other_key": state["my_key"]}

    def inner_2(state: InnerState):
        return {"my_key": " and there"}

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: str

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile(checkpointer=True))
    graph.add_edge(START, "inner")
    graph.add_conditional_edges(
        "inner", lambda s: "inner" if s["my_key"].count("there") < 2 else END
    )
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "2"}}
    assert [c for c in app.stream({"my_key": ""}, config, subgraphs=True)] == [
        (("inner",), {"inner_1": {"my_key": " got here", "my_other_key": ""}}),
        (("inner",), {"inner_2": {"my_key": " and there"}}),
        ((), {"inner": {"my_key": " got here and there"}}),
        (
            ("inner",),
            {
                "inner_1": {
                    "my_key": " got here",
                    "my_other_key": " got here and there got here and there",
                }
            },
        ),
        (("inner",), {"inner_2": {"my_key": " and there"}}),
        (
            (),
            {
                "inner": {
                    "my_key": " got here and there got here and there got here and there"
                }
            },
        ),
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_subgraph_checkpoint_true_interrupt(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    # Define subgraph
    class SubgraphState(TypedDict):
        # note that none of these keys are shared with the parent graph state
        bar: str
        baz: str

    def subgraph_node_1(state: SubgraphState):
        baz_value = interrupt("Provide baz value")
        return {"baz": baz_value}

    def subgraph_node_2(state: SubgraphState):
        return {"bar": state["bar"] + state["baz"]}

    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node(subgraph_node_1)
    subgraph_builder.add_node(subgraph_node_2)
    subgraph_builder.add_edge(START, "subgraph_node_1")
    subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
    subgraph = subgraph_builder.compile(checkpointer=True)

    class ParentState(TypedDict):
        foo: str

    def node_1(state: ParentState):
        return {"foo": "hi! " + state["foo"]}

    def node_2(state: ParentState):
        response = subgraph.invoke({"bar": state["foo"]})
        return {"foo": response["bar"]}

    builder = StateGraph(ParentState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2, subgraphs=[subgraph])
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    assert graph.invoke({"foo": "foo"}, config) == {"foo": "hi! foo"}
    print(graph.get_state(config, subgraphs=True).tasks)
    assert graph.get_state(config, subgraphs=True).tasks[0].state.values == {
        "bar": "hi! foo"
    }
    assert graph.invoke(Command(resume="baz"), config) == {"foo": "hi! foobaz"}


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_stream_subgraphs_during_execution(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class InnerState(TypedDict):
        my_key: Annotated[str, operator.add]
        my_other_key: str

    def inner_1(state: InnerState):
        return {"my_key": "got here", "my_other_key": state["my_key"]}

    def inner_2(state: InnerState):
        time.sleep(0.5)
        return {
            "my_key": " and there",
            "my_other_key": state["my_key"],
        }

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def outer_1(state: State):
        time.sleep(0.2)
        return {"my_key": " and parallel"}

    def outer_2(state: State):
        return {"my_key": " and back again"}

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile())
    graph.add_node("outer_1", outer_1)
    graph.add_node("outer_2", outer_2)

    graph.add_edge(START, "inner")
    graph.add_edge(START, "outer_1")
    graph.add_edge(["inner", "outer_1"], "outer_2")
    graph.add_edge("outer_2", END)

    app = graph.compile(checkpointer=checkpointer)

    start = time.perf_counter()
    chunks: list[tuple[float, Any]] = []
    config = {"configurable": {"thread_id": "2"}}
    for c in app.stream({"my_key": ""}, config, subgraphs=True):
        chunks.append((round(time.perf_counter() - start, 1), c))
    for idx in range(len(chunks)):
        elapsed, c = chunks[idx]
        chunks[idx] = (round(elapsed - chunks[0][0], 1), c)

    assert chunks == [
        # arrives before "inner" finishes
        (
            FloatBetween(0.0, 0.1),
            (
                (AnyStr("inner:"),),
                {"inner_1": {"my_key": "got here", "my_other_key": ""}},
            ),
        ),
        (FloatBetween(0.2, 0.3), ((), {"outer_1": {"my_key": " and parallel"}})),
        (
            FloatBetween(0.5, 0.8),
            (
                (AnyStr("inner:"),),
                {"inner_2": {"my_key": " and there", "my_other_key": "got here"}},
            ),
        ),
        (FloatBetween(0.5, 0.8), ((), {"inner": {"my_key": "got here and there"}})),
        (FloatBetween(0.5, 0.8), ((), {"outer_2": {"my_key": " and back again"}})),
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_stream_buffering_single_node(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def node(state: State):
        writer = get_stream_writer()
        writer("Before sleep")
        time.sleep(0.2)
        writer("After sleep")
        return {"my_key": "got here"}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    graph = builder.compile(checkpointer=checkpointer)

    start = time.perf_counter()
    chunks: list[tuple[float, Any]] = []
    config = {"configurable": {"thread_id": "2"}}
    for c in graph.stream({"my_key": ""}, config, stream_mode="custom"):
        chunks.append((round(time.perf_counter() - start, 1), c))

    assert chunks == [
        (FloatBetween(0.0, 0.1), "Before sleep"),
        (FloatBetween(0.2, 0.3), "After sleep"),
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_nested_graph_interrupts_parallel(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class InnerState(TypedDict):
        my_key: Annotated[str, operator.add]
        my_other_key: str

    def inner_1(state: InnerState):
        time.sleep(0.1)
        return {"my_key": "got here", "my_other_key": state["my_key"]}

    def inner_2(state: InnerState):
        return {
            "my_key": " and there",
            "my_other_key": state["my_key"],
        }

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def outer_1(state: State):
        return {"my_key": " and parallel"}

    def outer_2(state: State):
        return {"my_key": " and back again"}

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile(interrupt_before=["inner_2"]))
    graph.add_node("outer_1", outer_1)
    graph.add_node("outer_2", outer_2)

    graph.add_edge(START, "inner")
    graph.add_edge(START, "outer_1")
    graph.add_edge(["inner", "outer_1"], "outer_2")
    graph.set_finish_point("outer_2")

    app = graph.compile(checkpointer=checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert app.invoke({"my_key": ""}, config) == {
        "my_key": " and parallel",
    }

    assert app.invoke(None, config) == {
        "my_key": "got here and there and parallel and back again",
    }

    # below combo of assertions is asserting two things
    # - outer_1 finishes before inner interrupts (because we see its output in stream, which only happens after node finishes)
    # - the writes of outer are persisted in 1st call and used in 2nd call, ie outer isn't called again (because we dont see outer_1 output again in 2nd stream)
    # test stream updates w/ nested interrupt
    config = {"configurable": {"thread_id": "2"}}
    assert [*app.stream({"my_key": ""}, config, subgraphs=True)] == [
        # we got to parallel node first
        ((), {"outer_1": {"my_key": " and parallel"}}),
        ((AnyStr("inner:"),), {"inner_1": {"my_key": "got here", "my_other_key": ""}}),
        ((), {"__interrupt__": ()}),
    ]
    assert [*app.stream(None, config)] == [
        {"outer_1": {"my_key": " and parallel"}, "__metadata__": {"cached": True}},
        {"inner": {"my_key": "got here and there"}},
        {"outer_2": {"my_key": " and back again"}},
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [*app.stream({"my_key": ""}, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # test interrupts BEFORE the parallel node
    app = graph.compile(checkpointer=checkpointer, interrupt_before=["outer_1"])
    config = {"configurable": {"thread_id": "4"}}
    assert [*app.stream({"my_key": ""}, config, stream_mode="values")] == [
        {"my_key": ""}
    ]
    # while we're waiting for the node w/ interrupt inside to finish
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # test interrupts AFTER the parallel node
    app = graph.compile(checkpointer=checkpointer, interrupt_after=["outer_1"])
    config = {"configurable": {"thread_id": "5"}}
    assert [*app.stream({"my_key": ""}, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_doubly_nested_graph_interrupts(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class State(TypedDict):
        my_key: str

    class ChildState(TypedDict):
        my_key: str

    class GrandChildState(TypedDict):
        my_key: str

    def grandchild_1(state: ChildState):
        return {"my_key": state["my_key"] + " here"}

    def grandchild_2(state: ChildState):
        return {
            "my_key": state["my_key"] + " and there",
        }

    grandchild = StateGraph(GrandChildState)
    grandchild.add_node("grandchild_1", grandchild_1)
    grandchild.add_node("grandchild_2", grandchild_2)
    grandchild.add_edge("grandchild_1", "grandchild_2")
    grandchild.set_entry_point("grandchild_1")
    grandchild.set_finish_point("grandchild_2")

    child = StateGraph(ChildState)
    child.add_node(
        "child_1",
        grandchild.compile(interrupt_before=["grandchild_2"]),
    )
    child.set_entry_point("child_1")
    child.set_finish_point("child_1")

    def parent_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    def parent_2(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("parent_1", parent_1)
    graph.add_node("child", child.compile())
    graph.add_node("parent_2", parent_2)
    graph.set_entry_point("parent_1")
    graph.add_edge("parent_1", "child")
    graph.add_edge("child", "parent_2")
    graph.set_finish_point("parent_2")

    app = graph.compile(checkpointer=checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert app.invoke({"my_key": "my value"}, config) == {
        "my_key": "hi my value",
    }

    assert app.invoke(None, config) == {
        "my_key": "hi my value here and there and back again",
    }

    # test stream updates w/ nested interrupt
    nodes: list[str] = []
    config = {
        "configurable": {"thread_id": "2", CONFIG_KEY_NODE_FINISHED: nodes.append}
    }
    assert [*app.stream({"my_key": "my value"}, config)] == [
        {"parent_1": {"my_key": "hi my value"}},
        {"__interrupt__": ()},
    ]
    assert nodes == ["parent_1", "grandchild_1"]
    assert [*app.stream(None, config)] == [
        {"child": {"my_key": "hi my value here and there"}},
        {"parent_2": {"my_key": "hi my value here and there and back again"}},
    ]
    assert nodes == [
        "parent_1",
        "grandchild_1",
        "grandchild_2",
        "child_1",
        "child",
        "parent_2",
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [*app.stream({"my_key": "my value"}, config, stream_mode="values")] == [
        {"my_key": "my value"},
        {"my_key": "hi my value"},
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": "hi my value"},
        {"my_key": "hi my value here and there"},
        {"my_key": "hi my value here and there and back again"},
    ]


def test_subgraph_retries():
    class State(TypedDict):
        count: int

    class ChildState(State):
        some_list: Annotated[list, operator.add]

    called_times = 0

    class RandomError(ValueError):
        """This will be retried on."""

    def parent_node(state: State):
        return {"count": state["count"] + 1}

    def child_node_a(state: ChildState):
        nonlocal called_times
        # We want it to retry only on node_b
        # NOT re-compute the whole graph.
        assert not called_times
        called_times += 1
        return {"some_list": ["val"]}

    def child_node_b(state: ChildState):
        raise RandomError("First attempt fails")

    child = StateGraph(ChildState)
    child.add_node(child_node_a)
    child.add_node(child_node_b)
    child.add_edge("__start__", "child_node_a")
    child.add_edge("child_node_a", "child_node_b")

    parent = StateGraph(State)
    parent.add_node("parent_node", parent_node)
    parent.add_node(
        "child_graph",
        child.compile(),
        retry=RetryPolicy(
            max_attempts=3,
            retry_on=(RandomError,),
            backoff_factor=0.0001,
            initial_interval=0.0001,
        ),
    )
    parent.add_edge("parent_node", "child_graph")
    parent.set_entry_point("parent_node")

    checkpointer = InMemorySaver()
    app = parent.compile(checkpointer=checkpointer)
    with pytest.raises(RandomError):
        app.invoke({"count": 0}, {"configurable": {"thread_id": "foo"}})


def test_enum_node_names():
    class NodeName(str, enum.Enum):
        BAZ = "baz"

    class State(TypedDict):
        foo: str
        bar: str

    def baz(state: State):
        return {"bar": state["foo"] + "!"}

    graph = StateGraph(State)
    graph.add_node(NodeName.BAZ, baz)
    graph.add_edge(START, NodeName.BAZ)
    graph.add_edge(NodeName.BAZ, END)
    graph = graph.compile()

    assert graph.invoke({"foo": "hello"}) == {"foo": "hello", "bar": "hello!"}


def test_debug_retry():
    class State(TypedDict):
        messages: Annotated[list[str], operator.add]

    def node(name):
        def _node(state: State):
            return {"messages": [f"entered {name} node"]}

        return _node

    builder = StateGraph(State)
    builder.add_node("one", node("one"))
    builder.add_node("two", node("two"))
    builder.add_edge(START, "one")
    builder.add_edge("one", "two")
    builder.add_edge("two", END)

    saver = InMemorySaver()

    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": []}, config=config)

    # re-run step: 1
    target_config = next(
        c.parent_config for c in saver.list(config) if c.metadata["step"] == 1
    )
    update_config = graph.update_state(target_config, values=None)

    events = [*graph.stream(None, config=update_config, stream_mode="debug")]

    checkpoint_events = list(
        reversed([e["payload"] for e in events if e["type"] == "checkpoint"])
    )

    checkpoint_history = {
        c.config["configurable"]["checkpoint_id"]: c
        for c in graph.get_state_history(config)
    }

    def lax_normalize_config(config: Optional[dict]) -> Optional[dict]:
        if config is None:
            return None
        return config["configurable"]

    for stream in checkpoint_events:
        stream_conf = lax_normalize_config(stream["config"])
        stream_parent_conf = lax_normalize_config(stream["parent_config"])
        assert stream_conf != stream_parent_conf

        # ensure the streamed checkpoint == checkpoint from checkpointer.list()
        history = checkpoint_history[stream["config"]["configurable"]["checkpoint_id"]]
        history_conf = lax_normalize_config(history.config)
        assert stream_conf == history_conf

        history_parent_conf = lax_normalize_config(history.parent_config)
        assert stream_parent_conf == history_parent_conf


def test_debug_subgraphs():
    class State(TypedDict):
        messages: Annotated[list[str], operator.add]

    def node(name):
        def _node(state: State):
            return {"messages": [f"entered {name} node"]}

        return _node

    parent = StateGraph(State)
    child = StateGraph(State)

    child.add_node("c_one", node("c_one"))
    child.add_node("c_two", node("c_two"))
    child.add_edge(START, "c_one")
    child.add_edge("c_one", "c_two")
    child.add_edge("c_two", END)

    parent.add_node("p_one", node("p_one"))
    parent.add_node("p_two", child.compile())
    parent.add_edge(START, "p_one")
    parent.add_edge("p_one", "p_two")
    parent.add_edge("p_two", END)

    graph = parent.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "1"}}
    events = [
        *graph.stream(
            {"messages": []},
            config=config,
            stream_mode="debug",
        )
    ]

    checkpoint_events = list(
        reversed([e["payload"] for e in events if e["type"] == "checkpoint"])
    )
    checkpoint_history = list(graph.get_state_history(config))

    assert len(checkpoint_events) == len(checkpoint_history)

    def lax_normalize_config(config: Optional[dict]) -> Optional[dict]:
        if config is None:
            return None
        return config["configurable"]

    for stream, history in zip(checkpoint_events, checkpoint_history):
        assert stream["values"] == history.values
        assert stream["next"] == list(history.next)
        assert lax_normalize_config(stream["config"]) == lax_normalize_config(
            history.config
        )
        assert lax_normalize_config(stream["parent_config"]) == lax_normalize_config(
            history.parent_config
        )

        assert len(stream["tasks"]) == len(history.tasks)
        for stream_task, history_task in zip(stream["tasks"], history.tasks):
            assert stream_task["id"] == history_task.id
            assert stream_task["name"] == history_task.name
            assert stream_task["interrupts"] == history_task.interrupts
            assert stream_task.get("error") == history_task.error
            assert stream_task.get("state") == history_task.state


def test_debug_nested_subgraphs():
    from collections import defaultdict

    class State(TypedDict):
        messages: Annotated[list[str], operator.add]

    def node(name):
        def _node(state: State):
            return {"messages": [f"entered {name} node"]}

        return _node

    grand_parent = StateGraph(State)
    parent = StateGraph(State)
    child = StateGraph(State)

    child.add_node("c_one", node("c_one"))
    child.add_node("c_two", node("c_two"))
    child.add_edge(START, "c_one")
    child.add_edge("c_one", "c_two")
    child.add_edge("c_two", END)

    parent.add_node("p_one", node("p_one"))
    parent.add_node("p_two", child.compile())
    parent.add_edge(START, "p_one")
    parent.add_edge("p_one", "p_two")
    parent.add_edge("p_two", END)

    grand_parent.add_node("gp_one", node("gp_one"))
    grand_parent.add_node("gp_two", parent.compile())
    grand_parent.add_edge(START, "gp_one")
    grand_parent.add_edge("gp_one", "gp_two")
    grand_parent.add_edge("gp_two", END)

    graph = grand_parent.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "1"}}
    events = [
        *graph.stream(
            {"messages": []},
            config=config,
            stream_mode="debug",
            subgraphs=True,
        )
    ]

    stream_ns: dict[tuple, dict] = defaultdict(list)
    for ns, e in events:
        if e["type"] == "checkpoint":
            stream_ns[ns].append(e["payload"])

    assert list(stream_ns.keys()) == [
        (),
        (AnyStr("gp_two:"),),
        (AnyStr("gp_two:"), AnyStr("p_two:")),
    ]

    history_ns = {
        ns: list(
            graph.get_state_history(
                {"configurable": {"thread_id": "1", "checkpoint_ns": "|".join(ns)}}
            )
        )[::-1]
        for ns in stream_ns.keys()
    }

    def normalize_config(config: Optional[dict]) -> Optional[dict]:
        if config is None:
            return None

        clean_config = {}
        clean_config["thread_id"] = config["configurable"]["thread_id"]
        clean_config["checkpoint_id"] = config["configurable"]["checkpoint_id"]
        clean_config["checkpoint_ns"] = config["configurable"]["checkpoint_ns"]
        if "checkpoint_map" in config["configurable"]:
            clean_config["checkpoint_map"] = config["configurable"]["checkpoint_map"]

        return clean_config

    for checkpoint_events, checkpoint_history in zip(
        stream_ns.values(), history_ns.values()
    ):
        for stream, history in zip(checkpoint_events, checkpoint_history):
            assert stream["values"] == history.values
            assert stream["next"] == list(history.next)
            assert normalize_config(stream["config"]) == normalize_config(
                history.config
            )
            assert normalize_config(stream["parent_config"]) == normalize_config(
                history.parent_config
            )

            assert len(stream["tasks"]) == len(history.tasks)
            for stream_task, history_task in zip(stream["tasks"], history.tasks):
                assert stream_task["id"] == history_task.id
                assert stream_task["name"] == history_task.name
                assert stream_task["interrupts"] == history_task.interrupts
                assert stream_task.get("error") == history_task.error
                assert stream_task.get("state") == history_task.state


def test_add_sequence():
    class State(TypedDict):
        foo: Annotated[list[str], operator.add]
        bar: str

    def step1(state: State):
        return {"foo": ["step1"], "bar": "baz"}

    def step2(state: State):
        return {"foo": ["step2"]}

    # test raising if less than 1 steps
    with pytest.raises(ValueError):
        StateGraph(State).add_sequence([])

    # test raising if duplicate step names
    with pytest.raises(ValueError):
        StateGraph(State).add_sequence([step1, step1])

    with pytest.raises(ValueError):
        StateGraph(State).add_sequence([("foo", step1), ("foo", step1)])

    # test unnamed steps
    builder = StateGraph(State)
    builder.add_sequence([step1, step2])
    builder.add_edge(START, "step1")
    graph = builder.compile()
    result = graph.invoke({"foo": []})
    assert result == {"foo": ["step1", "step2"], "bar": "baz"}
    stream_chunks = list(graph.stream({"foo": []}))
    assert stream_chunks == [
        {"step1": {"foo": ["step1"], "bar": "baz"}},
        {"step2": {"foo": ["step2"]}},
    ]

    # test named steps
    builder_named_steps = StateGraph(State)
    builder_named_steps.add_sequence([("meow1", step1), ("meow2", step2)])
    builder_named_steps.add_edge(START, "meow1")
    graph_named_steps = builder_named_steps.compile()
    result = graph_named_steps.invoke({"foo": []})
    stream_chunks = list(graph_named_steps.stream({"foo": []}))
    assert result == {"foo": ["step1", "step2"], "bar": "baz"}
    assert stream_chunks == [
        {"meow1": {"foo": ["step1"], "bar": "baz"}},
        {"meow2": {"foo": ["step2"]}},
    ]

    builder_named_steps = StateGraph(State)
    builder_named_steps.add_sequence(
        [
            ("meow1", lambda state: {"foo": ["foo"]}),
            ("meow2", lambda state: {"bar": state["foo"][0] + "bar"}),
        ],
    )
    builder_named_steps.add_edge(START, "meow1")
    graph_named_steps = builder_named_steps.compile()
    result = graph_named_steps.invoke({"foo": []})
    stream_chunks = list(graph_named_steps.stream({"foo": []}))
    # filtered by output schema
    assert result == {"bar": "foobar", "foo": ["foo"]}
    assert stream_chunks == [
        {"meow1": {"foo": ["foo"]}},
        {"meow2": {"bar": "foobar"}},
    ]

    # test two sequences

    def a(state: State):
        return {"foo": ["a"]}

    def b(state: State):
        return {"foo": ["b"]}

    builder_two_sequences = StateGraph(State)
    builder_two_sequences.add_sequence([a])
    builder_two_sequences.add_sequence([b])
    builder_two_sequences.add_edge(START, "a")
    builder_two_sequences.add_edge("a", "b")
    graph_two_sequences = builder_two_sequences.compile()

    result = graph_two_sequences.invoke({"foo": []})
    assert result == {"foo": ["a", "b"]}

    stream_chunks = list(graph_two_sequences.stream({"foo": []}))
    assert stream_chunks == [
        {"a": {"foo": ["a"]}},
        {"b": {"foo": ["b"]}},
    ]

    # test mixed nodes and sequences

    def c(state: State):
        return {"foo": ["c"]}

    def d(state: State):
        return {"foo": ["d"]}

    def e(state: State):
        return {"foo": ["e"]}

    def foo(state: State):
        if state["foo"][0] == "a":
            return "d"
        else:
            return "c"

    builder_complex = StateGraph(State)
    builder_complex.add_sequence([a, b])
    builder_complex.add_conditional_edges("b", foo)
    builder_complex.add_node(c)
    builder_complex.add_sequence([d, e])
    builder_complex.add_edge(START, "a")
    graph_complex = builder_complex.compile()

    result = graph_complex.invoke({"foo": []})
    assert result == {"foo": ["a", "b", "d", "e"]}

    result = graph_complex.invoke({"foo": ["start"]})
    assert result == {"foo": ["start", "a", "b", "c"]}

    stream_chunks = list(graph_complex.stream({"foo": []}))
    assert stream_chunks == [
        {"a": {"foo": ["a"]}},
        {"b": {"foo": ["b"]}},
        {"d": {"foo": ["d"]}},
        {"e": {"foo": ["e"]}},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_parent_command(request: pytest.FixtureRequest, checkpointer_name: str) -> None:
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import tool

    @tool(return_direct=True)
    def get_user_name() -> Command:
        """Retrieve user name"""
        return Command(update={"user_name": "Meow"}, graph=Command.PARENT)

    subgraph_builder = StateGraph(MessagesState)
    subgraph_builder.add_node("tool", get_user_name)
    subgraph_builder.add_edge(START, "tool")
    subgraph = subgraph_builder.compile()

    class CustomParentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        # this key is not available to the child graph
        user_name: str

    builder = StateGraph(CustomParentState)
    builder.add_node("alice", subgraph)
    builder.add_edge(START, "alice")
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    assert graph.invoke({"messages": [("user", "get user name")]}, config) == {
        "messages": [
            _AnyIdHumanMessage(content="get user name"),
        ],
        "user_name": "Meow",
    }
    assert graph.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="get user name"),
            ],
            "user_name": "Meow",
        },
        next=(),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "writes": {
                "alice": {
                    "messages": [
                        _AnyIdHumanMessage(content="get user name"),
                    ],
                    "user_name": "Meow",
                }
            },
            "thread_id": "1",
            "step": 1,
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=(
            None
            if "shallow" in checkpointer_name
            else {
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            }
        ),
        tasks=(),
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_interrupt_subgraph(request: pytest.FixtureRequest, checkpointer_name: str):
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        baz: str

    def foo(state):
        return {"baz": "foo"}

    def bar(state):
        value = interrupt("Please provide baz value:")
        return {"baz": value}

    child_builder = StateGraph(State)
    child_builder.add_node(bar)
    child_builder.add_edge(START, "bar")

    builder = StateGraph(State)
    builder.add_node(foo)
    builder.add_node("bar", child_builder.compile())
    builder.add_edge(START, "foo")
    builder.add_edge("foo", "bar")
    graph = builder.compile(checkpointer=checkpointer)

    thread1 = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert graph.invoke({"baz": ""}, thread1)
    # Resume with answer
    assert graph.invoke(Command(resume="bar"), thread1)


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_interrupt_multiple(request: pytest.FixtureRequest, checkpointer_name: str):
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def node(s: State) -> State:
        answer = interrupt({"value": 1})
        answer2 = interrupt({"value": 2})
        return {"my_key": answer + " " + answer2}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")

    graph = builder.compile(checkpointer=checkpointer)
    thread1 = {"configurable": {"thread_id": "1"}}

    assert [e for e in graph.stream({"my_key": "DE", "market": "DE"}, thread1)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value={"value": 1},
                    resumable=True,
                    ns=[AnyStr("node:")],
                    when="during",
                ),
            )
        }
    ]

    assert [
        event
        for event in graph.stream(
            Command(resume="answer 1", update={"my_key": "foofoo"}), thread1
        )
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value={"value": 2},
                    resumable=True,
                    ns=[AnyStr("node:")],
                    when="during",
                ),
            )
        }
    ]

    assert [event for event in graph.stream(Command(resume="answer 2"), thread1)] == [
        {"node": {"my_key": "answer 1 answer 2"}},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_interrupt_loop(request: pytest.FixtureRequest, checkpointer_name: str):
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        age: int
        other: str

    def ask_age(s: State):
        """Ask an expert for help."""
        question = "How old are you?"
        value = None
        for _ in range(10):
            value: str = interrupt(question)
            if not value.isdigit() or int(value) < 18:
                question = "invalid response"
                value = None
            else:
                break

        return {"age": int(value)}

    builder = StateGraph(State)
    builder.add_node("node", ask_age)
    builder.add_edge(START, "node")

    graph = builder.compile(checkpointer=checkpointer)
    thread1 = {"configurable": {"thread_id": "1"}}

    assert [e for e in graph.stream({"other": ""}, thread1)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="How old are you?",
                    resumable=True,
                    ns=[AnyStr("node:")],
                    when="during",
                ),
            )
        }
    ]

    assert [
        event
        for event in graph.stream(
            Command(resume="13"),
            thread1,
        )
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="invalid response",
                    resumable=True,
                    ns=[AnyStr("node:")],
                    when="during",
                ),
            )
        }
    ]

    assert [
        event
        for event in graph.stream(
            Command(resume="15"),
            thread1,
        )
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="invalid response",
                    resumable=True,
                    ns=[AnyStr("node:")],
                    when="during",
                ),
            )
        }
    ]

    assert [event for event in graph.stream(Command(resume="19"), thread1)] == [
        {"node": {"age": 19}},
    ]


def test_dict_mixed_return() -> None:
    class State(TypedDict):
        foo: Annotated[str, operator.add]

    def my_node(state: State):
        return [Command(update={"foo": "a"}), {"foo": "b"}]

    graph = StateGraph(State)
    graph.add_node(my_node)
    graph.add_edge(START, "my_node")
    graph = graph.compile()

    assert graph.invoke({"foo": ""}) == {"foo": "ab"}


def test_command_pydantic_dataclass() -> None:
    from pydantic import BaseModel

    class PydanticState(BaseModel):
        foo: str

    @dataclass
    class DataclassState:
        foo: str

    for State in (PydanticState, DataclassState):

        def node_a(state) -> Command[Literal["node_b"]]:
            return Command(
                update=State(foo="foo"),
                goto="node_b",
            )

        def node_b(state):
            return {"foo": state.foo + "bar"}

        builder = StateGraph(State)
        builder.add_edge(START, "node_a")
        builder.add_node(node_a)
        builder.add_node(node_b)
        graph = builder.compile()
        assert graph.invoke(State(foo="")) == {"foo": "foobar"}


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_command_with_static_breakpoints(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    """Test that we can use Command to resume and update with static breakpoints."""

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        """The graph state."""

        foo: str

    def node1(state: State):
        return {
            "foo": state["foo"] + "|node-1",
        }

    def node2(state: State):
        return {
            "foo": state["foo"] + "|node-2",
        }

    builder = StateGraph(State)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")

    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["node1"])
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Start the graph and interrupt at the first node
    graph.invoke({"foo": "abc"}, config)
    result = graph.invoke(Command(resume="node1"), config)
    assert result == {"foo": "abc|node-1|node-2"}


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_multistep_plan(request: pytest.FixtureRequest, checkpointer_name: str):
    from langchain_core.messages import AnyMessage

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict, total=False):
        plan: list[Union[str, list[str]]]
        messages: Annotated[list[AnyMessage], add_messages]

    def planner(state: State):
        if state.get("plan") is None:
            # create plan somehow
            plan = ["step1", ["step2", "step3"], "step4"]
            # pick the first step to execute next
            first_step, *plan = plan
            # put the rest of plan in state
            return Command(goto=first_step, update={"plan": plan})
        elif state["plan"]:
            # go to the next step of the plan
            next_step, *next_plan = state["plan"]
            return Command(goto=next_step, update={"plan": next_plan})
        else:
            # the end of the plan
            pass

    def step1(state: State):
        return Command(goto="planner", update={"messages": [("user", "step1")]})

    def step2(state: State):
        return Command(goto="planner", update={"messages": [("user", "step2")]})

    def step3(state: State):
        return Command(goto="planner", update={"messages": [("user", "step3")]})

    def step4(state: State):
        return Command(goto="planner", update={"messages": [("user", "step4")]})

    builder = StateGraph(State)
    builder.add_node(planner)
    builder.add_node(step1)
    builder.add_node(step2)
    builder.add_node(step3)
    builder.add_node(step4)
    builder.add_edge(START, "planner")
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    assert graph.invoke({"messages": [("user", "start")]}, config) == {
        "messages": [
            _AnyIdHumanMessage(content="start"),
            _AnyIdHumanMessage(content="step1"),
            _AnyIdHumanMessage(content="step2"),
            _AnyIdHumanMessage(content="step3"),
            _AnyIdHumanMessage(content="step4"),
        ],
        "plan": [],
    }


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_command_goto_with_static_breakpoints(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    """Use Command goto with static breakpoints."""

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        """The graph state."""

        foo: Annotated[str, operator.add]

    def node1(state: State):
        return {
            "foo": "|node-1",
        }

    def node2(state: State):
        return {
            "foo": "|node-2",
        }

    builder = StateGraph(State)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")

    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["node1"])

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Start the graph and interrupt at the first node
    graph.invoke({"foo": "abc"}, config)
    result = graph.invoke(Command(goto=["node2"]), config)
    assert result == {"foo": "abc|node-1|node-2|node-2"}


def test_parallel_node_execution():
    """Test that parallel nodes execute concurrently."""

    class State(TypedDict):
        results: Annotated[list[str], operator.add]

    def slow_node(state: State):
        time.sleep(1)
        return {"results": ["slow"]}

    def fast_node(state: State):
        time.sleep(2)
        return {"results": ["fast"]}

    builder = StateGraph(State)
    builder.add_node("slow", slow_node)
    builder.add_node("fast", fast_node)
    builder.add_edge(START, "slow")
    builder.add_edge(START, "fast")

    graph = builder.compile()

    start = time.perf_counter()
    result = graph.invoke({"results": []})
    duration = time.perf_counter() - start

    # Fast node result should be available first
    assert "fast" in result["results"][0]

    # Total duration should be less than sum of both nodes
    assert duration < 3.0


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_multiple_interrupt_state_persistence(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    """Test that state is preserved correctly across multiple interrupts."""

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        steps: Annotated[list[str], operator.add]

    def interruptible_node(state: State):
        first = interrupt("First interrupt")
        second = interrupt("Second interrupt")
        return {"steps": [first, second]}

    builder = StateGraph(State)
    builder.add_node("node", interruptible_node)
    builder.add_edge(START, "node")

    app = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    # First execution - should hit first interrupt
    app.invoke({"steps": []}, config)

    # State should still be empty since node hasn't returned
    state = app.get_state(config)
    assert state.values == {"steps": []}

    # Resume after first interrupt - should hit second interrupt
    app.invoke(Command(resume="step1"), config)

    # State should still be empty since node hasn't returned
    state = app.get_state(config)
    assert state.values == {"steps": []}

    # Resume after second interrupt - node should complete
    result = app.invoke(Command(resume="step2"), config)

    # Now state should contain both steps since node returned
    assert result["steps"] == ["step1", "step2"]
    state = app.get_state(config)
    assert state.values["steps"] == ["step1", "step2"]


def test_concurrent_execution_thread_safety():
    """Test thread safety during concurrent execution."""

    class State(TypedDict):
        counter: Annotated[int, operator.add]

    results = deque()  # thread-safe queue
    threads: list[threading.Thread] = []

    def slow_node(state: State):
        time.sleep(0.1)
        return {"counter": 1}

    builder = StateGraph(State)
    builder.add_node("node", slow_node)
    builder.add_edge(START, "node")
    graph = builder.compile()

    def run_graph():
        result = graph.invoke({"counter": 0})
        results.append(result)

    # Start multiple threads
    for _ in range(10):
        thread = threading.Thread(target=run_graph)
        thread.start()
        threads.append(thread)

    # Wait for all threads
    for thread in threads:
        thread.join()

    # Verify results are independent
    assert len(results) == 10
    for result in results:
        assert result["counter"] == 1


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_checkpoint_recovery(request: pytest.FixtureRequest, checkpointer_name: str):
    """Test recovery from checkpoints after failures."""
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        steps: Annotated[list[str], operator.add]
        attempt: int  # Track number of attempts

    def failing_node(state: State):
        # Fail on first attempt, succeed on retry
        if state["attempt"] == 1:
            raise RuntimeError("Simulated failure")
        return {"steps": ["node1"]}

    def second_node(state: State):
        return {"steps": ["node2"]}

    builder = StateGraph(State)
    builder.add_node("node1", failing_node)
    builder.add_node("node2", second_node)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")

    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    # First attempt should fail
    with pytest.raises(RuntimeError):
        graph.invoke({"steps": ["start"], "attempt": 1}, config)

    # Verify checkpoint state
    state = graph.get_state(config)
    assert state is not None
    assert state.values == {"steps": ["start"], "attempt": 1}  # input state saved
    assert state.next == ("node1",)  # Should retry failed node
    assert "RuntimeError('Simulated failure')" in state.tasks[0].error

    # Retry with updated attempt count
    result = graph.invoke({"steps": [], "attempt": 2}, config)
    assert result == {"steps": ["start", "node1", "node2"], "attempt": 2}

    if "shallow" in checkpointer_name:
        return

    # Verify checkpoint history shows both attempts
    history = list(graph.get_state_history(config))
    assert len(history) == 6  # Initial + failed attempt + successful attempt

    # Verify the error was recorded in checkpoint
    failed_checkpoint = next(c for c in history if c.tasks and c.tasks[0].error)
    assert "RuntimeError('Simulated failure')" in failed_checkpoint.tasks[0].error


def test_multiple_updates() -> None:
    class State(TypedDict):
        foo: Annotated[str, operator.add]

    def node_a(state):
        return [Command(update={"foo": "a1"}), Command(update={"foo": "a2"})]

    def node_b(state):
        return {"foo": "b"}

    graph = (
        StateGraph(State)
        .add_sequence([node_a, node_b])
        .add_edge(START, "node_a")
        .compile()
    )

    assert graph.invoke({"foo": ""}) == {
        "foo": "a1a2b",
    }

    # only streams the last update from node_a
    assert [c for c in graph.stream({"foo": ""}, stream_mode="updates")] == [
        {"node_a": [{"foo": "a1"}, {"foo": "a2"}]},
        {"node_b": {"foo": "b"}},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_double_interrupt_subgraph(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class AgentState(TypedDict):
        input: str

    def node_1(state: AgentState):
        result = interrupt("interrupt node 1")
        return {"input": result}

    def node_2(state: AgentState):
        result = interrupt("interrupt node 2")
        return {"input": result}

    subgraph_builder = (
        StateGraph(AgentState)
        .add_node("node_1", node_1)
        .add_node("node_2", node_2)
        .add_edge(START, "node_1")
        .add_edge("node_1", "node_2")
        .add_edge("node_2", END)
    )

    # invoke the sub graph
    subgraph = subgraph_builder.compile(checkpointer=checkpointer)
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    assert [c for c in subgraph.stream({"input": "test"}, thread)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="interrupt node 1",
                    resumable=True,
                    ns=[AnyStr("node_1:")],
                    when="during",
                ),
            )
        },
    ]
    # resume from the first interrupt
    assert [c for c in subgraph.stream(Command(resume="123"), thread)] == [
        {
            "node_1": {"input": "123"},
        },
        {
            "__interrupt__": (
                Interrupt(
                    value="interrupt node 2",
                    resumable=True,
                    ns=[AnyStr("node_2:")],
                    when="during",
                ),
            )
        },
    ]
    # resume from the second interrupt
    assert [c for c in subgraph.stream(Command(resume="123"), thread)] == [
        {
            "node_2": {"input": "123"},
        },
    ]

    subgraph = subgraph_builder.compile()

    def invoke_sub_agent(state: AgentState):
        return subgraph.invoke(state)

    parent_agent = (
        StateGraph(AgentState)
        .add_node("invoke_sub_agent", invoke_sub_agent)
        .add_edge(START, "invoke_sub_agent")
        .add_edge("invoke_sub_agent", END)
        .compile(checkpointer=checkpointer)
    )

    assert [c for c in parent_agent.stream({"input": "test"}, thread)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="interrupt node 1",
                    resumable=True,
                    ns=[AnyStr("invoke_sub_agent:"), AnyStr("node_1:")],
                    when="during",
                ),
            )
        },
    ]

    # resume from the first interrupt
    assert [c for c in parent_agent.stream(Command(resume=True), thread)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="interrupt node 2",
                    resumable=True,
                    ns=[AnyStr("invoke_sub_agent:"), AnyStr("node_2:")],
                    when="during",
                ),
            )
        }
    ]

    # resume from 2nd interrupt
    assert [c for c in parent_agent.stream(Command(resume=True), thread)] == [
        {
            "invoke_sub_agent": {"input": True},
        },
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_multiple_subgraphs(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define the subgraphs
    def add(state):
        return {"result": state["a"] + state["b"]}

    add_subgraph = (
        StateGraph(State, output=Output).add_node(add).add_edge(START, "add").compile()
    )

    def multiply(state):
        return {"result": state["a"] * state["b"]}

    multiply_subgraph = (
        StateGraph(State, output=Output)
        .add_node(multiply)
        .add_edge(START, "multiply")
        .compile()
    )

    # Test calling the same subgraph multiple times
    def call_same_subgraph(state):
        result = add_subgraph.invoke(state)
        another_result = add_subgraph.invoke({"a": result["result"], "b": 10})
        return another_result

    parent_call_same_subgraph = (
        StateGraph(State, output=Output)
        .add_node(call_same_subgraph)
        .add_edge(START, "call_same_subgraph")
        .compile(checkpointer=checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}
    assert parent_call_same_subgraph.invoke({"a": 2, "b": 3}, config) == {"result": 15}

    # Test calling multiple subgraphs
    class Output(TypedDict):
        add_result: int
        multiply_result: int

    def call_multiple_subgraphs(state):
        add_result = add_subgraph.invoke(state)
        multiply_result = multiply_subgraph.invoke(state)
        return {
            "add_result": add_result["result"],
            "multiply_result": multiply_result["result"],
        }

    parent_call_multiple_subgraphs = (
        StateGraph(State, output=Output)
        .add_node(call_multiple_subgraphs)
        .add_edge(START, "call_multiple_subgraphs")
        .compile(checkpointer=checkpointer)
    )
    config = {"configurable": {"thread_id": "2"}}
    assert parent_call_multiple_subgraphs.invoke({"a": 2, "b": 3}, config) == {
        "add_result": 5,
        "multiply_result": 6,
    }


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_multiple_subgraphs_checkpointer(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class SubgraphState(TypedDict):
        sub_counter: Annotated[int, operator.add]

    def subgraph_node(state):
        return {"sub_counter": 2}

    sub_graph_1 = (
        StateGraph(SubgraphState)
        .add_node(subgraph_node)
        .add_edge(START, "subgraph_node")
        .compile(checkpointer=True)
    )

    class OtherSubgraphState(TypedDict):
        other_sub_counter: Annotated[int, operator.add]

    def other_subgraph_node(state):
        return {"other_sub_counter": 3}

    sub_graph_2 = (
        StateGraph(OtherSubgraphState)
        .add_node(other_subgraph_node)
        .add_edge(START, "other_subgraph_node")
        .compile()
    )

    class ParentState(TypedDict):
        parent_counter: int

    def parent_node(state):
        result = sub_graph_1.invoke({"sub_counter": state["parent_counter"]})
        other_result = sub_graph_2.invoke({"other_sub_counter": result["sub_counter"]})
        return {"parent_counter": other_result["other_sub_counter"]}

    parent_graph = (
        StateGraph(ParentState)
        .add_node(parent_node)
        .add_edge(START, "parent_node")
        .compile(checkpointer=checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    assert parent_graph.invoke({"parent_counter": 0}, config) == {"parent_counter": 5}
    assert parent_graph.invoke({"parent_counter": 0}, config) == {"parent_counter": 7}
    config = {"configurable": {"thread_id": "2"}}
    assert [
        c
        for c in parent_graph.stream(
            {"parent_counter": 0}, config, subgraphs=True, stream_mode="updates"
        )
    ] == [
        (("parent_node",), {"subgraph_node": {"sub_counter": 2}}),
        (
            (AnyStr("parent_node:"), "1"),
            {"other_subgraph_node": {"other_sub_counter": 3}},
        ),
        ((), {"parent_node": {"parent_counter": 5}}),
    ]
    assert [
        c
        for c in parent_graph.stream(
            {"parent_counter": 0}, config, subgraphs=True, stream_mode="updates"
        )
    ] == [
        (("parent_node",), {"subgraph_node": {"sub_counter": 2}}),
        (
            (AnyStr("parent_node:"), "1"),
            {"other_subgraph_node": {"other_sub_counter": 3}},
        ),
        ((), {"parent_node": {"parent_counter": 7}}),
    ]


def test_merging_updates_command_parent():
    # simple reducer
    def append_unique(left, right):
        combined = list(left)
        for item in right:
            if item in combined:
                continue
            else:
                combined.append(item)
        return combined

    class State(TypedDict):
        foo: str
        bar: Annotated[list[str], append_unique]

    # Define subgraph
    def subgraph_node_1(state: State):
        return Command(
            goto="subgraph_node_2",
            update={
                "foo": "foo",
                "bar": ["subgraph_node_1"],
            },
        )

    def subgraph_node_2(state: State):
        return Command(
            goto="node_3",
            update={"bar": ["subgraph_node_2"]},
            graph=Command.PARENT,
        )

    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node(subgraph_node_1)
    subgraph_builder.add_node(subgraph_node_2)
    subgraph_builder.add_edge(START, "subgraph_node_1")

    # Define main graph
    def node_1(state: State):
        return Command(
            goto="node_2",
            update={"bar": ["node_1"]},
        )

    def node_3(state: State):
        return Command(
            update={"bar": ["node_3"]},
        )

    main_builder = StateGraph(State)
    main_builder.add_node("node_1", node_1)
    main_builder.add_node("node_2", subgraph_builder.compile())
    main_builder.add_node("node_3", node_3)
    main_builder.add_edge(START, "node_1")
    main_builder.add_edge("node_2", "node_3")
    main_graph = main_builder.compile()

    assert main_graph.invoke({"foo": ""}) == {
        "foo": "foo",
        "bar": ["node_1", "subgraph_node_1", "subgraph_node_2", "node_3"],
    }

    assert list(
        main_graph.stream({"foo": ""}, stream_mode="updates", subgraphs=True)
    ) == [
        ((), {"node_1": {"bar": ["node_1"]}}),
        (
            (AnyStr("node_2:"),),
            {"subgraph_node_1": {"foo": "foo", "bar": ["subgraph_node_1"]}},
        ),
        (
            (),
            {
                "node_2": [
                    {"foo": "foo"},
                    {"bar": ["node_1", "subgraph_node_1"]},
                    {"bar": ["subgraph_node_2"]},
                ]
            },
        ),
        ((), {"node_3": {"bar": ["node_3"]}}),
    ]


def test_merging_non_overlapping_updates_command_parent():
    # simple reducer
    def append_unique(left, right):
        combined = list(left)
        for item in right:
            if item in combined:
                continue
            else:
                combined.append(item)
        return combined

    class State(TypedDict):
        foo: Annotated[list, append_unique]

    # Define subgraph
    def subgraph_node_1(state: State):
        return Command(
            goto="subgraph_node_2",
            update={
                "foo": ["bar"],
                "bar": ["subgraph_node_1"],
            },
        )

    def subgraph_node_2(state: State):
        return Command(
            goto="node_3",
            update={"bar": ["subgraph_node_2"]},
            graph=Command.PARENT,
        )

    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node(subgraph_node_1)
    subgraph_builder.add_node(subgraph_node_2)
    subgraph_builder.add_edge(START, "subgraph_node_1")

    # Define main graph
    def node_1(state: State):
        return Command(
            goto="node_2",
            update={"foo": ["foo"]},
        )

    def node_3(state: State):
        return Command(
            update={"foo": ["baz"]},
        )

    main_builder = StateGraph(State)
    main_builder.add_node("node_1", node_1)
    main_builder.add_node("node_2", subgraph_builder.compile())
    main_builder.add_node("node_3", node_3)
    main_builder.add_edge(START, "node_1")
    main_builder.add_edge("node_2", "node_3")
    main_graph = main_builder.compile()

    assert main_graph.invoke({"foo": []}) == {
        "foo": ["foo", "bar", "baz"],
    }


def test_pydantic_none_state_update() -> None:
    from pydantic import BaseModel

    class State(BaseModel):
        foo: Optional[str]

    def node_a(state: State) -> State:
        return State(foo=None)

    graph = StateGraph(State).add_node(node_a).add_edge(START, "node_a").compile()
    assert graph.invoke({"foo": ""}) == {"foo": None}


def test_get_stream_writer() -> None:
    class State(TypedDict):
        foo: str

    def my_node(state):
        writer = get_stream_writer()
        writer("custom!")
        return state

    graph = StateGraph(State).add_node(my_node).add_edge(START, "my_node").compile()
    assert list(graph.stream({"foo": "bar"}, stream_mode="custom")) == ["custom!"]
    assert list(graph.stream({"foo": "bar"}, stream_mode="values")) == [
        {"foo": "bar"},
        {"foo": "bar"},
    ]
    assert list(graph.stream({"foo": "bar"}, stream_mode=["custom", "updates"])) == [
        (
            "custom",
            "custom!",
        ),
        (
            "updates",
            {
                "my_node": {
                    "foo": "bar",
                },
            },
        ),
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_interrupt_subgraph_reenter_checkpointer_true(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class SubgraphState(TypedDict):
        foo: str
        bar: str

    class ParentState(TypedDict):
        foo: str
        counter: int

    called = []
    bar_values = []

    def subnode_1(state: SubgraphState):
        called.append("subnode_1")
        bar_values.append(state.get("bar"))
        return {"foo": "subgraph_1"}

    def subnode_2(state: SubgraphState):
        called.append("subnode_2")
        value = interrupt("Provide value")
        value += "baz"
        return {"foo": "subgraph_2", "bar": value}

    subgraph = (
        StateGraph(SubgraphState)
        .add_node(subnode_1)
        .add_node(subnode_2)
        .add_edge(START, "subnode_1")
        .add_edge("subnode_1", "subnode_2")
        .compile(checkpointer=True)
    )

    def call_subgraph(state: ParentState):
        called.append("call_subgraph")
        return subgraph.invoke(state)

    def node(state: ParentState):
        called.append("parent")
        if state["counter"] < 1:
            return Command(
                goto="call_subgraph", update={"counter": state["counter"] + 1}
            )

        return {"foo": state["foo"] + "|" + "parent"}

    parent = (
        StateGraph(ParentState)
        .add_node(call_subgraph)
        .add_node(node)
        .add_edge(START, "call_subgraph")
        .add_edge("call_subgraph", "node")
        .compile(checkpointer=checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    assert parent.invoke({"foo": "", "counter": 0}, config) == {"foo": "", "counter": 0}
    assert parent.invoke(Command(resume="bar"), config) == {
        "foo": "subgraph_2",
        "counter": 1,
    }
    assert parent.invoke(Command(resume="qux"), config) == {
        "foo": "subgraph_2|parent",
        "counter": 1,
    }
    assert called == [
        "call_subgraph",
        "subnode_1",
        "subnode_2",
        "call_subgraph",
        "subnode_2",
        "parent",
        "call_subgraph",
        "subnode_1",
        "subnode_2",
        "call_subgraph",
        "subnode_2",
        "parent",
    ]

    # invoke parent again (new turn)
    assert parent.invoke({"foo": "meow", "counter": 0}, config) == {
        "foo": "meow",
        "counter": 0,
    }
    # confirm that we preserve the state values from the previous invocation
    assert bar_values == [None, "barbaz", "quxbaz"]
