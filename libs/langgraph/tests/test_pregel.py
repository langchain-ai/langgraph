import enum
import functools
import gc
import json
import logging
import operator
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from random import randrange
from typing import Annotated, Any, Literal, get_type_hints

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.runnables.graph import Edge
from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion
from typing_extensions import NotRequired, TypedDict

from langgraph._internal._constants import CONFIG_KEY_NODE_FINISHED, ERROR, PULL
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError, InvalidUpdateError, ParentCommand
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from langgraph.pregel import (
    NodeBuilder,
    Pregel,
)
from langgraph.pregel._loop import SyncPregelLoop
from langgraph.pregel._runner import PregelRunner
from langgraph.types import (
    CachePolicy,
    Command,
    Durability,
    Interrupt,
    Overwrite,
    PregelTask,
    RetryPolicy,
    Send,
    StateSnapshot,
    StateUpdate,
    StreamWriter,
    interrupt,
)
from tests.agents import AgentAction, AgentFinish
from tests.any_str import AnyStr, AnyVersion, FloatBetween, UnsortedSequence
from tests.messages import (
    _AnyIdAIMessage,
    _AnyIdAIMessageChunk,
    _AnyIdHumanMessage,
    _AnyIdToolMessage,
)

pytestmark = pytest.mark.anyio

logger = logging.getLogger(__name__)


def test_graph_validation() -> None:
    class State(TypedDict):
        hello: str

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


def test_checkpoint_errors() -> None:
    class FaultyGetCheckpointer(InMemorySaver):
        def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
            raise ValueError("Faulty get_tuple")

    class FaultyPutCheckpointer(InMemorySaver):
        def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: dict[str, str | int | float] | None = None,
        ) -> RunnableConfig:
            raise ValueError("Faulty put")

    class FaultyPutWritesCheckpointer(InMemorySaver):
        def put_writes(
            self, config: RunnableConfig, writes: list[tuple[str, Any]], task_id: str
        ) -> RunnableConfig:
            raise ValueError("Faulty put_writes")

    class FaultyVersionCheckpointer(InMemorySaver):
        def get_next_version(self, current: int | None, channel: None) -> int:
            raise ValueError("Faulty get_next_version")

    def logic(inp: str) -> str:
        return ""

    builder = StateGraph(Annotated[str, operator.add])
    builder.add_node("agent", logic)
    builder.add_edge(START, "agent")

    graph = builder.compile(checkpointer=FaultyGetCheckpointer())
    with pytest.raises(ValueError, match="Faulty get_tuple"):
        graph.invoke("", {"configurable": {"thread_id": "thread-1"}})

    graph = builder.compile(checkpointer=FaultyPutCheckpointer())
    with pytest.raises(ValueError, match="Faulty put"):
        graph.invoke("", {"configurable": {"thread_id": "thread-1"}})

    graph = builder.compile(checkpointer=FaultyVersionCheckpointer())
    with pytest.raises(ValueError, match="Faulty get_next_version"):
        graph.invoke("", {"configurable": {"thread_id": "thread-1"}})

    # add parallel node
    builder.add_node("parallel", logic)
    builder.add_edge(START, "parallel")
    graph = builder.compile(checkpointer=FaultyPutWritesCheckpointer())
    with pytest.raises(ValueError, match="Faulty put_writes"):
        graph.invoke(
            "", {"configurable": {"thread_id": "thread-1"}}, durability="async"
        )


def test_context_json_schema() -> None:
    """Test that config json schema is generated properly."""
    chain = NodeBuilder().subscribe_only("input").write_to("output")

    @dataclass
    class Foo:
        x: int
        y: str = field(default="foo")

    app = Pregel(
        nodes={
            "one": chain,
        },
        channels={
            "ephemeral": EphemeralValue(Any),
            "input": LastValue(int),
            "output": LastValue(int),
        },
        input_channels=["input", "ephemeral"],
        output_channels="output",
        context_schema=Foo,
    )

    assert app.get_context_jsonschema() == {
        "properties": {
            "x": {
                "title": "X",
                "type": "integer",
            },
            "y": {
                "default": "foo",
                "title": "Y",
                "type": "string",
            },
        },
        "required": [
            "x",
        ],
        "title": "Foo",
        "type": "object",
    }


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

    builder = StateGraph(State, output_schema=Output)
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

    builder = StateGraph(State, output_schema=Output)
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
    chain = NodeBuilder().subscribe_only("input").do(add_one).write_to("output")

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

    assert app.input_schema.model_json_schema() == {
        "title": "LangGraphInput",
        "type": "integer",
    }
    assert app.output_schema.model_json_schema() == {
        "title": "LangGraphOutput",
        "type": "integer",
    }
    assert app.get_context_jsonschema() is None

    assert app.invoke(2) == 3
    assert app.invoke(2, output_keys=["output"]) == {"output": 3}
    assert repr(app), "does not raise recursion error"


def test_invoke_single_process_in_write_kwargs(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = (
        NodeBuilder()
        .subscribe_only("input")
        .do(add_one)
        .write_to("output", fixed=5, output_plus_one=lambda x: x + 1)
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

    assert app.input_schema.model_json_schema() == {
        "title": "LangGraphInput",
        "type": "integer",
    }
    assert app.output_schema.model_json_schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {
            "output": {"title": "Output", "type": "integer", "default": None},
            "fixed": {"title": "Fixed", "type": "integer", "default": None},
            "output_plus_one": {
                "title": "Output Plus One",
                "type": "integer",
                "default": None,
            },
        },
    }
    assert app.invoke(2) == {"output": 3, "fixed": 5, "output_plus_one": 4}


def test_invoke_single_process_in_out_dict(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = NodeBuilder().subscribe_only("input").do(add_one).write_to("output")

    app = Pregel(
        nodes={"one": chain},
        channels={"input": LastValue(int), "output": LastValue(int)},
        input_channels="input",
        output_channels=["output"],
    )

    assert app.input_schema.model_json_schema() == {
        "title": "LangGraphInput",
        "type": "integer",
    }
    assert app.output_schema.model_json_schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {
            "output": {"title": "Output", "type": "integer", "default": None}
        },
    }
    assert app.invoke(2) == {"output": 3}


def test_invoke_single_process_in_dict_out_dict(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = NodeBuilder().subscribe_only("input").do(add_one).write_to("output")

    app = Pregel(
        nodes={"one": chain},
        channels={"input": LastValue(int), "output": LastValue(int)},
        input_channels=["input"],
        output_channels=["output"],
    )
    assert app.input_schema.model_json_schema() == {
        "title": "LangGraphInput",
        "type": "object",
        "properties": {"input": {"title": "Input", "type": "integer", "default": None}},
    }
    assert app.output_schema.model_json_schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {
            "output": {"title": "Output", "type": "integer", "default": None}
        },
    }
    assert app.invoke({"input": 2}) == {"output": 3}


def test_invoke_two_processes_in_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = NodeBuilder().subscribe_only("input").do(add_one).write_to("inbox")
    two = NodeBuilder().subscribe_only("inbox").do(add_one).write_to("output")

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
        app.invoke(2, {"recursion_limit": 1}, debug=1)


def test_run_from_checkpoint_id_retains_previous_writes(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    class MyState(TypedDict):
        myval: Annotated[int, operator.add]
        otherval: bool

    class Anode:
        def __init__(self):
            self.switch = False

        def __call__(self, state: MyState):
            self.switch = not self.switch
            return {"myval": 2 if self.switch else 1, "otherval": self.switch}

    builder = StateGraph(MyState)
    thenode = Anode()  # Fun.
    builder.add_node("node_one", thenode)
    builder.add_node("node_two", thenode)
    builder.add_edge(START, "node_one")

    def _getedge(src: str):
        swap = "node_one" if src == "node_two" else "node_two"

        def _edge(st: MyState) -> Literal["__end__", "node_one", "node_two"]:
            if st["myval"] > 3:
                return END
            if st["otherval"]:
                return swap
            return src

        return _edge

    builder.add_conditional_edges("node_one", _getedge("node_one"))
    builder.add_conditional_edges("node_two", _getedge("node_two"))
    graph = builder.compile(checkpointer=sync_checkpointer)

    thread_id = uuid.uuid4()
    thread1 = {"configurable": {"thread_id": str(thread_id)}}

    result = graph.invoke({"myval": 1}, thread1, durability="async")
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


def test_batch_two_processes_in_out() -> None:
    def add_one_with_delay(inp: int) -> int:
        time.sleep(inp / 10)
        return inp + 1

    one = NodeBuilder().subscribe_only("input").do(add_one_with_delay).write_to("one")
    two = NodeBuilder().subscribe_only("one").do(add_one_with_delay).write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "one": LastValue(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    assert app.batch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]
    assert app.batch([3, 2, 1, 3, 5], output_keys=["output"]) == [
        {"output": 5},
        {"output": 4},
        {"output": 3},
        {"output": 5},
        {"output": 7},
    ]


def test_invoke_many_processes_in_out(mocker: MockerFixture) -> None:
    test_size = 100
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    nodes = {"-1": NodeBuilder().subscribe_only("input").do(add_one).write_to("-1")}
    for i in range(test_size - 2):
        nodes[str(i)] = (
            NodeBuilder().subscribe_only(str(i - 1)).do(add_one).write_to(str(i))
        )
    nodes["last"] = NodeBuilder().subscribe_only(str(i)).do(add_one).write_to("output")

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


def test_batch_many_processes_in_out(mocker: MockerFixture) -> None:
    test_size = 100
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    nodes = {"-1": NodeBuilder().subscribe_only("input").do(add_one).write_to("-1")}
    for i in range(test_size - 2):
        nodes[str(i)] = (
            NodeBuilder().subscribe_only(str(i - 1)).do(add_one).write_to(str(i))
        )
    nodes["last"] = NodeBuilder().subscribe_only(str(i)).do(add_one).write_to("output")

    app = Pregel(
        nodes=nodes,
        channels={str(i): LastValue(int) for i in range(-1, test_size - 2)}
        | {"input": LastValue(int), "output": LastValue(int)},
        input_channels="input",
        output_channels="output",
    )

    for _ in range(3):
        assert app.batch([2, 1, 3, 4, 5], {"recursion_limit": test_size}) == [
            2 + test_size,
            1 + test_size,
            3 + test_size,
            4 + test_size,
            5 + test_size,
        ]

    with ThreadPoolExecutor() as executor:
        assert [
            *executor.map(
                app.batch, [[2, 1, 3, 4, 5]] * 3, [{"recursion_limit": test_size}] * 3
            )
        ] == [
            [2 + test_size, 1 + test_size, 3 + test_size, 4 + test_size, 5 + test_size]
        ] * 3


def test_invoke_two_processes_two_in_two_out_invalid(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = NodeBuilder().subscribe_only("input").do(add_one).write_to("output")
    two = NodeBuilder().subscribe_only("input").do(add_one).write_to("output")

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
        graph.invoke({"hello": "there"}, debug=True)


def test_invoke_two_processes_two_in_two_out_valid(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = NodeBuilder().subscribe_only("input").do(add_one).write_to("output")
    two = NodeBuilder().subscribe_only("input").do(add_one).write_to("output")

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


def test_invoke_checkpoint_two(
    mocker: MockerFixture, sync_checkpointer: BaseCheckpointSaver
) -> None:
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
        .subscribe_to("input")
        .read_from("total")
        .do(add_one)
        .write_to("output", "total")
        .do(raise_if_above_10)
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
        checkpointer=sync_checkpointer,
        retry_policy=RetryPolicy(),
    )

    # total starts out as 0, so output is 0+2=2
    assert app.invoke(2, {"configurable": {"thread_id": "1"}}) == 2
    checkpoint = sync_checkpointer.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 2
    # total is now 2, so output is 2+3=5
    assert app.invoke(3, {"configurable": {"thread_id": "1"}}) == 5
    assert errored_once, "errored and retried"
    checkpoint_tup = sync_checkpointer.get_tuple({"configurable": {"thread_id": "1"}})
    assert checkpoint_tup is not None
    assert checkpoint_tup.checkpoint["channel_values"].get("total") == 7
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        app.invoke(4, {"configurable": {"thread_id": "1"}})
    # checkpoint is not updated, error is recorded
    checkpoint_tup = sync_checkpointer.get_tuple({"configurable": {"thread_id": "1"}})
    assert checkpoint_tup is not None
    assert checkpoint_tup.checkpoint["channel_values"].get("total") == 7
    assert checkpoint_tup.pending_writes == [
        (AnyStr(), ERROR, "ValueError('Input is too large')")
    ]
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert app.invoke(5, {"configurable": {"thread_id": "2"}}) == 5
    checkpoint = sync_checkpointer.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    checkpoint = sync_checkpointer.get({"configurable": {"thread_id": "2"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 5


def test_pending_writes_resume(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    class State(TypedDict):
        value: Annotated[int, operator.add]

    class AwhileMaker:
        def __init__(self, sleep: float, rtn: dict | Exception) -> None:
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
    two = AwhileMaker(0.2, ConnectionError("I'm not good"))
    builder = StateGraph(State)
    builder.add_node("one", one)
    builder.add_node(
        "two",
        two,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=0, jitter=False),
    )
    builder.add_edge(START, "one")
    builder.add_edge(START, "two")
    graph = builder.compile(checkpointer=sync_checkpointer)

    thread1: RunnableConfig = {"configurable": {"thread_id": "1"}}
    with pytest.raises(ConnectionError, match="I'm not good"):
        graph.invoke({"value": 1}, thread1, durability=durability)

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
    }
    # get_state with checkpoint_id should not apply any pending writes
    state = graph.get_state(state.config)
    assert state is not None
    assert state.values == {"value": 1}
    assert state.next == ("one", "two")
    # should contain pending write of "one"
    checkpoint = sync_checkpointer.get_tuple(thread1)
    assert checkpoint is not None
    # should contain error from "two"
    expected_writes = [
        (AnyStr(), "value", 2),
        (AnyStr(), ERROR, 'ConnectionError("I\'m not good")'),
    ]
    assert len(checkpoint.pending_writes) == 2
    assert all(w in expected_writes for w in checkpoint.pending_writes)
    # both non-error pending writes come from same task
    non_error_writes = [w for w in checkpoint.pending_writes if w[1] != ERROR]
    # error write is from the other task
    error_write = next(w for w in checkpoint.pending_writes if w[1] == ERROR)
    assert error_write[0] != non_error_writes[0][0]

    # resume execution
    with pytest.raises(ConnectionError, match="I'm not good"):
        graph.invoke(None, thread1, durability=durability)

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
    assert graph.invoke(None, thread1, durability=durability) == {"value": 6}

    # check all final checkpoints
    checkpoints = [c for c in sync_checkpointer.list(thread1)]
    # we should have 3
    assert len(checkpoints) == (3 if durability != "exit" else 2)
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
            "v": 4,
            "id": AnyStr(),
            "ts": AnyStr(),
            "versions_seen": {
                "one": {
                    "branch:to:one": AnyVersion(),
                },
                "two": {
                    "branch:to:two": AnyVersion(),
                },
                "__input__": {},
                "__start__": {
                    "__start__": AnyVersion(),
                },
                "__interrupt__": {
                    "value": AnyVersion(),
                    "__start__": AnyVersion(),
                    "branch:to:one": AnyVersion(),
                    "branch:to:two": AnyVersion(),
                },
            },
            "channel_versions": {
                "value": AnyVersion(),
                "__start__": AnyVersion(),
                "branch:to:one": AnyVersion(),
                "branch:to:two": AnyVersion(),
            },
            "channel_values": {"value": 6},
            "updated_channels": ["value"],
        },
        metadata={
            "parents": {},
            "step": 1,
            "source": "loop",
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
            "v": 4,
            "id": AnyStr(),
            "ts": AnyStr(),
            "versions_seen": {
                "__input__": {},
                "__start__": {
                    "__start__": AnyVersion(),
                },
            },
            "channel_versions": {
                "value": AnyVersion(),
                "__start__": AnyVersion(),
                "branch:to:one": AnyVersion(),
                "branch:to:two": AnyVersion(),
            },
            "channel_values": {
                "value": 1,
                "branch:to:one": None,
                "branch:to:two": None,
            },
            "updated_channels": ["branch:to:one", "branch:to:two", "value"],
        },
        metadata={
            "parents": {},
            "step": 0,
            "source": "loop",
        },
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": (
                    checkpoints[2].config["configurable"]["checkpoint_id"]
                ),
            }
        }
        if durability != "exit"
        else None,
        pending_writes=(
            UnsortedSequence(
                (AnyStr(), "value", 2),
                (AnyStr(), "__error__", 'ConnectionError("I\'m not good")'),
                (AnyStr(), "value", 3),
            )
            if durability != "exit"
            else UnsortedSequence(
                (AnyStr(), "value", 2),
                (AnyStr(), "__error__", 'ConnectionError("I\'m not good")'),
                # the write against the previous checkpoint is not saved, as it is
                # produced in a run where only the next checkpoint (the last) is saved
            )
        ),
    )
    if durability == "exit":
        return
    assert checkpoints[2] == CheckpointTuple(
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        checkpoint={
            "v": 4,
            "id": AnyStr(),
            "ts": AnyStr(),
            "versions_seen": {"__input__": {}},
            "channel_versions": {
                "__start__": AnyVersion(),
            },
            "channel_values": {"__start__": {"value": 1}},
            "updated_channels": ["__start__"],
        },
        metadata={
            "parents": {},
            "step": -1,
            "source": "input",
        },
        parent_config=None,
        pending_writes=UnsortedSequence(
            (AnyStr(), "value", 1),
            (AnyStr(), "branch:to:one", None),
            (AnyStr(), "branch:to:two", None),
        ),
    )


def test_cond_edge_after_send() -> None:
    class Node:
        def __init__(self, name: str):
            self.name = name
            setattr(self, "__name__", name)

        def __call__(self, state):
            return [self.name]

    def send_for_fun(state):
        return [Send("2", state), Send("2", state)]

    def route_to_three(state) -> Literal["3"]:
        return "3"

    builder = StateGraph(Annotated[list, operator.add])
    builder.add_node(Node("1"))
    builder.add_node(Node("2"))
    builder.add_node(Node("3"))
    builder.add_edge(START, "1")
    builder.add_conditional_edges("1", send_for_fun)
    builder.add_conditional_edges("2", route_to_three)
    graph = builder.compile()
    assert graph.invoke(["0"]) == ["0", "1", "2", "2", "3"]


def test_concurrent_emit_sends() -> None:
    class Node:
        def __init__(self, name: str):
            self.name = name
            setattr(self, "__name__", name)

        def __call__(self, state):
            return (
                [self.name]
                if isinstance(state, list)
                else ["|".join((self.name, str(state)))]
            )

    def send_for_fun(state):
        return [Send("2", 1), Send("2", 2), "3.1"]

    def send_for_profit(state):
        return [Send("2", 3), Send("2", 4)]

    def route_to_three(state) -> Literal["3"]:
        return "3"

    builder = StateGraph(Annotated[list, operator.add])
    builder.add_node(Node("1"))
    builder.add_node(Node("1.1"))
    builder.add_node(Node("2"))
    builder.add_node(Node("3"))
    builder.add_node(Node("3.1"))
    builder.add_edge(START, "1")
    builder.add_edge(START, "1.1")
    builder.add_conditional_edges("1", send_for_fun)
    builder.add_conditional_edges("1.1", send_for_profit)
    builder.add_conditional_edges("2", route_to_three)
    graph = builder.compile()
    assert graph.invoke(["0"]) == [
        "0",
        "1",
        "1.1",
        "3.1",
        "2|1",
        "2|2",
        "2|3",
        "2|4",
        "3",
    ]


def test_send_sequences() -> None:
    class Node:
        def __init__(self, name: str):
            self.name = name
            setattr(self, "__name__", name)

        def __call__(self, state):
            update = (
                [self.name]
                if isinstance(state, list)
                else ["|".join((self.name, str(state)))]
            )
            if isinstance(state, Command):
                return [state, Command(update=update)]
            else:
                return update

    def send_for_fun(state):
        return [
            Send("2", Command(goto=Send("2", 3))),
            Send("2", Command(goto=Send("2", 4))),
            "3.1",
        ]

    def route_to_three(state) -> Literal["3"]:
        return "3"

    builder = StateGraph(Annotated[list, operator.add])
    builder.add_node(Node("1"))
    builder.add_node(Node("2"))
    builder.add_node(Node("3"))
    builder.add_node(Node("3.1"))
    builder.add_edge(START, "1")
    builder.add_conditional_edges("1", send_for_fun)
    builder.add_conditional_edges("2", route_to_three)
    graph = builder.compile()
    assert graph.invoke(["0"]) == [
        "0",
        "1",
        "3.1",
        "2|Command(goto=Send(node='2', arg=3))",
        "2|Command(goto=Send(node='2', arg=4))",
        "3",
        "2|3",
        "2|4",
        "3",
    ]


def test_imp_task(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    mapper_calls = 0

    class Context(TypedDict):
        model: str

    @task()
    def mapper(input: int) -> str:
        nonlocal mapper_calls
        mapper_calls += 1
        time.sleep(input / 100)
        return str(input) * 2

    @entrypoint(checkpointer=sync_checkpointer, context_schema=Context)
    def graph(input: list[int]) -> list[str]:
        futures = [mapper(i) for i in input]
        mapped = [f.result() for f in futures]
        answer = interrupt("question")
        return [m + answer for m in mapped]

    assert graph.get_input_jsonschema() == {
        "type": "array",
        "items": {"type": "integer"},
        "title": "LangGraphInput",
    }
    assert graph.get_output_jsonschema() == {
        "type": "array",
        "items": {"type": "string"},
        "title": "LangGraphOutput",
    }
    assert graph.get_context_jsonschema() == {
        "properties": {"model": {"title": "Model", "type": "string"}},
        "required": ["model"],
        "title": "Context",
        "type": "object",
    }

    thread1 = {"configurable": {"thread_id": "1"}}
    assert [*graph.stream([0, 1], thread1, durability=durability)] == [
        {"mapper": "00"},
        {"mapper": "11"},
        {
            "__interrupt__": (
                Interrupt(
                    value="question",
                    id=AnyStr(),
                ),
            )
        },
    ]
    assert mapper_calls == 2

    assert graph.invoke(Command(resume="answer"), thread1, durability=durability) == [
        "00answer",
        "11answer",
    ]
    assert mapper_calls == 2


def test_imp_nested(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    def mynode(input: list[str]) -> list[str]:
        return [it + "a" for it in input]

    builder = StateGraph(list[str])
    builder.add_node(mynode)
    builder.add_edge(START, "mynode")
    add_a = builder.compile()

    @task
    def submapper(input: int) -> str:
        time.sleep(input / 100)
        return str(input)

    @task()
    def mapper(input: int) -> str:
        sub = submapper(input)
        time.sleep(input / 100)
        return sub.result() * 2

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(input: list[int]) -> list[str]:
        futures = [mapper(i) for i in input]
        mapped = [f.result() for f in futures]
        answer = interrupt("question")
        final = [m + answer for m in mapped]
        return add_a.invoke(final)

    assert graph.get_input_jsonschema() == {
        "type": "array",
        "items": {"type": "integer"},
        "title": "LangGraphInput",
    }
    assert graph.get_output_jsonschema() == {
        "type": "array",
        "items": {"type": "string"},
        "title": "LangGraphOutput",
    }

    thread1 = {"configurable": {"thread_id": "1"}}
    assert [*graph.stream([0, 1], thread1, durability=durability)] == [
        {"submapper": "0"},
        {"mapper": "00"},
        {"submapper": "1"},
        {"mapper": "11"},
        {
            "__interrupt__": (
                Interrupt(
                    value="question",
                    id=AnyStr(),
                ),
            )
        },
    ]

    assert graph.invoke(Command(resume="answer"), thread1, durability=durability) == [
        "00answera",
        "11answera",
    ]


def test_imp_stream_order(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    @task()
    def foo(state: dict) -> tuple:
        return state["a"] + "foo", "bar"

    @task
    def bar(a: str, b: str, c: str | None = None) -> dict:
        return {"a": a + b, "c": (c or "") + "bark"}

    @task
    def baz(state: dict) -> dict:
        return {"a": state["a"] + "baz", "c": "something else"}

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(state: dict) -> dict:
        fut_foo = foo(state)
        fut_bar = bar(*fut_foo.result())
        fut_baz = baz(fut_bar.result())
        return fut_baz.result()

    thread1 = {"configurable": {"thread_id": "1"}}
    assert [c for c in graph.stream({"a": "0"}, thread1, durability=durability)] == [
        {
            "foo": (
                "0foo",
                "bar",
            )
        },
        {"bar": {"a": "0foobar", "c": "bark"}},
        {"baz": {"a": "0foobarbaz", "c": "something else"}},
        {"graph": {"a": "0foobarbaz", "c": "something else"}},
    ]

    assert graph.get_state(thread1).values == {"a": "0foobarbaz", "c": "something else"}


def test_invoke_checkpoint_three(
    mocker: MockerFixture, sync_checkpointer: BaseCheckpointSaver
) -> None:
    adder = mocker.Mock(side_effect=lambda x: x["total"] + x["input"])

    def raise_if_above_10(input: int) -> int:
        if input > 10:
            raise ValueError("Input is too large")
        return input

    one = (
        NodeBuilder()
        .subscribe_to("input")
        .read_from("total")
        .do(adder)
        .write_to("output", "total")
        .do(raise_if_above_10)
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
        checkpointer=sync_checkpointer,
    )

    thread_1 = {"configurable": {"thread_id": "1"}}
    # total starts out as 0, so output is 0+2=2
    assert app.invoke(2, thread_1, durability="async") == 2
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 2
    assert state.next == ()
    assert (
        state.config["configurable"]["checkpoint_id"]
        == sync_checkpointer.get(thread_1)["id"]
    )
    # total is now 2, so output is 2+3=5
    assert app.invoke(3, thread_1, durability="async") == 5
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert (
        state.config["configurable"]["checkpoint_id"]
        == sync_checkpointer.get(thread_1)["id"]
    )
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        app.invoke(4, thread_1, durability="async")
    # checkpoint is updated with new input
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert state.next == ("one",)
    """we checkpoint inputs and it failed on "one", so the next node is one"""
    # we can recover from error by sending new inputs
    assert app.invoke(2, thread_1, durability="async") == 9
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 16, "total is now 7+9=16"
    assert state.next == ()

    thread_2 = {"configurable": {"thread_id": "2"}}
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert app.invoke(5, thread_2) == 5
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 16
    assert state.next == (), "checkpoint of other thread not touched"
    state = app.get_state(thread_2)
    assert state is not None
    assert state.values.get("total") == 5
    assert state.next == ()

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
        sync_checkpointer.get(thread_1_history[0].config)["id"]
        == thread_1_history[0].config["configurable"]["checkpoint_id"]
    )
    assert (
        sync_checkpointer.get(thread_1_history[1].config)["id"]
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

    one = NodeBuilder().subscribe_only("input").do(add_one).write_to("inbox")
    chain_three = NodeBuilder().subscribe_only("input").do(add_one).write_to("inbox")
    chain_four = (
        NodeBuilder().subscribe_only("inbox").do(add_10_each).write_to("output")
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


def test_invoke_join_then_call_other_pregel(
    mocker: MockerFixture, sync_checkpointer: BaseCheckpointSaver
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])

    inner_app = Pregel(
        nodes={
            "one": NodeBuilder().subscribe_only("input").do(add_one).write_to("output")
        },
        channels={
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    one = NodeBuilder().subscribe_only("input").do(add_10_each).write_to("inbox_one")
    two = (
        NodeBuilder()
        .subscribe_only("inbox_one")
        .do(inner_app.map())
        .write_to("outbox_one")
    )
    chain_three = NodeBuilder().subscribe_only("outbox_one").do(sum).write_to("output")

    app = Pregel(
        nodes={
            "one": one,
            "two": two,
            "chain_three": chain_three,
        },
        channels={
            "inbox_one": Topic(int),
            "outbox_one": LastValue(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    for _ in range(10):
        assert app.invoke([2, 3]) == 27

    with ThreadPoolExecutor() as executor:
        assert [*executor.map(app.invoke, [[2, 3]] * 10)] == [27] * 10

    # add checkpointer
    app.checkpointer = sync_checkpointer
    # subgraph is called twice in the same node, but that works
    assert app.invoke([2, 3], {"configurable": {"thread_id": "1"}}) == 27

    # set inner graph checkpointer NeverCheckpoint
    inner_app.checkpointer = False
    # subgraph still called twice, but checkpointing for inner graph is disabled
    assert app.invoke([2, 3], {"configurable": {"thread_id": "1"}}) == 27


def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = (
        NodeBuilder().subscribe_only("input").do(add_one).write_to("output", "between")
    )
    two = NodeBuilder().subscribe_only("between").do(add_one).write_to("output")

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
    one = NodeBuilder().subscribe_only("input").do(add_one).write_to("between")
    two = NodeBuilder().subscribe_only("between").do(add_one)

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

    one = NodeBuilder().subscribe_only("between").do(add_one).write_to("output")
    two = NodeBuilder().subscribe_only("between").do(add_one)

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
    workflow.set_conditional_entry_point(continue_to_weather, path_map=["get_weather"])

    app = workflow.compile()

    assert json.dumps(app.get_input_jsonschema()) == snapshot
    assert json.dumps(app.get_output_jsonschema()) == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke({"locations": ["sf", "nyc"]}, debug=True) == {
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

    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot


def test_state_graph_w_config_inherited_state_keys(snapshot: SnapshotAssertion) -> None:
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import tool

    class BaseState(TypedDict):
        input: str
        agent_outcome: AgentAction | AgentFinish | None

    class AgentState(BaseState, total=False):
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    assert get_type_hints(AgentState).keys() == {
        "input",
        "agent_outcome",
        "intermediate_steps",
    }

    class Context(TypedDict, total=False):
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

    def agent_parser(input: str) -> dict[str, AgentAction | AgentFinish]:
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

    agent = prompt | llm | agent_parser

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
    builder = StateGraph(AgentState, Context)

    builder.add_node("agent", agent)
    builder.add_node("tools", execute_tools)

    builder.set_entry_point("agent")

    builder.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    builder.add_edge("tools", "agent")

    app = builder.compile()

    assert json.dumps(app.get_context_jsonschema()) == snapshot
    assert json.dumps(app.get_input_jsonschema()) == snapshot
    assert json.dumps(app.get_output_jsonschema()) == snapshot

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

    assert json.dumps(app.get_input_jsonschema()) == snapshot
    assert json.dumps(app.get_output_jsonschema()) == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke({"input": "what is weather in sf"}) == {
        "input": "what is weather in sf",
        "output": "what is weather in sf->right",
        "steps": [],
    }

    assert [*app.stream({"input": "what is weather in sf"})] == [
        {"right": {"output": "what is weather in sf->right"}},
    ]


def test_in_one_fan_out_state_graph_waiting_edge(
    snapshot: SnapshotAssertion, sync_checkpointer: BaseCheckpointSaver
) -> None:
    def sorted_add(x: list[str], y: list[str] | list[tuple[str, str]]) -> list[str]:
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

    if isinstance(sync_checkpointer, InMemorySaver):
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

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
        checkpointer=sync_checkpointer,
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
        checkpointer=sync_checkpointer,
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
    expected_parent_config = list(app_w_interrupt.checkpointer.list(config, limit=2))[
        -1
    ].config
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
        },
        parent_config=expected_parent_config,
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config, debug=1)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4,doc5"}},
    ]


@pytest.mark.parametrize("use_waiting_edge", (True, False))
def test_in_one_fan_out_state_graph_defer_node(
    snapshot: SnapshotAssertion,
    sync_checkpointer: BaseCheckpointSaver,
    use_waiting_edge: bool,
) -> None:
    def sorted_add(x: list[str], y: list[str] | list[tuple[str, str]]) -> list[str]:
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
    workflow.add_node(qa, defer=True)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "retriever_one")
    workflow.add_edge("retriever_one", "analyzer_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    if use_waiting_edge:
        workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    else:
        workflow.add_edge("retriever_one", "qa")
        workflow.add_edge("retriever_two", "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    if isinstance(sync_checkpointer, InMemorySaver):
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    assert [*app.stream({"query": "what is weather in sf"}, stream_mode="debug")] == [
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": AnyStr(),
                "name": "rewrite_query",
                "input": {"query": "what is weather in sf", "docs": []},
                "triggers": ("branch:to:rewrite_query",),
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": AnyStr(),
                "name": "rewrite_query",
                "error": None,
                "result": {
                    "query": "query: what is weather in sf",
                },
                "interrupts": [],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {
                "id": AnyStr(),
                "name": "retriever_one",
                "input": {"query": "query: what is weather in sf", "docs": []},
                "triggers": ("branch:to:retriever_one",),
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {
                "id": AnyStr(),
                "name": "retriever_two",
                "input": {"query": "query: what is weather in sf", "docs": []},
                "triggers": ("branch:to:retriever_two",),
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {
                "id": AnyStr(),
                "name": "retriever_one",
                "error": None,
                "result": {
                    "docs": ["doc1", "doc2"],
                },
                "interrupts": [],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {
                "id": AnyStr(),
                "name": "retriever_two",
                "error": None,
                "result": {
                    "docs": ["doc3", "doc4"],
                },
                "interrupts": [],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 3,
            "payload": {
                "id": AnyStr(),
                "name": "analyzer_one",
                "input": {
                    "query": "query: what is weather in sf",
                    "docs": ["doc1", "doc2", "doc3", "doc4"],
                },
                "triggers": ("branch:to:analyzer_one",),
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 3,
            "payload": {
                "id": AnyStr(),
                "name": "analyzer_one",
                "error": None,
                "result": {
                    "query": "analyzed: query: what is weather in sf",
                },
                "interrupts": [],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 4,
            "payload": {
                "id": AnyStr(),
                "name": "qa",
                "input": {
                    "query": "analyzed: query: what is weather in sf",
                    "docs": ["doc1", "doc2", "doc3", "doc4"],
                },
                "triggers": ("branch:to:qa", "join:retriever_one+retriever_two:qa")
                if use_waiting_edge
                else ("branch:to:qa",),
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 4,
            "payload": {
                "id": AnyStr(),
                "name": "qa",
                "error": None,
                "result": {
                    "answer": "doc1,doc2,doc3,doc4",
                },
                "interrupts": [],
            },
        },
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_after=["analyzer_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"__interrupt__": ()},
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_before=["qa"],
    )
    config = {"configurable": {"thread_id": "2"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"__interrupt__": ()},
    ]

    app_w_interrupt.update_state(config, {"docs": ["doc5"]})
    expected_parent_config = list(app_w_interrupt.checkpointer.list(config, limit=2))[
        -1
    ].config
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
        },
        parent_config=expected_parent_config,
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config, debug=1)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4,doc5"}},
    ]


def test_in_one_fan_out_state_graph_waiting_edge_via_branch(
    snapshot: SnapshotAssertion, sync_checkpointer: BaseCheckpointSaver
) -> None:
    def sorted_add(x: list[str], y: list[str] | list[tuple[str, str]]) -> list[str]:
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

    if isinstance(sync_checkpointer, InMemorySaver):
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke({"query": "what is weather in sf"}, debug=True) == {
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
        checkpointer=sync_checkpointer,
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


def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic2(
    snapshot: SnapshotAssertion,
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    def sorted_add(x: list[str], y: list[str] | list[tuple[str, str]]) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class InnerObject(BaseModel):
        yo: int

    class State(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        query: str
        inner: Annotated[InnerObject, lambda x, y: y]
        answer: str | None = None
        docs: Annotated[list[str], sorted_add]

    class StateUpdate(BaseModel):
        query: str | None = None
        answer: str | None = None
        docs: list[str] | None = None

    class UpdateDocs34(BaseModel):
        docs: list[str] = Field(default_factory=lambda: ["doc3", "doc4"])

    class Input(BaseModel):
        query: str
        inner: InnerObject

    class Output(BaseModel):
        answer: str
        docs: list[str]

    def rewrite_query(data: State) -> State:
        assert isinstance(data.inner, InnerObject)
        return {"query": f"query: {data.query}"}

    def analyzer_one(data: State) -> State:
        assert isinstance(data.inner, InnerObject)
        return StateUpdate(query=f"analyzed: {data.query}")

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)
        return UpdateDocs34()

    def qa(data: State) -> State:
        return {"answer": ",".join(data.docs)}

    def decider(data: State) -> str:
        assert isinstance(data, State)
        return "retriever_two"

    workflow = StateGraph(State, input_schema=Input, output_schema=Output)

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

    if isinstance(sync_checkpointer, InMemorySaver):
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
        assert app.get_input_jsonschema() == snapshot
        assert app.get_output_jsonschema() == snapshot

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
        checkpointer=sync_checkpointer,
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


def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic_input(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    def sorted_add(x: list[str], y: list[str] | list[tuple[str, str]]) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class InnerObject(BaseModel):
        yo: int

    class QueryModel(BaseModel):
        query: str

    class State(QueryModel):
        inner: InnerObject
        answer: str | None = None
        docs: Annotated[list[str], sorted_add]

    class StateUpdate(BaseModel):
        query: str | None = None
        answer: str | None = None
        docs: list[str] | None = None

    class Input(QueryModel):
        inner: InnerObject

    class Output(BaseModel):
        answer: str
        docs: list[str]

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

    workflow = StateGraph(State, input_schema=Input, output_schema=Output)

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

    assert app.invoke(
        Input(query="what is weather in sf", inner=InnerObject(yo=1))
    ) == {
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [
        *app.stream(Input(query="what is weather in sf", inner=InnerObject(yo=1)))
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        for c in app_w_interrupt.stream(
            Input(query="what is weather in sf", inner=InnerObject(yo=1)), config
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


def test_in_one_fan_out_state_graph_waiting_edge_plus_regular(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    def sorted_add(x: list[str], y: list[str] | list[tuple[str, str]]) -> list[str]:
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

    assert [*app.stream({"query": "what is weather in sf"})] in (
        [
            {"rewrite_query": {"query": "query: what is weather in sf"}},
            {"qa": {"answer": ""}},
            {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
            {"retriever_two": {"docs": ["doc3", "doc4"]}},
            {"retriever_one": {"docs": ["doc1", "doc2"]}},
            {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        ],
        [
            {"rewrite_query": {"query": "query: what is weather in sf"}},
            {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
            {"qa": {"answer": ""}},
            {"retriever_two": {"docs": ["doc3", "doc4"]}},
            {"retriever_one": {"docs": ["doc1", "doc2"]}},
            {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        ],
    )

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] in (
        [
            {"rewrite_query": {"query": "query: what is weather in sf"}},
            {"qa": {"answer": ""}},
            {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
            {"retriever_two": {"docs": ["doc3", "doc4"]}},
            {"retriever_one": {"docs": ["doc1", "doc2"]}},
            {"__interrupt__": ()},
        ],
        [
            {"rewrite_query": {"query": "query: what is weather in sf"}},
            {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
            {"qa": {"answer": ""}},
            {"retriever_two": {"docs": ["doc3", "doc4"]}},
            {"retriever_one": {"docs": ["doc1", "doc2"]}},
            {"__interrupt__": ()},
        ],
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


@pytest.mark.parametrize("with_cache", [True, False])
def test_in_one_fan_out_state_graph_waiting_edge_multiple(
    with_cache: bool, cache: BaseCache
) -> None:
    def sorted_add(x: list[str], y: list[str] | list[tuple[str, str]]) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    rewrite_query_count = 0

    def rewrite_query(data: State) -> State:
        nonlocal rewrite_query_count
        rewrite_query_count += 1
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

    workflow.add_node(
        "rewrite_query",
        rewrite_query,
        cache_policy=CachePolicy() if with_cache else None,
    )
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

    app = workflow.compile(cache=cache)

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: analyzed: query: what is weather in sf",
        "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
        "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {
            "rewrite_query": {"query": "query: what is weather in sf"},
            "__metadata__": {"cached": True},
        }
        if with_cache
        else {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"decider": None},
        {
            "rewrite_query": {"query": "query: analyzed: query: what is weather in sf"},
            "__metadata__": {"cached": True},
        }
        if with_cache
        else {
            "rewrite_query": {"query": "query: analyzed: query: what is weather in sf"}
        },
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
    assert rewrite_query_count == 2 if with_cache else 4

    # clear the cache
    if with_cache:
        app.clear_cache()

        assert app.invoke({"query": "what is weather in sf"}) == {
            "query": "analyzed: query: analyzed: query: what is weather in sf",
            "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
            "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
        }
        assert rewrite_query_count == 4


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
    def sorted_add(x: list[str], y: list[str] | list[tuple[str, str]]) -> list[str]:
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

    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
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


def test_nested_graph_xray(snapshot: SnapshotAssertion) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    def logic(state: State):
        pass

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two_slow", logic)
    tool_two_graph.add_node("tool_two_fast", logic)
    tool_two_graph.set_conditional_entry_point(
        lambda s: "tool_two_slow" if s["market"] == "DE" else "tool_two_fast",
        ["tool_two_slow", "tool_two_fast"],
    )
    tool_two = tool_two_graph.compile()

    graph = StateGraph(State)
    graph.add_node("tool_one", logic)
    graph.add_node("tool_two", tool_two)
    graph.add_node("tool_three", logic)
    graph.set_conditional_entry_point(
        lambda s: "tool_one", ["tool_one", "tool_two", "tool_three"]
    )
    app = graph.compile()

    assert app.get_graph(xray=True).to_json() == snapshot
    assert app.get_graph(xray=True).draw_mermaid() == snapshot


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

    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
    assert app.get_graph(xray=True).draw_mermaid() == snapshot
    assert app.invoke(
        {"my_key": "my value", "never_called": never_called},
        print_mode=["values", "updates"],
    ) == {
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

    chain = app | RunnablePassthrough()

    assert chain.invoke({"my_key": "my value", "never_called": never_called}) == {
        "my_key": "my value there and back again",
        "never_called": never_called,
    }
    assert [*chain.stream({"my_key": "my value", "never_called": never_called})] == [
        {"inner": {"my_key": "my value there"}},
        {"side": {"my_key": "my value there and back again"}},
    ]


def test_subgraph_checkpoint_true(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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
    app = graph.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "2"}}
    assert [
        c
        for c in app.stream(
            {"my_key": ""}, config, subgraphs=True, durability=durability
        )
    ] == [
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

    checkpoints = list(app.get_state_history(config))
    if durability != "exit":
        assert len(checkpoints) == 4
    else:
        assert len(checkpoints) == 1


def test_subgraph_durability_inherited(durability: Durability) -> None:
    sync_checkpointer = InMemorySaver()

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

    inner_app = inner.compile(checkpointer=sync_checkpointer)
    graph = StateGraph(State)
    graph.add_node("inner", inner_app)
    graph.add_edge(START, "inner")
    graph.add_conditional_edges(
        "inner", lambda s: "inner" if s["my_key"].count("there") < 2 else END
    )
    app = graph.compile(checkpointer=sync_checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke({"my_key": ""}, config, subgraphs=True, durability=durability)
    if durability != "exit":
        checkpoints = list(sync_checkpointer.list(config))
        assert len(checkpoints) == 12
    else:
        checkpoints = list(sync_checkpointer.list(config))
        assert len(checkpoints) == 1


def test_subgraph_checkpoint_true_interrupt(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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
    builder.add_node("node_2", node_2)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")

    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    assert graph.invoke({"foo": "foo"}, config, durability=durability) == {
        "foo": "hi! foo",
        "__interrupt__": [
            Interrupt(
                value="Provide baz value",
                id=AnyStr(),
            )
        ],
    }
    assert graph.get_state(config, subgraphs=True).tasks[0].state.values == {
        "bar": "hi! foo"
    }
    assert graph.invoke(Command(resume="baz"), config, durability=durability) == {
        "foo": "hi! foobaz"
    }


def test_stream_subgraphs_during_execution(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
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

    app = graph.compile(checkpointer=sync_checkpointer)

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


def test_stream_buffering_single_node(sync_checkpointer: BaseCheckpointSaver) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def node(state: State, writer: StreamWriter):
        writer("Before sleep")
        time.sleep(0.2)
        writer("After sleep")
        return {"my_key": "got here"}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    graph = builder.compile(checkpointer=sync_checkpointer)

    start = time.perf_counter()
    chunks: list[tuple[float, Any]] = []
    config = {"configurable": {"thread_id": "2"}}
    for c in graph.stream({"my_key": ""}, config, stream_mode="custom"):
        chunks.append((round(time.perf_counter() - start, 1), c))

    assert chunks == [
        (FloatBetween(0.0, 0.1), "Before sleep"),
        (FloatBetween(0.2, 0.3), "After sleep"),
    ]


def test_nested_graph_interrupts_parallel(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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

    app = graph.compile(checkpointer=sync_checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert app.invoke({"my_key": ""}, config, durability=durability) == {
        "my_key": " and parallel",
    }

    assert app.invoke(None, config, durability=durability) == {
        "my_key": "got here and there and parallel and back again",
    }

    # below combo of assertions is asserting two things
    # - outer_1 finishes before inner interrupts (because we see its output in stream, which only happens after node finishes)
    # - the writes of outer are persisted in 1st call and used in 2nd call, ie outer isn't called again (because we dont see outer_1 output again in 2nd stream)
    # test stream updates w/ nested interrupt
    config = {"configurable": {"thread_id": "2"}}
    assert [
        *app.stream({"my_key": ""}, config, subgraphs=True, durability=durability)
    ] == [
        # we got to parallel node first
        ((), {"outer_1": {"my_key": " and parallel"}}),
        ((AnyStr("inner:"),), {"inner_1": {"my_key": "got here", "my_other_key": ""}}),
        ((), {"__interrupt__": ()}),
    ]
    assert [*app.stream(None, config, durability=durability)] == [
        {"outer_1": {"my_key": " and parallel"}, "__metadata__": {"cached": True}},
        {"inner": {"my_key": "got here and there"}},
        {"outer_2": {"my_key": " and back again"}},
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [
        *app.stream(
            {"my_key": ""},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values", durability=durability)] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # test interrupts BEFORE the parallel node
    app = graph.compile(checkpointer=sync_checkpointer, interrupt_before=["outer_1"])
    config = {"configurable": {"thread_id": "4"}}
    assert [
        *app.stream(
            {"my_key": ""},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [{"my_key": ""}]
    # while we're waiting for the node w/ interrupt inside to finish
    assert [*app.stream(None, config, stream_mode="values", durability=durability)] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values", durability=durability)] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # test interrupts AFTER the parallel node
    app = graph.compile(checkpointer=sync_checkpointer, interrupt_after=["outer_1"])
    config = {"configurable": {"thread_id": "5"}}
    assert [
        *app.stream(
            {"my_key": ""},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values", durability=durability)] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values", durability=durability)] == [
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]


def test_doubly_nested_graph_interrupts(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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

    app = graph.compile(checkpointer=sync_checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert app.invoke({"my_key": "my value"}, config, durability=durability) == {
        "my_key": "hi my value",
    }

    assert app.invoke(None, config, durability=durability) == {
        "my_key": "hi my value here and there and back again",
    }

    # test stream updates w/ nested interrupt
    nodes: list[str] = []
    config = {
        "configurable": {"thread_id": "2", CONFIG_KEY_NODE_FINISHED: nodes.append}
    }
    assert [*app.stream({"my_key": "my value"}, config, durability=durability)] == [
        {"parent_1": {"my_key": "hi my value"}},
        {"__interrupt__": ()},
    ]
    assert nodes == ["parent_1", "grandchild_1"]
    assert [*app.stream(None, config, durability=durability)] == [
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
    assert [
        *app.stream(
            {"my_key": "my value"},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [
        {"my_key": "my value"},
        {"my_key": "hi my value"},
    ]
    assert [*app.stream(None, config, stream_mode="values", durability=durability)] == [
        {"my_key": "hi my value"},
        {"my_key": "hi my value here and there"},
        {"my_key": "hi my value here and there and back again"},
    ]


def test_repeat_condition(snapshot: SnapshotAssertion) -> None:
    class AgentState(TypedDict):
        hello: str

    def router(state: AgentState) -> str:
        return "hmm"

    workflow = StateGraph(AgentState)
    workflow.add_node("Researcher", lambda x: x)
    workflow.add_node("Chart Generator", lambda x: x)
    workflow.add_node("Call Tool", lambda x: x)
    workflow.add_conditional_edges(
        "Researcher",
        router,
        {
            "redo": "Researcher",
            "continue": "Chart Generator",
            "call_tool": "Call Tool",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "Chart Generator",
        router,
        {"continue": "Researcher", "call_tool": "Call Tool", "end": END},
    )
    workflow.add_conditional_edges(
        "Call Tool",
        # Each agent node updates the 'sender' field
        # the tool calling node does not, meaning
        # this edge will route back to the original agent
        # who invoked the tool
        lambda x: x["sender"],
        {
            "Researcher": "Researcher",
            "Chart Generator": "Chart Generator",
        },
    )
    workflow.set_entry_point("Researcher")

    app = workflow.compile()
    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot


def test_checkpoint_metadata(sync_checkpointer: BaseCheckpointSaver) -> None:
    """This test verifies that a run's configurable fields are merged with the
    previous checkpoint config for each step in the run.
    """
    # set up test
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, AnyMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool

    # graph state
    class BaseState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # initialize graph nodes
    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a nice assistant."),
            ("placeholder", "{messages}"),
        ]
    )

    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    },
                ],
            ),
            AIMessage(content="answer"),
        ]
    )

    @traceable(run_type="llm")
    def agent(state: BaseState) -> BaseState:
        formatted = prompt.invoke(state)
        response = model.invoke(formatted)
        return {"messages": response, "usage_metadata": {"total_tokens": 123}}

    def should_continue(data: BaseState) -> str:
        # Logic to decide whether to continue in the loop or exit
        if not data["messages"][-1].tool_calls:
            return "exit"
        else:
            return "continue"

    # define graphs w/ and w/o interrupt
    workflow = StateGraph(BaseState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )
    workflow.add_edge("tools", "agent")

    # graph w/o interrupt
    app = workflow.compile(checkpointer=sync_checkpointer)

    # graph w/ interrupt
    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer, interrupt_before=["tools"]
    )

    # assertions

    # invoke graph w/o interrupt
    assert app.invoke(
        {"messages": ["what is weather in sf"]},
        {
            "configurable": {
                "thread_id": "1",
                "test_config_1": "foo",
                "test_config_2": "bar",
            },
        },
    ) == {
        "messages": [
            _AnyIdHumanMessage(content="what is weather in sf"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_api",
                        "args": {"query": "query"},
                        "id": "tool_call123",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            _AnyIdAIMessage(content="answer"),
        ]
    }

    config = {"configurable": {"thread_id": "1"}}

    # assert that checkpoint metadata contains the run's configurable fields
    chkpnt_metadata_1 = sync_checkpointer.get_tuple(config).metadata
    assert chkpnt_metadata_1["test_config_1"] == "foo"
    assert chkpnt_metadata_1["test_config_2"] == "bar"

    # Verify that all checkpoint metadata have the expected keys. This check
    # is needed because a run may have an arbitrary number of steps depending
    # on how the graph is constructed.
    chkpnt_tuples_1 = sync_checkpointer.list(config)
    for chkpnt_tuple in chkpnt_tuples_1:
        assert chkpnt_tuple.metadata["test_config_1"] == "foo"
        assert chkpnt_tuple.metadata["test_config_2"] == "bar"

    # invoke graph, but interrupt before tool call
    app_w_interrupt.invoke(
        {"messages": ["what is weather in sf"]},
        {
            "configurable": {
                "thread_id": "2",
                "test_config_3": "foo",
                "test_config_4": "bar",
            },
        },
    )

    config = {"configurable": {"thread_id": "2"}}

    # assert that checkpoint metadata contains the run's configurable fields
    chkpnt_metadata_2 = sync_checkpointer.get_tuple(config).metadata
    assert chkpnt_metadata_2["test_config_3"] == "foo"
    assert chkpnt_metadata_2["test_config_4"] == "bar"

    # resume graph execution
    app_w_interrupt.invoke(
        input=None,
        config={
            "configurable": {
                "thread_id": "2",
                "test_config_3": "foo",
                "test_config_4": "bar",
            }
        },
    )

    # assert that checkpoint metadata contains the run's configurable fields
    chkpnt_metadata_3 = sync_checkpointer.get_tuple(config).metadata
    assert chkpnt_metadata_3["test_config_3"] == "foo"
    assert chkpnt_metadata_3["test_config_4"] == "bar"

    # Verify that all checkpoint metadata have the expected keys. This check
    # is needed because a run may have an arbitrary number of steps depending
    # on how the graph is constructed.
    chkpnt_tuples_2 = sync_checkpointer.list(config)
    for chkpnt_tuple in chkpnt_tuples_2:
        assert chkpnt_tuple.metadata["test_config_3"] == "foo"
        assert chkpnt_tuple.metadata["test_config_4"] == "bar"


def test_remove_message_via_state_update(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

    workflow = StateGraph(state_schema=Annotated[list[AnyMessage], add_messages])  # type: ignore[arg-type]
    workflow.add_node(
        "chatbot",
        lambda state: [
            AIMessage(
                content="Hello! How can I help you",
            )
        ],
    )

    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", END)

    app = workflow.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    output = app.invoke([HumanMessage(content="Hi")], config=config)
    app.update_state(config, values=[RemoveMessage(id=output[-1].id)])

    updated_state = app.get_state(config)

    assert len(updated_state.values) == 1
    assert updated_state.values[-1].content == "Hi"

    app.checkpointer.delete_thread(config["configurable"]["thread_id"])

    # Verify that the message was removed from the checkpointer
    assert app.checkpointer.get_tuple(config) is None
    assert [*app.get_state_history(config)] == []


def test_remove_message_from_node():
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

    workflow = StateGraph(state_schema=Annotated[list[AnyMessage], add_messages])  # type: ignore[arg-type]
    workflow.add_node(
        "chatbot",
        lambda state: [
            AIMessage(
                content="Hello!",
            ),
            AIMessage(
                content="How can I help you?",
            ),
        ],
    )
    workflow.add_node("delete_messages", lambda state: [RemoveMessage(id=state[-2].id)])
    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", "delete_messages")
    workflow.add_edge("delete_messages", END)

    app = workflow.compile()
    output = app.invoke([HumanMessage(content="Hi")])
    assert len(output) == 2
    assert output[-1].content == "How can I help you?"


def test_xray_lance(snapshot: SnapshotAssertion):
    from langchain_core.messages import AnyMessage, HumanMessage

    class Analyst(BaseModel):
        affiliation: str = Field(
            description="Primary affiliation of the investment analyst.",
        )
        name: str = Field(
            description="Name of the investment analyst.",
            pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        )
        role: str = Field(
            description="Role of the investment analyst in the context of the topic.",
        )
        description: str = Field(
            description="Description of the investment analyst focus, concerns, and motives.",
        )

        @property
        def persona(self) -> str:
            return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

    class Perspectives(BaseModel):
        analysts: list[Analyst] = Field(
            description="Comprehensive list of investment analysts with their roles and affiliations.",
        )

    class Section(BaseModel):
        section_title: str = Field(..., title="Title of the section")
        context: str = Field(
            ..., title="Provide a clear summary of the focus area that you researched."
        )
        findings: str = Field(
            ...,
            title="Give a clear and detailed overview of your findings based upon the expert interview.",
        )
        thesis: str = Field(
            ...,
            title="Give a clear and specific investment thesis based upon these findings.",
        )

    class InterviewState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        analyst: Analyst
        section: Section

    class ResearchGraphState(TypedDict):
        analysts: list[Analyst]
        topic: str
        max_analysts: int
        sections: list[Section]
        interviews: Annotated[list, operator.add]

    # Conditional edge
    def route_messages(state):
        return "ask_question"

    def generate_question(state):
        return ...

    def generate_answer(state):
        return ...

    # Add nodes and edges
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("answer_question", generate_answer)

    # Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "answer_question")
    interview_builder.add_conditional_edges(
        "answer_question", route_messages, ["ask_question", END]
    )

    # Interview
    interview_graph = interview_builder.compile().with_config(
        run_name="Conduct Interviews"
    )

    # View
    assert interview_graph.get_graph().to_json() == snapshot

    def run_all_interviews(state: ResearchGraphState):
        """Edge to run the interview sub-graph using Send"""
        return [
            Send(
                "conduct_interview",
                {
                    "analyst": Analyst(),
                    "messages": [
                        HumanMessage(
                            content="So you said you were writing an article on ...?"
                        )
                    ],
                },
            )
            for s in state["analysts"]
        ]

    def generate_sections(state: ResearchGraphState):
        return ...

    def generate_analysts(state: ResearchGraphState):
        return ...

    builder = StateGraph(ResearchGraphState)
    builder.add_node("generate_analysts", generate_analysts)
    builder.add_node("conduct_interview", interview_builder.compile())
    builder.add_node("generate_sections", generate_sections)

    builder.add_edge(START, "generate_analysts")
    builder.add_conditional_edges(
        "generate_analysts", run_all_interviews, ["conduct_interview"]
    )
    builder.add_edge("conduct_interview", "generate_sections")
    builder.add_edge("generate_sections", END)

    graph = builder.compile()

    # View
    assert graph.get_graph().to_json() == snapshot
    assert graph.get_graph(xray=1).to_json() == snapshot


def test_channel_values(sync_checkpointer: BaseCheckpointSaver) -> None:
    config = {"configurable": {"thread_id": "1"}}
    chain = NodeBuilder().subscribe_only("input").write_to("output")
    app = Pregel(
        nodes={
            "one": chain,
        },
        channels={
            "ephemeral": EphemeralValue(Any),
            "input": LastValue(int),
            "output": LastValue(int),
        },
        input_channels=["input", "ephemeral"],
        output_channels="output",
        checkpointer=sync_checkpointer,
    )
    app.invoke({"input": 1, "ephemeral": "meow"}, config)
    assert sync_checkpointer.get(config)["channel_values"] == {"input": 1, "output": 1}


def test_xray_issue(snapshot: SnapshotAssertion) -> None:
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def node(name):
        def _node(state: State):
            return {"messages": [("human", f"entered {name} node")]}

        return _node

    parent = StateGraph(State)
    child = StateGraph(State)

    child.add_node("c_one", node("c_one"))
    child.add_node("c_two", node("c_two"))

    child.add_edge("__start__", "c_one")
    child.add_edge("c_two", "c_one")

    child.add_conditional_edges(
        "c_one", lambda x: str(randrange(0, 2)), {"0": "c_two", "1": "__end__"}
    )

    parent.add_node("p_one", node("p_one"))
    parent.add_node("p_two", child.compile())

    parent.add_edge("__start__", "p_one")
    parent.add_edge("p_two", "p_one")

    parent.add_conditional_edges(
        "p_one", lambda x: str(randrange(0, 2)), {"0": "p_two", "1": "__end__"}
    )

    app = parent.compile()

    assert app.get_graph(xray=True).draw_mermaid() == snapshot


def test_xray_bool(snapshot: SnapshotAssertion) -> None:
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def node(name):
        def _node(state: State):
            return {"messages": [("human", f"entered {name} node")]}

        return _node

    grand_parent = StateGraph(State)

    child = StateGraph(State)

    child.add_node("c_one", node("c_one"))
    child.add_node("c_two", node("c_two"))

    child.add_edge("__start__", "c_one")
    child.add_edge("c_two", "c_one")

    child.add_conditional_edges(
        "c_one", lambda x: str(randrange(0, 2)), {"0": "c_two", "1": "__end__"}
    )

    parent = StateGraph(State)
    parent.add_node("p_one", node("p_one"))
    parent.add_node("p_two", child.compile())
    parent.add_edge("__start__", "p_one")
    parent.add_edge("p_two", "p_one")
    parent.add_conditional_edges(
        "p_one", lambda x: str(randrange(0, 2)), {"0": "p_two", "1": "__end__"}
    )

    grand_parent.add_node("gp_one", node("gp_one"))
    grand_parent.add_node("gp_two", parent.compile())
    grand_parent.add_edge("__start__", "gp_one")
    grand_parent.add_edge("gp_two", "gp_one")
    grand_parent.add_conditional_edges(
        "gp_one", lambda x: str(randrange(0, 2)), {"0": "gp_two", "1": "__end__"}
    )

    app = grand_parent.compile()
    assert app.get_graph(xray=True).draw_mermaid() == snapshot


def test_multiple_sinks_subgraphs(snapshot: SnapshotAssertion) -> None:
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node("one", lambda x: x)
    subgraph_builder.add_node("two", lambda x: x)
    subgraph_builder.add_node("three", lambda x: x)
    subgraph_builder.add_edge("__start__", "one")
    subgraph_builder.add_conditional_edges("one", lambda x: "two", ["two", "three"])
    subgraph = subgraph_builder.compile()

    builder = StateGraph(State)
    builder.add_node("uno", lambda x: x)
    builder.add_node("dos", lambda x: x)
    builder.add_node("subgraph", subgraph)
    builder.add_edge("__start__", "uno")
    builder.add_conditional_edges("uno", lambda x: "dos", ["dos", "subgraph"])

    app = builder.compile()
    assert app.get_graph(xray=True).draw_mermaid() == snapshot


def test_store_injected(
    sync_checkpointer: BaseCheckpointSaver, sync_store: BaseStore
) -> None:
    class State(TypedDict):
        count: Annotated[int, operator.add]

    doc_id = str(uuid.uuid4())
    doc = {"some-key": "this-is-a-val"}
    uid = uuid.uuid4().hex
    namespace = (f"foo-{uid}", "bar")
    thread_1 = str(uuid.uuid4())
    thread_2 = str(uuid.uuid4())

    class Node:
        def __init__(self, i: int | None = None):
            self.i = i

        def __call__(self, inputs: State, config: RunnableConfig, store: BaseStore):
            assert isinstance(store, BaseStore)
            store.put(
                (
                    namespace
                    if self.i is not None
                    and config["configurable"]["thread_id"] in (thread_1, thread_2)
                    else (f"foo_{self.i}", "bar")
                ),
                doc_id,
                {
                    **doc,
                    "from_thread": config["configurable"]["thread_id"],
                    "some_val": inputs["count"],
                },
            )
            return {"count": 1}

    builder = StateGraph(State)
    builder.add_node("node", Node())
    builder.add_edge("__start__", "node")
    N = 50
    M = 1

    for i in range(N):
        builder.add_node(f"node_{i}", Node(i))
        builder.add_edge("__start__", f"node_{i}")

    graph = builder.compile(store=sync_store, checkpointer=sync_checkpointer)

    results = graph.batch(
        [{"count": 0}] * M,
        ([{"configurable": {"thread_id": str(uuid.uuid4())}}] * (M - 1))
        + [{"configurable": {"thread_id": thread_1}}],
    )
    result = results[-1]
    assert result == {"count": N + 1}
    returned_doc = sync_store.get(namespace, doc_id).value
    assert returned_doc == {**doc, "from_thread": thread_1, "some_val": 0}
    assert len(sync_store.search(namespace)) == 1
    # Check results after another turn of the same thread
    result = graph.invoke({"count": 0}, {"configurable": {"thread_id": thread_1}})
    assert result == {"count": (N + 1) * 2}
    returned_doc = sync_store.get(namespace, doc_id).value
    assert returned_doc == {**doc, "from_thread": thread_1, "some_val": N + 1}
    assert len(sync_store.search(namespace)) == 1

    result = graph.invoke({"count": 0}, {"configurable": {"thread_id": thread_2}})
    assert result == {"count": N + 1}
    returned_doc = sync_store.get(namespace, doc_id).value
    assert returned_doc == {
        **doc,
        "from_thread": thread_2,
        "some_val": 0,
    }  # Overwrites the whole doc
    assert len(sync_store.search(namespace)) == 1  # still overwriting the same one


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


def test_debug_retry(sync_checkpointer: BaseCheckpointSaver):
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

    graph = builder.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": []}, config=config, durability="async")

    # re-run step: 1
    target_config = next(
        c.parent_config
        for c in sync_checkpointer.list(config)
        if c.metadata["step"] == 1
    )
    update_config = graph.update_state(target_config, values=None)

    events = [
        *graph.stream(
            None, config=update_config, stream_mode="debug", durability="async"
        )
    ]

    checkpoint_events = list(
        reversed([e["payload"] for e in events if e["type"] == "checkpoint"])
    )

    checkpoint_history = {
        c.config["configurable"]["checkpoint_id"]: c
        for c in graph.get_state_history(config)
    }

    def lax_normalize_config(config: dict | None) -> dict | None:
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


def test_debug_subgraphs(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
):
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

    graph = parent.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    events = [
        *graph.stream(
            {"messages": []},
            config=config,
            stream_mode="debug",
            durability=durability,
        )
    ]

    checkpoint_events = list(
        reversed([e["payload"] for e in events if e["type"] == "checkpoint"])
    )
    if durability == "exit":
        checkpoint_events = checkpoint_events[:1]
    checkpoint_history = list(graph.get_state_history(config))

    assert len(checkpoint_events) == len(checkpoint_history)

    def lax_normalize_config(config: dict | None) -> dict | None:
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


def test_debug_nested_subgraphs(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
):
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

    graph = grand_parent.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    events = [
        *graph.stream(
            {"messages": []},
            config=config,
            stream_mode="debug",
            subgraphs=True,
            durability=durability,
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

    def normalize_config(config: dict | None) -> dict | None:
        if config is None:
            return None

        clean_config = {}
        clean_config["thread_id"] = config["configurable"]["thread_id"]
        clean_config["checkpoint_id"] = config["configurable"]["checkpoint_id"]
        clean_config["checkpoint_ns"] = config["configurable"]["checkpoint_ns"]
        if "checkpoint_map" in config["configurable"]:
            clean_config["checkpoint_map"] = config["configurable"]["checkpoint_map"]

        return clean_config

    for checkpoint_events, checkpoint_history, ns in zip(
        stream_ns.values(), history_ns.values(), stream_ns.keys()
    ):
        if durability == "exit":
            checkpoint_events = checkpoint_events[-1:]
            if ns:  # Save no checkpoints for subgraphs when durability="exit"
                assert not checkpoint_history
                continue
        assert len(checkpoint_events) == len(checkpoint_history)
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


def test_runnable_passthrough_node_graph() -> None:
    class State(TypedDict):
        changeme: str

    async def dummy(state):
        return state

    agent = dummy | RunnablePassthrough.assign(prediction=RunnableLambda(lambda x: x))

    graph_builder = StateGraph(State)

    graph_builder.add_node("agent", agent)
    graph_builder.add_edge(START, "agent")

    graph = graph_builder.compile()

    assert graph.get_graph(xray=True).to_json() == graph.get_graph(xray=False).to_json()


@pytest.mark.parametrize("subgraph_persist", [True, False])
def test_parent_command(
    sync_checkpointer: BaseCheckpointSaver, subgraph_persist: bool
) -> None:
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import tool

    @tool(return_direct=True)
    def get_user_name() -> Command:
        """Retrieve user name"""
        return Command(update={"user_name": "Meow"}, graph=Command.PARENT)

    subgraph_builder = StateGraph(MessagesState)
    subgraph_builder.add_node("tool", get_user_name)
    subgraph_builder.add_edge(START, "tool")
    subgraph = subgraph_builder.compile(checkpointer=subgraph_persist)

    class CustomParentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        # this key is not available to the child graph
        user_name: str

    builder = StateGraph(CustomParentState)
    builder.add_node("alice", subgraph)
    builder.add_edge(START, "alice")
    graph = builder.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    assert graph.invoke(
        {"messages": [("user", "get user name")]}, config, durability="exit"
    ) == {
        "messages": [
            _AnyIdHumanMessage(
                content="get user name", additional_kwargs={}, response_metadata={}
            ),
        ],
        "user_name": "Meow",
    }
    assert graph.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(
                    content="get user name", additional_kwargs={}, response_metadata={}
                ),
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
            "step": 1,
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        tasks=(),
        interrupts=(),
    )


def test_interrupt_subgraph(sync_checkpointer: BaseCheckpointSaver):
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
    graph = builder.compile(checkpointer=sync_checkpointer)

    thread1 = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert graph.invoke({"baz": ""}, thread1)
    # Resume with answer
    assert graph.invoke(Command(resume="bar"), thread1)


@pytest.mark.parametrize("resume_style", ["null", "map"])
def test_interrupt_multiple(
    sync_checkpointer: BaseCheckpointSaver, resume_style: Literal["null", "map"]
):
    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def node(s: State) -> State:
        answer = interrupt({"value": 1})
        answer2 = interrupt({"value": 2})
        return {"my_key": answer + " " + answer2}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")

    graph = builder.compile(checkpointer=sync_checkpointer)
    thread1 = {"configurable": {"thread_id": "1"}}

    result = [e for e in graph.stream({"my_key": "DE", "market": "DE"}, thread1)]
    assert result == [
        {
            "__interrupt__": (
                Interrupt(
                    value={"value": 1},
                    id=AnyStr(),
                ),
            )
        }
    ]

    result = [
        event
        for event in graph.stream(
            Command(
                resume="answer 1"
                if resume_style == "null"
                else {result[0]["__interrupt__"][0].id: "answer 1"},
                update={"my_key": " foofoo "},
            ),
            thread1,
        )
    ]
    assert result == [
        {
            "__interrupt__": (
                Interrupt(
                    value={"value": 2},
                    id=AnyStr(),
                ),
            )
        }
    ]

    assert [
        event
        for event in graph.stream(
            Command(
                resume="answer 2"
                if resume_style == "null"
                else {result[0]["__interrupt__"][0].id: "answer 2"}
            ),
            thread1,
            stream_mode="values",
        )
    ] == [
        {"my_key": "DE foofoo "},
        {"my_key": "DE foofoo answer 1 answer 2"},
    ]


def test_interrupt_loop(sync_checkpointer: BaseCheckpointSaver):
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

    graph = builder.compile(checkpointer=sync_checkpointer)
    thread1 = {"configurable": {"thread_id": "1"}}

    assert [e for e in graph.stream({"other": ""}, thread1)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="How old are you?",
                    id=AnyStr(),
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
                    id=AnyStr(),
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
                    id=AnyStr(),
                ),
            )
        }
    ]

    assert [event for event in graph.stream(Command(resume="19"), thread1)] == [
        {"node": {"age": 19}},
    ]


def test_interrupt_functional(
    sync_checkpointer: BaseCheckpointSaver, snapshot: SnapshotAssertion
) -> None:
    @task
    def foo(state: dict) -> dict:
        return {"a": state["a"] + "foo"}

    @task
    def bar(state: dict) -> dict:
        return {"a": state["a"] + "bar", "b": state["b"]}

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(inputs: dict) -> dict:
        fut_foo = foo(inputs)
        value = interrupt("Provide value for bar:")
        bar_input = {**fut_foo.result(), "b": value}
        fut_bar = bar(bar_input)
        return fut_bar.result()

    config = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert graph.invoke({"a": ""}, config) == {
        "__interrupt__": [
            Interrupt(
                value="Provide value for bar:",
                id=AnyStr(),
            )
        ]
    }
    # Resume with an answer
    res = graph.invoke(Command(resume="bar"), config)
    assert res == {"a": "foobar", "b": "bar"}


def test_interrupt_task_functional(
    sync_checkpointer: BaseCheckpointSaver, snapshot: SnapshotAssertion
) -> None:
    @task
    def foo(state: dict) -> dict:
        return {"a": state["a"] + "foo"}

    @task
    def bar(state: dict) -> dict:
        value = interrupt("Provide value for bar:")
        return {"a": state["a"] + value}

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(inputs: dict) -> dict:
        fut_foo = foo(inputs)
        fut_bar = bar(fut_foo.result())
        return fut_bar.result()

    config = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert graph.invoke({"a": ""}, config) == {
        "__interrupt__": [
            Interrupt(
                value="Provide value for bar:",
                id=AnyStr(),
            ),
        ]
    }
    # Resume with an answer
    res = graph.invoke(Command(resume="bar"), config)
    assert res == {"a": "foobar"}

    # Test that we can interrupt the same task multiple times
    config = {"configurable": {"thread_id": "2"}}

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(inputs: dict) -> dict:
        foo_result = foo(inputs).result()
        bar_result = bar(foo_result).result()
        baz_result = bar(bar_result).result()
        return baz_result

    # First run, interrupted at bar
    assert graph.invoke({"a": ""}, config) == {
        "__interrupt__": [
            Interrupt(
                value="Provide value for bar:",
                id=AnyStr(),
            ),
        ]
    }
    # Provide resumes
    graph.invoke(Command(resume="bar"), config)
    assert graph.invoke(Command(resume="baz"), config) == {"a": "foobarbaz"}


def test_root_mixed_return() -> None:
    def my_node(state: list[str]):
        return [Command(update=["a"]), ["b"]]

    graph = StateGraph(Annotated[list[str], operator.add])

    graph.add_node(my_node)
    graph.add_edge(START, "my_node")
    graph = graph.compile()

    assert graph.invoke([]) == ["a", "b"]


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


def test_command_with_static_breakpoints(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test that we can use Command to resume and update with static breakpoints."""

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

    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_before=["node1"])
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Start the graph and interrupt at the first node
    graph.invoke({"foo": "abc"}, config)
    result = graph.invoke(Command(resume="node1"), config)
    assert result == {"foo": "abc|node-1|node-2"}


def test_multistep_plan(sync_checkpointer: BaseCheckpointSaver):
    from langchain_core.messages import AnyMessage

    class State(TypedDict, total=False):
        plan: list[str | list[str]]
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
        return Command(goto="planner", update={"messages": [("human", "step1")]})

    def step2(state: State):
        return Command(goto="planner", update={"messages": [("human", "step2")]})

    def step3(state: State):
        return Command(goto="planner", update={"messages": [("human", "step3")]})

    def step4(state: State):
        return Command(goto="planner", update={"messages": [("human", "step4")]})

    builder = StateGraph(State)
    builder.add_node(planner)
    builder.add_node(step1)
    builder.add_node(step2)
    builder.add_node(step3)
    builder.add_node(step4)
    builder.add_edge(START, "planner")
    graph = builder.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    assert graph.invoke({"messages": [("human", "start")]}, config) == {
        "messages": [
            _AnyIdHumanMessage(content="start"),
            _AnyIdHumanMessage(content="step1"),
            _AnyIdHumanMessage(content="step2"),
            _AnyIdHumanMessage(content="step3"),
            _AnyIdHumanMessage(content="step4"),
        ],
        "plan": [],
    }


def test_command_goto_with_static_breakpoints(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Use Command goto with static breakpoints."""

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

    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_before=["node1"])

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


def test_multiple_interrupt_state_persistence(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test that state is preserved correctly across multiple interrupts."""

    class State(TypedDict):
        steps: Annotated[list[str], operator.add]

    def interruptible_node(state: State):
        first = interrupt("First interrupt")
        second = interrupt("Second interrupt")
        return {"steps": [first, second]}

    builder = StateGraph(State)
    builder.add_node("node", interruptible_node)
    builder.add_edge(START, "node")

    app = builder.compile(checkpointer=sync_checkpointer)
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


def test_checkpoint_recovery(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
):
    """Test recovery from checkpoints after failures."""

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

    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    # First attempt should fail
    with pytest.raises(RuntimeError):
        graph.invoke(
            {"steps": ["start"], "attempt": 1},
            config,
            durability=durability,
        )

    # Verify checkpoint state
    state = graph.get_state(config)
    assert state is not None
    assert state.values == {"steps": ["start"], "attempt": 1}  # input state saved
    assert state.next == ("node1",)  # Should retry failed node
    assert "RuntimeError('Simulated failure')" in state.tasks[0].error

    # Retry with updated attempt count
    result = graph.invoke({"steps": [], "attempt": 2}, config, durability=durability)
    assert result == {"steps": ["start", "node1", "node2"], "attempt": 2}

    # Verify checkpoint history shows both attempts
    history = list(graph.get_state_history(config))
    if durability != "exit":
        assert len(history) == 6  # Initial + failed attempt + successful attempt
    else:
        assert len(history) == 2  # error + success

    # Verify the error was recorded in checkpoint
    failed_checkpoint = next(c for c in history if c.tasks and c.tasks[0].error)
    assert "RuntimeError('Simulated failure')" in failed_checkpoint.tasks[0].error

    # Verify delete leaves it empty
    graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
    assert graph.checkpointer.get_tuple(config) is None
    assert [*graph.get_state_history(config)] == []


def test_multiple_updates_root() -> None:
    def node_a(state):
        return [Command(update="a1"), Command(update="a2")]

    def node_b(state):
        return "b"

    graph = (
        StateGraph(Annotated[str, operator.add])
        .add_sequence([node_a, node_b])
        .add_edge(START, "node_a")
        .compile()
    )

    assert graph.invoke("") == "a1a2b"

    # only streams the last update from node_a
    assert [c for c in graph.stream("", stream_mode="updates")] == [
        {"node_a": ["a1", "a2"]},
        {"node_b": "b"},
    ]


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


def test_falsy_return_from_task(sync_checkpointer: BaseCheckpointSaver):
    """Test with a falsy return from a task."""

    @task
    def falsy_task() -> bool:
        return False

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(state: dict) -> dict:
        """React tool."""
        falsy_task().result()
        interrupt("test")

    configurable = {"configurable": {"thread_id": uuid.uuid4()}}
    assert [
        chunk
        for chunk in graph.stream(
            {"a": 5}, configurable, stream_mode="debug", durability="exit"
        )
    ] == [
        {
            "payload": {
                "config": {
                    "configurable": {
                        "checkpoint_id": AnyStr(),
                        "checkpoint_ns": "",
                        "thread_id": AnyStr(),
                    },
                },
                "metadata": {
                    "parents": {},
                    "source": "input",
                    "step": -1,
                },
                "next": [
                    "graph",
                ],
                "parent_config": None,
                "tasks": [
                    {
                        "id": AnyStr(),
                        "interrupts": (),
                        "name": "graph",
                        "state": None,
                    },
                ],
                "values": None,
            },
            "step": -1,
            "timestamp": AnyStr(),
            "type": "checkpoint",
        },
        {
            "payload": {
                "id": AnyStr(),
                "input": {
                    "a": 5,
                },
                "name": "graph",
                "triggers": ("__start__",),
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task",
        },
        {
            "payload": {
                "id": AnyStr(),
                "input": (
                    (),
                    {},
                ),
                "name": "falsy_task",
                "triggers": ("__pregel_push",),
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task",
        },
        {
            "payload": {
                "error": None,
                "id": AnyStr(),
                "interrupts": [],
                "name": "falsy_task",
                "result": {
                    "__return__": False,
                },
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task_result",
        },
        {
            "payload": {
                "error": None,
                "id": AnyStr(),
                "interrupts": [
                    {
                        "id": AnyStr(),
                        "value": "test",
                    },
                ],
                "name": "graph",
                "result": {},
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task_result",
        },
    ]
    assert [
        c
        for c in graph.stream(
            Command(resume="123"),
            configurable,
            stream_mode="debug",
            durability="exit",
        )
    ] == [
        {
            "payload": {
                "config": {
                    "configurable": {
                        "checkpoint_id": AnyStr(),
                        "checkpoint_ns": "",
                        "thread_id": AnyStr(),
                    },
                },
                "metadata": {
                    "parents": {},
                    "source": "input",
                    "step": -1,
                },
                "next": [
                    "graph",
                ],
                "parent_config": None,
                "tasks": [
                    {
                        "id": AnyStr(),
                        "interrupts": (
                            {
                                "id": AnyStr(),
                                "value": "test",
                            },
                        ),
                        "name": "graph",
                        "state": None,
                    },
                ],
                "values": None,
            },
            "step": -1,
            "timestamp": AnyStr(),
            "type": "checkpoint",
        },
        {
            "payload": {
                "id": AnyStr(),
                "input": {
                    "a": 5,
                },
                "name": "graph",
                "triggers": ("__start__",),
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task",
        },
        {
            "payload": {
                "id": AnyStr(),
                "input": (
                    (),
                    {},
                ),
                "name": "falsy_task",
                "triggers": ("__pregel_push",),
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task",
        },
        {
            "payload": {
                "error": None,
                "id": AnyStr(),
                "interrupts": [],
                "name": "graph",
                "result": {
                    "__end__": None,
                },
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task_result",
        },
        {
            "payload": {
                "config": {
                    "configurable": {
                        "checkpoint_id": AnyStr(),
                        "checkpoint_ns": "",
                        "thread_id": AnyStr(),
                    },
                },
                "metadata": {
                    "parents": {},
                    "source": "loop",
                    "step": 0,
                },
                "next": [],
                "parent_config": None,
                "tasks": [],
                "values": None,
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "checkpoint",
        },
    ]


def test_multiple_interrupts_functional(sync_checkpointer: BaseCheckpointSaver):
    """Test multiple interrupts with functional API."""

    counter = 0

    @task
    def double(x: int) -> int:
        """Increment the counter."""
        nonlocal counter
        counter += 1
        return 2 * x

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(state: dict) -> dict:
        """React tool."""

        values = []

        for idx in [1, 2, 3]:
            values.extend([double(idx).result(), interrupt({"a": "boo"})])

        return {"values": values}

    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke({}, configurable)
    graph.invoke(Command(resume="a"), configurable)
    graph.invoke(Command(resume="b"), configurable)
    result = graph.invoke(Command(resume="c"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 4, "b", 6, "c"],
    }
    assert counter == 3


def test_multiple_interrupts_functional_cache(
    sync_checkpointer: BaseCheckpointSaver, cache: BaseCache
):
    """Test multiple interrupts with functional API."""

    counter = 0

    @task(cache_policy=CachePolicy())
    def double(x: int) -> int:
        """Increment the counter."""
        nonlocal counter
        counter += 1
        return 2 * x

    @entrypoint(checkpointer=sync_checkpointer, cache=cache)
    def graph(state: dict) -> dict:
        """React tool."""

        values = []

        for idx in [1, 1, 2, 2, 3, 3]:
            values.extend([double(idx).result(), interrupt({"a": "boo"})])

        return {"values": values}

    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke({}, configurable)
    graph.invoke(Command(resume="a"), configurable)
    graph.invoke(Command(resume="b"), configurable)
    graph.invoke(Command(resume="c"), configurable)
    graph.invoke(Command(resume="d"), configurable)
    graph.invoke(Command(resume="e"), configurable)
    result = graph.invoke(Command(resume="f"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 2, "b", 4, "c", 4, "d", 6, "e", 6, "f"],
    }
    assert counter == 3

    # should all be cached now
    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke({}, configurable)
    graph.invoke(Command(resume="a"), configurable)
    graph.invoke(Command(resume="b"), configurable)
    graph.invoke(Command(resume="c"), configurable)
    graph.invoke(Command(resume="d"), configurable)
    graph.invoke(Command(resume="e"), configurable)
    result = graph.invoke(Command(resume="f"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 2, "b", 4, "c", 4, "d", 6, "e", 6, "f"],
    }
    assert counter == 3

    # clear cache
    double.clear_cache(cache)

    # should recompute now
    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke({}, configurable)
    graph.invoke(Command(resume="a"), configurable)
    graph.invoke(Command(resume="b"), configurable)
    graph.invoke(Command(resume="c"), configurable)
    graph.invoke(Command(resume="d"), configurable)
    graph.invoke(Command(resume="e"), configurable)
    result = graph.invoke(Command(resume="f"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 2, "b", 4, "c", 4, "d", 6, "e", 6, "f"],
    }
    assert counter == 6


def test_double_interrupt_subgraph(sync_checkpointer: BaseCheckpointSaver) -> None:
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
    subgraph = subgraph_builder.compile(checkpointer=sync_checkpointer)
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    assert [c for c in subgraph.stream({"input": "test"}, thread)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="interrupt node 1",
                    id=AnyStr(),
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
                    id=AnyStr(),
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

    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    parent_agent = (
        StateGraph(AgentState)
        .add_node("invoke_sub_agent", invoke_sub_agent)
        .add_edge(START, "invoke_sub_agent")
        .add_edge("invoke_sub_agent", END)
        .compile(checkpointer=sync_checkpointer)
    )

    assert [c for c in parent_agent.stream({"input": "test"}, thread)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="interrupt node 1",
                    id=AnyStr(),
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
                    id=AnyStr(),
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


def test_multi_resume(sync_checkpointer: BaseCheckpointSaver) -> None:
    class ChildState(TypedDict):
        prompt: str
        human_input: str
        human_inputs: list[str]

    def get_human_input(state: ChildState):
        human_input = interrupt(state["prompt"])

        return {
            "human_input": human_input,
            "human_inputs": [human_input],
        }

    child_graph = (
        StateGraph(ChildState)
        .add_node("get_human_input", get_human_input)
        .add_edge(START, "get_human_input")
        .add_edge("get_human_input", END)
        .compile(checkpointer=sync_checkpointer)
    )

    class ParentState(TypedDict):
        prompts: list[str]
        human_inputs: Annotated[list[str], operator.add]

    def assign_workers(state: ParentState) -> list[Send]:
        return [
            Send(
                "child_graph",
                {"prompt": prompt},
            )
            for prompt in state["prompts"]
        ]

    def cleanup(state: ParentState):
        assert len(state["human_inputs"]) == len(state["prompts"])

    parent_graph = (
        StateGraph(ParentState)
        .add_node("child_graph", child_graph)
        .add_node("cleanup", cleanup)
        .add_conditional_edges(START, assign_workers, ["child_graph"])
        .add_edge("child_graph", "cleanup")
        .add_edge("cleanup", END)
        .compile(checkpointer=sync_checkpointer)
    )

    thread_config: RunnableConfig = {
        "configurable": {
            "thread_id": uuid.uuid4(),
        },
    }

    prompts = ["a", "b", "c", "d", "e"]

    events = parent_graph.invoke(
        {"prompts": prompts}, thread_config, stream_mode="values"
    )

    assert len(events["__interrupt__"]) == len(prompts)
    interrupt_values = {i.value for i in events["__interrupt__"]}
    assert interrupt_values == set(prompts)

    resume_map: dict[str, str] = {
        i.id: f"human input for prompt {i.value}"
        for i in parent_graph.get_state(thread_config).interrupts
    }

    result = parent_graph.invoke(Command(resume=resume_map), thread_config)
    assert result == {
        "prompts": prompts,
        "human_inputs": [f"human input for prompt {prompt}" for prompt in prompts],
    }


def test_sync_streaming_with_functional_api() -> None:
    """Test streaming with functional API.

    This test verifies that we're able to stream results as they're being generated
    rather than have all the results arrive at once after the graph has completed.

    The time of arrival between the two updates corresponding to the two `slow` tasks
    should be greater than the time delay between the two tasks.
    """

    time_delay = 0.05

    @task()
    def slow() -> dict:
        time.sleep(time_delay)  # Simulate a delay of 10 ms
        return {"tic": time.time()}

    @entrypoint()
    def graph(inputs: dict) -> list:
        first = slow().result()
        second = slow().result()
        return [first, second]

    arrival_times = []

    for chunk in graph.stream({}):
        if "slow" not in chunk:  # We'll just look at the updates from `slow`
            continue
        arrival_times.append(time.time())

    assert len(arrival_times) == 2
    delta = arrival_times[1] - arrival_times[0]
    # Delta cannot be less than 10 ms if it is streaming as results are generated.
    assert delta > time_delay


def test_entrypoint_without_checkpointer() -> None:
    """Test no checkpointer."""
    states = []
    config = {"configurable": {"thread_id": "1"}}

    # Test without previous
    @entrypoint()
    def foo(inputs: Any) -> Any:
        states.append(inputs)
        return inputs

    assert foo.invoke({"a": "1"}, config) == {"a": "1"}

    @entrypoint()
    def foo(inputs: Any, *, previous: Any) -> Any:
        states.append(previous)
        return {"previous": previous, "current": inputs}

    assert foo.invoke({"a": "1"}, config) == {"current": {"a": "1"}, "previous": None}
    assert foo.invoke({"a": "1"}, config) == {"current": {"a": "1"}, "previous": None}


def test_entrypoint_stateful(sync_checkpointer: BaseCheckpointSaver) -> None:
    """Test stateful entrypoint invoke."""

    # Test invoke
    states = []

    @entrypoint(checkpointer=sync_checkpointer)
    def foo(inputs, *, previous: Any) -> Any:
        states.append(previous)
        return {"previous": previous, "current": inputs}

    config = {"configurable": {"thread_id": "1"}}

    assert foo.invoke({"a": "1"}, config) == {"current": {"a": "1"}, "previous": None}
    assert foo.invoke({"a": "2"}, config) == {
        "current": {"a": "2"},
        "previous": {"current": {"a": "1"}, "previous": None},
    }
    assert foo.invoke({"a": "3"}, config) == {
        "current": {"a": "3"},
        "previous": {
            "current": {"a": "2"},
            "previous": {"current": {"a": "1"}, "previous": None},
        },
    }
    assert states == [
        None,
        {"current": {"a": "1"}, "previous": None},
        {"current": {"a": "2"}, "previous": {"current": {"a": "1"}, "previous": None}},
    ]

    # Test stream
    @entrypoint(checkpointer=sync_checkpointer)
    def foo(inputs, *, previous: Any) -> Any:
        return {"previous": previous, "current": inputs}

    config = {"configurable": {"thread_id": "2"}}
    items = [item for item in foo.stream({"a": "1"}, config)]
    assert items == [{"foo": {"current": {"a": "1"}, "previous": None}}]


def test_entrypoint_stateful_update_state(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test stateful entrypoint invoke."""

    # Test invoke
    states = []

    @entrypoint(checkpointer=sync_checkpointer)
    def foo(inputs, *, previous: Any) -> Any:
        states.append(previous)
        return {"previous": previous, "current": inputs}

    config = {"configurable": {"thread_id": "1"}}

    # assert print(foo.input_channels)
    foo.update_state(config, {"a": "-1"})
    assert foo.invoke({"a": "1"}, config) == {
        "current": {"a": "1"},
        "previous": {"a": "-1"},
    }
    assert foo.invoke({"a": "2"}, config) == {
        "current": {"a": "2"},
        "previous": {"current": {"a": "1"}, "previous": {"a": "-1"}},
    }
    assert foo.invoke({"a": "3"}, config) == {
        "current": {"a": "3"},
        "previous": {
            "current": {"a": "2"},
            "previous": {"current": {"a": "1"}, "previous": {"a": "-1"}},
        },
    }

    # update state
    foo.update_state(config, {"a": "3"})

    # Test stream
    assert [item for item in foo.stream({"a": "1"}, config)] == [
        {"foo": {"current": {"a": "1"}, "previous": {"a": "3"}}}
    ]
    assert states == [
        {"a": "-1"},
        {"current": {"a": "1"}, "previous": {"a": "-1"}},
        {
            "current": {"a": "2"},
            "previous": {"current": {"a": "1"}, "previous": {"a": "-1"}},
        },
        {"a": "3"},
    ]


def test_entrypoint_from_sync_generator() -> None:
    """@entrypoint does not support sync generators."""
    previous_return_values = []

    with pytest.raises(NotImplementedError):

        @entrypoint()
        def foo(inputs, previous=None) -> Any:
            previous_return_values.append(previous)
            yield "a"
            yield "b"


def test_multiple_subgraphs(sync_checkpointer: BaseCheckpointSaver) -> None:
    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define the subgraphs
    def add(state):
        return {"result": state["a"] + state["b"]}

    add_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(add)
        .add_edge(START, "add")
        .compile()
    )

    def multiply(state):
        return {"result": state["a"] * state["b"]}

    multiply_subgraph = (
        StateGraph(State, output_schema=Output)
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
        StateGraph(State, output_schema=Output)
        .add_node(call_same_subgraph)
        .add_edge(START, "call_same_subgraph")
        .compile(checkpointer=sync_checkpointer)
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
        StateGraph(State, output_schema=Output)
        .add_node(call_multiple_subgraphs)
        .add_edge(START, "call_multiple_subgraphs")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "2"}}
    assert parent_call_multiple_subgraphs.invoke({"a": 2, "b": 3}, config) == {
        "add_result": 5,
        "multiply_result": 6,
    }


def test_multiple_subgraphs_functional(sync_checkpointer: BaseCheckpointSaver) -> None:
    # Define addition subgraph
    @entrypoint()
    def add(inputs: tuple[int, int]):
        a, b = inputs
        return a + b

    # Define multiplication subgraph using tasks
    @task
    def multiply_task(a, b):
        return a * b

    @entrypoint()
    def multiply(inputs: tuple[int, int]):
        return multiply_task(*inputs).result()

    # Test calling the same subgraph multiple times
    @task
    def call_same_subgraph(a, b):
        result = add.invoke([a, b])
        another_result = add.invoke([result, 10])
        return another_result

    @entrypoint(checkpointer=sync_checkpointer)
    def parent_call_same_subgraph(inputs):
        return call_same_subgraph(*inputs).result()

    config = {"configurable": {"thread_id": "1"}}
    assert parent_call_same_subgraph.invoke([2, 3], config) == 15

    # Test calling multiple subgraphs
    @task
    def call_multiple_subgraphs(a, b):
        add_result = add.invoke([a, b])
        multiply_result = multiply.invoke([a, b])
        return [add_result, multiply_result]

    @entrypoint(checkpointer=sync_checkpointer)
    def parent_call_multiple_subgraphs(inputs):
        return call_multiple_subgraphs(*inputs).result()

    config = {"configurable": {"thread_id": "2"}}
    assert parent_call_multiple_subgraphs.invoke([2, 3], config) == [5, 6]


def test_multiple_subgraphs_mixed_entrypoint(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test calling multiple StateGraph subgraphs from an entrypoint."""

    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define the subgraphs
    def add(state):
        return {"result": state["a"] + state["b"]}

    add_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(add)
        .add_edge(START, "add")
        .compile()
    )

    def multiply(state):
        return {"result": state["a"] * state["b"]}

    multiply_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(multiply)
        .add_edge(START, "multiply")
        .compile()
    )

    # Test calling the same subgraph multiple times
    @task
    def call_same_subgraph(a, b):
        result = add_subgraph.invoke({"a": a, "b": b})["result"]
        another_result = add_subgraph.invoke({"a": result, "b": 10})["result"]
        return another_result

    @entrypoint(checkpointer=sync_checkpointer)
    def parent_call_same_subgraph(inputs):
        return call_same_subgraph(*inputs).result()

    config = {"configurable": {"thread_id": "1"}}
    assert parent_call_same_subgraph.invoke([2, 3], config) == 15

    # Test calling multiple subgraphs
    @task
    def call_multiple_subgraphs(a, b):
        add_result = add_subgraph.invoke({"a": a, "b": b})["result"]
        multiply_result = multiply_subgraph.invoke({"a": a, "b": b})["result"]
        return [add_result, multiply_result]

    @entrypoint(checkpointer=sync_checkpointer)
    def parent_call_multiple_subgraphs(inputs):
        return call_multiple_subgraphs(*inputs).result()

    config = {"configurable": {"thread_id": "2"}}
    assert parent_call_multiple_subgraphs.invoke([2, 3], config) == [5, 6]


def test_multiple_subgraphs_mixed_state_graph(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test calling multiple entrypoint "subgraphs" from a StateGraph."""

    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define addition subgraph
    @entrypoint()
    def add(inputs: tuple[int, int]):
        a, b = inputs
        return a + b

    # Define multiplication subgraph using tasks
    @task
    def multiply_task(a, b):
        return a * b

    @entrypoint()
    def multiply(inputs: tuple[int, int]):
        return multiply_task(*inputs).result()

    # Test calling the same subgraph multiple times
    def call_same_subgraph(state):
        result = add.invoke([state["a"], state["b"]])
        another_result = add.invoke([result, 10])
        return {"result": another_result}

    parent_call_same_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(call_same_subgraph)
        .add_edge(START, "call_same_subgraph")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}
    assert parent_call_same_subgraph.invoke({"a": 2, "b": 3}, config) == {"result": 15}

    # Test calling multiple subgraphs
    class Output(TypedDict):
        add_result: int
        multiply_result: int

    def call_multiple_subgraphs(state):
        add_result = add.invoke([state["a"], state["b"]])
        multiply_result = multiply.invoke([state["a"], state["b"]])
        return {
            "add_result": add_result,
            "multiply_result": multiply_result,
        }

    parent_call_multiple_subgraphs = (
        StateGraph(State, output_schema=Output)
        .add_node(call_multiple_subgraphs)
        .add_edge(START, "call_multiple_subgraphs")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "2"}}
    assert parent_call_multiple_subgraphs.invoke({"a": 2, "b": 3}, config) == {
        "add_result": 5,
        "multiply_result": 6,
    }


def test_multiple_subgraphs_checkpointer(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
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
        .compile(checkpointer=sync_checkpointer)
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


def test_entrypoint_output_schema_with_return_and_save() -> None:
    """Test output schema inference with entrypoint.final."""

    # Un-parameterized entrypoint.final is interpreted as entrypoint.final[Any, Any]
    @entrypoint()
    def foo2(inputs, *, previous: Any) -> entrypoint.final:
        return entrypoint.final(value="foo", save=1)

    assert foo2.get_output_jsonschema() == {
        "title": "LangGraphOutput",
    }

    @entrypoint()
    def foo(inputs, *, previous: Any) -> entrypoint.final[str, int]:
        return entrypoint.final(value="foo", save=1)

    assert foo.get_output_jsonschema() == {
        "title": "LangGraphOutput",
        "type": "string",
    }

    with pytest.raises(TypeError):
        # Raise an exception on an improperly parameterized entrypoint.final
        # User is attempting to parameterize in this case, so we'll offer
        # a bit of help if it's not done correctly.
        @entrypoint()
        def foo(inputs, *, previous: Any) -> entrypoint.final[int]:
            return entrypoint.final(value=1, save=1)  # type: ignore


def test_entrypoint_with_return_and_save(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test entrypoint with return and save."""
    previous_ = None

    @entrypoint(checkpointer=sync_checkpointer)
    def foo(msg: str, *, previous: Any) -> entrypoint.final[int, list[str]]:
        nonlocal previous_
        previous_ = previous
        previous = previous or []
        return entrypoint.final(value=len(previous), save=previous + [msg])

    assert foo.get_output_jsonschema() == {
        "title": "LangGraphOutput",
        "type": "integer",
    }

    config = {"configurable": {"thread_id": "1"}}
    assert foo.invoke("hello", config) == 0
    assert previous_ is None
    assert foo.invoke("goodbye", config) == 1
    assert previous_ == ["hello"]
    assert foo.invoke("definitely", config) == 2
    assert previous_ == ["hello", "goodbye"]


def test_overriding_injectable_args_with_tasks(sync_store: BaseStore) -> None:
    """Test overriding injectable args in tasks."""

    @task
    def foo(store: BaseStore, writer: StreamWriter, value: Any) -> None:
        assert store is value
        assert writer is value

    @entrypoint(store=sync_store)
    def main(inputs, store: BaseStore) -> str:
        assert store is not None
        foo(store=None, writer=None, value=None).result()
        foo(store="hello", writer="hello", value="hello").result()
        return "OK"

    assert main.invoke({}) == "OK"


def test_named_tasks_functional() -> None:
    class Foo:
        def foo(self, value: str) -> dict:
            return value + "foo"

    f = Foo()

    # class method task
    foo = task(f.foo, name="custom_foo")
    other_foo = task(f.foo, name="other_foo")

    # regular function task
    @task(name="custom_bar")
    def bar(value: str) -> dict:
        return value + "|bar"

    def baz(update: str, value: str) -> dict:
        return value + f"|{update}"

    # partial function task (unnamed)
    baz_task = task(functools.partial(baz, "baz"))
    # partial function task (named_)
    custom_baz_task = task(functools.partial(baz, "custom_baz"), name="custom_baz")

    class Qux:
        def __call__(self, value: str) -> dict:
            return value + "|qux"

    qux_task = task(Qux(), name="qux")

    @entrypoint()
    def workflow(inputs: dict) -> dict:
        foo_result = foo(inputs).result()
        other_foo(inputs).result()
        fut_bar = bar(foo_result)
        fut_baz = baz_task(fut_bar.result())
        fut_custom_baz = custom_baz_task(fut_baz.result())
        fut_qux = qux_task(fut_custom_baz.result())
        return fut_qux.result()

    assert list(workflow.stream("", stream_mode="updates")) == [
        {"custom_foo": "foo"},
        {"other_foo": "foo"},
        {"custom_bar": "foo|bar"},
        {"baz": "foo|bar|baz"},
        {"custom_baz": "foo|bar|baz|custom_baz"},
        {"qux": "foo|bar|baz|custom_baz|qux"},
        {"workflow": "foo|bar|baz|custom_baz|qux"},
    ]


def test_tags_stream_mode_messages() -> None:
    model = GenericFakeChatModel(messages=iter(["foo"]), tags=["meow"])
    graph = (
        StateGraph(MessagesState)
        .add_node(
            "call_model", lambda state: {"messages": model.invoke(state["messages"])}
        )
        .add_edge(START, "call_model")
        .compile()
    )
    assert list(
        graph.stream(
            {
                "messages": "hi",
            },
            stream_mode="messages",
        )
    ) == [
        (
            _AnyIdAIMessageChunk(content="foo", chunk_position="last"),
            {
                "langgraph_step": 1,
                "langgraph_node": "call_model",
                "langgraph_triggers": ("branch:to:call_model",),
                "langgraph_path": ("__pregel_pull", "call_model"),
                "langgraph_checkpoint_ns": AnyStr("call_model:"),
                "checkpoint_ns": AnyStr("call_model:"),
                "ls_provider": "genericfakechatmodel",
                "ls_model_type": "chat",
                "tags": ["meow"],
            },
        )
    ]


def test_stream_mode_messages_command() -> None:
    from langchain_core.messages import HumanMessage

    def my_node(state):
        return {"messages": HumanMessage(content="foo")}

    def my_other_node(state):
        return Command(update={"messages": HumanMessage(content="bar")})

    def my_last_node(state):
        return [Command(update={"messages": HumanMessage(content="baz")})]

    graph = (
        StateGraph(MessagesState)
        .add_sequence([my_node, my_other_node, my_last_node])
        .add_edge(START, "my_node")
        .compile()
    )
    assert list(
        graph.stream(
            {
                "messages": [],
            },
            stream_mode="messages",
        )
    ) == [
        (
            _AnyIdHumanMessage(content="foo"),
            {
                "langgraph_step": 1,
                "langgraph_node": "my_node",
                "langgraph_triggers": ("branch:to:my_node",),
                "langgraph_path": ("__pregel_pull", "my_node"),
                "langgraph_checkpoint_ns": AnyStr("my_node:"),
            },
        ),
        (
            _AnyIdHumanMessage(content="bar"),
            {
                "langgraph_step": 2,
                "langgraph_node": "my_other_node",
                "langgraph_triggers": ("branch:to:my_other_node",),
                "langgraph_path": ("__pregel_pull", "my_other_node"),
                "langgraph_checkpoint_ns": AnyStr("my_other_node:"),
            },
        ),
        (
            _AnyIdHumanMessage(content="baz"),
            {
                "langgraph_step": 3,
                "langgraph_node": "my_last_node",
                "langgraph_triggers": ("branch:to:my_last_node",),
                "langgraph_path": ("__pregel_pull", "my_last_node"),
                "langgraph_checkpoint_ns": AnyStr("my_last_node:"),
            },
        ),
    ]


def test_node_destinations() -> None:
    class State(TypedDict):
        foo: Annotated[str, operator.add]

    def node_a(state: State):
        value = state["foo"]
        if value == "a":
            goto = "node_b"
        else:
            goto = "node_c"

        return Command(
            update={"foo": value},
            goto=goto,
            graph=Command.PARENT,
        )

    subgraph = StateGraph(State).add_node(node_a).add_edge(START, "node_a").compile()

    # test calling subgraph inside a node function
    def call_subgraph(state: State):
        return subgraph.invoke(state)

    def node_b(state: State):
        return {"foo": "b"}

    def node_c(state: State):
        return {"foo": "c"}

    for subgraph_node in (subgraph, call_subgraph):
        # destinations w/ tuples
        builder = StateGraph(State)
        builder.add_edge(START, "child")
        builder.add_node("child", subgraph_node, destinations=("node_b", "node_c"))
        builder.add_node(node_b)
        builder.add_node(node_c)
        compiled_graph = builder.compile()
        assert compiled_graph.invoke({"foo": ""}) == {"foo": "c"}

        graph = compiled_graph.get_graph()
        assert [
            Edge(source="__start__", target="child", data=None, conditional=False),
            Edge(source="child", target="node_b", data=None, conditional=True),
            Edge(source="child", target="node_c", data=None, conditional=True),
            Edge(source="node_b", target="__end__", data=None, conditional=False),
            Edge(source="node_c", target="__end__", data=None, conditional=False),
        ] == graph.edges

        # destinations w/ dicts
        builder = StateGraph(State)
        builder.add_edge(START, "child")
        builder.add_node(
            "child", subgraph_node, destinations={"node_b": "foo", "node_c": "bar"}
        )
        builder.add_node(node_b)
        builder.add_node(node_c)
        compiled_graph = builder.compile()
        assert compiled_graph.invoke({"foo": ""}) == {"foo": "c"}

        graph = compiled_graph.get_graph()
        assert [
            Edge(source="__start__", target="child", data=None, conditional=False),
            Edge(source="child", target="node_b", data="foo", conditional=True),
            Edge(source="child", target="node_c", data="bar", conditional=True),
            Edge(source="node_b", target="__end__", data=None, conditional=False),
            Edge(source="node_c", target="__end__", data=None, conditional=False),
        ] == graph.edges


def test_pydantic_none_state_update() -> None:
    class State(BaseModel):
        foo: str | None

    def node_a(state: State) -> State:
        return State(foo=None)

    graph = StateGraph(State).add_node(node_a).add_edge(START, "node_a").compile()
    assert graph.invoke({"foo": ""}) == {"foo": None}


def test_pydantic_state_update_command() -> None:
    class State(BaseModel):
        foo: str | None

    def node_a(state: State) -> State:
        return Command(update=State(foo=None))

    graph = StateGraph(State).add_node(node_a).add_edge(START, "node_a").compile()
    assert graph.invoke({"foo": ""}) == {"foo": None}

    class State(BaseModel):
        foo: str | None = None
        bar: str | None = None

    def node_a(state: State):
        return State(foo="foo")

    def node_b(state: State):
        return Command(update=State(bar="bar"))

    builder = StateGraph(State)
    builder.add_node(node_a)
    builder.add_node(node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    graph = builder.compile()

    assert graph.invoke(State()) == {"foo": "foo", "bar": "bar"}


def test_pydantic_state_mutation() -> None:
    class Inner(BaseModel):
        a: int = 0

    class State(BaseModel):
        inner: Inner = Inner()
        outer: int = 0

    def my_node(state: State) -> State:
        state.inner.a = 5
        state.outer = 10
        return state

    graph = StateGraph(State).add_node(my_node).add_edge(START, "my_node").compile()

    assert graph.invoke({"outer": 1}) == {"outer": 10, "inner": Inner(a=5)}

    # test w/ default_factory
    class State(BaseModel):
        inner: Inner = Field(default_factory=Inner)
        outer: int = 0

    def my_node(state: State) -> State:
        state.inner.a = 5
        state.outer = 10
        return state

    graph = StateGraph(State).add_node(my_node).add_edge(START, "my_node").compile()

    assert graph.invoke({"outer": 1}) == {"outer": 10, "inner": Inner(a=5)}


def test_pydantic_state_mutation_command() -> None:
    class Inner(BaseModel):
        a: int = 0

    class State(BaseModel):
        inner: Inner = Inner()
        outer: int = 0

    def my_node(state: State) -> State:
        state.inner.a = 5
        state.outer = 10
        return Command(update=state)

    graph = StateGraph(State).add_node(my_node).add_edge(START, "my_node").compile()

    assert graph.invoke({"outer": 1}) == {"outer": 10, "inner": Inner(a=5)}

    # test w/ default_factory
    class State(BaseModel):
        inner: Inner = Field(default_factory=Inner)
        outer: int = 0

    def my_node(state: State) -> State:
        state.inner.a = 5
        state.outer = 10
        return Command(update=state)

    graph = StateGraph(State).add_node(my_node).add_edge(START, "my_node").compile()

    assert graph.invoke({"outer": 1}) == {"outer": 10, "inner": Inner(a=5)}


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


def test_stream_messages_dedupe_inputs() -> None:
    from langchain_core.messages import AIMessage

    def call_model(state):
        return {"messages": AIMessage("hi", id="1")}

    def route(state):
        return Command(goto="node_2", graph=Command.PARENT)

    subgraph = (
        StateGraph(MessagesState)
        .add_node(call_model)
        .add_node(route)
        .add_edge(START, "call_model")
        .add_edge("call_model", "route")
        .compile()
    )

    graph = (
        StateGraph(MessagesState)
        .add_node("node_1", subgraph)
        .add_node("node_2", lambda state: state)
        .add_edge(START, "node_1")
        .compile()
    )

    chunks = [
        chunk
        for ns, chunk in graph.stream(
            {"messages": "hi"}, stream_mode="messages", subgraphs=True
        )
    ]

    assert len(chunks) == 1
    assert chunks[0][0] == AIMessage("hi", id="1")
    assert chunks[0][1]["langgraph_node"] == "call_model"


def test_stream_messages_dedupe_state(sync_checkpointer: BaseCheckpointSaver) -> None:
    from langchain_core.messages import AIMessage

    to_emit = [AIMessage("bye", id="1"), AIMessage("bye again", id="2")]

    def call_model(state):
        return {"messages": to_emit.pop(0)}

    def route(state):
        return Command(goto="node_2", graph=Command.PARENT)

    subgraph = (
        StateGraph(MessagesState)
        .add_node(call_model)
        .add_node(route)
        .add_edge(START, "call_model")
        .add_edge("call_model", "route")
        .compile()
    )

    graph = (
        StateGraph(MessagesState)
        .add_node("node_1", subgraph)
        .add_node("node_2", lambda state: state)
        .add_edge(START, "node_1")
        .compile(checkpointer=sync_checkpointer)
    )

    thread1 = {"configurable": {"thread_id": "1"}}

    chunks = [
        chunk
        for ns, chunk in graph.stream(
            {"messages": "hi"}, thread1, stream_mode="messages", subgraphs=True
        )
    ]

    assert len(chunks) == 1
    assert chunks[0][0] == AIMessage("bye", id="1")
    assert chunks[0][1]["langgraph_node"] == "call_model"

    chunks = [
        chunk
        for ns, chunk in graph.stream(
            {"messages": "hi again"},
            thread1,
            stream_mode="messages",
            subgraphs=True,
        )
    ]

    assert len(chunks) == 1
    assert chunks[0][0] == AIMessage("bye again", id="2")
    assert chunks[0][1]["langgraph_node"] == "call_model"


def test_interrupt_subgraph_reenter_checkpointer_true(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
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
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    assert parent.invoke({"foo": "", "counter": 0}, config) == {
        "foo": "",
        "counter": 0,
        "__interrupt__": [
            Interrupt(
                value="Provide value",
                id=AnyStr(),
            )
        ],
    }
    assert parent.invoke(Command(resume="bar"), config) == {
        "foo": "subgraph_2",
        "counter": 1,
        "__interrupt__": [
            Interrupt(
                value="Provide value",
                id=AnyStr(),
            )
        ],
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
        "__interrupt__": [
            Interrupt(
                value="Provide value",
                id=AnyStr(),
            )
        ],
    }
    # confirm that we preserve the state values from the previous invocation
    assert bar_values == [None, "barbaz", "quxbaz"]


def test_empty_invoke() -> None:
    def reducer_merge_dicts(
        dict1: dict[Any, Any], dict2: dict[Any, Any]
    ) -> dict[Any, Any]:
        merged = {**dict1, **dict2}
        return merged

    class SimpleGraphState(BaseModel):
        x1: Annotated[list[str], operator.add] = []
        x2: Annotated[dict[str, Any], reducer_merge_dicts] = {}

    def update_x1_1(state: SimpleGraphState):
        print(state)
        return {"x1": ["111"]}

    def update_x1_2(state: SimpleGraphState):
        print(state)
        state.x1.append("222")
        return {"x1": ["222"]}

    def update_x2_1(state: SimpleGraphState):
        print(state)
        return {"x2": {"111": 111}}

    def update_x2_2(state: SimpleGraphState):
        print(state)
        return {"x2": {"222": 222}}

    graph = StateGraph(SimpleGraphState)
    graph.add_node("x1_1_node", update_x1_1)
    graph.add_node("x1_2_node", update_x1_2)
    graph.add_node("x2_1_node", update_x2_1)
    graph.add_node("x2_2_node", update_x2_2)
    graph.add_edge("x1_1_node", "x1_2_node")
    graph.add_edge("x1_2_node", "x2_1_node")
    graph.add_edge("x2_1_node", "x2_2_node")

    graph.add_edge(START, "x1_1_node")
    graph.add_edge("x2_2_node", END)

    compiled = graph.compile()

    assert compiled.invoke(SimpleGraphState()).get("x2") == {
        "111": 111,
        "222": 222,
    }


def test_parallel_interrupts(sync_checkpointer: BaseCheckpointSaver) -> None:
    # --- CHILD GRAPH ---

    class ChildState(BaseModel):
        prompt: str = Field(..., description="What is going to be asked to the user?")
        human_input: str | None = Field(None, description="What the human said")
        human_inputs: Annotated[list[str], operator.add] = Field(
            default_factory=list, description="All of my messages"
        )

    def get_human_input(state: ChildState):
        human_input = interrupt(state.prompt)

        return dict(
            human_input=human_input,  # update child state
            human_inputs=[human_input],  # update parent state
        )

    child_graph_builder = StateGraph(ChildState)
    child_graph_builder.add_node("get_human_input", get_human_input)
    child_graph_builder.add_edge(START, "get_human_input")
    child_graph_builder.add_edge("get_human_input", END)
    child_graph = child_graph_builder.compile()

    # --- PARENT GRAPH ---

    class ParentState(BaseModel):
        prompts: list[str] = Field(
            ..., description="What is going to be asked to the user?"
        )
        human_inputs: Annotated[list[str], operator.add] = Field(
            default_factory=list, description="All of my messages"
        )

    def assign_workers(state: ParentState):
        return [
            Send(
                "child_graph",
                dict(
                    prompt=prompt,
                ),
            )
            for prompt in state.prompts
        ]

    def cleanup(state: ParentState):
        assert len(state.human_inputs) == len(state.prompts)

    parent_graph_builder = StateGraph(ParentState)
    parent_graph_builder.add_node("child_graph", child_graph)
    parent_graph_builder.add_node("cleanup", cleanup)

    parent_graph_builder.add_conditional_edges(START, assign_workers, ["child_graph"])
    parent_graph_builder.add_edge("child_graph", "cleanup")
    parent_graph_builder.add_edge("cleanup", END)

    parent_graph = parent_graph_builder.compile(checkpointer=sync_checkpointer)

    # --- CLIENT INVOCATION ---

    thread_config = dict(
        configurable=dict(
            thread_id=str(uuid.uuid4()),
        )
    )
    current_input = dict(
        prompts=["a", "b"],
    )

    invokes = 0
    events: dict[int, list[dict]] = {}
    while invokes < 10:
        # reset interrupt
        invokes += 1
        events[invokes] = []
        current_interrupts: list[Interrupt] = []

        # start / resume the graph
        for event in parent_graph.stream(
            input=current_input,
            config=thread_config,
            stream_mode="updates",
        ):
            events[invokes].append(event)
            # handle the interrupt
            if "__interrupt__" in event:
                current_interrupts.extend(event["__interrupt__"])
                # assume that it breaks here, because it is an interrupt

        # get human input and resume
        if len(current_interrupts) > 0:
            # we resume one at a time to preserve original test behavior,
            # but we could also resume all at once if we wanted
            # with a single dict mapping of interrupt ids to resume values
            resume = {current_interrupts[0].id: f"Resume #{invokes}"}
            current_input = Command(resume=resume)

        # not more human input required, must be completed
        else:
            break
    else:
        assert False, "Detected infinite loop"

    assert invokes == 3
    assert len(events) == 3

    assert events[1] == UnsortedSequence(
        {
            "__interrupt__": (
                Interrupt(
                    value="a",
                    id=AnyStr(),
                ),
            )
        },
        {
            "__interrupt__": (
                Interrupt(
                    value="b",
                    id=AnyStr(),
                ),
            )
        },
    )
    assert events[2] in (
        UnsortedSequence(
            {
                "__interrupt__": (
                    Interrupt(
                        value="a",
                        id=AnyStr(),
                    ),
                )
            },
            {"child_graph": {"human_inputs": ["Resume #1"]}},
        ),
        UnsortedSequence(
            {
                "__interrupt__": (
                    Interrupt(
                        value="b",
                        id=AnyStr(),
                    ),
                )
            },
            {"child_graph": {"human_inputs": ["Resume #1"]}},
        ),
    )
    assert events[3] == UnsortedSequence(
        {
            "child_graph": {"human_inputs": ["Resume #1"]},
            "__metadata__": {"cached": True},
        },
        {"child_graph": {"human_inputs": ["Resume #2"]}},
        {"cleanup": None},
    )


def test_parallel_interrupts_double(sync_checkpointer: BaseCheckpointSaver) -> None:
    # --- CHILD GRAPH ---

    class ChildState(BaseModel):
        prompt: str = Field(..., description="What is going to be asked to the user?")
        human_input: str | None = Field(None, description="What the human said")
        human_inputs: Annotated[list[str], operator.add] = Field(
            default_factory=list, description="All of my messages"
        )

    def get_human_input(state: ChildState):
        human_input = interrupt(state.prompt)

        return dict(
            human_inputs=[human_input],  # update parent state
        )

    def get_dolphin_input(state: ChildState):
        human_input = interrupt(state.prompt)

        return dict(
            human_inputs=[human_input],  # update parent state
        )

    child_graph_builder = StateGraph(ChildState)
    child_graph_builder.add_node("get_human_input", get_human_input)
    child_graph_builder.add_node("get_dolphin_input", get_dolphin_input)
    child_graph_builder.add_edge(START, "get_human_input")
    child_graph_builder.add_edge(START, "get_dolphin_input")
    child_graph = child_graph_builder.compile()

    # --- PARENT GRAPH ---

    class ParentState(BaseModel):
        prompts: list[str] = Field(
            ..., description="What is going to be asked to the user?"
        )
        human_inputs: Annotated[list[str], operator.add] = Field(
            default_factory=list, description="All of my messages"
        )

    def assign_workers(state: ParentState):
        return [
            Send(
                "child_graph",
                dict(
                    prompt=prompt,
                ),
            )
            for prompt in state.prompts
        ]

    def cleanup(state: ParentState):
        assert len(state.human_inputs) == len(state.prompts) * 2

    parent_graph_builder = StateGraph(ParentState)
    parent_graph_builder.add_node("child_graph", child_graph)
    parent_graph_builder.add_node("cleanup", cleanup)

    parent_graph_builder.add_conditional_edges(START, assign_workers, ["child_graph"])
    parent_graph_builder.add_edge("child_graph", "cleanup")
    parent_graph_builder.add_edge("cleanup", END)

    parent_graph = parent_graph_builder.compile(checkpointer=sync_checkpointer)

    # --- CLIENT INVOCATION ---

    thread_config = dict(
        configurable=dict(
            thread_id=str(uuid.uuid4()),
        )
    )
    current_input = dict(
        prompts=["a", "b"],
    )

    invokes = 0
    events: dict[int, list[dict]] = {}
    while invokes < 10:
        # reset interrupt
        invokes += 1
        events[invokes] = []
        current_interrupts: list[Interrupt] = []

        # start / resume the graph
        for event in parent_graph.stream(
            input=current_input,
            config=thread_config,
            stream_mode="updates",
        ):
            events[invokes].append(event)
            # handle the interrupt
            if "__interrupt__" in event:
                current_interrupts.extend(event["__interrupt__"])
                # assume that it breaks here, because it is an interrupt

        # get human input and resume
        if len(current_interrupts) > 0:
            # we resume one at a time to preserve original test behavior,
            # but we could also resume all at once if we wanted
            # with a single dict mapping of interrupt ids to resume values
            resume = {current_interrupts[0].id: f"Resume #{invokes}"}
            current_input = Command(resume=resume)

        # not more human input required, must be completed
        else:
            break
    else:
        assert False, "Detected infinite loop"

    assert invokes == 5
    assert len(events) == 5


def test_pregel_loop_refcount():
    gc.collect()
    try:
        gc.disable()

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        graph_builder = StateGraph(State)

        def chatbot(state: State):
            return {"messages": [("ai", "HIYA")]}

        graph_builder.add_node("chatbot", chatbot)
        graph_builder.set_entry_point("chatbot")
        graph_builder.set_finish_point("chatbot")
        graph = graph_builder.compile()

        for _ in range(5):
            graph.invoke({"messages": [{"role": "user", "content": "hi"}]})
            assert (
                len(
                    [obj for obj in gc.get_objects() if isinstance(obj, SyncPregelLoop)]
                )
                == 0
            )
            assert (
                len([obj for obj in gc.get_objects() if isinstance(obj, PregelRunner)])
                == 0
            )
    finally:
        gc.enable()


def test_bulk_state_updates(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    class State(TypedDict):
        foo: str
        baz: str

    def node_a(state: State) -> State:
        return {"foo": "bar"}

    def node_b(state: State) -> State:
        return {"baz": "qux"}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # First update with node_a
    graph.bulk_update_state(
        config,
        [
            [
                StateUpdate(values={"foo": "bar"}, as_node="node_a"),
            ]
        ],
    )

    # Then bulk update with both nodes
    graph.bulk_update_state(
        config,
        [
            [
                StateUpdate(values={"foo": "updated"}, as_node="node_a"),
                StateUpdate(values={"baz": "new"}, as_node="node_b"),
            ]
        ],
    )

    state = graph.get_state(config)
    assert state.values == {"foo": "updated", "baz": "new"}

    # Check if there are only two checkpoints
    checkpoints = list(sync_checkpointer.list(config))
    assert len(checkpoints) == 2

    # perform multiple steps at the same time
    config = {"configurable": {"thread_id": "2"}}

    graph.bulk_update_state(
        config,
        [
            [
                StateUpdate(values={"foo": "bar"}, as_node="node_a"),
            ],
            [
                StateUpdate(values={"foo": "updated"}, as_node="node_a"),
                StateUpdate(values={"baz": "new"}, as_node="node_b"),
            ],
        ],
    )

    state = graph.get_state(config)
    assert state.values == {"foo": "updated", "baz": "new"}

    checkpoints = list(sync_checkpointer.list(config))
    assert len(checkpoints) == 2

    # Should raise error if updating without as_node
    with pytest.raises(InvalidUpdateError):
        graph.bulk_update_state(
            config,
            [
                [
                    StateUpdate(values={"foo": "error"}, as_node=None),
                    StateUpdate(values={"bar": "error"}, as_node=None),
                ]
            ],
        )

    # Should raise if no updates are provided
    with pytest.raises(ValueError, match="No supersteps provided"):
        graph.bulk_update_state(config, [])

    # Should raise if no updates are provided
    with pytest.raises(ValueError, match="No updates provided"):
        graph.bulk_update_state(config, [[], []])

    # Should raise if __end__ or __copy__ update is applied in bulk
    with pytest.raises(InvalidUpdateError):
        graph.bulk_update_state(
            config,
            [
                [
                    StateUpdate(values=None, as_node="__end__"),
                    StateUpdate(values=None, as_node="__copy__"),
                ],
            ],
        )


def test_pregel_node_copy() -> None:
    class State(TypedDict):
        foo: str

    def agent(state: State) -> State:
        return {"foo": "agent"}

    def tool(state: State) -> State:
        return {"foo": "tool"}

    graph = (
        StateGraph(State)
        .add_node("agent", agent)
        .add_node("tool", tool)
        .add_edge(START, "agent")
        .add_edge("agent", "tool")
        .compile()
    )

    graph.invoke({"foo": "input"}, {"configurable": {"thread_id": "1"}})
    graph.copy()
    graph.nodes["agent"].copy({})


def test_update_as_input(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    class State(TypedDict):
        foo: str

    def agent(state: State) -> State:
        return {"foo": "agent"}

    def tool(state: State) -> State:
        return {"foo": "tool"}

    graph = (
        StateGraph(State)
        .add_node("agent", agent)
        .add_node("tool", tool)
        .add_edge(START, "agent")
        .add_edge("agent", "tool")
        .compile(checkpointer=sync_checkpointer)
    )

    assert graph.invoke(
        {"foo": "input"},
        {"configurable": {"thread_id": "1"}},
        durability=durability,
    ) == {"foo": "tool"}

    assert graph.invoke(
        {"foo": "input"},
        {"configurable": {"thread_id": "1"}},
        durability=durability,
    ) == {"foo": "tool"}

    def map_snapshot(i: StateSnapshot) -> dict:
        return {
            "values": i.values,
            "next": i.next,
            "step": i.metadata.get("step"),
        }

    history = [
        map_snapshot(s)
        for s in graph.get_state_history({"configurable": {"thread_id": "1"}})
    ]

    graph.bulk_update_state(
        {"configurable": {"thread_id": "2"}},
        [
            # First turn
            [StateUpdate({"foo": "input"}, "__input__")],
            [StateUpdate({"foo": "input"}, "__start__")],
            [StateUpdate({"foo": "agent"}, "agent")],
            [StateUpdate({"foo": "tool"}, "tool")],
            # Second turn
            [StateUpdate({"foo": "input"}, "__input__")],
            [StateUpdate({"foo": "input"}, "__start__")],
            [StateUpdate({"foo": "agent"}, "agent")],
            [StateUpdate({"foo": "tool"}, "tool")],
        ],
    )

    state = graph.get_state({"configurable": {"thread_id": "2"}})
    assert state.values == {"foo": "tool"}

    new_history = [
        map_snapshot(s)
        for s in graph.get_state_history({"configurable": {"thread_id": "2"}})
    ]

    if durability != "exit":
        assert new_history == history
    else:
        assert [new_history[0], new_history[4]] == history


def test_batch_update_as_input(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    class State(TypedDict):
        foo: str
        tasks: Annotated[list[int], operator.add]

    def agent(state: State) -> State:
        return {"foo": "agent"}

    def map(state: State) -> Command["task"]:
        return Command(
            goto=[
                Send("task", {"index": 0}),
                Send("task", {"index": 1}),
                Send("task", {"index": 2}),
            ],
            update={"foo": "map"},
        )

    def task(state: dict) -> State:
        return {"tasks": [state["index"]]}

    graph = (
        StateGraph(State)
        .add_node("agent", agent)
        .add_node("map", map)
        .add_node("task", task)
        .add_edge(START, "agent")
        .add_edge("agent", "map")
        .compile(checkpointer=sync_checkpointer)
    )

    assert graph.invoke(
        {"foo": "input"},
        {"configurable": {"thread_id": "1"}},
        durability=durability,
    ) == {
        "foo": "map",
        "tasks": [0, 1, 2],
    }

    def map_snapshot(i: StateSnapshot) -> dict:
        return {
            "values": i.values,
            "next": i.next,
            "step": i.metadata.get("step"),
            "tasks": [t.name for t in i.tasks],
        }

    history = [
        map_snapshot(s)
        for s in graph.get_state_history({"configurable": {"thread_id": "1"}})
    ]

    graph.bulk_update_state(
        {"configurable": {"thread_id": "2"}},
        [
            [StateUpdate({"foo": "input"}, "__input__")],
            [StateUpdate({"foo": "input"}, "__start__")],
            [StateUpdate({"foo": "agent", "tasks": []}, "agent")],
            [
                StateUpdate(
                    Command(
                        goto=[
                            Send("task", {"index": 0}),
                            Send("task", {"index": 1}),
                            Send("task", {"index": 2}),
                        ],
                        update={"foo": "map"},
                    ),
                    "map",
                )
            ],
            [
                StateUpdate({"tasks": [0]}, "task"),
                StateUpdate({"tasks": [1]}, "task"),
                StateUpdate({"tasks": [2]}, "task"),
            ],
        ],
    )

    state = graph.get_state({"configurable": {"thread_id": "2"}})
    assert state.values == {"foo": "map", "tasks": [0, 1, 2]}

    new_history = [
        map_snapshot(s)
        for s in graph.get_state_history({"configurable": {"thread_id": "2"}})
    ]

    if durability != "exit":
        assert new_history == history
    else:
        assert new_history[:1] == history


def test_migration_graph(snapshot: SnapshotAssertion) -> None:
    class DummyState(BaseModel):
        pass_count: int = 0

    def increment_pass_count(state: DummyState):
        state.pass_count += 1
        return state

    def route_b(state: DummyState):
        if state.pass_count == 0:
            return "X"
        else:
            return "Y"

    migration_graph = StateGraph(DummyState)

    migration_graph.add_node("B", increment_pass_count)
    migration_graph.add_node("C", increment_pass_count)
    migration_graph.add_node("D", increment_pass_count)

    migration_graph.add_edge(START, "B")

    migration_graph.add_conditional_edges(
        "B",
        route_b,
        {
            "X": "C",
            "Y": "D",
        },
    )

    migration_graph.add_edge("D", "B")
    migration_graph.add_edge("C", END)

    app = migration_graph.compile()

    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot


def test_get_graph_loop(snapshot: SnapshotAssertion) -> None:
    class State(TypedDict):
        foo: str

    def human_node(state: State) -> State:
        value = interrupt()
        return {"foo": value}

    def agent_node(state: State) -> State:
        return {"foo": "Hi " + state["foo"]}

    workflow = StateGraph(State)
    workflow.add_node("human", human_node)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "human")
    workflow.add_edge("human", "agent")
    workflow.add_edge("agent", "human")

    app = workflow.compile()
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot


def test_get_graph_self_loop(snapshot: SnapshotAssertion) -> None:
    import random

    subgraph_builder = StateGraph(MessagesState)
    subgraph_builder.add_node("agent", lambda x: x)
    subgraph_builder.add_edge(START, "agent")
    subgraph = subgraph_builder.compile()

    def worker_node(state: MessagesState) -> Command[Literal["worker_node", "__end__"]]:
        subgraph_result = subgraph.invoke(state)

        if random.choice([True, False]):
            next_node_name = "worker_node"
        else:
            next_node_name = END

        return Command(update=subgraph_result, goto=next_node_name)

    self_loop_builder = StateGraph(MessagesState)
    self_loop_builder.add_node("worker_node", worker_node)
    self_loop_builder.add_edge(START, "worker_node")
    self_loop_graph = self_loop_builder.compile()

    assert json.dumps(self_loop_graph.get_graph().to_json(), indent=2) == snapshot
    assert self_loop_graph.get_graph().draw_mermaid(with_styles=False) == snapshot


def test_get_graph_root_channel(snapshot: SnapshotAssertion) -> None:
    child_builder = StateGraph(list)
    child_builder.add_node("child_node", lambda x: x)
    child_builder.add_edge(START, "child_node")
    child_graph = child_builder.compile()

    graph_builder = StateGraph(list)
    graph_builder.add_node("child", child_graph)
    graph_builder.add_edge(START, "child")
    graph = graph_builder.compile()

    assert json.dumps(graph.get_graph().to_json(), indent=2) == snapshot
    assert graph.get_graph().draw_mermaid(with_styles=False) == snapshot


def test_imp_exception(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    @task()
    def my_task(number: int):
        time.sleep(0.1)
        return number * 2

    @task()
    def task_with_exception(number: int):
        time.sleep(0.1)
        raise Exception("This is a test exception")

    @entrypoint(checkpointer=sync_checkpointer)
    def my_workflow(number: int):
        my_task(number).result()
        try:
            task_with_exception(number).result()
        except Exception as e:
            print(f"Exception caught: {e}")
        my_task(number).result()
        return "done"

    thread1 = {"configurable": {"thread_id": "1"}}
    assert my_workflow.invoke(1, thread1) == "done"

    assert [c for c in my_workflow.stream(1, thread1)] == [
        {"my_task": 2},
        {"my_task": 2},
        {"my_workflow": "done"},
    ]


@pytest.mark.parametrize("with_timeout", [False, "inner", "outer", "both"])
@pytest.mark.parametrize("subgraph_persist", [True, False])
def test_parent_command_goto(
    sync_checkpointer: BaseCheckpointSaver, subgraph_persist: bool, with_timeout: bool
) -> None:
    class State(TypedDict):
        dialog_state: Annotated[list[str], operator.add]

    def node_a_child(state):
        return {"dialog_state": ["a_child_state"]}

    def node_b_child(state):
        return Command(
            graph=Command.PARENT,
            goto="node_b_parent",
            update={"dialog_state": ["b_child_state"]},
        )

    sub_builder = StateGraph(State)
    sub_builder.add_node(node_a_child)
    sub_builder.add_node(node_b_child)
    sub_builder.add_edge(START, "node_a_child")
    sub_builder.add_edge("node_a_child", "node_b_child")
    sub_graph = sub_builder.compile(checkpointer=subgraph_persist)
    if with_timeout in ("inner", "both"):
        sub_graph.step_timeout = 1

    def node_b_parent(state):
        return {"dialog_state": ["node_b_parent"]}

    main_builder = StateGraph(State)
    main_builder.add_node(node_b_parent)
    main_builder.add_edge(START, "subgraph_node")
    main_builder.add_node("subgraph_node", sub_graph, destinations=("node_b_parent",))
    main_graph = main_builder.compile(sync_checkpointer, name="parent")
    if with_timeout in ("outer", "both"):
        main_graph.step_timeout = 1

    config = {"configurable": {"thread_id": 1}}

    assert main_graph.invoke(input={"dialog_state": ["init_state"]}, config=config) == {
        "dialog_state": ["init_state", "b_child_state", "node_b_parent"]
    }


@pytest.mark.parametrize("with_timeout", [True, False])
def test_timeout_with_parent_command(
    sync_checkpointer: BaseCheckpointSaver, with_timeout: bool
) -> None:
    """Test that parent commands are properly propagated during timeouts."""

    class State(TypedDict):
        value: str

    def parent_command_node(state: State) -> State:
        time.sleep(0.1)  # Add some delay before raising
        return Command(graph=Command.PARENT, goto="test_cmd", update={"key": "value"})

    builder = StateGraph(State)
    builder.add_node("parent_cmd", parent_command_node)
    builder.set_entry_point("parent_cmd")
    graph = builder.compile(checkpointer=sync_checkpointer)
    if with_timeout:
        graph.step_timeout = 1

    # Should propagate parent command, not timeout
    thread1 = {"configurable": {"thread_id": "1"}}
    with pytest.raises(ParentCommand) as exc_info:
        graph.invoke({"value": "start"}, thread1)
    assert exc_info.value.args[0].goto == "test_cmd"
    assert exc_info.value.args[0].update == {"key": "value"}


def test_fork_and_update_task_results(sync_checkpointer: BaseCheckpointSaver) -> None:
    """Test forking and updating task results with state history."""

    def checkpoint(values: dict[str, Any]):
        return ("checkpoint", {"values": values})

    def task(name: str, result: Any):
        return ("task", {"name": name, "result": result})

    def get_tree(history: list[StateSnapshot]) -> list:
        """Build a tree structure from state history for comparison."""
        if not history:
            return []

        # Build a tree structure similar to renderForks
        node_map: dict[str, dict] = {}
        root_nodes: list[dict] = []

        # Second pass: establish parent-child relationships
        for item in reversed(history):
            checkpoint_id = item.config["configurable"]["checkpoint_id"]
            parent_checkpoint_id = (
                item.parent_config["configurable"]["checkpoint_id"]
                if item.parent_config
                else None
            )
            node_map[checkpoint_id] = {"item": item, "children": []}

            parent = node_map.get(parent_checkpoint_id)
            (parent["children"] if parent else root_nodes).append(
                node_map[checkpoint_id]
            )

        def node_to_tree(node: dict) -> list:
            """Convert a node to tree structure."""
            result = [
                checkpoint(node["item"].values),
            ] + [
                task(task_info.name, task_info.result)
                for task_info in node["item"].tasks
            ]

            if len(node["children"]) > 1:
                branches = [node_to_tree(child) for child in node["children"]]
                return result + [branches]
            elif len(node["children"]) == 1:
                return result + node_to_tree(node["children"][0])
            else:
                return result

        if len(root_nodes) == 1:
            # Process all root nodes
            return node_to_tree(root_nodes[0])

        elif len(root_nodes) > 1:
            # Multiple root nodes - treat as branches
            branches = [node_to_tree(node) for node in root_nodes]
            return branches
        else:
            return []

    class State(TypedDict):
        name: Annotated[str, lambda a, b: " > ".join([a, b]) if a else b]

    # Define the graph with a sequence of nodes
    def one(state: State) -> Command:
        return Command(goto=[Send("two", {})], update={"name": "one"})

    def two(state: State) -> State:
        return {"name": "two"}

    def three(state: State) -> State:
        return {"name": "three"}

    graph = (
        StateGraph(State)
        .add_node("one", one)
        .add_node("two", two)
        .add_node("three", three)
        .add_edge(START, "one")
        .add_edge("one", "two")
        .add_edge("two", "three")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    history: list[StateSnapshot] = []

    # Initial run
    graph.invoke({"name": "start"}, config)
    history = list(graph.get_state_history(config))

    assert get_tree(history) == [
        checkpoint({"name": ""}),
        task("__start__", {"name": "start"}),
        checkpoint({"name": "start"}),
        task("one", {"name": "one"}),
        checkpoint({"name": "start > one"}),
        task("two", {"name": "two"}),
        task("two", {"name": "two"}),
        checkpoint({"name": "start > one > two > two"}),
        task("three", {"name": "three"}),
        checkpoint({"name": "start > one > two > two > three"}),
    ]

    # Update the start state
    graph.invoke(
        None,
        graph.update_state(
            history[4].config,
            values=[StateUpdate(values={"name": "start*"}, as_node="__start__")],
            as_node="__copy__",
        ),
    )

    history = list(graph.get_state_history(config))
    assert get_tree(history) == [
        [
            checkpoint({"name": ""}),
            task("__start__", {"name": "start"}),
            checkpoint({"name": "start"}),
            task("one", {"name": "one"}),
            checkpoint({"name": "start > one"}),
            task("two", {"name": "two"}),
            task("two", {"name": "two"}),
            checkpoint({"name": "start > one > two > two"}),
            task("three", {"name": "three"}),
            checkpoint({"name": "start > one > two > two > three"}),
        ],
        [
            checkpoint({"name": ""}),
            task("__start__", {"name": "start*"}),
            checkpoint({"name": "start*"}),
            task("one", {"name": "one"}),
            checkpoint({"name": "start* > one"}),
            task("two", {"name": "two"}),
            task("two", {"name": "two"}),
            checkpoint({"name": "start* > one > two > two"}),
            task("three", {"name": "three"}),
            checkpoint({"name": "start* > one > two > two > three"}),
        ],
    ]

    # Fork from task "one"
    # Start from the checkpoint that has the task "one"
    assert history[3].values == {"name": "start*"}
    assert len(history[3].tasks) == 1
    assert history[3].tasks[0].name == "one"

    graph.invoke(
        None,
        graph.update_state(
            history[3].config,
            [StateUpdate(values={"name": "one*"}, as_node="one")],
            "__copy__",
        ),
    )

    history = list(graph.get_state_history(config))
    assert get_tree(history) == [
        [
            checkpoint({"name": ""}),
            task("__start__", {"name": "start"}),
            checkpoint({"name": "start"}),
            task("one", {"name": "one"}),
            checkpoint({"name": "start > one"}),
            task("two", {"name": "two"}),
            task("two", {"name": "two"}),
            checkpoint({"name": "start > one > two > two"}),
            task("three", {"name": "three"}),
            checkpoint({"name": "start > one > two > two > three"}),
        ],
        [
            checkpoint({"name": ""}),
            task("__start__", {"name": "start*"}),
            [
                [
                    checkpoint({"name": "start*"}),
                    task("one", {"name": "one"}),
                    checkpoint({"name": "start* > one"}),
                    task("two", {"name": "two"}),
                    task("two", {"name": "two"}),
                    checkpoint({"name": "start* > one > two > two"}),
                    task("three", {"name": "three"}),
                    checkpoint({"name": "start* > one > two > two > three"}),
                ],
                [
                    checkpoint({"name": "start*"}),
                    task("one", {"name": "one*"}),
                    checkpoint({"name": "start* > one*"}),
                    task("two", {"name": "two"}),
                    checkpoint({"name": "start* > one* > two"}),
                    task("three", {"name": "three"}),
                    checkpoint({"name": "start* > one* > two > three"}),
                ],
            ],
        ],
    ]

    config = {"configurable": {"thread_id": "2"}}

    # Initialize the thread once again
    graph.invoke({"name": "start"}, config)
    history = list(graph.get_state_history(config))

    # Fork from task "two"
    # Start from the checkpoint that has the task "two"
    assert history[2].values == {"name": "start > one"}

    graph.invoke(
        None,
        graph.update_state(
            history[2].config,
            [
                StateUpdate(values={"name": "two"}, as_node="two"),
                StateUpdate(values={"name": "two"}, as_node="two"),
            ],
            "__copy__",
        ),
    )

    history = list(graph.get_state_history(config))
    assert get_tree(history) == [
        checkpoint({"name": ""}),
        task("__start__", {"name": "start"}),
        checkpoint({"name": "start"}),
        task("one", {"name": "one"}),
        [
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                checkpoint({"name": "start > one > two > two"}),
                task("three", {"name": "three"}),
                checkpoint({"name": "start > one > two > two > three"}),
            ],
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                checkpoint({"name": "start > one > two > two"}),
                task("three", {"name": "three"}),
                checkpoint({"name": "start > one > two > two > three"}),
            ],
        ],
    ]

    # Fork task three
    assert history[1].values == {"name": "start > one > two > two"}
    assert len(history[1].tasks) == 1
    assert history[1].tasks[0].name == "three"

    graph.invoke(
        None,
        graph.update_state(
            history[1].config,
            [StateUpdate(values={"name": "three*"}, as_node="three")],
            "__copy__",
        ),
    )

    history = list(graph.get_state_history(config))
    assert get_tree(history) == [
        checkpoint({"name": ""}),
        task("__start__", {"name": "start"}),
        checkpoint({"name": "start"}),
        task("one", {"name": "one"}),
        [
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                checkpoint({"name": "start > one > two > two"}),
                task("three", {"name": "three"}),
                checkpoint({"name": "start > one > two > two > three"}),
            ],
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                [
                    [
                        checkpoint({"name": "start > one > two > two"}),
                        task("three", {"name": "three"}),
                        checkpoint({"name": "start > one > two > two > three"}),
                    ],
                    [
                        checkpoint({"name": "start > one > two > two"}),
                        task("three", {"name": "three*"}),
                        checkpoint({"name": "start > one > two > two > three*"}),
                    ],
                ],
            ],
        ],
    ]

    # Regenerate task three
    assert history[3].values == {"name": "start > one > two > two"}
    assert len(history[3].tasks) == 1
    assert history[3].tasks[0].name == "three"

    graph.invoke(None, graph.update_state(history[3].config, None, "__copy__"))

    history = list(graph.get_state_history(config))
    assert get_tree(history) == [
        checkpoint({"name": ""}),
        task("__start__", {"name": "start"}),
        checkpoint({"name": "start"}),
        task("one", {"name": "one"}),
        [
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                checkpoint({"name": "start > one > two > two"}),
                task("three", {"name": "three"}),
                checkpoint({"name": "start > one > two > two > three"}),
            ],
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                [
                    [
                        checkpoint({"name": "start > one > two > two"}),
                        task("three", {"name": "three"}),
                        checkpoint({"name": "start > one > two > two > three"}),
                    ],
                    [
                        checkpoint({"name": "start > one > two > two"}),
                        task("three", {"name": "three*"}),
                        checkpoint({"name": "start > one > two > two > three*"}),
                    ],
                    [
                        checkpoint({"name": "start > one > two > two"}),
                        task("three", {"name": "three"}),
                        checkpoint({"name": "start > one > two > two > three"}),
                    ],
                ],
            ],
        ],
    ]


def test_subgraph_streaming_sync() -> None:
    """Test subgraph streaming when used as a node in sync version"""

    # Create a fake chat model that returns a simple response
    model = GenericFakeChatModel(messages=iter(["The weather is sunny today."]))

    # Create a subgraph that uses the fake chat model
    def call_model_node(state: MessagesState, config: RunnableConfig) -> MessagesState:
        """Node that calls the model with the last message."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        response = model.invoke([("user", last_message)], config)
        return {"messages": [response]}

    # Build the subgraph
    subgraph = StateGraph(MessagesState)
    subgraph.add_node("call_model", call_model_node)
    subgraph.add_edge(START, "call_model")
    compiled_subgraph = subgraph.compile()

    class SomeCustomState(TypedDict):
        last_chunk: NotRequired[str]
        num_chunks: NotRequired[int]

    # Will invoke a subgraph as a function
    def parent_node(state: SomeCustomState, config: RunnableConfig) -> dict:
        """Node that runs the subgraph."""
        msgs = {"messages": [("user", "What is the weather in Tokyo?")]}
        events = []
        for event in compiled_subgraph.stream(msgs, config, stream_mode="messages"):
            events.append(event)
        ai_msg_chunks = [ai_msg_chunk for ai_msg_chunk, _ in events]
        return {
            "last_chunk": ai_msg_chunks[-1],
            "num_chunks": len(ai_msg_chunks),
        }

    # Build the main workflow
    workflow = StateGraph(SomeCustomState)
    workflow.add_node("subgraph", parent_node)
    workflow.add_edge(START, "subgraph")
    compiled_workflow = workflow.compile()

    # Test the basic functionality
    result = compiled_workflow.invoke({})

    assert result["last_chunk"].content == "today."
    assert result["num_chunks"] == 9


def test_get_graph_nonterminal_last_step_source(snapshot: SnapshotAssertion) -> None:
    class State(TypedDict):
        messages: list[str]

    def chatbot_node(state: State) -> State:
        return {"messages": state["messages"] + ["chatbot"]}

    def tools_node(state: State) -> State:
        return {"messages": state["messages"] + ["tools"]}

    def human_node(state: State) -> State:
        return {"messages": state["messages"] + ["human"]}

    def tools_condition(_: State) -> str:
        return "tools"

    def end_condition(_: State) -> str:
        return "chatbot"

    workflow = StateGraph(State)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("human", human_node)

    workflow.add_edge(START, "human")
    workflow.add_edge("tools", "chatbot")

    workflow.add_conditional_edges(
        "chatbot", tools_condition, {"tools": "tools", "human": "human"}
    )
    workflow.add_conditional_edges(
        "human", end_condition, {"chatbot": "chatbot", END: END}
    )

    app = workflow.compile()
    graph = app.get_graph()
    graph_json = graph.to_json()

    assert json.dumps(graph_json, indent=2, sort_keys=True) == snapshot


def test_null_resume_disallowed_with_multiple_interrupts(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    class State(TypedDict):
        text_1: str
        text_2: str

    def human_node_1(state: State):
        value = interrupt({"text_to_revise": state["text_1"]})
        return {"text_1": value}

    def human_node_2(state: State):
        value = interrupt({"text_to_revise": state["text_2"]})
        return {"text_2": value}

    graph_builder = StateGraph(State)
    graph_builder.add_node("human_node_1", human_node_1)
    graph_builder.add_node("human_node_2", human_node_2)

    # Add both nodes in parallel from START
    graph_builder.add_edge(START, "human_node_1")
    graph_builder.add_edge(START, "human_node_2")

    checkpointer = InMemorySaver()
    graph = graph_builder.compile(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    graph.invoke(
        {"text_1": "original text 1", "text_2": "original text 2"}, config=config
    )

    resume_map = {
        i.id: f"resume for prompt: {i.value['text_to_revise']}"
        for i in graph.get_state(config).interrupts
    }
    with pytest.raises(
        RuntimeError,
        match="When there are multiple pending interrupts, you must specify the interrupt id when resuming.",
    ):
        graph.invoke(Command(resume="singular resume"), config=config)

    assert graph.invoke(Command(resume=resume_map), config=config) == {
        "text_1": "resume for prompt: original text 1",
        "text_2": "resume for prompt: original text 2",
    }


def test_interrupt_stream_mode_values():
    """Test that interrupts are surfaced when steam_mode='values'"""

    class State(TypedDict):
        human_input: str

    def human_input_node(state: State) -> Command:
        human_input = interrupt("interrupt")
        return Command(update={"human_input": human_input})

    builder = StateGraph(State)
    builder.add_node(human_input_node)
    builder.add_edge(START, "human_input_node")
    app = builder.compile()

    result = [*app.stream(State(), stream_mode="values")]
    assert "__interrupt__" in result[-1]


def test_supersteps_populate_task_results(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    class State(TypedDict):
        num: int
        text: str

    def double(state: State) -> State:
        return {"num": state["num"] * 2, "text": state["text"] * 2}

    graph = (
        StateGraph(State)
        .add_node("double", double)
        .add_edge(START, "double")
        .add_edge("double", END)
        .compile(checkpointer=sync_checkpointer)
    )

    def first_task_result(history: list[StateSnapshot], node: str) -> Any:
        for s in history:
            for t in s.tasks:
                if t.name == node:
                    return t.result
        return None

    # reference run with invoke
    ref_cfg = {"configurable": {"thread_id": "ref"}}
    graph.invoke({"num": 1, "text": "one"}, ref_cfg)
    ref_history = list(graph.get_state_history(ref_cfg))

    ref_start_result = first_task_result(ref_history, "__start__")
    ref_double_result = first_task_result(ref_history, "double")
    assert ref_start_result == {"num": 1, "text": "one"}
    assert ref_double_result == {"num": 2, "text": "oneone"}

    # using supersteps
    bulk_cfg = {"configurable": {"thread_id": "bulk"}}
    graph.bulk_update_state(
        bulk_cfg,
        [
            [StateUpdate(values={}, as_node="__input__")],
            [StateUpdate(values={"num": 1, "text": "one"}, as_node="__start__")],
            [StateUpdate(values={"num": 2, "text": "oneone"}, as_node="double")],
        ],
    )
    bulk_history = list(graph.get_state_history(bulk_cfg))

    bulk_start_result = first_task_result(bulk_history, "__start__")
    bulk_double_result = first_task_result(bulk_history, "double")

    assert bulk_start_result == ref_start_result == {"num": 1, "text": "one"}
    assert bulk_double_result == ref_double_result == {"num": 2, "text": "oneone"}


def test_multiple_writes_same_channel_from_same_node(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test that a node can write multiple times to the same channel and that writes are ordered, reduced, and reflected in streamed events and state history."""

    class State(TypedDict):
        foo: Annotated[str, lambda a, b: ", ".join([x for x in [a, b] if x])]

    def one(_: State) -> Command:
        return Command(update=[("foo", "one.0"), ("foo", "one.1")])

    def two(_: State) -> State:
        return {"foo": "two"}

    graph = (
        StateGraph(State)
        .add_node("one", one)
        .add_node("two", two)
        .add_edge(START, "one")
        .add_edge("one", "two")
        .add_edge("two", END)
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    events = [
        (ns, ev)
        for ns, ev in graph.stream(
            {"foo": "input"}, config, stream_mode=["updates", "tasks"]
        )
    ]

    assert events == [
        (
            "tasks",
            {
                "id": AnyStr(),
                "name": "one",
                "input": {"foo": "input"},
                "triggers": ("branch:to:one",),
            },
        ),
        ("updates", {"one": [{"foo": "one.0"}, {"foo": "one.1"}]}),
        (
            "tasks",
            {
                "id": AnyStr(),
                "name": "one",
                "error": None,
                "result": {"foo": {"$writes": ["one.0", "one.1"]}},
                "interrupts": [],
            },
        ),
        (
            "tasks",
            {
                "id": AnyStr(),
                "name": "two",
                "input": {"foo": "input, one.0, one.1"},
                "triggers": ("branch:to:two",),
            },
        ),
        ("updates", {"two": {"foo": "two"}}),
        (
            "tasks",
            {
                "id": AnyStr(),
                "name": "two",
                "error": None,
                "result": {"foo": "two"},
                "interrupts": [],
            },
        ),
    ]

    def map_snapshot(s: StateSnapshot) -> dict:
        return {
            "tasks": [{"name": t.name, "result": t.result} for t in s.tasks],
            "values": s.values,
        }

    history = [map_snapshot(s) for s in graph.get_state_history(config)]

    assert history == [
        {
            "tasks": [],
            "values": {"foo": "input, one.0, one.1, two"},
        },
        {
            "tasks": [{"name": "two", "result": {"foo": "two"}}],
            "values": {"foo": "input, one.0, one.1"},
        },
        {
            "tasks": [
                {"name": "one", "result": {"foo": {"$writes": ["one.0", "one.1"]}}}
            ],
            "values": {"foo": "input"},
        },
        {
            "tasks": [{"name": "__start__", "result": {"foo": "input"}}],
            "values": {"foo": ""},
        },
    ]


def test_send_with_untracked_value(sync_checkpointer: BaseCheckpointSaver):
    """Test that Send objects work correctly with untracked values in state."""

    class UnserializableResource:
        def __init__(self, name: str):
            self.name = name
            self.lock = threading.Lock()

    class State(TypedDict):
        messages: Annotated[list[str], operator.add]
        session_resource: Annotated[UnserializableResource, UntrackedValue]

    def setup_node(state: State) -> State:
        resource = UnserializableResource("test_session")
        return {"messages": ["setup complete"], "session_resource": resource}

    def send_to_tool(state: State):
        return [Send("tool_node", state)]

    def tool_node(state: State) -> State:
        resource = state["session_resource"]
        assert isinstance(resource, UnserializableResource)
        assert resource.name == "test_session"

        new_resource = UnserializableResource("new_session")

        return {
            "messages": [f"tool used resource: {resource.name}"],
            "session_resource": new_resource,
        }

    graph = StateGraph(State)
    graph.add_node("setup", setup_node)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "setup")
    graph.add_conditional_edges("setup", send_to_tool)

    app = graph.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    result = app.invoke({}, config)

    assert len(result["messages"]) == 2
    assert result["messages"][0] == "setup complete"
    assert result["messages"][1] == "tool used resource: test_session"
    assert result["session_resource"].name == "new_session"

    state = app.get_state(config)
    assert "session_resource" not in state.values


def test_send_with_untracked_value_overlapping_keys(
    sync_checkpointer: BaseCheckpointSaver,
):
    """Test that Send objects work correctly with untracked values in state."""

    class State(TypedDict):
        dictionary: dict
        session_resource: Annotated[str, UntrackedValue]

    def setup_node(state: State) -> State:
        return {}

    def send_to_tool(state: State):
        return [
            Send(
                "tool_node",
                {
                    "dictionary": {"session_resource": "legal_value"},
                    "session_resource": "illegal_value",
                },
            )
        ]

    def tool_node(state: State) -> State:
        print(f"STATE: {state}")
        assert state["dictionary"] == {"session_resource": "legal_value"}
        assert state["session_resource"] == "illegal_value"

        return {
            "dictionary": state["dictionary"],
            "session_resource": "new_illegal_value",
        }

    graph = StateGraph(State)
    graph.add_node("setup", setup_node)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "setup")
    graph.add_conditional_edges("setup", send_to_tool)

    app = graph.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    result = app.invoke({}, config)

    assert result["session_resource"] == "new_illegal_value"
    state = app.get_state(config)
    assert "session_resource" not in state.values
    assert state.values.get("dictionary") == {"session_resource": "legal_value"}


@pytest.mark.parametrize("as_json", [False, True])
def test_overwrite_sequential(
    sync_checkpointer: BaseCheckpointSaver, as_json: bool
) -> None:
    """Test a sequential chain of nodes where the last node uses Overwrite to bypass a reducer and write a value directly to the channel."""

    class State(TypedDict):
        messages: Annotated[list, operator.add]

    def node_a(state: State):
        return {"messages": ["a"]}

    def node_b(state: State):
        overwrite = {"__overwrite__": ["b"]} if as_json else Overwrite(["b"])
        return {"messages": overwrite}

    builder = StateGraph(State)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")

    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"messages": ["START"]}, config)
    # a is overwritten by b
    assert result == {"messages": ["b"]}


@pytest.mark.parametrize("as_json", [False, True])
def test_overwrite_parallel(
    sync_checkpointer: BaseCheckpointSaver, as_json: bool
) -> None:
    """Test parallel nodes where max one node uses Overwrite to bypass a reducer and write a value directly to the channel."""

    class State(TypedDict):
        messages: Annotated[list, operator.add]

    def node_a(state: State):
        return {"messages": ["a"]}

    def node_b(state: State):
        overwrite = {"__overwrite__": ["b"]} if as_json else Overwrite(["b"])
        return {"messages": overwrite}

    def node_c(state: State):
        return {"messages": ["c"]}

    def node_d(state: State):
        return {"messages": ["d"]}

    builder = StateGraph(State)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)
    builder.add_node("node_d", node_d)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_a", "node_c")
    builder.add_edge("node_b", "node_d")
    builder.add_edge("node_c", "node_d")

    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"messages": ["START"]}, config)
    # a, c are overwritten by b, then d is written
    assert result == {"messages": ["b", "d"]}


@pytest.mark.parametrize("as_json", [False, True])
def test_overwrite_parallel_error(
    sync_checkpointer: BaseCheckpointSaver, as_json: bool
) -> None:
    """Test parallel nodes where more than one node uses Overwrite to bypass a reducer and write a value directly to the channel. In this case, InvalidUpdateError should be raised."""

    class State(TypedDict):
        messages: Annotated[list, operator.add]

    def node_a(state: State):
        return {"messages": ["a"]}

    def node_b(state: State):
        overwrite = {"__overwrite__": ["b"]} if as_json else Overwrite(["b"])
        return {"messages": overwrite}

    def node_c(state: State):
        overwrite = {"__overwrite__": ["c"]} if as_json else Overwrite(["c"])
        return {"messages": overwrite}

    builder = StateGraph(State)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_a", "node_c")
    builder.add_edge("node_b", END)
    builder.add_edge("node_c", END)

    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    with pytest.raises(
        InvalidUpdateError, match="Can receive only one Overwrite value per super-step."
    ):
        graph.invoke({"messages": ["START"]}, config)


def test_fork_does_not_apply_pending_writes(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test that forking with update_state does not apply pending writes from original execution."""

    class State(TypedDict):
        value: Annotated[int, operator.add]

    def node_a(state: State) -> State:
        return {"value": 10}

    def node_b(state: State) -> State:
        return {"value": 100}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    thread1 = {"configurable": {"thread_id": "1"}}
    graph.invoke({"value": 1}, thread1)

    history = list(graph.get_state_history(thread1))
    checkpoint_before_a = next(s for s in history if s.next == ("node_a",))

    fork_config = graph.update_state(
        checkpoint_before_a.config, {"value": 20}, as_node="node_a"
    )

    # Continue from fork (should run node_b)
    result = graph.invoke(None, fork_config)

    # Should be: 1 (input) + 20 (forked node_a) + 100 (node_b) = 121
    assert result == {"value": 121}
