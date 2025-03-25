import enum
import functools
import gc
import json
import logging
import operator
import threading
import time
import uuid
import warnings
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from random import randrange
from typing import (
    Annotated,
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_type_hints,
)

import httpx
import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.runnables.graph import Edge
from langsmith import traceable
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion
from typing_extensions import TypedDict

from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.context import Context
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import CONFIG_KEY_NODE_FINISHED, ERROR, PULL, START
from langgraph.errors import InvalidUpdateError
from langgraph.func import entrypoint, task
from langgraph.graph import END, Graph, StateGraph
from langgraph.graph.message import MessageGraph, MessagesState, add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.pregel import Channel, GraphRecursionError, Pregel, StateSnapshot
from langgraph.pregel.loop import SyncPregelLoop
from langgraph.pregel.retry import RetryPolicy
from langgraph.pregel.runner import PregelRunner
from langgraph.store.base import BaseStore
from langgraph.types import (
    Command,
    Interrupt,
    PregelTask,
    Send,
    StateUpdate,
    StreamWriter,
    interrupt,
)
from tests.agents import AgentAction, AgentFinish
from tests.any_str import AnyStr, AnyVersion, FloatBetween, UnsortedSequence
from tests.conftest import (
    ALL_CHECKPOINTERS_SYNC,
    ALL_STORES_SYNC,
    REGULAR_CHECKPOINTERS_SYNC,
    SHOULD_CHECK_SNAPSHOTS,
)
from tests.messages import (
    _AnyIdAIMessage,
    _AnyIdAIMessageChunk,
    _AnyIdHumanMessage,
    _AnyIdToolMessage,
)

pytestmark = pytest.mark.anyio

logger = logging.getLogger(__name__)


def test_graph_validation() -> None:
    def logic(inp: str) -> str:
        return ""

    workflow = Graph()
    workflow.add_node("agent", logic)
    workflow.set_entry_point("agent")
    workflow.set_finish_point("agent")
    assert workflow.compile(), "valid graph"

    # Accept a dead-end
    workflow = Graph()
    workflow.add_node("agent", logic)
    workflow.set_entry_point("agent")
    workflow.compile()

    workflow = Graph()
    workflow.add_node("agent", logic)
    workflow.set_finish_point("agent")
    with pytest.raises(ValueError, match="must have an entrypoint"):
        workflow.compile()

    workflow = Graph()
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", logic, {"continue": "tools", "exit": END})
    workflow.add_edge("tools", "agent")
    assert workflow.compile(), "valid graph"

    workflow = Graph()
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    workflow.set_entry_point("tools")
    workflow.add_conditional_edges("agent", logic, {"continue": "tools", "exit": END})
    workflow.add_edge("tools", "agent")
    assert workflow.compile(), "valid graph"

    workflow = Graph()
    workflow.set_entry_point("tools")
    workflow.add_conditional_edges("agent", logic, {"continue": "tools", "exit": END})
    workflow.add_edge("tools", "agent")
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    assert workflow.compile(), "valid graph"

    workflow = Graph()
    workflow.set_entry_point("tools")
    workflow.add_conditional_edges(
        "agent", logic, {"continue": "tools", "exit": END, "hmm": "extra"}
    )
    workflow.add_edge("tools", "agent")
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    with pytest.raises(ValueError, match="unknown"):  # extra is not defined
        workflow.compile()

    workflow = Graph()
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", logic, {"continue": "tools", "exit": END})
    workflow.add_edge("tools", "extra")
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    with pytest.raises(ValueError, match="unknown"):  # extra is not defined
        workflow.compile()

    workflow = Graph()
    workflow.add_node("agent", logic)
    workflow.add_node("tools", logic)
    workflow.add_node("extra", logic)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", logic)
    workflow.add_edge("tools", "agent")
    # Accept, even though extra is dead-end
    workflow.compile()

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
        def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
            raise ValueError("Faulty get_tuple")

    class FaultyPutCheckpointer(MemorySaver):
        def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: Optional[dict[str, Union[str, int, float]]] = None,
        ) -> RunnableConfig:
            raise ValueError("Faulty put")

    class FaultyPutWritesCheckpointer(InMemorySaver):
        def put_writes(
            self, config: RunnableConfig, writes: List[Tuple[str, Any]], task_id: str
        ) -> RunnableConfig:
            raise ValueError("Faulty put_writes")

    class FaultyVersionCheckpointer(InMemorySaver):
        def get_next_version(self, current: Optional[int], channel: BaseChannel) -> int:
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
        graph.invoke("", {"configurable": {"thread_id": "thread-1"}})


def test_config_json_schema() -> None:
    """Test that config json schema is generated properly."""
    chain = Channel.subscribe_to("input") | Channel.write_to("output")

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
        config_type=Foo,
    )

    assert app.get_config_jsonschema() == {
        "$defs": {
            "Foo": {
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
            },
        },
        "properties": {
            "configurable": {
                "$ref": "#/$defs/Foo",
                "default": None,
            },
        },
        "title": "LangGraphConfig",
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
    chain = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

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
    graph = Graph()
    graph.add_node("add_one", add_one)
    graph.set_entry_point("add_one")
    graph.set_finish_point("add_one")
    gapp = graph.compile()

    if SHOULD_CHECK_SNAPSHOTS:
        assert app.input_schema.model_json_schema() == {
            "title": "LangGraphInput",
            "type": "integer",
        }
        assert app.output_schema.model_json_schema() == {
            "title": "LangGraphOutput",
            "type": "integer",
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # raise warnings as errors
            assert app.config_schema().model_json_schema() == {
                "properties": {},
                "title": "LangGraphConfig",
                "type": "object",
            }

    assert app.invoke(2) == 3
    assert app.invoke(2, output_keys=["output"]) == {"output": 3}
    assert repr(app), "does not raise recursion error"

    assert gapp.invoke(2, debug=True) == 3


@pytest.mark.parametrize(
    "falsy_value",
    [None, False, 0, "", [], {}, set(), frozenset(), 0.0, 0j],
)
def test_invoke_single_process_in_out_falsy_values(falsy_value: Any) -> None:
    graph = Graph()
    graph.add_node("return_falsy_const", lambda *args, **kwargs: falsy_value)
    graph.set_entry_point("return_falsy_const")
    graph.set_finish_point("return_falsy_const")
    gapp = graph.compile()
    assert gapp.invoke(1) == falsy_value


def test_invoke_single_process_in_write_kwargs(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    chain = (
        Channel.subscribe_to("input")
        | add_one
        | Channel.write_to("output", fixed=5, output_plus_one=lambda x: x + 1)
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

    if SHOULD_CHECK_SNAPSHOTS:
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
    chain = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": chain},
        channels={"input": LastValue(int), "output": LastValue(int)},
        input_channels="input",
        output_channels=["output"],
    )

    if SHOULD_CHECK_SNAPSHOTS:
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
    chain = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": chain},
        channels={"input": LastValue(int), "output": LastValue(int)},
        input_channels=["input"],
        output_channels=["output"],
    )
    if SHOULD_CHECK_SNAPSHOTS:
        assert app.input_schema.model_json_schema() == {
            "title": "LangGraphInput",
            "type": "object",
            "properties": {
                "input": {"title": "Input", "type": "integer", "default": None}
            },
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
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = Channel.subscribe_to("inbox") | add_one | Channel.write_to("output")

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

    graph = Graph()
    graph.add_node("add_one", add_one)
    graph.add_node("add_one_more", add_one)
    graph.set_entry_point("add_one")
    graph.set_finish_point("add_one_more")
    graph.add_edge("add_one", "add_one_more")
    gapp = graph.compile()

    assert gapp.invoke(2) == 4

    for step, values in enumerate(gapp.stream(2, debug=1), start=1):
        if step == 1:
            assert values == {
                "add_one": 3,
            }
        elif step == 2:
            assert values == {
                "add_one_more": 4,
            }
        else:
            assert 0, f"{step}:{values}"
    assert step == 2


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
    graph = builder.compile(checkpointer=checkpointer)

    thread_id = uuid.uuid4()
    thread1 = {"configurable": {"thread_id": str(thread_id)}}

    result = graph.invoke({"myval": 1}, thread1)
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


def test_invoke_two_processes_in_dict_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = (
        Channel.subscribe_to("inbox")
        | RunnableLambda(add_one).batch
        | RunnablePassthrough(lambda _: time.sleep(0.1))
        | Channel.write_to("output").batch
    )

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "inbox": Topic(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels=["input", "inbox"],
        stream_channels=["output", "inbox"],
        output_channels=["output"],
    )

    # [12 + 1, 2 + 1 + 1]
    assert [
        *app.stream(
            {"input": 2, "inbox": 12}, output_keys="output", stream_mode="updates"
        )
    ] == [
        {"one": None},
        {"two": 13},
        {"two": 4},
    ]
    assert [*app.stream({"input": 2, "inbox": 12}, output_keys="output")] == [
        13,
        4,
    ]

    assert [*app.stream({"input": 2, "inbox": 12}, stream_mode="updates")] == [
        {"one": {"inbox": 3}},
        {"two": {"output": 13}},
        {"two": {"output": 4}},
    ]
    assert [*app.stream({"input": 2, "inbox": 12})] == [
        {"inbox": [3], "output": 13},
        {"output": 4},
    ]
    assert [*app.stream({"input": 2, "inbox": 12}, stream_mode="debug")] == [
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "id": AnyStr(),
                "name": "one",
                "input": 2,
                "triggers": ("input",),
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "id": AnyStr(),
                "name": "two",
                "input": [12],
                "triggers": ("inbox",),
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "id": AnyStr(),
                "name": "one",
                "result": [("inbox", 3)],
                "error": None,
                "interrupts": [],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "id": AnyStr(),
                "name": "two",
                "result": [("output", 13)],
                "error": None,
                "interrupts": [],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": AnyStr(),
                "name": "two",
                "input": [3],
                "triggers": ("inbox",),
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": AnyStr(),
                "name": "two",
                "result": [("output", 4)],
                "error": None,
                "interrupts": [],
            },
        },
    ]


def test_batch_two_processes_in_out() -> None:
    def add_one_with_delay(inp: int) -> int:
        time.sleep(inp / 10)
        return inp + 1

    one = Channel.subscribe_to("input") | add_one_with_delay | Channel.write_to("one")
    two = Channel.subscribe_to("one") | add_one_with_delay | Channel.write_to("output")

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

    graph = Graph()
    graph.add_node("add_one", add_one_with_delay)
    graph.add_node("add_one_more", add_one_with_delay)
    graph.set_entry_point("add_one")
    graph.set_finish_point("add_one_more")
    graph.add_edge("add_one", "add_one_more")
    gapp = graph.compile()

    assert gapp.batch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]


def test_invoke_many_processes_in_out(mocker: MockerFixture) -> None:
    test_size = 100
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    nodes = {"-1": Channel.subscribe_to("input") | add_one | Channel.write_to("-1")}
    for i in range(test_size - 2):
        nodes[str(i)] = (
            Channel.subscribe_to(str(i - 1)) | add_one | Channel.write_to(str(i))
        )
    nodes["last"] = Channel.subscribe_to(str(i)) | add_one | Channel.write_to("output")

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

    nodes = {"-1": Channel.subscribe_to("input") | add_one | Channel.write_to("-1")}
    for i in range(test_size - 2):
        nodes[str(i)] = (
            Channel.subscribe_to(str(i - 1)) | add_one | Channel.write_to(str(i))
        )
    nodes["last"] = Channel.subscribe_to(str(i)) | add_one | Channel.write_to("output")

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

    one = Channel.subscribe_to("input") | add_one | Channel.write_to("output")
    two = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

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

    one = Channel.subscribe_to("input") | add_one | Channel.write_to("output")
    two = Channel.subscribe_to("input") | add_one | Channel.write_to("output")

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
        Channel.subscribe_to(["input"]).join(["total"])
        | add_one
        | Channel.write_to("output", "total")
        | raise_if_above_10
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
            "v": 2,
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
            "v": 2,
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
            "v": 2,
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_imp_task(request: pytest.FixtureRequest, checkpointer_name: str) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    mapper_calls = 0

    class Configurable:
        model: str

    @task()
    def mapper(input: int) -> str:
        nonlocal mapper_calls
        mapper_calls += 1
        time.sleep(input / 100)
        return str(input) * 2

    @entrypoint(checkpointer=checkpointer, config_schema=Configurable)
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
    assert graph.get_config_jsonschema() == {
        "$defs": {
            "Configurable": {
                "properties": {
                    "model": {"default": None, "title": "Model", "type": "string"},
                    "checkpoint_id": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Pass to fetch a past checkpoint. If None, fetches the latest checkpoint.",
                        "title": "Checkpoint ID",
                    },
                    "checkpoint_ns": {
                        "default": "",
                        "description": 'Checkpoint namespace. Denotes the path to the subgraph node the checkpoint originates from, separated by `|` character, e.g. `"child|grandchild"`. Defaults to "" (root graph).',
                        "title": "Checkpoint NS",
                        "type": "string",
                    },
                    "thread_id": {
                        "default": "",
                        "title": "Thread ID",
                        "type": "string",
                    },
                },
                "title": "Configurable",
                "type": "object",
            }
        },
        "properties": {
            "configurable": {"$ref": "#/$defs/Configurable", "default": None}
        },
        "title": "LangGraphConfig",
        "type": "object",
    }

    thread1 = {"configurable": {"thread_id": "1"}}
    assert [*graph.stream([0, 1], thread1)] == [
        {"mapper": "00"},
        {"mapper": "11"},
        {
            "__interrupt__": (
                Interrupt(
                    value="question",
                    resumable=True,
                    ns=[AnyStr("graph:")],
                    when="during",
                ),
            )
        },
    ]
    assert mapper_calls == 2

    assert graph.invoke(Command(resume="answer"), thread1) == [
        "00answer",
        "11answer",
    ]
    assert mapper_calls == 2


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_imp_nested(
    request: pytest.FixtureRequest, checkpointer_name: str, snapshot: SnapshotAssertion
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

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

    @entrypoint(checkpointer=checkpointer)
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
    assert [*graph.stream([0, 1], thread1)] == [
        {"submapper": "0"},
        {"mapper": "00"},
        {"submapper": "1"},
        {"mapper": "11"},
        {
            "__interrupt__": (
                Interrupt(
                    value="question",
                    resumable=True,
                    ns=[AnyStr("graph:")],
                    when="during",
                ),
            )
        },
    ]

    assert graph.invoke(Command(resume="answer"), thread1) == [
        "00answera",
        "11answera",
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_imp_stream_order(
    request: pytest.FixtureRequest, checkpointer_name: str, snapshot: SnapshotAssertion
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    @task()
    def foo(state: dict) -> tuple:
        return state["a"] + "foo", "bar"

    @task
    def bar(a: str, b: str, c: Optional[str] = None) -> dict:
        return {"a": a + b, "c": (c or "") + "bark"}

    @task
    def baz(state: dict) -> dict:
        return {"a": state["a"] + "baz", "c": "something else"}

    @entrypoint(checkpointer=checkpointer)
    def graph(state: dict) -> dict:
        fut_foo = foo(state)
        fut_bar = bar(*fut_foo.result())
        fut_baz = baz(fut_bar.result())
        return fut_baz.result()

    thread1 = {"configurable": {"thread_id": "1"}}
    assert [c for c in graph.stream({"a": "0"}, thread1)] == [
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
        Channel.subscribe_to(["input"]).join(["total"])
        | adder
        | Channel.write_to("output", "total")
        | raise_if_above_10
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
    assert app.invoke(2, thread_1, debug=1) == 2
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
    assert app.invoke(5, thread_2, debug=True) == 5
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

    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    chain_three = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    chain_four = (
        Channel.subscribe_to("inbox") | add_10_each | Channel.write_to("output")
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_invoke_join_then_call_other_pregel(
    mocker: MockerFixture, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])

    inner_app = Pregel(
        nodes={
            "one": Channel.subscribe_to("input") | add_one | Channel.write_to("output")
        },
        channels={
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    one = (
        Channel.subscribe_to("input")
        | add_10_each
        | Channel.write_to("inbox_one").map()
    )
    two = (
        Channel.subscribe_to("inbox_one")
        | inner_app.map()
        | sorted
        | Channel.write_to("outbox_one")
    )
    chain_three = Channel.subscribe_to("outbox_one") | sum | Channel.write_to("output")

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
    app.checkpointer = checkpointer
    # subgraph is called twice in the same node, but that works
    assert app.invoke([2, 3], {"configurable": {"thread_id": "1"}}) == 27

    # set inner graph checkpointer NeverCheckpoint
    inner_app.checkpointer = False
    # subgraph still called twice, but checkpointing for inner graph is disabled
    assert app.invoke([2, 3], {"configurable": {"thread_id": "1"}}) == 27


def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = (
        Channel.subscribe_to("input") | add_one | Channel.write_to("output", "between")
    )
    two = Channel.subscribe_to("between") | add_one | Channel.write_to("output")

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
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("between")
    two = Channel.subscribe_to("between") | add_one

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

    one = Channel.subscribe_to("between") | add_one | Channel.write_to("output")
    two = Channel.subscribe_to("between") | add_one

    with pytest.raises(TypeError):
        Pregel(nodes={"one": one, "two": two})


def test_channel_enter_exit_timing(mocker: MockerFixture) -> None:
    setup = mocker.Mock()
    cleanup = mocker.Mock()

    @contextmanager
    def an_int() -> Generator[int, None, None]:
        setup()
        try:
            yield 5
        finally:
            cleanup()

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = (
        Channel.subscribe_to("inbox")
        | RunnableLambda(add_one).batch
        | Channel.write_to("output").batch
    )

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "inbox": Topic(int),
            "ctx": Context(an_int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels=["inbox", "output"],
        stream_channels=["inbox", "output"],
    )

    assert setup.call_count == 0
    assert cleanup.call_count == 0
    for i, chunk in enumerate(app.stream(2)):
        assert setup.call_count == 1, "Expected setup to be called once"
        if i == 0:
            assert chunk == {"inbox": [3]}
        elif i == 1:
            assert chunk == {"output": 4}
        else:
            assert False, "Expected only two chunks"
    assert cleanup.call_count == 1, "Expected cleanup to be called once"


def test_conditional_entrypoint_graph(snapshot: SnapshotAssertion) -> None:
    def left(data: str) -> str:
        return data + "->left"

    def right(data: str) -> str:
        return data + "->right"

    def should_start(data: str) -> str:
        # Logic to decide where to start
        if len(data) > 10:
            return "go-right"
        else:
            return "go-left"

    # Define a new graph
    workflow = Graph()

    workflow.add_node("left", left)
    workflow.add_node("right", right)

    workflow.set_conditional_entry_point(
        should_start, {"go-left": "left", "go-right": "right"}
    )

    workflow.add_conditional_edges("left", lambda data: END, {END: END})
    workflow.add_edge("right", END)

    app = workflow.compile()

    if SHOULD_CHECK_SNAPSHOTS:
        assert json.dumps(app.get_input_schema().model_json_schema()) == snapshot
        assert json.dumps(app.get_output_schema().model_json_schema()) == snapshot
        assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert (
        app.invoke("what is weather in sf", debug=True)
        == "what is weather in sf->right"
    )

    assert [*app.stream("what is weather in sf")] == [
        {"right": "what is weather in sf->right"},
    ]


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

    if SHOULD_CHECK_SNAPSHOTS:
        assert json.dumps(app.get_input_schema().model_json_schema()) == snapshot
        assert json.dumps(app.get_output_schema().model_json_schema()) == snapshot
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
    builder = StateGraph(AgentState, Config)

    builder.add_node("agent", agent)
    builder.add_node("tools", execute_tools)

    builder.set_entry_point("agent")

    builder.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    builder.add_edge("tools", "agent")

    app = builder.compile()

    if SHOULD_CHECK_SNAPSHOTS:
        assert json.dumps(app.config_schema().model_json_schema()) == snapshot
        assert json.dumps(app.get_input_schema().model_json_schema()) == snapshot
        assert json.dumps(app.get_output_schema().model_json_schema()) == snapshot

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

    if SHOULD_CHECK_SNAPSHOTS:
        assert json.dumps(app.get_input_schema().model_json_schema()) == snapshot
        assert json.dumps(app.get_output_schema().model_json_schema()) == snapshot
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

    assert [c for c in app_w_interrupt.stream(None, config, debug=1)] == [
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
    setup = mocker.Mock()
    teardown = mocker.Mock()

    @contextmanager
    def assert_ctx_once() -> Iterator[None]:
        assert setup.call_count == 0
        assert teardown.call_count == 0
        try:
            yield
        finally:
            assert setup.call_count == 1
            assert teardown.call_count == 1
            setup.reset_mock()
            teardown.reset_mock()

    @contextmanager
    def make_httpx_client() -> Iterator[httpx.Client]:
        setup()
        with httpx.Client() as client:
            try:
                yield client
            finally:
                teardown()

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
        inner: Annotated[InnerObject, lambda x, y: y]
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]
        client: Annotated[httpx.Client, Context(make_httpx_client)]

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

    class UpdateDocs34(BaseModel):
        docs: list[str] = ["doc3", "doc4"]

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

    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
    assert app.get_input_jsonschema() == snapshot
    assert app.get_output_jsonschema() == snapshot

    with pytest.raises(ValidationError), assert_ctx_once():
        app.invoke({"query": {}})

    with assert_ctx_once():
        assert app.invoke({"query": "what is weather in sf", "inner": {"yo": 1}}) == {
            "docs": ["doc1", "doc2", "doc3", "doc4"],
            "answer": "doc1,doc2,doc3,doc4",
        }

    with assert_ctx_once():
        assert [
            *app.stream({"query": "what is weather in sf", "inner": {"yo": 1}})
        ] == [
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

    with assert_ctx_once():
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

    with assert_ctx_once():
        assert [c for c in app_w_interrupt.stream(None, config)] == [
            {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        ]

    with assert_ctx_once():
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
def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic2(
    snapshot: SnapshotAssertion,
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
    checkpointer_name: str,
) -> None:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    setup = mocker.Mock()
    teardown = mocker.Mock()

    @contextmanager
    def assert_ctx_once() -> Iterator[None]:
        assert setup.call_count == 0
        assert teardown.call_count == 0
        try:
            yield
        finally:
            assert setup.call_count == 1
            assert teardown.call_count == 1
            setup.reset_mock()
            teardown.reset_mock()

    @contextmanager
    def make_httpx_client() -> Iterator[httpx.Client]:
        setup()
        with httpx.Client() as client:
            try:
                yield client
            finally:
                teardown()

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
        model_config = ConfigDict(arbitrary_types_allowed=True)

        query: str
        inner: Annotated[InnerObject, lambda x, y: y]
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]
        client: Annotated[httpx.Client, Context(make_httpx_client)]

    class StateUpdate(BaseModel):
        query: Optional[str] = None
        answer: Optional[str] = None
        docs: Optional[list[str]] = None

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

    if SHOULD_CHECK_SNAPSHOTS:
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
        assert app.get_input_schema().model_json_schema() == snapshot
        assert app.get_output_schema().model_json_schema() == snapshot

    with pytest.raises(ValidationError), assert_ctx_once():
        app.invoke({"query": {}})

    with assert_ctx_once():
        assert app.invoke({"query": "what is weather in sf", "inner": {"yo": 1}}) == {
            "docs": ["doc1", "doc2", "doc3", "doc4"],
            "answer": "doc1,doc2,doc3,doc4",
        }

    with assert_ctx_once():
        assert [
            *app.stream({"query": "what is weather in sf", "inner": {"yo": 1}})
        ] == [
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

    with assert_ctx_once():
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

    with assert_ctx_once():
        assert [c for c in app_w_interrupt.stream(None, config)] == [
            {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        ]

    with assert_ctx_once():
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
def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic_input(
    snapshot: SnapshotAssertion,
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
    checkpointer_name: str,
) -> None:
    from pydantic import BaseModel

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

    class QueryModel(BaseModel):
        query: str

    class State(QueryModel):
        inner: InnerObject
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]

    class StateUpdate(BaseModel):
        query: Optional[str] = None
        answer: Optional[str] = None
        docs: Optional[list[str]] = None

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
        checkpointer=checkpointer,
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


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_nested_pydantic_models(version: str) -> None:
    """Test that nested Pydantic models are properly constructed from leaf nodes up."""

    # Define nested Pydantic models
    if version == "v1":
        from pydantic.v1 import BaseModel, Field
    else:
        from pydantic import BaseModel, Field

    class NestedModel(BaseModel):
        value: int
        name: str

    # Forward reference model
    class RecursiveModel(BaseModel):
        value: str
        child: Optional["RecursiveModel"] = None

    # Discriminated union models
    class Cat(BaseModel):
        pet_type: Literal["cat"]
        meow: str

    class Dog(BaseModel):
        pet_type: Literal["dog"]
        bark: str

    # Cyclic reference model
    class Person(BaseModel):
        id: str
        name: str
        friends: list[str] = Field(default_factory=list)  # IDs of friends

    class State(BaseModel):
        # Basic nested model tests
        top_level: str
        nested: NestedModel
        optional_nested: Annotated[Optional[NestedModel], lambda x, y: y, "Foo"]
        dict_nested: dict[str, NestedModel]
        list_nested: Annotated[
            Union[dict, list[dict[str, NestedModel]]], lambda x, y: (x or []) + [y]
        ]
        tuple_nested: tuple[str, NestedModel]
        tuple_list_nested: list[tuple[int, NestedModel]]
        complex_tuple: tuple[str, dict[str, tuple[int, NestedModel]]]

        # Forward reference test
        recursive: RecursiveModel

        # Discriminated union test
        pet: Union[Cat, Dog]

        # Cyclic reference test
        people: dict[str, Person]  # Map of ID -> Person

    inputs = {
        # Basic nested models
        "top_level": "initial",
        "nested": {"value": 42, "name": "test"},
        "optional_nested": {"value": 10, "name": "optional"},
        "dict_nested": {"a": {"value": 5, "name": "a"}},
        "list_nested": [{"a": {"value": 6, "name": "b"}}],
        "tuple_nested": ["tuple-key", {"value": 7, "name": "tuple-value"}],
        "tuple_list_nested": [[1, {"value": 8, "name": "tuple-in-list"}]],
        "complex_tuple": [
            "complex",
            {"nested": [9, {"value": 10, "name": "deep"}]},
        ],
        # Forward reference
        "recursive": {"value": "parent", "child": {"value": "child", "child": None}},
        # Discriminated union (using a cat in this case)
        "pet": {"pet_type": "cat", "meow": "meow!"},
        # Cyclic references
        "people": {
            "1": {
                "id": "1",
                "name": "Alice",
                "friends": ["2", "3"],  # Alice is friends with Bob and Charlie
            },
            "2": {
                "id": "2",
                "name": "Bob",
                "friends": ["1"],  # Bob is friends with Alice
            },
            "3": {
                "id": "3",
                "name": "Charlie",
                "friends": ["1", "2"],  # Charlie is friends with Alice and Bob
            },
        },
    }

    update = {"top_level": "updated", "nested": {"value": 100, "name": "updated"}}

    expected = State(**inputs)

    def node_fn(state: State) -> dict:
        assert state == expected
        return update

    builder = StateGraph(State)
    builder.add_node("process", node_fn)
    builder.set_entry_point("process")
    builder.set_finish_point("process")
    graph = builder.compile()

    result = graph.invoke(inputs.copy())

    assert result == {**inputs, **update}

    new_inputs = inputs.copy()
    new_inputs["list_nested"] = {"foo": "bar"}
    expected = State(**new_inputs)
    assert {**new_inputs, **update} == graph.invoke(new_inputs.copy())


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
        checkpointer=checkpointer,
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
        then=END,
    )
    tool_two = tool_two_graph.compile()

    graph = StateGraph(State)
    graph.add_node("tool_one", logic)
    graph.add_node("tool_two", tool_two)
    graph.add_node("tool_three", logic)
    graph.set_conditional_entry_point(lambda s: "tool_one", then=END)
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
        {"my_key": "my value", "never_called": never_called}, debug=True
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
    builder.add_node("node_2", node_2)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    assert graph.invoke({"foo": "foo"}, config) == {"foo": "hi! foo"}
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

    def node(state: State, writer: StreamWriter):
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
    assert app.invoke({"my_key": ""}, config, debug=True) == {
        "my_key": " and parallel",
    }

    assert app.invoke(None, config, debug=True) == {
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
    assert app.invoke({"my_key": "my value"}, config, debug=True) == {
        "my_key": "hi my value",
    }

    assert app.invoke(None, config, debug=True) == {
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


def test_checkpoint_metadata() -> None:
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
    checkpointer_1 = InMemorySaver()
    app = workflow.compile(checkpointer=checkpointer_1)

    # graph w/ interrupt
    checkpointer_2 = InMemorySaver()
    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer_2, interrupt_before=["tools"]
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
    chkpnt_metadata_1 = checkpointer_1.get_tuple(config).metadata
    assert chkpnt_metadata_1["thread_id"] == "1"
    assert chkpnt_metadata_1["test_config_1"] == "foo"
    assert chkpnt_metadata_1["test_config_2"] == "bar"

    # Verify that all checkpoint metadata have the expected keys. This check
    # is needed because a run may have an arbitrary number of steps depending
    # on how the graph is constructed.
    chkpnt_tuples_1 = checkpointer_1.list(config)
    for chkpnt_tuple in chkpnt_tuples_1:
        assert chkpnt_tuple.metadata["thread_id"] == "1"
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
    chkpnt_metadata_2 = checkpointer_2.get_tuple(config).metadata
    assert chkpnt_metadata_2["thread_id"] == "2"
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
    chkpnt_metadata_3 = checkpointer_2.get_tuple(config).metadata
    assert chkpnt_metadata_3["thread_id"] == "2"
    assert chkpnt_metadata_3["test_config_3"] == "foo"
    assert chkpnt_metadata_3["test_config_4"] == "bar"

    # Verify that all checkpoint metadata have the expected keys. This check
    # is needed because a run may have an arbitrary number of steps depending
    # on how the graph is constructed.
    chkpnt_tuples_2 = checkpointer_2.list(config)
    for chkpnt_tuple in chkpnt_tuples_2:
        assert chkpnt_tuple.metadata["thread_id"] == "2"
        assert chkpnt_tuple.metadata["test_config_3"] == "foo"
        assert chkpnt_tuple.metadata["test_config_4"] == "bar"


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_remove_message_via_state_update(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

    workflow = MessageGraph()
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

    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)
    app = workflow.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    output = app.invoke([HumanMessage(content="Hi")], config=config)
    app.update_state(config, values=[RemoveMessage(id=output[-1].id)])

    updated_state = app.get_state(config)

    assert len(updated_state.values) == 1
    assert updated_state.values[-1].content == "Hi"


def test_remove_message_from_node():
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

    workflow = MessageGraph()
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
    from pydantic import BaseModel, Field

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
        analysts: List[Analyst] = Field(
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
        messages: Annotated[List[AnyMessage], add_messages]
        analyst: Analyst
        section: Section

    class ResearchGraphState(TypedDict):
        analysts: List[Analyst]
        topic: str
        max_analysts: int
        sections: List[Section]
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
    interview_builder.add_conditional_edges("answer_question", route_messages)

    # Set up memory
    memory = InMemorySaver()

    # Interview
    interview_graph = interview_builder.compile(checkpointer=memory).with_config(
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_channel_values(request: pytest.FixtureRequest, checkpointer_name: str) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    config = {"configurable": {"thread_id": "1"}}
    chain = Channel.subscribe_to("input") | Channel.write_to("output")
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
        checkpointer=checkpointer,
    )
    app.invoke({"input": 1, "ephemeral": "meow"}, config)
    assert checkpointer.get(config)["channel_values"] == {"input": 1, "output": 1}


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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
@pytest.mark.parametrize("store_name", ALL_STORES_SYNC)
def test_store_injected(
    request: pytest.FixtureRequest, checkpointer_name: str, store_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    the_store = request.getfixturevalue(f"store_{store_name}")

    class State(TypedDict):
        count: Annotated[int, operator.add]

    doc_id = str(uuid.uuid4())
    doc = {"some-key": "this-is-a-val"}
    uid = uuid.uuid4().hex
    namespace = (f"foo-{uid}", "bar")
    thread_1 = str(uuid.uuid4())
    thread_2 = str(uuid.uuid4())

    class Node:
        def __init__(self, i: Optional[int] = None):
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
    N = 500
    M = 1

    for i in range(N):
        builder.add_node(f"node_{i}", Node(i))
        builder.add_edge("__start__", f"node_{i}")

    graph = builder.compile(store=the_store, checkpointer=checkpointer)

    results = graph.batch(
        [{"count": 0}] * M,
        ([{"configurable": {"thread_id": str(uuid.uuid4())}}] * (M - 1))
        + [{"configurable": {"thread_id": thread_1}}],
    )
    result = results[-1]
    assert result == {"count": N + 1}
    returned_doc = the_store.get(namespace, doc_id).value
    assert returned_doc == {**doc, "from_thread": thread_1, "some_val": 0}
    assert len(the_store.search(namespace)) == 1
    # Check results after another turn of the same thread
    result = graph.invoke({"count": 0}, {"configurable": {"thread_id": thread_1}})
    assert result == {"count": (N + 1) * 2}
    returned_doc = the_store.get(namespace, doc_id).value
    assert returned_doc == {**doc, "from_thread": thread_1, "some_val": N + 1}
    assert len(the_store.search(namespace)) == 1

    result = graph.invoke({"count": 0}, {"configurable": {"thread_id": thread_2}})
    assert result == {"count": N + 1}
    returned_doc = the_store.get(namespace, doc_id).value
    assert returned_doc == {
        **doc,
        "from_thread": thread_2,
        "some_val": 0,
    }  # Overwrites the whole doc
    assert len(the_store.search(namespace)) == 1  # still overwriting the same one


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
            "writes": {
                "alice": {
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_interrupt_functional(
    request: pytest.FixtureRequest, checkpointer_name: str, snapshot: SnapshotAssertion
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    @task
    def foo(state: dict) -> dict:
        return {"a": state["a"] + "foo"}

    @task
    def bar(state: dict) -> dict:
        return {"a": state["a"] + "bar", "b": state["b"]}

    @entrypoint(checkpointer=checkpointer)
    def graph(inputs: dict) -> dict:
        fut_foo = foo(inputs)
        value = interrupt("Provide value for bar:")
        bar_input = {**fut_foo.result(), "b": value}
        fut_bar = bar(bar_input)
        return fut_bar.result()

    config = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    graph.invoke({"a": ""}, config)
    # Resume with an answer
    res = graph.invoke(Command(resume="bar"), config)
    assert res == {"a": "foobar", "b": "bar"}


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_interrupt_task_functional(
    request: pytest.FixtureRequest, checkpointer_name: str, snapshot: SnapshotAssertion
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    @task
    def foo(state: dict) -> dict:
        return {"a": state["a"] + "foo"}

    @task
    def bar(state: dict) -> dict:
        value = interrupt("Provide value for bar:")
        return {"a": state["a"] + value}

    @entrypoint(checkpointer=checkpointer)
    def graph(inputs: dict) -> dict:
        fut_foo = foo(inputs)
        fut_bar = bar(fut_foo.result())
        return fut_bar.result()

    config = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert not graph.invoke({"a": ""}, config)
    # Resume with an answer
    res = graph.invoke(Command(resume="bar"), config)
    assert res == {"a": "foobar"}

    # Test that we can interrupt the same task multiple times
    config = {"configurable": {"thread_id": "2"}}

    @entrypoint(checkpointer=checkpointer)
    def graph(inputs: dict) -> dict:
        foo_result = foo(inputs).result()
        bar_result = bar(foo_result).result()
        baz_result = bar(bar_result).result()
        return baz_result

    # First run, interrupted at bar
    assert not graph.invoke({"a": ""}, config)
    # Provide resumes
    assert not graph.invoke(Command(resume="bar"), config)
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
    graph = builder.compile(checkpointer=checkpointer)

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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_falsy_return_from_task(
    request: pytest.FixtureRequest, checkpointer_name: str, snapshot: SnapshotAssertion
):
    """Test with a falsy return from a task."""
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    @task
    def falsy_task() -> bool:
        return False

    @entrypoint(checkpointer=checkpointer)
    def graph(state: dict) -> dict:
        """React tool."""
        falsy_task().result()
        interrupt("test")

    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    assert [
        chunk for chunk in graph.stream({"a": 5}, configurable, stream_mode="debug")
    ] == [
        {
            "payload": {
                "config": {
                    "callbacks": None,
                    "configurable": {
                        "checkpoint_id": AnyStr(),
                        "checkpoint_ns": "",
                        "thread_id": AnyStr(),
                    },
                    "metadata": configurable["configurable"],
                    "recursion_limit": 25,
                    "tags": [],
                },
                "metadata": {
                    "parents": {},
                    "source": "input",
                    "step": -1,
                    "thread_id": AnyStr(),
                    "writes": {
                        "__start__": {
                            "a": 5,
                        },
                    },
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
                "result": [
                    (
                        "__return__",
                        False,
                    ),
                ],
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
                        "ns": [
                            AnyStr(),
                        ],
                        "resumable": True,
                        "value": "test",
                        "when": "during",
                    },
                ],
                "name": "graph",
                "result": [],
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task_result",
        },
    ]
    assert [
        c
        for c in graph.stream(Command(resume="123"), configurable, stream_mode="debug")
    ] == [
        {
            "payload": {
                "config": {
                    "callbacks": None,
                    "configurable": {
                        "checkpoint_id": AnyStr(),
                        "checkpoint_ns": "",
                        "thread_id": AnyStr(),
                    },
                    "metadata": configurable["configurable"],
                    "recursion_limit": 25,
                    "tags": [],
                },
                "metadata": {
                    "parents": {},
                    "source": "input",
                    "step": -1,
                    "thread_id": AnyStr(),
                    "writes": {
                        "__start__": {
                            "a": 5,
                        },
                    },
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
                                "ns": [
                                    AnyStr(),
                                ],
                                "resumable": True,
                                "value": "test",
                                "when": "during",
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
                "result": [
                    (
                        "__end__",
                        None,
                    ),
                ],
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task_result",
        },
        {
            "payload": {
                "config": {
                    "callbacks": None,
                    "configurable": {
                        "checkpoint_id": AnyStr(),
                        "checkpoint_ns": "",
                        "thread_id": AnyStr(),
                    },
                    "metadata": configurable["configurable"],
                    "recursion_limit": 25,
                    "tags": [],
                },
                "metadata": {
                    "parents": {},
                    "source": "loop",
                    "step": 0,
                    "thread_id": AnyStr(),
                    "writes": {
                        "falsy_task": False,
                        "graph": None,
                    },
                },
                "next": [],
                "parent_config": {
                    "callbacks": None,
                    "configurable": {
                        "checkpoint_id": AnyStr(),
                        "checkpoint_ns": "",
                        "thread_id": AnyStr(),
                    },
                    "metadata": configurable["configurable"],
                    "recursion_limit": 25,
                    "tags": [],
                },
                "tasks": [],
                "values": None,
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "checkpoint",
        },
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_multiple_interrupts_functional(
    request: pytest.FixtureRequest, checkpointer_name: str, snapshot: SnapshotAssertion
):
    """Test multiple interrupts with functional API."""
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    counter = 0

    @task
    def double(x: int) -> int:
        """Increment the counter."""
        nonlocal counter
        counter += 1
        return 2 * x

    @entrypoint(checkpointer=checkpointer)
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

    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
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


def test_sync_streaming_with_functional_api() -> None:
    """Test streaming with functional API.

    This test verifies that we're able to stream results as they're being generated
    rather than have all the results arrive at once after the graph has completed.

    The time of arrival between the two updates corresponding to the two `slow` tasks
    should be greater than the time delay between the two tasks.
    """

    time_delay = 0.01

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


def test_entrypoint_stateful() -> None:
    """Test stateful entrypoint invoke."""

    # Test invoke
    states = []

    @entrypoint(checkpointer=MemorySaver())
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
    @entrypoint(checkpointer=MemorySaver())
    def foo(inputs, *, previous: Any) -> Any:
        return {"previous": previous, "current": inputs}

    config = {"configurable": {"thread_id": "1"}}
    items = [item for item in foo.stream({"a": "1"}, config)]
    assert items == [{"foo": {"current": {"a": "1"}, "previous": None}}]


def test_entrypoint_from_sync_generator() -> None:
    """@entrypoint does not support sync generators."""
    previous_return_values = []

    with pytest.raises(NotImplementedError):

        @entrypoint(checkpointer=MemorySaver())
        def foo(inputs, previous=None) -> Any:
            previous_return_values.append(previous)
            yield "a"
            yield "b"


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
def test_multiple_subgraphs_functional(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

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

    @entrypoint(checkpointer=checkpointer)
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

    @entrypoint(checkpointer=checkpointer)
    def parent_call_multiple_subgraphs(inputs):
        return call_multiple_subgraphs(*inputs).result()

    config = {"configurable": {"thread_id": "2"}}
    assert parent_call_multiple_subgraphs.invoke([2, 3], config) == [5, 6]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_multiple_subgraphs_mixed_entrypoint(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    """Test calling multiple StateGraph subgraphs from an entrypoint."""
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
    @task
    def call_same_subgraph(a, b):
        result = add_subgraph.invoke({"a": a, "b": b})["result"]
        another_result = add_subgraph.invoke({"a": result, "b": 10})["result"]
        return another_result

    @entrypoint(checkpointer=checkpointer)
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

    @entrypoint(checkpointer=checkpointer)
    def parent_call_multiple_subgraphs(inputs):
        return call_multiple_subgraphs(*inputs).result()

    config = {"configurable": {"thread_id": "2"}}
    assert parent_call_multiple_subgraphs.invoke([2, 3], config) == [5, 6]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_multiple_subgraphs_mixed_state_graph(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    """Test calling multiple entrypoint "subgraphs" from a StateGraph."""
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

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
        add_result = add.invoke([state["a"], state["b"]])
        multiply_result = multiply.invoke([state["a"], state["b"]])
        return {
            "add_result": add_result,
            "multiply_result": multiply_result,
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


def test_entrypoint_output_schema_with_return_and_save() -> None:
    """Test output schema inference with entrypoint.final."""

    # Un-parameterized entrypoint.final is interpreted as entrypoint.final[Any, Any]
    @entrypoint()
    def foo2(inputs, *, previous: Any) -> entrypoint.final:
        return entrypoint.final(value="foo", save=1)

    assert foo2.get_output_schema().model_json_schema() == {
        "title": "LangGraphOutput",
    }

    @entrypoint()
    def foo(inputs, *, previous: Any) -> entrypoint.final[str, int]:
        return entrypoint.final(value="foo", save=1)

    assert foo.get_output_schema().model_json_schema() == {
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


def test_entrypoint_with_return_and_save() -> None:
    """Test entrypoint with return and save."""
    previous_ = None

    @entrypoint(checkpointer=MemorySaver())
    def foo(msg: str, *, previous: Any) -> entrypoint.final[int, list[str]]:
        nonlocal previous_
        previous_ = previous
        previous = previous or []
        return entrypoint.final(value=len(previous), save=previous + [msg])

    assert foo.get_output_schema().model_json_schema() == {
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


def test_overriding_injectable_args_with_tasks() -> None:
    """Test overriding injectable args in tasks."""
    from langgraph.store.memory import InMemoryStore

    @task
    def foo(store: BaseStore, writer: StreamWriter, value: Any) -> None:
        assert store is value
        assert writer is value

    @entrypoint(store=InMemoryStore())
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
            _AnyIdAIMessageChunk(content="foo"),
            {
                "langgraph_step": 1,
                "langgraph_node": "call_model",
                "langgraph_triggers": (
                    "branch:to:call_model",
                    "start:call_model",
                ),
                "langgraph_path": ("__pregel_pull", "call_model"),
                "langgraph_checkpoint_ns": AnyStr("call_model:"),
                "checkpoint_ns": AnyStr("call_model:"),
                "ls_provider": "genericfakechatmodel",
                "ls_model_type": "chat",
                "tags": ["meow"],
            },
        )
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
        ] == graph.edges


def test_pydantic_none_state_update() -> None:
    from pydantic import BaseModel

    class State(BaseModel):
        foo: Optional[str]

    def node_a(state: State) -> State:
        return State(foo=None)

    graph = StateGraph(State).add_node(node_a).add_edge(START, "node_a").compile()
    assert graph.invoke({"foo": ""}) == {"foo": None}


def test_pydantic_state_mutation() -> None:
    from pydantic import BaseModel, Field

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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_stream_messages_dedupe_state(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    from langchain_core.messages import AIMessage

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
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
        .compile(checkpointer=checkpointer)
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


def test_empty_invoke() -> None:
    from pydantic import BaseModel

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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_parallel_interrupts(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    from pydantic import BaseModel, Field

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    # --- CHILD GRAPH ---

    class ChildState(BaseModel):
        prompt: str = Field(..., description="What is going to be asked to the user?")
        human_input: Optional[str] = Field(None, description="What the human said")
        human_inputs: Annotated[List[str], operator.add] = Field(
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
        prompts: List[str] = Field(
            ..., description="What is going to be asked to the user?"
        )
        human_inputs: Annotated[List[str], operator.add] = Field(
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

    parent_graph = parent_graph_builder.compile(checkpointer=checkpointer)

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
        if any(i.resumable for i in current_interrupts):
            current_input = Command(resume=f"Resume #{invokes}")

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
                    resumable=True,
                    ns=[
                        AnyStr("child_graph:"),
                        AnyStr("get_human_input:"),
                    ],
                ),
            )
        },
        {
            "__interrupt__": (
                Interrupt(
                    value="b",
                    resumable=True,
                    ns=[
                        AnyStr("child_graph:"),
                        AnyStr("get_human_input:"),
                    ],
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
                        resumable=True,
                        ns=[
                            AnyStr("child_graph:"),
                            AnyStr("get_human_input:"),
                        ],
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
                        resumable=True,
                        ns=[
                            AnyStr("child_graph:"),
                            AnyStr("get_human_input:"),
                        ],
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_parallel_interrupts_double(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    from pydantic import BaseModel, Field

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    # --- CHILD GRAPH ---

    class ChildState(BaseModel):
        prompt: str = Field(..., description="What is going to be asked to the user?")
        human_input: Optional[str] = Field(None, description="What the human said")
        human_inputs: Annotated[List[str], operator.add] = Field(
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
        prompts: List[str] = Field(
            ..., description="What is going to be asked to the user?"
        )
        human_inputs: Annotated[List[str], operator.add] = Field(
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

    parent_graph = parent_graph_builder.compile(checkpointer=checkpointer)

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
        if any(i.resumable for i in current_interrupts):
            current_input = Command(resume=f"Resume #{invokes}")

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


@pytest.mark.parametrize("checkpointer_name", REGULAR_CHECKPOINTERS_SYNC)
def test_bulk_state_updates(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

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
        .compile(checkpointer=checkpointer)
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
    checkpoints = list(checkpointer.list(config))
    assert len(checkpoints) == 2
    assert checkpoints[0].metadata["writes"] == {
        "node_a": {"foo": "updated"},
        "node_b": {"baz": "new"},
    }
    assert checkpoints[1].metadata["writes"] == {"node_a": {"foo": "bar"}}

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

    checkpoints = list(checkpointer.list(config))
    assert len(checkpoints) == 2
    assert checkpoints[0].metadata["writes"] == {
        "node_a": {"foo": "updated"},
        "node_b": {"baz": "new"},
    }
    assert checkpoints[1].metadata["writes"] == {"node_a": {"foo": "bar"}}

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


@pytest.mark.parametrize("checkpointer_name", REGULAR_CHECKPOINTERS_SYNC)
def test_update_as_input(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

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
        .compile(checkpointer=checkpointer)
    )

    assert graph.invoke({"foo": "input"}, {"configurable": {"thread_id": "1"}}) == {
        "foo": "tool"
    }

    assert graph.invoke({"foo": "input"}, {"configurable": {"thread_id": "1"}}) == {
        "foo": "tool"
    }

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

    assert new_history == history


@pytest.mark.parametrize("checkpointer_name", REGULAR_CHECKPOINTERS_SYNC)
def test_batch_update_as_input(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

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
        .compile(checkpointer=checkpointer)
    )

    assert graph.invoke({"foo": "input"}, {"configurable": {"thread_id": "1"}}) == {
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

    assert new_history == history
