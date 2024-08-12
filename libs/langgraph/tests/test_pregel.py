import json
import operator
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
    TypedDict,
    Union,
    get_type_hints,
)

import httpx
import pytest
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
    RunnablePick,
)
from langsmith import traceable
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion

from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.context import Context
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import Send
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, Graph
from langgraph.graph.graph import START
from langgraph.graph.message import MessageGraph, add_messages
from langgraph.graph.state import StateGraph
from langgraph.prebuilt.chat_agent_executor import (
    create_tool_calling_executor,
)
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.pregel import (
    INHERIT_CHECKPOINTER,
    Channel,
    GraphRecursionError,
    Pregel,
    StateSnapshot,
)
from langgraph.pregel.retry import RetryPolicy
from tests.any_str import AnyStr
from tests.memory_assert import (
    MemorySaverAssertCheckpointMetadata,
    MemorySaverAssertImmutable,
    MemorySaverNoPending,
    NoopSerializer,
)
from tests.messages import _AnyIdAIMessage, _AnyIdHumanMessage


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
    with pytest.raises(ValueError, match="not reachable"):
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
    workflow.add_conditional_edges("agent", logic, {"continue": "tools", "exit": END})
    workflow.add_edge("tools", "agent")
    with pytest.raises(
        ValueError, match="Node `extra` is not reachable"
    ):  # extra is not reachable
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

    def node_a(state: State) -> State:
        # typo
        return {"hell": "world"}

    builder = StateGraph(State)
    builder.add_node("a", node_a)
    builder.set_entry_point("a")
    builder.set_finish_point("a")
    graph = builder.compile()
    with pytest.raises(InvalidUpdateError):
        graph.invoke({"hello": "there"})

    graph = StateGraph(State)
    graph.add_node("start", lambda x: x)
    graph.add_edge("__start__", "start")
    graph.add_edge("unknown", "start")
    graph.add_edge("start", "__end__")
    with pytest.raises(ValueError, match="Found edge starting at unknown node "):
        graph.compile()

    def bad_reducer(a):
        ...

    class BadReducerState(TypedDict):
        hello: Annotated[str, bad_reducer]

    with pytest.raises(ValueError, match="Invalid reducer"):
        StateGraph(BadReducerState)


def test_checkpoint_errors() -> None:
    class FaultyGetCheckpointer(MemorySaver):
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

    class FaultyPutWritesCheckpointer(MemorySaver):
        def put_writes(
            self, config: RunnableConfig, writes: List[Tuple[str, Any]], task_id: str
        ) -> RunnableConfig:
            raise ValueError("Faulty put_writes")

    class FaultyVersionCheckpointer(MemorySaver):
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
            "now": None,
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

    builder = StateGraph(input=State, output=Output)
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

    assert app.input_schema.schema() == {"title": "LangGraphInput", "type": "integer"}
    assert app.output_schema.schema() == {"title": "LangGraphOutput", "type": "integer"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # raise warnings as errors
        assert app.config_schema().schema() == {
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

    assert app.input_schema.schema() == {"title": "LangGraphInput", "type": "integer"}
    assert app.output_schema.schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {
            "output": {"title": "Output", "type": "integer"},
            "fixed": {"title": "Fixed", "type": "integer"},
            "output_plus_one": {"title": "Output Plus One", "type": "integer"},
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

    assert app.input_schema.schema() == {"title": "LangGraphInput", "type": "integer"}
    assert app.output_schema.schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {"output": {"title": "Output", "type": "integer"}},
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

    assert app.input_schema.schema() == {
        "title": "LangGraphInput",
        "type": "object",
        "properties": {"input": {"title": "Input", "type": "integer"}},
    }
    assert app.output_schema.schema() == {
        "title": "LangGraphOutput",
        "type": "object",
        "properties": {"output": {"title": "Output", "type": "integer"}},
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


@pytest.mark.parametrize(
    "checkpointer_name",
    ["memory", "sqlite", "postgres", "postgres_pipe"],
)
def test_invoke_two_processes_in_out_interrupt(
    request: pytest.FixtureRequest, checkpointer_name: str, mocker: MockerFixture
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
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
        checkpointer=checkpointer,
        interrupt_after_nodes=["one"],
    )
    thread1 = {"configurable": {"thread_id": "1"}}
    thread2 = {"configurable": {"thread_id": "2"}}

    # start execution, stop at inbox
    assert app.invoke(2, thread1) is None

    # inbox == 3
    checkpoint = checkpointer.get(thread1)
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 3

    # resume execution, finish
    assert app.invoke(None, thread1) == 4

    # start execution again, stop at inbox
    assert app.invoke(20, thread1) is None

    # inbox == 21
    checkpoint = checkpointer.get(thread1)
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 21

    # send a new value in, interrupting the previous execution
    assert app.invoke(3, thread1) is None
    assert app.invoke(None, thread1) == 5

    # start execution again, stopping at inbox
    assert app.invoke(20, thread2) is None

    # inbox == 21
    snapshot = app.get_state(thread2)
    assert snapshot.values["inbox"] == 21
    assert snapshot.next == ("two",)

    # update the state, resume
    app.update_state(thread2, 25, as_node="one")
    assert app.invoke(None, thread2) == 26

    # no pending tasks
    snapshot = app.get_state(thread2)
    assert snapshot.next == ()

    # list history
    history = [c for c in app.get_state_history(thread1)]
    assert history == [
        StateSnapshot(
            values={"inbox": 4, "output": 5, "input": 3},
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "step": 6, "writes": 5},
            created_at=AnyStr(),
            parent_config=history[1].config,
        ),
        StateSnapshot(
            values={"inbox": 4, "output": 4, "input": 3},
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "step": 5, "writes": None},
            created_at=AnyStr(),
            parent_config=history[2].config,
        ),
        StateSnapshot(
            values={"inbox": 21, "output": 4, "input": 3},
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "input", "step": 4, "writes": 3},
            created_at=AnyStr(),
            parent_config=history[3].config,
        ),
        StateSnapshot(
            values={"inbox": 21, "output": 4, "input": 20},
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "step": 3, "writes": None},
            created_at=AnyStr(),
            parent_config=history[4].config,
        ),
        StateSnapshot(
            values={"inbox": 3, "output": 4, "input": 20},
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "input", "step": 2, "writes": 20},
            created_at=AnyStr(),
            parent_config=history[5].config,
        ),
        StateSnapshot(
            values={"inbox": 3, "output": 4, "input": 2},
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "step": 1, "writes": 4},
            created_at=AnyStr(),
            parent_config=history[6].config,
        ),
        StateSnapshot(
            values={"inbox": 3, "input": 2},
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "step": 0, "writes": None},
            created_at=AnyStr(),
            parent_config=history[7].config,
        ),
        StateSnapshot(
            values={"input": 2},
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "input", "step": -1, "writes": 2},
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]

    # forking from any previous checkpoint w/out forking should do nothing
    assert [c for c in app.stream(None, history[0].config, stream_mode="updates")] == []
    assert [c for c in app.stream(None, history[1].config, stream_mode="updates")] == []
    assert [c for c in app.stream(None, history[2].config, stream_mode="updates")] == []

    # forking and re-running from any prev checkpoint should re-run nodes
    fork_config = app.update_state(history[0].config, None)
    assert [c for c in app.stream(None, fork_config, stream_mode="updates")] == []

    fork_config = app.update_state(history[1].config, None)
    assert [c for c in app.stream(None, fork_config, stream_mode="updates")] == [
        {"two": {"output": 5}}
    ]

    fork_config = app.update_state(history[2].config, None)
    assert [c for c in app.stream(None, fork_config, stream_mode="updates")] == [
        {"one": {"inbox": 4}}
    ]


@pytest.mark.parametrize(
    "checkpointer_name",
    ["memory", "sqlite", "postgres", "postgres_pipe"],
)
def test_fork_always_re_runs_nodes(
    request: pytest.FixtureRequest, checkpointer_name: str, mocker: MockerFixture
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    add_one = mocker.Mock(side_effect=lambda _: 1)

    builder = StateGraph(Annotated[int, operator.add])
    builder.add_node("add_one", add_one)
    builder.add_edge(START, "add_one")
    builder.add_conditional_edges("add_one", lambda cnt: "add_one" if cnt < 6 else END)
    graph = builder.compile(checkpointer=checkpointer)

    thread1 = {"configurable": {"thread_id": "1"}}

    # start execution, stop at inbox
    assert [*graph.stream(1, thread1, stream_mode=["values", "updates"])] == [
        ("values", 1),
        ("updates", {"add_one": 1}),
        ("values", 2),
        ("updates", {"add_one": 1}),
        ("values", 3),
        ("updates", {"add_one": 1}),
        ("values", 4),
        ("updates", {"add_one": 1}),
        ("values", 5),
        ("updates", {"add_one": 1}),
        ("values", 6),
    ]

    # list history
    history = [c for c in graph.get_state_history(thread1)]
    assert history == [
        StateSnapshot(
            values=6,
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "step": 5, "writes": {"add_one": 1}},
            created_at=AnyStr(),
            parent_config=history[1].config,
        ),
        StateSnapshot(
            values=5,
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "step": 4, "writes": {"add_one": 1}},
            created_at=AnyStr(),
            parent_config=history[2].config,
        ),
        StateSnapshot(
            values=4,
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "step": 3, "writes": {"add_one": 1}},
            created_at=AnyStr(),
            parent_config=history[3].config,
        ),
        StateSnapshot(
            values=3,
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "step": 2, "writes": {"add_one": 1}},
            created_at=AnyStr(),
            parent_config=history[4].config,
        ),
        StateSnapshot(
            values=2,
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "step": 1, "writes": {"add_one": 1}},
            created_at=AnyStr(),
            parent_config=history[5].config,
        ),
        StateSnapshot(
            values=1,
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "step": 0, "writes": None},
            created_at=AnyStr(),
            parent_config=history[6].config,
        ),
        StateSnapshot(
            values=0,
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "input", "step": -1, "writes": 1},
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]

    # forking from any previous checkpoint w/out forking should do nothing
    assert [
        c for c in graph.stream(None, history[0].config, stream_mode="updates")
    ] == []
    assert [
        c for c in graph.stream(None, history[1].config, stream_mode="updates")
    ] == []

    # forking and re-running from any prev checkpoint should re-run nodes
    fork_config = graph.update_state(history[0].config, None)
    assert [c for c in graph.stream(None, fork_config, stream_mode="updates")] == []

    fork_config = graph.update_state(history[1].config, None)
    assert [c for c in graph.stream(None, fork_config, stream_mode="updates")] == [
        {"add_one": 1}
    ]

    fork_config = graph.update_state(history[2].config, None)
    assert [c for c in graph.stream(None, fork_config, stream_mode="updates")] == [
        {"add_one": 1},
        {"add_one": 1},
    ]


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
                "id": "2687f72c-e3a8-5f6f-9afa-047cbf24e923",
                "name": "one",
                "input": 2,
                "triggers": ["input"],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "id": "18f52f6a-828d-58a1-a501-53cc0c7af33e",
                "name": "two",
                "input": [12],
                "triggers": ["inbox"],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "id": "2687f72c-e3a8-5f6f-9afa-047cbf24e923",
                "name": "one",
                "result": [("inbox", 3)],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "id": "18f52f6a-828d-58a1-a501-53cc0c7af33e",
                "name": "two",
                "result": [("output", 13)],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": "871d6e74-7bb3-565f-a4fe-cef4b8f19b62",
                "name": "two",
                "input": [3],
                "triggers": ["inbox"],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": "871d6e74-7bb3-565f-a4fe-cef4b8f19b62",
                "name": "two",
                "result": [("output", 4)],
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


def test_invoke_checkpoint(mocker: MockerFixture) -> None:
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

    memory = MemorySaverAssertImmutable()

    app = Pregel(
        nodes={"one": one},
        channels={
            "total": BinaryOperatorAggregate(int, operator.add),
            "input": LastValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
        checkpointer=memory,
        retry_policy=RetryPolicy(),
    )

    # total starts out as 0, so output is 0+2=2
    assert app.invoke(2, {"configurable": {"thread_id": "1"}}) == 2
    checkpoint = memory.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 2
    # total is now 2, so output is 2+3=5
    assert app.invoke(3, {"configurable": {"thread_id": "1"}}) == 5
    assert errored_once, "errored and retried"
    checkpoint = memory.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        app.invoke(4, {"configurable": {"thread_id": "1"}})
    # checkpoint is not updated
    checkpoint = memory.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert app.invoke(5, {"configurable": {"thread_id": "2"}}) == 5
    checkpoint = memory.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    checkpoint = memory.get({"configurable": {"thread_id": "2"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 5


@pytest.mark.parametrize(
    "checkpointer_name",
    ["memory", "sqlite", "postgres", "postgres_pipe"],
)
def test_pending_writes_resume(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

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

    one = AwhileMaker(0.2, {"value": 2})
    two = AwhileMaker(0.6, ConnectionError("I'm not good"))
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
    state = graph.get_state(thread1)
    assert state is not None
    assert state.values == {"value": 1}
    assert state.next == ("one", "two")
    assert state.metadata == {"source": "loop", "step": 0, "writes": None}
    # should contain pending write of "one"
    checkpoint = checkpointer.get_tuple(thread1)
    assert checkpoint is not None
    assert checkpoint.pending_writes == [
        (AnyStr(), "one", "one"),
        (AnyStr(), "value", 2),
    ]
    # both pending writes come from same task
    assert checkpoint.pending_writes[0][0] == checkpoint.pending_writes[1][0]

    # resume execution
    with pytest.raises(ConnectionError, match="I'm not good"):
        graph.invoke(None, thread1)

    # node "one" succeeded previously, so shouldn't be called again
    assert one.calls == 1
    # node "two" should have been called once again
    assert two.calls == 4  # two attempts before + two attempts now

    # confirm no new checkpoints saved
    state_two = graph.get_state(thread1)
    assert state_two == state

    # resume execution, without exception
    two.rtn = {"value": 3}
    # both the pending write and the new write were applied, 1 + 2 + 3 = 6
    assert graph.invoke(None, thread1) == {"value": 6}


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


async def test_checkpointer_null_pending_writes() -> None:
    class Node:
        def __init__(self, name: str):
            self.name = name
            setattr(self, "__name__", name)

        def __call__(self, state):
            return [self.name]

    builder = StateGraph(Annotated[list, operator.add])
    builder.add_node(Node("1"))
    builder.add_edge(START, "1")
    graph = builder.compile(checkpointer=MemorySaverNoPending())
    assert graph.invoke([], {"configurable": {"thread_id": "foo"}}) == ["1"]
    assert graph.invoke([], {"configurable": {"thread_id": "foo"}}) == ["1"] * 2
    assert (await graph.ainvoke([], {"configurable": {"thread_id": "foo"}})) == [
        "1"
    ] * 3
    assert (await graph.ainvoke([], {"configurable": {"thread_id": "foo"}})) == [
        "1"
    ] * 4


def test_invoke_checkpoint_sqlite(mocker: MockerFixture) -> None:
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

    with SqliteSaver.from_conn_string(":memory:") as memory:
        app = Pregel(
            nodes={"one": one},
            channels={
                "total": BinaryOperatorAggregate(int, operator.add),
                "input": LastValue(int),
                "output": LastValue(int),
            },
            input_channels="input",
            output_channels="output",
            checkpointer=memory,
        )

        thread_1 = {"configurable": {"thread_id": "1"}}
        # total starts out as 0, so output is 0+2=2
        assert app.invoke(2, thread_1, debug=1) == 2
        state = app.get_state(thread_1)
        assert state is not None
        assert state.values.get("total") == 2
        assert state.next == ()
        assert (
            state.config["configurable"]["checkpoint_id"] == memory.get(thread_1)["id"]
        )
        # total is now 2, so output is 2+3=5
        assert app.invoke(3, thread_1) == 5
        state = app.get_state(thread_1)
        assert state is not None
        assert state.values.get("total") == 7
        assert (
            state.config["configurable"]["checkpoint_id"] == memory.get(thread_1)["id"]
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
            memory.get(thread_1_history[0].config)["id"]
            == thread_1_history[0].config["configurable"]["checkpoint_id"]
        )
        assert (
            memory.get(thread_1_history[1].config)["id"]
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
            app.get_state(thread_1_next_config).parent_config
            == thread_1_history[1].config
        )
        # 1 more checkpoint in history
        assert len(list(app.get_state_history(thread_1))) == 8
        assert Counter(
            c.metadata["source"] for c in app.get_state_history(thread_1)
        ) == {
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


def test_invoke_join_then_call_other_pregel(mocker: MockerFixture) -> None:
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


def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)

    one = (
        Channel.subscribe_to("input")
        | add_one
        | Channel.write_to(output=RunnablePassthrough(), between=RunnablePassthrough())
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

    with pytest.raises(ValueError):
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
        assert cleanup.call_count == 0, "Expected cleanup to not be called yet"
        if i == 0:
            assert chunk == {"inbox": [3]}
        elif i == 1:
            assert chunk == {"output": 4}
        else:
            assert False, "Expected only two chunks"
    assert cleanup.call_count == 1, "Expected cleanup to be called once"


def test_conditional_graph(snapshot: SnapshotAssertion) -> None:
    from copy import deepcopy

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.tools import tool

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

    def agent_parser(input: str) -> Union[AgentAction, AgentFinish]:
        if input.startswith("finish"):
            _, answer = input.split(":")
            return AgentFinish(return_values={"answer": answer}, log=input)
        else:
            _, tool_name, tool_input = input.split(":")
            return AgentAction(tool=tool_name, tool_input=tool_input, log=input)

    agent = RunnablePassthrough.assign(agent_outcome=prompt | llm | agent_parser)

    # Define tool execution logic
    def execute_tools(data: dict) -> dict:
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = {t.name: t for t in tools}[agent_action.tool].invoke(
            agent_action.tool_input
        )
        if data.get("intermediate_steps") is None:
            data["intermediate_steps"] = []
        data["intermediate_steps"].append((agent_action, observation))
        return data

    # Define decision-making logic
    def should_continue(data: dict) -> str:
        # Logic to decide whether to continue in the loop or exit
        if isinstance(data["agent_outcome"], AgentFinish):
            return "exit"
        else:
            return "continue"

    # Define a new graph
    workflow = Graph()

    workflow.add_node("agent", agent)
    workflow.add_node("tools", execute_tools, metadata={"version": 2, "variant": "b"})

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
    assert app.get_graph().draw_mermaid() == snapshot
    assert json.dumps(app.get_graph(xray=True).to_json(), indent=2) == snapshot
    assert app.get_graph(xray=True).draw_mermaid(with_styles=False) == snapshot

    assert app.invoke({"input": "what is weather in sf"}) == {
        "input": "what is weather in sf",
        "intermediate_steps": [
            (
                AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:query",
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
        "agent_outcome": AgentFinish(
            return_values={"answer": "answer"}, log="finish:answer"
        ),
    }

    # deepcopy because the nodes mutate the data
    assert [deepcopy(c) for c in app.stream({"input": "what is weather in sf"})] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
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
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
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
                "agent_outcome": AgentFinish(
                    return_values={"answer": "answer"}, log="finish:answer"
                ),
            }
        },
    ]

    # test state get/update methods with interrupt_after

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert app_w_interrupt.get_graph().to_json() == snapshot
    assert app_w_interrupt.get_graph().draw_mermaid() == snapshot

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        }
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        next=("tools",),
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        metadata={
            "source": "loop",
            "step": 0,
            "writes": {
                "agent": {
                    "input": "what is weather in sf",
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                },
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )
    assert (
        app_w_interrupt.checkpointer.get_tuple(config).config["configurable"][
            "checkpoint_id"
        ]
        is not None
    )

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "input": "what is weather in sf",
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            },
        },
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "input": "what is weather in sf",
                },
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    app_w_interrupt.update_state(
        config,
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                )
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                ),
            },
        },
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 4,
            "writes": {
                "agent": {
                    "input": "what is weather in sf",
                    "intermediate_steps": [
                        (
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:a different query",
                            ),
                            "result for query",
                        )
                    ],
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    ),
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        }
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 0,
            "writes": {
                "agent": {
                    "input": "what is weather in sf",
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "input": "what is weather in sf",
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            },
        },
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "input": "what is weather in sf",
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    app_w_interrupt.update_state(
        config,
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                )
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                ),
            },
        },
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 4,
            "writes": {
                "agent": {
                    "input": "what is weather in sf",
                    "intermediate_steps": [
                        (
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:a different query",
                            ),
                            "result for query",
                        )
                    ],
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    ),
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # test re-invoke to continue with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        }
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 0,
            "writes": {
                "agent": {
                    "input": "what is weather in sf",
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    )
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
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
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
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
                "agent_outcome": AgentFinish(
                    return_values={"answer": "answer"}, log="finish:answer"
                ),
            }
        },
    ]


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

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
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

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
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


def test_conditional_state_graph(
    snapshot: SnapshotAssertion, mocker: MockerFixture
) -> None:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import tool

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

    class AgentState(TypedDict, total=False):
        input: Annotated[str, UntrackedValue]
        agent_outcome: Optional[Union[AgentAction, AgentFinish]]
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
        session: Annotated[httpx.Client, Context(make_httpx_client)]

    class ToolState(TypedDict, total=False):
        agent_outcome: Union[AgentAction, AgentFinish]
        session: Annotated[httpx.Client, Context(make_httpx_client)]

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
    def execute_tools(data: ToolState) -> dict:
        # check session in data
        assert isinstance(data["session"], httpx.Client)
        assert "input" not in data
        assert "intermediate_steps" not in data
        # execute the tool
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = {t.name: t for t in tools}[agent_action.tool].invoke(
            agent_action.tool_input
        )
        return {"intermediate_steps": [(agent_action, observation)]}

    # Define decision-making logic
    def should_continue(data: AgentState) -> str:
        # check session in data
        assert isinstance(data["session"], httpx.Client)
        # Logic to decide whether to continue in the loop or exit
        if isinstance(data["agent_outcome"], AgentFinish):
            return "exit"
        else:
            return "continue"

    # Define a new graph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent)
    workflow.add_node("tools", execute_tools, input=ToolState)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    with assert_ctx_once():
        assert app.invoke({"input": "what is weather in sf"}) == {
            "input": "what is weather in sf",
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
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
            "agent_outcome": AgentFinish(
                return_values={"answer": "answer"}, log="finish:answer"
            ),
        }

    with assert_ctx_once():
        assert [*app.stream({"input": "what is weather in sf"})] == [
            {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
            {
                "tools": {
                    "intermediate_steps": [
                        (
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:query",
                            ),
                            "result for query",
                        )
                    ],
                }
            },
            {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="another",
                        log="tool:search_api:another",
                    ),
                }
            },
            {
                "tools": {
                    "intermediate_steps": [
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
            },
            {
                "agent": {
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "answer"}, log="finish:answer"
                    ),
                }
            },
        ]

    # test state get/update methods with interrupt_after

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    with assert_ctx_once():
        assert [
            c
            for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
        ] == [
            {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
        ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    with assert_ctx_once():
        app_w_interrupt.update_state(
            config,
            {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                )
            },
        )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 2,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    )
                },
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    with assert_ctx_once():
        assert [c for c in app_w_interrupt.stream(None, config)] == [
            {
                "tools": {
                    "intermediate_steps": [
                        (
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:a different query",
                            ),
                            "result for query",
                        )
                    ],
                }
            },
            {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="another",
                        log="tool:search_api:another",
                    ),
                }
            },
        ]

    with assert_ctx_once():
        app_w_interrupt.update_state(
            config,
            {
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                )
            },
        )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                )
            ],
        },
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 5,
            "writes": {
                "agent": {
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    )
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_before=["tools"],
        debug=True,
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            )
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 2,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    )
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            )
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                )
            ],
        },
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 5,
            "writes": {
                "agent": {
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    )
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # test w interrupt before all
    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_before="*",
        debug=True,
    )
    config = {"configurable": {"thread_id": "3"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == []

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "intermediate_steps": [],
        },
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={"source": "loop", "step": 0, "writes": None},
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                    "result for query",
                )
            ],
        },
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 2,
            "writes": {
                "tools": {
                    "intermediate_steps": [
                        (
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:query",
                            ),
                            "result for query",
                        )
                    ],
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    # test w interrupt after all
    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_after="*",
    )
    config = {"configurable": {"thread_id": "4"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "intermediate_steps": [
                    (
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    )
                ],
            }
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [
                (
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                    "result for query",
                )
            ],
        },
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 2,
            "writes": {
                "tools": {
                    "intermediate_steps": [
                        (
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:query",
                            ),
                            "result for query",
                        )
                    ],
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]


def test_state_graph_w_config_inherited_state_keys(snapshot: SnapshotAssertion) -> None:
    from langchain_core.agents import AgentAction, AgentFinish
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

    assert app.config_schema().schema_json() == snapshot
    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot

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

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
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


def test_prebuilt_tool_chat(snapshot: SnapshotAssertion) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_tools(self, functions: list):
            return self

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    model = FakeFuntionChatModel(
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
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call234",
                        "name": "search_api",
                        "args": {"query": "another"},
                    },
                    {
                        "id": "tool_call567",
                        "name": "search_api",
                        "args": {"query": "a third one"},
                    },
                ],
            ),
            AIMessage(content="answer"),
        ]
    )

    app = create_tool_calling_executor(model, tools)

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke(
        {"messages": [HumanMessage(content="what is weather in sf")]}
    ) == {
        "messages": [
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                id=AnyStr(),
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    },
                ],
            ),
            ToolMessage(
                content="result for query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(
                id=AnyStr(),
                content="",
                tool_calls=[
                    {
                        "id": "tool_call234",
                        "name": "search_api",
                        "args": {"query": "another"},
                    },
                    {
                        "id": "tool_call567",
                        "name": "search_api",
                        "args": {"query": "a third one"},
                    },
                ],
            ),
            ToolMessage(
                content="result for another",
                name="search_api",
                tool_call_id="tool_call234",
                id=AnyStr(),
            ),
            ToolMessage(
                content="result for a third one",
                name="search_api",
                tool_call_id="tool_call567",
                id=AnyStr(),
            ),
            _AnyIdAIMessage(content="answer"),
        ]
    }

    assert app.invoke(
        {"messages": [HumanMessage(content="what is weather in sf")]},
        {"recursion_limit": 2},
        debug=True,
    ) == {
        "messages": [
            _AnyIdHumanMessage(content="what is weather in sf"),
            _AnyIdAIMessage(content="Sorry, need more steps to process this request."),
        ]
    }

    model.i = 0  # reset the model

    assert app.invoke(
        {"messages": [HumanMessage(content="what is weather in sf")]},
        stream_mode="updates",
    ) == [
        {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call123",
                                "name": "search_api",
                                "args": {"query": "query"},
                            },
                        ],
                        id=AnyStr(),
                    )
                ]
            }
        },
        {
            "tools": {
                "messages": [
                    ToolMessage(
                        content="result for query",
                        name="search_api",
                        tool_call_id="tool_call123",
                        id=AnyStr(),
                    )
                ]
            }
        },
        {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call234",
                                "name": "search_api",
                                "args": {"query": "another"},
                            },
                            {
                                "id": "tool_call567",
                                "name": "search_api",
                                "args": {"query": "a third one"},
                            },
                        ],
                        id=AnyStr(),
                    )
                ]
            }
        },
        {
            "tools": {
                "messages": [
                    ToolMessage(
                        content="result for another",
                        name="search_api",
                        tool_call_id="tool_call234",
                        id=AnyStr(),
                    ),
                    ToolMessage(
                        content="result for a third one",
                        name="search_api",
                        tool_call_id="tool_call567",
                        id=AnyStr(),
                    ),
                ]
            }
        },
        {"agent": {"messages": [_AnyIdAIMessage(content="answer")]}},
    ]

    assert [
        *app.stream({"messages": [HumanMessage(content="what is weather in sf")]})
    ] == [
        {
            "agent": {
                "messages": [
                    AIMessage(
                        id=AnyStr(),
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call123",
                                "name": "search_api",
                                "args": {"query": "query"},
                            },
                        ],
                    )
                ]
            }
        },
        {
            "tools": {
                "messages": [
                    ToolMessage(
                        content="result for query",
                        name="search_api",
                        tool_call_id="tool_call123",
                        id=AnyStr(),
                    )
                ]
            }
        },
        {
            "agent": {
                "messages": [
                    AIMessage(
                        id=AnyStr(),
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call234",
                                "name": "search_api",
                                "args": {"query": "another"},
                            },
                            {
                                "id": "tool_call567",
                                "name": "search_api",
                                "args": {"query": "a third one"},
                            },
                        ],
                    )
                ]
            }
        },
        {
            "tools": {
                "messages": [
                    ToolMessage(
                        content="result for another",
                        name="search_api",
                        tool_call_id="tool_call234",
                        id=AnyStr(),
                    ),
                    ToolMessage(
                        content="result for a third one",
                        name="search_api",
                        tool_call_id="tool_call567",
                        id=AnyStr(),
                    ),
                ]
            }
        },
        {"agent": {"messages": [_AnyIdAIMessage(content="answer")]}},
    ]


@pytest.mark.parametrize("serde", [NoopSerializer(), JsonPlusSerializer()])
def test_state_graph_packets(serde: SerializerProtocol) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolCall,
        ToolMessage,
    )
    from langchain_core.tools import tool

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]
    tools_by_name = {t.name: t for t in tools}

    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    },
                ],
            ),
            AIMessage(
                id="ai2",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call234",
                        "name": "search_api",
                        "args": {"query": "another", "idx": 0},
                    },
                    {
                        "id": "tool_call567",
                        "name": "search_api",
                        "args": {"query": "a third one", "idx": 1},
                    },
                ],
            ),
            AIMessage(id="ai3", content="answer"),
        ]
    )

    def agent(data: AgentState) -> AgentState:
        return {
            "messages": model.invoke(data["messages"]),
            "something_extra": "hi there",
        }

    # Define decision-making logic
    def should_continue(data: AgentState) -> str:
        assert (
            data["something_extra"] == "hi there"
        ), "nodes can pass extra data to their cond edges, which isn't saved in state"
        # Logic to decide whether to continue in the loop or exit
        if tool_calls := data["messages"][-1].tool_calls:
            return [Send("tools", tool_call) for tool_call in tool_calls]
        else:
            return END

    def tools_node(tool_call: ToolCall, config: RunnableConfig) -> AgentState:
        time.sleep(tool_call["args"].get("idx", 0) / 10)
        output = tools_by_name[tool_call["name"]].invoke(tool_call["args"], config)
        return {
            "messages": ToolMessage(
                content=output, name=tool_call["name"], tool_call_id=tool_call["id"]
            )
        }

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tools_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges("agent", should_continue)

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    assert app.invoke({"messages": HumanMessage(content="what is weather in sf")}) == {
        "messages": [
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    },
                ],
            ),
            ToolMessage(
                content="result for query",
                name="search_api",
                id=AnyStr(),
                tool_call_id="tool_call123",
            ),
            AIMessage(
                id="ai2",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call234",
                        "name": "search_api",
                        "args": {"query": "another", "idx": 0},
                    },
                    {
                        "id": "tool_call567",
                        "name": "search_api",
                        "args": {"query": "a third one", "idx": 1},
                    },
                ],
            ),
            ToolMessage(
                content="result for another",
                name="search_api",
                id=AnyStr(),
                tool_call_id="tool_call234",
            ),
            ToolMessage(
                content="result for a third one",
                name="search_api",
                id=AnyStr(),
                tool_call_id="tool_call567",
            ),
            AIMessage(content="answer", id="ai3"),
        ]
    }

    assert [
        c
        for c in app.stream(
            {"messages": [HumanMessage(content="what is weather in sf")]}
        )
    ] == [
        {
            "agent": {
                "messages": AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                )
            },
        },
        {
            "tools": {
                "messages": ToolMessage(
                    content="result for query",
                    name="search_api",
                    id=AnyStr(),
                    tool_call_id="tool_call123",
                )
            }
        },
        {
            "agent": {
                "messages": AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                )
            }
        },
        {
            "tools": {
                "messages": ToolMessage(
                    content="result for another",
                    name="search_api",
                    id=AnyStr(),
                    tool_call_id="tool_call234",
                )
            },
        },
        {
            "tools": {
                "messages": ToolMessage(
                    content="result for a third one",
                    name="search_api",
                    id=AnyStr(),
                    tool_call_id="tool_call567",
                ),
            },
        },
        {"agent": {"messages": AIMessage(content="answer", id="ai3")}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(serde=serde),
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        for c in app_w_interrupt.stream(
            {"messages": HumanMessage(content="what is weather in sf")}, config
        )
    ] == [
        {
            "agent": {
                "messages": AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                )
            }
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                ),
            ]
        },
        next=("tools",),
        config=(app_w_interrupt.checkpointer.get_tuple(config)).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": {
                    "messages": AIMessage(
                        id="ai1",
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call123",
                                "name": "search_api",
                                "args": {"query": "query"},
                            },
                        ],
                    )
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # modify ai message
    last_message = (app_w_interrupt.get_state(config)).values["messages"][-1]
    last_message.tool_calls[0]["args"]["query"] = "a different query"
    app_w_interrupt.update_state(
        config, {"messages": last_message, "something_extra": "hi there"}
    )

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
            ]
        },
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 2,
            "writes": {
                "agent": {
                    "messages": AIMessage(
                        id="ai1",
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call123",
                                "name": "search_api",
                                "args": {"query": "a different query"},
                            },
                        ],
                    ),
                    "something_extra": "hi there",
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "messages": ToolMessage(
                    content="result for a different query",
                    name="search_api",
                    id=AnyStr(),
                    tool_call_id="tool_call123",
                )
            }
        },
        {
            "agent": {
                "messages": AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                )
            },
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
                ToolMessage(
                    content="result for a different query",
                    name="search_api",
                    id=AnyStr(),
                    tool_call_id="tool_call123",
                ),
                AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                ),
            ]
        },
        next=("tools", "tools"),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 4,
            "writes": {
                "agent": {
                    "messages": AIMessage(
                        id="ai2",
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call234",
                                "name": "search_api",
                                "args": {"query": "another", "idx": 0},
                            },
                            {
                                "id": "tool_call567",
                                "name": "search_api",
                                "args": {"query": "a third one", "idx": 1},
                            },
                        ],
                    )
                },
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        {
            "messages": AIMessage(content="answer", id="ai2"),
            "something_extra": "hi there",
        },
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
                ToolMessage(
                    content="result for a different query",
                    name="search_api",
                    id=AnyStr(),
                    tool_call_id="tool_call123",
                ),
                AIMessage(content="answer", id="ai2"),
            ]
        },
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 5,
            "writes": {
                "agent": {
                    "messages": AIMessage(content="answer", id="ai2"),
                    "something_extra": "hi there",
                }
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )


def test_message_graph(
    snapshot: SnapshotAssertion,
    deterministic_uuids: MockerFixture,
) -> None:
    from copy import deepcopy

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list):
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = deepcopy(self.responses[self.i])
            if self.i < len(self.responses) - 1:
                self.i += 1
            else:
                self.i = 0
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation])

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    model = FakeFuntionChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
            AIMessage(content="answer", id="ai3"),
        ]
    )

    # Define the function that determines whether to continue or not
    def should_continue(messages):
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    # Define a new graph
    workflow = MessageGraph()

    # Define the two nodes we will cycle between
    workflow.add_node("agent", model)
    workflow.add_node("tools", ToolNode(tools))

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    assert app.get_input_schema().schema_json() == snapshot
    assert app.get_output_schema().schema_json() == snapshot
    assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke(HumanMessage(content="what is weather in sf")) == [
        HumanMessage(
            content="what is weather in sf",
            id="00000000-0000-4000-8000-000000000002",  # adds missing ids
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call123",
                    "name": "search_api",
                    "args": {"query": "query"},
                }
            ],
            id="ai1",  # respects ids passed in
        ),
        ToolMessage(
            content="result for query",
            name="search_api",
            tool_call_id="tool_call123",
            id="00000000-0000-4000-8000-000000000011",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call456",
                    "name": "search_api",
                    "args": {"query": "another"},
                }
            ],
            id="ai2",
        ),
        ToolMessage(
            content="result for another",
            name="search_api",
            tool_call_id="tool_call456",
            id="00000000-0000-4000-8000-000000000020",
        ),
        AIMessage(content="answer", id="ai3"),
    ]

    assert [*app.stream([HumanMessage(content="what is weather in sf")])] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
        {
            "tools": [
                ToolMessage(
                    content="result for query",
                    name="search_api",
                    tool_call_id="tool_call123",
                    id="00000000-0000-4000-8000-000000000036",
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
        {
            "tools": [
                ToolMessage(
                    content="result for another",
                    name="search_api",
                    tool_call_id="tool_call456",
                    id="00000000-0000-4000-8000-000000000045",
                )
            ]
        },
        {"agent": AIMessage(content="answer", id="ai3")},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream(("human", "what is weather in sf"), config)
    ] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
        ],
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        }
                    ],
                    id="ai1",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # modify ai message
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    next_config = app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
        ],
        next=("tools",),
        config=next_config,
        created_at=AnyStr(),
        metadata={
            "source": "update",
            "step": 2,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        }
                    ],
                    id="ai1",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": [
                ToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                    id=AnyStr(),
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            ToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
        ],
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 4,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call456",
                            "name": "search_api",
                            "args": {"query": "another"},
                        }
                    ],
                    id="ai2",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),  # replace existing message
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            ToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 5,
            "writes": {"agent": AIMessage(content="answer", id="ai2")},
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    model.i = 0  # reset the llm

    assert [c for c in app_w_interrupt.stream("what is weather in sf", config)] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
        ],
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        }
                    ],
                    id="ai1",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # modify ai message
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
        ],
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 2,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        }
                    ],
                    id="ai1",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": [
                ToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                    id=AnyStr(),
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            ToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
        ],
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 4,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call456",
                            "name": "search_api",
                            "args": {"query": "another"},
                        }
                    ],
                    id="ai2",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            ToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 5,
            "writes": {"agent": AIMessage(content="answer", id="ai2")},
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # add an extra message as if it came from "tools" node
    app_w_interrupt.update_state(config, ("ai", "an extra message"), as_node="tools")

    # extra message is coerced BaseMessge and appended
    # now the next node is "agent" per the graph edges
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            ToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
            _AnyIdAIMessage(content="an extra message"),
        ],
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 6,
            "writes": {"tools": ("ai", "an extra message")},
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )


def test_root_graph(
    snapshot: SnapshotAssertion,
    deterministic_uuids: MockerFixture,
) -> None:
    from copy import deepcopy

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list):
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = deepcopy(self.responses[self.i])
            if self.i < len(self.responses) - 1:
                self.i += 1
            else:
                self.i = 0
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation])

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    model = FakeFuntionChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
            AIMessage(content="answer", id="ai3"),
        ]
    )

    # Define the function that determines whether to continue or not
    def should_continue(messages):
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    class State(TypedDict):
        __root__: Annotated[list[BaseMessage], add_messages]

    # Define a new graph
    workflow = StateGraph(State)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", model)
    workflow.add_node("tools", ToolNode(tools))

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    assert app.invoke(HumanMessage(content="what is weather in sf")) == [
        HumanMessage(
            content="what is weather in sf",
            id="00000000-0000-4000-8000-000000000002",  # adds missing ids
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call123",
                    "name": "search_api",
                    "args": {"query": "query"},
                }
            ],
            id="ai1",  # respects ids passed in
        ),
        ToolMessage(
            content="result for query",
            name="search_api",
            tool_call_id="tool_call123",
            id="00000000-0000-4000-8000-000000000011",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call456",
                    "name": "search_api",
                    "args": {"query": "another"},
                }
            ],
            id="ai2",
        ),
        ToolMessage(
            content="result for another",
            name="search_api",
            tool_call_id="tool_call456",
            id="00000000-0000-4000-8000-000000000020",
        ),
        AIMessage(content="answer", id="ai3"),
    ]

    assert [*app.stream([HumanMessage(content="what is weather in sf")])] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
        {
            "tools": [
                ToolMessage(
                    content="result for query",
                    name="search_api",
                    tool_call_id="tool_call123",
                    id="00000000-0000-4000-8000-000000000036",
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
        {
            "tools": [
                ToolMessage(
                    content="result for another",
                    name="search_api",
                    tool_call_id="tool_call456",
                    id="00000000-0000-4000-8000-000000000045",
                )
            ]
        },
        {"agent": AIMessage(content="answer", id="ai3")},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream(("human", "what is weather in sf"), config)
    ] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
        ],
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        }
                    ],
                    id="ai1",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # modify ai message
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    next_config = app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
        ],
        next=("tools",),
        config=next_config,
        created_at=AnyStr(),
        metadata={
            "source": "update",
            "step": 2,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        }
                    ],
                    id="ai1",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": [
                ToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                    id=AnyStr(),
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            ToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
        ],
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 4,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call456",
                            "name": "search_api",
                            "args": {"query": "another"},
                        }
                    ],
                    id="ai2",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),  # replace existing message
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            ToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 5,
            "writes": {"agent": AIMessage(content="answer", id="ai2")},
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    model.i = 0  # reset the llm

    assert [c for c in app_w_interrupt.stream("what is weather in sf", config)] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
        ],
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        }
                    ],
                    id="ai1",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # modify ai message
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
        ],
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 2,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        }
                    ],
                    id="ai1",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": [
                ToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                    id=AnyStr(),
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            ToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
        ],
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "loop",
            "step": 4,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call456",
                            "name": "search_api",
                            "args": {"query": "another"},
                        }
                    ],
                    id="ai2",
                )
            },
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            ToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 5,
            "writes": {"agent": AIMessage(content="answer", id="ai2")},
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # add an extra message as if it came from "tools" node
    app_w_interrupt.update_state(config, ("ai", "an extra message"), as_node="tools")

    # extra message is coerced BaseMessge and appended
    # now the next node is "agent" per the graph edges
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            ToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
            _AnyIdAIMessage(content="an extra message"),
        ],
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 6,
            "writes": {"tools": ("ai", "an extra message")},
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # create new graph with one more state key, reuse previous thread history

    def simple_add(left, right):
        if not isinstance(right, list):
            right = [right]
        return left + right

    class MoreState(TypedDict):
        __root__: Annotated[list[BaseMessage], simple_add]
        something_else: str

    # Define a new graph
    new_workflow = StateGraph(MoreState)
    new_workflow.add_node(
        "agent", RunnableMap(__root__=RunnablePick("__root__") | model)
    )
    new_workflow.add_node(
        "tools", RunnableMap(__root__=RunnablePick("__root__") | ToolNode(tools))
    )
    new_workflow.set_entry_point("agent")
    new_workflow.add_conditional_edges(
        "agent",
        RunnablePick("__root__") | should_continue,
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )
    new_workflow.add_edge("tools", "agent")
    new_app = new_workflow.compile(checkpointer=app_w_interrupt.checkpointer)
    model.i = 0  # reset the llm

    # previous state is converted to new schema
    assert new_app.get_state(config) == StateSnapshot(
        values={
            "__root__": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    content="",
                    id="ai1",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        }
                    ],
                ),
                ToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                    id=AnyStr(),
                ),
                AIMessage(content="answer", id="ai2"),
                _AnyIdAIMessage(content="an extra message"),
            ]
        },
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 6,
            "writes": {"tools": ("ai", "an extra message")},
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # new input is merged to old state
    assert new_app.invoke(
        {
            "__root__": [HumanMessage(content="what is weather in la")],
            "something_else": "value",
        },
        config,
        interrupt_before=["agent"],
    ) == {
        "__root__": [
            HumanMessage(
                content="what is weather in sf",
                id="00000000-0000-4000-8000-000000000077",
            ),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "name": "search_api",
                        "args": {"query": "a different query"},
                        "id": "tool_call123",
                    }
                ],
            ),
            ToolMessage(
                content="result for a different query",
                name="search_api",
                id="00000000-0000-4000-8000-000000000091",
                tool_call_id="tool_call123",
            ),
            AIMessage(content="answer", id="ai2"),
            AIMessage(
                content="an extra message", id="00000000-0000-4000-8000-000000000101"
            ),
            HumanMessage(content="what is weather in la"),
        ],
        "something_else": "value",
    }


def test_in_one_fan_out_out_one_graph_state() -> None:
    def sorted_add(x: list[str], y: list[str]) -> list[str]:
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    def retriever_one(data: State) -> State:
        # timer ensures stream output order is stable
        # also, it confirms that the update order is not dependent on finishing order
        # instead being defined by the order of the nodes/edges in the graph definition
        # ie. stable between invocations
        time.sleep(0.1)
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "retriever_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    workflow.add_edge("retriever_one", "qa")
    workflow.add_edge("retriever_two", "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    assert [*app.stream({"query": "what is weather in sf"}, stream_mode="values")] == [
        {"query": "what is weather in sf", "docs": []},
        {"query": "query: what is weather in sf", "docs": []},
        {
            "query": "query: what is weather in sf",
            "docs": ["doc1", "doc2", "doc3", "doc4"],
        },
        {
            "query": "query: what is weather in sf",
            "docs": ["doc1", "doc2", "doc3", "doc4"],
            "answer": "doc1,doc2,doc3,doc4",
        },
    ]

    assert [
        *app.stream(
            {"query": "what is weather in sf"},
            stream_mode=["values", "updates", "debug"],
        )
    ] == [
        ("values", {"query": "what is weather in sf", "docs": []}),
        (
            "debug",
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "id": "592f3430-c17c-5d1c-831f-fecebb2c05bf",
                    "name": "rewrite_query",
                    "input": {
                        "query": "what is weather in sf",
                        "answer": None,
                        "docs": [],
                    },
                    "triggers": ["start:rewrite_query"],
                },
            },
        ),
        ("updates", {"rewrite_query": {"query": "query: what is weather in sf"}}),
        (
            "debug",
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "id": "592f3430-c17c-5d1c-831f-fecebb2c05bf",
                    "name": "rewrite_query",
                    "result": [("query", "query: what is weather in sf")],
                },
            },
        ),
        ("values", {"query": "query: what is weather in sf", "docs": []}),
        (
            "debug",
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": "7db5e9d8-e132-5079-ab99-ced15e67d48b",
                    "name": "retriever_one",
                    "input": {
                        "query": "query: what is weather in sf",
                        "answer": None,
                        "docs": [],
                    },
                    "triggers": ["rewrite_query"],
                },
            },
        ),
        (
            "debug",
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": "96965ed0-2c10-52a1-86eb-081ba6de73b2",
                    "name": "retriever_two",
                    "input": {
                        "query": "query: what is weather in sf",
                        "answer": None,
                        "docs": [],
                    },
                    "triggers": ["rewrite_query"],
                },
            },
        ),
        (
            "updates",
            {"retriever_two": {"docs": ["doc3", "doc4"]}},
        ),
        (
            "debug",
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": "96965ed0-2c10-52a1-86eb-081ba6de73b2",
                    "name": "retriever_two",
                    "result": [("docs", ["doc3", "doc4"])],
                },
            },
        ),
        (
            "updates",
            {"retriever_one": {"docs": ["doc1", "doc2"]}},
        ),
        (
            "debug",
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": "7db5e9d8-e132-5079-ab99-ced15e67d48b",
                    "name": "retriever_one",
                    "result": [("docs", ["doc1", "doc2"])],
                },
            },
        ),
        (
            "values",
            {
                "query": "query: what is weather in sf",
                "docs": ["doc1", "doc2", "doc3", "doc4"],
            },
        ),
        (
            "debug",
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 3,
                "payload": {
                    "id": "8959fb57-d0f5-5725-9ac4-ec1c554fb0a0",
                    "name": "qa",
                    "input": {
                        "query": "query: what is weather in sf",
                        "answer": None,
                        "docs": ["doc1", "doc2", "doc3", "doc4"],
                    },
                    "triggers": ["retriever_one", "retriever_two"],
                },
            },
        ),
        ("updates", {"qa": {"answer": "doc1,doc2,doc3,doc4"}}),
        (
            "debug",
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 3,
                "payload": {
                    "id": "8959fb57-d0f5-5725-9ac4-ec1c554fb0a0",
                    "name": "qa",
                    "result": [("answer", "doc1,doc2,doc3,doc4")],
                },
            },
        ),
        (
            "values",
            {
                "query": "query: what is weather in sf",
                "answer": "doc1,doc2,doc3,doc4",
                "docs": ["doc1", "doc2", "doc3", "doc4"],
            },
        ),
    ]


def test_start_branch_then(snapshot: SnapshotAssertion) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two_slow", lambda s: {"my_key": " slow"})
    tool_two_graph.add_node("tool_two_fast", lambda s: {"my_key": " fast"})
    tool_two_graph.set_conditional_entry_point(
        lambda s: "tool_two_slow" if s["market"] == "DE" else "tool_two_fast", then=END
    )
    tool_two = tool_two_graph.compile()
    assert tool_two.get_graph().draw_mermaid() == snapshot

    assert tool_two.invoke({"my_key": "value", "market": "DE"}) == {
        "my_key": "value slow",
        "market": "DE",
    }
    assert tool_two.invoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value fast",
        "market": "US",
    }

    with SqliteSaver.from_conn_string(":memory:") as saver:
        tool_two = tool_two_graph.compile(
            checkpointer=saver, interrupt_before=["tool_two_fast", "tool_two_slow"]
        )

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            tool_two.invoke({"my_key": "value", "market": "DE"})

        thread1 = {"configurable": {"thread_id": "1"}}
        # stop when about to enter node
        assert tool_two.invoke({"my_key": "value ⛰️", "market": "DE"}, thread1) == {
            "my_key": "value ⛰️",
            "market": "DE",
        }
        assert [c.metadata for c in tool_two.checkpointer.list(thread1)] == [
            {
                "source": "loop",
                "step": 0,
                "writes": None,
            },
            {
                "source": "input",
                "step": -1,
                "writes": {"my_key": "value ⛰️", "market": "DE"},
            },
        ]
        assert tool_two.get_state(thread1) == StateSnapshot(
            values={"my_key": "value ⛰️", "market": "DE"},
            next=("tool_two_slow",),
            config=tool_two.checkpointer.get_tuple(thread1).config,
            created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
            metadata={"source": "loop", "step": 0, "writes": None},
            parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
        )
        # resume, for same result as above
        assert tool_two.invoke(None, thread1, debug=1) == {
            "my_key": "value ⛰️ slow",
            "market": "DE",
        }
        assert tool_two.get_state(thread1) == StateSnapshot(
            values={"my_key": "value ⛰️ slow", "market": "DE"},
            next=(),
            config=tool_two.checkpointer.get_tuple(thread1).config,
            created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 1,
                "writes": {"tool_two_slow": {"my_key": " slow"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
        )

        thread2 = {"configurable": {"thread_id": "2"}}
        # stop when about to enter node
        assert tool_two.invoke({"my_key": "value", "market": "US"}, thread2) == {
            "my_key": "value",
            "market": "US",
        }
        assert tool_two.get_state(thread2) == StateSnapshot(
            values={"my_key": "value", "market": "US"},
            next=("tool_two_fast",),
            config=tool_two.checkpointer.get_tuple(thread2).config,
            created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
            metadata={"source": "loop", "step": 0, "writes": None},
            parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
        )
        # resume, for same result as above
        assert tool_two.invoke(None, thread2, debug=1) == {
            "my_key": "value fast",
            "market": "US",
        }
        assert tool_two.get_state(thread2) == StateSnapshot(
            values={"my_key": "value fast", "market": "US"},
            next=(),
            config=tool_two.checkpointer.get_tuple(thread2).config,
            created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 1,
                "writes": {"tool_two_fast": {"my_key": " fast"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
        )

        thread3 = {"configurable": {"thread_id": "3"}}
        # stop when about to enter node
        assert tool_two.invoke({"my_key": "value", "market": "US"}, thread3) == {
            "my_key": "value",
            "market": "US",
        }
        assert tool_two.get_state(thread3) == StateSnapshot(
            values={"my_key": "value", "market": "US"},
            next=("tool_two_fast",),
            config=tool_two.checkpointer.get_tuple(thread3).config,
            created_at=tool_two.checkpointer.get_tuple(thread3).checkpoint["ts"],
            metadata={"source": "loop", "step": 0, "writes": None},
            parent_config=[*tool_two.checkpointer.list(thread3, limit=2)][-1].config,
        )
        # update state
        tool_two.update_state(thread3, {"my_key": "key"})  # appends to my_key
        assert tool_two.get_state(thread3) == StateSnapshot(
            values={"my_key": "valuekey", "market": "US"},
            next=("tool_two_fast",),
            config=tool_two.checkpointer.get_tuple(thread3).config,
            created_at=tool_two.checkpointer.get_tuple(thread3).checkpoint["ts"],
            metadata={
                "source": "update",
                "step": 1,
                "writes": {START: {"my_key": "key"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread3, limit=2)][-1].config,
        )
        # resume, for same result as above
        assert tool_two.invoke(None, thread3, debug=1) == {
            "my_key": "valuekey fast",
            "market": "US",
        }
        assert tool_two.get_state(thread3) == StateSnapshot(
            values={"my_key": "valuekey fast", "market": "US"},
            next=(),
            config=tool_two.checkpointer.get_tuple(thread3).config,
            created_at=tool_two.checkpointer.get_tuple(thread3).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 2,
                "writes": {"tool_two_fast": {"my_key": " fast"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread3, limit=2)][-1].config,
        )


def test_branch_then(snapshot: SnapshotAssertion) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_graph = StateGraph(State)
    tool_two_graph.set_entry_point("prepare")
    tool_two_graph.set_finish_point("finish")
    tool_two_graph.add_conditional_edges(
        source="prepare",
        path=lambda s: "tool_two_slow" if s["market"] == "DE" else "tool_two_fast",
        then="finish",
    )
    tool_two_graph.add_node("prepare", lambda s: {"my_key": " prepared"})
    tool_two_graph.add_node("tool_two_slow", lambda s: {"my_key": " slow"})
    tool_two_graph.add_node("tool_two_fast", lambda s: {"my_key": " fast"})
    tool_two_graph.add_node("finish", lambda s: {"my_key": " finished"})
    tool_two = tool_two_graph.compile()
    assert tool_two.get_graph().draw_mermaid(with_styles=False) == snapshot
    assert tool_two.get_graph().draw_mermaid() == snapshot

    assert tool_two.invoke({"my_key": "value", "market": "DE"}, debug=1) == {
        "my_key": "value prepared slow finished",
        "market": "DE",
    }
    assert tool_two.invoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value prepared fast finished",
        "market": "US",
    }

    with SqliteSaver.from_conn_string(":memory:") as saver:
        # test stream_mode=debug
        tool_two = tool_two_graph.compile(checkpointer=saver)
        thread10 = {"configurable": {"thread_id": "10"}}
        assert [
            *tool_two.stream(
                {"my_key": "value", "market": "DE"}, thread10, stream_mode="debug"
            )
        ] == [
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": -1,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {"my_key": ""},
                    "metadata": {
                        "source": "input",
                        "step": -1,
                        "writes": {"my_key": "value", "market": "DE"},
                    },
                },
            },
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": 0,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {
                        "my_key": "value",
                        "market": "DE",
                    },
                    "metadata": {
                        "source": "loop",
                        "step": 0,
                        "writes": None,
                    },
                },
            },
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "id": "7b7b0713-e958-5d07-803c-c9910a7cc162",
                    "name": "prepare",
                    "input": {"my_key": "value", "market": "DE"},
                    "triggers": ["start:prepare"],
                },
            },
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "id": "7b7b0713-e958-5d07-803c-c9910a7cc162",
                    "name": "prepare",
                    "result": [("my_key", " prepared")],
                },
            },
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": 1,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {
                        "my_key": "value prepared",
                        "market": "DE",
                    },
                    "metadata": {
                        "source": "loop",
                        "step": 1,
                        "writes": {"prepare": {"my_key": " prepared"}},
                    },
                },
            },
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": "dd9f2fa5-ccfa-5d12-81ec-942563056a08",
                    "name": "tool_two_slow",
                    "input": {"my_key": "value prepared", "market": "DE"},
                    "triggers": ["branch:prepare:condition:tool_two_slow"],
                },
            },
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "id": "dd9f2fa5-ccfa-5d12-81ec-942563056a08",
                    "name": "tool_two_slow",
                    "result": [("my_key", " slow")],
                },
            },
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": 2,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {
                        "my_key": "value prepared slow",
                        "market": "DE",
                    },
                    "metadata": {
                        "source": "loop",
                        "step": 2,
                        "writes": {"tool_two_slow": {"my_key": " slow"}},
                    },
                },
            },
            {
                "type": "task",
                "timestamp": AnyStr(),
                "step": 3,
                "payload": {
                    "id": "9b590c54-15ef-54b1-83a7-140d27b0bc52",
                    "name": "finish",
                    "input": {"my_key": "value prepared slow", "market": "DE"},
                    "triggers": ["branch:prepare:condition::then"],
                },
            },
            {
                "type": "task_result",
                "timestamp": AnyStr(),
                "step": 3,
                "payload": {
                    "id": "9b590c54-15ef-54b1-83a7-140d27b0bc52",
                    "name": "finish",
                    "result": [("my_key", " finished")],
                },
            },
            {
                "type": "checkpoint",
                "timestamp": AnyStr(),
                "step": 3,
                "payload": {
                    "config": {
                        "tags": [],
                        "metadata": {"thread_id": "10"},
                        "callbacks": None,
                        "recursion_limit": 25,
                        "configurable": {
                            "thread_id": "10",
                            "checkpoint_ns": "",
                            "checkpoint_id": AnyStr(),
                        },
                    },
                    "values": {
                        "my_key": "value prepared slow finished",
                        "market": "DE",
                    },
                    "metadata": {
                        "source": "loop",
                        "step": 3,
                        "writes": {"finish": {"my_key": " finished"}},
                    },
                },
            },
        ]

        tool_two = tool_two_graph.compile(
            checkpointer=saver, interrupt_before=["tool_two_fast", "tool_two_slow"]
        )

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            tool_two.invoke({"my_key": "value", "market": "DE"})

        thread1 = {"configurable": {"thread_id": "1"}}
        # stop when about to enter node
        assert tool_two.invoke({"my_key": "value", "market": "DE"}, thread1) == {
            "my_key": "value prepared",
            "market": "DE",
        }
        assert tool_two.get_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared", "market": "DE"},
            next=("tool_two_slow",),
            config=tool_two.checkpointer.get_tuple(thread1).config,
            created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 1,
                "writes": {"prepare": {"my_key": " prepared"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
        )
        # resume, for same result as above
        assert tool_two.invoke(None, thread1, debug=1) == {
            "my_key": "value prepared slow finished",
            "market": "DE",
        }
        assert tool_two.get_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared slow finished", "market": "DE"},
            next=(),
            config=tool_two.checkpointer.get_tuple(thread1).config,
            created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 3,
                "writes": {"finish": {"my_key": " finished"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
        )

        thread2 = {"configurable": {"thread_id": "2"}}
        # stop when about to enter node
        assert tool_two.invoke({"my_key": "value", "market": "US"}, thread2) == {
            "my_key": "value prepared",
            "market": "US",
        }
        assert tool_two.get_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared", "market": "US"},
            next=("tool_two_fast",),
            config=tool_two.checkpointer.get_tuple(thread2).config,
            created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 1,
                "writes": {"prepare": {"my_key": " prepared"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
        )
        # resume, for same result as above
        assert tool_two.invoke(None, thread2, debug=1) == {
            "my_key": "value prepared fast finished",
            "market": "US",
        }
        assert tool_two.get_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared fast finished", "market": "US"},
            next=(),
            config=tool_two.checkpointer.get_tuple(thread2).config,
            created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 3,
                "writes": {"finish": {"my_key": " finished"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
        )

    with SqliteSaver.from_conn_string(":memory:") as saver:
        tool_two = tool_two_graph.compile(
            checkpointer=saver, interrupt_before=["finish"]
        )

        thread1 = {"configurable": {"thread_id": "1"}}

        # stop when about to enter node
        assert tool_two.invoke({"my_key": "value", "market": "DE"}, thread1) == {
            "my_key": "value prepared slow",
            "market": "DE",
        }
        assert tool_two.get_state(thread1) == StateSnapshot(
            values={
                "my_key": "value prepared slow",
                "market": "DE",
            },
            next=("finish",),
            config=tool_two.checkpointer.get_tuple(thread1).config,
            created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 2,
                "writes": {"tool_two_slow": {"my_key": " slow"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
        )

        # update state
        tool_two.update_state(thread1, {"my_key": "er"})
        assert tool_two.get_state(thread1) == StateSnapshot(
            values={
                "my_key": "value prepared slower",
                "market": "DE",
            },
            next=("finish",),
            config=tool_two.checkpointer.get_tuple(thread1).config,
            created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
            metadata={
                "source": "update",
                "step": 3,
                "writes": {"tool_two_slow": {"my_key": "er"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
        )

    with SqliteSaver.from_conn_string(":memory:") as saver:
        tool_two = tool_two_graph.compile(
            checkpointer=saver, interrupt_after=["prepare"]
        )

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            tool_two.invoke({"my_key": "value", "market": "DE"})

        thread1 = {"configurable": {"thread_id": "1"}}
        # stop when about to enter node
        assert tool_two.invoke({"my_key": "value", "market": "DE"}, thread1) == {
            "my_key": "value prepared",
            "market": "DE",
        }
        assert tool_two.get_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared", "market": "DE"},
            next=("tool_two_slow",),
            config=tool_two.checkpointer.get_tuple(thread1).config,
            created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 1,
                "writes": {"prepare": {"my_key": " prepared"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
        )
        # resume, for same result as above
        assert tool_two.invoke(None, thread1, debug=1) == {
            "my_key": "value prepared slow finished",
            "market": "DE",
        }
        assert tool_two.get_state(thread1) == StateSnapshot(
            values={"my_key": "value prepared slow finished", "market": "DE"},
            next=(),
            config=tool_two.checkpointer.get_tuple(thread1).config,
            created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 3,
                "writes": {"finish": {"my_key": " finished"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
        )

        thread2 = {"configurable": {"thread_id": "2"}}
        # stop when about to enter node
        assert tool_two.invoke({"my_key": "value", "market": "US"}, thread2) == {
            "my_key": "value prepared",
            "market": "US",
        }
        assert tool_two.get_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared", "market": "US"},
            next=("tool_two_fast",),
            config=tool_two.checkpointer.get_tuple(thread2).config,
            created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 1,
                "writes": {"prepare": {"my_key": " prepared"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
        )
        # resume, for same result as above
        assert tool_two.invoke(None, thread2, debug=1) == {
            "my_key": "value prepared fast finished",
            "market": "US",
        }
        assert tool_two.get_state(thread2) == StateSnapshot(
            values={"my_key": "value prepared fast finished", "market": "US"},
            next=(),
            config=tool_two.checkpointer.get_tuple(thread2).config,
            created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 3,
                "writes": {"finish": {"my_key": " finished"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
        )

        thread3 = {"configurable": {"thread_id": "3"}}
        # update an empty thread before first run
        uconfig = tool_two.update_state(thread3, {"my_key": "key", "market": "DE"})
        # check current state
        assert tool_two.get_state(thread3) == StateSnapshot(
            values={"my_key": "key", "market": "DE"},
            next=("prepare",),
            config=uconfig,
            created_at=AnyStr(),
            metadata={
                "source": "update",
                "step": 0,
                "writes": {START: {"my_key": "key", "market": "DE"}},
            },
        )
        # run from this point
        assert tool_two.invoke(None, thread3) == {
            "my_key": "key prepared",
            "market": "DE",
        }
        # get state after first node
        assert tool_two.get_state(thread3) == StateSnapshot(
            values={"my_key": "key prepared", "market": "DE"},
            next=("tool_two_slow",),
            config=tool_two.checkpointer.get_tuple(thread3).config,
            created_at=tool_two.checkpointer.get_tuple(thread3).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 1,
                "writes": {"prepare": {"my_key": " prepared"}},
            },
            parent_config=uconfig,
        )
        # resume, for same result as above
        assert tool_two.invoke(None, thread3, debug=1) == {
            "my_key": "key prepared slow finished",
            "market": "DE",
        }
        assert tool_two.get_state(thread3) == StateSnapshot(
            values={"my_key": "key prepared slow finished", "market": "DE"},
            next=(),
            config=tool_two.checkpointer.get_tuple(thread3).config,
            created_at=tool_two.checkpointer.get_tuple(thread3).checkpoint["ts"],
            metadata={
                "source": "loop",
                "step": 3,
                "writes": {"finish": {"my_key": " finished"}},
            },
            parent_config=[*tool_two.checkpointer.list(thread3, limit=2)][-1].config,
        )


def test_in_one_fan_out_state_graph_waiting_edge(snapshot: SnapshotAssertion) -> None:
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
        return {"query": f'query: {data["query"]}'}

    def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

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
        checkpointer=MemorySaverAssertImmutable(),
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
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_before=["qa"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
    ]

    app_w_interrupt.update_state(config, {"docs": ["doc5"]})
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "query": "analyzed: query: what is weather in sf",
            "docs": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        },
        next=("qa",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "source": "update",
            "step": 4,
            "writes": {"retriever_one": {"docs": ["doc5"]}},
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config, debug=1)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4,doc5"}},
    ]


def test_in_one_fan_out_state_graph_waiting_edge_via_branch(
    snapshot: SnapshotAssertion,
) -> None:
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
        return {"query": f'query: {data["query"]}'}

    def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

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
        checkpointer=MemorySaverAssertImmutable(),
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
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic1(
    snapshot: SnapshotAssertion, mocker: MockerFixture
) -> None:
    from langchain_core.pydantic_v1 import BaseModel, ValidationError

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
        inner: InnerObject
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
        print("decider", data)
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
    assert app.get_input_schema().schema() == snapshot
    assert app.get_output_schema().schema() == snapshot

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
        checkpointer=MemorySaverAssertImmutable(),
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


def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic2(
    snapshot: SnapshotAssertion, mocker: MockerFixture
) -> None:
    from pydantic import BaseModel, ConfigDict, ValidationError

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
        inner: InnerObject
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]
        client: Annotated[httpx.Client, Context(make_httpx_client)]

    class StateUpdate(BaseModel):
        query: Optional[str] = None
        answer: Optional[str] = None
        docs: Optional[list[str]] = None

    class Input(BaseModel):
        query: str
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

    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
    assert app.get_input_schema().schema() == snapshot
    assert app.get_output_schema().schema() == snapshot

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
        checkpointer=MemorySaverAssertImmutable(),
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


def test_in_one_fan_out_state_graph_waiting_edge_plus_regular() -> None:
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
        return {"query": f'query: {data["query"]}'}

    def analyzer_one(data: State) -> State:
        time.sleep(0.1)
        return {"query": f'analyzed: {data["query"]}'}

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
        checkpointer=MemorySaverAssertImmutable(),
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
        return {"query": f'query: {data["query"]}'}

    def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

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
        return {"query": f'query: {data["query"]}'}

    def analyze(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

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
        return {"query": f'query: {data["query"]}'}

    def analyze(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

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
        return {"query": f'query: {data["query"]}'}

    def retriever_picker(data: State) -> list[str]:
        return ["analyzer_one", "retriever_two"]

    def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

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


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "checkpointer_name",
    ["memory", "sqlite", "postgres", "postgres_pipe"],
)
def test_nested_graph_interrupts(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class InnerState(TypedDict):
        my_key: str
        my_other_key: str

    def inner_1(state: InnerState):
        return {
            "my_key": state["my_key"] + " here",
            "my_other_key": state["my_key"],
        }

    def inner_2(state: InnerState):
        return {
            "my_key": state["my_key"] + " and there",
            "my_other_key": state["my_key"],
        }

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: str

    def outer_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    def outer_2(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("outer_1", outer_1)
    graph.add_node(
        "inner",
        inner.compile(interrupt_before=["inner_2"], checkpointer=INHERIT_CHECKPOINTER),
    )
    graph.add_node("outer_2", outer_2)
    graph.set_entry_point("outer_1")
    graph.add_edge("outer_1", "inner")
    graph.add_edge("inner", "outer_2")
    graph.set_finish_point("outer_2")

    app = graph.compile(checkpointer=checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert app.invoke({"my_key": "my value"}, config, debug=True) == {
        "my_key": "hi my value",
    }
    assert list(app.get_state_history(config)) == [
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    assert app.invoke(None, config, debug=True) == {
        "my_key": "hi my value here and there and back again",
    }
    assert list(app.get_state_history(config)) == [
        StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
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
                    "outer_2": {"my_key": "hi my value here and there and back again"}
                },
                "step": 3,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            next=("outer_2",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"inner": {"my_key": "hi my value here and there"}},
                "step": 2,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]

    # test stream updates w/ nested interrupt
    config = {"configurable": {"thread_id": "2"}}
    assert [*app.stream({"my_key": "my value"}, config)] == [
        {"outer_1": {"my_key": "hi my value"}},
    ]
    assert [*app.stream(None, config)] == [
        {"inner": {"my_key": "hi my value here and there"}},
        {"outer_2": {"my_key": "hi my value here and there and back again"}},
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [*app.stream({"my_key": "my value"}, config, stream_mode="values")] == [
        {
            "my_key": "my value",
        },
        {
            "my_key": "hi my value",
        },
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {
            "my_key": "hi my value here and there",
        },
        {
            "my_key": "hi my value here and there and back again",
        },
    ]

    # test interrupts BEFORE the node w/ interrupts
    app = graph.compile(checkpointer=checkpointer, interrupt_before=["inner"])
    config = {"configurable": {"thread_id": "4"}}
    assert [*app.stream({"my_key": "my value"}, config, stream_mode="values")] == [
        {
            "my_key": "my value",
        },
        {
            "my_key": "hi my value",
        },
    ]
    assert list(app.get_state_history(config)) == [
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    # while we're waiting for the node w/ interrupt inside to finish
    assert [*app.stream(None, config, stream_mode="values")] == []
    assert list(app.get_state_history(config)) == [
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {
            "my_key": "hi my value here and there",
        },
        {
            "my_key": "hi my value here and there and back again",
        },
    ]
    assert list(app.get_state_history(config)) == [
        StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
            next=(),
            config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {
                    "outer_2": {"my_key": "hi my value here and there and back again"}
                },
                "step": 3,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            next=("outer_2",),
            config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"inner": {"my_key": "hi my value here and there"}},
                "step": 2,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "4",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]

    # test interrupts AFTER the node w/ interrupts
    app = graph.compile(checkpointer=checkpointer, interrupt_after=["inner"])
    config = {"configurable": {"thread_id": "5"}}
    assert [*app.stream({"my_key": "my value"}, config, stream_mode="values")] == [
        {
            "my_key": "my value",
        },
        {
            "my_key": "hi my value",
        },
    ]
    # interrupted after "inner"
    assert list(app.get_state_history(config)) == [
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {
            "my_key": "hi my value here and there",
        },
    ]
    assert list(app.get_state_history(config)) == [
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            next=("outer_2",),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"inner": {"my_key": "hi my value here and there"}},
                "step": 2,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {
            "my_key": "hi my value here and there and back again",
        },
    ]
    assert list(app.get_state_history(config)) == [
        StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
            next=(),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {
                    "outer_2": {"my_key": "hi my value here and there and back again"}
                },
                "step": 3,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            next=("outer_2",),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"inner": {"my_key": "hi my value here and there"}},
                "step": 2,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "5",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]

    # test restarting from checkpoint_id
    config = {"configurable": {"thread_id": "6"}}
    app = graph.compile(checkpointer=checkpointer)
    assert app.invoke({"my_key": "my value"}, config, debug=True) == {
        "my_key": "hi my value"
    }
    state_history = [c for c in app.get_state_history(config)]
    assert state_history == [
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    child_state_history = [
        c
        for c in app.get_state_history(
            {"configurable": {"thread_id": "6", "checkpoint_ns": "inner"}}
        )
    ]
    assert child_state_history == [
        StateSnapshot(
            values={"my_key": "hi my value here"},
            next=(),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "inner",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {
                    "inner_1": {
                        "my_key": "hi my value here",
                        "my_other_key": "hi my value",
                    }
                },
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "inner",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        # there should be a single child checkpoint because we only keep
        # one child checkpoint per parent checkpoint (in which child ran)
    ]

    # check that child snapshot matches id of parent
    child_snapshot = child_state_history[0]
    assert (
        child_snapshot.config["configurable"]["checkpoint_id"]
        == state_history[0].config["configurable"]["checkpoint_id"]
    )
    # check resuming from interrupt w/ checkpoint_id
    interrupt_state_snapshot, before_interrupt_state_snapshot = state_history[:2]
    before_interrupt_config = before_interrupt_state_snapshot.config
    # going to get to interrupt again here, so the output is None
    assert app.invoke(None, before_interrupt_config, debug=True) == {
        "my_key": "hi my value"
    }
    # one more "identical" snapshot than before, at top of list
    assert list(app.get_state_history(config)) == [
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    # going to restart from interrupt
    interrupt_config = interrupt_state_snapshot.config
    assert app.invoke(None, interrupt_config, debug=True) == {
        "my_key": "hi my value here and there and back again",
    }
    assert list(app.get_state_history(config)) == [
        StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
            next=(),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {
                    "outer_2": {"my_key": "hi my value here and there and back again"}
                },
                "step": 3,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            next=("outer_2",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"inner": {"my_key": "hi my value here and there"}},
                "step": 2,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "6",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]


@pytest.mark.parametrize(
    "checkpointer_name",
    ["memory", "sqlite", "postgres", "postgres_pipe"],
)
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
    graph.add_node(
        "inner",
        inner.compile(interrupt_before=["inner_2"], checkpointer=INHERIT_CHECKPOINTER),
    )
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
        "my_key": "",
    }

    assert app.invoke(None, config, debug=True) == {
        "my_key": "got here and there and parallel and back again",
    }

    # below combo of assertions is asserting two things
    # - outer_1 finishes before inner interrupts (because we see its output in stream, which only happens after node finishes)
    # - the writes of outer are persisted in 1st call and used in 2nd call, ie outer isn't called again (because we dont see outer_1 output again in 2nd stream)
    # test stream updates w/ nested interrupt
    config = {"configurable": {"thread_id": "2"}}
    assert [*app.stream({"my_key": ""}, config)] == [
        # we got to parallel node first
        {"outer_1": {"my_key": " and parallel"}},
    ]
    assert [*app.stream(None, config)] == [
        {"inner": {"my_key": "got here and there"}},
        {"outer_2": {"my_key": " and back again"}},
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [*app.stream({"my_key": ""}, config, stream_mode="values")] == [
        {
            "my_key": "",
        },
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {
            "my_key": "got here and there and parallel",
        },
        {
            "my_key": "got here and there and parallel and back again",
        },
    ]

    # test interrupts BEFORE the parallel node
    app = graph.compile(checkpointer=checkpointer, interrupt_before=["outer_1"])
    config = {"configurable": {"thread_id": "4"}}
    assert [*app.stream({"my_key": ""}, config, stream_mode="values")] == [
        {"my_key": ""}
    ]
    # while we're waiting for the node w/ interrupt inside to finish
    assert [*app.stream(None, config, stream_mode="values")] == []
    assert [*app.stream(None, config, stream_mode="values")] == [
        {
            "my_key": "got here and there and parallel",
        },
        {
            "my_key": "got here and there and parallel and back again",
        },
    ]

    # test interrupts AFTER the parallel node
    app = graph.compile(checkpointer=checkpointer, interrupt_after=["outer_1"])
    config = {"configurable": {"thread_id": "5"}}
    assert [*app.stream({"my_key": ""}, config, stream_mode="values")] == [
        {"my_key": ""}
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": "got here and there and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {
            "my_key": "got here and there and parallel and back again",
        },
    ]


@pytest.mark.parametrize(
    "checkpointer_name",
    ["memory", "sqlite", "postgres", "postgres_pipe"],
)
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
        grandchild.compile(
            interrupt_before=["grandchild_2"], checkpointer=INHERIT_CHECKPOINTER
        ),
    )
    child.set_entry_point("child_1")
    child.set_finish_point("child_1")

    def parent_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    def parent_2(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("parent_1", parent_1)
    graph.add_node("child", child.compile(checkpointer=INHERIT_CHECKPOINTER))
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
    config = {"configurable": {"thread_id": "2"}}
    assert [*app.stream({"my_key": "my value"}, config)] == [
        {"parent_1": {"my_key": "hi my value"}},
    ]
    assert [*app.stream(None, config)] == [
        {"child": {"my_key": "hi my value here and there"}},
        {"parent_2": {"my_key": "hi my value here and there and back again"}},
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [*app.stream({"my_key": "my value"}, config, stream_mode="values")] == [
        {
            "my_key": "my value",
        },
        {
            "my_key": "hi my value",
        },
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {
            "my_key": "hi my value here and there",
        },
        {
            "my_key": "hi my value here and there and back again",
        },
    ]


@pytest.mark.parametrize(
    "checkpointer_name",
    ["memory", "sqlite", "postgres", "postgres_pipe"],
)
def test_nested_graph_state(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class InnerState(TypedDict):
        my_key: str
        my_other_key: str

    def inner_1(state: InnerState):
        return {
            "my_key": state["my_key"] + " here",
            "my_other_key": state["my_key"],
        }

    def inner_2(state: InnerState):
        return {
            "my_key": state["my_key"] + " and there",
            "my_other_key": state["my_key"],
        }

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: str

    def outer_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    def outer_2(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("outer_1", outer_1)
    graph.add_node(
        "inner",
        inner.compile(interrupt_before=["inner_2"], checkpointer=INHERIT_CHECKPOINTER),
    )
    graph.add_node("outer_2", outer_2)
    graph.set_entry_point("outer_1")
    graph.add_edge("outer_1", "inner")
    graph.add_edge("inner", "outer_2")
    graph.set_finish_point("outer_2")

    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    app.invoke({"my_key": "my value"}, config, debug=True)
    # test state w/ nested subgraph state (right after interrupt)
    assert app.get_state(config, include_subgraph_state=False) == StateSnapshot(
        values={"my_key": "hi my value"},
        next=("inner",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "writes": {"outer_1": {"my_key": "hi my value"}},
            "step": 1,
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        subgraph_state_snapshots=None,
    )
    assert app.get_state(config, include_subgraph_state=True) == StateSnapshot(
        values={"my_key": "hi my value"},
        next=("inner",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "writes": {"outer_1": {"my_key": "hi my value"}},
            "step": 1,
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        subgraph_state_snapshots={
            "inner": StateSnapshot(
                values={"my_key": "hi my value here"},
                next=(),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "inner",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "source": "loop",
                    "writes": {
                        "inner_1": {
                            "my_key": "hi my value here",
                            "my_other_key": "hi my value",
                        }
                    },
                    "step": 1,
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "inner",
                        "checkpoint_id": AnyStr(),
                    }
                },
                subgraph_state_snapshots=None,
            )
        },
    )
    assert list(app.get_state_history(config, include_subgraph_state=True)) == [
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            subgraph_state_snapshots={
                "inner": StateSnapshot(
                    values={"my_key": "hi my value here"},
                    next=(),
                    config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": "inner",
                            "checkpoint_id": AnyStr(),
                        }
                    },
                    metadata={
                        "source": "loop",
                        "writes": {
                            "inner_1": {
                                "my_key": "hi my value here",
                                "my_other_key": "hi my value",
                            }
                        },
                        "step": 1,
                    },
                    created_at=AnyStr(),
                    parent_config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": "inner",
                            "checkpoint_id": AnyStr(),
                        }
                    },
                    subgraph_state_snapshots=None,
                )
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            subgraph_state_snapshots=None,
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
            subgraph_state_snapshots=None,
        ),
    ]
    app.invoke(None, config, debug=True)
    # test state w/ nested subgraph state (after resuming from interrupt)
    assert app.get_state(config, include_subgraph_state=True) == StateSnapshot(
        values={"my_key": "hi my value here and there and back again"},
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
                "outer_2": {"my_key": "hi my value here and there and back again"}
            },
            "step": 3,
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        subgraph_state_snapshots={
            "inner": StateSnapshot(
                values={"my_key": "hi my value here and there"},
                next=(),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "inner",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "source": "loop",
                    "writes": {
                        "inner_2": {
                            "my_key": "hi my value here and there",
                            "my_other_key": "hi my value here",
                        }
                    },
                    "step": 2,
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "inner",
                        "checkpoint_id": AnyStr(),
                    }
                },
                subgraph_state_snapshots=None,
            )
        },
    )
    assert list(app.get_state_history(config, include_subgraph_state=True)) == [
        StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
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
                    "outer_2": {"my_key": "hi my value here and there and back again"}
                },
                "step": 3,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            subgraph_state_snapshots=None,
        ),
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            next=("outer_2",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"inner": {"my_key": "hi my value here and there"}},
                "step": 2,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            subgraph_state_snapshots=None,
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            # TODO: this is likely very confusing for an end user, and we'll probably need to update this.
            # right now this is happening due to us overwriting the
            # subgraph snapshot after we finish the graph with while the checkpoint_id
            # is the same as when we interrupted
            subgraph_state_snapshots={
                "inner": StateSnapshot(
                    values={"my_key": "hi my value here and there"},
                    next=(),
                    config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": "inner",
                            "checkpoint_id": AnyStr(),
                        }
                    },
                    metadata={
                        "source": "loop",
                        "writes": {
                            "inner_2": {
                                "my_key": "hi my value here and there",
                                "my_other_key": "hi my value here",
                            }
                        },
                        "step": 2,
                    },
                    created_at=AnyStr(),
                    parent_config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": "inner",
                            "checkpoint_id": AnyStr(),
                        }
                    },
                    subgraph_state_snapshots=None,
                )
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"source": "loop", "writes": None, "step": 0},
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            subgraph_state_snapshots=None,
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
            subgraph_state_snapshots=None,
        ),
    ]


@pytest.mark.parametrize(
    "checkpointer_name",
    ["memory", "sqlite", "postgres", "postgres_pipe"],
)
def test_doubly_nested_graph_state(
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
        grandchild.compile(
            interrupt_before=["grandchild_2"], checkpointer=INHERIT_CHECKPOINTER
        ),
    )
    child.set_entry_point("child_1")
    child.set_finish_point("child_1")

    def parent_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    def parent_2(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("parent_1", parent_1)
    graph.add_node("child", child.compile(checkpointer=INHERIT_CHECKPOINTER))
    graph.add_node("parent_2", parent_2)
    graph.set_entry_point("parent_1")
    graph.add_edge("parent_1", "child")
    graph.add_edge("child", "parent_2")
    graph.set_finish_point("parent_2")

    app = graph.compile(checkpointer=checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({"my_key": "my value"}, config, debug=True)
    assert app.get_state(config) == StateSnapshot(
        values={"my_key": "hi my value"},
        next=("child",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "writes": {"parent_1": {"my_key": "hi my value"}},
            "step": 1,
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        subgraph_state_snapshots=None,
    )
    assert app.get_state(config, include_subgraph_state=True) == StateSnapshot(
        values={"my_key": "hi my value"},
        next=("child",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "writes": {"parent_1": {"my_key": "hi my value"}},
            "step": 1,
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        subgraph_state_snapshots={
            "child": StateSnapshot(
                values={"my_key": "hi my value"},
                next=(),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "child",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={"source": "loop", "writes": None, "step": 0},
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "child",
                        "checkpoint_id": AnyStr(),
                    }
                },
                subgraph_state_snapshots={
                    "child_1": StateSnapshot(
                        values={"my_key": "hi my value here"},
                        next=(),
                        config={
                            "configurable": {
                                "thread_id": "1",
                                "checkpoint_ns": "child|child_1",
                                "checkpoint_id": AnyStr(),
                            }
                        },
                        metadata={
                            "source": "loop",
                            "writes": {"grandchild_1": {"my_key": "hi my value here"}},
                            "step": 1,
                        },
                        created_at=AnyStr(),
                        parent_config={
                            "configurable": {
                                "thread_id": "1",
                                "checkpoint_ns": "child|child_1",
                                "checkpoint_id": AnyStr(),
                            }
                        },
                        subgraph_state_snapshots=None,
                    )
                },
            )
        },
    )
    app.invoke(None, config, debug=True)
    assert app.get_state(config, include_subgraph_state=True) == StateSnapshot(
        values={"my_key": "hi my value here and there and back again"},
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
                "parent_2": {"my_key": "hi my value here and there and back again"}
            },
            "step": 3,
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        subgraph_state_snapshots={
            "child": StateSnapshot(
                values={"my_key": "hi my value here and there"},
                next=(),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "child",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "source": "loop",
                    "writes": {"child_1": {"my_key": "hi my value here and there"}},
                    "step": 1,
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "child",
                        "checkpoint_id": AnyStr(),
                    }
                },
                subgraph_state_snapshots={
                    "child_1": StateSnapshot(
                        values={"my_key": "hi my value here and there"},
                        next=(),
                        config={
                            "configurable": {
                                "thread_id": "1",
                                "checkpoint_ns": "child|child_1",
                                "checkpoint_id": AnyStr(),
                            }
                        },
                        metadata={
                            "source": "loop",
                            "writes": {
                                "grandchild_2": {"my_key": "hi my value here and there"}
                            },
                            "step": 2,
                        },
                        created_at=AnyStr(),
                        parent_config={
                            "configurable": {
                                "thread_id": "1",
                                "checkpoint_ns": "child|child_1",
                                "checkpoint_id": AnyStr(),
                            }
                        },
                        subgraph_state_snapshots=None,
                    )
                },
            )
        },
    )


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
    from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
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
    checkpointer_1 = MemorySaverAssertCheckpointMetadata()
    app = workflow.compile(checkpointer=checkpointer_1)

    # graph w/ interrupt
    checkpointer_2 = MemorySaverAssertCheckpointMetadata()
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
            AIMessage(
                content="",
                id=AnyStr(),
                tool_calls=[
                    {
                        "name": "search_api",
                        "args": {"query": "query"},
                        "id": "tool_call123",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="result for query",
                name="search_api",
                id=AnyStr(),
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


@pytest.mark.parametrize(
    "checkpointer_name",
    ["memory", "sqlite", "postgres", "postgres_pipe"],
)
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
    from langchain_core.pydantic_v1 import BaseModel, Field

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
    memory = MemorySaver()

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


@pytest.mark.parametrize(
    "checkpointer_name",
    ["memory", "sqlite", "postgres", "postgres_pipe"],
)
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
