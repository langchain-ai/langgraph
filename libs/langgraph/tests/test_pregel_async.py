import asyncio
import json
import operator
import re
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)
from uuid import UUID

import httpx
import pytest
from langchain_core.messages import (
    ToolCall,
)
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
    RunnablePick,
)
from langchain_core.utils.aiter import aclosing
from pydantic import BaseModel
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion

from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.context import Context
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import ERROR, Interrupt, Send
from langgraph.errors import InvalidUpdateError, NodeInterrupt
from langgraph.graph import END, Graph, StateGraph
from langgraph.graph.graph import START
from langgraph.graph.message import MessageGraph, add_messages
from langgraph.managed.shared_value import SharedValue
from langgraph.prebuilt.chat_agent_executor import (
    create_tool_calling_executor,
)
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.pregel import (
    Channel,
    GraphRecursionError,
    Pregel,
    StateSnapshot,
)
from langgraph.pregel.retry import RetryPolicy
from langgraph.pregel.types import PregelTask
from langgraph.store.memory import MemoryStore
from tests.any_str import AnyDict, AnyStr, AnyVersion, UnsortedSequence
from tests.conftest import ALL_CHECKPOINTERS_ASYNC
from tests.fake_tracer import FakeTracer
from tests.memory_assert import (
    MemorySaverAssertCheckpointMetadata,
    MemorySaverAssertImmutable,
)
from tests.messages import _AnyIdAIMessage, _AnyIdHumanMessage


async def test_checkpoint_errors() -> None:
    class FaultyGetCheckpointer(MemorySaver):
        async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
            raise ValueError("Faulty get_tuple")

    class FaultyPutCheckpointer(MemorySaver):
        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            raise ValueError("Faulty put")

    class FaultyPutWritesCheckpointer(MemorySaver):
        async def aput_writes(
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
        await graph.ainvoke("", {"configurable": {"thread_id": "thread-1"}})
    with pytest.raises(ValueError, match="Faulty get_tuple"):
        async for _ in graph.astream("", {"configurable": {"thread_id": "thread-2"}}):
            pass
    with pytest.raises(ValueError, match="Faulty get_tuple"):
        async for _ in graph.astream_events(
            "", {"configurable": {"thread_id": "thread-3"}}, version="v2"
        ):
            pass

    graph = builder.compile(checkpointer=FaultyPutCheckpointer())
    with pytest.raises(ValueError, match="Faulty put"):
        await graph.ainvoke("", {"configurable": {"thread_id": "thread-1"}})
    with pytest.raises(ValueError, match="Faulty put"):
        async for _ in graph.astream("", {"configurable": {"thread_id": "thread-2"}}):
            pass
    with pytest.raises(ValueError, match="Faulty put"):
        async for _ in graph.astream_events(
            "", {"configurable": {"thread_id": "thread-3"}}, version="v2"
        ):
            pass

    graph = builder.compile(checkpointer=FaultyVersionCheckpointer())
    with pytest.raises(ValueError, match="Faulty get_next_version"):
        await graph.ainvoke("", {"configurable": {"thread_id": "thread-1"}})
    with pytest.raises(ValueError, match="Faulty get_next_version"):
        async for _ in graph.astream("", {"configurable": {"thread_id": "thread-2"}}):
            pass
    with pytest.raises(ValueError, match="Faulty get_next_version"):
        async for _ in graph.astream_events(
            "", {"configurable": {"thread_id": "thread-3"}}, version="v2"
        ):
            pass

    # add a parallel node
    builder.add_node("parallel", logic)
    builder.add_edge(START, "parallel")
    graph = builder.compile(checkpointer=FaultyPutWritesCheckpointer())
    with pytest.raises(ValueError, match="Faulty put_writes"):
        await graph.ainvoke("", {"configurable": {"thread_id": "thread-1"}})
    with pytest.raises(ValueError, match="Faulty put_writes"):
        async for _ in graph.astream("", {"configurable": {"thread_id": "thread-2"}}):
            pass
    with pytest.raises(ValueError, match="Faulty put_writes"):
        async for _ in graph.astream_events(
            "", {"configurable": {"thread_id": "thread-3"}}, version="v2"
        ):
            pass


async def test_node_cancellation_on_external_cancel() -> None:
    inner_task_cancelled = False

    async def awhile(input: Any) -> None:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            nonlocal inner_task_cancelled
            inner_task_cancelled = True
            raise

    builder = Graph()
    builder.add_node("agent", awhile)
    builder.set_entry_point("agent")
    builder.set_finish_point("agent")

    graph = builder.compile()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(graph.ainvoke(1), 0.5)

    assert inner_task_cancelled


async def test_node_cancellation_on_other_node_exception() -> None:
    inner_task_cancelled = False

    async def awhile(input: Any) -> None:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            nonlocal inner_task_cancelled
            inner_task_cancelled = True
            raise

    async def iambad(input: Any) -> None:
        raise ValueError("I am bad")

    builder = Graph()
    builder.add_node("agent", awhile)
    builder.add_node("bad", iambad)
    builder.set_conditional_entry_point(lambda _: ["agent", "bad"], then=END)

    graph = builder.compile()

    with pytest.raises(ValueError, match="I am bad"):
        # This will raise ValueError, not TimeoutError
        await asyncio.wait_for(graph.ainvoke(1), 0.5)

    assert inner_task_cancelled


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_dynamic_interrupt(
    checkpointer_name: str, snapshot: SnapshotAssertion, request: pytest.FixtureRequest
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_node_count = 0

    async def tool_two_node(s: State) -> State:
        nonlocal tool_two_node_count
        tool_two_node_count += 1
        if s["market"] == "DE":
            raise NodeInterrupt("Just because...")
        return {"my_key": " all good"}

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two", tool_two_node, retry=RetryPolicy())
    tool_two_graph.add_edge(START, "tool_two")
    tool_two = tool_two_graph.compile()

    tracer = FakeTracer()
    assert await tool_two.ainvoke(
        {"my_key": "value", "market": "DE"}, {"callbacks": [tracer]}
    ) == {
        "my_key": "value",
        "market": "DE",
    }
    assert tool_two_node_count == 1, "interrupts aren't retried"
    assert len(tracer.runs) == 1
    run = tracer.runs[0]
    assert run.end_time is not None
    assert run.error is None
    assert run.outputs == {"market": "DE", "my_key": "value"}

    assert await tool_two.ainvoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value all good",
        "market": "US",
    }

    tool_two = tool_two_graph.compile(checkpointer=checkpointer)

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        await tool_two.ainvoke({"my_key": "value", "market": "DE"})

    thread1 = {"configurable": {"thread_id": "1"}}
    # stop when about to enter node
    assert await tool_two.ainvoke({"my_key": "value ⛰️", "market": "DE"}, thread1) == {
        "my_key": "value ⛰️",
        "market": "DE",
    }
    assert [c.metadata async for c in tool_two.checkpointer.alist(thread1)] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
        },
        {
            "parents": {},
            "source": "input",
            "step": -1,
            "writes": {"my_key": "value ⛰️", "market": "DE"},
        },
    ]
    tup = await tool_two.checkpointer.aget_tuple(thread1)
    assert await tool_two.aget_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️", "market": "DE"},
        next=("tool_two",),
        tasks=(
            PregelTask(
                AnyStr(),
                "tool_two",
                interrupts=(Interrupt("Just because..."),),
            ),
        ),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={"parents": {}, "source": "loop", "step": 0, "writes": None},
        parent_config=[c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
            -1
        ].config,
    )
    # TODO use aget_state_history


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_node_not_cancelled_on_other_node_interrupted(
    checkpointer_name: str, request: pytest.FixtureRequest
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        hello: str

    awhiles = 0
    inner_task_cancelled = False

    async def awhile(input: State) -> None:
        nonlocal awhiles

        awhiles += 1
        try:
            await asyncio.sleep(1)
            return {"hello": "again"}
        except asyncio.CancelledError:
            nonlocal inner_task_cancelled
            inner_task_cancelled = True
            raise

    async def iambad(input: State) -> None:
        if input["hello"] != "bye":
            raise NodeInterrupt("I am bad")

    builder = StateGraph(State)
    builder.add_node("agent", awhile)
    builder.add_node("bad", iambad)
    builder.set_conditional_entry_point(lambda _: ["agent", "bad"], then=END)

    graph = builder.compile(checkpointer=checkpointer)
    thread = {"configurable": {"thread_id": "1"}}

    assert await graph.ainvoke({"hello": "world"}, thread) == {"hello": "world"}

    assert not inner_task_cancelled
    assert awhiles == 1

    assert await graph.ainvoke(None, thread, debug=True) is None

    assert not inner_task_cancelled
    assert awhiles == 1

    assert await graph.ainvoke({"hello": "bye"}, thread) == {"hello": "again"}

    assert not inner_task_cancelled
    assert awhiles == 2


async def test_step_timeout_on_stream_hang() -> None:
    inner_task_cancelled = False

    async def awhile(input: Any) -> None:
        try:
            await asyncio.sleep(1.5)
        except asyncio.CancelledError:
            nonlocal inner_task_cancelled
            inner_task_cancelled = True
            raise

    async def alittlewhile(input: Any) -> None:
        await asyncio.sleep(0.6)
        return "1"

    builder = Graph()
    builder.add_node(awhile)
    builder.add_node(alittlewhile)
    builder.set_conditional_entry_point(lambda _: ["awhile", "alittlewhile"], then=END)
    graph = builder.compile()
    graph.step_timeout = 1

    with pytest.raises(asyncio.TimeoutError):
        async for chunk in graph.astream(1, stream_mode="updates"):
            assert chunk == {"alittlewhile": {"alittlewhile": "1"}}
            await asyncio.sleep(0.6)

    assert inner_task_cancelled


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_cancel_graph_astream(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        value: Annotated[int, operator.add]

    class AwhileMaker:
        def __init__(self) -> None:
            self.reset()

        async def __call__(self, input: State) -> Any:
            self.started = True
            try:
                await asyncio.sleep(1.5)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        def reset(self):
            self.started = False
            self.cancelled = False

    async def alittlewhile(input: State) -> None:
        await asyncio.sleep(0.6)
        return {"value": 2}

    awhile = AwhileMaker()
    aparallelwhile = AwhileMaker()
    builder = StateGraph(State)
    builder.add_node("awhile", awhile)
    builder.add_node("aparallelwhile", aparallelwhile)
    builder.add_node(alittlewhile)
    builder.add_edge(START, "alittlewhile")
    builder.add_edge(START, "aparallelwhile")
    builder.add_edge("alittlewhile", "awhile")
    graph = builder.compile(checkpointer=checkpointer)

    # test interrupting astream
    got_event = False
    thread1: RunnableConfig = {"configurable": {"thread_id": "1"}}
    async with aclosing(graph.astream({"value": 1}, thread1)) as stream:
        async for chunk in stream:
            assert chunk == {"alittlewhile": {"value": 2}}
            got_event = True
            break

    assert got_event

    # node aparallelwhile should start, but be cancelled
    assert aparallelwhile.started is True
    assert aparallelwhile.cancelled is True

    # node "awhile" should never start
    assert awhile.started is False

    # checkpoint with output of "alittlewhile" should not be saved
    if checkpointer is not None:
        state = await graph.aget_state(thread1)
        assert state is not None
        assert state.values == {"value": 1}
        assert state.next == (
            "aparallelwhile",
            "alittlewhile",
        )
        assert state.metadata == {
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
        }


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_cancel_graph_astream_events_v2(
    request: pytest.FixtureRequest, checkpointer_name: Optional[str]
) -> None:
    checkpointer = (
        request.getfixturevalue(f"checkpointer_{checkpointer_name}")
        if checkpointer_name
        else None
    )

    class State(TypedDict):
        value: int

    class AwhileMaker:
        def __init__(self) -> None:
            self.reset()

        async def __call__(self, input: State) -> Any:
            self.started = True
            try:
                await asyncio.sleep(1.5)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        def reset(self):
            self.started = False
            self.cancelled = False

    async def alittlewhile(input: State) -> None:
        await asyncio.sleep(0.6)
        return {"value": 2}

    awhile = AwhileMaker()
    anotherwhile = AwhileMaker()
    builder = StateGraph(State)
    builder.add_node(alittlewhile)
    builder.add_node("awhile", awhile)
    builder.add_node("anotherwhile", anotherwhile)
    builder.add_edge(START, "alittlewhile")
    builder.add_edge("alittlewhile", "awhile")
    builder.add_edge("awhile", "anotherwhile")
    graph = builder.compile(checkpointer=checkpointer)

    # test interrupting astream_events v2
    got_event = False
    thread2: RunnableConfig = {"configurable": {"thread_id": "2"}}
    async with aclosing(
        graph.astream_events({"value": 1}, thread2, version="v2")
    ) as stream:
        async for chunk in stream:
            if chunk["event"] == "on_chain_stream" and not chunk["parent_ids"]:
                got_event = True
                assert chunk["data"]["chunk"] == {"alittlewhile": {"value": 2}}
                break

    # did break
    assert got_event

    # node "awhile" maybe starts (impl detail of astream_events)
    # if it does start, it must be cancelled
    if awhile.started:
        assert awhile.cancelled is True

    # node "anotherwhile" should never start
    assert anotherwhile.started is False

    # checkpoint with output of "alittlewhile" should not be saved
    if checkpointer is not None:
        state = await graph.aget_state(thread2)
        assert state is not None
        assert state.values == {"value": 2}
        assert state.next == ("awhile",)
        assert state.metadata == {
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"alittlewhile": {"value": 2}},
        }


async def test_node_schemas_custom_output() -> None:
    class State(TypedDict):
        hello: str
        bye: str
        messages: Annotated[list[str], add_messages]

    class Output(TypedDict):
        messages: list[str]

    class StateForA(TypedDict):
        hello: str
        messages: Annotated[list[str], add_messages]

    async def node_a(state: StateForA):
        assert state == {
            "hello": "there",
            "messages": [_AnyIdHumanMessage(content="hello")],
        }

    class StateForB(TypedDict):
        bye: str
        now: int

    async def node_b(state: StateForB):
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

    async def node_c(state: StateForC):
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

    assert await graph.ainvoke(
        {"hello": "there", "bye": "world", "messages": "hello"}
    ) == {
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

    assert await graph.ainvoke(
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
        async for c in graph.astream(
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


async def test_invoke_single_process_in_out(mocker: MockerFixture) -> None:
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
    assert await app.ainvoke(2) == 3
    assert await app.ainvoke(2, output_keys=["output"]) == {"output": 3}

    assert await gapp.ainvoke(2) == 3


@pytest.mark.parametrize(
    "falsy_value",
    [None, False, 0, "", [], {}, set(), frozenset(), 0.0, 0j],
)
async def test_invoke_single_process_in_out_falsy_values(falsy_value: Any) -> None:
    graph = Graph()
    graph.add_node("return_falsy_const", lambda *args, **kwargs: falsy_value)
    graph.set_entry_point("return_falsy_const")
    graph.set_finish_point("return_falsy_const")
    gapp = graph.compile()
    assert falsy_value == await gapp.ainvoke(1)


async def test_invoke_single_process_in_write_kwargs(mocker: MockerFixture) -> None:
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
    assert await app.ainvoke(2) == {"output": 3, "fixed": 5, "output_plus_one": 4}


async def test_invoke_single_process_in_out_dict(mocker: MockerFixture) -> None:
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
    assert await app.ainvoke(2) == {"output": 3}


async def test_invoke_single_process_in_dict_out_dict(mocker: MockerFixture) -> None:
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
    assert await app.ainvoke({"input": 2}) == {"output": 3}


async def test_invoke_two_processes_in_out(mocker: MockerFixture) -> None:
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
        stream_channels=["inbox", "output"],
    )

    assert await app.ainvoke(2) == 4

    with pytest.raises(GraphRecursionError):
        await app.ainvoke(2, {"recursion_limit": 1})

    step = 0
    async for values in app.astream(2):
        step += 1
        if step == 1:
            assert values == {
                "inbox": 3,
            }
        elif step == 2:
            assert values == {
                "inbox": 3,
                "output": 4,
            }
    assert step == 2

    graph = Graph()
    graph.add_node("add_one", add_one)
    graph.add_node("add_one_more", add_one)
    graph.set_entry_point("add_one")
    graph.set_finish_point("add_one_more")
    graph.add_edge("add_one", "add_one_more")
    gapp = graph.compile()

    assert await gapp.ainvoke(2) == 4

    step = 0
    async for values in gapp.astream(2):
        step += 1
        if step == 1:
            assert values == {
                "add_one": 3,
            }
        elif step == 2:
            assert values == {
                "add_one_more": 4,
            }
    assert step == 2


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_invoke_two_processes_in_out_interrupt(
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
    assert await app.ainvoke(2, thread1) is None

    # inbox == 3
    checkpoint = await checkpointer.aget(thread1)
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 3

    # resume execution, finish
    assert await app.ainvoke(None, thread1) == 4

    # start execution again, stop at inbox
    assert await app.ainvoke(20, thread1) is None

    # inbox == 21
    checkpoint = await checkpointer.aget(thread1)
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 21

    # send a new value in, interrupting the previous execution
    assert await app.ainvoke(3, thread1) is None
    assert await app.ainvoke(None, thread1) == 5

    # start execution again, stopping at inbox
    assert await app.ainvoke(20, thread2) is None

    # inbox == 21
    snapshot = await app.aget_state(thread2)
    assert snapshot.values["inbox"] == 21
    assert snapshot.next == ("two",)

    # update the state, resume
    await app.aupdate_state(thread2, 25, as_node="one")
    assert await app.ainvoke(None, thread2) == 26

    # no pending tasks
    snapshot = await app.aget_state(thread2)
    assert snapshot.next == ()

    # list history
    history = [c async for c in app.aget_state_history(thread1)]
    assert history == [
        StateSnapshot(
            values={"inbox": 4, "output": 5, "input": 3},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"parents": {}, "source": "loop", "step": 6, "writes": {"two": 5}},
            created_at=AnyStr(),
            parent_config=history[1].config,
        ),
        StateSnapshot(
            values={"inbox": 4, "output": 4, "input": 3},
            tasks=(PregelTask(AnyStr(), "two"),),
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 5,
                "writes": {"one": None},
            },
            created_at=AnyStr(),
            parent_config=history[2].config,
        ),
        StateSnapshot(
            values={"inbox": 21, "output": 4, "input": 3},
            tasks=(PregelTask(AnyStr(), "one"),),
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"parents": {}, "source": "input", "step": 4, "writes": 3},
            created_at=AnyStr(),
            parent_config=history[3].config,
        ),
        StateSnapshot(
            values={"inbox": 21, "output": 4, "input": 20},
            tasks=(PregelTask(AnyStr(), "two"),),
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
                "writes": {"one": None},
            },
            created_at=AnyStr(),
            parent_config=history[4].config,
        ),
        StateSnapshot(
            values={"inbox": 3, "output": 4, "input": 20},
            tasks=(PregelTask(AnyStr(), "one"),),
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"parents": {}, "source": "input", "step": 2, "writes": 20},
            created_at=AnyStr(),
            parent_config=history[5].config,
        ),
        StateSnapshot(
            values={"inbox": 3, "output": 4, "input": 2},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"parents": {}, "source": "loop", "step": 1, "writes": {"two": 4}},
            created_at=AnyStr(),
            parent_config=history[6].config,
        ),
        StateSnapshot(
            values={"inbox": 3, "input": 2},
            tasks=(PregelTask(AnyStr(), "two"),),
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": {"one": None},
            },
            created_at=AnyStr(),
            parent_config=history[7].config,
        ),
        StateSnapshot(
            values={"input": 2},
            tasks=(PregelTask(AnyStr(), "one"),),
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"parents": {}, "source": "input", "step": -1, "writes": 2},
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]

    # forking from any previous checkpoint w/out forking should do nothing
    assert [
        c async for c in app.astream(None, history[0].config, stream_mode="updates")
    ] == []
    assert [
        c async for c in app.astream(None, history[1].config, stream_mode="updates")
    ] == [
        {"two": {"output": 5}, "__metadata__": {"cached": True}},
    ]
    assert [
        c async for c in app.astream(None, history[2].config, stream_mode="updates")
    ] == [
        {"one": {"inbox": 4}, "__metadata__": {"cached": True}},
    ]

    # forking and re-running from any prev checkpoint should re-run nodes
    fork_config = await app.aupdate_state(history[0].config, None)
    assert [
        c async for c in app.astream(None, fork_config, stream_mode="updates")
    ] == []

    fork_config = await app.aupdate_state(history[1].config, None)
    assert [c async for c in app.astream(None, fork_config, stream_mode="updates")] == [
        {"two": {"output": 5}}
    ]

    fork_config = await app.aupdate_state(history[2].config, None)
    assert [c async for c in app.astream(None, fork_config, stream_mode="updates")] == [
        {"one": {"inbox": 4}}
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_fork_always_re_runs_nodes(
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
    assert [
        c async for c in graph.astream(1, thread1, stream_mode=["values", "updates"])
    ] == [
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
    history = [c async for c in graph.aget_state_history(thread1)]
    assert history == [
        StateSnapshot(
            values=6,
            next=(),
            tasks=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 5,
                "writes": {"add_one": 1},
            },
            created_at=AnyStr(),
            parent_config=history[1].config,
        ),
        StateSnapshot(
            values=5,
            tasks=(PregelTask(AnyStr(), "add_one"),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 4,
                "writes": {"add_one": 1},
            },
            created_at=AnyStr(),
            parent_config=history[2].config,
        ),
        StateSnapshot(
            values=4,
            tasks=(PregelTask(AnyStr(), "add_one"),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
                "writes": {"add_one": 1},
            },
            created_at=AnyStr(),
            parent_config=history[3].config,
        ),
        StateSnapshot(
            values=3,
            tasks=(PregelTask(AnyStr(), "add_one"),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 2,
                "writes": {"add_one": 1},
            },
            created_at=AnyStr(),
            parent_config=history[4].config,
        ),
        StateSnapshot(
            values=2,
            tasks=(PregelTask(AnyStr(), "add_one"),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
                "writes": {"add_one": 1},
            },
            created_at=AnyStr(),
            parent_config=history[5].config,
        ),
        StateSnapshot(
            values=1,
            tasks=(PregelTask(AnyStr(), "add_one"),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"parents": {}, "source": "loop", "step": 0, "writes": None},
            created_at=AnyStr(),
            parent_config=history[6].config,
        ),
        StateSnapshot(
            values=0,
            tasks=(PregelTask(AnyStr(), "__start__"),),
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"parents": {}, "source": "input", "step": -1, "writes": 1},
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]

    # forking from any previous checkpoint w/out forking should do nothing
    assert [
        c async for c in graph.astream(None, history[0].config, stream_mode="updates")
    ] == []
    assert [
        c async for c in graph.astream(None, history[1].config, stream_mode="updates")
    ] == [{"add_one": 1, "__metadata__": {"cached": True}}]

    # forking and re-running from any prev checkpoint should re-run nodes
    fork_config = await graph.aupdate_state(history[0].config, None)
    assert [
        c async for c in graph.astream(None, fork_config, stream_mode="updates")
    ] == []

    fork_config = await graph.aupdate_state(history[1].config, None)
    assert [
        c async for c in graph.astream(None, fork_config, stream_mode="updates")
    ] == [{"add_one": 1}]

    fork_config = await graph.aupdate_state(history[2].config, None)
    assert [
        c async for c in graph.astream(None, fork_config, stream_mode="updates")
    ] == [
        {"add_one": 1},
        {"add_one": 1},
    ]


async def test_invoke_two_processes_in_dict_out(mocker: MockerFixture) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = (
        Channel.subscribe_to("inbox")
        | RunnableLambda(add_one).abatch
        | Channel.write_to("output").abatch
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
        c
        async for c in app.astream(
            {"input": 2, "inbox": 12}, output_keys="output", stream_mode="updates"
        )
    ] == [
        {"one": None},
        {"two": 13},
        {"two": 4},
    ]
    assert [
        c async for c in app.astream({"input": 2, "inbox": 12}, output_keys="output")
    ] == [13, 4]

    assert [
        c async for c in app.astream({"input": 2, "inbox": 12}, stream_mode="updates")
    ] == [
        {"one": {"inbox": 3}},
        {"two": {"output": 13}},
        {"two": {"output": 4}},
    ]
    assert [c async for c in app.astream({"input": 2, "inbox": 12})] == [
        {"inbox": [3], "output": 13},
        {"output": 4},
    ]
    assert [
        c async for c in app.astream({"input": 2, "inbox": 12}, stream_mode="debug")
    ] == [
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
                "error": None,
                "interrupts": [],
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
                "error": None,
                "interrupts": [],
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
                "error": None,
                "interrupts": [],
            },
        },
    ]


async def test_batch_two_processes_in_out() -> None:
    async def add_one_with_delay(inp: int) -> int:
        await asyncio.sleep(inp / 10)
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

    assert await app.abatch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]
    assert await app.abatch([3, 2, 1, 3, 5], output_keys=["output"]) == [
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

    assert await gapp.abatch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]


async def test_invoke_many_processes_in_out(mocker: MockerFixture) -> None:
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

    # No state is left over from previous invocations
    for _ in range(10):
        assert await app.ainvoke(2, {"recursion_limit": test_size}) == 2 + test_size

    # Concurrent invocations do not interfere with each other
    assert await asyncio.gather(
        *(app.ainvoke(2, {"recursion_limit": test_size}) for _ in range(10))
    ) == [2 + test_size for _ in range(10)]


async def test_batch_many_processes_in_out(mocker: MockerFixture) -> None:
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

    # No state is left over from previous invocations
    for _ in range(3):
        # Then invoke pubsub
        assert await app.abatch([2, 1, 3, 4, 5], {"recursion_limit": test_size}) == [
            2 + test_size,
            1 + test_size,
            3 + test_size,
            4 + test_size,
            5 + test_size,
        ]

    # Concurrent invocations do not interfere with each other
    assert await asyncio.gather(
        *(app.abatch([2, 1, 3, 4, 5], {"recursion_limit": test_size}) for _ in range(3))
    ) == [
        [2 + test_size, 1 + test_size, 3 + test_size, 4 + test_size, 5 + test_size]
        for _ in range(3)
    ]


async def test_invoke_two_processes_two_in_two_out_invalid(
    mocker: MockerFixture,
) -> None:
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
        await app.ainvoke(2)


async def test_invoke_two_processes_two_in_two_out_valid(mocker: MockerFixture) -> None:
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

    # An Topic channel accumulates updates into a sequence
    assert await app.ainvoke(2) == [3, 3]


async def test_invoke_checkpoint(mocker: MockerFixture) -> None:
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
    assert await app.ainvoke(2, {"configurable": {"thread_id": "1"}}) == 2
    checkpoint = await memory.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 2
    # total is now 2, so output is 2+3=5
    assert await app.ainvoke(3, {"configurable": {"thread_id": "1"}}) == 5
    assert errored_once, "errored and retried"
    checkpoint = await memory.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        await app.ainvoke(4, {"configurable": {"thread_id": "1"}})
    # checkpoint is not updated
    checkpoint = await memory.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert await app.ainvoke(5, {"configurable": {"thread_id": "2"}}) == 5
    checkpoint = await memory.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    checkpoint = await memory.aget({"configurable": {"thread_id": "2"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 5


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_pending_writes_resume(
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

        async def __call__(self, input: State) -> Any:
            self.calls += 1
            await asyncio.sleep(self.sleep)
            if isinstance(self.rtn, Exception):
                raise self.rtn
            else:
                return self.rtn

        def reset(self):
            self.calls = 0

    one = AwhileMaker(0.2, {"value": 2})
    two = AwhileMaker(0.6, ValueError("I'm not good"))
    builder = StateGraph(State)
    builder.add_node("one", one)
    builder.add_node("two", two)
    builder.add_edge(START, "one")
    builder.add_edge(START, "two")
    graph = builder.compile(checkpointer=checkpointer)

    thread1: RunnableConfig = {"configurable": {"thread_id": "1"}}
    with pytest.raises(ValueError, match="I'm not good"):
        await graph.ainvoke({"value": 1}, thread1)

    # both nodes should have been called once
    assert one.calls == 1
    assert two.calls == 1

    # latest checkpoint should be before nodes "one", "two"
    state = await graph.aget_state(thread1)
    assert state is not None
    assert state.values == {"value": 1}
    assert state.next == ("one", "two")
    assert state.tasks == (
        PregelTask(AnyStr(), "one"),
        PregelTask(AnyStr(), "two", 'ValueError("I\'m not good")'),
    )
    assert state.metadata == {
        "parents": {},
        "source": "loop",
        "step": 0,
        "writes": None,
    }
    # should contain pending write of "one"
    checkpoint = await checkpointer.aget_tuple(thread1)
    assert checkpoint is not None
    # should contain error from "two"
    expected_writes = [
        (AnyStr(), "one", "one"),
        (AnyStr(), "value", 2),
        (AnyStr(), ERROR, 'ValueError("I\'m not good")'),
    ]
    assert len(checkpoint.pending_writes) == 3
    assert all(w in expected_writes for w in checkpoint.pending_writes)
    # both non-error pending writes come from same task
    non_error_writes = [w for w in checkpoint.pending_writes if w[1] != ERROR]
    assert non_error_writes[0][0] == non_error_writes[1][0]
    # error write is from the other task
    error_write = next(w for w in checkpoint.pending_writes if w[1] == ERROR)
    assert error_write[0] != non_error_writes[0][0]

    # TODO arguably this shouldn't even run the failed task again,
    # and should require empty update_state (ie new checkpoint_id)
    # in order to try again

    # resume execution
    with pytest.raises(ValueError, match="I'm not good"):
        await graph.ainvoke(None, thread1)

    # node "one" succeeded previously, so shouldn't be called again
    assert one.calls == 1
    # node "two" should have been called once again
    assert two.calls == 2

    # confirm no new checkpoints saved
    state_two = await graph.aget_state(thread1)
    assert state_two.metadata == state.metadata

    # resume execution, without exception
    two.rtn = {"value": 3}
    # both the pending write and the new write were applied, 1 + 2 + 3 = 6
    assert await graph.ainvoke(None, thread1) == {"value": 6}

    # check all final checkpoints
    checkpoints = [c async for c in checkpointer.alist(thread1)]
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
        metadata={"parents": {}, "step": 0, "source": "loop", "writes": None},
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
            (AnyStr(), "__error__", 'ValueError("I\'m not good")'),
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
        metadata={"parents": {}, "step": -1, "source": "input", "writes": {"value": 1}},
        parent_config=None,
        pending_writes=UnsortedSequence(
            (AnyStr(), "value", 1),
            (AnyStr(), "start:one", "__start__"),
            (AnyStr(), "start:two", "__start__"),
        ),
    )


async def test_cond_edge_after_send() -> None:
    class Node:
        def __init__(self, name: str):
            self.name = name
            setattr(self, "__name__", name)

        async def __call__(self, state):
            return [self.name]

    async def send_for_fun(state):
        return [Send("2", state), Send("2", state)]

    async def route_to_three(state) -> Literal["3"]:
        return "3"

    builder = StateGraph(Annotated[list, operator.add])
    builder.add_node(Node("1"))
    builder.add_node(Node("2"))
    builder.add_node(Node("3"))
    builder.add_edge(START, "1")
    builder.add_conditional_edges("1", send_for_fun)
    builder.add_conditional_edges("2", route_to_three)
    graph = builder.compile()

    assert await graph.ainvoke(["0"]) == ["0", "1", "2", "2", "3"]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_invoke_checkpoint_three(
    mocker: MockerFixture, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)
    add_one = mocker.Mock(side_effect=lambda x: x["total"] + x["input"])

    def raise_if_above_10(input: int) -> int:
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
        debug=True,
    )

    thread_1 = {"configurable": {"thread_id": "1"}}
    # total starts out as 0, so output is 0+2=2
    assert await app.ainvoke(2, thread_1) == 2
    state = await app.aget_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 2
    assert (
        state.config["configurable"]["checkpoint_id"]
        == (await checkpointer.aget(thread_1))["id"]
    )
    # total is now 2, so output is 2+3=5
    assert await app.ainvoke(3, thread_1) == 5
    state = await app.aget_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert (
        state.config["configurable"]["checkpoint_id"]
        == (await checkpointer.aget(thread_1))["id"]
    )
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        await app.ainvoke(4, thread_1)
    # checkpoint is not updated
    state = await app.aget_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert state.next == ("one",)
    """we checkpoint inputs and it failed on "one", so the next node is one"""
    # we can recover from error by sending new inputs
    assert await app.ainvoke(2, thread_1) == 9
    state = await app.aget_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 16, "total is now 7+9=16"
    assert state.next == ()

    thread_2 = {"configurable": {"thread_id": "2"}}
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert await app.ainvoke(5, thread_2) == 5
    state = await app.aget_state({"configurable": {"thread_id": "1"}})
    assert state is not None
    assert state.values.get("total") == 16
    assert state.next == ()
    state = await app.aget_state(thread_2)
    assert state is not None
    assert state.values.get("total") == 5
    assert state.next == ()

    assert len([c async for c in app.aget_state_history(thread_1, limit=1)]) == 1
    # list all checkpoints for thread 1
    thread_1_history = [c async for c in app.aget_state_history(thread_1)]
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
    cursored = [
        c
        async for c in app.aget_state_history(
            thread_1, limit=1, before=thread_1_history[0].config
        )
    ]
    assert len(cursored) == 1
    assert cursored[0].config == thread_1_history[1].config
    # the last checkpoint
    assert thread_1_history[0].values["total"] == 16
    # the first "loop" checkpoint
    assert thread_1_history[-2].values["total"] == 2
    # can get each checkpoint using aget with config
    assert (await checkpointer.aget(thread_1_history[0].config))[
        "id"
    ] == thread_1_history[0].config["configurable"]["checkpoint_id"]
    assert (await checkpointer.aget(thread_1_history[1].config))[
        "id"
    ] == thread_1_history[1].config["configurable"]["checkpoint_id"]

    thread_1_next_config = await app.aupdate_state(thread_1_history[1].config, 10)
    # update creates a new checkpoint
    assert (
        thread_1_next_config["configurable"]["checkpoint_id"]
        > thread_1_history[0].config["configurable"]["checkpoint_id"]
    )
    # 1 more checkpoint in history
    assert len([c async for c in app.aget_state_history(thread_1)]) == 8
    assert Counter(
        [c.metadata["source"] async for c in app.aget_state_history(thread_1)]
    ) == {
        "update": 1,
        "input": 4,
        "loop": 3,
    }
    # the latest checkpoint is the updated one
    assert await app.aget_state(thread_1) == await app.aget_state(thread_1_next_config)


async def test_invoke_two_processes_two_in_join_two_out(mocker: MockerFixture) -> None:
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
        assert await app.ainvoke(2) == [13, 13]

    assert await asyncio.gather(*(app.ainvoke(2) for _ in range(100))) == [
        [13, 13] for _ in range(100)
    ]


async def test_invoke_join_then_call_other_pregel(mocker: MockerFixture) -> None:
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

    # Then invoke pubsub
    for _ in range(10):
        assert await app.ainvoke([2, 3]) == 27

    assert await asyncio.gather(*(app.ainvoke([2, 3]) for _ in range(10))) == [
        27 for _ in range(10)
    ]


async def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture) -> None:
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

    # Then invoke pubsub
    assert [c async for c in app.astream(2)] == [
        {"between": 3, "output": 3},
        {"between": 3, "output": 4},
    ]


async def test_invoke_two_processes_no_out(mocker: MockerFixture) -> None:
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
    # but returns nothing, as nothing was published to "output" topic
    assert await app.ainvoke(2) is None


async def test_channel_enter_exit_timing(mocker: MockerFixture) -> None:
    setup_sync = mocker.Mock()
    cleanup_sync = mocker.Mock()
    setup_async = mocker.Mock()
    cleanup_async = mocker.Mock()

    @contextmanager
    def an_int() -> Generator[int, None, None]:
        setup_sync()
        try:
            yield 5
        finally:
            cleanup_sync()

    @asynccontextmanager
    async def an_int_async() -> AsyncGenerator[int, None]:
        setup_async()
        try:
            yield 5
        finally:
            cleanup_async()

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = (
        Channel.subscribe_to("inbox")
        | RunnableLambda(add_one).abatch
        | Channel.write_to("output").abatch
    )

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "input": LastValue(int),
            "output": LastValue(int),
            "inbox": Topic(int),
            "ctx": Context(an_int, an_int_async),
        },
        input_channels="input",
        output_channels=["inbox", "output"],
        stream_channels=["inbox", "output"],
    )

    async def aenumerate(aiter: AsyncIterator[Any]) -> AsyncIterator[tuple[int, Any]]:
        i = 0
        async for chunk in aiter:
            yield i, chunk
            i += 1

    assert setup_sync.call_count == 0
    assert cleanup_sync.call_count == 0
    assert setup_async.call_count == 0
    assert cleanup_async.call_count == 0
    async for i, chunk in aenumerate(app.astream(2)):
        assert setup_sync.call_count == 0, "Sync context manager should not be used"
        assert cleanup_sync.call_count == 0, "Sync context manager should not be used"
        assert setup_async.call_count == 1, "Expected setup to be called once"
        if i == 0:
            assert chunk == {"inbox": [3]}
        elif i == 1:
            assert chunk == {"output": 4}
        else:
            assert False, "Expected only two chunks"
    assert setup_sync.call_count == 0
    assert cleanup_sync.call_count == 0
    assert setup_async.call_count == 1, "Expected setup to be called once"
    assert cleanup_async.call_count == 1, "Expected cleanup to be called once"


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_conditional_graph(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    from copy import deepcopy

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.tools import tool

    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

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

    async def agent_parser(input: str) -> Union[AgentAction, AgentFinish]:
        if input.startswith("finish"):
            _, answer = input.split(":")
            return AgentFinish(return_values={"answer": answer}, log=input)
        else:
            _, tool_name, tool_input = input.split(":")
            return AgentAction(tool=tool_name, tool_input=tool_input, log=input)

    agent = RunnablePassthrough.assign(agent_outcome=prompt | llm | agent_parser)

    # Define tool execution logic
    async def execute_tools(data: dict) -> dict:
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = await {t.name: t for t in tools}[agent_action.tool].ainvoke(
            agent_action.tool_input
        )
        if data.get("intermediate_steps") is None:
            data["intermediate_steps"] = []
        data["intermediate_steps"].append([agent_action, observation])
        return data

    # Define decision-making logic
    async def should_continue(data: dict, config: RunnableConfig) -> str:
        # Logic to decide whether to continue in the loop or exit
        if isinstance(data["agent_outcome"], AgentFinish):
            return "exit"
        else:
            return "continue"

    # Define a new graph
    workflow = Graph()

    workflow.add_node("agent", agent)
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    assert await app.ainvoke({"input": "what is weather in sf"}) == {
        "input": "what is weather in sf",
        "intermediate_steps": [
            [
                AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:query",
                ),
                "result for query",
            ],
            [
                AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
                "result for another",
            ],
        ],
        "agent_outcome": AgentFinish(
            return_values={"answer": "answer"}, log="finish:answer"
        ),
    }

    # deepcopy because the nodes mutate the data
    assert [
        deepcopy(c) async for c in app.astream({"input": "what is weather in sf"})
    ] == [
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
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
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
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ],
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ],
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ],
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ],
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "answer"}, log="finish:answer"
                ),
            }
        },
    ]

    patches = [c async for c in app.astream_log({"input": "what is weather in sf"})]
    patch_paths = {op["path"] for log in patches for op in log.ops}

    # Check that agent (one of the nodes) has its output streamed to the logs
    assert "/logs/agent/streamed_output/-" in patch_paths
    assert "/logs/agent:2/streamed_output/-" in patch_paths
    assert "/logs/agent:3/streamed_output/-" in patch_paths
    # Check that agent (one of the nodes) has its final output set in the logs
    assert "/logs/agent/final_output" in patch_paths
    assert "/logs/agent:2/final_output" in patch_paths
    assert "/logs/agent:3/final_output" in patch_paths
    assert [
        p["value"]
        for log in patches
        for p in log.ops
        if p["path"] == "/logs/agent/final_output"
        or p["path"] == "/logs/agent:2/final_output"
        or p["path"] == "/logs/agent:3/final_output"
    ] == [
        {
            "input": "what is weather in sf",
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
        },
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                    "result for query",
                ]
            ],
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="another",
                log="tool:search_api:another",
            ),
        },
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                    "result for query",
                ],
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="another",
                        log="tool:search_api:another",
                    ),
                    "result for another",
                ],
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "answer"}, log="finish:answer"
            ),
        },
    ]

    # test state get/update methods with interrupt_after

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"input": "what is weather in sf"}, config
        )
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

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": {
                "agent": {
                    "agent": {
                        "input": "what is weather in sf",
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                    }
                }
            },
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    await app_w_interrupt.aupdate_state(
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

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
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
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
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
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    await app_w_interrupt.aupdate_state(
        config,
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                ]
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                ),
            },
        },
        tasks=(),
        next=(),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
            "source": "update",
            "step": 4,
            "writes": {
                "agent": {
                    "input": "what is weather in sf",
                    "intermediate_steps": [
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:a different query",
                            ),
                            "result for query",
                        ]
                    ],
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    ),
                }
            },
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"input": "what is weather in sf"}, config
        )
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

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": {
                "agent": {
                    "agent": {
                        "input": "what is weather in sf",
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                    }
                }
            },
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    await app_w_interrupt.aupdate_state(
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

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
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
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
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
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    await app_w_interrupt.aupdate_state(
        config,
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                ]
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                ),
            },
        },
        tasks=(),
        next=(),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
            "source": "update",
            "step": 4,
            "writes": {
                "agent": {
                    "input": "what is weather in sf",
                    "intermediate_steps": [
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:a different query",
                            ),
                            "result for query",
                        ]
                    ],
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    ),
                }
            },
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    # test re-invoke to continue with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "3"}}
    llm.i = 0  # reset the llm

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"input": "what is weather in sf"}, config
        )
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

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": {
                "agent": {
                    "agent": {
                        "input": "what is weather in sf",
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                    }
                }
            },
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ],
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ],
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ],
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ],
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "answer"}, log="finish:answer"
                ),
            }
        },
    ]


async def test_conditional_graph_state(mocker: MockerFixture) -> None:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import tool

    setup = mocker.Mock()
    teardown = mocker.Mock()

    @asynccontextmanager
    async def assert_ctx_once() -> AsyncIterator[None]:
        assert setup.call_count == 0
        assert teardown.call_count == 0
        try:
            yield
        finally:
            assert setup.call_count == 1
            assert teardown.call_count == 1
            setup.reset_mock()
            teardown.reset_mock()

    class MyPydanticContextModel(BaseModel):
        class Config:
            arbitrary_types_allowed = True

        session: httpx.AsyncClient
        something_else: str

    @asynccontextmanager
    async def make_context(
        config: RunnableConfig,
    ) -> AsyncIterator[MyPydanticContextModel]:
        assert isinstance(config, dict)
        setup()
        session = httpx.AsyncClient()
        try:
            yield MyPydanticContextModel(session=session, something_else="hello")
        finally:
            await session.aclose()
            teardown()

    class AgentState(TypedDict):
        input: Annotated[str, UntrackedValue]
        agent_outcome: Optional[Union[AgentAction, AgentFinish]]
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
        context: Annotated[MyPydanticContextModel, Context(make_context)]

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
        # check we have httpx session in AgentState
        assert isinstance(data["context"], MyPydanticContextModel)
        # execute the tool
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = {t.name: t for t in tools}[agent_action.tool].invoke(
            agent_action.tool_input
        )
        return {"intermediate_steps": [[agent_action, observation]]}

    # Define decision-making logic
    def should_continue(data: AgentState) -> str:
        # check we have httpx session in AgentState
        assert isinstance(data["context"], MyPydanticContextModel)
        # Logic to decide whether to continue in the loop or exit
        if isinstance(data["agent_outcome"], AgentFinish):
            return "exit"
        else:
            return "continue"

    # Define a new graph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent)
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    async with assert_ctx_once():
        assert await app.ainvoke({"input": "what is weather in sf"}) == {
            "input": "what is weather in sf",
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                    "result for query",
                ],
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="another",
                        log="tool:search_api:another",
                    ),
                    "result for another",
                ],
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "answer"}, log="finish:answer"
            ),
        }

    async with assert_ctx_once():
        assert [c async for c in app.astream({"input": "what is weather in sf"})] == [
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
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:query",
                            ),
                            "result for query",
                        ]
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
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="another",
                                log="tool:search_api:another",
                            ),
                            "result for another",
                        ],
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

    async with assert_ctx_once():
        patches = [c async for c in app.astream_log({"input": "what is weather in sf"})]
    patch_paths = {op["path"] for log in patches for op in log.ops}

    # Check that agent (one of the nodes) has its output streamed to the logs
    assert "/logs/agent/streamed_output/-" in patch_paths
    # Check that agent (one of the nodes) has its final output set in the logs
    assert "/logs/agent/final_output" in patch_paths
    assert [
        p["value"]
        for log in patches
        for p in log.ops
        if p["path"] == "/logs/agent/final_output"
        or p["path"] == "/logs/agent:2/final_output"
        or p["path"] == "/logs/agent:3/final_output"
    ] == [
        {
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            )
        },
        {
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="another", log="tool:search_api:another"
            )
        },
        {
            "agent_outcome": AgentFinish(
                return_values={"answer": "answer"}, log="finish:answer"
            ),
        },
    ]

    # test state get/update methods with interrupt_after

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    async with assert_ctx_once():
        assert [
            c
            async for c in app_w_interrupt.astream(
                {"input": "what is weather in sf"}, config
            )
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

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:query",
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
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
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    async with assert_ctx_once():
        await app_w_interrupt.aupdate_state(
            config,
            {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                )
            },
        )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
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
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    async with assert_ctx_once():
        assert [c async for c in app_w_interrupt.astream(None, config)] == [
            {
                "tools": {
                    "intermediate_steps": [
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:a different query",
                            ),
                            "result for query",
                        ]
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

    async with assert_ctx_once():
        await app_w_interrupt.aupdate_state(
            config,
            {
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                )
            },
        )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                ]
            ],
        },
        tasks=(),
        next=(),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
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
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=MemorySaverAssertImmutable(),
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"input": "what is weather in sf"}, config
        )
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
    ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
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
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    await app_w_interrupt.aupdate_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            )
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
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
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": {
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
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

    await app_w_interrupt.aupdate_state(
        config,
        {
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            )
        },
    )

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                ]
            ],
        },
        tasks=(),
        next=(),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
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
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )


async def test_conditional_entrypoint_graph() -> None:
    async def left(data: str) -> str:
        return data + "->left"

    async def right(data: str) -> str:
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

    workflow.add_conditional_edges("left", lambda data: END)
    workflow.add_edge("right", END)

    app = workflow.compile()

    assert await app.ainvoke("what is weather in sf") == "what is weather in sf->right"

    assert [c async for c in app.astream("what is weather in sf")] == [
        {"right": "what is weather in sf->right"},
    ]


async def test_conditional_entrypoint_graph_state() -> None:
    class AgentState(TypedDict, total=False):
        input: str
        output: str
        steps: Annotated[list[str], operator.add]

    async def left(data: AgentState) -> AgentState:
        return {"output": data["input"] + "->left"}

    async def right(data: AgentState) -> AgentState:
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

    workflow.add_conditional_edges("left", lambda data: END)
    workflow.add_edge("right", END)

    app = workflow.compile()

    assert await app.ainvoke({"input": "what is weather in sf"}) == {
        "input": "what is weather in sf",
        "output": "what is weather in sf->right",
        "steps": [],
    }

    assert [c async for c in app.astream({"input": "what is weather in sf"})] == [
        {"right": {"output": "what is weather in sf->right"}},
    ]


async def test_prebuilt_tool_chat() -> None:
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

    app = create_tool_calling_executor(
        FakeFuntionChatModel(
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
        ),
        tools,
    )

    assert await app.ainvoke(
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

    assert [
        c
        async for c in app.astream(
            {"messages": [HumanMessage(content="what is weather in sf")]}
        )
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
                        tool_call_id="tool_call234",
                        name="search_api",
                        id=AnyStr(),
                    ),
                    ToolMessage(
                        content="result for a third one",
                        tool_call_id="tool_call567",
                        name="search_api",
                        id=AnyStr(),
                    ),
                ]
            }
        },
        {"agent": {"messages": [_AnyIdAIMessage(content="answer")]}},
    ]


# defined outside to allow deserializer to see it
class ToolInput(BaseModel, arbitrary_types_allowed=True):
    call: ToolCall
    my_session: httpx.AsyncClient


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_state_graph_packets(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolMessage,
    )
    from langchain_core.tools import tool

    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        session: Annotated[httpx.AsyncClient, Context(httpx.AsyncClient)]

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

    # Define decision-making logic
    def should_continue(data: AgentState) -> str:
        assert isinstance(data["session"], httpx.AsyncClient)
        # Logic to decide whether to continue in the loop or exit
        if tool_calls := data["messages"][-1].tool_calls:
            return [
                Send("tools", ToolInput(call=tool_call, my_session=data["session"]))
                for tool_call in tool_calls
            ]
        else:
            return END

    async def tools_node(input: ToolInput, config: RunnableConfig) -> AgentState:
        assert isinstance(input.my_session, httpx.AsyncClient)
        tool_call = input.call
        await asyncio.sleep(tool_call["args"].get("idx", 0) / 10)
        output = await tools_by_name[tool_call["name"]].ainvoke(
            tool_call["args"], config
        )
        return {
            "messages": ToolMessage(
                content=output, name=tool_call["name"], tool_call_id=tool_call["id"]
            )
        }

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", {"messages": RunnablePick("messages") | model})
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

    assert await app.ainvoke(
        {"messages": HumanMessage(content="what is weather in sf")}
    ) == {
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
        async for c in app.astream(
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
        checkpointer=checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
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

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
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
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=(await app_w_interrupt.checkpointer.aget_tuple(config)).config,
        created_at=(await app_w_interrupt.checkpointer.aget_tuple(config)).checkpoint[
            "ts"
        ],
        metadata={
            "parents": {},
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
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    # modify ai message
    last_message = (await app_w_interrupt.aget_state(config)).values["messages"][-1]
    last_message.tool_calls[0]["args"]["query"] = "a different query"
    await app_w_interrupt.aupdate_state(config, {"messages": last_message})

    # message was replaced instead of appended
    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
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
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
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
                    )
                }
            },
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
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

    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
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
        tasks=(PregelTask(AnyStr(), "tools"), PregelTask(AnyStr(), "tools")),
        next=("tools", "tools"),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
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
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    await app_w_interrupt.aupdate_state(
        config,
        {"messages": AIMessage(content="answer", id="ai2")},
    )

    # replaces message even if object identity is different, as long as id is the same
    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
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
        tasks=(),
        next=(),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
            "writes": {"agent": {"messages": AIMessage(content="answer", id="ai2")}},
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_message_graph(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    from langchain_core.agents import AgentAction
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
    from langchain_core.tools import tool

    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list):
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
                additional_kwargs={
                    "function_call": {
                        "name": "search_api",
                        "arguments": json.dumps("query"),
                    }
                },
                id="ai1",
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "search_api",
                        "arguments": json.dumps("another"),
                    }
                },
                id="ai2",
            ),
            AIMessage(content="answer", id="ai3"),
        ]
    )

    tool_executor = ToolExecutor(tools)

    # Define the function that determines whether to continue or not
    def should_continue(messages):
        last_message = messages[-1]
        # If there is no function call, then we finish
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    async def call_tool(messages):
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an AgentAction from the function_call
        action = AgentAction(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(
                last_message.additional_kwargs["function_call"]["arguments"]
            ),
            log="",
        )
        # We call the tool_executor and get back a response
        response = await tool_executor.ainvoke(action)
        # We use the response to create a FunctionMessage
        return FunctionMessage(content=str(response), name=action.tool)

    # Define a new graph
    workflow = MessageGraph()

    # Define the two nodes we will cycle between
    workflow.add_node("agent", model)
    workflow.add_node("tools", call_tool)

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

    assert await app.ainvoke(HumanMessage(content="what is weather in sf")) == [
        _AnyIdHumanMessage(content="what is weather in sf"),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {"name": "search_api", "arguments": '"query"'}
            },
            id="ai1",
        ),
        FunctionMessage(content="result for query", name="search_api", id=AnyStr()),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {"name": "search_api", "arguments": '"another"'}
            },
            id="ai2",
        ),
        FunctionMessage(content="result for another", name="search_api", id=AnyStr()),
        AIMessage(content="answer", id="ai3"),
    ]

    assert [
        c async for c in app.astream([HumanMessage(content="what is weather in sf")])
    ] == [
        {
            "agent": AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"query"'}
                },
                id="ai1",
            )
        },
        {
            "tools": FunctionMessage(
                content="result for query", name="search_api", id=AnyStr()
            )
        },
        {
            "agent": AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"another"'}
                },
                id="ai2",
            )
        },
        {
            "tools": FunctionMessage(
                content="result for another", name="search_api", id=AnyStr()
            )
        },
        {"agent": AIMessage(content="answer", id="ai3")},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app_w_interrupt.astream(
            HumanMessage(content="what is weather in sf"), config
        )
    ] == [
        {
            "agent": AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"query"'}
                },
                id="ai1",
            )
        },
    ]

    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"query"'}
                },
                id="ai1",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {"name": "search_api", "arguments": '"query"'}
                    },
                    id="ai1",
                )
            },
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    # modify ai message
    last_message = (await app_w_interrupt.aget_state(config)).values[-1]
    last_message.additional_kwargs["function_call"]["arguments"] = '"a different query"'
    await app_w_interrupt.aupdate_state(config, last_message)

    # message was replaced instead of appended
    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "search_api",
                        "arguments": '"a different query"',
                    }
                },
                id="ai1",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
            "writes": {
                "agent": AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "search_api",
                            "arguments": '"a different query"',
                        }
                    },
                    id="ai1",
                )
            },
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {
            "tools": FunctionMessage(
                content="result for a different query",
                name="search_api",
                id=AnyStr(),
            )
        },
        {
            "agent": AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"another"'}
                },
                id="ai2",
            )
        },
    ]

    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "search_api",
                        "arguments": '"a different query"',
                    }
                },
                id="ai1",
            ),
            FunctionMessage(
                content="result for a different query",
                name="search_api",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {"name": "search_api", "arguments": '"another"'}
                },
                id="ai2",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools"),),
        next=("tools",),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 4,
            "writes": {
                "agent": AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "search_api",
                            "arguments": '"another"',
                        }
                    },
                    id="ai2",
                )
            },
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )

    await app_w_interrupt.aupdate_state(
        config,
        AIMessage(content="answer", id="ai2"),
    )

    # replaces message even if object identity is different, as long as id is the same
    tup = await app_w_interrupt.checkpointer.aget_tuple(config)
    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "search_api",
                        "arguments": '"a different query"',
                    }
                },
                id="ai1",
            ),
            FunctionMessage(
                content="result for a different query",
                name="search_api",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        tasks=(),
        next=(),
        config=tup.config,
        created_at=tup.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
            "writes": {"agent": AIMessage(content="answer", id="ai2")},
        },
        parent_config=[
            c async for c in app_w_interrupt.checkpointer.alist(config, limit=2)
        ][-1].config,
    )


async def test_in_one_fan_out_out_one_graph_state() -> None:
    def sorted_add(x: list[str], y: list[str]) -> list[str]:
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], operator.add]

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        await asyncio.sleep(0.1)
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
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

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    assert [
        c
        async for c in app.astream(
            {"query": "what is weather in sf"}, stream_mode="values"
        )
    ] == [
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
        c
        async for c in app.astream(
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
                    "error": None,
                    "interrupts": [],
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
                    "error": None,
                    "interrupts": [],
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
                    "error": None,
                    "interrupts": [],
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
                    "error": None,
                    "interrupts": [],
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_start_branch_then(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str
        shared: Annotated[dict[str, dict[str, Any]], SharedValue.on("assistant_id")]
        other: Annotated[dict[str, dict[str, Any]], SharedValue.on("assistant_id")]

    def assert_shared_value(data: State, config: RunnableConfig) -> State:
        assert "shared" in data
        if thread_id := config["configurable"].get("thread_id"):
            if thread_id == "1":
                # this is the first thread, so should not see a value
                assert data["shared"] == {}
                return {"shared": {"1": {"hello": "world"}}, "other": {"2": {1: 2}}}
            elif thread_id == "2":
                # this should get value saved by thread 1
                assert data["shared"] == {"1": {"hello": "world"}}
            elif thread_id == "3":
                # this is a different assistant, so should not see previous value
                assert data["shared"] == {}
        return {}

    def tool_two_slow(data: State, config: RunnableConfig) -> State:
        return {"my_key": " slow", **assert_shared_value(data, config)}

    def tool_two_fast(data: State, config: RunnableConfig) -> State:
        return {"my_key": " fast", **assert_shared_value(data, config)}

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two_slow", tool_two_slow)
    tool_two_graph.add_node("tool_two_fast", tool_two_fast)
    tool_two_graph.set_conditional_entry_point(
        lambda s: "tool_two_slow" if s["market"] == "DE" else "tool_two_fast", then=END
    )
    tool_two = tool_two_graph.compile()

    assert await tool_two.ainvoke({"my_key": "value", "market": "DE"}) == {
        "my_key": "value slow",
        "market": "DE",
    }
    assert await tool_two.ainvoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value fast",
        "market": "US",
    }

    tool_two = tool_two_graph.compile(
        store=MemoryStore(),
        checkpointer=checkpointer,
        interrupt_before=["tool_two_fast", "tool_two_slow"],
    )

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        await tool_two.ainvoke({"my_key": "value", "market": "DE"})

    thread1 = {"configurable": {"thread_id": "1", "assistant_id": "a"}}
    # stop when about to enter node
    assert await tool_two.ainvoke({"my_key": "value", "market": "DE"}, thread1) == {
        "my_key": "value",
        "market": "DE",
    }
    assert [c.metadata async for c in tool_two.checkpointer.alist(thread1)] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
        },
        {
            "parents": {},
            "source": "input",
            "step": -1,
            "writes": {"my_key": "value", "market": "DE"},
        },
    ]
    assert await tool_two.aget_state(thread1) == StateSnapshot(
        values={"my_key": "value", "market": "DE"},
        tasks=(PregelTask(AnyStr(), "tool_two_slow"),),
        next=("tool_two_slow",),
        config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread1)).checkpoint["ts"],
        metadata={"parents": {}, "source": "loop", "step": 0, "writes": None},
        parent_config=[c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
            -1
        ].config,
    )
    # resume, for same result as above
    assert await tool_two.ainvoke(None, thread1, debug=1) == {
        "my_key": "value slow",
        "market": "DE",
    }
    assert await tool_two.aget_state(thread1) == StateSnapshot(
        values={"my_key": "value slow", "market": "DE"},
        tasks=(),
        next=(),
        config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread1)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"tool_two_slow": {"my_key": " slow"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
            -1
        ].config,
    )

    thread2 = {"configurable": {"thread_id": "2", "assistant_id": "a"}}
    # stop when about to enter node
    assert await tool_two.ainvoke({"my_key": "value", "market": "US"}, thread2) == {
        "my_key": "value",
        "market": "US",
    }
    assert await tool_two.aget_state(thread2) == StateSnapshot(
        values={"my_key": "value", "market": "US"},
        tasks=(PregelTask(AnyStr(), "tool_two_fast"),),
        next=("tool_two_fast",),
        config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread2)).checkpoint["ts"],
        metadata={"parents": {}, "source": "loop", "step": 0, "writes": None},
        parent_config=[c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
            -1
        ].config,
    )
    # resume, for same result as above
    assert await tool_two.ainvoke(None, thread2, debug=1) == {
        "my_key": "value fast",
        "market": "US",
    }
    assert await tool_two.aget_state(thread2) == StateSnapshot(
        values={"my_key": "value fast", "market": "US"},
        tasks=(),
        next=(),
        config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread2)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"tool_two_fast": {"my_key": " fast"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
            -1
        ].config,
    )

    thread3 = {"configurable": {"thread_id": "3", "assistant_id": "b"}}
    # stop when about to enter node
    assert await tool_two.ainvoke({"my_key": "value", "market": "US"}, thread3) == {
        "my_key": "value",
        "market": "US",
    }
    assert await tool_two.aget_state(thread3) == StateSnapshot(
        values={"my_key": "value", "market": "US"},
        tasks=(PregelTask(AnyStr(), "tool_two_fast"),),
        next=("tool_two_fast",),
        config=(await tool_two.checkpointer.aget_tuple(thread3)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread3)).checkpoint["ts"],
        metadata={"parents": {}, "source": "loop", "step": 0, "writes": None},
        parent_config=[c async for c in tool_two.checkpointer.alist(thread3, limit=2)][
            -1
        ].config,
    )
    # update state
    await tool_two.aupdate_state(thread3, {"my_key": "key"})  # appends to my_key
    assert await tool_two.aget_state(thread3) == StateSnapshot(
        values={"my_key": "valuekey", "market": "US"},
        tasks=(PregelTask(AnyStr(), "tool_two_fast"),),
        next=("tool_two_fast",),
        config=(await tool_two.checkpointer.aget_tuple(thread3)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread3)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 1,
            "writes": {START: {"my_key": "key"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread3, limit=2)][
            -1
        ].config,
    )
    # resume, for same result as above
    assert await tool_two.ainvoke(None, thread3, debug=1) == {
        "my_key": "valuekey fast",
        "market": "US",
    }
    assert await tool_two.aget_state(thread3) == StateSnapshot(
        values={"my_key": "valuekey fast", "market": "US"},
        tasks=(),
        next=(),
        config=(await tool_two.checkpointer.aget_tuple(thread3)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread3)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 2,
            "writes": {"tool_two_fast": {"my_key": " fast"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread3, limit=2)][
            -1
        ].config,
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_branch_then(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

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

    assert await tool_two.ainvoke({"my_key": "value", "market": "DE"}, debug=1) == {
        "my_key": "value prepared slow finished",
        "market": "DE",
    }
    assert await tool_two.ainvoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value prepared fast finished",
        "market": "US",
    }

    # test stream_mode=debug
    tool_two = tool_two_graph.compile(checkpointer=checkpointer)
    thread10 = {"configurable": {"thread_id": "10"}}
    assert [
        c
        async for c in tool_two.astream(
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
                    "parents": {},
                    "source": "input",
                    "step": -1,
                    "writes": {"my_key": "value", "market": "DE"},
                },
                "next": ["__start__"],
                "tasks": [{"id": AnyStr(), "name": "__start__", "interrupts": ()}],
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
                    "parents": {},
                    "source": "loop",
                    "step": 0,
                    "writes": None,
                },
                "next": ["prepare"],
                "tasks": [{"id": AnyStr(), "name": "prepare", "interrupts": ()}],
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
                "error": None,
                "interrupts": [],
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
                    "parents": {},
                    "source": "loop",
                    "step": 1,
                    "writes": {"prepare": {"my_key": " prepared"}},
                },
                "next": ["tool_two_slow"],
                "tasks": [{"id": AnyStr(), "name": "tool_two_slow", "interrupts": ()}],
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
                "error": None,
                "interrupts": [],
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
                    "parents": {},
                    "source": "loop",
                    "step": 2,
                    "writes": {"tool_two_slow": {"my_key": " slow"}},
                },
                "next": ["finish"],
                "tasks": [{"id": AnyStr(), "name": "finish", "interrupts": ()}],
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
                "error": None,
                "interrupts": [],
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
                    "parents": {},
                    "source": "loop",
                    "step": 3,
                    "writes": {"finish": {"my_key": " finished"}},
                },
                "next": [],
                "tasks": [],
            },
        },
    ]

    tool_two = tool_two_graph.compile(
        checkpointer=checkpointer, interrupt_before=["tool_two_fast", "tool_two_slow"]
    )

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        await tool_two.ainvoke({"my_key": "value", "market": "DE"})

    thread1 = {"configurable": {"thread_id": "11"}}
    # stop when about to enter node
    assert [
        c
        async for c in tool_two.astream(
            {"my_key": "value", "market": "DE"}, thread1, stream_mode="debug"
        )
    ] == [
        {
            "type": "checkpoint",
            "timestamp": AnyStr(),
            "step": -1,
            "payload": {
                "config": {
                    "tags": [],
                    "metadata": {"thread_id": "11"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "11",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "values": {"my_key": ""},
                "metadata": {
                    "parents": {},
                    "source": "input",
                    "step": -1,
                    "writes": {"my_key": "value", "market": "DE"},
                },
                "next": ["__start__"],
                "tasks": [{"id": AnyStr(), "name": "__start__", "interrupts": ()}],
            },
        },
        {
            "type": "checkpoint",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "config": {
                    "tags": [],
                    "metadata": {"thread_id": "11"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "11",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "values": {
                    "my_key": "value",
                    "market": "DE",
                },
                "metadata": {
                    "parents": {},
                    "source": "loop",
                    "step": 0,
                    "writes": None,
                },
                "next": ["prepare"],
                "tasks": [{"id": AnyStr(), "name": "prepare", "interrupts": ()}],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": "1a591be4-f85c-558f-8d00-1ccac0d1877f",
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
                "id": "1a591be4-f85c-558f-8d00-1ccac0d1877f",
                "name": "prepare",
                "result": [("my_key", " prepared")],
                "error": None,
                "interrupts": [],
            },
        },
        {
            "type": "checkpoint",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "config": {
                    "tags": [],
                    "metadata": {"thread_id": "11"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "11",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "values": {
                    "my_key": "value prepared",
                    "market": "DE",
                },
                "metadata": {
                    "parents": {},
                    "source": "loop",
                    "step": 1,
                    "writes": {"prepare": {"my_key": " prepared"}},
                },
                "next": ["tool_two_slow"],
                "tasks": [{"id": AnyStr(), "name": "tool_two_slow", "interrupts": ()}],
            },
        },
    ]
    assert await tool_two.aget_state(thread1) == StateSnapshot(
        values={"my_key": "value prepared", "market": "DE"},
        tasks=(PregelTask(AnyStr(), "tool_two_slow"),),
        next=("tool_two_slow",),
        config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread1)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"prepare": {"my_key": " prepared"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
            -1
        ].config,
    )
    # resume, for same result as above
    assert await tool_two.ainvoke(None, thread1, debug=1) == {
        "my_key": "value prepared slow finished",
        "market": "DE",
    }
    assert await tool_two.aget_state(thread1) == StateSnapshot(
        values={"my_key": "value prepared slow finished", "market": "DE"},
        tasks=(),
        next=(),
        config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread1)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 3,
            "writes": {"finish": {"my_key": " finished"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
            -1
        ].config,
    )

    thread2 = {"configurable": {"thread_id": "12"}}
    # stop when about to enter node
    assert await tool_two.ainvoke({"my_key": "value", "market": "US"}, thread2) == {
        "my_key": "value prepared",
        "market": "US",
    }
    assert await tool_two.aget_state(thread2) == StateSnapshot(
        values={"my_key": "value prepared", "market": "US"},
        tasks=(PregelTask(AnyStr(), "tool_two_fast"),),
        next=("tool_two_fast",),
        config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread2)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"prepare": {"my_key": " prepared"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
            -1
        ].config,
    )
    # resume, for same result as above
    assert await tool_two.ainvoke(None, thread2, debug=1) == {
        "my_key": "value prepared fast finished",
        "market": "US",
    }
    assert await tool_two.aget_state(thread2) == StateSnapshot(
        values={"my_key": "value prepared fast finished", "market": "US"},
        tasks=(),
        next=(),
        config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread2)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 3,
            "writes": {"finish": {"my_key": " finished"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
            -1
        ].config,
    )

    tool_two = tool_two_graph.compile(
        checkpointer=checkpointer, interrupt_after=["prepare"]
    )

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        await tool_two.ainvoke({"my_key": "value", "market": "DE"})

    thread1 = {"configurable": {"thread_id": "21"}}
    # stop when about to enter node
    assert await tool_two.ainvoke({"my_key": "value", "market": "DE"}, thread1) == {
        "my_key": "value prepared",
        "market": "DE",
    }
    assert await tool_two.aget_state(thread1) == StateSnapshot(
        values={"my_key": "value prepared", "market": "DE"},
        tasks=(PregelTask(AnyStr(), "tool_two_slow"),),
        next=("tool_two_slow",),
        config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread1)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"prepare": {"my_key": " prepared"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
            -1
        ].config,
    )
    # resume, for same result as above
    assert await tool_two.ainvoke(None, thread1, debug=1) == {
        "my_key": "value prepared slow finished",
        "market": "DE",
    }
    assert await tool_two.aget_state(thread1) == StateSnapshot(
        values={"my_key": "value prepared slow finished", "market": "DE"},
        tasks=(),
        next=(),
        config=(await tool_two.checkpointer.aget_tuple(thread1)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread1)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 3,
            "writes": {"finish": {"my_key": " finished"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread1, limit=2)][
            -1
        ].config,
    )

    thread2 = {"configurable": {"thread_id": "22"}}
    # stop when about to enter node
    assert await tool_two.ainvoke({"my_key": "value", "market": "US"}, thread2) == {
        "my_key": "value prepared",
        "market": "US",
    }
    assert await tool_two.aget_state(thread2) == StateSnapshot(
        values={"my_key": "value prepared", "market": "US"},
        tasks=(PregelTask(AnyStr(), "tool_two_fast"),),
        next=("tool_two_fast",),
        config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread2)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"prepare": {"my_key": " prepared"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
            -1
        ].config,
    )
    # resume, for same result as above
    assert await tool_two.ainvoke(None, thread2, debug=1) == {
        "my_key": "value prepared fast finished",
        "market": "US",
    }
    assert await tool_two.aget_state(thread2) == StateSnapshot(
        values={"my_key": "value prepared fast finished", "market": "US"},
        tasks=(),
        next=(),
        config=(await tool_two.checkpointer.aget_tuple(thread2)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread2)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 3,
            "writes": {"finish": {"my_key": " finished"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread2, limit=2)][
            -1
        ].config,
    )

    thread3 = {"configurable": {"thread_id": "23"}}
    # update an empty thread before first run
    uconfig = await tool_two.aupdate_state(thread3, {"my_key": "key", "market": "DE"})
    # check current state
    assert await tool_two.aget_state(thread3) == StateSnapshot(
        values={"my_key": "key", "market": "DE"},
        tasks=(PregelTask(AnyStr(), "prepare"),),
        next=("prepare",),
        config=uconfig,
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 0,
            "writes": {START: {"my_key": "key", "market": "DE"}},
        },
        parent_config=None,
    )
    # run from this point
    assert await tool_two.ainvoke(None, thread3) == {
        "my_key": "key prepared",
        "market": "DE",
    }
    # get state after first node
    assert await tool_two.aget_state(thread3) == StateSnapshot(
        values={"my_key": "key prepared", "market": "DE"},
        tasks=(PregelTask(AnyStr(), "tool_two_slow"),),
        next=("tool_two_slow",),
        config=(await tool_two.checkpointer.aget_tuple(thread3)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread3)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"prepare": {"my_key": " prepared"}},
        },
        parent_config=uconfig,
    )
    # resume, for same result as above
    assert await tool_two.ainvoke(None, thread3, debug=1) == {
        "my_key": "key prepared slow finished",
        "market": "DE",
    }
    assert await tool_two.aget_state(thread3) == StateSnapshot(
        values={"my_key": "key prepared slow finished", "market": "DE"},
        tasks=(),
        next=(),
        config=(await tool_two.checkpointer.aget_tuple(thread3)).config,
        created_at=(await tool_two.checkpointer.aget_tuple(thread3)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 3,
            "writes": {"finish": {"my_key": " finished"}},
        },
        parent_config=[c async for c in tool_two.checkpointer.alist(thread3, limit=2)][
            -1
        ].config,
    )


async def test_in_one_fan_out_state_graph_waiting_edge() -> None:
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

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        await asyncio.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
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

    app = workflow.compile()

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
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
        c
        async for c in app_w_interrupt.astream(
            {"query": "what is weather in sf"}, config
        )
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
    ]

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


async def test_in_one_fan_out_state_graph_waiting_edge_via_branch(
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

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        await asyncio.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
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
    workflow.add_conditional_edges(
        "rewrite_query", lambda _: "retriever_two", {"retriever_two": "retriever_two"}
    )
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.get_graph().draw_ascii() == snapshot

    assert await app.ainvoke({"query": "what is weather in sf"}, debug=True) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
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
        c
        async for c in app_w_interrupt.astream(
            {"query": "what is weather in sf"}, config
        )
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
    ]

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


async def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class(
    snapshot: SnapshotAssertion, mocker: MockerFixture
) -> None:
    from langchain_core.pydantic_v1 import BaseModel, ValidationError

    setup = mocker.Mock()
    teardown = mocker.Mock()

    @asynccontextmanager
    async def assert_ctx_once() -> AsyncIterator[None]:
        assert setup.call_count == 0
        assert teardown.call_count == 0
        try:
            yield
        finally:
            assert setup.call_count == 1
            assert teardown.call_count == 1
            setup.reset_mock()
            teardown.reset_mock()

    @asynccontextmanager
    async def make_httpx_client() -> AsyncIterator[httpx.AsyncClient]:
        setup()
        async with httpx.AsyncClient() as client:
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

    class State(BaseModel):
        class Config:
            arbitrary_types_allowed = True

        query: str
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]
        client: Annotated[httpx.AsyncClient, Context(make_httpx_client)]

    class Input(BaseModel):
        query: str

    class Output(BaseModel):
        answer: str
        docs: list[str]

    class StateUpdate(BaseModel):
        query: Optional[str] = None
        answer: Optional[str] = None
        docs: Optional[list[str]] = None

    async def rewrite_query(data: State) -> State:
        return {"query": f"query: {data.query}"}

    async def analyzer_one(data: State) -> State:
        return StateUpdate(query=f"analyzed: {data.query}")

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        await asyncio.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
        return {"answer": ",".join(data.docs)}

    async def decider(data: State) -> str:
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

    assert app.get_graph().draw_ascii() == snapshot

    async with assert_ctx_once():
        with pytest.raises(ValidationError):
            await app.ainvoke({"query": {}})

    async with assert_ctx_once():
        assert await app.ainvoke({"query": "what is weather in sf"}) == {
            "docs": ["doc1", "doc2", "doc3", "doc4"],
            "answer": "doc1,doc2,doc3,doc4",
        }

    async with assert_ctx_once():
        assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
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

    async with assert_ctx_once():
        assert [
            c
            async for c in app_w_interrupt.astream(
                {"query": "what is weather in sf"}, config
            )
        ] == [
            {"rewrite_query": {"query": "query: what is weather in sf"}},
            {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
            {"retriever_two": {"docs": ["doc3", "doc4"]}},
            {"retriever_one": {"docs": ["doc1", "doc2"]}},
        ]

    async with assert_ctx_once():
        assert [c async for c in app_w_interrupt.astream(None, config)] == [
            {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        ]

    assert await app_w_interrupt.aget_state(config) == StateSnapshot(
        values={
            "query": "analyzed: query: what is weather in sf",
            "answer": "doc1,doc2,doc3,doc4",
            "docs": ["doc1", "doc2", "doc3", "doc4"],
        },
        tasks=(),
        next=(),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "writes": {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
            "step": 4,
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
    )

    async with assert_ctx_once():
        assert await app_w_interrupt.aupdate_state(
            config, {"docs": ["doc5"]}, as_node="rewrite_query"
        ) == {
            "configurable": {
                "thread_id": "1",
                "checkpoint_id": AnyStr(),
                "checkpoint_ns": "",
            }
        }


async def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic2(
    snapshot: SnapshotAssertion,
) -> None:
    from pydantic import BaseModel, ValidationError

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
        query: str
        inner: InnerObject
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]

    class StateUpdate(BaseModel):
        query: Optional[str] = None
        answer: Optional[str] = None
        docs: Optional[list[str]] = None

    async def rewrite_query(data: State) -> State:
        return {"query": f"query: {data.query}"}

    async def analyzer_one(data: State) -> State:
        return StateUpdate(query=f"analyzed: {data.query}")

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        await asyncio.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
        return {"answer": ",".join(data.docs)}

    async def decider(data: State) -> str:
        assert isinstance(data, State)
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
    workflow.add_conditional_edges(
        "rewrite_query", decider, {"retriever_two": "retriever_two"}
    )
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
    assert app.get_input_schema().schema() == snapshot
    assert app.get_output_schema().schema() == snapshot

    with pytest.raises(ValidationError):
        await app.ainvoke({"query": {}})

    assert await app.ainvoke(
        {"query": "what is weather in sf", "inner": {"yo": 1}}
    ) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
        "inner": {"yo": 1},
    }

    assert [
        c
        async for c in app.astream(
            {"query": "what is weather in sf", "inner": {"yo": 1}}
        )
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

    assert [
        c
        async for c in app_w_interrupt.astream(
            {"query": "what is weather in sf", "inner": {"yo": 1}}, config
        )
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
    ]

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    assert await app_w_interrupt.aupdate_state(
        config, {"docs": ["doc5"]}, as_node="rewrite_query"
    ) == {
        "configurable": {
            "thread_id": "1",
            "checkpoint_id": AnyStr(),
            "checkpoint_ns": "",
        }
    }


async def test_in_one_fan_out_state_graph_waiting_edge_plus_regular() -> None:
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

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def analyzer_one(data: State) -> State:
        await asyncio.sleep(0.1)
        return {"query": f'analyzed: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        await asyncio.sleep(0.2)
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
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

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
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
        c
        async for c in app_w_interrupt.astream(
            {"query": "what is weather in sf"}, config
        )
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"qa": {"answer": ""}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
    ]

    assert [c async for c in app_w_interrupt.astream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


async def test_in_one_fan_out_state_graph_waiting_edge_multiple() -> None:
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

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        await asyncio.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    async def decider(data: State) -> None:
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

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: analyzed: query: what is weather in sf",
        "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
        "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
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


async def test_in_one_fan_out_state_graph_waiting_edge_multiple_cond_edge() -> None:
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

    async def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    async def retriever_picker(data: State) -> list[str]:
        return ["analyzer_one", "retriever_two"]

    async def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    async def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    async def retriever_two(data: State) -> State:
        await asyncio.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    async def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    async def decider(data: State) -> None:
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

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: analyzed: query: what is weather in sf",
        "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
        "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
    }

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
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


async def test_nested_graph(snapshot: SnapshotAssertion) -> None:
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

    async def side(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile())
    graph.add_node("side", side)
    graph.set_entry_point("inner")
    graph.add_edge("inner", "side")
    graph.set_finish_point("side")

    app = graph.compile()

    assert app.get_graph().draw_ascii() == snapshot
    assert await app.ainvoke({"my_key": "my value", "never_called": never_called}) == {
        "my_key": "my value there and back again",
        "never_called": never_called,
    }
    assert [
        chunk
        async for chunk in app.astream(
            {"my_key": "my value", "never_called": never_called}
        )
    ] == [
        {"inner": {"my_key": "my value there"}},
        {"side": {"my_key": "my value there and back again"}},
    ]
    assert [
        chunk
        async for chunk in app.astream(
            {"my_key": "my value", "never_called": never_called}, stream_mode="values"
        )
    ] == [
        {"my_key": "my value", "never_called": never_called},
        {"my_key": "my value there", "never_called": never_called},
        {"my_key": "my value there and back again", "never_called": never_called},
    ]
    times_called = 0
    async for event in app.astream_events(
        {"my_key": "my value", "never_called": never_called},
        version="v2",
        config={"run_id": UUID(int=0)},
        stream_mode="values",
    ):
        if event["event"] == "on_chain_end" and event["run_id"] == str(UUID(int=0)):
            times_called += 1
            assert event["data"] == {
                "output": {
                    "my_key": "my value there and back again",
                    "never_called": never_called,
                }
            }
    assert times_called == 1
    times_called = 0
    async for event in app.astream_events(
        {"my_key": "my value", "never_called": never_called},
        version="v2",
        config={"run_id": UUID(int=0)},
    ):
        if event["event"] == "on_chain_end" and event["run_id"] == str(UUID(int=0)):
            times_called += 1
            assert event["data"] == {
                "output": {
                    "my_key": "my value there and back again",
                    "never_called": never_called,
                }
            }
    assert times_called == 1

    chain = app | RunnablePassthrough()

    assert await chain.ainvoke(
        {"my_key": "my value", "never_called": never_called}
    ) == {
        "my_key": "my value there and back again",
        "never_called": never_called,
    }
    assert [
        chunk
        async for chunk in chain.astream(
            {"my_key": "my value", "never_called": never_called}
        )
    ] == [
        {"inner": {"my_key": "my value there"}},
        {"side": {"my_key": "my value there and back again"}},
    ]
    times_called = 0
    async for event in chain.astream_events(
        {"my_key": "my value", "never_called": never_called},
        version="v2",
        config={"run_id": UUID(int=0)},
    ):
        if event["event"] == "on_chain_end" and event["run_id"] == str(UUID(int=0)):
            times_called += 1
            assert event["data"] == {
                "output": [
                    {"inner": {"my_key": "my value there"}},
                    {"side": {"my_key": "my value there and back again"}},
                ]
            }
    assert times_called == 1


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_nested_graph_interrupts_parallel(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class InnerState(TypedDict):
        my_key: Annotated[str, operator.add]
        my_other_key: str

    async def inner_1(state: InnerState):
        await asyncio.sleep(0.1)
        return {"my_key": "got here", "my_other_key": state["my_key"]}

    async def inner_2(state: InnerState):
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

    async def outer_1(state: State):
        return {"my_key": " and parallel"}

    async def outer_2(state: State):
        return {"my_key": " and back again"}

    graph = StateGraph(State)
    graph.add_node(
        "inner",
        inner.compile(interrupt_before=["inner_2"]),
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
    assert await app.ainvoke({"my_key": ""}, config, debug=True) == {
        "my_key": "",
    }

    assert await app.ainvoke(None, config, debug=True) == {
        "my_key": "got here and there and parallel and back again",
    }

    # below combo of assertions is asserting two things
    # - outer_1 finishes before inner interrupts (because we see its output in stream, which only happens after node finishes)
    # - the writes of outer are persisted in 1st call and used in 2nd call, ie outer isn't called again (because we dont see outer_1 output again in 2nd stream)
    # test stream updates w/ nested interrupt
    config = {"configurable": {"thread_id": "2"}}
    assert [c async for c in app.astream({"my_key": ""}, config, subgraphs=True)] == [
        # we got to parallel node first
        ((), {"outer_1": {"my_key": " and parallel"}}),
        ((AnyStr("inner:"),), {"inner_1": {"my_key": "got here", "my_other_key": ""}}),
    ]
    assert [c async for c in app.astream(None, config)] == [
        {"outer_1": {"my_key": " and parallel"}, "__metadata__": {"cached": True}},
        {"inner": {"my_key": "got here and there"}},
        {"outer_2": {"my_key": " and back again"}},
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [
        c async for c in app.astream({"my_key": ""}, config, stream_mode="values")
    ] == [
        {"my_key": ""},
    ]
    assert [c async for c in app.astream(None, config, stream_mode="values")] == [
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # # test interrupts BEFORE the parallel node
    app = graph.compile(checkpointer=checkpointer, interrupt_before=["outer_1"])
    config = {"configurable": {"thread_id": "4"}}
    assert [
        c async for c in app.astream({"my_key": ""}, config, stream_mode="values")
    ] == [
        {"my_key": ""},
    ]
    # while we're waiting for the node w/ interrupt inside to finish
    assert [c async for c in app.astream(None, config, stream_mode="values")] == []
    assert [c async for c in app.astream(None, config, stream_mode="values")] == [
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # test interrupts AFTER the parallel node
    app = graph.compile(checkpointer=checkpointer, interrupt_after=["outer_1"])
    config = {"configurable": {"thread_id": "5"}}
    assert [
        c async for c in app.astream({"my_key": ""}, config, stream_mode="values")
    ] == [
        {"my_key": ""},
    ]
    assert [c async for c in app.astream(None, config, stream_mode="values")] == [
        {"my_key": "got here and there and parallel"},
    ]
    assert [c async for c in app.astream(None, config, stream_mode="values")] == [
        {"my_key": "got here and there and parallel and back again"},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_doubly_nested_graph_interrupts(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        my_key: str

    class ChildState(TypedDict):
        my_key: str

    class GrandChildState(TypedDict):
        my_key: str

    async def grandchild_1(state: ChildState):
        return {"my_key": state["my_key"] + " here"}

    async def grandchild_2(state: ChildState):
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

    async def parent_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    async def parent_2(state: State):
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
    assert await app.ainvoke({"my_key": "my value"}, config, debug=True) == {
        "my_key": "hi my value",
    }

    assert await app.ainvoke(None, config, debug=True) == {
        "my_key": "hi my value here and there and back again",
    }

    # test stream updates w/ nested interrupt
    config = {"configurable": {"thread_id": "2"}}
    assert [c async for c in app.astream({"my_key": "my value"}, config)] == [
        {"parent_1": {"my_key": "hi my value"}},
    ]
    assert [c async for c in app.astream(None, config)] == [
        {"child": {"my_key": "hi my value here and there"}},
        {"parent_2": {"my_key": "hi my value here and there and back again"}},
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [
        c
        async for c in app.astream({"my_key": "my value"}, config, stream_mode="values")
    ] == [
        {
            "my_key": "my value",
        },
        {
            "my_key": "hi my value",
        },
    ]
    assert [c async for c in app.astream(None, config, stream_mode="values")] == [
        {
            "my_key": "hi my value here and there",
        },
        {
            "my_key": "hi my value here and there and back again",
        },
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_nested_graph_state(
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
        other_parent_key: str

    def outer_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    def outer_2(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("outer_1", outer_1)
    graph.add_node(
        "inner",
        inner.compile(interrupt_before=["inner_2"]),
    )
    graph.add_node("outer_2", outer_2)
    graph.set_entry_point("outer_1")
    graph.add_edge("outer_1", "inner")
    graph.add_edge("inner", "outer_2")
    graph.set_finish_point("outer_2")

    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    await app.ainvoke({"my_key": "my value"}, config, debug=True)
    # test state w/ nested subgraph state (right after interrupt)
    # first get_state without subgraph state
    assert await app.aget_state(config) == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "inner",
                state={"configurable": {"thread_id": "1", "checkpoint_ns": AnyStr()}},
            ),
        ),
        next=("inner",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
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
    )
    # now, get_state with subgraphs state
    assert await app.aget_state(config, subgraphs=True) == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "inner",
                state=StateSnapshot(
                    values={
                        "my_key": "hi my value here",
                        "my_other_key": "hi my value",
                    },
                    tasks=(
                        PregelTask(
                            AnyStr(),
                            name="inner_2",
                            error=None,
                        ),
                    ),
                    next=("inner_2",),
                    config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("inner:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {"": AnyStr(), AnyStr("child:"): AnyStr()}
                            ),
                        }
                    },
                    metadata={
                        "parents": {
                            "": AnyStr(),
                        },
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
                            "checkpoint_ns": AnyStr("inner:"),
                            "checkpoint_id": AnyStr(),
                        }
                    },
                ),
            ),
        ),
        next=("inner",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
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
    )
    # get_state_history returns outer graph checkpoints
    history = [c async for c in app.aget_state_history(config)]
    assert history == [
        StateSnapshot(
            values={"my_key": "hi my value"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "inner",
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("inner:"),
                        }
                    },
                ),
            ),
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
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
            tasks=(PregelTask(AnyStr(), "outer_1"),),
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": None,
                "step": 0,
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
            values={},
            tasks=(PregelTask(AnyStr(), "__start__"),),
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    # get_state_history for a subgraph returns its checkpoints
    child_history = [c async for c in app.aget_state_history(history[0].tasks[0].state)]
    assert child_history == [
        StateSnapshot(
            values={"my_key": "hi my value here", "my_other_key": "hi my value"},
            next=("inner_2",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("inner:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("inner:"): AnyStr()}
                    ),
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
                "parents": {"": AnyStr()},
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("inner:"),
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(PregelTask(id=AnyStr(), name="inner_2"),),
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("inner:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("inner:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": None,
                "step": 0,
                "parents": {"": AnyStr()},
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("inner:"),
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(PregelTask(id=AnyStr(), name="inner_1"),),
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("inner:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("inner:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "hi my value", "other_parent_key": None},
                "step": -1,
                "parents": {"": AnyStr()},
            },
            created_at=AnyStr(),
            parent_config=None,
            tasks=(PregelTask(id=AnyStr(), name="__start__"),),
        ),
    ]

    # resume
    await app.ainvoke(None, config, debug=True)
    # test state w/ nested subgraph state (after resuming from interrupt)
    assert await app.aget_state(config) == StateSnapshot(
        values={"my_key": "hi my value here and there and back again"},
        tasks=(),
        next=(),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
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
    )
    # test full history at the end
    actual_history = [c async for c in app.aget_state_history(config)]
    expected_history = [
        StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
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
            tasks=(PregelTask(AnyStr(), "outer_2"),),
            next=("outer_2",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
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
            tasks=(
                PregelTask(
                    AnyStr(),
                    "inner",
                    state={
                        "configurable": {"thread_id": "1", "checkpoint_ns": AnyStr()}
                    },
                ),
            ),
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
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
            tasks=(PregelTask(AnyStr(), "outer_1"),),
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": None,
                "step": 0,
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
            values={},
            tasks=(PregelTask(AnyStr(), "__start__"),),
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "writes": {"my_key": "my value"},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    assert actual_history == expected_history
    # test looking up parent state by checkpoint ID
    for actual_snapshot, expected_snapshot in zip(actual_history, expected_history):
        assert await app.aget_state(actual_snapshot.config) == expected_snapshot


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_doubly_nested_graph_state(
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
    assert [
        c async for c in app.astream({"my_key": "my value"}, config, subgraphs=True)
    ] == [
        ((), {"parent_1": {"my_key": "hi my value"}}),
        (
            (AnyStr("child:"), AnyStr("child_1:")),
            {"grandchild_1": {"my_key": "hi my value here"}},
        ),
    ]
    # get state without subgraphs
    outer_state = await app.aget_state(config)
    assert outer_state == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "child",
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("child"),
                    }
                },
            ),
        ),
        next=("child",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
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
    )
    child_state = await app.aget_state(outer_state.tasks[0].state)
    assert (
        child_state.tasks[0]
        == StateSnapshot(
            values={"my_key": "hi my value"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "child_1",
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr(),
                        }
                    },
                ),
            ),
            next=("child_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "parents": {"": AnyStr()},
                "source": "loop",
                "writes": None,
                "step": 0,
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                }
            },
        ).tasks[0]
    )
    grandchild_state = await app.aget_state(child_state.tasks[0].state)
    assert grandchild_state == StateSnapshot(
        values={"my_key": "hi my value here"},
        tasks=(
            PregelTask(
                AnyStr(),
                "grandchild_2",
            ),
        ),
        next=("grandchild_2",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": AnyStr(),
                "checkpoint_id": AnyStr(),
                "checkpoint_map": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                        AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                    }
                ),
            }
        },
        metadata={
            "parents": AnyDict(
                {
                    "": AnyStr(),
                    AnyStr("child:"): AnyStr(),
                }
            ),
            "source": "loop",
            "writes": {"grandchild_1": {"my_key": "hi my value here"}},
            "step": 1,
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": AnyStr(),
                "checkpoint_id": AnyStr(),
            }
        },
    )
    # get state with subgraphs
    assert await app.aget_state(config, subgraphs=True) == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "child",
                state=StateSnapshot(
                    values={"my_key": "hi my value"},
                    tasks=(
                        PregelTask(
                            AnyStr(),
                            "child_1",
                            state=StateSnapshot(
                                values={"my_key": "hi my value here"},
                                tasks=(
                                    PregelTask(
                                        AnyStr(),
                                        "grandchild_2",
                                    ),
                                ),
                                next=("grandchild_2",),
                                config={
                                    "configurable": {
                                        "thread_id": "1",
                                        "checkpoint_ns": AnyStr(),
                                        "checkpoint_id": AnyStr(),
                                        "checkpoint_map": AnyDict(
                                            {
                                                "": AnyStr(),
                                                AnyStr("child:"): AnyStr(),
                                                AnyStr(
                                                    re.compile(r"child:.+|child1:")
                                                ): AnyStr(),
                                            }
                                        ),
                                    }
                                },
                                metadata={
                                    "parents": AnyDict(
                                        {
                                            "": AnyStr(),
                                            AnyStr("child:"): AnyStr(),
                                        }
                                    ),
                                    "source": "loop",
                                    "writes": {
                                        "grandchild_1": {"my_key": "hi my value here"}
                                    },
                                    "step": 1,
                                },
                                created_at=AnyStr(),
                                parent_config={
                                    "configurable": {
                                        "thread_id": "1",
                                        "checkpoint_ns": AnyStr(),
                                        "checkpoint_id": AnyStr(),
                                    }
                                },
                            ),
                        ),
                    ),
                    next=("child_1",),
                    config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("child:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {"": AnyStr(), AnyStr("child:"): AnyStr()}
                            ),
                        }
                    },
                    metadata={
                        "parents": {"": AnyStr()},
                        "source": "loop",
                        "writes": None,
                        "step": 0,
                    },
                    created_at=AnyStr(),
                    parent_config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("child:"),
                            "checkpoint_id": AnyStr(),
                        }
                    },
                ),
            ),
        ),
        next=("child",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
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
    )
    # resume
    assert [c async for c in app.astream(None, config, subgraphs=True)] == [
        (
            (AnyStr("child:"), AnyStr("child_1:")),
            {"grandchild_2": {"my_key": "hi my value here and there"}},
        ),
        ((AnyStr("child:"),), {"child_1": {"my_key": "hi my value here and there"}}),
        ((), {"child": {"my_key": "hi my value here and there"}}),
        ((), {"parent_2": {"my_key": "hi my value here and there and back again"}}),
    ]
    # get state with and without subgraphs
    assert (
        await app.aget_state(config)
        == await app.aget_state(config, subgraphs=True)
        == StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
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
        )
    )
    # get outer graph history
    outer_history = [c async for c in app.aget_state_history(config)]
    assert (
        outer_history[0]
        == [
            StateSnapshot(
                values={"my_key": "hi my value here and there and back again"},
                tasks=(),
                next=(),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "parents": {},
                    "source": "loop",
                    "writes": {
                        "parent_2": {
                            "my_key": "hi my value here and there and back again"
                        }
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
                next=("parent_2",),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "parents": {},
                    "source": "loop",
                    "writes": {"child": {"my_key": "hi my value here and there"}},
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
                tasks=(PregelTask(id=AnyStr(), name="parent_2"),),
            ),
            StateSnapshot(
                values={"my_key": "hi my value"},
                tasks=(
                    PregelTask(
                        AnyStr(),
                        "child",
                        state={
                            "configurable": {
                                "thread_id": "1",
                                "checkpoint_ns": AnyStr("child"),
                            }
                        },
                    ),
                ),
                next=("child",),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "parents": {},
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
            ),
            StateSnapshot(
                values={"my_key": "my value"},
                next=("parent_1",),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "parents": {},
                    "source": "loop",
                    "writes": None,
                    "step": 0,
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                tasks=(PregelTask(id=AnyStr(), name="parent_1"),),
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
                    "parents": {},
                    "source": "input",
                    "writes": {"my_key": "my value"},
                    "step": -1,
                },
                created_at=AnyStr(),
                parent_config=None,
                tasks=(PregelTask(id=AnyStr(), name="__start__"),),
            ),
        ][0]
    )
    # get child graph history
    child_history = [
        c async for c in app.aget_state_history(outer_history[2].tasks[0].state)
    ]
    assert child_history == [
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"child_1": {"my_key": "hi my value here and there"}},
                "step": 1,
                "parents": {"": AnyStr()},
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(),
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("child_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": None,
                "step": 0,
                "parents": {"": AnyStr()},
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="child_1",
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("child:"),
                        }
                    },
                ),
            ),
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "hi my value"},
                "step": -1,
                "parents": {"": AnyStr()},
            },
            created_at=AnyStr(),
            parent_config=None,
            tasks=(PregelTask(id=AnyStr(), name="__start__"),),
        ),
    ]
    # get grandchild graph history
    grandchild_history = [
        c async for c in app.aget_state_history(child_history[1].tasks[0].state)
    ]
    assert grandchild_history == [
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"grandchild_2": {"my_key": "hi my value here and there"}},
                "step": 2,
                "parents": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                    }
                ),
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(),
        ),
        StateSnapshot(
            values={"my_key": "hi my value here"},
            next=("grandchild_2",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"grandchild_1": {"my_key": "hi my value here"}},
                "step": 1,
                "parents": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                    }
                ),
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(PregelTask(id=AnyStr(), name="grandchild_2"),),
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("grandchild_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": None,
                "step": 0,
                "parents": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                    }
                ),
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(PregelTask(id=AnyStr(), name="grandchild_1"),),
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            metadata={
                "source": "input",
                "writes": {"my_key": "hi my value"},
                "step": -1,
                "parents": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                    }
                ),
            },
            created_at=AnyStr(),
            parent_config=None,
            tasks=(PregelTask(id=AnyStr(), name="__start__"),),
        ),
    ]

    # replay grandchild checkpoint
    assert [
        c async for c in app.astream(None, grandchild_history[2].config, subgraphs=True)
    ] == [
        (
            (AnyStr("child:"), AnyStr("child_1:")),
            {
                "grandchild_1": {"my_key": "hi my value here"},
                "__metadata__": {"cached": True},
            },
        )
    ]

    # fork and replay
    fork = app.update_state(grandchild_history[2].config, None)
    assert [c for c in app.stream(None, fork, subgraphs=True)] == [
        (
            (AnyStr("child:"), AnyStr("child_1:")),
            {"grandchild_1": {"my_key": "hi my value here"}},
        )
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_send_to_nested_graphs(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class OverallState(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], operator.add]

    async def continue_to_jokes(state: OverallState):
        return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

    class JokeState(TypedDict):
        subject: str

    async def edit(state: JokeState):
        subject = state["subject"]
        return {"subject": f"{subject} - hohoho"}

    # subgraph
    subgraph = StateGraph(input=JokeState, output=OverallState)
    subgraph.add_node("edit", edit)
    subgraph.add_node(
        "generate", lambda state: {"jokes": [f"Joke about {state['subject']}"]}
    )
    subgraph.set_entry_point("edit")
    subgraph.add_edge("edit", "generate")
    subgraph.set_finish_point("generate")

    # parent graph
    builder = StateGraph(OverallState)
    builder.add_node(
        "generate_joke",
        subgraph.compile(interrupt_before=["generate"]),
    )
    builder.add_conditional_edges(START, continue_to_jokes)
    builder.add_edge("generate_joke", END)

    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    # invoke and pause at nested interrupt
    assert await graph.ainvoke({"subjects": ["cats", "dogs"]}, config=config) == {
        "subjects": ["cats", "dogs"],
        "jokes": [],
    }
    # check state
    outer_state = await graph.aget_state(config)

    assert outer_state == StateSnapshot(
        values={"subjects": ["cats", "dogs"], "jokes": []},
        tasks=(
            PregelTask(
                AnyStr(),
                "generate_joke",
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("generate_joke:"),
                    }
                },
            ),
            PregelTask(
                AnyStr(),
                "generate_joke",
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("generate_joke:"),
                    }
                },
            ),
        ),
        next=("generate_joke", "generate_joke"),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={"parents": {}, "source": "loop", "writes": None, "step": 0},
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
    )

    # update state of dogs joke graph
    await graph.aupdate_state(
        outer_state.tasks[1].state, {"subject": "turtles - hohoho"}
    )

    # continue past interrupt
    assert await graph.ainvoke(None, config=config) == {
        "subjects": ["cats", "dogs"],
        "jokes": ["Joke about cats - hohoho", "Joke about turtles - hohoho"],
    }

    actual_snapshot = await graph.aget_state(config)
    expected_snapshot = StateSnapshot(
        values={
            "subjects": ["cats", "dogs"],
            "jokes": ["Joke about cats - hohoho", "Joke about turtles - hohoho"],
        },
        tasks=(),
        next=(),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "writes": {
                "generate_joke": [
                    {"jokes": ["Joke about cats - hohoho"]},
                    {"jokes": ["Joke about turtles - hohoho"]},
                ]
            },
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
    )
    assert actual_snapshot == expected_snapshot

    # test full history
    actual_history = [c async for c in graph.aget_state_history(config)]
    expected_history = [
        StateSnapshot(
            values={
                "subjects": ["cats", "dogs"],
                "jokes": ["Joke about cats - hohoho", "Joke about turtles - hohoho"],
            },
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": {
                    "generate_joke": [
                        {"jokes": ["Joke about cats - hohoho"]},
                        {"jokes": ["Joke about turtles - hohoho"]},
                    ]
                },
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
            values={"subjects": ["cats", "dogs"], "jokes": []},
            next=("generate_joke", "generate_joke"),
            tasks=(
                PregelTask(
                    AnyStr(),
                    "generate_joke",
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("generate_joke:"),
                        }
                    },
                ),
                PregelTask(
                    AnyStr(),
                    "generate_joke",
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("generate_joke:"),
                        }
                    },
                ),
            ),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={"parents": {}, "source": "loop", "writes": None, "step": 0},
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
            values={"jokes": []},
            tasks=(PregelTask(AnyStr(), "__start__"),),
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "writes": {"subjects": ["cats", "dogs"]},
                "step": -1,
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    assert actual_history == expected_history


async def test_checkpoint_metadata() -> None:
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

    def agent(state: BaseState, config: RunnableConfig) -> BaseState:
        formatted = prompt.invoke(state)
        response = model.invoke(formatted)
        return {"messages": response}

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
    await app.ainvoke(
        {"messages": ["what is weather in sf"]},
        {
            "configurable": {
                "thread_id": "1",
                "test_config_1": "foo",
                "test_config_2": "bar",
            },
        },
    )

    config = {"configurable": {"thread_id": "1"}}

    # assert that checkpoint metadata contains the run's configurable fields
    chkpnt_metadata_1 = (await checkpointer_1.aget_tuple(config)).metadata
    assert chkpnt_metadata_1["thread_id"] == "1"
    assert chkpnt_metadata_1["test_config_1"] == "foo"
    assert chkpnt_metadata_1["test_config_2"] == "bar"

    # Verify that all checkpoint metadata have the expected keys. This check
    # is needed because a run may have an arbitrary number of steps depending
    # on how the graph is constructed.
    chkpnt_tuples_1 = checkpointer_1.alist(config)
    async for chkpnt_tuple in chkpnt_tuples_1:
        assert chkpnt_tuple.metadata["thread_id"] == "1"
        assert chkpnt_tuple.metadata["test_config_1"] == "foo"
        assert chkpnt_tuple.metadata["test_config_2"] == "bar"

    # invoke graph, but interrupt before tool call
    await app_w_interrupt.ainvoke(
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
    chkpnt_metadata_2 = (await checkpointer_2.aget_tuple(config)).metadata
    assert chkpnt_metadata_2["thread_id"] == "2"
    assert chkpnt_metadata_2["test_config_3"] == "foo"
    assert chkpnt_metadata_2["test_config_4"] == "bar"

    # resume graph execution
    await app_w_interrupt.ainvoke(
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
    chkpnt_metadata_3 = (await checkpointer_2.aget_tuple(config)).metadata
    assert chkpnt_metadata_3["thread_id"] == "2"
    assert chkpnt_metadata_3["test_config_3"] == "foo"
    assert chkpnt_metadata_3["test_config_4"] == "bar"

    # Verify that all checkpoint metadata have the expected keys. This check
    # is needed because a run may have an arbitrary number of steps depending
    # on how the graph is constructed.
    chkpnt_tuples_2 = checkpointer_2.alist(config)
    async for chkpnt_tuple in chkpnt_tuples_2:
        assert chkpnt_tuple.metadata["thread_id"] == "2"
        assert chkpnt_tuple.metadata["test_config_3"] == "foo"
        assert chkpnt_tuple.metadata["test_config_4"] == "bar"
