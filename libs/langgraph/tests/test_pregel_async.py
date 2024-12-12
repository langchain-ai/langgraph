import asyncio
import logging
import operator
import random
import sys
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from time import perf_counter
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
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.utils.aiter import aclosing
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion

from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.context import Context
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import (
    CONFIG_KEY_NODE_FINISHED,
    ERROR,
    FF_SEND_V2,
    PULL,
    START,
)
from langgraph.errors import InvalidUpdateError, MultipleSubgraphsError, NodeInterrupt
from langgraph.func import entrypoint, task
from langgraph.graph import END, Graph, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.pregel import Channel, GraphRecursionError, Pregel, StateSnapshot
from langgraph.pregel.retry import RetryPolicy
from langgraph.store.base import BaseStore
from langgraph.types import (
    Command,
    Interrupt,
    PregelTask,
    Send,
    StreamWriter,
    interrupt,
)
from tests.any_str import AnyStr, AnyVersion, FloatBetween, UnsortedSequence
from tests.conftest import (
    ALL_CHECKPOINTERS_ASYNC,
    ALL_CHECKPOINTERS_ASYNC_PLUS_NONE,
    ALL_STORES_ASYNC,
    SHOULD_CHECK_SNAPSHOTS,
    awith_checkpointer,
    awith_store,
)
from tests.fake_tracer import FakeTracer
from tests.memory_assert import (
    MemorySaverAssertCheckpointMetadata,
    MemorySaverNoPending,
)
from tests.messages import (
    _AnyIdAIMessage,
    _AnyIdHumanMessage,
    _AnyIdToolMessage,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.anyio


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


async def test_node_cancellation_on_other_node_exception_two() -> None:
    async def awhile(input: Any) -> None:
        await asyncio.sleep(1)

    async def iambad(input: Any) -> None:
        raise ValueError("I am bad")

    builder = Graph()
    builder.add_node("agent", awhile)
    builder.add_node("bad", iambad)
    builder.set_conditional_entry_point(lambda _: ["agent", "bad"], then=END)

    graph = builder.compile()

    with pytest.raises(ValueError, match="I am bad"):
        # This will raise ValueError, not CancelledError
        await graph.ainvoke(1)


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_dynamic_interrupt(checkpointer_name: str) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_node_count = 0

    async def tool_two_node(s: State) -> State:
        nonlocal tool_two_node_count
        tool_two_node_count += 1
        if s["market"] == "DE":
            answer = interrupt("Just because...")
        else:
            answer = " all good"
        return {"my_key": answer}

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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        tool_two = tool_two_graph.compile(checkpointer=checkpointer)

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            await tool_two.ainvoke({"my_key": "value", "market": "DE"})

        # flow: interrupt -> resume with answer
        thread2 = {"configurable": {"thread_id": "2"}}
        # stop when about to enter node
        assert [
            c
            async for c in tool_two.astream(
                {"my_key": "value ⛰️", "market": "DE"}, thread2
            )
        ] == [
            {
                "__interrupt__": (
                    Interrupt(
                        value="Just because...",
                        resumable=True,
                        ns=[AnyStr("tool_two:")],
                    ),
                )
            },
        ]
        # resume with answer
        assert [
            c async for c in tool_two.astream(Command(resume=" my answer"), thread2)
        ] == [
            {"tool_two": {"my_key": " my answer"}},
        ]

        # flow: interrupt -> clear
        thread1 = {"configurable": {"thread_id": "1"}}
        # stop when about to enter node
        assert [
            c
            async for c in tool_two.astream(
                {"my_key": "value ⛰️", "market": "DE"}, thread1
            )
        ] == [
            {
                "__interrupt__": (
                    Interrupt(
                        value="Just because...",
                        resumable=True,
                        ns=[AnyStr("tool_two:")],
                    ),
                )
            },
        ]
        assert [c.metadata async for c in tool_two.checkpointer.alist(thread1)] == [
            {
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": None,
                "thread_id": "1",
            },
            {
                "parents": {},
                "source": "input",
                "step": -1,
                "writes": {"__start__": {"my_key": "value ⛰️", "market": "DE"}},
                "thread_id": "1",
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
                    (PULL, "tool_two"),
                    interrupts=(
                        Interrupt(
                            value="Just because...",
                            resumable=True,
                            ns=[AnyStr("tool_two:")],
                        ),
                    ),
                ),
            ),
            config=tup.config,
            created_at=tup.checkpoint["ts"],
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": None,
                "thread_id": "1",
            },
            parent_config=[
                c async for c in tool_two.checkpointer.alist(thread1, limit=2)
            ][-1].config,
        )

        # clear the interrupt and next tasks
        await tool_two.aupdate_state(thread1, None, as_node=END)
        # interrupt is cleared, as well as the next tasks
        tup = await tool_two.checkpointer.aget_tuple(thread1)
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value ⛰️", "market": "DE"},
            next=(),
            tasks=(),
            config=tup.config,
            created_at=tup.checkpoint["ts"],
            metadata={
                "parents": {},
                "source": "update",
                "step": 1,
                "writes": {},
                "thread_id": "1",
            },
            parent_config=[
                c async for c in tool_two.checkpointer.alist(thread1, limit=2)
            ][-1].config,
        )


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_dynamic_interrupt_subgraph(checkpointer_name: str) -> None:
    class SubgraphState(TypedDict):
        my_key: str
        market: str

    tool_two_node_count = 0

    def tool_two_node(s: SubgraphState) -> SubgraphState:
        nonlocal tool_two_node_count
        tool_two_node_count += 1
        if s["market"] == "DE":
            answer = interrupt("Just because...")
        else:
            answer = " all good"
        return {"my_key": answer}

    subgraph = StateGraph(SubgraphState)
    subgraph.add_node("do", tool_two_node, retry=RetryPolicy())
    subgraph.add_edge(START, "do")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two", subgraph.compile())
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        tool_two = tool_two_graph.compile(checkpointer=checkpointer)

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            await tool_two.ainvoke({"my_key": "value", "market": "DE"})

        # flow: interrupt -> resume with answer
        thread2 = {"configurable": {"thread_id": "2"}}
        # stop when about to enter node
        assert [
            c
            async for c in tool_two.astream(
                {"my_key": "value ⛰️", "market": "DE"}, thread2
            )
        ] == [
            {
                "__interrupt__": (
                    Interrupt(
                        value="Just because...",
                        resumable=True,
                        ns=[AnyStr("tool_two:"), AnyStr("do:")],
                    ),
                )
            },
        ]
        # resume with answer
        assert [
            c async for c in tool_two.astream(Command(resume=" my answer"), thread2)
        ] == [
            {"tool_two": {"my_key": " my answer", "market": "DE"}},
        ]

        # flow: interrupt -> clear
        thread1 = {"configurable": {"thread_id": "1"}}
        thread1root = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
        # stop when about to enter node
        assert [
            c
            async for c in tool_two.astream(
                {"my_key": "value ⛰️", "market": "DE"}, thread1
            )
        ] == [
            {
                "__interrupt__": (
                    Interrupt(
                        value="Just because...",
                        resumable=True,
                        ns=[AnyStr("tool_two:"), AnyStr("do:")],
                    ),
                )
            },
        ]
        assert [c.metadata async for c in tool_two.checkpointer.alist(thread1root)] == [
            {
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": None,
                "thread_id": "1",
            },
            {
                "parents": {},
                "source": "input",
                "step": -1,
                "writes": {"__start__": {"my_key": "value ⛰️", "market": "DE"}},
                "thread_id": "1",
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
                    (PULL, "tool_two"),
                    interrupts=(
                        Interrupt(
                            value="Just because...",
                            resumable=True,
                            ns=[AnyStr("tool_two:"), AnyStr("do:")],
                        ),
                    ),
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("tool_two:"),
                        }
                    },
                ),
            ),
            config=tup.config,
            created_at=tup.checkpoint["ts"],
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": None,
                "thread_id": "1",
            },
            parent_config=[
                c async for c in tool_two.checkpointer.alist(thread1root, limit=2)
            ][-1].config,
        )

        # clear the interrupt and next tasks
        await tool_two.aupdate_state(thread1, None, as_node=END)
        # interrupt is cleared, as well as the next tasks
        tup = await tool_two.checkpointer.aget_tuple(thread1)
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value ⛰️", "market": "DE"},
            next=(),
            tasks=(),
            config=tup.config,
            created_at=tup.checkpoint["ts"],
            metadata={
                "parents": {},
                "source": "update",
                "step": 1,
                "writes": {},
                "thread_id": "1",
            },
            parent_config=[
                c async for c in tool_two.checkpointer.alist(thread1root, limit=2)
            ][-1].config,
        )


@pytest.mark.skipif(not FF_SEND_V2, reason="send v2 is not enabled")
@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_copy_checkpoint(checkpointer_name: str) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    def tool_one(s: State) -> State:
        return {"my_key": " one"}

    tool_two_node_count = 0

    def tool_two_node(s: State) -> State:
        nonlocal tool_two_node_count
        tool_two_node_count += 1
        if s["market"] == "DE":
            answer = interrupt("Just because...")
        else:
            answer = " all good"
        return {"my_key": answer}

    def start(state: State) -> list[Union[Send, str]]:
        return ["tool_two", Send("tool_one", state)]

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two", tool_two_node, retry=RetryPolicy())
    tool_two_graph.add_node("tool_one", tool_one)
    tool_two_graph.set_conditional_entry_point(start)
    tool_two = tool_two_graph.compile()

    tracer = FakeTracer()
    assert await tool_two.ainvoke(
        {"my_key": "value", "market": "DE"}, {"callbacks": [tracer]}
    ) == {
        "my_key": "value one",
        "market": "DE",
    }
    assert tool_two_node_count == 1, "interrupts aren't retried"
    assert len(tracer.runs) == 1
    run = tracer.runs[0]
    assert run.end_time is not None
    assert run.error is None
    assert run.outputs == {"market": "DE", "my_key": "value one"}

    assert await tool_two.ainvoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value one all good",
        "market": "US",
    }

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        tool_two = tool_two_graph.compile(checkpointer=checkpointer)

        # missing thread_id
        with pytest.raises(ValueError, match="thread_id"):
            await tool_two.ainvoke({"my_key": "value", "market": "DE"})

        # flow: interrupt -> resume with answer
        thread2 = {"configurable": {"thread_id": "2"}}
        # stop when about to enter node
        assert [
            c
            async for c in tool_two.astream(
                {"my_key": "value ⛰️", "market": "DE"}, thread2
            )
        ] == [
            {
                "tool_one": {"my_key": " one"},
            },
            {
                "__interrupt__": (
                    Interrupt(
                        value="Just because...",
                        resumable=True,
                        ns=[AnyStr("tool_two:")],
                    ),
                )
            },
        ]
        # resume with answer
        assert [
            c async for c in tool_two.astream(Command(resume=" my answer"), thread2)
        ] == [
            {"tool_two": {"my_key": " my answer"}},
        ]

        # flow: interrupt -> clear tasks
        thread1 = {"configurable": {"thread_id": "1"}}
        # stop when about to enter node
        assert await tool_two.ainvoke(
            {"my_key": "value ⛰️", "market": "DE"}, thread1
        ) == {
            "my_key": "value ⛰️ one",
            "market": "DE",
        }
        assert [c.metadata async for c in tool_two.checkpointer.alist(thread1)] == [
            {
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": {"tool_one": {"my_key": " one"}},
                "thread_id": "1",
            },
            {
                "parents": {},
                "source": "input",
                "step": -1,
                "writes": {"__start__": {"my_key": "value ⛰️", "market": "DE"}},
                "thread_id": "1",
            },
        ]
        tup = await tool_two.checkpointer.aget_tuple(thread1)
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value ⛰️ one", "market": "DE"},
            next=("tool_two",),
            tasks=(
                PregelTask(
                    AnyStr(),
                    "tool_two",
                    (PULL, "tool_two"),
                    interrupts=(
                        Interrupt(
                            value="Just because...",
                            resumable=True,
                            ns=[AnyStr("tool_two:")],
                        ),
                    ),
                ),
            ),
            config=tup.config,
            created_at=tup.checkpoint["ts"],
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": {"tool_one": {"my_key": " one"}},
                "thread_id": "1",
            },
            parent_config=[
                c async for c in tool_two.checkpointer.alist(thread1, limit=2)
            ][-1].config,
        )
        # clear the interrupt and next tasks
        await tool_two.aupdate_state(thread1, None)
        # interrupt is cleared, next task is kept
        tup = await tool_two.checkpointer.aget_tuple(thread1)
        assert await tool_two.aget_state(thread1) == StateSnapshot(
            values={"my_key": "value ⛰️ one", "market": "DE"},
            next=("tool_two",),
            tasks=(
                PregelTask(
                    AnyStr(),
                    "tool_two",
                    (PULL, "tool_two"),
                    interrupts=(),
                ),
            ),
            config=tup.config,
            created_at=tup.checkpoint["ts"],
            metadata={
                "parents": {},
                "source": "update",
                "step": 1,
                "writes": {},
                "thread_id": "1",
            },
            parent_config=[
                c async for c in tool_two.checkpointer.alist(thread1, limit=2)
            ][-1].config,
        )


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_node_not_cancelled_on_other_node_interrupted(
    checkpointer_name: str,
) -> None:
    class State(TypedDict):
        hello: Annotated[str, operator.add]

    awhiles = 0
    inner_task_cancelled = False

    async def awhile(input: State) -> None:
        nonlocal awhiles

        awhiles += 1
        try:
            await asyncio.sleep(1)
            return {"hello": " again"}
        except asyncio.CancelledError:
            nonlocal inner_task_cancelled
            inner_task_cancelled = True
            raise

    async def iambad(input: State) -> None:
        return {"hello": interrupt("I am bad")}

    builder = StateGraph(State)
    builder.add_node("agent", awhile)
    builder.add_node("bad", iambad)
    builder.set_conditional_entry_point(lambda _: ["agent", "bad"], then=END)

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        thread = {"configurable": {"thread_id": "1"}}

        # writes from "awhile" are applied to last chunk
        assert await graph.ainvoke({"hello": "world"}, thread) == {
            "hello": "world again"
        }

        assert not inner_task_cancelled
        assert awhiles == 1

        assert await graph.ainvoke(None, thread, debug=True) == {"hello": "world again"}

        assert not inner_task_cancelled
        assert awhiles == 1

        # resume with answer
        assert await graph.ainvoke(Command(resume=" okay"), thread) == {
            "hello": "world again okay"
        }

        assert not inner_task_cancelled
        assert awhiles == 1


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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC_PLUS_NONE)
async def test_cancel_graph_astream(checkpointer_name: str) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
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
        # but we should have applied pending writes
        if checkpointer is not None:
            state = await graph.aget_state(thread1)
            assert state is not None
            assert state.values == {"value": 3}  # 1 + 2
            assert state.next == ("aparallelwhile",)
            assert state.metadata == {
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": None,
                "thread_id": "1",
            }


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC_PLUS_NONE)
async def test_cancel_graph_astream_events_v2(checkpointer_name: Optional[str]) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
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
                    await asyncio.sleep(0.1)
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
                "thread_id": "2",
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

    builder = StateGraph(State, output=Output)
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

    if SHOULD_CHECK_SNAPSHOTS:
        assert app.input_schema.model_json_schema() == {
            "title": "LangGraphInput",
            "type": "integer",
        }
        assert app.output_schema.model_json_schema() == {
            "title": "LangGraphOutput",
            "type": "integer",
        }
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
                "id": AnyStr(),
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
                "id": AnyStr(),
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
                "triggers": ["inbox"],
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_invoke_checkpoint(mocker: MockerFixture, checkpointer_name: str) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
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
        assert await app.ainvoke(2, {"configurable": {"thread_id": "1"}}) == 2
        checkpoint = await checkpointer.aget({"configurable": {"thread_id": "1"}})
        assert checkpoint is not None
        assert checkpoint["channel_values"].get("total") == 2
        # total is now 2, so output is 2+3=5
        assert await app.ainvoke(3, {"configurable": {"thread_id": "1"}}) == 5
        assert errored_once, "errored and retried"
        checkpoint = await checkpointer.aget({"configurable": {"thread_id": "1"}})
        assert checkpoint is not None
        assert checkpoint["channel_values"].get("total") == 7
        # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
        with pytest.raises(ValueError):
            await app.ainvoke(4, {"configurable": {"thread_id": "1"}})
        # checkpoint is not updated
        checkpoint = await checkpointer.aget({"configurable": {"thread_id": "1"}})
        assert checkpoint is not None
        assert checkpoint["channel_values"].get("total") == 7
        # on a new thread, total starts out as 0, so output is 0+5=5
        assert await app.ainvoke(5, {"configurable": {"thread_id": "2"}}) == 5
        checkpoint = await checkpointer.aget({"configurable": {"thread_id": "1"}})
        assert checkpoint is not None
        assert checkpoint["channel_values"].get("total") == 7
        checkpoint = await checkpointer.aget({"configurable": {"thread_id": "2"}})
        assert checkpoint is not None
        assert checkpoint["channel_values"].get("total") == 5


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_pending_writes_resume(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
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

    one = AwhileMaker(0.1, {"value": 2})
    two = AwhileMaker(0.3, ConnectionError("I'm not good"))
    builder = StateGraph(State)
    builder.add_node("one", one)
    builder.add_node("two", two, retry=RetryPolicy(max_attempts=2))
    builder.add_edge(START, "one")
    builder.add_edge(START, "two")
    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        thread1: RunnableConfig = {"configurable": {"thread_id": "1"}}
        with pytest.raises(ConnectionError, match="I'm not good"):
            await graph.ainvoke({"value": 1}, thread1)

        # both nodes should have been called once
        assert one.calls == 1
        assert two.calls == 2

        # latest checkpoint should be before nodes "one", "two"
        # but we should have applied pending writes from "one"
        state = await graph.aget_state(thread1)
        assert state is not None
        assert state.values == {"value": 3}
        assert state.next == ("two",)
        assert state.tasks == (
            PregelTask(AnyStr(), "one", (PULL, "one"), result={"value": 2}),
            PregelTask(
                AnyStr(),
                "two",
                (PULL, "two"),
                'ConnectionError("I\'m not good")',
            ),
        )
        assert state.metadata == {
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "thread_id": "1",
        }
        # get_state with checkpoint_id should not apply any pending writes
        state = await graph.aget_state(state.config)
        assert state is not None
        assert state.values == {"value": 1}
        assert state.next == ("one", "two")
        # should contain pending write of "one"
        checkpoint = await checkpointer.aget_tuple(thread1)
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
            await graph.ainvoke(None, thread1)

        # node "one" succeeded previously, so shouldn't be called again
        assert one.calls == 1
        # node "two" should have been called once again
        assert two.calls == 4

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
                "thread_id": "1",
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": checkpoints[1].config["configurable"][
                        "checkpoint_id"
                    ],
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
                    "checkpoint_id": checkpoints[2].config["configurable"][
                        "checkpoint_id"
                    ],
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_run_from_checkpoint_id_retains_previous_writes(
    request: pytest.FixtureRequest, checkpointer_name: str, mocker: MockerFixture
) -> None:
    class MyState(TypedDict):
        myval: Annotated[int, operator.add]
        otherval: bool

    class Anode:
        def __init__(self):
            self.switch = False

        async def __call__(self, state: MyState):
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
    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        thread_id = uuid.uuid4()
        thread1 = {"configurable": {"thread_id": str(thread_id)}}

        result = await graph.ainvoke({"myval": 1}, thread1)
        assert result["myval"] == 4
        history = [c async for c in graph.aget_state_history(thread1)]

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
        second_result = await graph.ainvoke(None, second_run_config)
        assert second_result == {"myval": 5, "otherval": True}

        new_history = [
            c
            async for c in graph.aget_state_history(
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


async def test_concurrent_emit_sends() -> None:
    class Node:
        def __init__(self, name: str):
            self.name = name
            setattr(self, "__name__", name)

        async def __call__(self, state):
            return (
                [self.name]
                if isinstance(state, list)
                else ["|".join((self.name, str(state)))]
            )

    async def send_for_fun(state):
        return [Send("2", 1), Send("2", 2), "3.1"]

    async def send_for_profit(state):
        return [Send("2", 3), Send("2", 4)]

    async def route_to_three(state) -> Literal["3"]:
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
    assert await graph.ainvoke(["0"]) == (
        [
            "0",
            "1",
            "1.1",
            "2|1",
            "2|2",
            "2|3",
            "2|4",
            "3",
            "3.1",
        ]
        if FF_SEND_V2
        else [
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
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_send_sequences(checkpointer_name: str) -> None:
    class Node:
        def __init__(self, name: str):
            self.name = name
            setattr(self, "__name__", name)

        async def __call__(self, state):
            update = (
                [self.name]
                if isinstance(state, list)  # or isinstance(state, Control)
                else ["|".join((self.name, str(state)))]
            )
            if isinstance(state, Command):
                return replace(state, update=update)
            else:
                return update

    async def send_for_fun(state):
        return [
            Send("2", Command(goto=Send("2", 3))),
            Send("2", Command(goto=Send("2", 4))),
            "3.1",
        ]

    async def route_to_three(state) -> Literal["3"]:
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
    assert (
        await graph.ainvoke(["0"])
        == [
            "0",
            "1",
            "2|Command(goto=Send(node='2', arg=3))",
            "2|Command(goto=Send(node='2', arg=4))",
            "2|3",
            "2|4",
            "3",
            "3.1",
        ]
        if FF_SEND_V2
        else [
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
    )

    if not FF_SEND_V2:
        return

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer, interrupt_before=["3.1"])
        thread1 = {"configurable": {"thread_id": "1"}}
        assert await graph.ainvoke(["0"], thread1) == [
            "0",
            "1",
            "2|Command(goto=Send(node='2', arg=3))",
            "2|Command(goto=Send(node='2', arg=4))",
            "2|3",
            "2|4",
        ]
        assert await graph.ainvoke(None, thread1) == [
            "0",
            "1",
            "2|Command(goto=Send(node='2', arg=3))",
            "2|Command(goto=Send(node='2', arg=4))",
            "2|3",
            "2|4",
            "3",
            "3.1",
        ]


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_imp_task(checkpointer_name: str) -> None:
    async with awith_checkpointer(checkpointer_name) as checkpointer:
        mapper_calls = 0

        @task()
        async def mapper(input: int) -> str:
            nonlocal mapper_calls
            mapper_calls += 1
            return str(input) * 2

        @entrypoint(checkpointer=checkpointer)
        async def graph(input: list[int]) -> list[str]:
            futures = [mapper(i) for i in input]
            mapped = await asyncio.gather(*futures)
            answer = interrupt("question")
            return [m + answer for m in mapped]

        thread1 = {"configurable": {"thread_id": "1"}}
        assert [c async for c in graph.astream([0, 1], thread1)] == [
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

        assert await graph.ainvoke(Command(resume="answer"), thread1) == [
            "00answer",
            "11answer",
        ]
        assert mapper_calls == 2


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_imp_task_cancel(checkpointer_name: str) -> None:
    async with awith_checkpointer(checkpointer_name) as checkpointer:
        mapper_calls = 0
        mapper_cancels = 0

        @task()
        async def mapper(input: int) -> str:
            nonlocal mapper_calls, mapper_cancels
            mapper_calls += 1
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                mapper_cancels += 1
                raise
            return str(input) * 2

        @entrypoint(checkpointer=checkpointer)
        async def graph(input: list[int]) -> list[str]:
            futures = [mapper(i) for i in input]
            await asyncio.sleep(0.1)
            futures.pop().cancel()  # cancel one
            mapped = await asyncio.gather(*futures)
            answer = interrupt("question")
            return [m + answer for m in mapped]

        thread1 = {"configurable": {"thread_id": "1"}}
        assert [c async for c in graph.astream([0, 1], thread1)] == [
            {"mapper": "00"},
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
        assert mapper_cancels == 1

        assert await graph.ainvoke(Command(resume="answer"), thread1) == [
            "00answer",
        ]
        assert mapper_calls == 3
        assert mapper_cancels == 2


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_imp_sync_from_async(checkpointer_name: str) -> None:
    async with awith_checkpointer(checkpointer_name) as checkpointer:

        @task()
        def foo(state: dict) -> dict:
            return {"a": state["a"] + "foo", "b": "bar"}

        @task()
        def bar(state: dict) -> dict:
            return {"a": state["a"] + state["b"], "c": "bark"}

        @task()
        def baz(state: dict) -> dict:
            return {"a": state["a"] + "baz", "c": "something else"}

        @entrypoint(checkpointer=checkpointer)
        def graph(state: dict) -> dict:
            fut_foo = foo(state)
            fut_bar = bar(fut_foo.result())
            fut_baz = baz(fut_bar.result())
            return fut_baz.result()

        thread1 = {"configurable": {"thread_id": "1"}}
        assert [c async for c in graph.astream({"a": "0"}, thread1)] == [
            {"foo": {"a": "0foo", "b": "bar"}},
            {"bar": {"a": "0foobar", "c": "bark"}},
            {"baz": {"a": "0foobarbaz", "c": "something else"}},
            {"graph": {"a": "0foobarbaz", "c": "something else"}},
        ]


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_imp_stream_order(checkpointer_name: str) -> None:
    async with awith_checkpointer(checkpointer_name) as checkpointer:

        @task()
        async def foo(state: dict) -> dict:
            return {"a": state["a"] + "foo", "b": "bar"}

        @task()
        async def bar(state: dict) -> dict:
            return {"a": state["a"] + state["b"], "c": "bark"}

        @task()
        async def baz(state: dict) -> dict:
            return {"a": state["a"] + "baz", "c": "something else"}

        @entrypoint(checkpointer=checkpointer)
        async def graph(state: dict) -> dict:
            fut_foo = foo(state)
            fut_bar = bar(await fut_foo)
            fut_baz = baz(await fut_bar)
            return await fut_baz

        thread1 = {"configurable": {"thread_id": "1"}}
        assert [c async for c in graph.astream({"a": "0"}, thread1)] == [
            {"foo": {"a": "0foo", "b": "bar"}},
            {"bar": {"a": "0foobar", "c": "bark"}},
            {"baz": {"a": "0foobarbaz", "c": "something else"}},
            {"graph": {"a": "0foobarbaz", "c": "something else"}},
        ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_send_dedupe_on_resume(checkpointer_name: str) -> None:
    if not FF_SEND_V2:
        pytest.skip("Send deduplication is only available in Send V2")

    class InterruptOnce:
        ticks: int = 0

        def __call__(self, state):
            self.ticks += 1
            if self.ticks == 1:
                raise NodeInterrupt("Bahh")
            return ["|".join(("flaky", str(state)))]

    class Node:
        def __init__(self, name: str):
            self.name = name
            self.ticks = 0
            setattr(self, "__name__", name)

        def __call__(self, state):
            self.ticks += 1
            update = (
                [self.name]
                if isinstance(state, list)
                else ["|".join((self.name, str(state)))]
            )
            if isinstance(state, Command):
                return replace(state, update=update)
            else:
                return update

    def send_for_fun(state):
        return [
            Send("2", Command(goto=Send("2", 3))),
            Send("2", Command(goto=Send("flaky", 4))),
            "3.1",
        ]

    def route_to_three(state) -> Literal["3"]:
        return "3"

    builder = StateGraph(Annotated[list, operator.add])
    builder.add_node(Node("1"))
    builder.add_node(Node("2"))
    builder.add_node(Node("3"))
    builder.add_node(Node("3.1"))
    builder.add_node("flaky", InterruptOnce())
    builder.add_edge(START, "1")
    builder.add_conditional_edges("1", send_for_fun)
    builder.add_conditional_edges("2", route_to_three)

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        thread1 = {"configurable": {"thread_id": "1"}}
        assert await graph.ainvoke(["0"], thread1, debug=1) == [
            "0",
            "1",
            "2|Command(goto=Send(node='2', arg=3))",
            "2|Command(goto=Send(node='flaky', arg=4))",
            "2|3",
        ]
        assert builder.nodes["2"].runnable.func.ticks == 3
        assert builder.nodes["flaky"].runnable.func.ticks == 1
        # resume execution
        assert await graph.ainvoke(None, thread1, debug=1) == [
            "0",
            "1",
            "2|Command(goto=Send(node='2', arg=3))",
            "2|Command(goto=Send(node='flaky', arg=4))",
            "2|3",
            "flaky|4",
            "3",
            "3.1",
        ]
        # node "2" doesn't get called again, as we recover writes saved before
        assert builder.nodes["2"].runnable.func.ticks == 3
        # node "flaky" gets called again, as it was interrupted
        assert builder.nodes["flaky"].runnable.func.ticks == 2
        # check history
        history = [c async for c in graph.aget_state_history(thread1)]
        assert history == [
            StateSnapshot(
                values=[
                    "0",
                    "1",
                    "2|Command(goto=Send(node='2', arg=3))",
                    "2|Command(goto=Send(node='flaky', arg=4))",
                    "2|3",
                    "flaky|4",
                    "3",
                    "3.1",
                ],
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
                    "writes": {"3": ["3"], "3.1": ["3.1"]},
                    "thread_id": "1",
                    "step": 2,
                    "parents": {},
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                tasks=(),
            ),
            StateSnapshot(
                values=[
                    "0",
                    "1",
                    "2|Command(goto=Send(node='2', arg=3))",
                    "2|Command(goto=Send(node='flaky', arg=4))",
                    "2|3",
                    "flaky|4",
                ],
                next=("3", "3.1"),
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
                        "1": ["1"],
                        "2": [
                            ["2|Command(goto=Send(node='2', arg=3))"],
                            ["2|Command(goto=Send(node='flaky', arg=4))"],
                            ["2|3"],
                        ],
                        "flaky": ["flaky|4"],
                    },
                    "thread_id": "1",
                    "step": 1,
                    "parents": {},
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                tasks=(
                    PregelTask(
                        id=AnyStr(),
                        name="3",
                        path=("__pregel_pull", "3"),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["3"],
                    ),
                    PregelTask(
                        id=AnyStr(),
                        name="3.1",
                        path=("__pregel_pull", "3.1"),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["3.1"],
                    ),
                ),
            ),
            StateSnapshot(
                values=["0"],
                next=("1", "2", "2", "2", "flaky"),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "source": "loop",
                    "writes": None,
                    "thread_id": "1",
                    "step": 0,
                    "parents": {},
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                tasks=(
                    PregelTask(
                        id=AnyStr(),
                        name="1",
                        path=("__pregel_pull", "1"),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["1"],
                    ),
                    PregelTask(
                        id=AnyStr(),
                        name="2",
                        path=("__pregel_push", ("__pregel_pull", "1"), 2),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["2|Command(goto=Send(node='2', arg=3))"],
                    ),
                    PregelTask(
                        id=AnyStr(),
                        name="2",
                        path=("__pregel_push", ("__pregel_pull", "1"), 3),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["2|Command(goto=Send(node='flaky', arg=4))"],
                    ),
                    PregelTask(
                        id=AnyStr(),
                        name="2",
                        path=(
                            "__pregel_push",
                            ("__pregel_push", ("__pregel_pull", "1"), 2),
                            2,
                        ),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["2|3"],
                    ),
                    PregelTask(
                        id=AnyStr(),
                        name="flaky",
                        path=(
                            "__pregel_push",
                            ("__pregel_push", ("__pregel_pull", "1"), 3),
                            2,
                        ),
                        error=None,
                        interrupts=(Interrupt(value="Bahh", when="during"),),
                        state=None,
                        result=["flaky|4"],
                    ),
                ),
            ),
            StateSnapshot(
                values=[],
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
                    "writes": {"__start__": ["0"]},
                    "thread_id": "1",
                    "step": -1,
                    "parents": {},
                },
                created_at=AnyStr(),
                parent_config=None,
                tasks=(
                    PregelTask(
                        id=AnyStr(),
                        name="__start__",
                        path=("__pregel_pull", "__start__"),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["0"],
                    ),
                ),
            ),
        ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_send_react_interrupt(checkpointer_name: str) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

    ai_message = AIMessage(
        "",
        id="ai1",
        tool_calls=[ToolCall(name="foo", args={"hi": [1, 2, 3]}, id=AnyStr())],
    )

    async def agent(state):
        return {"messages": ai_message}

    def route(state):
        if isinstance(state["messages"][-1], AIMessage):
            return [
                Send(call["name"], call) for call in state["messages"][-1].tool_calls
            ]

    foo_called = 0

    async def foo(call: ToolCall):
        nonlocal foo_called
        foo_called += 1
        return {"messages": ToolMessage(str(call["args"]), tool_call_id=call["id"])}

    builder = StateGraph(MessagesState)
    builder.add_node(agent)
    builder.add_node(foo)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route)
    graph = builder.compile()

    assert await graph.ainvoke({"messages": [HumanMessage("hello")]}) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="{'hi': [1, 2, 3]}",
                tool_call_id=AnyStr(),
            ),
        ]
    }
    assert foo_called == 1

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        # simple interrupt-resume flow
        foo_called = 0
        graph = builder.compile(checkpointer=checkpointer, interrupt_before=["foo"])
        thread1 = {"configurable": {"thread_id": "1"}}
        assert await graph.ainvoke({"messages": [HumanMessage("hello")]}, thread1) == {
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        }
        assert foo_called == 0
        assert await graph.ainvoke(None, thread1) == {
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
                _AnyIdToolMessage(
                    content="{'hi': [1, 2, 3]}",
                    tool_call_id=AnyStr(),
                ),
            ]
        }
        assert foo_called == 1

        # interrupt-update-resume flow
        foo_called = 0
        graph = builder.compile(checkpointer=checkpointer, interrupt_before=["foo"])
        thread1 = {"configurable": {"thread_id": "2"}}
        assert await graph.ainvoke({"messages": [HumanMessage("hello")]}, thread1) == {
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        }
        assert foo_called == 0

        if not FF_SEND_V2:
            return

        # get state should show the pending task
        state = await graph.aget_state(thread1)
        assert state == StateSnapshot(
            values={
                "messages": [
                    _AnyIdHumanMessage(content="hello"),
                    _AnyIdAIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "foo",
                                "args": {"hi": [1, 2, 3]},
                                "id": "",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            },
            next=("foo",),
            config={
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "step": 0,
                "source": "loop",
                "writes": None,
                "parents": {},
                "thread_id": "2",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="agent",
                    path=("__pregel_pull", "agent"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result={
                        "messages": AIMessage(
                            content="",
                            additional_kwargs={},
                            response_metadata={},
                            id="ai1",
                            tool_calls=[
                                {
                                    "name": "foo",
                                    "args": {"hi": [1, 2, 3]},
                                    "id": "",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    },
                ),
                PregelTask(
                    id=AnyStr(),
                    name="foo",
                    path=("__pregel_push", ("__pregel_pull", "agent"), 2),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=None,
                ),
            ),
        )

        # remove the tool call, clearing the pending task
        await graph.aupdate_state(
            thread1, {"messages": AIMessage("Bye now", id=ai_message.id, tool_calls=[])}
        )

        # tool call no longer in pending tasks
        assert await graph.aget_state(thread1) == StateSnapshot(
            values={
                "messages": [
                    _AnyIdHumanMessage(content="hello"),
                    _AnyIdAIMessage(
                        content="Bye now",
                        tool_calls=[],
                    ),
                ]
            },
            next=(),
            config={
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "step": 1,
                "source": "update",
                "writes": {
                    "agent": {
                        "messages": _AnyIdAIMessage(
                            content="Bye now",
                            tool_calls=[],
                        )
                    }
                },
                "parents": {},
                "thread_id": "2",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(),
        )

        # tool call not executed
        assert await graph.ainvoke(None, thread1) == {
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(content="Bye now"),
            ]
        }
        assert foo_called == 0

        # interrupt-update-resume flow, creating new Send in update call
        foo_called = 0
        graph = builder.compile(checkpointer=checkpointer, interrupt_before=["foo"])
        thread1 = {"configurable": {"thread_id": "3"}}
        assert await graph.ainvoke({"messages": [HumanMessage("hello")]}, thread1) == {
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        }
        assert foo_called == 0

        # get state should show the pending task
        state = await graph.aget_state(thread1)
        assert state == StateSnapshot(
            values={
                "messages": [
                    _AnyIdHumanMessage(content="hello"),
                    _AnyIdAIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "foo",
                                "args": {"hi": [1, 2, 3]},
                                "id": "",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            },
            next=("foo",),
            config={
                "configurable": {
                    "thread_id": "3",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "step": 0,
                "source": "loop",
                "writes": None,
                "parents": {},
                "thread_id": "3",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "3",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="agent",
                    path=("__pregel_pull", "agent"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result={
                        "messages": AIMessage(
                            "",
                            id="ai1",
                            tool_calls=[
                                {
                                    "name": "foo",
                                    "args": {"hi": [1, 2, 3]},
                                    "id": "",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    },
                ),
                PregelTask(
                    id=AnyStr(),
                    name="foo",
                    path=("__pregel_push", ("__pregel_pull", "agent"), 2),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=None,
                ),
            ),
        )

        # replace the tool call, should clear previous send, create new one
        await graph.aupdate_state(
            thread1,
            {
                "messages": AIMessage(
                    "",
                    id=ai_message.id,
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [4, 5, 6]},
                            "id": "tool1",
                            "type": "tool_call",
                        }
                    ],
                )
            },
        )

        # prev tool call no longer in pending tasks, new tool call is
        assert await graph.aget_state(thread1) == StateSnapshot(
            values={
                "messages": [
                    _AnyIdHumanMessage(content="hello"),
                    _AnyIdAIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "foo",
                                "args": {"hi": [4, 5, 6]},
                                "id": "tool1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            },
            next=("foo",),
            config={
                "configurable": {
                    "thread_id": "3",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "step": 1,
                "source": "update",
                "writes": {
                    "agent": {
                        "messages": _AnyIdAIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "foo",
                                    "args": {"hi": [4, 5, 6]},
                                    "id": "tool1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    }
                },
                "parents": {},
                "thread_id": "3",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "3",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="foo",
                    path=("__pregel_push", (), 0),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=None,
                ),
            ),
        )

        # prev tool call not executed, new tool call is
        assert await graph.ainvoke(None, thread1) == {
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                AIMessage(
                    "",
                    id="ai1",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [4, 5, 6]},
                            "id": "tool1",
                            "type": "tool_call",
                        }
                    ],
                ),
                _AnyIdToolMessage(content="{'hi': [4, 5, 6]}", tool_call_id="tool1"),
            ]
        }
        assert foo_called == 1


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_send_react_interrupt_control(
    checkpointer_name: str, snapshot: SnapshotAssertion
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

    ai_message = AIMessage(
        "",
        id="ai1",
        tool_calls=[ToolCall(name="foo", args={"hi": [1, 2, 3]}, id=AnyStr())],
    )

    async def agent(state) -> Command[Literal["foo"]]:
        return Command(
            update={"messages": ai_message},
            goto=[Send(call["name"], call) for call in ai_message.tool_calls],
        )

    foo_called = 0

    async def foo(call: ToolCall):
        nonlocal foo_called
        foo_called += 1
        return {"messages": ToolMessage(str(call["args"]), tool_call_id=call["id"])}

    builder = StateGraph(MessagesState)
    builder.add_node(agent)
    builder.add_node(foo)
    builder.add_edge(START, "agent")
    graph = builder.compile()
    assert graph.get_graph().draw_mermaid() == snapshot

    assert await graph.ainvoke({"messages": [HumanMessage("hello")]}) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="{'hi': [1, 2, 3]}",
                tool_call_id=AnyStr(),
            ),
        ]
    }
    assert foo_called == 1

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        # simple interrupt-resume flow
        foo_called = 0
        graph = builder.compile(checkpointer=checkpointer, interrupt_before=["foo"])
        thread1 = {"configurable": {"thread_id": "1"}}
        assert await graph.ainvoke({"messages": [HumanMessage("hello")]}, thread1) == {
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        }
        assert foo_called == 0
        assert await graph.ainvoke(None, thread1) == {
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
                _AnyIdToolMessage(
                    content="{'hi': [1, 2, 3]}",
                    tool_call_id=AnyStr(),
                ),
            ]
        }
        assert foo_called == 1

        # interrupt-update-resume flow
        foo_called = 0
        graph = builder.compile(checkpointer=checkpointer, interrupt_before=["foo"])
        thread1 = {"configurable": {"thread_id": "2"}}
        assert await graph.ainvoke({"messages": [HumanMessage("hello")]}, thread1) == {
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        }
        assert foo_called == 0

        if not FF_SEND_V2:
            return

        # get state should show the pending task
        state = await graph.aget_state(thread1)
        assert state == StateSnapshot(
            values={
                "messages": [
                    _AnyIdHumanMessage(content="hello"),
                    _AnyIdAIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "foo",
                                "args": {"hi": [1, 2, 3]},
                                "id": "",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            },
            next=("foo",),
            config={
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "step": 0,
                "source": "loop",
                "writes": None,
                "parents": {},
                "thread_id": "2",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="agent",
                    path=("__pregel_pull", "agent"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result={
                        "messages": AIMessage(
                            content="",
                            additional_kwargs={},
                            response_metadata={},
                            id="ai1",
                            tool_calls=[
                                {
                                    "name": "foo",
                                    "args": {"hi": [1, 2, 3]},
                                    "id": "",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    },
                ),
                PregelTask(
                    id=AnyStr(),
                    name="foo",
                    path=("__pregel_push", ("__pregel_pull", "agent"), 2),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=None,
                ),
            ),
        )

        # remove the tool call, clearing the pending task
        await graph.aupdate_state(
            thread1, {"messages": AIMessage("Bye now", id=ai_message.id, tool_calls=[])}
        )

        # tool call no longer in pending tasks
        assert await graph.aget_state(thread1) == StateSnapshot(
            values={
                "messages": [
                    _AnyIdHumanMessage(content="hello"),
                    _AnyIdAIMessage(
                        content="Bye now",
                        tool_calls=[],
                    ),
                ]
            },
            next=(),
            config={
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "step": 1,
                "source": "update",
                "writes": {
                    "agent": {
                        "messages": _AnyIdAIMessage(
                            content="Bye now",
                            tool_calls=[],
                        )
                    }
                },
                "parents": {},
                "thread_id": "2",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(),
        )

        # tool call not executed
        assert await graph.ainvoke(None, thread1) == {
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(content="Bye now"),
            ]
        }
        assert foo_called == 0


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_max_concurrency(checkpointer_name: str) -> None:
    class Node:
        def __init__(self, name: str):
            self.name = name
            setattr(self, "__name__", name)
            self.currently = 0
            self.max_currently = 0

        async def __call__(self, state):
            self.currently += 1
            if self.currently > self.max_currently:
                self.max_currently = self.currently
            await asyncio.sleep(random.random() / 10)
            self.currently -= 1
            return [state]

    def one(state):
        return ["1"]

    def three(state):
        return ["3"]

    async def send_to_many(state):
        return [Send("2", idx) for idx in range(100)]

    async def route_to_three(state) -> Literal["3"]:
        return "3"

    node2 = Node("2")
    builder = StateGraph(Annotated[list, operator.add])
    builder.add_node("1", one)
    builder.add_node(node2)
    builder.add_node("3", three)
    builder.add_edge(START, "1")
    builder.add_conditional_edges("1", send_to_many)
    builder.add_conditional_edges("2", route_to_three)
    graph = builder.compile()

    assert await graph.ainvoke(["0"]) == ["0", "1", *range(100), "3"]
    assert node2.max_currently == 100
    assert node2.currently == 0
    node2.max_currently = 0

    assert await graph.ainvoke(["0"], {"max_concurrency": 10}) == [
        "0",
        "1",
        *range(100),
        "3",
    ]
    assert node2.max_currently == 10
    assert node2.currently == 0

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer, interrupt_before=["2"])
        thread1 = {"max_concurrency": 10, "configurable": {"thread_id": "1"}}

        assert await graph.ainvoke(["0"], thread1, debug=True) == ["0", "1"]
        state = await graph.aget_state(thread1)
        assert state.values == ["0", "1"]
        assert await graph.ainvoke(None, thread1) == ["0", "1", *range(100), "3"]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_max_concurrency_control(checkpointer_name: str) -> None:
    async def node1(state) -> Command[Literal["2"]]:
        return Command(update=["1"], goto=[Send("2", idx) for idx in range(100)])

    node2_currently = 0
    node2_max_currently = 0

    async def node2(state) -> Command[Literal["3"]]:
        nonlocal node2_currently, node2_max_currently
        node2_currently += 1
        if node2_currently > node2_max_currently:
            node2_max_currently = node2_currently
        await asyncio.sleep(0.1)
        node2_currently -= 1

        return Command(update=[state], goto="3")

    async def node3(state) -> Literal["3"]:
        return ["3"]

    builder = StateGraph(Annotated[list, operator.add])
    builder.add_node("1", node1)
    builder.add_node("2", node2)
    builder.add_node("3", node3)
    builder.add_edge(START, "1")
    graph = builder.compile()

    assert (
        graph.get_graph().draw_mermaid()
        == """%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
	__start__([<p>__start__</p>]):::first
	1(1)
	2(2)
	3([3]):::last
	__start__ --> 1;
	1 -.-> 2;
	2 -.-> 3;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
"""
    )

    assert await graph.ainvoke(["0"], debug=True) == ["0", "1", *range(100), "3"]
    assert node2_max_currently == 100
    assert node2_currently == 0
    node2_max_currently = 0

    assert await graph.ainvoke(["0"], {"max_concurrency": 10}) == [
        "0",
        "1",
        *range(100),
        "3",
    ]
    assert node2_max_currently == 10
    assert node2_currently == 0

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer, interrupt_before=["2"])
        thread1 = {"max_concurrency": 10, "configurable": {"thread_id": "1"}}

        assert await graph.ainvoke(["0"], thread1) == ["0", "1"]
        assert await graph.ainvoke(None, thread1) == ["0", "1", *range(100), "3"]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_invoke_checkpoint_three(
    mocker: MockerFixture, checkpointer_name: str
) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
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
        assert await app.aget_state(thread_1) == await app.aget_state(
            thread_1_next_config
        )


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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_invoke_join_then_call_other_pregel(
    mocker: MockerFixture, checkpointer_name: str
) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        # add checkpointer
        app.checkpointer = checkpointer
        # subgraph is called twice in the same node, through .map(), so raises
        with pytest.raises(MultipleSubgraphsError):
            await app.ainvoke([2, 3], {"configurable": {"thread_id": "1"}})

        # set inner graph checkpointer NeverCheckpoint
        inner_app.checkpointer = False
        # subgraph still called twice, but checkpointing for inner graph is disabled
        assert await app.ainvoke([2, 3], {"configurable": {"thread_id": "1"}}) == 27


async def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture) -> None:
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_in_one_fan_out_state_graph_waiting_edge(checkpointer_name: str) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
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
            {"__interrupt__": ()},
        ]

        assert [c async for c in app_w_interrupt.astream(None, config)] == [
            {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_in_one_fan_out_state_graph_waiting_edge_via_branch(
    snapshot: SnapshotAssertion, checkpointer_name: str
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
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
            {"__interrupt__": ()},
        ]

        assert [c async for c in app_w_interrupt.astream(None, config)] == [
            {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class(
    snapshot: SnapshotAssertion, mocker: MockerFixture, checkpointer_name: str
) -> None:
    from pydantic.v1 import BaseModel, ValidationError

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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
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
                {"__interrupt__": ()},
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
                "thread_id": "1",
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic2(
    snapshot: SnapshotAssertion, checkpointer_name: str
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

    if SHOULD_CHECK_SNAPSHOTS:
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
        assert app.get_input_schema().model_json_schema() == snapshot
        assert app.get_output_schema().model_json_schema() == snapshot

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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
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
            {"__interrupt__": ()},
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_in_one_fan_out_state_graph_waiting_edge_plus_regular(
    checkpointer_name: str,
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app_w_interrupt = workflow.compile(
            checkpointer=checkpointer,
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
            {"__interrupt__": ()},
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
async def test_stream_subgraphs_during_execution(checkpointer_name: str) -> None:
    class InnerState(TypedDict):
        my_key: Annotated[str, operator.add]
        my_other_key: str

    async def inner_1(state: InnerState):
        return {"my_key": "got here", "my_other_key": state["my_key"]}

    async def inner_2(state: InnerState):
        await asyncio.sleep(0.5)
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
        await asyncio.sleep(0.2)
        return {"my_key": " and parallel"}

    async def outer_2(state: State):
        return {"my_key": " and back again"}

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile())
    graph.add_node("outer_1", outer_1)
    graph.add_node("outer_2", outer_2)

    graph.add_edge(START, "inner")
    graph.add_edge(START, "outer_1")
    graph.add_edge(["inner", "outer_1"], "outer_2")
    graph.add_edge("outer_2", END)

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        start = perf_counter()
        chunks: list[tuple[float, Any]] = []
        config = {"configurable": {"thread_id": "2"}}
        async for c in app.astream({"my_key": ""}, config, subgraphs=True):
            chunks.append((round(perf_counter() - start, 1), c))
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
            (FloatBetween(0.2, 0.4), ((), {"outer_1": {"my_key": " and parallel"}})),
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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_stream_buffering_single_node(checkpointer_name: str) -> None:
    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    async def node(state: State, writer: StreamWriter):
        writer("Before sleep")
        await asyncio.sleep(0.2)
        writer("After sleep")
        return {"my_key": "got here"}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        start = perf_counter()
        chunks: list[tuple[float, Any]] = []
        config = {"configurable": {"thread_id": "2"}}
        async for c in graph.astream({"my_key": ""}, config, stream_mode="custom"):
            chunks.append((round(perf_counter() - start, 1), c))

        assert chunks == [
            (FloatBetween(0.0, 0.1), "Before sleep"),
            (FloatBetween(0.2, 0.3), "After sleep"),
        ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_nested_graph_interrupts_parallel(checkpointer_name: str) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        # test invoke w/ nested interrupt
        config = {"configurable": {"thread_id": "1"}}
        assert await app.ainvoke({"my_key": ""}, config, debug=True) == {
            "my_key": " and parallel",
        }

        assert await app.ainvoke(None, config, debug=True) == {
            "my_key": "got here and there and parallel and back again",
        }

        # below combo of assertions is asserting two things
        # - outer_1 finishes before inner interrupts (because we see its output in stream, which only happens after node finishes)
        # - the writes of outer are persisted in 1st call and used in 2nd call, ie outer isn't called again (because we dont see outer_1 output again in 2nd stream)
        # test stream updates w/ nested interrupt
        config = {"configurable": {"thread_id": "2"}}
        assert [
            c async for c in app.astream({"my_key": ""}, config, subgraphs=True)
        ] == [
            # we got to parallel node first
            ((), {"outer_1": {"my_key": " and parallel"}}),
            (
                (AnyStr("inner:"),),
                {"inner_1": {"my_key": "got here", "my_other_key": ""}},
            ),
            ((), {"__interrupt__": ()}),
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
            {"my_key": " and parallel"},
        ]
        assert [c async for c in app.astream(None, config, stream_mode="values")] == [
            {"my_key": ""},
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
        assert [c async for c in app.astream(None, config, stream_mode="values")] == [
            {"my_key": ""},
            {"my_key": " and parallel"},
        ]
        assert [c async for c in app.astream(None, config, stream_mode="values")] == [
            {"my_key": ""},
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
            {"my_key": " and parallel"},
        ]
        assert [c async for c in app.astream(None, config, stream_mode="values")] == [
            {"my_key": ""},
            {"my_key": "got here and there and parallel"},
        ]
        assert [c async for c in app.astream(None, config, stream_mode="values")] == [
            {"my_key": "got here and there and parallel"},
            {"my_key": "got here and there and parallel and back again"},
        ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_doubly_nested_graph_interrupts(checkpointer_name: str) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
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
        nodes: list[str] = []
        config = {
            "configurable": {"thread_id": "2", CONFIG_KEY_NODE_FINISHED: nodes.append}
        }
        assert [c async for c in app.astream({"my_key": "my value"}, config)] == [
            {"parent_1": {"my_key": "hi my value"}},
            {"__interrupt__": ()},
        ]
        assert nodes == ["parent_1", "grandchild_1"]
        assert [c async for c in app.astream(None, config)] == [
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
            c
            async for c in app.astream(
                {"my_key": "my value"}, config, stream_mode="values"
            )
        ] == [
            {"my_key": "my value"},
            {"my_key": "hi my value"},
        ]
        assert [c async for c in app.astream(None, config, stream_mode="values")] == [
            {"my_key": "hi my value"},
            {"my_key": "hi my value here and there"},
            {"my_key": "hi my value here and there and back again"},
        ]


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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
@pytest.mark.parametrize("store_name", ALL_STORES_ASYNC)
async def test_store_injected_async(checkpointer_name: str, store_name: str) -> None:
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

        async def __call__(
            self, inputs: State, config: RunnableConfig, store: BaseStore
        ):
            assert isinstance(store, BaseStore)
            await store.aput(
                namespace
                if self.i is not None
                and config["configurable"]["thread_id"] in (thread_1, thread_2)
                else (f"foo_{self.i}", "bar"),
                doc_id,
                {
                    **doc,
                    "from_thread": config["configurable"]["thread_id"],
                    "some_val": inputs["count"],
                },
            )
            return {"count": 1}

    def other_node(inputs: State, config: RunnableConfig, store: BaseStore):
        assert isinstance(store, BaseStore)
        store.put(("not", "interesting"), "key", {"val": "val"})
        item = store.get(("not", "interesting"), "key")
        assert item is not None
        assert item.value == {"val": "val"}
        return {"count": 0}

    builder = StateGraph(State)
    builder.add_node("node", Node())
    builder.add_node("other_node", other_node)
    builder.add_edge("__start__", "node")
    builder.add_edge("node", "other_node")

    N = 500
    M = 1
    if "duckdb" in store_name:
        logger.warning(
            "DuckDB store implementation has a known issue that does not"
            " support concurrent writes, so we're reducing the test scope"
        )
        N = M = 1

    for i in range(N):
        builder.add_node(f"node_{i}", Node(i))
        builder.add_edge("__start__", f"node_{i}")

    async with awith_checkpointer(checkpointer_name) as checkpointer, awith_store(
        store_name
    ) as the_store:
        graph = builder.compile(store=the_store, checkpointer=checkpointer)

        # Test batch operations with multiple threads
        results = await graph.abatch(
            [{"count": 0}] * M,
            ([{"configurable": {"thread_id": str(uuid.uuid4())}}] * (M - 1))
            + [{"configurable": {"thread_id": thread_1}}],
        )
        result = results[-1]
        assert result == {"count": N + 1}
        returned_doc = (await the_store.aget(namespace, doc_id)).value
        assert returned_doc == {**doc, "from_thread": thread_1, "some_val": 0}
        assert len((await the_store.asearch(namespace))) == 1

        # Check results after another turn of the same thread
        result = await graph.ainvoke(
            {"count": 0}, {"configurable": {"thread_id": thread_1}}
        )
        assert result == {"count": (N + 1) * 2}
        returned_doc = (await the_store.aget(namespace, doc_id)).value
        assert returned_doc == {**doc, "from_thread": thread_1, "some_val": N + 1}
        assert len((await the_store.asearch(namespace))) == 1

        # Test with a different thread
        result = await graph.ainvoke(
            {"count": 0}, {"configurable": {"thread_id": thread_2}}
        )
        assert result == {"count": N + 1}
        returned_doc = (await the_store.aget(namespace, doc_id)).value
        assert returned_doc == {
            **doc,
            "from_thread": thread_2,
            "some_val": 0,
        }  # Overwrites the whole doc
        assert (
            len((await the_store.asearch(namespace))) == 1
        )  # still overwriting the same one


async def test_debug_retry():
    class State(TypedDict):
        messages: Annotated[list[str], operator.add]

    def node(name):
        async def _node(state: State):
            return {"messages": [f"entered {name} node"]}

        return _node

    builder = StateGraph(State)
    builder.add_node("one", node("one"))
    builder.add_node("two", node("two"))
    builder.add_edge(START, "one")
    builder.add_edge("one", "two")
    builder.add_edge("two", END)

    saver = MemorySaver()

    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "1"}}
    await graph.ainvoke({"messages": []}, config=config)

    # re-run step: 1
    async for c in saver.alist(config):
        if c.metadata["step"] == 1:
            target_config = c.parent_config
            break
    assert target_config is not None

    update_config = await graph.aupdate_state(target_config, values=None)

    events = [
        c async for c in graph.astream(None, config=update_config, stream_mode="debug")
    ]

    checkpoint_events = list(
        reversed([e["payload"] for e in events if e["type"] == "checkpoint"])
    )

    checkpoint_history = {
        c.config["configurable"]["checkpoint_id"]: c
        async for c in graph.aget_state_history(config)
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


async def test_debug_subgraphs():
    class State(TypedDict):
        messages: Annotated[list[str], operator.add]

    def node(name):
        async def _node(state: State):
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

    graph = parent.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "1"}}
    events = [
        c
        async for c in graph.astream(
            {"messages": []},
            config=config,
            stream_mode="debug",
        )
    ]

    checkpoint_events = list(
        reversed([e["payload"] for e in events if e["type"] == "checkpoint"])
    )
    checkpoint_history = [c async for c in graph.aget_state_history(config)]

    assert len(checkpoint_events) == len(checkpoint_history)

    def normalize_config(config: Optional[dict]) -> Optional[dict]:
        if config is None:
            return None
        return config["configurable"]

    for stream, history in zip(checkpoint_events, checkpoint_history):
        assert stream["values"] == history.values
        assert stream["next"] == list(history.next)
        assert normalize_config(stream["config"]) == normalize_config(history.config)
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


async def test_debug_nested_subgraphs():
    from collections import defaultdict

    class State(TypedDict):
        messages: Annotated[list[str], operator.add]

    def node(name):
        async def _node(state: State):
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

    graph = grand_parent.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "1"}}
    events = [
        c
        async for c in graph.astream(
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

    history_ns = {}
    for ns in stream_ns.keys():

        async def get_history():
            history = [
                c
                async for c in graph.aget_state_history(
                    {"configurable": {"thread_id": "1", "checkpoint_ns": "|".join(ns)}}
                )
            ]
            return history[::-1]

        history_ns[ns] = await get_history()

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


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_parent_command(checkpointer_name: str) -> None:
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
    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "1"}}

        assert await graph.ainvoke(
            {"messages": [("user", "get user name")]}, config
        ) == {
            "messages": [
                _AnyIdHumanMessage(
                    content="get user name", additional_kwargs={}, response_metadata={}
                ),
            ],
            "user_name": "Meow",
        }
        assert await graph.aget_state(config) == StateSnapshot(
            values={
                "messages": [
                    _AnyIdHumanMessage(
                        content="get user name",
                        additional_kwargs={},
                        response_metadata={},
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
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(),
        )


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_interrupt_subgraph(checkpointer_name: str):
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        thread1 = {"configurable": {"thread_id": "1"}}
        # First run, interrupted at bar
        assert await graph.ainvoke({"baz": ""}, thread1)
        # Resume with answer
        assert await graph.ainvoke(Command(resume="bar"), thread1)


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_interrupt_multiple(checkpointer_name: str):
    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    async def node(s: State) -> State:
        answer = interrupt({"value": 1})
        answer2 = interrupt({"value": 2})
        return {"my_key": answer + " " + answer2}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        thread1 = {"configurable": {"thread_id": "1"}}

        assert [
            e async for e in graph.astream({"my_key": "DE", "market": "DE"}, thread1)
        ] == [
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
            async for event in graph.astream(
                Command(resume="answer 1", update={"my_key": "foofoo"}),
                thread1,
                stream_mode="updates",
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

        assert [
            event
            async for event in graph.astream(
                Command(resume="answer 2"), thread1, stream_mode="updates"
            )
        ] == [
            {"node": {"my_key": "answer 1 answer 2"}},
        ]


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_interrupt_loop(checkpointer_name: str):
    class State(TypedDict):
        age: int
        other: str

    async def ask_age(s: State):
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        thread1 = {"configurable": {"thread_id": "1"}}

        assert [e async for e in graph.astream({"other": ""}, thread1)] == [
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
            async for event in graph.astream(
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
            async for event in graph.astream(
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

        assert [
            event async for event in graph.astream(Command(resume="19"), thread1)
        ] == [
            {"node": {"age": 19}},
        ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_command_with_static_breakpoints(checkpointer_name: str) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer, interrupt_before=["node1"])
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # Start the graph and interrupt at the first node
        await graph.ainvoke({"foo": "abc"}, config)
        result = await graph.ainvoke(Command(update={"foo": "def"}), config)
        assert result == {"foo": "def|node-1|node-2"}


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_multistep_plan(checkpointer_name: str):
    from langchain_core.messages import AnyMessage

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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "1"}}

        assert await graph.ainvoke({"messages": [("human", "start")]}, config) == {
            "messages": [
                _AnyIdHumanMessage(content="start"),
                _AnyIdHumanMessage(content="step1"),
                _AnyIdHumanMessage(content="step2"),
                _AnyIdHumanMessage(content="step3"),
                _AnyIdHumanMessage(content="step4"),
            ],
            "plan": [],
        }


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_command_goto_with_static_breakpoints(checkpointer_name: str) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer, interrupt_before=["node1"])

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # Start the graph and interrupt at the first node
        await graph.ainvoke({"foo": "abc"}, config)
        result = await graph.ainvoke(Command(goto=["node2"]), config)
        assert result == {"foo": "abc|node-1|node-2|node-2"}


async def test_nested_graph_state_error_handling():
    """Test error handling when updating state in nested graphs."""

    class State(TypedDict):
        count: int

    def child_node(state: State):
        return {"count": state["count"] + 1}

    child = StateGraph(State)
    child.add_node("child", child_node)
    child.add_edge(START, "child")

    parent = StateGraph(State)
    parent.add_node("child_graph", child.compile())
    parent.add_edge(START, "child_graph")

    app = parent.compile(checkpointer=MemorySaver())

    # Test invalid state update on parent
    with pytest.raises(InvalidUpdateError):
        await app.aupdate_state(
            {"configurable": {"thread_id": "1"}}, {"invalid_key": "value"}
        )

    # Test invalid state update on child
    with pytest.raises(InvalidUpdateError):
        await app.aupdate_state(
            {"configurable": {"thread_id": "1", "checkpoint_ns": "child_graph"}},
            {"invalid_key": "value"},
        )


async def test_parallel_node_execution():
    """Test that parallel nodes execute concurrently."""

    class State(TypedDict):
        results: Annotated[list[str], operator.add]

    async def slow_node(state: State):
        await asyncio.sleep(1)
        return {"results": ["slow"]}

    async def fast_node(state: State):
        await asyncio.sleep(2)
        return {"results": ["fast"]}

    builder = StateGraph(State)
    builder.add_node("slow", slow_node)
    builder.add_node("fast", fast_node)
    builder.add_edge(START, "slow")
    builder.add_edge(START, "fast")

    graph = builder.compile()

    start = perf_counter()
    result = await graph.ainvoke({"results": []})
    duration = perf_counter() - start

    # Fast node result should be available first
    assert "fast" in result["results"][0]

    # Total duration should be less than sum of both nodes
    assert duration < 3.0


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_multiple_interrupt_state_persistence(checkpointer_name: str) -> None:
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        app = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "1"}}

        # First execution - should hit first interrupt
        await app.ainvoke({"steps": []}, config)

        # State should still be empty since node hasn't returned
        state = await app.aget_state(config)
        assert state.values == {"steps": []}

        # Resume after first interrupt - should hit second interrupt
        await app.ainvoke(Command(resume="step1"), config)

        # State should still be empty since node hasn't returned
        state = await app.aget_state(config)
        assert state.values == {"steps": []}

        # Resume after second interrupt - node should complete
        result = await app.ainvoke(Command(resume="step2"), config)

        # Now state should contain both steps since node returned
        assert result["steps"] == ["step1", "step2"]
        state = await app.aget_state(config)
        assert state.values["steps"] == ["step1", "step2"]


async def test_concurrent_execution():
    """Test concurrent execution with async nodes."""

    class State(TypedDict):
        counter: Annotated[int, operator.add]

    results = deque()

    async def slow_node(state: State):
        await asyncio.sleep(0.1)
        return {"counter": 1}

    builder = StateGraph(State)
    builder.add_node("node", slow_node)
    builder.add_edge(START, "node")
    graph = builder.compile()

    async def run_graph():
        result = await graph.ainvoke({"counter": 0})
        results.append(result)

    # Create and gather tasks
    tasks = [run_graph() for _ in range(10)]
    await asyncio.gather(*tasks)

    # Verify results are independent
    assert len(results) == 10
    for result in results:
        assert result["counter"] == 1


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_checkpoint_recovery_async(checkpointer_name: str):
    """Test recovery from checkpoints after failures with async nodes."""

    class State(TypedDict):
        steps: Annotated[list[str], operator.add]
        attempt: int  # Track number of attempts

    async def failing_node(state: State):
        # Fail on first attempt, succeed on retry
        if state["attempt"] == 1:
            raise RuntimeError("Simulated failure")
        await asyncio.sleep(0.1)  # Simulate async work
        return {"steps": ["node1"]}

    async def second_node(state: State):
        await asyncio.sleep(0.1)  # Simulate async work
        return {"steps": ["node2"]}

    builder = StateGraph(State)
    builder.add_node("node1", failing_node)
    builder.add_node("node2", second_node)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "1"}}

        # First attempt should fail
        with pytest.raises(RuntimeError):
            await graph.ainvoke({"steps": ["start"], "attempt": 1}, config)

        # Verify checkpoint state
        state = await graph.aget_state(config)
        assert state is not None
        assert state.values == {"steps": ["start"], "attempt": 1}  # input state saved
        assert state.next == ("node1",)  # Should retry failed node

        # Retry with updated attempt count
        result = await graph.ainvoke({"steps": [], "attempt": 2}, config)
        assert result == {"steps": ["start", "node1", "node2"], "attempt": 2}

        # Verify checkpoint history shows both attempts
        history = [c async for c in graph.aget_state_history(config)]
        assert len(history) == 6  # Initial + failed attempt + successful attempt

        # Verify the error was recorded in checkpoint
        failed_checkpoint = next(c for c in history if c.tasks and c.tasks[0].error)
        assert "RuntimeError('Simulated failure')" in failed_checkpoint.tasks[0].error
