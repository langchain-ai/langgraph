import asyncio
import enum
import functools
import gc
import logging
import operator
import random
import sys
import uuid
from collections import Counter, deque
from dataclasses import replace
from time import perf_counter
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
)
from uuid import UUID

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough
from langchain_core.utils.aiter import aclosing
from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion
from typing_extensions import NotRequired, TypedDict

from langgraph._internal._constants import CONFIG_KEY_NODE_FINISHED, ERROR, PULL
from langgraph._internal._queue import AsyncQueue
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.errors import (
    GraphRecursionError,
    InvalidUpdateError,
    ParentCommand,
)
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.pregel._loop import AsyncPregelLoop
from langgraph.pregel._runner import PregelRunner
from langgraph.types import (
    CachePolicy,
    Command,
    Durability,
    Interrupt,
    PregelTask,
    RetryPolicy,
    Send,
    StateSnapshot,
    StateUpdate,
    StreamWriter,
    interrupt,
)
from tests.any_str import AnyStr, AnyVersion, FloatBetween, UnsortedSequence
from tests.fake_tracer import FakeTracer
from tests.memory_assert import MemorySaverNoPending
from tests.messages import (
    _AnyIdAIMessage,
    _AnyIdAIMessageChunk,
    _AnyIdHumanMessage,
    _AnyIdToolMessage,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.anyio

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


async def test_checkpoint_errors() -> None:
    class FaultyGetCheckpointer(InMemorySaver):
        async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
            raise ValueError("Faulty get_tuple")

    class FaultyPutCheckpointer(InMemorySaver):
        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            raise ValueError("Faulty put")

    class FaultyPutWritesCheckpointer(InMemorySaver):
        async def aput_writes(
            self, config: RunnableConfig, writes: list[tuple[str, Any]], task_id: str
        ) -> RunnableConfig:
            raise ValueError("Faulty put_writes")

    class FaultyVersionCheckpointer(InMemorySaver):
        def get_next_version(self, current: int | None, channel: None) -> int:
            raise ValueError("Faulty get_next_version")

    class FaultySerializer(JsonPlusSerializer):
        def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
            raise ValueError("Faulty serializer")

    def logic(inp: str) -> str:
        return ""

    builder = StateGraph(Annotated[str, operator.add])
    builder.add_node("agent", logic)
    builder.add_edge(START, "agent")

    graph = builder.compile(checkpointer=InMemorySaver(serde=FaultySerializer()))
    with pytest.raises(ValueError, match="Faulty serializer"):
        await graph.ainvoke("", {"configurable": {"thread_id": "thread-1"}})
    with pytest.raises(ValueError, match="Faulty serializer"):
        async for _ in graph.astream("", {"configurable": {"thread_id": "thread-2"}}):
            pass
    with pytest.raises(ValueError, match="Faulty serializer"):
        async for _ in graph.astream_events(
            "", {"configurable": {"thread_id": "thread-3"}}, version="v2"
        ):
            pass

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
        await graph.ainvoke(
            "", {"configurable": {"thread_id": "thread-1"}}, durability="async"
        )
    with pytest.raises(ValueError, match="Faulty put_writes"):
        async for _ in graph.astream(
            "", {"configurable": {"thread_id": "thread-2"}}, durability="async"
        ):
            pass
    with pytest.raises(ValueError, match="Faulty put_writes"):
        async for _ in graph.astream_events(
            "",
            {"configurable": {"thread_id": "thread-3"}},
            version="v2",
            durability="async",
        ):
            pass

    def faulty_reducer(a: Any, b: Any) -> Any:
        raise ValueError("Faulty reducer")

    builder = StateGraph(Annotated[str, faulty_reducer])
    builder.add_node("agent", logic)
    builder.add_edge(START, "agent")
    graph = builder.compile(checkpointer=InMemorySaver())

    with pytest.raises(ValueError, match="Faulty reducer"):
        await graph.ainvoke("", {"configurable": {"thread_id": "thread-1"}})
    with pytest.raises(ValueError, match="Faulty reducer"):
        async for _ in graph.astream("", {"configurable": {"thread_id": "thread-2"}}):
            pass
    with pytest.raises(ValueError, match="Faulty reducer"):
        async for _ in graph.astream_events(
            "", {"configurable": {"thread_id": "thread-3"}}, version="v2"
        ):
            pass


async def test_py_async_with_cancel_behavior() -> None:
    """This test confirms that in all versions of Python we support, __aexit__
    is not cancelled when the coroutine containing the async with block is cancelled."""

    logs: list[str] = []

    class MyContextManager:
        async def __aenter__(self):
            logs.append("Entering")
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            logs.append("Starting exit")
            try:
                # Simulate some cleanup work
                await asyncio.sleep(2)
                logs.append("Cleanup completed")
            except asyncio.CancelledError:
                logs.append("Cleanup was cancelled!")
                raise
            logs.append("Exit finished")

    async def main():
        try:
            async with MyContextManager():
                logs.append("In context")
                await asyncio.sleep(1)
                logs.append("This won't print if cancelled")
        except asyncio.CancelledError:
            logs.append("Context was cancelled")
            raise

    # create task
    t = asyncio.create_task(main())
    # cancel after 0.2 seconds
    await asyncio.sleep(0.2)
    t.cancel()
    # check logs before cancellation is handled
    assert logs == [
        "Entering",
        "In context",
    ], "Cancelled before cleanup started"
    # wait for task to finish
    try:
        await t
    except asyncio.CancelledError:
        # check logs after cancellation is handled
        assert logs == [
            "Entering",
            "In context",
            "Starting exit",
            "Cleanup completed",
            "Exit finished",
            "Context was cancelled",
        ], "Cleanup started and finished after cancellation"
    else:
        assert False, "Task should be cancelled"


async def test_checkpoint_put_after_cancellation() -> None:
    logs: list[str] = []

    class LongPutCheckpointer(InMemorySaver):
        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            logs.append("checkpoint.aput.start")
            try:
                await asyncio.sleep(1)
                return await super().aput(config, checkpoint, metadata, new_versions)
            finally:
                logs.append("checkpoint.aput.end")

    inner_task_cancelled = False

    async def awhile(input: Any) -> None:
        logs.append("awhile.start")
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            nonlocal inner_task_cancelled
            inner_task_cancelled = True
            raise
        finally:
            logs.append("awhile.end")

    class State(TypedDict):
        hello: str

    builder = StateGraph(State)
    builder.add_node("agent", awhile)
    builder.set_entry_point("agent")
    builder.set_finish_point("agent")

    graph = builder.compile(checkpointer=LongPutCheckpointer())
    thread1 = {"configurable": {"thread_id": "1"}}

    # start the task
    t = asyncio.create_task(
        graph.ainvoke({"hello": "world"}, thread1, durability="exit")
    )
    # cancel after 0.2 seconds
    await asyncio.sleep(0.2)
    t.cancel()
    # check logs before cancellation is handled
    assert sorted(logs) == [
        "awhile.start",
    ], "Cancelled before checkpoint put started"
    # wait for task to finish
    try:
        await t
    except asyncio.CancelledError:
        # check logs after cancellation is handled
        assert sorted(logs) == [
            "awhile.end",
            "awhile.start",
            "checkpoint.aput.end",
            "checkpoint.aput.start",
        ], "Checkpoint put is not cancelled"
    else:
        assert False, "Task should be cancelled"


async def test_checkpoint_put_after_cancellation_stream_anext() -> None:
    logs: list[str] = []

    class LongPutCheckpointer(InMemorySaver):
        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            logs.append("checkpoint.aput.start")
            try:
                await asyncio.sleep(1)
                return await super().aput(config, checkpoint, metadata, new_versions)
            finally:
                logs.append("checkpoint.aput.end")

    inner_task_cancelled = False

    async def awhile(input: Any) -> None:
        logs.append("awhile.start")
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            nonlocal inner_task_cancelled
            inner_task_cancelled = True
            raise
        finally:
            logs.append("awhile.end")

    class State(TypedDict):
        hello: str

    builder = StateGraph(State)
    builder.add_node("agent", awhile)
    builder.set_entry_point("agent")
    builder.set_finish_point("agent")

    graph = builder.compile(checkpointer=LongPutCheckpointer())
    thread1 = {"configurable": {"thread_id": "1"}}

    # start the task
    s = graph.astream({"hello": "world"}, thread1, durability="exit")
    t = asyncio.create_task(s.__anext__())
    # cancel after 0.2 seconds
    await asyncio.sleep(0.2)
    t.cancel()
    # check logs before cancellation is handled
    assert sorted(logs) == [
        "awhile.start",
    ], "Cancelled before checkpoint put started"
    # wait for task to finish
    try:
        await t
    except asyncio.CancelledError:
        # check logs after cancellation is handled
        assert sorted(logs) == [
            "awhile.end",
            "awhile.start",
            "checkpoint.aput.end",
            "checkpoint.aput.start",
        ], "Checkpoint put is not cancelled"
    else:
        assert False, "Task should be cancelled"


async def test_checkpoint_put_after_cancellation_stream_events_anext() -> None:
    logs: list[str] = []

    class LongPutCheckpointer(InMemorySaver):
        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> RunnableConfig:
            logs.append("checkpoint.aput.start")
            try:
                await asyncio.sleep(1)
                return await super().aput(config, checkpoint, metadata, new_versions)
            finally:
                logs.append("checkpoint.aput.end")

    inner_task_cancelled = False

    async def awhile(input: Any) -> None:
        logs.append("awhile.start")
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            nonlocal inner_task_cancelled
            inner_task_cancelled = True
            raise
        finally:
            logs.append("awhile.end")

    class State(TypedDict):
        hello: str

    builder = StateGraph(State)
    builder.add_node("agent", awhile)
    builder.set_entry_point("agent")
    builder.set_finish_point("agent")

    graph = builder.compile(checkpointer=LongPutCheckpointer())
    thread1 = {"configurable": {"thread_id": "1"}}

    # start the task
    s = graph.astream_events(
        {"hello": "world"},
        thread1,
        version="v2",
        include_names=["LangGraph"],
        durability="exit",
    )
    # skip first event (happens right away)
    await s.__anext__()
    # start the task for 2nd event
    t = asyncio.create_task(s.__anext__())
    # cancel after 0.2 seconds
    await asyncio.sleep(0.2)
    t.cancel()
    # check logs before cancellation is handled
    assert logs == [
        "awhile.start",
    ], "Cancelled before checkpoint put started"
    # wait for task to finish
    try:
        await t
    except asyncio.CancelledError:
        # check logs after cancellation is handled
        assert logs == [
            "awhile.start",
            "awhile.end",
            "checkpoint.aput.start",
            "checkpoint.aput.end",
        ], "Checkpoint put is not cancelled"
    else:
        assert False, "Task should be cancelled"


async def test_node_cancellation_on_external_cancel() -> None:
    inner_task_cancelled = False

    async def awhile(input: Any) -> None:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            nonlocal inner_task_cancelled
            inner_task_cancelled = True
            raise

    class State(TypedDict):
        hello: str

    builder = StateGraph(State)
    builder.add_node("agent", awhile)
    builder.set_entry_point("agent")
    builder.set_finish_point("agent")

    graph = builder.compile()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(graph.ainvoke({"hello": "world"}), 0.5)

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

    class State(TypedDict):
        hello: str

    builder = StateGraph(State)
    builder.add_node("agent", awhile)
    builder.add_node("bad", iambad)
    builder.set_conditional_entry_point(lambda _: ["agent", "bad"])

    graph = builder.compile()

    with pytest.raises(ValueError, match="I am bad"):
        # This will raise ValueError, not TimeoutError
        await asyncio.wait_for(graph.ainvoke({"hello": "world"}), 0.5)

    assert inner_task_cancelled


async def test_node_cancellation_on_other_node_exception_two() -> None:
    async def awhile(input: Any) -> None:
        await asyncio.sleep(1)

    async def iambad(input: Any) -> None:
        raise ValueError("I am bad")

    class State(TypedDict):
        hello: str

    builder = StateGraph(State)
    builder.add_node("agent", awhile)
    builder.add_node("bad", iambad)
    builder.set_conditional_entry_point(lambda _: ["agent", "bad"])

    graph = builder.compile()

    with pytest.raises(ValueError, match="I am bad"):
        # This will raise ValueError, not CancelledError
        await graph.ainvoke({"hello": "world"})


@NEEDS_CONTEXTVARS
async def test_dynamic_interrupt(async_checkpointer: BaseCheckpointSaver) -> None:
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
    tool_two_graph.add_node("tool_two", tool_two_node, retry_policy=RetryPolicy())
    tool_two_graph.add_edge(START, "tool_two")
    tool_two = tool_two_graph.compile()

    tracer = FakeTracer()
    assert await tool_two.ainvoke(
        {"my_key": "value", "market": "DE"}, {"callbacks": [tracer]}
    ) == {
        "my_key": "value",
        "market": "DE",
        "__interrupt__": [Interrupt(value="Just because...", id=AnyStr())],
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

    tool_two = tool_two_graph.compile(checkpointer=async_checkpointer)

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        await tool_two.ainvoke({"my_key": "value", "market": "DE"})

    # flow: interrupt -> resume with answer
    thread2 = {"configurable": {"thread_id": "2"}}
    # stop when about to enter node
    assert [
        c
        async for c in tool_two.astream({"my_key": "value ⛰️", "market": "DE"}, thread2)
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="Just because...",
                    id=AnyStr(),
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
            {"my_key": "value ⛰️", "market": "DE"}, thread1, durability="exit"
        )
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="Just because...",
                    id=AnyStr(),
                ),
            )
        },
    ]
    assert [c.metadata async for c in tool_two.checkpointer.alist(thread1)] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
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
                        id=AnyStr(),
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
        },
        parent_config=None,
        interrupts=(
            Interrupt(
                value="Just because...",
                id=AnyStr(),
            ),
        ),
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
        },
        parent_config=(
            [c async for c in tool_two.checkpointer.alist(thread1, limit=2)][-1].config
        ),
        interrupts=(),
    )


@NEEDS_CONTEXTVARS
async def test_dynamic_interrupt_subgraph(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
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
    subgraph.add_node("do", tool_two_node, retry_policy=RetryPolicy())
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
        "__interrupt__": [
            Interrupt(
                value="Just because...",
                id=AnyStr(),
            )
        ],
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

    tool_two = tool_two_graph.compile(checkpointer=async_checkpointer)

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        await tool_two.ainvoke({"my_key": "value", "market": "DE"})

    # flow: interrupt -> resume with answer
    thread2 = {"configurable": {"thread_id": "2"}}
    # stop when about to enter node
    assert [
        c
        async for c in tool_two.astream({"my_key": "value ⛰️", "market": "DE"}, thread2)
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="Just because...",
                    id=AnyStr(),
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
            {"my_key": "value ⛰️", "market": "DE"}, thread1, durability="exit"
        )
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="Just because...",
                    id=AnyStr(),
                ),
            )
        },
    ]
    assert [c.metadata async for c in tool_two.checkpointer.alist(thread1root)] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
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
                        id=AnyStr(),
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
        },
        parent_config=None,
        interrupts=(
            Interrupt(
                value="Just because...",
                id=AnyStr(),
            ),
        ),
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
        },
        parent_config=(
            [c async for c in tool_two.checkpointer.alist(thread1root, limit=2)][
                -1
            ].config
        ),
        interrupts=(),
    )


@NEEDS_CONTEXTVARS
async def test_partial_pending_checkpoint(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
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

    def start(state: State) -> list[Send | str]:
        return ["tool_two", Send("tool_one", state)]

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two", tool_two_node, retry_policy=RetryPolicy())
    tool_two_graph.add_node("tool_one", tool_one)
    tool_two_graph.set_conditional_entry_point(start)
    tool_two = tool_two_graph.compile()

    tracer = FakeTracer()
    assert await tool_two.ainvoke(
        {"my_key": "value", "market": "DE"}, {"callbacks": [tracer]}, debug=True
    ) == {
        "my_key": "value one",
        "market": "DE",
        "__interrupt__": [Interrupt(value="Just because...", id=AnyStr())],
    }
    assert tool_two_node_count == 1, "interrupts aren't retried"
    assert len(tracer.runs) == 1
    run = tracer.runs[0]
    assert run.end_time is not None
    assert run.error is None
    assert run.outputs == {"market": "DE", "my_key": "value one"}

    assert await tool_two.ainvoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value all good one",
        "market": "US",
    }

    tool_two = tool_two_graph.compile(checkpointer=async_checkpointer)

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        await tool_two.ainvoke({"my_key": "value", "market": "DE"})

    # flow: interrupt -> resume with answer
    thread2 = {"configurable": {"thread_id": "2"}}
    # stop when about to enter node
    assert [
        c
        async for c in tool_two.astream({"my_key": "value ⛰️", "market": "DE"}, thread2)
    ] == UnsortedSequence(
        {
            "__interrupt__": (
                Interrupt(
                    value="Just because...",
                    id=AnyStr(),
                ),
            )
        },
        {
            "tool_one": {"my_key": " one"},
        },
    )
    # resume with answer
    assert [
        c async for c in tool_two.astream(Command(resume=" my answer"), thread2)
    ] == [
        {
            "__metadata__": {"cached": True},
            "tool_one": {"my_key": " one"},
        },
        {"tool_two": {"my_key": " my answer"}},
    ]

    # flow: interrupt -> clear tasks
    thread1 = {"configurable": {"thread_id": "1"}}
    # stop when about to enter node
    assert await tool_two.ainvoke(
        {"my_key": "value ⛰️", "market": "DE"}, thread1, durability="exit"
    ) == {
        "my_key": "value ⛰️ one",
        "market": "DE",
        "__interrupt__": [
            Interrupt(
                value="Just because...",
                id=AnyStr(),
            )
        ],
    }

    assert [c.metadata async for c in tool_two.checkpointer.alist(thread1)] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
        },
    ]

    tup = await tool_two.checkpointer.aget_tuple(thread1)
    assert await tool_two.aget_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️ one", "market": "DE"},
        next=("tool_two",),
        tasks=(
            PregelTask(
                AnyStr(),
                name="tool_one",
                path=("__pregel_push", 0, False),
                error=None,
                interrupts=(),
                state=None,
                result={"my_key": " one"},
            ),
            PregelTask(
                AnyStr(),
                "tool_two",
                (PULL, "tool_two"),
                interrupts=(
                    Interrupt(
                        value="Just because...",
                        id=AnyStr(),
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
        },
        parent_config=None,
        interrupts=(
            Interrupt(
                value="Just because...",
                id=AnyStr(),
            ),
        ),
    )

    # clear the interrupt and next tasks
    await tool_two.aupdate_state(thread1, None, as_node=END)
    # interrupt and next tasks are cleared, finished tasks are kept
    tup_upd = await tool_two.checkpointer.aget_tuple(thread1)
    assert await tool_two.aget_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️ one", "market": "DE"},
        next=(),
        tasks=(),
        config=tup_upd.config,
        created_at=tup_upd.checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 1,
        },
        parent_config=tup.config,
        interrupts=(),
    )


@NEEDS_CONTEXTVARS
async def test_node_not_cancelled_on_other_node_interrupted(
    async_checkpointer: BaseCheckpointSaver,
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
    builder.set_conditional_entry_point(lambda _: ["agent", "bad"])

    graph = builder.compile(checkpointer=async_checkpointer)
    thread = {"configurable": {"thread_id": "1"}}

    # writes from "awhile" are applied to last chunk
    assert await graph.ainvoke({"hello": "world"}, thread) == {
        "hello": "world again",
        "__interrupt__": [
            Interrupt(
                value="I am bad",
                id=AnyStr(),
            )
        ],
    }

    assert not inner_task_cancelled
    assert awhiles == 1

    assert await graph.ainvoke(None, thread, debug=True) == {
        "hello": "world again",
        "__interrupt__": [
            Interrupt(
                value="I am bad",
                id=AnyStr(),
            )
        ],
    }

    assert not inner_task_cancelled
    assert awhiles == 1

    # resume with answer
    assert await graph.ainvoke(Command(resume=" okay"), thread) == {
        "hello": "world again okay"
    }

    assert not inner_task_cancelled
    assert awhiles == 1


@pytest.mark.parametrize("stream_hang_s", [0.3, 0.6])
async def test_step_timeout_on_stream_hang(stream_hang_s: float) -> None:
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
        return {"hello": "1"}

    class State(TypedDict):
        hello: str

    builder = StateGraph(State)
    builder.add_node(awhile)
    builder.add_node(alittlewhile)
    builder.set_conditional_entry_point(lambda _: ["awhile", "alittlewhile"])
    graph = builder.compile()
    graph.step_timeout = 1

    with pytest.raises(asyncio.TimeoutError):
        async for chunk in graph.astream({"hello": "world"}, stream_mode="updates"):
            assert chunk == {"alittlewhile": {"hello": "1"}}
            await asyncio.sleep(stream_hang_s)

    assert inner_task_cancelled


async def test_cancel_graph_astream(async_checkpointer: BaseCheckpointSaver) -> None:
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

    graph = builder.compile(checkpointer=async_checkpointer)

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
    state = await graph.aget_state(thread1)
    assert state is not None
    assert state.values == {"value": 3}  # 1 + 2
    assert state.next == ("aparallelwhile",)
    assert state.metadata == {
        "parents": {},
        "source": "loop",
        "step": 0,
    }


async def test_cancel_graph_astream_events_v2(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
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

    graph = builder.compile(checkpointer=async_checkpointer)

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
    state = await graph.aget_state(thread2)
    assert state is not None
    assert state.values == {"value": 2}
    assert state.next == ("awhile",)
    assert state.metadata == {
        "parents": {},
        "source": "loop",
        "step": 1,
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

    builder = StateGraph(State, output_schema=Output)
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

    builder = StateGraph(State, output_schema=Output)
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
    assert await app.ainvoke(2) == 3
    assert await app.ainvoke(2, output_keys=["output"]) == {"output": 3}


async def test_invoke_single_process_in_write_kwargs(mocker: MockerFixture) -> None:
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
    assert await app.ainvoke(2) == {"output": 3, "fixed": 5, "output_plus_one": 4}


async def test_invoke_single_process_in_out_dict(mocker: MockerFixture) -> None:
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
    assert await app.ainvoke(2) == {"output": 3}


async def test_invoke_single_process_in_dict_out_dict(mocker: MockerFixture) -> None:
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
    assert await app.ainvoke({"input": 2}) == {"output": 3}


async def test_invoke_two_processes_in_out(mocker: MockerFixture) -> None:
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


async def test_batch_two_processes_in_out() -> None:
    async def add_one_with_delay(inp: int) -> int:
        await asyncio.sleep(inp / 10)
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

    assert await app.abatch([3, 2, 1, 3, 5]) == [5, 4, 3, 5, 7]
    assert await app.abatch([3, 2, 1, 3, 5], output_keys=["output"]) == [
        {"output": 5},
        {"output": 4},
        {"output": 3},
        {"output": 5},
        {"output": 7},
    ]


async def test_invoke_many_processes_in_out(mocker: MockerFixture) -> None:
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
        await app.ainvoke(2)


async def test_invoke_two_processes_two_in_two_out_valid(mocker: MockerFixture) -> None:
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

    # An Topic channel accumulates updates into a sequence
    assert await app.ainvoke(2) == [3, 3]


async def test_invoke_checkpoint(
    mocker: MockerFixture, async_checkpointer: BaseCheckpointSaver
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
        checkpointer=async_checkpointer,
        retry_policy=RetryPolicy(),
    )

    # total starts out as 0, so output is 0+2=2
    assert await app.ainvoke(2, {"configurable": {"thread_id": "1"}}) == 2
    checkpoint = await async_checkpointer.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 2
    # total is now 2, so output is 2+3=5
    assert await app.ainvoke(3, {"configurable": {"thread_id": "1"}}) == 5
    assert errored_once, "errored and retried"
    checkpoint = await async_checkpointer.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        await app.ainvoke(4, {"configurable": {"thread_id": "1"}})
    # checkpoint is not updated
    checkpoint = await async_checkpointer.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert await app.ainvoke(5, {"configurable": {"thread_id": "2"}}) == 5
    checkpoint = await async_checkpointer.aget({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    checkpoint = await async_checkpointer.aget({"configurable": {"thread_id": "2"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 5


async def test_pending_writes_resume(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    class State(TypedDict):
        value: Annotated[int, operator.add]

    class AwhileMaker:
        def __init__(self, sleep: float, rtn: dict | Exception) -> None:
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
    graph = builder.compile(checkpointer=async_checkpointer)

    thread1: RunnableConfig = {"configurable": {"thread_id": "1"}}
    with pytest.raises(ConnectionError, match="I'm not good"):
        await graph.ainvoke({"value": 1}, thread1, durability=durability)

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
    }
    # get_state with checkpoint_id should not apply any pending writes
    state = await graph.aget_state(state.config)
    assert state is not None
    assert state.values == {"value": 1}
    assert state.next == ("one", "two")
    # should contain pending write of "one"
    checkpoint = await async_checkpointer.aget_tuple(thread1)
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
        await graph.ainvoke(None, thread1, durability=durability)

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
    assert await graph.ainvoke(None, thread1, durability=durability) == {"value": 6}

    # check all final checkpoints
    checkpoints = [c async for c in async_checkpointer.alist(thread1)]
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
                "checkpoint_id": checkpoints[2].config["configurable"]["checkpoint_id"],
            }
        }
        if durability != "exit"
        else None,
        pending_writes=UnsortedSequence(
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


async def test_run_from_checkpoint_id_retains_previous_writes(
    async_checkpointer: BaseCheckpointSaver,
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
    graph = builder.compile(checkpointer=async_checkpointer)

    thread_id = uuid.uuid4()
    thread1 = {"configurable": {"thread_id": str(thread_id)}}

    result = await graph.ainvoke({"myval": 1}, thread1, durability="async")
    assert result["myval"] == 4
    history = [c async for c in graph.aget_state_history(thread1)]

    assert len(history) == 4
    assert history[0].values == {"myval": 4, "otherval": False}
    assert history[-1].values == {"myval": 0}

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
            "3.1",
            "2|1",
            "2|2",
            "2|3",
            "2|4",
            "3",
        ]
    )


async def test_send_sequences(async_checkpointer: BaseCheckpointSaver) -> None:
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
    assert await graph.ainvoke(["0"]) == [
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

    graph = builder.compile(checkpointer=async_checkpointer, interrupt_before=["3.1"])
    thread1 = {"configurable": {"thread_id": "1"}}
    assert await graph.ainvoke(["0"], thread1) == [
        "0",
        "1",
    ]
    assert await graph.ainvoke(None, thread1) == [
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


@NEEDS_CONTEXTVARS
async def test_imp_task(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    mapper_calls = 0

    @task()
    async def mapper(input: int) -> str:
        nonlocal mapper_calls
        mapper_calls += 1
        await asyncio.sleep(0.1 * input)
        return str(input) * 2

    @entrypoint(checkpointer=async_checkpointer)
    async def graph(input: list[int]) -> list[str]:
        futures = [mapper(i) for i in input]
        mapped = await asyncio.gather(*futures)
        answer = interrupt("question")
        return [m + answer for m in mapped]

    tracer = FakeTracer()
    thread1 = {"configurable": {"thread_id": "1"}, "callbacks": [tracer]}
    assert [c async for c in graph.astream([0, 1], thread1, durability=durability)] == [
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
    assert len(tracer.runs) == 1
    assert len(tracer.runs[0].child_runs) == 1
    entrypoint_run = tracer.runs[0].child_runs[0]
    assert entrypoint_run.name == "graph"
    mapper_runs = [r for r in entrypoint_run.child_runs if r.name == "mapper"]
    assert len(mapper_runs) == 2
    assert any(r.inputs == {"input": 0} for r in mapper_runs)
    assert any(r.inputs == {"input": 1} for r in mapper_runs)

    assert await graph.ainvoke(
        Command(resume="answer"), thread1, durability=durability
    ) == [
        "00answer",
        "11answer",
    ]
    assert mapper_calls == 2


@NEEDS_CONTEXTVARS
async def test_imp_nested(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    async def mynode(input: list[str]) -> list[str]:
        return [it + "a" for it in input]

    builder = StateGraph(list[str])
    builder.add_node(mynode)
    builder.add_edge(START, "mynode")
    add_a = builder.compile()

    @task
    def submapper(input: int) -> str:
        return str(input)

    @task
    async def mapper(input: int) -> str:
        await asyncio.sleep(input / 100)
        return await submapper(input) * 2

    @entrypoint(checkpointer=async_checkpointer)
    async def graph(input: list[int]) -> list[str]:
        futures = [mapper(i) for i in input]
        mapped = await asyncio.gather(*futures)
        answer = interrupt("question")
        final = [m + answer for m in mapped]
        return await add_a.ainvoke(final)

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
    assert [c async for c in graph.astream([0, 1], thread1, durability=durability)] == [
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

    assert await graph.ainvoke(
        Command(resume="answer"), thread1, durability=durability
    ) == [
        "00answera",
        "11answera",
    ]


@NEEDS_CONTEXTVARS
async def test_imp_task_cancel(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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

    @entrypoint(checkpointer=async_checkpointer)
    async def graph(input: list[int]) -> list[str]:
        futures = [mapper(i) for i in input]
        await asyncio.sleep(0.1)
        futures.pop().cancel()  # cancel one
        mapped = await asyncio.gather(*futures)
        answer = interrupt("question")
        return [m + answer for m in mapped]

    thread1 = {"configurable": {"thread_id": "1"}}
    assert [c async for c in graph.astream([0, 1], thread1, durability=durability)] == [
        {"mapper": "00"},
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
    assert mapper_cancels == 1

    assert await graph.ainvoke(
        Command(resume="answer"), thread1, durability=durability
    ) == [
        "00answer",
    ]
    assert mapper_calls == 3
    assert mapper_cancels == 2


@NEEDS_CONTEXTVARS
async def test_imp_sync_from_async(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    @task()
    def foo(state: dict) -> dict:
        return {"a": state["a"] + "foo", "b": "bar"}

    @task
    def bar(a: str, b: str, c: str | None = None) -> dict:
        return {"a": a + b, "c": (c or "") + "bark"}

    @task()
    def baz(state: dict) -> dict:
        return {"a": state["a"] + "baz", "c": "something else"}

    @entrypoint(checkpointer=async_checkpointer)
    def graph(state: dict) -> dict:
        foo_result = foo(state).result()
        fut_bar = bar(foo_result["a"], foo_result["b"])
        fut_baz = baz(fut_bar.result())
        return fut_baz.result()

    thread1 = {"configurable": {"thread_id": "1"}}
    assert [
        c async for c in graph.astream({"a": "0"}, thread1, durability=durability)
    ] == [
        {"foo": {"a": "0foo", "b": "bar"}},
        {"bar": {"a": "0foobar", "c": "bark"}},
        {"baz": {"a": "0foobarbaz", "c": "something else"}},
        {"graph": {"a": "0foobarbaz", "c": "something else"}},
    ]


@NEEDS_CONTEXTVARS
async def test_imp_stream_order(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    @task()
    async def foo(state: dict) -> dict:
        return {"a": state["a"] + "foo", "b": "bar"}

    @task
    async def bar(a: str, b: str, c: str | None = None) -> dict:
        return {"a": a + b, "c": (c or "") + "bark"}

    @task()
    async def baz(state: dict) -> dict:
        return {"a": state["a"] + "baz", "c": "something else"}

    @entrypoint(checkpointer=async_checkpointer)
    async def graph(state: dict) -> dict:
        foo_res = await foo(state)

        fut_bar = bar(foo_res["a"], foo_res["b"])
        fut_baz = baz(await fut_bar)
        return await fut_baz

    thread1 = {"configurable": {"thread_id": "1"}}
    assert [
        c async for c in graph.astream({"a": "0"}, thread1, durability=durability)
    ] == [
        {"foo": {"a": "0foo", "b": "bar"}},
        {"bar": {"a": "0foobar", "c": "bark"}},
        {"baz": {"a": "0foobarbaz", "c": "something else"}},
        {"graph": {"a": "0foobarbaz", "c": "something else"}},
    ]


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Requires Python 3.11 or higher for context management",
)
async def test_send_dedupe_on_resume(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    class InterruptOnce:
        ticks: int = 0

        def __call__(self, state):
            self.ticks += 1
            if self.ticks == 1:
                interrupt("Bahh")
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

    graph = builder.compile(checkpointer=async_checkpointer)
    thread1 = {"configurable": {"thread_id": "1"}}
    assert await graph.ainvoke(["0"], thread1, durability=durability) == {
        "__interrupt__": [
            Interrupt(
                value="Bahh",
                id=AnyStr(),
            ),
        ],
    }
    assert builder.nodes["2"].runnable.func.ticks == 3
    assert builder.nodes["flaky"].runnable.func.ticks == 1
    # resume execution
    assert await graph.ainvoke(None, thread1, durability=durability) == [
        "0",
        "1",
        "3.1",
        "2|Command(goto=Send(node='2', arg=3))",
        "2|Command(goto=Send(node='flaky', arg=4))",
        "3",
        "2|3",
        "flaky|4",
        "3",
    ]
    # node "2" doesn't get called again, as we recover writes saved before
    assert builder.nodes["2"].runnable.func.ticks == 3
    # node "flaky" gets called again, as it was interrupted
    assert builder.nodes["flaky"].runnable.func.ticks == 2
    # check history
    history = [c async for c in graph.aget_state_history(thread1)]
    assert len(history) == (6 if durability != "exit" else 2)
    expected_history = [
        StateSnapshot(
            values=[
                "0",
                "1",
                "3.1",
                "2|Command(goto=Send(node='2', arg=3))",
                "2|Command(goto=Send(node='flaky', arg=4))",
                "3",
                "2|3",
                "flaky|4",
                "3",
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
                "step": 4,
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
            interrupts=(),
        ),
        StateSnapshot(
            values=[
                "0",
                "1",
                "3.1",
                "2|Command(goto=Send(node='2', arg=3))",
                "2|Command(goto=Send(node='flaky', arg=4))",
                "3",
                "2|3",
                "flaky|4",
            ],
            next=("3",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "step": 3,
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
            ),
            interrupts=(),
        ),
        StateSnapshot(
            values=[
                "0",
                "1",
                "3.1",
                "2|Command(goto=Send(node='2', arg=3))",
                "2|Command(goto=Send(node='flaky', arg=4))",
            ],
            next=("2", "flaky", "3"),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
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
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="2",
                    path=("__pregel_push", 0, False),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["2|3"],
                ),
                PregelTask(
                    id=AnyStr(),
                    name="flaky",
                    path=("__pregel_push", 1, False),
                    error=None,
                    interrupts=(Interrupt(value="Bahh", id=AnyStr()),),
                    state=None,
                    result=["flaky|4"] if durability != "exit" else None,
                ),
                PregelTask(
                    id=AnyStr(),
                    name="3",
                    path=("__pregel_pull", "3"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["3"],
                ),
            ),
            interrupts=(Interrupt(value="Bahh", id=AnyStr()),),
        ),
        StateSnapshot(
            values=["0", "1"],
            next=("2", "2", "3.1"),
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
                    name="2",
                    path=("__pregel_push", 0, False),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["2|Command(goto=Send(node='2', arg=3))"],
                ),
                PregelTask(
                    id=AnyStr(),
                    name="2",
                    path=("__pregel_push", 1, False),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=["2|Command(goto=Send(node='flaky', arg=4))"],
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
            interrupts=(),
        ),
        StateSnapshot(
            values=["0"],
            next=("1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
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
            ),
            interrupts=(),
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
            interrupts=(),
        ),
    ]
    if durability != "exit":
        assert history == expected_history
    else:
        assert history[0] == expected_history[0]._replace(
            parent_config=history[1].config
        )
        assert history[1] == expected_history[2]._replace(parent_config=None)


async def test_send_react_interrupt(async_checkpointer: BaseCheckpointSaver) -> None:
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

    # simple interrupt-resume flow
    foo_called = 0
    graph = builder.compile(checkpointer=async_checkpointer, interrupt_before=["foo"])
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
    graph = builder.compile(checkpointer=async_checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "2"}}
    assert await graph.ainvoke(
        {"messages": [HumanMessage("hello")]}, thread1, durability="exit"
    ) == {
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
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 1,
            "source": "loop",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", 0, False),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
        interrupts=(),
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
            "step": 2,
            "source": "update",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=(
            {
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            }
        ),
        tasks=(),
        interrupts=(),
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
    graph = builder.compile(checkpointer=async_checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "3"}}
    assert await graph.ainvoke(
        {"messages": [HumanMessage("hello")]}, thread1, durability="exit"
    ) == {
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
            "step": 1,
            "source": "loop",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", 0, False),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
        interrupts=(),
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
            "step": 2,
            "source": "update",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=(
            {
                "configurable": {
                    "thread_id": "3",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            }
        ),
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", 0, False),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
        interrupts=(),
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


async def test_send_react_interrupt_control(
    async_checkpointer: BaseCheckpointSaver, snapshot: SnapshotAssertion
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
    if isinstance(async_checkpointer, InMemorySaver):
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

    # simple interrupt-resume flow
    foo_called = 0
    graph = builder.compile(checkpointer=async_checkpointer, interrupt_before=["foo"])
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
    graph = builder.compile(checkpointer=async_checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "2"}}
    assert await graph.ainvoke(
        {"messages": [HumanMessage("hello")]}, thread1, durability="exit"
    ) == {
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
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 1,
            "source": "loop",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", 0, False),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
        interrupts=(),
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
            "step": 2,
            "source": "update",
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=(
            {
                "configurable": {
                    "thread_id": "2",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            }
        ),
        tasks=(),
        interrupts=(),
    )

    # tool call not executed
    assert await graph.ainvoke(None, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(content="Bye now"),
        ]
    }
    assert foo_called == 0


async def test_max_concurrency(async_checkpointer: BaseCheckpointSaver) -> None:
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

    graph = builder.compile(checkpointer=async_checkpointer, interrupt_before=["2"])
    thread1 = {"max_concurrency": 10, "configurable": {"thread_id": "1"}}

    assert await graph.ainvoke(["0"], thread1, debug=True) == ["0", "1"]
    state = await graph.aget_state(thread1)
    assert state.values == ["0", "1"]
    assert await graph.ainvoke(None, thread1) == ["0", "1", *range(100), "3"]


async def test_max_concurrency_control(async_checkpointer: BaseCheckpointSaver) -> None:
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

    if isinstance(async_checkpointer, InMemorySaver):
        assert (
            graph.get_graph().draw_mermaid()
            == """---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	1(1)
	2(2)
	3(3)
	__end__([<p>__end__</p>]):::last
	1 -.-> 2;
	2 -.-> 3;
	__start__ --> 1;
	3 --> __end__;
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

    graph = builder.compile(checkpointer=async_checkpointer, interrupt_before=["2"])
    thread1 = {"max_concurrency": 10, "configurable": {"thread_id": "1"}}

    assert await graph.ainvoke(["0"], thread1) == ["0", "1"]
    assert await graph.ainvoke(None, thread1) == ["0", "1", *range(100), "3"]


async def test_invoke_checkpoint_three(
    mocker: MockerFixture, async_checkpointer: BaseCheckpointSaver
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x["total"] + x["input"])

    def raise_if_above_10(input: int) -> int:
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
        checkpointer=async_checkpointer,
        debug=True,
    )

    thread_1 = {"configurable": {"thread_id": "1"}}
    # total starts out as 0, so output is 0+2=2
    assert await app.ainvoke(2, thread_1, durability="async") == 2
    state = await app.aget_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 2
    assert (
        state.config["configurable"]["checkpoint_id"]
        == (await async_checkpointer.aget(thread_1))["id"]
    )
    # total is now 2, so output is 2+3=5
    assert await app.ainvoke(3, thread_1, durability="async") == 5
    state = await app.aget_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert (
        state.config["configurable"]["checkpoint_id"]
        == (await async_checkpointer.aget(thread_1))["id"]
    )
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        await app.ainvoke(4, thread_1, durability="async")
    # checkpoint is not updated
    state = await app.aget_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert state.next == ("one",)
    """we checkpoint inputs and it failed on "one", so the next node is one"""
    # we can recover from error by sending new inputs
    assert await app.ainvoke(2, thread_1, durability="async") == 9
    state = await app.aget_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 16, "total is now 7+9=16"
    assert state.next == ()

    thread_2 = {"configurable": {"thread_id": "2"}}
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert await app.ainvoke(5, thread_2) == 5
    state = await app.aget_state(thread_1)
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
    assert (await async_checkpointer.aget(thread_1_history[0].config))[
        "id"
    ] == thread_1_history[0].config["configurable"]["checkpoint_id"]
    assert (await async_checkpointer.aget(thread_1_history[1].config))[
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
        assert await app.ainvoke(2) == [13, 13]

    assert await asyncio.gather(*(app.ainvoke(2) for _ in range(100))) == [
        [13, 13] for _ in range(100)
    ]


async def test_invoke_two_processes_one_in_two_out(mocker: MockerFixture) -> None:
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

    # Then invoke pubsub
    assert [c async for c in app.astream(2)] == [
        {"between": 3, "output": 3},
        {"between": 3, "output": 4},
    ]


async def test_invoke_two_processes_no_out(mocker: MockerFixture) -> None:
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
    # but returns nothing, as nothing was published to "output" topic
    assert await app.ainvoke(2) is None


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


async def test_in_one_fan_out_state_graph_waiting_edge(
    async_checkpointer: BaseCheckpointSaver,
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

    async def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    async def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

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
        checkpointer=async_checkpointer,
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


@pytest.mark.parametrize("use_waiting_edge", (True, False))
async def test_in_one_fan_out_state_graph_defer_node(
    async_checkpointer: BaseCheckpointSaver, use_waiting_edge: bool
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

    async def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    async def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

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
    workflow.add_node("qa", qa, defer=True)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    if use_waiting_edge:
        workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    else:
        workflow.add_edge("retriever_one", "qa")
        workflow.add_edge("retriever_two", "qa")
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

    app_w_interrupt = workflow.compile(
        checkpointer=async_checkpointer,
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


async def test_in_one_fan_out_state_graph_waiting_edge_via_branch(
    async_checkpointer: BaseCheckpointSaver,
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

    async def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    async def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

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

    app_w_interrupt = workflow.compile(
        checkpointer=async_checkpointer,
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


async def test_nested_pydantic_models() -> None:
    """Test that nested Pydantic models are properly constructed from leaf nodes up."""

    class NestedModel(BaseModel):
        value: int
        name: str
        something: str | None = None

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

    class MyEnum(enum.Enum):
        A = 1
        B = 2

    class MyTypedDict(TypedDict):
        x: int
        my_enum: MyEnum

    class State(BaseModel):
        # Basic nested model tests
        top_level: str
        nested: NestedModel
        optional_nested: NestedModel | None = None
        dict_nested: dict[str, NestedModel]
        my_set: set[int]
        another_set: set
        my_enum: MyEnum
        list_nested: Annotated[
            dict | list[dict[str, NestedModel]], lambda x, y: (x or []) + [y]
        ]
        list_nested_reversed: Annotated[
            list[dict[str, NestedModel]] | NestedModel | dict | list,
            lambda x, y: (x or []) + [y],
        ]
        tuple_nested: tuple[str, NestedModel]
        tuple_list_nested: list[tuple[int, NestedModel]]
        complex_tuple: tuple[str, dict[str, tuple[int, NestedModel]]]
        my_typed_dict: MyTypedDict

        # Forward reference test
        recursive: RecursiveModel

        # Discriminated union test
        pet: Cat | Dog

        # Cyclic reference test
        people: dict[str, Person]  # Map of ID -> Person

    inputs = {
        # Basic nested models
        "top_level": "initial",
        "nested": {"value": 42, "name": "test"},
        "optional_nested": {"value": 10, "name": "optional"},
        "my_set": [1, 2, 7],
        "another_set": ["foo", 3],
        "my_enum": MyEnum.B,
        "my_typed_dict": {"x": 1, "my_enum": MyEnum.A},
        "dict_nested": {"a": {"value": 5, "name": "a"}},
        "list_nested": [{"a": {"value": 6, "name": "b"}}],
        "list_nested_reversed": ["foo", "bar"],
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

    async def node_fn(state: State) -> dict:
        assert state == State(**inputs)
        return update

    builder = StateGraph(State)
    builder.add_node("process", node_fn)
    builder.set_entry_point("process")
    builder.set_finish_point("process")
    graph = builder.compile()

    result = await graph.ainvoke(inputs.copy())

    assert result == {**inputs, **update}


async def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    def sorted_add(x: list[str], y: list[str] | list[tuple[str, str]]) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        query: str
        answer: str | None = None
        docs: Annotated[list[str], sorted_add]

    class Input(BaseModel):
        query: str

    class Output(BaseModel):
        answer: str
        docs: list[str]

    class StateUpdate(BaseModel):
        query: str | None = None
        answer: str | None = None
        docs: list[str] | None = None

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

    with pytest.raises(ValidationError):
        await app.ainvoke({"query": {}})

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
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
        checkpointer=async_checkpointer,
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
            "step": 4,
        },
        created_at=AnyStr(),
        parent_config=(
            {
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            }
        ),
        interrupts=(),
    )

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
    snapshot: SnapshotAssertion, async_checkpointer: BaseCheckpointSaver
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
        query: str
        inner: InnerObject
        answer: str | None = None
        docs: Annotated[list[str], sorted_add]

    class StateUpdate(BaseModel):
        query: str | None = None
        answer: str | None = None
        docs: list[str] | None = None

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

    if isinstance(async_checkpointer, InMemorySaver):
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
        assert app.get_input_jsonschema() == snapshot
        assert app.get_output_jsonschema() == snapshot

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
        checkpointer=async_checkpointer,
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


async def test_in_one_fan_out_state_graph_waiting_edge_plus_regular(
    async_checkpointer: BaseCheckpointSaver,
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

    async def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    async def analyzer_one(data: State) -> State:
        await asyncio.sleep(0.1)
        return {"query": f"analyzed: {data['query']}"}

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
        checkpointer=async_checkpointer,
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


@pytest.mark.parametrize("with_cache", [True, False])
async def test_in_one_fan_out_state_graph_waiting_edge_multiple(
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

    async def rewrite_query(data: State) -> State:
        nonlocal rewrite_query_count
        rewrite_query_count += 1
        return {"query": f"query: {data['query']}"}

    async def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

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

    assert await app.ainvoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: analyzed: query: what is weather in sf",
        "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
        "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
    }
    assert rewrite_query_count == 2

    assert [c async for c in app.astream({"query": "what is weather in sf"})] == [
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
        await app.aclear_cache()

        assert await app.ainvoke({"query": "what is weather in sf"}) == {
            "query": "analyzed: query: analyzed: query: what is weather in sf",
            "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
            "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
        }
        assert rewrite_query_count == 4


async def test_in_one_fan_out_state_graph_waiting_edge_multiple_cond_edge() -> None:
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

    async def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    async def retriever_picker(data: State) -> list[str]:
        return ["analyzer_one", "retriever_two"]

    async def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

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
                "output": {"side": {"my_key": "my value there and back again"}}
            }
    assert times_called == 1


async def test_subgraph_checkpoint_true(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
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

    app = graph.compile(checkpointer=async_checkpointer)

    config = {"configurable": {"thread_id": "2"}}
    assert [
        c
        async for c in app.astream(
            {"my_key": ""},
            config,
            subgraphs=True,
            durability=durability,
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


async def test_subgraph_durability_inherited(
    durability: Durability,
) -> None:
    async_checkpointer = InMemorySaver()

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

    inner_app = inner.compile(checkpointer=async_checkpointer)
    graph = StateGraph(State)
    graph.add_node("inner", inner_app)
    graph.add_edge(START, "inner")
    graph.add_conditional_edges(
        "inner", lambda s: "inner" if s["my_key"].count("there") < 2 else END
    )
    app = graph.compile(checkpointer=async_checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    await app.ainvoke({"my_key": ""}, config, subgraphs=True, durability=durability)
    if durability != "exit":
        checkpoints = list(async_checkpointer.list(config))
        assert len(checkpoints) == 12
    else:
        checkpoints = list(async_checkpointer.list(config))
        assert len(checkpoints) == 1


@NEEDS_CONTEXTVARS
async def test_subgraph_checkpoint_true_interrupt(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
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

    async def node_2(state: ParentState, config: RunnableConfig):
        response = await subgraph.ainvoke({"bar": state["foo"]})
        return {"foo": response["bar"]}

    builder = StateGraph(ParentState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")

    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    assert await graph.ainvoke({"foo": "foo"}, config, durability=durability) == {
        "foo": "hi! foo",
        "__interrupt__": [
            Interrupt(
                value="Provide baz value",
                id=AnyStr(),
            )
        ],
    }
    assert (await graph.aget_state(config, subgraphs=True)).tasks[0].state.values == {
        "bar": "hi! foo"
    }
    assert await graph.ainvoke(
        Command(resume="baz"), config, durability=durability
    ) == {"foo": "hi! foobaz"}


async def test_stream_subgraphs_during_execution(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
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

    app = graph.compile(checkpointer=async_checkpointer)

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


@NEEDS_CONTEXTVARS
async def test_stream_buffering_single_node(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
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

    graph = builder.compile(checkpointer=async_checkpointer)

    start = perf_counter()
    chunks: list[tuple[float, Any]] = []
    config = {"configurable": {"thread_id": "2"}}
    async for c in graph.astream({"my_key": ""}, config, stream_mode="custom"):
        chunks.append((round(perf_counter() - start, 1), c))

    assert chunks == [
        (FloatBetween(0.0, 0.1), "Before sleep"),
        (FloatBetween(0.2, 0.3), "After sleep"),
    ]


async def test_nested_graph_interrupts_parallel(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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

    app = graph.compile(checkpointer=async_checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert await app.ainvoke({"my_key": ""}, config, durability=durability) == {
        "my_key": " and parallel",
    }

    assert await app.ainvoke(None, config, durability=durability) == {
        "my_key": "got here and there and parallel and back again",
    }

    # below combo of assertions is asserting two things
    # - outer_1 finishes before inner interrupts (because we see its output in stream, which only happens after node finishes)
    # - the writes of outer are persisted in 1st call and used in 2nd call, ie outer isn't called again (because we dont see outer_1 output again in 2nd stream)
    # test stream updates w/ nested interrupt
    config = {"configurable": {"thread_id": "2"}}
    assert [
        c
        async for c in app.astream(
            {"my_key": ""},
            config,
            subgraphs=True,
            durability=durability,
        )
    ] == [
        # we got to parallel node first
        ((), {"outer_1": {"my_key": " and parallel"}}),
        (
            (AnyStr("inner:"),),
            {"inner_1": {"my_key": "got here", "my_other_key": ""}},
        ),
        ((), {"__interrupt__": ()}),
    ]
    assert [c async for c in app.astream(None, config, durability=durability)] == [
        {"outer_1": {"my_key": " and parallel"}, "__metadata__": {"cached": True}},
        {"inner": {"my_key": "got here and there"}},
        {"outer_2": {"my_key": " and back again"}},
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [
        c
        async for c in app.astream(
            {"my_key": ""},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [
        c
        async for c in app.astream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # # test interrupts BEFORE the parallel node
    app = graph.compile(checkpointer=async_checkpointer, interrupt_before=["outer_1"])
    config = {"configurable": {"thread_id": "4"}}
    assert [
        c
        async for c in app.astream(
            {"my_key": ""},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [
        {"my_key": ""},
    ]
    # while we're waiting for the node w/ interrupt inside to finish
    assert [
        c
        async for c in app.astream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [
        c
        async for c in app.astream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # test interrupts AFTER the parallel node
    app = graph.compile(checkpointer=async_checkpointer, interrupt_after=["outer_1"])
    config = {"configurable": {"thread_id": "5"}}
    assert [
        c
        async for c in app.astream(
            {"my_key": ""},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [
        c
        async for c in app.astream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
    ]
    assert [
        c
        async for c in app.astream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]


async def test_doubly_nested_graph_interrupts(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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

    app = graph.compile(checkpointer=async_checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert await app.ainvoke({"my_key": "my value"}, config, durability=durability) == {
        "my_key": "hi my value",
    }

    assert await app.ainvoke(None, config, durability=durability) == {
        "my_key": "hi my value here and there and back again",
    }

    # test stream updates w/ nested interrupt
    nodes: list[str] = []
    config = {
        "configurable": {"thread_id": "2", CONFIG_KEY_NODE_FINISHED: nodes.append}
    }
    assert [
        c
        async for c in app.astream(
            {"my_key": "my value"}, config, durability=durability
        )
    ] == [
        {"parent_1": {"my_key": "hi my value"}},
        {"__interrupt__": ()},
    ]
    assert nodes == ["parent_1", "grandchild_1"]
    assert [c async for c in app.astream(None, config, durability=durability)] == [
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
            {"my_key": "my value"},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [
        {"my_key": "my value"},
        {"my_key": "hi my value"},
    ]
    assert [
        c
        async for c in app.astream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": "hi my value"},
        {"my_key": "hi my value here and there"},
        {"my_key": "hi my value here and there and back again"},
    ]


async def test_checkpoint_metadata(async_checkpointer: BaseCheckpointSaver) -> None:
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
    app = workflow.compile(checkpointer=async_checkpointer)

    # graph w/ interrupt
    app_w_interrupt = workflow.compile(
        checkpointer=async_checkpointer, interrupt_before=["tools"]
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
    chkpnt_metadata_1 = (await async_checkpointer.aget_tuple(config)).metadata
    assert chkpnt_metadata_1["test_config_1"] == "foo"
    assert chkpnt_metadata_1["test_config_2"] == "bar"

    # Verify that all checkpoint metadata have the expected keys. This check
    # is needed because a run may have an arbitrary number of steps depending
    # on how the graph is constructed.
    chkpnt_tuples_1 = async_checkpointer.alist(config)
    async for chkpnt_tuple in chkpnt_tuples_1:
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
    chkpnt_metadata_2 = (await async_checkpointer.aget_tuple(config)).metadata
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
    chkpnt_metadata_3 = (await async_checkpointer.aget_tuple(config)).metadata
    assert chkpnt_metadata_3["test_config_3"] == "foo"
    assert chkpnt_metadata_3["test_config_4"] == "bar"

    # Verify that all checkpoint metadata have the expected keys. This check
    # is needed because a run may have an arbitrary number of steps depending
    # on how the graph is constructed.
    chkpnt_tuples_2 = async_checkpointer.alist(config)
    async for chkpnt_tuple in chkpnt_tuples_2:
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


async def test_store_injected_async(
    async_checkpointer: BaseCheckpointSaver, async_store: BaseStore
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

        async def __call__(
            self, inputs: State, config: RunnableConfig, store: BaseStore
        ):
            assert isinstance(store, BaseStore)
            await store.aput(
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

    N = 50
    M = 1

    for i in range(N):
        builder.add_node(f"node_{i}", Node(i))
        builder.add_edge("__start__", f"node_{i}")

    graph = builder.compile(store=async_store, checkpointer=async_checkpointer)

    # Test batch operations with multiple threads
    results = await graph.abatch(
        [{"count": 0}] * M,
        ([{"configurable": {"thread_id": str(uuid.uuid4())}}] * (M - 1))
        + [{"configurable": {"thread_id": thread_1}}],
    )
    result = results[-1]
    assert result == {"count": N + 1}
    returned_doc = (await async_store.aget(namespace, doc_id)).value
    assert returned_doc == {**doc, "from_thread": thread_1, "some_val": 0}
    assert len(await async_store.asearch(namespace)) == 1

    # Check results after another turn of the same thread
    result = await graph.ainvoke(
        {"count": 0}, {"configurable": {"thread_id": thread_1}}
    )
    assert result == {"count": (N + 1) * 2}
    returned_doc = (await async_store.aget(namespace, doc_id)).value
    assert returned_doc == {**doc, "from_thread": thread_1, "some_val": N + 1}
    assert len(await async_store.asearch(namespace)) == 1

    # Test with a different thread
    result = await graph.ainvoke(
        {"count": 0}, {"configurable": {"thread_id": thread_2}}
    )
    assert result == {"count": N + 1}
    returned_doc = (await async_store.aget(namespace, doc_id)).value
    assert returned_doc == {
        **doc,
        "from_thread": thread_2,
        "some_val": 0,
    }  # Overwrites the whole doc
    assert (
        len(await async_store.asearch(namespace)) == 1
    )  # still overwriting the same one


async def test_debug_retry(async_checkpointer: BaseCheckpointSaver):
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

    graph = builder.compile(checkpointer=async_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    await graph.ainvoke({"messages": []}, config=config, durability="async")

    # re-run step: 1
    async for c in async_checkpointer.alist(config):
        if c.metadata["step"] == 1:
            target_config = c.parent_config
            break
    assert target_config is not None

    update_config = await graph.aupdate_state(target_config, values=None)

    events = [
        c
        async for c in graph.astream(
            None, config=update_config, stream_mode="debug", durability="async"
        )
    ]

    checkpoint_events = list(
        reversed([e["payload"] for e in events if e["type"] == "checkpoint"])
    )

    checkpoint_history = {
        c.config["configurable"]["checkpoint_id"]: c
        async for c in graph.aget_state_history(config)
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


async def test_debug_subgraphs(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
):
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

    graph = parent.compile(checkpointer=async_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    events = [
        c
        async for c in graph.astream(
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
    checkpoint_history = [c async for c in graph.aget_state_history(config)]

    assert len(checkpoint_events) == len(checkpoint_history)

    def normalize_config(config: dict | None) -> dict | None:
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


async def test_debug_nested_subgraphs(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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

    graph = grand_parent.compile(checkpointer=async_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    events = [
        c
        async for c in graph.astream(
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


@pytest.mark.parametrize("subgraph_persist", [True, False])
async def test_parent_command(
    async_checkpointer: BaseCheckpointSaver, subgraph_persist: bool
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

    graph = builder.compile(checkpointer=async_checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    assert await graph.ainvoke(
        {"messages": [("user", "get user name")]}, config, durability="exit"
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
            "step": 1,
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        tasks=(),
        interrupts=(),
    )


@NEEDS_CONTEXTVARS
async def test_interrupt_subgraph(async_checkpointer: BaseCheckpointSaver) -> None:
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

    graph = builder.compile(checkpointer=async_checkpointer)

    thread1 = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert await graph.ainvoke({"baz": ""}, thread1)
    # Resume with answer
    assert await graph.ainvoke(Command(resume="bar"), thread1)


@NEEDS_CONTEXTVARS
async def test_interrupt_multiple(async_checkpointer: BaseCheckpointSaver):
    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    async def node(s: State) -> State:
        answer = interrupt({"value": 1})
        answer2 = interrupt({"value": 2})
        return {"my_key": answer + " " + answer2}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")

    graph = builder.compile(checkpointer=async_checkpointer)
    thread1 = {"configurable": {"thread_id": "1"}}

    assert [
        e async for e in graph.astream({"my_key": "DE", "market": "DE"}, thread1)
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value={"value": 1},
                    id=AnyStr(),
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
                    id=AnyStr(),
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


@NEEDS_CONTEXTVARS
async def test_interrupt_loop(async_checkpointer: BaseCheckpointSaver) -> None:
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

    graph = builder.compile(checkpointer=async_checkpointer)
    thread1 = {"configurable": {"thread_id": "1"}}

    assert [e async for e in graph.astream({"other": ""}, thread1)] == [
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
        async for event in graph.astream(
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
        async for event in graph.astream(
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

    assert [event async for event in graph.astream(Command(resume="19"), thread1)] == [
        {"node": {"age": 19}},
    ]


@NEEDS_CONTEXTVARS
async def test_interrupt_functional(async_checkpointer: BaseCheckpointSaver) -> None:
    @task
    async def foo(state: dict) -> dict:
        return {"a": state["a"] + "foo"}

    @task
    async def bar(state: dict) -> dict:
        return {"a": state["a"] + "bar", "b": state["b"]}

    @entrypoint(checkpointer=async_checkpointer)
    async def graph(inputs: dict) -> dict:
        foo_result = await foo(inputs)
        value = interrupt("Provide value for bar:")
        bar_input = {**foo_result, "b": value}
        bar_result = await bar(bar_input)
        return bar_result

    config = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    await graph.ainvoke({"a": ""}, config)
    # Resume with an answer
    res = await graph.ainvoke(Command(resume="bar"), config)
    assert res == {"a": "foobar", "b": "bar"}


@NEEDS_CONTEXTVARS
async def test_interrupt_task_functional(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    @task
    async def foo(state: dict) -> dict:
        return {"a": state["a"] + "foo"}

    @task
    async def bar(state: dict) -> dict:
        value = interrupt("Provide value for bar:")
        return {"a": state["a"] + value}

    @entrypoint(checkpointer=async_checkpointer)
    async def graph(inputs: dict) -> dict:
        foo_result = await foo(inputs)
        bar_result = await bar(foo_result)
        return bar_result

    config = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert await graph.ainvoke({"a": ""}, config) == {
        "__interrupt__": [
            Interrupt(
                value="Provide value for bar:",
                id=AnyStr(),
            ),
        ]
    }
    # Resume with an answer
    res = await graph.ainvoke(Command(resume="bar"), config)
    assert res == {"a": "foobar"}


async def test_command_with_static_breakpoints(
    async_checkpointer: BaseCheckpointSaver,
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

    graph = builder.compile(checkpointer=async_checkpointer, interrupt_before=["node1"])
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Start the graph and interrupt at the first node
    await graph.ainvoke({"foo": "abc"}, config)
    result = await graph.ainvoke(Command(update={"foo": "def"}), config)
    assert result == {"foo": "def|node-1|node-2"}


async def test_multistep_plan(async_checkpointer: BaseCheckpointSaver) -> None:
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

    graph = builder.compile(checkpointer=async_checkpointer)

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


async def test_command_goto_with_static_breakpoints(
    async_checkpointer: BaseCheckpointSaver,
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

    graph = builder.compile(checkpointer=async_checkpointer, interrupt_before=["node1"])

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Start the graph and interrupt at the first node
    await graph.ainvoke({"foo": "abc"}, config)
    result = await graph.ainvoke(Command(goto=["node2"]), config)
    assert result == {"foo": "abc|node-1|node-2|node-2"}


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


@NEEDS_CONTEXTVARS
async def test_multiple_interrupt_state_persistence(
    async_checkpointer: BaseCheckpointSaver,
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

    app = builder.compile(checkpointer=async_checkpointer)
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


async def test_checkpoint_recovery_async(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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

    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    # First attempt should fail
    with pytest.raises(RuntimeError):
        await graph.ainvoke(
            {"steps": ["start"], "attempt": 1},
            config,
            durability=durability,
        )

    # Verify checkpoint state
    state = await graph.aget_state(config)
    assert state is not None
    assert state.values == {"steps": ["start"], "attempt": 1}  # input state saved
    assert state.next == ("node1",)  # Should retry failed node

    # Retry with updated attempt count
    result = await graph.ainvoke(
        {"steps": [], "attempt": 2}, config, durability=durability
    )
    assert result == {"steps": ["start", "node1", "node2"], "attempt": 2}

    # Verify checkpoint history shows both attempts
    history = [c async for c in graph.aget_state_history(config)]
    if durability != "exit":
        assert len(history) == 6  # Initial + failed attempt + successful attempt
    else:
        assert len(history) == 2  # error + success

    # Verify the error was recorded in checkpoint
    failed_checkpoint = next(c for c in history if c.tasks and c.tasks[0].error)
    assert "RuntimeError('Simulated failure')" in failed_checkpoint.tasks[0].error


async def test_multiple_updates_root() -> None:
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

    assert await graph.ainvoke("") == "a1a2b"

    # only streams the last update from node_a
    assert [c async for c in graph.astream("", stream_mode="updates")] == [
        {"node_a": ["a1", "a2"]},
        {"node_b": "b"},
    ]


async def test_multiple_updates() -> None:
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

    assert await graph.ainvoke({"foo": ""}) == {
        "foo": "a1a2b",
    }

    # only streams the last update from node_a
    assert [c async for c in graph.astream({"foo": ""}, stream_mode="updates")] == [
        {"node_a": [{"foo": "a1"}, {"foo": "a2"}]},
        {"node_b": {"foo": "b"}},
    ]


@NEEDS_CONTEXTVARS
async def test_falsy_return_from_task(async_checkpointer: BaseCheckpointSaver) -> None:
    """Test with a falsy return from a task."""

    @task
    async def falsy_task() -> bool:
        return False

    @entrypoint(checkpointer=async_checkpointer)
    async def graph(state: dict) -> dict:
        """React tool."""
        await falsy_task()
        interrupt("test")

    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    await graph.ainvoke({"a": 5}, configurable)
    await graph.ainvoke(Command(resume="123"), configurable)


@NEEDS_CONTEXTVARS
async def test_multiple_interrupts_functional(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test multiple interrupts with functional API."""
    from langgraph.func import entrypoint, task

    counter = 0

    @task
    async def double(x: int) -> int:
        """Increment the counter."""
        nonlocal counter
        counter += 1
        return 2 * x

    @entrypoint(checkpointer=async_checkpointer)
    async def graph(state: dict) -> dict:
        """React tool."""

        values = []

        for idx in [1, 2, 3]:
            values.extend([await double(idx), interrupt({"a": "boo"})])

        return {"values": values}

    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    await graph.ainvoke({}, configurable)
    await graph.ainvoke(Command(resume="a"), configurable)
    await graph.ainvoke(Command(resume="b"), configurable)
    result = await graph.ainvoke(Command(resume="c"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 4, "b", 6, "c"],
    }
    assert counter == 3


@NEEDS_CONTEXTVARS
async def test_multiple_interrupts_functional_cache(
    async_checkpointer: BaseCheckpointSaver, cache: BaseCache
):
    """Test multiple interrupts with functional API."""
    counter = 0

    @task(cache_policy=CachePolicy())
    def double(x: int) -> int:
        """Increment the counter."""
        nonlocal counter
        counter += 1
        return 2 * x

    @entrypoint(checkpointer=async_checkpointer, cache=cache)
    def graph(state: dict) -> dict:
        """React tool."""

        values = []

        for idx in [1, 1, 2, 2, 3, 3]:
            values.extend([double(idx).result(), interrupt({"a": "boo"})])

        return {"values": values}

    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    await graph.ainvoke({}, configurable)
    await graph.ainvoke(Command(resume="a"), configurable)
    await graph.ainvoke(Command(resume="b"), configurable)
    await graph.ainvoke(Command(resume="c"), configurable)
    await graph.ainvoke(Command(resume="d"), configurable)
    await graph.ainvoke(Command(resume="e"), configurable)
    result = await graph.ainvoke(Command(resume="f"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 2, "b", 4, "c", 4, "d", 6, "e", 6, "f"],
    }
    assert counter == 3

    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    await graph.ainvoke({}, configurable)
    await graph.ainvoke(Command(resume="a"), configurable)
    await graph.ainvoke(Command(resume="b"), configurable)
    await graph.ainvoke(Command(resume="c"), configurable)
    await graph.ainvoke(Command(resume="d"), configurable)
    await graph.ainvoke(Command(resume="e"), configurable)
    result = await graph.ainvoke(Command(resume="f"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 2, "b", 4, "c", 4, "d", 6, "e", 6, "f"],
    }
    assert counter == 3

    # clear the cache
    await double.aclear_cache(cache)

    # now should recompute
    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    await graph.ainvoke({}, configurable)
    await graph.ainvoke(Command(resume="a"), configurable)
    await graph.ainvoke(Command(resume="b"), configurable)
    await graph.ainvoke(Command(resume="c"), configurable)
    await graph.ainvoke(Command(resume="d"), configurable)
    await graph.ainvoke(Command(resume="e"), configurable)
    result = await graph.ainvoke(Command(resume="f"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 2, "b", 4, "c", 4, "d", 6, "e", 6, "f"],
    }
    assert counter == 6


@NEEDS_CONTEXTVARS
async def test_double_interrupt_subgraph(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
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
    subgraph = subgraph_builder.compile(checkpointer=async_checkpointer)
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    assert [c async for c in subgraph.astream({"input": "test"}, thread)] == [
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
    assert [c async for c in subgraph.astream(Command(resume="123"), thread)] == [
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
    assert [c async for c in subgraph.astream(Command(resume="123"), thread)] == [
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
        .compile(checkpointer=async_checkpointer)
    )

    assert [c async for c in parent_agent.astream({"input": "test"}, thread)] == [
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
    assert [c async for c in parent_agent.astream(Command(resume=True), thread)] == [
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
    assert [c async for c in parent_agent.astream(Command(resume=True), thread)] == [
        {
            "invoke_sub_agent": {"input": True},
        },
    ]


@NEEDS_CONTEXTVARS
async def test_async_streaming_with_functional_api() -> None:
    """Test streaming with functional API.

    This test verifies that we're able to stream results as they're being generated
    rather than have all the results arrive at once after the graph has completed.

    The time of arrival between the two updates corresponding to the two `slow` tasks
    should be greater than the time delay between the two tasks.
    """

    time_delay = 0.01

    @task()
    async def slow() -> dict:
        await asyncio.sleep(time_delay)  # Simulate a delay of 10 ms
        return {"tic": asyncio.get_running_loop().time()}

    @entrypoint()
    async def graph(inputs: dict) -> list:
        first = await slow()
        second = await slow()
        return [first, second]

    arrival_times = []

    async for chunk in graph.astream({}):
        if "slow" not in chunk:  # We'll just look at the updates from `slow`
            continue
        arrival_times.append(asyncio.get_running_loop().time())

    assert len(arrival_times) == 2
    delta = arrival_times[1] - arrival_times[0]
    # Delta cannot be less than 10 ms if it is streaming as results are generated.
    assert delta > time_delay


@NEEDS_CONTEXTVARS
async def test_multiple_subgraphs(async_checkpointer: BaseCheckpointSaver) -> None:
    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define the subgraphs
    async def add(state):
        return {"result": state["a"] + state["b"]}

    add_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(add)
        .add_edge(START, "add")
        .compile()
    )

    async def multiply(state):
        return {"result": state["a"] * state["b"]}

    multiply_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(multiply)
        .add_edge(START, "multiply")
        .compile()
    )

    # Test calling the same subgraph multiple times
    async def call_same_subgraph(state):
        result = await add_subgraph.ainvoke(state)
        another_result = await add_subgraph.ainvoke({"a": result["result"], "b": 10})
        return another_result

    parent_call_same_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(call_same_subgraph)
        .add_edge(START, "call_same_subgraph")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}
    assert await parent_call_same_subgraph.ainvoke({"a": 2, "b": 3}, config) == {
        "result": 15
    }

    # Test calling multiple subgraphs
    class Output(TypedDict):
        add_result: int
        multiply_result: int

    async def call_multiple_subgraphs(state):
        add_result = await add_subgraph.ainvoke(state)
        multiply_result = await multiply_subgraph.ainvoke(state)
        return {
            "add_result": add_result["result"],
            "multiply_result": multiply_result["result"],
        }

    parent_call_multiple_subgraphs = (
        StateGraph(State, output_schema=Output)
        .add_node(call_multiple_subgraphs)
        .add_edge(START, "call_multiple_subgraphs")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "2"}}
    assert await parent_call_multiple_subgraphs.ainvoke({"a": 2, "b": 3}, config) == {
        "add_result": 5,
        "multiply_result": 6,
    }


@NEEDS_CONTEXTVARS
async def test_multiple_subgraphs_functional(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    # Define addition subgraph
    @entrypoint()
    async def add(inputs):
        a, b = inputs
        return a + b

    # Define multiplication subgraph using tasks
    @task
    async def multiply_task(a, b):
        return a * b

    @entrypoint()
    async def multiply(inputs):
        return await multiply_task(*inputs)

    # Test calling the same subgraph multiple times
    @task
    async def call_same_subgraph(a, b):
        result = await add.ainvoke([a, b])
        another_result = await add.ainvoke([result, 10])
        return another_result

    @entrypoint(checkpointer=async_checkpointer)
    async def parent_call_same_subgraph(inputs):
        return await call_same_subgraph(*inputs)

    config = {"configurable": {"thread_id": "1"}}
    assert await parent_call_same_subgraph.ainvoke([2, 3], config) == 15

    # Test calling multiple subgraphs
    @task
    async def call_multiple_subgraphs(a, b):
        add_result = await add.ainvoke([a, b])
        multiply_result = await multiply.ainvoke([a, b])
        return [add_result, multiply_result]

    @entrypoint(checkpointer=async_checkpointer)
    async def parent_call_multiple_subgraphs(inputs):
        return await call_multiple_subgraphs(*inputs)

    config = {"configurable": {"thread_id": "2"}}
    assert await parent_call_multiple_subgraphs.ainvoke([2, 3], config) == [5, 6]


@NEEDS_CONTEXTVARS
async def test_multiple_subgraphs_mixed_entrypoint(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test calling multiple StateGraph subgraphs from an entrypoint."""

    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define the subgraphs
    async def add(state):
        return {"result": state["a"] + state["b"]}

    add_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(add)
        .add_edge(START, "add")
        .compile()
    )

    async def multiply(state):
        return {"result": state["a"] * state["b"]}

    multiply_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(multiply)
        .add_edge(START, "multiply")
        .compile()
    )

    # Test calling the same subgraph multiple times
    @task
    async def call_same_subgraph(a, b):
        result = (await add_subgraph.ainvoke({"a": a, "b": b}))["result"]
        another_result = (await add_subgraph.ainvoke({"a": result, "b": 10}))["result"]
        return another_result

    @entrypoint(checkpointer=async_checkpointer)
    async def parent_call_same_subgraph(inputs):
        return await call_same_subgraph(*inputs)

    config = {"configurable": {"thread_id": "1"}}
    assert await parent_call_same_subgraph.ainvoke([2, 3], config) == 15

    # Test calling multiple subgraphs
    @task
    async def call_multiple_subgraphs(a, b):
        add_result = (await add_subgraph.ainvoke({"a": a, "b": b}))["result"]
        multiply_result = (await multiply_subgraph.ainvoke({"a": a, "b": b}))["result"]
        return [add_result, multiply_result]

    @entrypoint(checkpointer=async_checkpointer)
    async def parent_call_multiple_subgraphs(inputs):
        return await call_multiple_subgraphs(*inputs)

    config = {"configurable": {"thread_id": "2"}}
    assert await parent_call_multiple_subgraphs.ainvoke([2, 3], config) == [5, 6]


@NEEDS_CONTEXTVARS
async def test_multiple_subgraphs_mixed_state_graph(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test calling multiple entrypoint "subgraphs" from a StateGraph."""

    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define addition subgraph
    @entrypoint()
    async def add(inputs):
        a, b = inputs
        return a + b

    # Define multiplication subgraph using tasks
    @task
    async def multiply_task(a, b):
        return a * b

    @entrypoint()
    async def multiply(inputs):
        return await multiply_task(*inputs)

    # Test calling the same subgraph multiple times
    async def call_same_subgraph(state):
        result = await add.ainvoke([state["a"], state["b"]])
        another_result = await add.ainvoke([result, 10])
        return {"result": another_result}

    parent_call_same_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(call_same_subgraph)
        .add_edge(START, "call_same_subgraph")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}
    assert await parent_call_same_subgraph.ainvoke({"a": 2, "b": 3}, config) == {
        "result": 15
    }

    # Test calling multiple subgraphs
    class Output(TypedDict):
        add_result: int
        multiply_result: int

    async def call_multiple_subgraphs(state):
        add_result = await add.ainvoke([state["a"], state["b"]])
        multiply_result = await multiply.ainvoke([state["a"], state["b"]])
        return {
            "add_result": add_result,
            "multiply_result": multiply_result,
        }

    parent_call_multiple_subgraphs = (
        StateGraph(State, output_schema=Output)
        .add_node(call_multiple_subgraphs)
        .add_edge(START, "call_multiple_subgraphs")
        .compile(checkpointer=async_checkpointer)
    )
    config = {"configurable": {"thread_id": "2"}}
    assert await parent_call_multiple_subgraphs.ainvoke({"a": 2, "b": 3}, config) == {
        "add_result": 5,
        "multiply_result": 6,
    }


@NEEDS_CONTEXTVARS
async def test_multiple_subgraphs_checkpointer(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    class SubgraphState(TypedDict):
        sub_counter: Annotated[int, operator.add]

    async def subgraph_node(state):
        return {"sub_counter": 2}

    sub_graph_1 = (
        StateGraph(SubgraphState)
        .add_node(subgraph_node)
        .add_edge(START, "subgraph_node")
        .compile(checkpointer=True)
    )

    class OtherSubgraphState(TypedDict):
        other_sub_counter: Annotated[int, operator.add]

    async def other_subgraph_node(state):
        return {"other_sub_counter": 3}

    sub_graph_2 = (
        StateGraph(OtherSubgraphState)
        .add_node(other_subgraph_node)
        .add_edge(START, "other_subgraph_node")
        .compile()
    )

    class ParentState(TypedDict):
        parent_counter: int

    async def parent_node(state):
        result = await sub_graph_1.ainvoke({"sub_counter": state["parent_counter"]})
        other_result = await sub_graph_2.ainvoke(
            {"other_sub_counter": result["sub_counter"]}
        )
        return {"parent_counter": other_result["other_sub_counter"]}

    parent_graph = (
        StateGraph(ParentState)
        .add_node(parent_node)
        .add_edge(START, "parent_node")
        .compile(checkpointer=async_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    assert await parent_graph.ainvoke({"parent_counter": 0}, config) == {
        "parent_counter": 5
    }
    assert await parent_graph.ainvoke({"parent_counter": 0}, config) == {
        "parent_counter": 7
    }
    config = {"configurable": {"thread_id": "2"}}
    assert [
        c
        async for c in parent_graph.astream(
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
        async for c in parent_graph.astream(
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


@NEEDS_CONTEXTVARS
async def test_async_entrypoint_without_checkpointer() -> None:
    """Test no checkpointer."""
    states = []
    config = {"configurable": {"thread_id": "1"}}

    # Test without previous
    @entrypoint()
    async def foo(inputs: Any) -> Any:
        states.append(inputs)
        return inputs

    assert (await foo.ainvoke({"a": "1"}, config)) == {"a": "1"}

    @entrypoint()
    async def foo(inputs: Any, *, previous: Any) -> Any:
        states.append(previous)
        return {"previous": previous, "current": inputs}

    assert (await foo.ainvoke({"a": "1"}, config)) == {
        "current": {"a": "1"},
        "previous": None,
    }
    assert (await foo.ainvoke({"a": "1"}, config)) == {
        "current": {"a": "1"},
        "previous": None,
    }


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


async def test_entrypoint_stateful(async_checkpointer: BaseCheckpointSaver) -> None:
    """Test stateful entrypoint invoke."""

    # Test invoke
    states = []

    @entrypoint(checkpointer=async_checkpointer)
    async def foo(inputs: Any, *, previous: Any) -> Any:
        states.append(previous)
        return {"previous": previous, "current": inputs}

    config = {"configurable": {"thread_id": "1"}}

    assert await foo.ainvoke({"a": "1"}, config) == {
        "current": {"a": "1"},
        "previous": None,
    }
    assert await foo.ainvoke({"a": "2"}, config) == {
        "current": {"a": "2"},
        "previous": {"current": {"a": "1"}, "previous": None},
    }
    assert await foo.ainvoke({"a": "3"}, config) == {
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
    @entrypoint(checkpointer=async_checkpointer)
    async def foo(inputs, *, previous: Any) -> Any:
        return {"previous": previous, "current": inputs}

    config = {"configurable": {"thread_id": "2"}}
    items = [item async for item in foo.astream({"a": "1"}, config)]
    assert items == [{"foo": {"current": {"a": "1"}, "previous": None}}]


async def test_entrypoint_stateful_update_state(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test stateful entrypoint invoke."""

    # Test invoke
    states = []

    @entrypoint(checkpointer=async_checkpointer)
    async def foo(inputs: Any, *, previous: Any) -> Any:
        states.append(previous)
        return {"previous": previous, "current": inputs}

    config = {"configurable": {"thread_id": "1"}}

    # assert print(foo.input_channels)
    await foo.aupdate_state(config, {"a": "-1"})
    assert await foo.ainvoke({"a": "1"}, config) == {
        "current": {"a": "1"},
        "previous": {"a": "-1"},
    }
    assert await foo.ainvoke({"a": "2"}, config) == {
        "current": {"a": "2"},
        "previous": {"current": {"a": "1"}, "previous": {"a": "-1"}},
    }
    assert await foo.ainvoke({"a": "3"}, config) == {
        "current": {"a": "3"},
        "previous": {
            "current": {"a": "2"},
            "previous": {"current": {"a": "1"}, "previous": {"a": "-1"}},
        },
    }

    # update state
    await foo.aupdate_state(config, {"a": "3"})

    # Test stream
    assert [item async for item in foo.astream({"a": "1"}, config)] == [
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


async def test_entrypoint_from_async_generator() -> None:
    """@entrypoint does not support async generators."""
    with pytest.raises(NotImplementedError):

        @entrypoint()
        async def foo(inputs) -> Any:
            yield "a"
            yield "b"


@NEEDS_CONTEXTVARS
async def test_named_tasks_functional() -> None:
    class Foo:
        async def foo(self, value: str) -> dict:
            return value + "foo"

    f = Foo()

    # class method task
    foo = task(f.foo, name="custom_foo")
    other_foo = task(f.foo, name="other_foo")

    # regular function task
    @task(name="custom_bar")
    async def bar(value: str) -> dict:
        return value + "|bar"

    async def baz(update: str, value: str) -> dict:
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
    async def workflow(inputs: dict) -> dict:
        foo_result = await foo(inputs)
        await other_foo(inputs)
        bar_result = await bar(foo_result)
        baz_result = await baz_task(bar_result)
        custom_baz_result = await custom_baz_task(baz_result)
        qux_result = await qux_task(custom_baz_result)
        return qux_result

    assert [c async for c in workflow.astream("", stream_mode="updates")] == [
        {"custom_foo": "foo"},
        {"other_foo": "foo"},
        {"custom_bar": "foo|bar"},
        {"baz": "foo|bar|baz"},
        {"custom_baz": "foo|bar|baz|custom_baz"},
        {"qux": "foo|bar|baz|custom_baz|qux"},
        {"workflow": "foo|bar|baz|custom_baz|qux"},
    ]


@NEEDS_CONTEXTVARS
async def test_overriding_injectable_args_with_async_task(
    async_store: BaseStore,
) -> None:
    """Test overriding injectable args in tasks."""

    @task
    async def foo(store: BaseStore, writer: StreamWriter, value: Any) -> None:
        assert store is value
        assert writer is value

    @entrypoint(store=async_store)
    async def main(inputs, store: BaseStore) -> str:
        assert store is not None
        await foo(store=None, writer=None, value=None)
        await foo(store="hello", writer="hello", value="hello")
        return "OK"

    assert await main.ainvoke({}) == "OK"


async def test_tags_stream_mode_messages() -> None:
    model = GenericFakeChatModel(messages=iter(["foo"]), tags=["meow"])

    async def call_model(state, config):
        return {"messages": await model.ainvoke(state["messages"], config)}

    graph = (
        StateGraph(MessagesState)
        .add_node(call_model)
        .add_edge(START, "call_model")
        .compile()
    )
    assert [
        c
        async for c in graph.astream(
            {
                "messages": "hi",
            },
            stream_mode="messages",
        )
    ] == [
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


async def test_stream_mode_messages_command() -> None:
    from langchain_core.messages import HumanMessage

    async def my_node(state):
        return {"messages": HumanMessage(content="foo")}

    async def my_other_node(state):
        return Command(update={"messages": HumanMessage(content="bar")})

    graph = (
        StateGraph(MessagesState)
        .add_sequence([my_node, my_other_node])
        .add_edge(START, "my_node")
        .compile()
    )
    assert [
        c
        async for c in graph.astream(
            {
                "messages": [],
            },
            stream_mode="messages",
        )
    ] == [
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
    ]


async def test_stream_messages_dedupe_inputs() -> None:
    from langchain_core.messages import AIMessage

    async def call_model(state):
        return {"messages": AIMessage("hi", id="1")}

    async def route(state):
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
        async for ns, chunk in graph.astream(
            {"messages": "hi"}, stream_mode="messages", subgraphs=True
        )
    ]

    assert len(chunks) == 1
    assert chunks[0][0] == AIMessage("hi", id="1")
    assert chunks[0][1]["langgraph_node"] == "call_model"


async def test_stream_messages_dedupe_state(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    from langchain_core.messages import AIMessage

    to_emit = [AIMessage("bye", id="1"), AIMessage("bye again", id="2")]

    async def call_model(state):
        return {"messages": to_emit.pop(0)}

    async def route(state):
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
        .compile(checkpointer=async_checkpointer)
    )

    thread1 = {"configurable": {"thread_id": "1"}}

    chunks = [
        chunk
        async for ns, chunk in graph.astream(
            {"messages": "hi"}, thread1, stream_mode="messages", subgraphs=True
        )
    ]

    assert len(chunks) == 1
    assert chunks[0][0] == AIMessage("bye", id="1")
    assert chunks[0][1]["langgraph_node"] == "call_model"

    chunks = [
        chunk
        async for ns, chunk in graph.astream(
            {"messages": "hi again"},
            thread1,
            stream_mode="messages",
            subgraphs=True,
        )
    ]

    assert len(chunks) == 1
    assert chunks[0][0] == AIMessage("bye again", id="2")
    assert chunks[0][1]["langgraph_node"] == "call_model"


@NEEDS_CONTEXTVARS
async def test_interrupt_subgraph_reenter_checkpointer_true(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    class SubgraphState(TypedDict):
        foo: str
        bar: str

    class ParentState(TypedDict):
        foo: str
        counter: int

    called = []
    bar_values = []

    async def subnode_1(state: SubgraphState):
        called.append("subnode_1")
        bar_values.append(state.get("bar"))
        return {"foo": "subgraph_1"}

    async def subnode_2(state: SubgraphState):
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

    async def call_subgraph(state: ParentState):
        called.append("call_subgraph")
        return await subgraph.ainvoke(state)

    async def node(state: ParentState):
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
        .compile(checkpointer=async_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    assert await parent.ainvoke({"foo": "", "counter": 0}, config) == {
        "foo": "",
        "counter": 0,
        "__interrupt__": [
            Interrupt(
                value="Provide value",
                id=AnyStr(),
            )
        ],
    }
    assert await parent.ainvoke(Command(resume="bar"), config) == {
        "foo": "subgraph_2",
        "counter": 1,
        "__interrupt__": [
            Interrupt(
                value="Provide value",
                id=AnyStr(),
            )
        ],
    }
    assert await parent.ainvoke(Command(resume="qux"), config) == {
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
    assert await parent.ainvoke({"foo": "meow", "counter": 0}, config) == {
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


@NEEDS_CONTEXTVARS
async def test_handles_multiple_interrupts_from_tasks(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    @task
    async def add_participant(name: str) -> str:
        feedback = interrupt(f"Hey do you want to add {name}?")

        if feedback is False:
            return f"The user changed their mind and doesn't want to add {name}!"

        if feedback is True:
            return f"Added {name}!"

        raise ValueError("Invalid feedback")

    @entrypoint(checkpointer=async_checkpointer)
    async def program(_state: Any) -> list[str]:
        first = await add_participant("James")
        second = await add_participant("Will")
        return [first, second]

    config = {"configurable": {"thread_id": "1"}}

    result = await program.ainvoke("this is ignored", config=config)
    assert result == {
        "__interrupt__": [
            Interrupt(
                value="Hey do you want to add James?",
                id=AnyStr(),
            ),
        ]
    }

    state = await program.aget_state(config=config)
    assert len(state.tasks[0].interrupts) == 1
    task_interrupt = state.tasks[0].interrupts[0]
    assert task_interrupt.value == "Hey do you want to add James?"

    result = await program.ainvoke(Command(resume=True), config=config)
    assert result == {
        "__interrupt__": [
            Interrupt(
                value="Hey do you want to add Will?",
                id=AnyStr(),
            ),
        ]
    }

    state = await program.aget_state(config=config)
    assert len(state.tasks[0].interrupts) == 1
    task_interrupt = state.tasks[0].interrupts[0]
    assert task_interrupt.value == "Hey do you want to add Will?"

    result = await program.ainvoke(Command(resume=True), config=config)
    assert result is not None
    assert len(result) == 2
    assert result[0] == "Added James!"
    assert result[1] == "Added Will!"


@NEEDS_CONTEXTVARS
async def test_interrupts_in_tasks_surfaced_once(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    @task
    async def add_participant(name: str) -> str:
        feedback = interrupt(f"Hey do you want to add {name}?")

        if feedback is False:
            return f"The user changed their mind and doesn't want to add {name}!"

        if feedback is True:
            return f"Added {name}!"

        raise ValueError("Invalid feedback")

    @entrypoint(checkpointer=async_checkpointer)
    async def program(_state: Any) -> list[str]:
        first = await add_participant("James")
        second = await add_participant("Will")
        return [first, second]

    config = {"configurable": {"thread_id": "1"}}

    interrupts = [
        e
        async for e in program.astream("this is ignored", config=config)
        if "__interrupt__" in e
    ]
    assert len(interrupts) == 1

    state = await program.aget_state(config=config)
    assert len(state.tasks[0].interrupts) == 1
    task_interrupt = state.tasks[0].interrupts[0]
    assert task_interrupt.value == "Hey do you want to add James?"

    interrupts = [
        e
        async for e in program.astream(Command(resume=True), config=config)
        if "__interrupt__" in e
    ]
    assert len(interrupts) == 1

    state = await program.aget_state(config=config)
    assert len(state.tasks[0].interrupts) == 1
    task_interrupt = state.tasks[0].interrupts[0]
    assert task_interrupt.value == "Hey do you want to add Will?"

    result = await program.ainvoke(Command(resume=True), config=config)
    assert result is not None
    assert len(result) == 2
    assert result[0] == "Added James!"
    assert result[1] == "Added Will!"


async def test_pregel_loop_refcount():
    gc.collect()
    try:
        gc.disable()

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        graph_builder = StateGraph(State)

        async def chatbot(state: State):
            return {"messages": [("ai", "HIYA")]}

        graph_builder.add_node("chatbot", chatbot)
        graph_builder.set_entry_point("chatbot")
        graph_builder.set_finish_point("chatbot")
        graph = graph_builder.compile()

        for _ in range(5):
            await graph.ainvoke({"messages": [{"role": "user", "content": "hi"}]})
            assert (
                len(
                    [
                        obj
                        for obj in gc.get_objects()
                        if isinstance(obj, AsyncPregelLoop)
                    ]
                )
                == 0
            )
            assert (
                len([obj for obj in gc.get_objects() if isinstance(obj, PregelRunner)])
                == 0
            )
    finally:
        gc.enable()


async def test_bulk_state_updates(async_checkpointer: BaseCheckpointSaver) -> None:
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
        .compile(checkpointer=async_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # First update with node_a
    await graph.abulk_update_state(
        config,
        [
            [
                StateUpdate({"foo": "bar"}, "node_a"),
            ]
        ],
    )

    # Then bulk update with both nodes
    await graph.abulk_update_state(
        config,
        [
            [
                StateUpdate({"foo": "updated"}, "node_a"),
                StateUpdate({"baz": "new"}, "node_b"),
            ]
        ],
    )

    state = await graph.aget_state(config)
    assert state.values == {"foo": "updated", "baz": "new"}

    # Check if there are only two checkpoints
    checkpoints = [
        c async for c in async_checkpointer.alist({"configurable": {"thread_id": "1"}})
    ]
    assert len(checkpoints) == 2

    # perform multiple steps at the same time
    config = {"configurable": {"thread_id": "2"}}

    await graph.abulk_update_state(
        config,
        [
            [
                StateUpdate({"foo": "bar"}, "node_a"),
            ],
            [
                StateUpdate({"foo": "updated"}, "node_a"),
                StateUpdate({"baz": "new"}, "node_b"),
            ],
        ],
    )

    state = await graph.aget_state(config)
    assert state.values == {"foo": "updated", "baz": "new"}

    checkpoints = [
        c async for c in async_checkpointer.alist({"configurable": {"thread_id": "1"}})
    ]
    assert len(checkpoints) == 2

    # Should raise error if updating without as_node
    with pytest.raises(InvalidUpdateError):
        await graph.abulk_update_state(
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
        await graph.abulk_update_state(config, [])

    # Should raise if no updates are provided
    with pytest.raises(ValueError, match="No updates provided"):
        await graph.abulk_update_state(config, [[], []])

    # Should raise if __end__ or __copy__ update is applied in bulk
    with pytest.raises(InvalidUpdateError):
        await graph.abulk_update_state(
            config,
            [
                [
                    StateUpdate(values=None, as_node="__end__"),
                    StateUpdate(values=None, as_node="__copy__"),
                ],
            ],
        )


async def test_update_as_input(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
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
        .compile(checkpointer=async_checkpointer)
    )

    assert await graph.ainvoke(
        {"foo": "input"},
        {"configurable": {"thread_id": "1"}},
        durability=durability,
    ) == {"foo": "tool"}

    assert await graph.ainvoke(
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
        async for s in graph.aget_state_history({"configurable": {"thread_id": "1"}})
    ]

    await graph.abulk_update_state(
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

    state = await graph.aget_state({"configurable": {"thread_id": "2"}})
    assert state.values == {"foo": "tool"}

    new_history = [
        map_snapshot(s)
        async for s in graph.aget_state_history({"configurable": {"thread_id": "2"}})
    ]

    if durability != "exit":
        assert new_history == history
    else:
        assert [new_history[0], new_history[4]] == history


async def test_batch_update_as_input(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
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
        .compile(checkpointer=async_checkpointer)
    )

    assert await graph.ainvoke(
        {"foo": "input"},
        {"configurable": {"thread_id": "1"}},
        durability=durability,
    ) == {"foo": "map", "tasks": [0, 1, 2]}

    def map_snapshot(i: StateSnapshot) -> dict:
        return {
            "values": i.values,
            "next": i.next,
            "step": i.metadata.get("step"),
            "tasks": [t.name for t in i.tasks],
        }

    history = [
        map_snapshot(s)
        async for s in graph.aget_state_history({"configurable": {"thread_id": "1"}})
    ]

    await graph.abulk_update_state(
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

    state = await graph.aget_state({"configurable": {"thread_id": "2"}})
    assert state.values == {"foo": "map", "tasks": [0, 1, 2]}

    new_history = [
        map_snapshot(s)
        async for s in graph.aget_state_history({"configurable": {"thread_id": "2"}})
    ]

    if durability != "exit":
        assert new_history == history
    else:
        assert new_history[:1] == history


async def test_draw_invalid():
    from langchain_core.messages import BaseMessage

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    workflow = StateGraph(AgentState)

    async def call_model(state: AgentState) -> AgentState:
        return state

    async def call_tool(state: AgentState) -> AgentState:
        return state

    async def do_nothing(state: AgentState) -> AgentState:
        return state

    def should_continue(state):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.content.startswith("end"):
            return END
        else:
            return [Send("tool", last_message), Send("nothing", last_message)]

    workflow.add_node("agent", call_model)
    workflow.add_node("tool", call_tool)
    workflow.add_node("nothing", do_nothing)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        path_map=["tool", "nothing", END],
    )
    workflow.add_edge("tool", "agent")

    graph = workflow.compile()

    assert graph.get_graph().to_json() == {
        "nodes": [
            {
                "id": "__start__",
                "type": "runnable",
                "data": {
                    "id": ["langgraph", "_internal", "_runnable", "RunnableCallable"],
                    "name": "__start__",
                },
            },
            {
                "id": "agent",
                "type": "runnable",
                "data": {
                    "id": ["langgraph", "_internal", "_runnable", "RunnableCallable"],
                    "name": "agent",
                },
            },
            {
                "id": "tool",
                "type": "runnable",
                "data": {
                    "id": ["langgraph", "_internal", "_runnable", "RunnableCallable"],
                    "name": "tool",
                },
            },
            {
                "id": "nothing",
                "type": "runnable",
                "data": {
                    "id": ["langgraph", "_internal", "_runnable", "RunnableCallable"],
                    "name": "nothing",
                },
            },
            {"id": "__end__"},
        ],
        "edges": [
            {"source": "__start__", "target": "agent"},
            {"source": "agent", "target": "__end__", "conditional": True},
            {"source": "agent", "target": "nothing", "conditional": True},
            {"source": "agent", "target": "tool", "conditional": True},
            {"source": "tool", "target": "agent"},
            {"source": "nothing", "target": "__end__"},
        ],
    }


@NEEDS_CONTEXTVARS
async def test_imp_exception(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    @task()
    async def my_task(number: int):
        await asyncio.sleep(0.1)
        return number * 2

    @task()
    async def task_with_exception(number: int):
        await asyncio.sleep(0.1)
        raise Exception("This is a test exception")

    @entrypoint(checkpointer=async_checkpointer)
    async def my_workflow(number: int):
        await my_task(number)
        try:
            await task_with_exception(number)
        except Exception as e:
            print(f"Exception caught: {e}")
        await my_task(number)
        return "done"

    thread1 = {"configurable": {"thread_id": "1"}}
    assert await my_workflow.ainvoke(1, thread1) == "done"

    assert [c async for c in my_workflow.astream(1, thread1)] == [
        {"my_task": 2},
        {"my_task": 2},
        {"my_workflow": "done"},
    ]

    assert [c async for c in my_workflow.astream_events(1, thread1)] == [
        {
            "event": "on_chain_start",
            "data": {"input": 1},
            "name": "LangGraph",
            "tags": [],
            "run_id": AnyStr(),
            "metadata": {"thread_id": "1"},
            "parent_ids": [],
        },
        {
            "event": "on_chain_start",
            "data": {"input": 1},
            "name": "my_workflow",
            "tags": ["graph:step:4"],
            "run_id": AnyStr(),
            "metadata": {
                "thread_id": "1",
                "langgraph_step": 4,
                "langgraph_node": "my_workflow",
                "langgraph_triggers": ("__start__",),
                "langgraph_path": ("__pregel_pull", "my_workflow"),
                "langgraph_checkpoint_ns": AnyStr(),
            },
            "parent_ids": [AnyStr()],
        },
        {
            "event": "on_chain_start",
            "data": {"input": {"number": 1}},
            "name": "my_task",
            "tags": ["seq:step:1"],
            "run_id": AnyStr(),
            "metadata": {
                "thread_id": "1",
                "langgraph_step": 4,
                "langgraph_node": "my_task",
                "langgraph_triggers": ("__pregel_push",),
                "langgraph_path": (
                    "__pregel_push",
                    ("__pregel_pull", "my_workflow"),
                    2,
                    True,
                ),
                "langgraph_checkpoint_ns": AnyStr(),
            },
            "parent_ids": [
                AnyStr(),
                AnyStr(),
            ],
        },
        {
            "event": "on_chain_stream",
            "run_id": AnyStr(),
            "name": "my_task",
            "tags": ["seq:step:1"],
            "metadata": {
                "thread_id": "1",
                "langgraph_step": 4,
                "langgraph_node": "my_task",
                "langgraph_triggers": ("__pregel_push",),
                "langgraph_path": (
                    "__pregel_push",
                    ("__pregel_pull", "my_workflow"),
                    2,
                    True,
                ),
                "langgraph_checkpoint_ns": AnyStr(),
            },
            "data": {"chunk": 2},
            "parent_ids": [
                AnyStr(),
                AnyStr(),
            ],
        },
        {
            "event": "on_chain_end",
            "data": {"output": 2, "input": {"number": 1}},
            "run_id": AnyStr(),
            "name": "my_task",
            "tags": ["seq:step:1"],
            "metadata": {
                "thread_id": "1",
                "langgraph_step": 4,
                "langgraph_node": "my_task",
                "langgraph_triggers": ("__pregel_push",),
                "langgraph_path": (
                    "__pregel_push",
                    ("__pregel_pull", "my_workflow"),
                    2,
                    True,
                ),
                "langgraph_checkpoint_ns": AnyStr(),
            },
            "parent_ids": [
                AnyStr(),
                AnyStr(),
            ],
        },
        {
            "event": "on_chain_stream",
            "run_id": AnyStr(),
            "name": "LangGraph",
            "tags": [],
            "metadata": {"thread_id": "1"},
            "data": {"chunk": {"my_task": 2}},
            "parent_ids": [],
        },
        {
            "event": "on_chain_start",
            "data": {"input": {"number": 1}},
            "name": "task_with_exception",
            "tags": ["seq:step:1"],
            "run_id": AnyStr(),
            "metadata": {
                "thread_id": "1",
                "langgraph_step": 4,
                "langgraph_node": "my_task",
                "langgraph_triggers": ("__pregel_push",),
                "langgraph_path": (
                    "__pregel_push",
                    ("__pregel_pull", "my_workflow"),
                    2,
                    True,
                ),
                "langgraph_checkpoint_ns": AnyStr(),
            },
            "parent_ids": [
                AnyStr(),
                AnyStr(),
            ],
        },
        {
            "event": "on_chain_start",
            "data": {"input": {"number": 1}},
            "name": "my_task",
            "tags": ["seq:step:1"],
            "run_id": AnyStr(),
            "metadata": {
                "thread_id": "1",
                "langgraph_step": 4,
                "langgraph_node": "my_task",
                "langgraph_triggers": ("__pregel_push",),
                "langgraph_path": (
                    "__pregel_push",
                    ("__pregel_pull", "my_workflow"),
                    2,
                    True,
                ),
                "langgraph_checkpoint_ns": AnyStr(),
            },
            "parent_ids": [
                AnyStr(),
                AnyStr(),
            ],
        },
        {
            "event": "on_chain_stream",
            "run_id": AnyStr(),
            "name": "my_task",
            "tags": ["seq:step:1"],
            "metadata": {
                "thread_id": "1",
                "langgraph_step": 4,
                "langgraph_node": "my_task",
                "langgraph_triggers": ("__pregel_push",),
                "langgraph_path": (
                    "__pregel_push",
                    ("__pregel_pull", "my_workflow"),
                    2,
                    True,
                ),
                "langgraph_checkpoint_ns": AnyStr(),
            },
            "data": {"chunk": 2},
            "parent_ids": [
                AnyStr(),
                AnyStr(),
            ],
        },
        {
            "event": "on_chain_end",
            "data": {"output": 2, "input": {"number": 1}},
            "run_id": AnyStr(),
            "name": "my_task",
            "tags": ["seq:step:1"],
            "metadata": {
                "thread_id": "1",
                "langgraph_step": 4,
                "langgraph_node": "my_task",
                "langgraph_triggers": ("__pregel_push",),
                "langgraph_path": (
                    "__pregel_push",
                    ("__pregel_pull", "my_workflow"),
                    2,
                    True,
                ),
                "langgraph_checkpoint_ns": AnyStr(),
            },
            "parent_ids": [
                AnyStr(),
                AnyStr(),
            ],
        },
        {
            "event": "on_chain_stream",
            "run_id": AnyStr(),
            "name": "my_workflow",
            "tags": ["graph:step:4"],
            "metadata": {
                "thread_id": "1",
                "langgraph_step": 4,
                "langgraph_node": "my_workflow",
                "langgraph_triggers": ("__start__",),
                "langgraph_path": ("__pregel_pull", "my_workflow"),
                "langgraph_checkpoint_ns": AnyStr(),
            },
            "data": {"chunk": "done"},
            "parent_ids": [AnyStr()],
        },
        {
            "event": "on_chain_stream",
            "run_id": AnyStr(),
            "name": "LangGraph",
            "tags": [],
            "metadata": {"thread_id": "1"},
            "data": {"chunk": {"my_task": 2}},
            "parent_ids": [],
        },
        {
            "event": "on_chain_end",
            "data": {"output": "done", "input": 1},
            "run_id": AnyStr(),
            "name": "my_workflow",
            "tags": ["graph:step:4"],
            "metadata": {
                "thread_id": "1",
                "langgraph_step": 4,
                "langgraph_node": "my_workflow",
                "langgraph_triggers": ("__start__",),
                "langgraph_path": ("__pregel_pull", "my_workflow"),
                "langgraph_checkpoint_ns": AnyStr(),
            },
            "parent_ids": [AnyStr()],
        },
        {
            "event": "on_chain_stream",
            "run_id": AnyStr(),
            "name": "LangGraph",
            "tags": [],
            "metadata": {"thread_id": "1"},
            "data": {"chunk": {"my_workflow": "done"}},
            "parent_ids": [],
        },
        {
            "event": "on_chain_end",
            "data": {"output": "done"},
            "run_id": AnyStr(),
            "name": "LangGraph",
            "tags": [],
            "metadata": {"thread_id": "1"},
            "parent_ids": [],
        },
    ]


@pytest.mark.parametrize("with_timeout", [False, "inner", "outer", "both"])
@pytest.mark.parametrize("subgraph_persist", [True, False])
async def test_parent_command_goto(
    async_checkpointer: BaseCheckpointSaver,
    subgraph_persist: bool,
    with_timeout: Literal[False, "inner", "outer", "both"],
) -> None:
    class State(TypedDict):
        dialog_state: Annotated[list[str], operator.add]

    async def node_a_child(state):
        return {"dialog_state": ["a_child_state"]}

    async def node_b_child(state):
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

    async def node_b_parent(state):
        return {"dialog_state": ["node_b_parent"]}

    main_builder = StateGraph(State)
    main_builder.add_node(node_b_parent)
    main_builder.add_edge(START, "subgraph_node")
    main_builder.add_node("subgraph_node", sub_graph, destinations=("node_b_parent",))
    main_graph = main_builder.compile(async_checkpointer, name="parent")
    if with_timeout in ("outer", "both"):
        main_graph.step_timeout = 1

    config = {"configurable": {"thread_id": 1}}

    assert await main_graph.ainvoke(
        input={"dialog_state": ["init_state"]}, config=config
    ) == {"dialog_state": ["init_state", "b_child_state", "node_b_parent"]}


@pytest.mark.parametrize("with_timeout", [True, False])
async def test_timeout_with_parent_command(
    async_checkpointer: BaseCheckpointSaver, with_timeout: bool
) -> None:
    """Test that parent commands are properly propagated during timeouts."""

    class State(TypedDict):
        value: str

    async def parent_command_node(state: State) -> State:
        await asyncio.sleep(0.1)  # Add some delay before raising
        return Command(graph=Command.PARENT, goto="test_cmd", update={"key": "value"})

    builder = StateGraph(State)
    builder.add_node("parent_cmd", parent_command_node)
    builder.set_entry_point("parent_cmd")
    graph = builder.compile(checkpointer=async_checkpointer)
    if with_timeout:
        graph.step_timeout = 1

    # Should propagate parent command, not timeout
    thread1 = {"configurable": {"thread_id": "1"}}
    with pytest.raises(ParentCommand) as exc_info:
        await graph.ainvoke({"value": "start"}, thread1)
    assert exc_info.value.args[0].goto == "test_cmd"
    assert exc_info.value.args[0].update == {"key": "value"}


async def test_fork_and_update_task_results(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
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
        .compile(checkpointer=async_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    history: list[StateSnapshot] = []

    # Initial run
    await graph.ainvoke({"name": "start"}, config)
    history = [c async for c in graph.aget_state_history(config)]

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
    await graph.ainvoke(
        None,
        await graph.aupdate_state(
            history[4].config,
            values=[StateUpdate(values={"name": "start*"}, as_node="__start__")],
            as_node="__copy__",
        ),
    )

    history = [c async for c in graph.aget_state_history(config)]
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

    await graph.ainvoke(
        None,
        await graph.aupdate_state(
            history[3].config,
            [StateUpdate(values={"name": "one*"}, as_node="one")],
            "__copy__",
        ),
    )

    history = [c async for c in graph.aget_state_history(config)]
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
    await graph.ainvoke({"name": "start"}, config)
    history = [c async for c in graph.aget_state_history(config)]

    # Fork from task "two"
    # Start from the checkpoint that has the task "two"
    assert history[2].values == {"name": "start > one"}

    await graph.ainvoke(
        None,
        await graph.aupdate_state(
            history[2].config,
            [
                StateUpdate(values={"name": "two"}, as_node="two"),
                StateUpdate(values={"name": "two"}, as_node="two"),
            ],
            "__copy__",
        ),
    )

    history = [c async for c in graph.aget_state_history(config)]
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

    await graph.ainvoke(
        None,
        await graph.aupdate_state(
            history[1].config,
            [StateUpdate(values={"name": "three*"}, as_node="three")],
            "__copy__",
        ),
    )

    history = [c async for c in graph.aget_state_history(config)]
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

    await graph.ainvoke(
        None, await graph.aupdate_state(history[3].config, None, "__copy__")
    )

    history = [c async for c in graph.aget_state_history(config)]
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


async def test_subgraph_streaming_async() -> None:
    """Test subgraph streaming when used as a node in async version"""

    # Create a fake chat model that returns a simple response
    model = GenericFakeChatModel(messages=iter(["The weather is sunny today."]))

    # Create a subgraph that uses the fake chat model
    async def call_model_node(
        state: MessagesState, config: RunnableConfig
    ) -> MessagesState:
        """Node that calls the model with the last message."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        response = await model.ainvoke([("user", last_message)], config)
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
    async def parent_node(state: SomeCustomState, config: RunnableConfig) -> dict:
        """Node that runs the subgraph."""
        msgs = {"messages": [("user", "What is the weather in Tokyo?")]}
        events = []
        async for event in compiled_subgraph.astream(
            msgs, config, stream_mode="messages"
        ):
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
    result = await compiled_workflow.ainvoke({})

    assert result["last_chunk"].content == "today."
    assert result["num_chunks"] == 9


@NEEDS_CONTEXTVARS
async def test_null_resume_disallowed_with_multiple_interrupts(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    class State(TypedDict):
        text_1: str
        text_2: str

    async def human_node_1(state: State):
        value = interrupt(state["text_1"])
        return {"text_1": value}

    async def human_node_2(state: State):
        value = interrupt(state["text_2"])
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
    await graph.ainvoke(
        {"text_1": "original text 1", "text_2": "original text 2"}, config=config
    )

    resume_map = {
        i.id: f"resume for prompt: {i.value}"
        for i in (await graph.aget_state(config)).interrupts
    }
    with pytest.raises(
        RuntimeError,
        match="When there are multiple pending interrupts, you must specify the interrupt id when resuming.",
    ):
        await graph.ainvoke(Command(resume="singular resume"), config=config)

    assert await graph.ainvoke(Command(resume=resume_map), config=config) == {
        "text_1": "resume for prompt: original text 1",
        "text_2": "resume for prompt: original text 2",
    }


async def test_astream_waiter_cleanup_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that AsyncPregelLoop cleans up waiter tasks after cancellation."""

    recorded_tasks: list[asyncio.Task[None]] = []
    finished_tasks: list[asyncio.Task[None]] = []

    original_wait = AsyncQueue.wait

    async def tracked_wait(self: AsyncQueue) -> None:
        task = asyncio.current_task()
        assert task is not None
        recorded_tasks.append(task)
        try:
            await original_wait(self)
        finally:
            finished_tasks.append(task)

    monkeypatch.setattr(AsyncQueue, "wait", tracked_wait)

    class State(TypedDict, total=False):
        count: int

    async def slow_node(state: State) -> State:
        await asyncio.sleep(0.05)
        state = dict(state)
        state["count"] = state.get("count", 0) + 1
        await asyncio.sleep(0.1)
        return state

    builder = StateGraph(State)
    builder.add_node("slow", slow_node)
    builder.add_edge(START, "slow")
    builder.add_edge("slow", END)
    graph = builder.compile()

    async def consumer() -> None:
        async for _ in graph.astream({"msg": "hi"}, stream_mode="messages"):
            await asyncio.sleep(0.01)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.05)

    assert recorded_tasks, "expected stream.wait() task to be created"
    assert set(finished_tasks) == set(recorded_tasks)
    assert all(t.done() for t in recorded_tasks)


async def test_supersteps_populate_task_results(
    async_checkpointer: BaseCheckpointSaver,
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
        .compile(checkpointer=async_checkpointer)
    )

    # reference run with ainvoke
    ref_cfg = {"configurable": {"thread_id": "ref"}}
    await graph.ainvoke({"num": 1, "text": "one"}, ref_cfg)
    ref_history = [h async for h in graph.aget_state_history(ref_cfg)]

    # Helper: pull first task result for a node name from history
    def first_task_result(history: list[StateSnapshot], node: str) -> Any:
        for s in history:
            for t in s.tasks:
                if t.name == node:
                    return t.result
        return None

    ref_start_result = first_task_result(ref_history, "__start__")
    ref_double_result = first_task_result(ref_history, "double")
    assert ref_start_result == {"num": 1, "text": "one"}
    assert ref_double_result == {"num": 2, "text": "oneone"}

    # using supersteps
    bulk_cfg = {"configurable": {"thread_id": "bulk"}}
    await graph.abulk_update_state(
        bulk_cfg,
        [
            [StateUpdate(values={}, as_node="__input__")],
            [StateUpdate(values={"num": 1, "text": "one"}, as_node="__start__")],
            [StateUpdate(values={"num": 2, "text": "oneone"}, as_node="double")],
        ],
    )
    bulk_history = [h async for h in graph.aget_state_history(bulk_cfg)]

    bulk_start_result = first_task_result(bulk_history, "__start__")
    bulk_double_result = first_task_result(bulk_history, "double")

    assert bulk_start_result == ref_start_result == {"num": 1, "text": "one"}
    assert bulk_double_result == ref_double_result == {"num": 2, "text": "oneone"}


async def test_fork_does_not_apply_pending_writes(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test that forking with aupdate_state does not apply pending writes from original execution."""

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
        .compile(checkpointer=async_checkpointer)
    )

    thread1 = {"configurable": {"thread_id": "1"}}
    await graph.ainvoke({"value": 1}, thread1)

    history = [c async for c in graph.aget_state_history(thread1)]
    checkpoint_before_a = next(s for s in history if s.next == ("node_a",))

    fork_config = await graph.aupdate_state(
        checkpoint_before_a.config, {"value": 20}, as_node="node_a"
    )
    result = await graph.ainvoke(None, fork_config)

    # 1 (input) + 20 (forked node_a) + 100 (node_b) = 121
    assert result == {"value": 121}
