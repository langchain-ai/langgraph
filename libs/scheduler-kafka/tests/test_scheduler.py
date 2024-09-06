import asyncio
import functools
import operator
from typing import Annotated, Callable, ParamSpec, Sequence, TypedDict, TypeVar, Union

import anyio
import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import StateGraph
from langgraph.pregel import Pregel
from langgraph.scheduler.kafka import serde
from langgraph.scheduler.kafka.executor import KafkaExecutor
from langgraph.scheduler.kafka.orchestrator import KafkaOrchestrator
from langgraph.scheduler.kafka.types import MessageToOrchestrator, Topics

pytestmark = pytest.mark.anyio
C = ParamSpec("C")
R = TypeVar("R")


def timeout(delay: int):
    def decorator(func: Callable[C, R]) -> Callable[C, R]:
        @functools.wraps(func)
        async def new_func(*args: C.args, **kwargs: C.kwargs) -> R:
            async with asyncio.timeout(delay):
                return await func(*args, **kwargs)

        return new_func

    return decorator


def mk_fanout_graph(
    checkpointer: BaseCheckpointSaver, interrupt_before: Sequence[str] = ()
) -> Pregel:
    # copied from test_in_one_fan_out_state_graph_waiting_edge_multiple_cond_edge
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

    builder = StateGraph(State)

    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("analyzer_one", analyzer_one)
    builder.add_node("retriever_one", retriever_one)
    builder.add_node("retriever_two", retriever_two)
    builder.add_node("decider", decider)
    builder.add_node("qa", qa)

    builder.set_entry_point("rewrite_query")
    builder.add_conditional_edges("rewrite_query", retriever_picker)
    builder.add_edge("analyzer_one", "retriever_one")
    builder.add_edge(["retriever_one", "retriever_two"], "decider")
    builder.add_conditional_edges("decider", decider_cond)
    builder.set_finish_point("qa")

    return builder.compile(checkpointer, interrupt_before=interrupt_before)


@timeout(10)
async def test_fanout_graph(topics: Topics, checkpointer: BaseCheckpointSaver) -> None:
    input = {"query": "what is weather in sf"}
    config = {"configurable": {"thread_id": "1"}}
    graph = mk_fanout_graph(checkpointer)
    n_orch_msgs = 0
    n_exec_msgs = 0

    async def orchestrator(expected: int) -> None:
        nonlocal n_orch_msgs
        async with KafkaOrchestrator(graph, topics) as orch:
            async for msgs in orch:
                print("orch", msgs)
                n_orch_msgs += len(msgs)
                if n_orch_msgs == expected:
                    break

    async def executor(expected: int) -> None:
        nonlocal n_exec_msgs
        async with KafkaExecutor(graph, topics) as exec:
            async for msgs in exec:
                print("exec", msgs)
                n_exec_msgs += len(msgs)
                if n_exec_msgs == expected:
                    break

    # start a new run
    async with AIOKafkaProducer(value_serializer=serde.dumps) as producer:
        await producer.send_and_wait(
            topics.orchestrator,
            MessageToOrchestrator(input=input, config=config),
        )

    # run the orchestrator and executor
    async with anyio.create_task_group() as tg:
        tg.start_soon(orchestrator, 13, name="orchestrator")
        tg.start_soon(executor, 12, name="executor")

    # check no errors
    async with AIOKafkaConsumer(topics.error) as consumer:
        assert len(consumer.assignment()) > 0
        for tp in consumer.assignment():
            assert await consumer.position(tp) == 0

    state = await graph.aget_state(config)
    assert n_orch_msgs == 13
    assert n_exec_msgs == 12
    assert (
        state.values
        == await graph.ainvoke(input, {"configurable": {"thread_id": "2"}})
        == {
            "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
            "query": "analyzed: query: analyzed: query: what is weather in sf",
            "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
        }
    )


@timeout(10)
async def test_fanout_graph_w_interrupt(
    topics: Topics, checkpointer: BaseCheckpointSaver
) -> None:
    input = {"query": "what is weather in sf"}
    config = {"configurable": {"thread_id": "1"}}
    graph = mk_fanout_graph(checkpointer, interrupt_before=["qa"])
    n_orch_msgs = 0
    n_exec_msgs = 0

    async def orchestrator(expected: int) -> None:
        nonlocal n_orch_msgs
        async with KafkaOrchestrator(graph, topics) as orch:
            async for msgs in orch:
                print("orch", msgs)
                n_orch_msgs += len(msgs)
                if n_orch_msgs == expected:
                    break

    async def executor(expected: int) -> None:
        nonlocal n_exec_msgs
        async with KafkaExecutor(graph, topics) as exec:
            async for msgs in exec:
                print("exec", msgs)
                n_exec_msgs += len(msgs)
                if n_exec_msgs == expected:
                    break

    # start a new run
    async with AIOKafkaProducer(value_serializer=serde.dumps) as producer:
        await producer.send_and_wait(
            topics.orchestrator,
            MessageToOrchestrator(input=input, config=config),
        )

    # run the orchestrator and executor
    async with anyio.create_task_group() as tg:
        tg.start_soon(orchestrator, 12, name="orchestrator")
        tg.start_soon(executor, 11, name="executor")

    # check no errors
    async with AIOKafkaConsumer(topics.error) as consumer:
        assert len(consumer.assignment()) > 0
        for tp in consumer.assignment():
            assert await consumer.position(tp) == 0

    state = await graph.aget_state(config)
    assert n_orch_msgs == 12
    assert n_exec_msgs == 11
    assert state.next == ("qa",)
    assert (
        state.values
        == await graph.ainvoke(input, {"configurable": {"thread_id": "2"}})
        == {
            "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
            "query": "analyzed: query: analyzed: query: what is weather in sf",
        }
    )

    # resume the thread
    async with AIOKafkaProducer(value_serializer=serde.dumps) as producer:
        await producer.send_and_wait(
            topics.orchestrator,
            MessageToOrchestrator(input=None, config=config),
        )

    # run the orchestrator and executor
    async with anyio.create_task_group() as tg:
        tg.start_soon(orchestrator, 14, name="orchestrator")
        tg.start_soon(executor, 12, name="executor")

    # check no errors
    async with AIOKafkaConsumer(topics.error) as consumer:
        assert len(consumer.assignment()) > 0
        for tp in consumer.assignment():
            assert await consumer.position(tp) == 0

    state = await graph.aget_state(config)
    assert n_orch_msgs == 14
    assert n_exec_msgs == 12
    assert state.next == ()
    assert (
        state.values
        == await graph.ainvoke(None, {"configurable": {"thread_id": "2"}})
        == {
            "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
            "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
            "query": "analyzed: query: analyzed: query: what is weather in sf",
        }
    )
