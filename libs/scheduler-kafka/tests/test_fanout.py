import asyncio
import operator
from typing import (
    Annotated,
    Sequence,
    TypedDict,
    Union,
)

import pytest
from aiokafka import AIOKafkaProducer

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import StateGraph
from langgraph.pregel import Pregel
from langgraph.scheduler.kafka import serde
from langgraph.scheduler.kafka.types import MessageToOrchestrator, Topics
from tests.any import AnyDict
from tests.run import drain_topics

pytestmark = pytest.mark.anyio


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


async def test_fanout_graph(topics: Topics, checkpointer: BaseCheckpointSaver) -> None:
    input = {"query": "what is weather in sf"}
    config = {"configurable": {"thread_id": "1"}}
    graph = mk_fanout_graph(checkpointer)

    # start a new run
    async with AIOKafkaProducer(value_serializer=serde.dumps) as producer:
        await producer.send_and_wait(
            topics.orchestrator,
            MessageToOrchestrator(input=input, config=config),
        )

    # drain topics
    orch_msgs, exec_msgs = await drain_topics(
        topics, graph, config, until=lambda s: s.values and s.next == ()
    )

    # check state
    state = await graph.aget_state(config)
    assert state.next == ()
    assert (
        state.values
        == await graph.ainvoke(input, {"configurable": {"thread_id": "2"}})
        == {
            "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
            "query": "analyzed: query: analyzed: query: what is weather in sf",
            "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
        }
    )

    # check history
    history = [c async for c in graph.aget_state_history(config)]
    assert len(history) == 11

    # check messages
    assert orch_msgs == [MessageToOrchestrator(input=input, config=config)] + [
        {
            "config": {
                "callbacks": None,
                "configurable": {
                    "__pregel_ensure_latest": True,
                    "__pregel_dedupe_tasks": True,
                    "__pregel_resuming": False,
                    "checkpoint_id": c.config["configurable"]["checkpoint_id"],
                    "checkpoint_ns": "",
                    "thread_id": "1",
                },
                "metadata": AnyDict(),
                "recursion_limit": 25,
                "tags": [],
            },
            "input": None,
        }
        for c in reversed(history)
        for _ in c.tasks
    ]
    assert exec_msgs == [
        {
            "config": {
                "callbacks": None,
                "configurable": {
                    "__pregel_ensure_latest": True,
                    "__pregel_dedupe_tasks": True,
                    "__pregel_resuming": False,
                    "checkpoint_id": c.config["configurable"]["checkpoint_id"],
                    "checkpoint_ns": "",
                    "thread_id": "1",
                },
                "metadata": AnyDict(),
                "recursion_limit": 25,
                "tags": [],
            },
            "task": {
                "id": t.id,
                "path": list(t.path),
            },
        }
        for c in reversed(history)
        for t in c.tasks
    ]


async def test_fanout_graph_w_interrupt(
    topics: Topics, checkpointer: BaseCheckpointSaver
) -> None:
    input = {"query": "what is weather in sf"}
    config = {"configurable": {"thread_id": "1"}}
    graph = mk_fanout_graph(checkpointer, interrupt_before=["qa"])

    # start a new run
    async with AIOKafkaProducer(value_serializer=serde.dumps) as producer:
        await producer.send_and_wait(
            topics.orchestrator,
            MessageToOrchestrator(input=input, config=config),
        )

    orch_msgs, exec_msgs = await drain_topics(
        topics, graph, config, until=lambda s: s.values and s.next == ("qa",)
    )

    # check interrupted state
    state = await graph.aget_state(config)
    assert state.next == ("qa",)
    assert (
        state.values
        == await graph.ainvoke(input, {"configurable": {"thread_id": "2"}})
        == {
            "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
            "query": "analyzed: query: analyzed: query: what is weather in sf",
        }
    )

    # check history
    history = [c async for c in graph.aget_state_history(config)]
    assert len(history) == 10

    # check messages
    assert orch_msgs == [MessageToOrchestrator(input=input, config=config)] + [
        {
            "config": {
                "callbacks": None,
                "configurable": {
                    "__pregel_ensure_latest": True,
                    "__pregel_dedupe_tasks": True,
                    "__pregel_resuming": False,
                    "checkpoint_id": c.config["configurable"]["checkpoint_id"],
                    "checkpoint_ns": "",
                    "thread_id": "1",
                },
                "metadata": AnyDict(),
                "recursion_limit": 25,
                "tags": [],
            },
            "input": None,
        }
        for c in reversed(history[1:])  # the last one wasn't executed
        for _ in c.tasks
    ]
    assert exec_msgs == [
        {
            "config": {
                "callbacks": None,
                "configurable": {
                    "__pregel_ensure_latest": True,
                    "__pregel_dedupe_tasks": True,
                    "__pregel_resuming": False,
                    "checkpoint_id": c.config["configurable"]["checkpoint_id"],
                    "checkpoint_ns": "",
                    "thread_id": "1",
                },
                "metadata": AnyDict(),
                "recursion_limit": 25,
                "tags": [],
            },
            "task": {
                "id": t.id,
                "path": list(t.path),
            },
        }
        for c in reversed(history[1:])  # the last one wasn't executed
        for t in c.tasks
    ]

    # resume the thread
    async with AIOKafkaProducer(value_serializer=serde.dumps) as producer:
        await producer.send_and_wait(
            topics.orchestrator,
            MessageToOrchestrator(input=None, config=config),
        )

    orch_msgs, exec_msgs = await drain_topics(
        topics, graph, config, until=lambda s: s.values and s.next == ()
    )

    # check final state
    state = await graph.aget_state(config)
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
