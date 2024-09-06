import asyncio
import operator
from typing import Annotated, TypedDict, Union

import pytest
from aiokafka import AIOKafkaProducer

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import StateGraph
from langgraph.pregel import Pregel
from langgraph.scheduler.kafka import serde
from langgraph.scheduler.kafka.executor import KafkaExecutor
from langgraph.scheduler.kafka.orchestrator import KafkaOrchestrator
from langgraph.scheduler.kafka.types import MessageToOrchestrator, Topics

pytestmark = pytest.mark.anyio


def mk_fanout_graph(checkpointer: BaseCheckpointSaver) -> Pregel:
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
        print("rewrite_query", data)
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

    return builder.compile(checkpointer)


async def test_fanout_graph(topics: Topics, checkpointer: BaseCheckpointSaver) -> None:
    graph = mk_fanout_graph(checkpointer)
    n_orch_msgs = 0
    n_exec_msgs = 0

    async def orchestrator() -> None:
        nonlocal n_orch_msgs
        async with KafkaOrchestrator(graph, topics) as orch:
            async for msg in orch:
                n_orch_msgs += 1
                print("orch", msg)

    async def executor() -> None:
        nonlocal n_exec_msgs
        async with KafkaExecutor(graph, topics) as exec:
            async for msg in exec:
                n_exec_msgs += 1
                print("exec", msg)

    async with asyncio.TaskGroup() as tg:
        o = tg.create_task(orchestrator(), name="orchestrator")
        e = tg.create_task(executor(), name="executor")

        # start a new run
        producer = AIOKafkaProducer(value_serializer=serde.dumps)
        await producer.start()
        await producer.send_and_wait(
            topics.orchestrator,
            MessageToOrchestrator(
                input={"query": "what is weather in sf"},
                config={"configurable": {"thread_id": "1"}},
            ),
        )
        await producer.stop()

        # wait for the run to finish
        await asyncio.sleep(5)
        o.cancel()
        e.cancel()

    assert n_orch_msgs == 13
    assert n_exec_msgs == 12
