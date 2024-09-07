import asyncio
import functools
import re
from typing import Callable, Literal, Optional, ParamSpec, TypeVar, Union, cast

import anyio
import pytest
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.tools import tool

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState
from langgraph.graph.state import StateGraph
from langgraph.pregel import Pregel
from langgraph.scheduler.kafka import serde
from langgraph.scheduler.kafka.executor import KafkaExecutor
from langgraph.scheduler.kafka.orchestrator import KafkaOrchestrator
from langgraph.scheduler.kafka.types import MessageToOrchestrator, Topics

pytestmark = pytest.mark.anyio
C = ParamSpec("C")
R = TypeVar("R")


class AnyStr(str):
    def __init__(self, prefix: Union[str, re.Pattern] = "") -> None:
        super().__init__()
        self.prefix = prefix

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and (
            other.startswith(self.prefix)
            if isinstance(self.prefix, str)
            else self.prefix.match(other)
        )

    def __hash__(self) -> int:
        return hash((str(self), self.prefix))


def timeout(delay: int):
    def decorator(func: Callable[C, R]) -> Callable[C, R]:
        @functools.wraps(func)
        async def new_func(*args: C.args, **kwargs: C.kwargs) -> R:
            async with asyncio.timeout(delay):
                return await func(*args, **kwargs)

        return new_func

    return decorator


def mk_weather_graph(checkpointer: BaseCheckpointSaver) -> Pregel:
    # copied from test_weather_subgraph

    # setup subgraph

    @tool
    def get_weather(city: str):
        """Get the weather for a specific city"""
        return f"I'ts sunny in {city}!"

    weather_model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tool_call123",
                        name="get_weather",
                        args={"city": "San Francisco"},
                    )
                ],
            )
        ]
    )

    class SubGraphState(MessagesState):
        city: str

    def model_node(state: SubGraphState):
        result = weather_model.invoke(state["messages"])
        return {"city": cast(AIMessage, result).tool_calls[0]["args"]["city"]}

    def weather_node(state: SubGraphState):
        result = get_weather.invoke({"city": state["city"]})
        return {"messages": [{"role": "assistant", "content": result}]}

    subgraph = StateGraph(SubGraphState)
    subgraph.add_node(model_node)
    subgraph.add_node(weather_node)
    subgraph.add_edge(START, "model_node")
    subgraph.add_edge("model_node", "weather_node")
    subgraph.add_edge("weather_node", END)
    subgraph = subgraph.compile(interrupt_before=["weather_node"])

    # setup main graph

    class RouterState(MessagesState):
        route: Literal["weather", "other"]

    router_model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tool_call123",
                        name="router",
                        args={"dest": "weather"},
                    )
                ],
            )
        ]
    )

    def router_node(state: RouterState):
        system_message = "Classify the incoming query as either about weather or not."
        messages = [{"role": "system", "content": system_message}] + state["messages"]
        route = router_model.invoke(messages)
        return {"route": cast(AIMessage, route).tool_calls[0]["args"]["dest"]}

    def normal_llm_node(state: RouterState):
        return {"messages": [AIMessage("Hello!")]}

    def route_after_prediction(state: RouterState):
        if state["route"] == "weather":
            return "weather_graph"
        else:
            return "normal_llm_node"

    async def weather_graph(state: RouterState):
        return await subgraph.ainvoke(state)

    graph = StateGraph(RouterState)
    graph.add_node(router_node)
    graph.add_node(normal_llm_node)
    graph.add_node("weather_graph", weather_graph)
    graph.add_edge(START, "router_node")
    graph.add_conditional_edges("router_node", route_after_prediction)
    graph.add_edge("normal_llm_node", END)
    graph.add_edge("weather_graph", END)

    return graph.compile(checkpointer=checkpointer)


@timeout(10)
async def test_subgraph_w_interrupt(
    topics: Topics, checkpointer: BaseCheckpointSaver
) -> None:
    input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    config = {"configurable": {"thread_id": "1"}}
    graph = mk_weather_graph(checkpointer)
    n_orch_msgs = 0
    n_exec_msgs = 0
    errors = []
    scope: Optional[anyio.CancelScope] = None

    async def orchestrator(expected: int) -> None:
        nonlocal n_orch_msgs
        async with KafkaOrchestrator(graph, topics) as orch:
            async for msgs in orch:
                n_orch_msgs += len(msgs)
                print("orch", n_orch_msgs, msgs)
                if n_orch_msgs == expected:
                    break

    async def executor(expected: int) -> None:
        nonlocal n_exec_msgs
        async with KafkaExecutor(graph, topics) as exec:
            async for msgs in exec:
                n_exec_msgs += len(msgs)
                print("exec", n_exec_msgs, msgs)
                if n_exec_msgs == expected:
                    break

    async def error_consumer() -> None:
        async with AIOKafkaConsumer(topics.error) as consumer:
            async for msg in consumer:
                errors.append(msg)
                if scope:
                    scope.cancel()

    # start error consumer
    error_task = asyncio.create_task(error_consumer(), name="error_consumer")

    # start a new run
    async with AIOKafkaProducer(value_serializer=serde.dumps) as producer:
        await producer.send_and_wait(
            topics.orchestrator,
            MessageToOrchestrator(input=input, config=config),
        )

    # run the orchestrator and executor
    async with anyio.create_task_group() as tg:
        scope = tg.cancel_scope
        tg.start_soon(orchestrator, 4, name="orchestrator")
        tg.start_soon(executor, 3, name="executor")

    # check no errors
    assert not errors

    # check interrupted state
    state = await graph.aget_state(config)
    assert n_orch_msgs == 4
    assert n_exec_msgs == 3
    assert state.next == ("weather_graph",)
    assert state.values == {
        "messages": [HumanMessage(id=AnyStr(), content="what's the weather in sf")],
        "route": "weather",
    }

    # resume the thread
    async with AIOKafkaProducer(value_serializer=serde.dumps) as producer:
        await producer.send_and_wait(
            topics.orchestrator,
            MessageToOrchestrator(input=None, config=config),
        )

    # run the orchestrator and executor
    async with anyio.create_task_group() as tg:
        scope = tg.cancel_scope
        tg.start_soon(orchestrator, 6, name="orchestrator")
        tg.start_soon(executor, 4, name="executor")

    # check no errors
    assert not errors

    # check final state
    state = await graph.aget_state(config)
    assert n_orch_msgs == 6
    assert n_exec_msgs == 4
    assert state.next == ()
    assert state.values == {
        "answer": "doc1,doc1,doc2,doc2,doc3,doc3,doc4,doc4",
        "docs": ["doc1", "doc1", "doc2", "doc2", "doc3", "doc3", "doc4", "doc4"],
        "query": "analyzed: query: analyzed: query: what is weather in sf",
    }

    error_task.cancel()
