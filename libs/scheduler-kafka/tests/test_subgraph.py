from typing import Literal, ParamSpec, TypeVar, cast

import pytest
from aiokafka import AIOKafkaProducer
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
from langgraph.scheduler.kafka.types import MessageToOrchestrator, Topics
from tests.any import AnyStr
from tests.run import drain_topics

pytestmark = pytest.mark.anyio
C = ParamSpec("C")
R = TypeVar("R")


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


async def test_subgraph_w_interrupt(
    topics: Topics, checkpointer: BaseCheckpointSaver
) -> None:
    input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    config = {"configurable": {"thread_id": "1"}}
    graph = mk_weather_graph(checkpointer)

    # start a new run
    async with AIOKafkaProducer(value_serializer=serde.dumps) as producer:
        await producer.send_and_wait(
            topics.orchestrator,
            MessageToOrchestrator(input=input, config=config),
        )

    orch_msgs, exec_msgs = await drain_topics(
        topics,
        graph,
        config,
        until=lambda state: state.next == ("weather_graph",),
    )

    # check interrupted state
    state = await graph.aget_state(config)
    assert len(orch_msgs) == 4
    assert len(exec_msgs) == 3
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

    orch_msgs, exec_msgs = await drain_topics(
        topics, graph, config, until=lambda state: state.next == (), debug=True
    )

    # check final state
    state = await graph.aget_state(config)
    assert len(orch_msgs) == 2
    assert len(exec_msgs) == 1
    assert state.next == ()
    assert state.values == {
        "messages": [
            HumanMessage(id=AnyStr(), content="what's the weather in sf"),
            AIMessage(content="I'ts sunny in San Francisco!", id=AnyStr()),
        ],
        "route": "weather",
    }
