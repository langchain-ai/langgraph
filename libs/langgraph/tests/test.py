from typing import Literal, cast

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import tool

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, PULL, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import (
    PregelTask,
    StateSnapshot,
    StreamWriter,
)
from tests.any_str import AnyDict, AnyStr
from tests.messages import (
    _AnyIdAIMessage,
    _AnyIdHumanMessage,
)

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


def model_node(state: SubGraphState, writer: StreamWriter):
    writer(" very")
    result = weather_model.invoke(state["messages"])
    return {"city": cast(AIMessage, result).tool_calls[0]["args"]["city"]}


def weather_node(state: SubGraphState, writer: StreamWriter):
    writer(" good")
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


def router_node(state: RouterState, writer: StreamWriter):
    writer("I'm")
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


def weather_graph(state: RouterState):
    return subgraph.invoke(state)


graph = StateGraph(RouterState)
graph.add_node(router_node)
graph.add_node(normal_llm_node)
graph.add_node("weather_graph", weather_graph)
graph.add_edge(START, "router_node")
graph.add_conditional_edges(
    "router_node",
    route_after_prediction,
    path_map=["weather_graph", "normal_llm_node"],
)
graph.add_edge("normal_llm_node", END)
graph.add_edge("weather_graph", END)
graph = graph.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}
thread2 = {"configurable": {"thread_id": "2"}}
inputs = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}

# run with custom output
assert [
    c for c in graph.stream(inputs, thread2, stream_mode="custom", subgraphs=True)
] == [
    ((), "I'm"),
    ((AnyStr("weather_graph:"),), " very"),
]
assert [
    c for c in graph.stream(None, thread2, stream_mode="custom", subgraphs=True)
] == [
    ((AnyStr("weather_graph:"),), " good"),
]

# run until interrupt
assert [
    c
    for c in graph.stream(
        inputs,
        config=config,
        stream_mode="updates",
        subgraphs=True,
        checkpoint_during=False,
    )
] == [
    ((), {"router_node": {"route": "weather"}}),
    ((AnyStr("weather_graph:"),), {"model_node": {"city": "San Francisco"}}),
    ((), {"__interrupt__": ()}),
]

# check current state
state = graph.get_state(config)
assert state == StateSnapshot(
    values={
        "messages": [_AnyIdHumanMessage(content="what's the weather in sf")],
        "route": "weather",
    },
    next=("weather_graph",),
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
    tasks=(
        PregelTask(
            id=AnyStr(),
            name="weather_graph",
            path=(PULL, "weather_graph"),
            state={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("weather_graph:"),
                }
            },
        ),
    ),
    interrupts=(),
)

# update
graph.update_state(state.tasks[0].state, {"city": "la"})

# run after update
assert [
    c for c in graph.stream(None, config=config, stream_mode="updates", subgraphs=True)
] == [
    (
        (AnyStr("weather_graph:"),),
        {
            "weather_node": {
                "messages": [{"role": "assistant", "content": "I'ts sunny in la!"}]
            }
        },
    ),
    (
        (),
        {
            "weather_graph": {
                "messages": [
                    _AnyIdHumanMessage(content="what's the weather in sf"),
                    _AnyIdAIMessage(content="I'ts sunny in la!"),
                ]
            }
        },
    ),
]

# try updating acting as weather node
config = {"configurable": {"thread_id": "14"}}
inputs = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
assert [
    c
    for c in graph.stream(
        inputs,
        config=config,
        stream_mode="updates",
        subgraphs=True,
        checkpoint_during=False,
    )
] == [
    ((), {"router_node": {"route": "weather"}}),
    ((AnyStr("weather_graph:"),), {"model_node": {"city": "San Francisco"}}),
    ((), {"__interrupt__": ()}),
]
state = graph.get_state(config, subgraphs=True)
assert state == StateSnapshot(
    values={
        "messages": [_AnyIdHumanMessage(content="what's the weather in sf")],
        "route": "weather",
    },
    next=("weather_graph",),
    config={
        "configurable": {
            "thread_id": "14",
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
    tasks=(
        PregelTask(
            id=AnyStr(),
            name="weather_graph",
            path=(PULL, "weather_graph"),
            state=StateSnapshot(
                values={
                    "messages": [
                        _AnyIdHumanMessage(content="what's the weather in sf")
                    ],
                    "city": "San Francisco",
                },
                next=("weather_node",),
                config={
                    "configurable": {
                        "thread_id": "14",
                        "checkpoint_ns": AnyStr("weather_graph:"),
                        "checkpoint_id": AnyStr(),
                        "checkpoint_map": AnyDict(
                            {
                                "": AnyStr(),
                                AnyStr("weather_graph:"): AnyStr(),
                            }
                        ),
                    }
                },
                metadata={
                    "source": "loop",
                    "step": 1,
                    "parents": {"": AnyStr()},
                },
                created_at=AnyStr(),
                parent_config=None,
                tasks=(
                    PregelTask(
                        id=AnyStr(),
                        name="weather_node",
                        path=(PULL, "weather_node"),
                    ),
                ),
                interrupts=(),
            ),
        ),
    ),
    interrupts=(),
)
graph.update_state(
    state.tasks[0].state.config,
    {"messages": [{"role": "assistant", "content": "rainy"}]},
    as_node="weather_node",
)
state = graph.get_state(config, subgraphs=True)
assert state == StateSnapshot(
    values={
        "messages": [_AnyIdHumanMessage(content="what's the weather in sf")],
        "route": "weather",
    },
    next=("weather_graph",),
    config={
        "configurable": {
            "thread_id": "14",
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
    tasks=(
        PregelTask(
            id=AnyStr(),
            name="weather_graph",
            path=(PULL, "weather_graph"),
            state=StateSnapshot(
                values={
                    "messages": [
                        _AnyIdHumanMessage(content="what's the weather in sf"),
                        _AnyIdAIMessage(content="rainy"),
                    ],
                    "city": "San Francisco",
                },
                next=(),
                config={
                    "configurable": {
                        "thread_id": "14",
                        "checkpoint_ns": AnyStr("weather_graph:"),
                        "checkpoint_id": AnyStr(),
                        "checkpoint_map": AnyDict(
                            {
                                "": AnyStr(),
                                AnyStr("weather_graph:"): AnyStr(),
                            }
                        ),
                    }
                },
                metadata={
                    "step": 2,
                    "source": "update",
                    "parents": {"": AnyStr()},
                },
                created_at=AnyStr(),
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": "14",
                            "checkpoint_ns": AnyStr("weather_graph:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {
                                    "": AnyStr(),
                                    AnyStr("weather_graph:"): AnyStr(),
                                }
                            ),
                        }
                    }
                ),
                interrupts=(),
                tasks=(),
            ),
        ),
    ),
    interrupts=(),
)
assert [
    c for c in graph.stream(None, config=config, stream_mode="updates", subgraphs=True)
] == [
    (
        (),
        {
            "weather_graph": {
                "messages": [
                    _AnyIdHumanMessage(content="what's the weather in sf"),
                    _AnyIdAIMessage(content="rainy"),
                ]
            }
        },
    ),
]

# run with custom output, without subgraph streaming, should omit subgraph chunks
assert [
    c
    for c in graph.stream(
        inputs, {"configurable": {"thread_id": "3"}}, stream_mode="custom"
    )
] == [
    "I'm",
]

# run with messages output, with subgraph streaming, should inc subgraph messages
assert [
    c
    for c in graph.stream(
        inputs,
        {"configurable": {"thread_id": "4"}},
        stream_mode="messages",
        subgraphs=True,
    )
] == [
    (
        (),
        (
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tool_call123",
                        name="router",
                        args={"dest": "weather"},
                    )
                ],
            ),
            {
                "thread_id": "4",
                "langgraph_step": 1,
                "langgraph_node": "router_node",
                "langgraph_triggers": ("branch:to:router_node",),
                "langgraph_path": ("__pregel_pull", "router_node"),
                "langgraph_checkpoint_ns": AnyStr("router_node:"),
                "checkpoint_ns": AnyStr("router_node:"),
                "ls_provider": "fakemessageslistchatmodel",
                "ls_model_type": "chat",
            },
        ),
    ),
    (
        (AnyStr("weather_graph:"),),
        (
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tool_call123",
                        name="get_weather",
                        args={"city": "San Francisco"},
                    )
                ],
            ),
            {
                "thread_id": "4",
                "langgraph_step": 1,
                "langgraph_node": "model_node",
                "langgraph_triggers": ("branch:to:model_node",),
                "langgraph_path": ("__pregel_pull", "model_node"),
                "langgraph_checkpoint_ns": AnyStr("weather_graph:"),
                "checkpoint_ns": AnyStr("weather_graph:"),
                "ls_provider": "fakemessageslistchatmodel",
                "ls_model_type": "chat",
            },
        ),
    ),
]

# run with messages output, without subgraph streaming, should exc subgraph messages
assert [
    c
    for c in graph.stream(
        inputs,
        {"configurable": {"thread_id": "5"}},
        stream_mode="messages",
    )
] == [
    (
        _AnyIdAIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="tool_call123",
                    name="router",
                    args={"dest": "weather"},
                )
            ],
        ),
        {
            "thread_id": "5",
            "langgraph_step": 1,
            "langgraph_node": "router_node",
            "langgraph_triggers": ("branch:to:router_node",),
            "langgraph_path": ("__pregel_pull", "router_node"),
            "langgraph_checkpoint_ns": AnyStr("router_node:"),
            "checkpoint_ns": AnyStr("router_node:"),
            "ls_provider": "fakemessageslistchatmodel",
            "ls_model_type": "chat",
        },
    ),
]
