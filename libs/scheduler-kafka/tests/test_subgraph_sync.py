from typing import Literal, cast

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import tool

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState
from langgraph.graph.state import StateGraph
from langgraph.pregel import Pregel
from langgraph.scheduler.kafka import serde
from langgraph.scheduler.kafka.default_sync import DefaultProducer
from langgraph.scheduler.kafka.types import MessageToOrchestrator, Topics
from tests.any import AnyDict
from tests.drain import drain_topics
from tests.messages import _AnyIdAIMessage, _AnyIdHumanMessage

pytestmark = pytest.mark.anyio


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

    def weather_graph(state: RouterState):
        return subgraph.invoke(state)

    graph = StateGraph(RouterState)
    graph.add_node(router_node)
    graph.add_node(normal_llm_node)
    graph.add_node("weather_graph", weather_graph)
    graph.add_edge(START, "router_node")
    graph.add_conditional_edges("router_node", route_after_prediction)
    graph.add_edge("normal_llm_node", END)
    graph.add_edge("weather_graph", END)

    return graph.compile(checkpointer=checkpointer)


def test_subgraph_w_interrupt(
    topics: Topics, checkpointer: BaseCheckpointSaver
) -> None:
    input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    config = {"configurable": {"thread_id": "1"}}
    graph = mk_weather_graph(checkpointer)

    # start a new run
    with DefaultProducer() as producer:
        producer.send(
            topics.orchestrator,
            value=serde.dumps(MessageToOrchestrator(input=input, config=config)),
        )
        producer.flush()

    orch_msgs, exec_msgs = drain_topics(topics, graph)

    # check interrupted state
    state = graph.get_state(config)
    assert state.next == ("weather_graph",)
    assert state.values == {
        "messages": [_AnyIdHumanMessage(content="what's the weather in sf")],
        "route": "weather",
    }

    # check outer history
    history = [c for c in graph.get_state_history(config)]
    assert len(history) == 3

    # check child history
    child_history = [c for c in graph.get_state_history(history[0].tasks[0].state)]
    assert len(child_history) == 3

    # check messages
    assert (
        orch_msgs
        == (
            # initial message to outer graph
            [MessageToOrchestrator(input=input, config=config)]
            # outer graph messages, until interrupted
            + [
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
                    "finally_send": None,
                }
                for c in reversed(history[1:])  # the last one wasn't executed
                # orchestrator messages appear only after tasks for that checkpoint
                # finish executing, ie. after executor sends message to resume checkpoint
                for _ in c.tasks
            ]
            # initial message to child graph
            + [
                {
                    "config": {
                        "callbacks": None,
                        "configurable": {
                            "__pregel_checkpointer": None,
                            "__pregel_delegate": False,
                            "__pregel_read": None,
                            "__pregel_send": None,
                            "__pregel_call": None,
                            "__pregel_ensure_latest": True,
                            "__pregel_dedupe_tasks": True,
                            "__pregel_resuming": False,
                            "__pregel_previous": None,
                            "__pregel_store": None,
                            "__pregel_task_id": history[0].tasks[0].id,
                            "__pregel_scratchpad": {
                                "subgraph_counter": None,
                                "call_counter": None,
                                "interrupt_counter": None,
                                "get_null_resume": None,
                                "resume": [],
                            },
                            "checkpoint_id": None,
                            "checkpoint_map": {
                                "": history[0].config["configurable"]["checkpoint_id"]
                            },
                            "checkpoint_ns": history[0]
                            .tasks[0]
                            .state["configurable"]["checkpoint_ns"],
                            "thread_id": "1",
                        },
                        "metadata": AnyDict(),
                        "recursion_limit": 25,
                        "tags": [],
                    },
                    "input": {
                        "messages": [
                            _AnyIdHumanMessage(content="what's the weather in sf")
                        ],
                        "route": "weather",
                    },
                    "finally_send": [
                        {
                            "topic": topics.executor,
                            "value": {
                                "config": {
                                    "callbacks": None,
                                    "configurable": {
                                        "__pregel_dedupe_tasks": True,
                                        "__pregel_ensure_latest": True,
                                        "__pregel_resuming": False,
                                        "checkpoint_id": history[0].config[
                                            "configurable"
                                        ]["checkpoint_id"],
                                        "checkpoint_ns": "",
                                        "thread_id": "1",
                                    },
                                    "metadata": AnyDict(),
                                    "recursion_limit": 25,
                                    "tags": [],
                                },
                                "finally_send": None,
                                "task": {
                                    "id": history[0].tasks[0].id,
                                    "path": list(history[0].tasks[0].path),
                                },
                            },
                        }
                    ],
                }
            ]
            # child graph messages, until interrupted
            + [
                {
                    "config": {
                        "callbacks": None,
                        "configurable": {
                            "__pregel_checkpointer": None,
                            "__pregel_delegate": False,
                            "__pregel_read": None,
                            "__pregel_send": None,
                            "__pregel_call": None,
                            "__pregel_ensure_latest": True,
                            "__pregel_store": None,
                            "__pregel_dedupe_tasks": True,
                            "__pregel_resuming": False,
                            "__pregel_previous": None,
                            "__pregel_task_id": history[0].tasks[0].id,
                            "__pregel_scratchpad": {
                                "subgraph_counter": None,
                                "call_counter": None,
                                "interrupt_counter": None,
                                "get_null_resume": None,
                                "resume": [],
                            },
                            "checkpoint_id": c.config["configurable"]["checkpoint_id"],
                            "checkpoint_map": {
                                "": history[0].config["configurable"]["checkpoint_id"]
                            },
                            "checkpoint_ns": history[0]
                            .tasks[0]
                            .state["configurable"]["checkpoint_ns"],
                            "thread_id": "1",
                        },
                        "metadata": AnyDict(),
                        "recursion_limit": 25,
                        "tags": [],
                    },
                    "input": None,
                    "finally_send": [
                        {
                            "topic": topics.executor,
                            "value": {
                                "config": {
                                    "callbacks": None,
                                    "configurable": {
                                        "__pregel_dedupe_tasks": True,
                                        "__pregel_ensure_latest": True,
                                        "__pregel_resuming": False,
                                        "checkpoint_id": history[0].config[
                                            "configurable"
                                        ]["checkpoint_id"],
                                        "checkpoint_ns": "",
                                        "thread_id": "1",
                                    },
                                    "metadata": AnyDict(),
                                    "recursion_limit": 25,
                                    "tags": [],
                                },
                                "finally_send": None,
                                "task": {
                                    "id": history[0].tasks[0].id,
                                    "path": list(history[0].tasks[0].path),
                                },
                            },
                        }
                    ],
                }
                for c in reversed(child_history[1:])  # the last one wasn't executed
                # orchestrator messages appear only after tasks for that checkpoint
                # finish executing, ie. after executor sends message to resume checkpoint
                for _ in c.tasks
            ]
        )
    )
    assert (
        exec_msgs
        == (
            # outer graph tasks
            [
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
                    "finally_send": None,
                }
                for c in reversed(history)
                for t in c.tasks
            ]
            # child graph tasks
            + [
                {
                    "config": {
                        "callbacks": None,
                        "configurable": {
                            "__pregel_checkpointer": None,
                            "__pregel_delegate": False,
                            "__pregel_read": None,
                            "__pregel_send": None,
                            "__pregel_call": None,
                            "__pregel_ensure_latest": True,
                            "__pregel_dedupe_tasks": True,
                            "__pregel_store": None,
                            "__pregel_resuming": False,
                            "__pregel_task_id": history[0].tasks[0].id,
                            "__pregel_previous": None,
                            "__pregel_scratchpad": {
                                "subgraph_counter": None,
                                "call_counter": None,
                                "interrupt_counter": None,
                                "get_null_resume": None,
                                "resume": [],
                            },
                            "checkpoint_id": c.config["configurable"]["checkpoint_id"],
                            "checkpoint_map": {
                                "": history[0].config["configurable"]["checkpoint_id"]
                            },
                            "checkpoint_ns": history[0]
                            .tasks[0]
                            .state["configurable"]["checkpoint_ns"],
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
                    "finally_send": [
                        {
                            "topic": topics.executor,
                            "value": {
                                "config": {
                                    "callbacks": None,
                                    "configurable": {
                                        "__pregel_dedupe_tasks": True,
                                        "__pregel_ensure_latest": True,
                                        "__pregel_resuming": False,
                                        "checkpoint_id": history[0].config[
                                            "configurable"
                                        ]["checkpoint_id"],
                                        "checkpoint_ns": "",
                                        "thread_id": "1",
                                    },
                                    "metadata": AnyDict(),
                                    "recursion_limit": 25,
                                    "tags": [],
                                },
                                "finally_send": None,
                                "task": {
                                    "id": history[0].tasks[0].id,
                                    "path": list(history[0].tasks[0].path),
                                },
                            },
                        }
                    ],
                }
                for c in reversed(child_history[1:])  # the last one wasn't executed
                for t in c.tasks
            ]
        )
    )

    # resume the thread
    with DefaultProducer() as producer:
        producer.send(
            topics.orchestrator,
            value=serde.dumps(MessageToOrchestrator(input=None, config=config)),
        )
        producer.flush()

    orch_msgs, exec_msgs = drain_topics(topics, graph)

    # check final state
    state = graph.get_state(config)
    assert state.next == ()
    assert state.values == {
        "messages": [
            _AnyIdHumanMessage(content="what's the weather in sf"),
            _AnyIdAIMessage(content="I'ts sunny in San Francisco!"),
        ],
        "route": "weather",
    }

    # check outer history
    history = [c for c in graph.get_state_history(config)]
    assert len(history) == 4

    # check child history
    # accessing second to last checkpoint, since that's the one w/ subgraph task
    child_history = [c for c in graph.get_state_history(history[1].tasks[0].state)]
    assert len(child_history) == 4

    # check messages
    assert (
        orch_msgs
        == (
            # initial message to outer graph
            [MessageToOrchestrator(input=None, config=config)]
            # initial message to child graph
            + [
                {
                    "config": {
                        "callbacks": None,
                        "configurable": {
                            "__pregel_checkpointer": None,
                            "__pregel_delegate": False,
                            "__pregel_read": None,
                            "__pregel_send": None,
                            "__pregel_call": None,
                            "__pregel_ensure_latest": True,
                            "__pregel_dedupe_tasks": True,
                            "__pregel_store": None,
                            "__pregel_resuming": True,
                            "__pregel_previous": None,
                            "__pregel_task_id": history[1].tasks[0].id,
                            "__pregel_scratchpad": {
                                "subgraph_counter": None,
                                "call_counter": None,
                                "interrupt_counter": None,
                                "get_null_resume": None,
                                "resume": [],
                            },
                            "checkpoint_id": None,
                            "checkpoint_map": {
                                "": history[1].config["configurable"]["checkpoint_id"]
                            },
                            "checkpoint_ns": history[1]
                            .tasks[0]
                            .state["configurable"]["checkpoint_ns"],
                            "thread_id": "1",
                        },
                        "metadata": AnyDict(),
                        "recursion_limit": 25,
                        "tags": [],
                    },
                    "input": None,
                    "finally_send": [
                        {
                            "topic": topics.executor,
                            "value": {
                                "config": {
                                    "callbacks": None,
                                    "configurable": {
                                        "__pregel_dedupe_tasks": True,
                                        "__pregel_ensure_latest": True,
                                        "__pregel_resuming": True,
                                        "checkpoint_id": history[1].config[
                                            "configurable"
                                        ]["checkpoint_id"],
                                        "checkpoint_ns": "",
                                        "thread_id": "1",
                                    },
                                    "metadata": AnyDict(),
                                    "recursion_limit": 25,
                                    "tags": [],
                                },
                                "finally_send": None,
                                "task": {
                                    "id": history[1].tasks[0].id,
                                    "path": list(history[1].tasks[0].path),
                                },
                            },
                        }
                    ],
                }
            ]
            # child graph messages, from previous last checkpoint onwards
            + [
                {
                    "config": {
                        "callbacks": None,
                        "configurable": {
                            "__pregel_checkpointer": None,
                            "__pregel_delegate": False,
                            "__pregel_read": None,
                            "__pregel_send": None,
                            "__pregel_call": None,
                            "__pregel_ensure_latest": True,
                            "__pregel_dedupe_tasks": True,
                            "__pregel_store": None,
                            "__pregel_resuming": True,
                            "__pregel_previous": None,
                            "__pregel_task_id": history[1].tasks[0].id,
                            "__pregel_scratchpad": {
                                "subgraph_counter": None,
                                "call_counter": None,
                                "interrupt_counter": None,
                                "get_null_resume": None,
                                "resume": [],
                            },
                            "checkpoint_id": c.config["configurable"]["checkpoint_id"],
                            "checkpoint_map": {
                                "": history[1].config["configurable"]["checkpoint_id"]
                            },
                            "checkpoint_ns": history[1]
                            .tasks[0]
                            .state["configurable"]["checkpoint_ns"],
                            "thread_id": "1",
                        },
                        "metadata": AnyDict(),
                        "recursion_limit": 25,
                        "tags": [],
                    },
                    "input": None,
                    "finally_send": [
                        {
                            "topic": topics.executor,
                            "value": {
                                "config": {
                                    "callbacks": None,
                                    "configurable": {
                                        "__pregel_dedupe_tasks": True,
                                        "__pregel_ensure_latest": True,
                                        "__pregel_resuming": True,
                                        "checkpoint_id": history[1].config[
                                            "configurable"
                                        ]["checkpoint_id"],
                                        "checkpoint_ns": "",
                                        "thread_id": "1",
                                    },
                                    "metadata": AnyDict(),
                                    "recursion_limit": 25,
                                    "tags": [],
                                },
                                "finally_send": None,
                                "task": {
                                    "id": history[1].tasks[0].id,
                                    "path": list(history[1].tasks[0].path),
                                },
                            },
                        }
                    ],
                }
                for c in reversed(child_history[:2])
                for _ in c.tasks
            ]
            # outer graph messages, from previous last checkpoint onwards
            + [
                {
                    "config": {
                        "callbacks": None,
                        "configurable": {
                            "__pregel_ensure_latest": True,
                            "__pregel_dedupe_tasks": True,
                            "__pregel_resuming": True,
                            "checkpoint_id": c.config["configurable"]["checkpoint_id"],
                            "checkpoint_ns": "",
                            "thread_id": "1",
                        },
                        "metadata": AnyDict(),
                        "recursion_limit": 25,
                        "tags": [],
                    },
                    "input": None,
                    "finally_send": None,
                }
                for c in reversed(history[:2])
                for _ in c.tasks
            ]
        )
    )
    assert (
        exec_msgs
        == (
            # outer graph tasks
            [
                {
                    "config": {
                        "callbacks": None,
                        "configurable": {
                            "__pregel_ensure_latest": True,
                            "__pregel_dedupe_tasks": True,
                            "__pregel_resuming": True,
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
                    "finally_send": None,
                }
                for c in reversed(history[:2])
                for t in c.tasks
            ]
            # child graph tasks
            + [
                {
                    "config": {
                        "callbacks": None,
                        "configurable": {
                            "__pregel_checkpointer": None,
                            "__pregel_delegate": False,
                            "__pregel_read": None,
                            "__pregel_send": None,
                            "__pregel_call": None,
                            "__pregel_ensure_latest": True,
                            "__pregel_dedupe_tasks": True,
                            "__pregel_resuming": True,
                            "__pregel_previous": None,
                            "__pregel_store": None,
                            "__pregel_task_id": history[1].tasks[0].id,
                            "__pregel_scratchpad": {
                                "subgraph_counter": None,
                                "call_counter": None,
                                "interrupt_counter": None,
                                "get_null_resume": None,
                                "resume": [],
                            },
                            "checkpoint_id": c.config["configurable"]["checkpoint_id"],
                            "checkpoint_map": {
                                "": history[1].config["configurable"]["checkpoint_id"]
                            },
                            "checkpoint_ns": history[1]
                            .tasks[0]
                            .state["configurable"]["checkpoint_ns"],
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
                    "finally_send": [
                        {
                            "topic": topics.executor,
                            "value": {
                                "config": {
                                    "callbacks": None,
                                    "configurable": {
                                        "__pregel_dedupe_tasks": True,
                                        "__pregel_ensure_latest": True,
                                        "__pregel_resuming": True,
                                        "checkpoint_id": history[1].config[
                                            "configurable"
                                        ]["checkpoint_id"],
                                        "checkpoint_ns": "",
                                        "thread_id": "1",
                                    },
                                    "metadata": AnyDict(),
                                    "recursion_limit": 25,
                                    "tags": [],
                                },
                                "finally_send": None,
                                "task": {
                                    "id": history[1].tasks[0].id,
                                    "path": list(history[1].tasks[0].path),
                                },
                            },
                        }
                    ],
                }
                for c in reversed(child_history[:2])
                for t in c.tasks
            ]
            # "finally" tasks
            + [
                {
                    "config": {
                        "callbacks": None,
                        "configurable": {
                            "__pregel_dedupe_tasks": True,
                            "__pregel_ensure_latest": True,
                            "__pregel_resuming": True,
                            "checkpoint_id": history[1].config["configurable"][
                                "checkpoint_id"
                            ],
                            "checkpoint_ns": "",
                            "thread_id": "1",
                        },
                        "metadata": AnyDict(),
                        "recursion_limit": 25,
                        "tags": [],
                    },
                    "finally_send": None,
                    "task": {
                        "id": history[1].tasks[0].id,
                        "path": list(history[1].tasks[0].path),
                    },
                }
            ]
        )
    )
