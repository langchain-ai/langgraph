import operator
from typing import (
    Annotated,
    Literal,
    Union,
)

import pytest

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import START
from langgraph.errors import NodeInterrupt
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.scheduler.kafka import serde
from langgraph.scheduler.kafka.default_sync import DefaultProducer
from langgraph.scheduler.kafka.types import MessageToOrchestrator, Topics
from langgraph.types import Command, Send
from tests.any import AnyDict
from tests.drain import drain_topics

pytestmark = pytest.mark.anyio


def mk_push_graph(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    # copied from test_send_dedupe_on_resume

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
            self.__name__ = name

        def __call__(self, state):
            self.ticks += 1
            update = (
                [self.name]
                if isinstance(state, list)
                else ["|".join((self.name, str(state)))]
            )
            if isinstance(state, Command):
                return state.copy(update=update)
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

    return builder.compile(checkpointer=checkpointer)


@pytest.mark.skip("TODO: re-enable in next PR")
def test_push_graph(topics: Topics, acheckpointer: BaseCheckpointSaver) -> None:
    input = ["0"]
    config = {"configurable": {"thread_id": "1"}}
    graph = mk_push_graph(acheckpointer)
    graph_compare = mk_push_graph(acheckpointer)

    # start a new run
    with DefaultProducer() as producer:
        producer.send(
            topics.orchestrator,
            value=serde.dumps(MessageToOrchestrator(input=input, config=config)),
        )
        producer.flush()

    # drain topics
    orch_msgs, exec_msgs = drain_topics(topics, graph)

    # check state
    state = graph.get_state(config)
    assert all(not t.error for t in state.tasks)
    assert state.next == ("flaky",)
    assert (
        state.values
        == graph_compare.invoke(input, {"configurable": {"thread_id": "2"}})
        == [
            "0",
            "1",
            "2|Control(goto=Send(node='2', arg=3))",
            "2|Control(goto=Send(node='flaky', arg=4))",
            "2|3",
        ]
    )

    # check history
    history = [c for c in graph.get_state_history(config)]
    assert len(history) == 2

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
            "finally_send": None,
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
                "path": _convert_path(t.path),
            },
            "finally_send": None,
        }
        for c in reversed(history)
        for t in c.tasks
    ]

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
    assert (
        state.values
        == graph_compare.invoke(None, {"configurable": {"thread_id": "2"}})
        == [
            "0",
            "1",
            "2|Control(goto=Send(node='2', arg=3))",
            "2|Control(goto=Send(node='flaky', arg=4))",
            "2|3",
            "flaky|4",
            "3",
            "3.1",
        ]
    )

    # check history
    history = [c for c in graph.get_state_history(config)]
    assert len(history) == 4

    # check executions
    # node "2" doesn't get called again, as we recover writes saved before
    assert graph.builder.nodes["2"].runnable.func.ticks == 3
    # node "flaky" gets called again, as it was interrupted
    assert graph.builder.nodes["flaky"].runnable.func.ticks == 2


def _convert_path(
    path: tuple[Union[str, int, tuple], ...],
) -> list[Union[str, int, list]]:
    return list(_convert_path(p) if isinstance(p, tuple) else p for p in path)
