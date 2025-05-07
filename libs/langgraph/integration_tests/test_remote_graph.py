"""Test RemoteGraph against an actual server."""

from typing import Annotated

import pytest
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict

from integration_tests.example_app.example_graph import app
from langgraph.graph import StateGraph, add_messages
from langgraph.pregel import Pregel
from langgraph.pregel.remote import RemoteGraph

pytestmark = pytest.mark.anyio


@pytest.fixture
def remote_graph() -> RemoteGraph:
    return RemoteGraph("app", url="http://localhost:2024")


@pytest.fixture
def nested_remote_graph(remote_graph: RemoteGraph) -> Pregel:
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    async def nested(state: State) -> State:
        return await remote_graph.ainvoke(state)

    return StateGraph(State).add_node(nested).add_edge("__start__", "nested").compile()


@pytest.fixture
async def nested_graph() -> Pregel:
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    async def nested(state: State) -> State:
        return await app.ainvoke(state)

    return StateGraph(State).add_node(nested).add_edge("__start__", "nested").compile()


async def test_remote_graph(remote_graph: RemoteGraph) -> None:
    # Basic smoke test of the remote graph
    response = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "hello"}]}
    )
    assert response == {
        "content": "answer",
        "additional_kwargs": {},
        "response_metadata": {},
        "type": "ai",
        "name": None,
        "id": "ai3",
        "example": False,
        "tool_calls": [],
        "invalid_tool_calls": [],
        "usage_metadata": None,
    }


async def test_remote_graph_stream_messages_tuple(
    nested_graph: Pregel, nested_remote_graph: Pregel
) -> None:
    events = []
    async for event, metadata in nested_remote_graph.astream(
        {"messages": [{"role": "user", "content": "hello"}]},
        stream_mode="messages-tuple",
        subgraphs=True,
    ):
        events.append((metadata, event))

    inmem_events = []
    async for event, metadata in nested_graph.astream(
        {"messages": [{"role": "user", "content": "hello"}]},
        # messages is the local equivalent. Sad.
        stream_mode="messages",
        subgraphs=True,
    ):
        inmem_events.append((metadata, event))
    assert len(events) == len(inmem_events)
