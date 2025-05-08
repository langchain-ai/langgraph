"""Test RemoteGraph against an actual server."""

import re
import sys
from typing import Annotated

import pytest
from langchain_core.messages import AnyMessage, BaseMessage
from typing_extensions import TypedDict

from integration_tests.example_app.example_graph import app
from langgraph.graph import StateGraph, add_messages
from langgraph.pregel import Pregel
from langgraph.pregel.remote import RemoteGraph

pytestmark = pytest.mark.anyio

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


@pytest.fixture
def remote_graph() -> RemoteGraph:
    return RemoteGraph("app", url="http://localhost:2024")


@pytest.fixture
def nested_remote_graph(remote_graph: RemoteGraph) -> Pregel:
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    return (
        StateGraph(State)
        .add_node("nested", remote_graph)
        .add_edge("__start__", "nested")
        .compile(name="nested_remote_graph")
    )


@pytest.fixture
async def nested_graph() -> Pregel:
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    return (
        StateGraph(State)
        .add_node("nested", app)
        .add_edge("__start__", "nested")
        .compile(name="nested_graph")
    )


def get_message_dict(msg: BaseMessage | dict):
    # just get the core stuff from within the message
    if isinstance(msg, dict):
        return {
            "content": msg.get("content"),
            "type": msg.get("type"),
            "name": msg.get("name"),
            "tool_calls": msg.get("tool_calls"),
            "invalid_tool_calls": msg.get("invalid_tool_calls"),
        }
    return {
        "content": msg.content,
        "type": msg.type,
        "name": msg.name,
        "tool_calls": getattr(msg, "tool_calls", None),
        "invalid_tool_calls": getattr(msg, "invalid_tool_calls", None),
    }


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


class monotonic_uid:
    def __init__(self):
        self._uid = 0

    def __call__(self, match=None):
        val = self._uid
        self._uid += 1
        hexval = f"{val:032x}"
        uuid_str = f"{hexval[:8]}-{hexval[8:12]}-{hexval[12:16]}-{hexval[16:20]}-{hexval[20:32]}"
        return uuid_str


uid_pattern = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


@NEEDS_CONTEXTVARS
async def test_remote_graph_stream_messages_tuple(
    nested_graph: Pregel, nested_remote_graph: Pregel
) -> None:
    events = []
    namespaces = []
    uid_generator = monotonic_uid()
    async for ns, messages in nested_remote_graph.astream(
        {"messages": [{"role": "user", "content": "hello"}]},
        stream_mode="messages",
        subgraphs=True,
    ):
        events.extend(messages)
        namespaces.append(
            tuple(uid_pattern.sub(uid_generator, ns_part) for ns_part in ns)
        )
    inmem_events = []
    inmem_namespaces = []
    uid_generator = monotonic_uid()
    async for ns, messages in nested_graph.astream(
        {"messages": [{"role": "user", "content": "hello"}]},
        stream_mode="messages",
        subgraphs=True,
    ):
        inmem_events.extend(messages)
        inmem_namespaces.append(
            tuple(uid_pattern.sub(uid_generator, ns_part) for ns_part in ns)
        )
    assert len(events) == len(inmem_events)
    assert len(namespaces) == len(inmem_namespaces)

    coerced_events = [get_message_dict(e) for e in events]
    coerced_inmem_events = [get_message_dict(e) for e in inmem_events]
    assert coerced_events == coerced_inmem_events
    # TODO: Fix the namespace matching in the next api release.
    # assert namespaces == inmem_namespaces
