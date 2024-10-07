import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import Graph as DrawableGraph

from langgraph.pregel.remote import RemotePregel
from langgraph.pregel.types import StateSnapshot


def test_with_config():
    # set up test
    remote_pregel = RemotePregel(
        client=AsyncMock(id="async_client_1"),
        sync_client=MagicMock(id="sync_client_1"),
        graph_id="test_graph_id",
    )

    # call method / assertions
    new_remote_pregel = remote_pregel.with_config()

    assert new_remote_pregel.client.id == "async_client_1"
    assert new_remote_pregel.sync_client.id == "sync_client_1"
    assert new_remote_pregel.graph_id == "test_graph_id"


def test_get_graph():
    # set up test
    mock_sync_client = MagicMock()
    mock_sync_client.assistants.get_graph.return_value = {
        "nodes": [
            {"id": "__start__", "type": "schema", "data": "__start__"},
            {"id": "__end__", "type": "schema", "data": "__end__"},
            {
                "id": "agent",
                "type": "runnable",
                "data": {
                    "id": ["langgraph", "utils", "RunnableCallable"],
                    "name": "agent",
                },
            },
        ],
        "edges": [
            {"source": "__start__", "target": "agent"},
            {"source": "agent", "target": "__end__"},
        ],
    }

    remote_pregel = RemotePregel(
        client=AsyncMock(), sync_client=mock_sync_client, graph_id="test_graph_id"
    )

    # call method / assertions
    drawable_graph = remote_pregel.get_graph()

    assert drawable_graph.nodes == [
        {"id": "__start__", "type": "schema", "data": "__start__"},
        {"id": "__end__", "type": "schema", "data": "__end__"},
        {
            "id": "agent",
            "type": "runnable",
            "data": {"id": ["langgraph", "utils", "RunnableCallable"], "name": "agent"},
        },
    ]

    assert drawable_graph.edges == [
        {"source": "__start__", "target": "agent"},
        {"source": "agent", "target": "__end__"},
    ]


@pytest.mark.anyio
async def test_aget_graph():
    # set up test
    mock_async_client = AsyncMock()
    mock_async_client.assistants.get_graph.return_value = {
        "nodes": [
            {"id": "__start__", "type": "schema", "data": "__start__"},
            {"id": "__end__", "type": "schema", "data": "__end__"},
            {
                "id": "agent",
                "type": "runnable",
                "data": {
                    "id": ["langgraph", "utils", "RunnableCallable"],
                    "name": "agent",
                },
            },
        ],
        "edges": [
            {"source": "__start__", "target": "agent"},
            {"source": "agent", "target": "__end__"},
        ],
    }

    remote_pregel = RemotePregel(
        client=mock_async_client, sync_client=MagicMock(), graph_id="test_graph_id"
    )

    # call method / assertions
    drawable_graph = await remote_pregel.aget_graph()

    assert drawable_graph.nodes == [
        {"id": "__start__", "type": "schema", "data": "__start__"},
        {"id": "__end__", "type": "schema", "data": "__end__"},
        {
            "id": "agent",
            "type": "runnable",
            "data": {"id": ["langgraph", "utils", "RunnableCallable"], "name": "agent"},
        },
    ]

    assert drawable_graph.edges == [
        {"source": "__start__", "target": "agent"},
        {"source": "agent", "target": "__end__"},
    ]


def test_get_subgraphs():
    # set up test
    mock_sync_client = MagicMock(id=2)
    mock_sync_client.assistants.get_subgraphs.return_value = {
        "namespace_1": {
            "graph_id": "test_graph_id_2",
            "input_schema": {},
            "output_schema": {},
            "state_schema": {},
            "config_schema": {},
        },
        "namespace_2": {
            "graph_id": "test_graph_id_3",
            "input_schema": {},
            "output_schema": {},
            "state_schema": {},
            "config_schema": {},
        },
    }

    remote_pregel = RemotePregel(AsyncMock(id=1), mock_sync_client, "test_graph_id_1")

    # call method / assertions
    subgraphs = list(remote_pregel.get_subgraphs())
    assert len(subgraphs) == 2

    subgraph_1 = subgraphs[0]
    ns_1 = subgraph_1[0]
    remote_pregel_1: RemotePregel = subgraph_1[1]
    assert ns_1 == "namespace_1"
    assert remote_pregel_1.graph_id == "test_graph_id_2"
    assert remote_pregel_1.client.id == 1
    assert remote_pregel_1.sync_client.id == 2

    subgraph_2 = subgraphs[1]
    ns_2 = subgraph_2[0]
    remote_pregel_2: RemotePregel = subgraph_2[1]
    assert ns_2 == "namespace_2"
    assert remote_pregel_2.graph_id == "test_graph_id_3"
    assert remote_pregel_2.client.id == 1
    assert remote_pregel_2.sync_client.id == 2


@pytest.mark.anyio
async def test_aget_subgraphs():
    # set up test
    mock_async_client = AsyncMock(id=1)
    mock_async_client.assistants.get_subgraphs.return_value = {
        "namespace_1": {
            "graph_id": "test_graph_id_2",
            "input_schema": {},
            "output_schema": {},
            "state_schema": {},
            "config_schema": {},
        },
        "namespace_2": {
            "graph_id": "test_graph_id_3",
            "input_schema": {},
            "output_schema": {},
            "state_schema": {},
            "config_schema": {},
        },
    }

    remote_pregel = RemotePregel(mock_async_client, MagicMock(id=2), "test_graph_id_1")

    # call method / assertions
    subgraphs = []
    async for subgraph in remote_pregel.aget_subgraphs():
        subgraphs.append(subgraph)
    assert len(subgraphs) == 2

    subgraph_1 = subgraphs[0]
    ns_1 = subgraph_1[0]
    remote_pregel_1: RemotePregel = subgraph_1[1]
    assert ns_1 == "namespace_1"
    assert remote_pregel_1.graph_id == "test_graph_id_2"
    assert remote_pregel_1.client.id == 1
    assert remote_pregel_1.sync_client.id == 2

    subgraph_2 = subgraphs[1]
    ns_2 = subgraph_2[0]
    remote_pregel_2: RemotePregel = subgraph_2[1]
    assert ns_2 == "namespace_2"
    assert remote_pregel_2.graph_id == "test_graph_id_3"
    assert remote_pregel_2.client.id == 1
    assert remote_pregel_2.sync_client.id == 2


def test_get_state():
    # set up test
    mock_sync_client = MagicMock()
    state_mock = {
        "values": {"messages": [{"type": "human", "content": "hello"}]},
        "next": None,
        "checkpoint": {
            "thread_id": "thread_1",
            "checkpoint_ns": "ns",
            "checkpoint_id": "checkpoint_1",
            "checkpoint_map": {},
        },
        "metadata": {},
        "created_at": "timestamp",
        "parent_checkpoint": None,
        "tasks": [],
    }
    mock_sync_client.threads.get_state.return_value = state_mock

    # call method / assertions
    remote_pregel = RemotePregel(AsyncMock(), mock_sync_client, "test_graph_id")

    config = {"configurable": {"thread_id": "thread1"}}
    state_snapshot = remote_pregel.get_state(config)

    assert state_snapshot == StateSnapshot(
        values={"messages": [{"type": "human", "content": "hello"}]},
        next=None,
        config={
            "configurable": {
                "thread_id": "thread_1",
                "checkpoint_ns": "ns",
                "checkpoint_id": "checkpoint_1",
                "checkpoint_map": {},
            }
        },
        metadata={},
        created_at="timestamp",
        parent_config=None,
        tasks=[],
    )


@pytest.mark.anyio
async def test_aget_state():
    mock_async_client = AsyncMock()
    state_mock = {
        "values": {"messages": [{"type": "human", "content": "hello"}]},
        "next": None,
        "checkpoint": {
            "thread_id": "thread_1",
            "checkpoint_ns": "ns",
            "checkpoint_id": "checkpoint_2",
            "checkpoint_map": {},
        },
        "metadata": {},
        "created_at": "timestamp",
        "parent_checkpoint": {
            "thread_id": "thread_1",
            "checkpoint_ns": "ns",
            "checkpoint_id": "checkpoint_1",
            "checkpoint_map": {},
        },
        "tasks": [],
    }
    mock_async_client.threads.get_state.return_value = state_mock

    # call method / assertions
    remote_pregel = RemotePregel(mock_async_client, MagicMock(), "test_graph_id")

    config = {"configurable": {"thread_id": "thread1"}}
    state_snapshot = await remote_pregel.aget_state(config)

    assert state_snapshot == StateSnapshot(
        values={"messages": [{"type": "human", "content": "hello"}]},
        next=None,
        config={
            "configurable": {
                "thread_id": "thread_1",
                "checkpoint_ns": "ns",
                "checkpoint_id": "checkpoint_2",
                "checkpoint_map": {},
            }
        },
        metadata={},
        created_at="timestamp",
        parent_config={
            "configurable": {
                "thread_id": "thread_1",
                "checkpoint_ns": "ns",
                "checkpoint_id": "checkpoint_1",
                "checkpoint_map": {},
            }
        },
        tasks=[],
    )
