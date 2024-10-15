from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.runnables.graph import (
    Edge as DrawableEdge,
)
from langchain_core.runnables.graph import (
    Node as DrawableNode,
)

from langgraph.pregel.remote import RemoteGraph
from langgraph.pregel.types import StateSnapshot


def test_with_config():
    # set up test
    remote_pregel = RemoteGraph(
        graph_id="test_graph_id",
        config={
            "configurable": {
                "foo": "bar",
                "thread_id": "thread_id_1",
            }
        },
    )

    # call method / assertions
    config = {"configurable": {"hello": "world"}}
    remote_pregel_copy = remote_pregel.with_config(config)

    # assert that a copy was returned
    assert remote_pregel_copy != remote_pregel
    # assert that configs were merged
    assert remote_pregel_copy.config == {
        "configurable": {
            "foo": "bar",
            "thread_id": "thread_id_1",
            "hello": "world",
        }
    }


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

    remote_pregel = RemoteGraph(sync_client=mock_sync_client, graph_id="test_graph_id")

    # call method / assertions
    drawable_graph = remote_pregel.get_graph()

    assert drawable_graph.nodes == {
        "__start__": DrawableNode(
            id="__start__", name="", data="__start__", metadata=None
        ),
        "__end__": DrawableNode(id="__end__", name="", data="__end__", metadata=None),
        "agent": DrawableNode(
            id="agent",
            name="",
            data={"id": ["langgraph", "utils", "RunnableCallable"], "name": "agent"},
            metadata=None,
        ),
    }

    assert drawable_graph.edges == [
        DrawableEdge(source="__start__", target="agent"),
        DrawableEdge(source="agent", target="__end__"),
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

    remote_pregel = RemoteGraph(client=mock_async_client, graph_id="test_graph_id")

    # call method / assertions
    drawable_graph = await remote_pregel.aget_graph()

    assert drawable_graph.nodes == {
        "__start__": DrawableNode(
            id="__start__", name="", data="__start__", metadata=None
        ),
        "__end__": DrawableNode(id="__end__", name="", data="__end__", metadata=None),
        "agent": DrawableNode(
            id="agent",
            name="",
            data={"id": ["langgraph", "utils", "RunnableCallable"], "name": "agent"},
            metadata=None,
        ),
    }

    assert drawable_graph.edges == [
        DrawableEdge(source="__start__", target="agent"),
        DrawableEdge(source="agent", target="__end__"),
    ]


def test_get_subgraphs():
    # set up test
    mock_sync_client = MagicMock()
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

    remote_pregel = RemoteGraph(
        sync_client=mock_sync_client, graph_id="test_graph_id_1"
    )

    # call method / assertions
    subgraphs = list(remote_pregel.get_subgraphs())
    assert len(subgraphs) == 2

    subgraph_1 = subgraphs[0]
    ns_1 = subgraph_1[0]
    remote_pregel_1: RemoteGraph = subgraph_1[1]
    assert ns_1 == "namespace_1"
    assert remote_pregel_1.graph_id == "test_graph_id_2"

    subgraph_2 = subgraphs[1]
    ns_2 = subgraph_2[0]
    remote_pregel_2: RemoteGraph = subgraph_2[1]
    assert ns_2 == "namespace_2"
    assert remote_pregel_2.graph_id == "test_graph_id_3"


@pytest.mark.anyio
async def test_aget_subgraphs():
    # set up test
    mock_async_client = AsyncMock()
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

    remote_pregel = RemoteGraph(
        client=mock_async_client,
        graph_id="test_graph_id_1",
    )

    # call method / assertions
    subgraphs = []
    async for subgraph in remote_pregel.aget_subgraphs():
        subgraphs.append(subgraph)
    assert len(subgraphs) == 2

    subgraph_1 = subgraphs[0]
    ns_1 = subgraph_1[0]
    remote_pregel_1: RemoteGraph = subgraph_1[1]
    assert ns_1 == "namespace_1"
    assert remote_pregel_1.graph_id == "test_graph_id_2"

    subgraph_2 = subgraphs[1]
    ns_2 = subgraph_2[0]
    remote_pregel_2: RemoteGraph = subgraph_2[1]
    assert ns_2 == "namespace_2"
    assert remote_pregel_2.graph_id == "test_graph_id_3"


def test_get_state():
    # set up test
    mock_sync_client = MagicMock()
    mock_sync_client.threads.get_state.return_value = {
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

    # call method / assertions
    remote_pregel = RemoteGraph(sync_client=mock_sync_client, graph_id="test_graph_id")

    config = {"configurable": {"thread_id": "thread1"}}
    state_snapshot = remote_pregel.get_state(config)

    assert state_snapshot == StateSnapshot(
        values={"messages": [{"type": "human", "content": "hello"}]},
        next=(),
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
        tasks=(),
    )


@pytest.mark.anyio
async def test_aget_state():
    mock_async_client = AsyncMock()
    mock_async_client.threads.get_state.return_value = {
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

    # call method / assertions
    remote_pregel = RemoteGraph(client=mock_async_client, graph_id="test_graph_id")

    config = {"configurable": {"thread_id": "thread1"}}
    state_snapshot = await remote_pregel.aget_state(config)

    assert state_snapshot == StateSnapshot(
        values={"messages": [{"type": "human", "content": "hello"}]},
        next=(),
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
        tasks=(),
    )


def test_get_state_history():
    # set up test
    mock_sync_client = MagicMock()
    mock_sync_client.threads.get_history.return_value = [
        {
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
    ]

    # call method / assertions
    remote_pregel = RemoteGraph(sync_client=mock_sync_client, graph_id="test_graph_id")

    config = {"configurable": {"thread_id": "thread1"}}
    state_history_snapshot = list(
        remote_pregel.get_state_history(config, filter=None, before=None, limit=None)
    )

    assert len(state_history_snapshot) == 1
    assert state_history_snapshot[0] == StateSnapshot(
        values={"messages": [{"type": "human", "content": "hello"}]},
        next=(),
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
        tasks=(),
    )


@pytest.mark.anyio
async def test_aget_state_history():
    # set up test
    mock_async_client = AsyncMock()
    mock_async_client.threads.get_history.return_value = [
        {
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
    ]

    # call method / assertions
    remote_pregel = RemoteGraph(client=mock_async_client, graph_id="test_graph_id")

    config = {"configurable": {"thread_id": "thread1"}}
    state_history_snapshot = []
    async for state_snapshot in remote_pregel.aget_state_history(
        config, filter=None, before=None, limit=None
    ):
        state_history_snapshot.append(state_snapshot)

    assert len(state_history_snapshot) == 1
    assert state_history_snapshot[0] == StateSnapshot(
        values={"messages": [{"type": "human", "content": "hello"}]},
        next=(),
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
        tasks=(),
    )


def test_update_state():
    # set up test
    mock_sync_client = MagicMock()
    mock_sync_client.threads.update_state.return_value = {
        "checkpoint": {
            "thread_id": "thread_1",
            "checkpoint_ns": "ns",
            "checkpoint_id": "checkpoint_1",
            "checkpoint_map": {},
        }
    }

    # call method / assertions
    remote_pregel = RemoteGraph(sync_client=mock_sync_client, graph_id="test_graph_id")

    config = {"configurable": {"thread_id": "thread1"}}
    response = remote_pregel.update_state(config, {"key": "value"})

    assert response == {
        "configurable": {
            "thread_id": "thread_1",
            "checkpoint_ns": "ns",
            "checkpoint_id": "checkpoint_1",
            "checkpoint_map": {},
        }
    }


@pytest.mark.anyio
async def test_aupdate_state():
    # set up test
    mock_async_client = AsyncMock()
    mock_async_client.threads.update_state.return_value = {
        "checkpoint": {
            "thread_id": "thread_1",
            "checkpoint_ns": "ns",
            "checkpoint_id": "checkpoint_1",
            "checkpoint_map": {},
        }
    }

    # call method / assertions
    remote_pregel = RemoteGraph(client=mock_async_client, graph_id="test_graph_id")

    config = {"configurable": {"thread_id": "thread1"}}
    response = await remote_pregel.aupdate_state(config, {"key": "value"})

    assert response == {
        "configurable": {
            "thread_id": "thread_1",
            "checkpoint_ns": "ns",
            "checkpoint_id": "checkpoint_1",
            "checkpoint_map": {},
        }
    }


def test_stream():
    # set up test
    mock_sync_client = MagicMock()
    mock_sync_client.runs.stream.return_value = [
        {"chunk": "data1"},
        {"chunk": "data2"},
        {"chunk": "data3"},
    ]

    # call method / assertions
    remote_pregel = RemoteGraph(sync_client=mock_sync_client, graph_id="test_graph_id")

    config = {"configurable": {"thread_id": "thread_1"}}
    result = list(remote_pregel.stream({"input": "data"}, config))
    assert result == [{"chunk": "data1"}, {"chunk": "data2"}, {"chunk": "data3"}]


@pytest.mark.anyio
async def test_astream():
    # set up test
    mock_async_client = MagicMock()
    async_iter = MagicMock()
    async_iter.__aiter__.return_value = [
        {"chunk": "data1"},
        {"chunk": "data2"},
        {"chunk": "data3"},
    ]
    mock_async_client.runs.stream.return_value = async_iter

    # call method / assertions
    remote_pregel = RemoteGraph(client=mock_async_client, graph_id="test_graph_id")

    config = {"configurable": {"thread_id": "thread_1"}}
    chunks = []
    async for chunk in remote_pregel.astream({"input": "data"}, config):
        chunks.append(chunk)
    assert chunks == [{"chunk": "data1"}, {"chunk": "data2"}, {"chunk": "data3"}]


def test_invoke():
    # set up test
    mock_sync_client = MagicMock()
    mock_sync_client.runs.wait.return_value = {
        "values": {"messages": [{"type": "human", "content": "world"}]}
    }

    # call method / assertions
    remote_pregel = RemoteGraph(sync_client=mock_sync_client, graph_id="test_graph_id")

    config = {"configurable": {"thread_id": "thread_1"}}
    result = remote_pregel.invoke(
        {"input": {"messages": [{"type": "human", "content": "hello"}]}}, config
    )

    assert result == {"values": {"messages": [{"type": "human", "content": "world"}]}}


@pytest.mark.anyio
async def test_ainvoke():
    # set up test
    mock_async_client = AsyncMock()
    mock_async_client.runs.wait.return_value = {
        "values": {"messages": [{"type": "human", "content": "world"}]}
    }

    # call method / assertions
    remote_pregel = RemoteGraph(client=mock_async_client, graph_id="test_graph_id")

    config = {"configurable": {"thread_id": "thread_1"}}
    result = await remote_pregel.ainvoke(
        {"input": {"messages": [{"type": "human", "content": "hello"}]}}, config
    )

    assert result == {"values": {"messages": [{"type": "human", "content": "world"}]}}


@pytest.mark.skip("Unskip this test to manually test the LangGraph Cloud integration")
@pytest.mark.anyio
async def test_langgraph_cloud_integration():
    from langgraph_sdk.client import get_client, get_sync_client

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph

    # create RemotePregel instance
    client = get_client()
    sync_client = get_sync_client()
    remote_pregel = RemoteGraph(
        client=client, sync_client=sync_client, graph_id="agent"
    )

    # define graph
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", remote_pregel)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    app = workflow.compile(checkpointer=MemorySaver())

    # test invocation
    input = {
        "messages": [
            {
                "role": "human",
                "content": "Hello world!",
            }
        ]
    }

    # test invoke
    response = app.invoke(
        input,
        config={"configurable": {"thread_id": "2dc3e3e7-39ac-4597-aa57-4404b944e82a"}},
    )
    print("response:", response["messages"][-1].content)

    # test stream
    async for chunk in app.astream(
        input,
        config={"configurable": {"thread_id": "2dc3e3e7-39ac-4597-aa57-4404b944e82a"}},
        subgraphs=True,
        stream_mode=["debug", "messages"],
    ):
        print("chunk:", chunk)

    # test stream events
    async for chunk in remote_pregel.astream_events(
        input,
        config={"configurable": {"thread_id": "2dc3e3e7-39ac-4597-aa57-4404b944e82a"}},
        version="v2",
        subgraphs=True,
        stream_mode=[],
    ):
        print("chunk:", chunk)

    # test get state
    state_snapshot = await remote_pregel.aget_state(
        config={"configurable": {"thread_id": "2dc3e3e7-39ac-4597-aa57-4404b944e82a"}},
        subgraphs=True,
    )
    print("state snapshot:", state_snapshot)

    # test update state
    response = await remote_pregel.aupdate_state(
        config={"configurable": {"thread_id": "6645e002-ed50-4022-92a3-d0d186fdf812"}},
        values={
            "messages": [
                {
                    "role": "ai",
                    "content": "Hello world again!",
                }
            ]
        },
    )
    print("response:", response)

    # test get history
    async for state in remote_pregel.aget_state_history(
        config={"configurable": {"thread_id": "2dc3e3e7-39ac-4597-aa57-4404b944e82a"}},
    ):
        print("state snapshot:", state)

    # test get graph
    remote_pregel.graph_id = "fe096781-5601-53d2-b2f6-0d3403f7e9ca"  # must be UUID
    graph = await remote_pregel.aget_graph(xray=True)
    print("graph:", graph)

    # test get subgraphs
    remote_pregel.graph_id = "fe096781-5601-53d2-b2f6-0d3403f7e9ca"  # must be UUID
    async for name, pregel in remote_pregel.aget_subgraphs():
        print("name:", name)
        print("pregel:", pregel)
