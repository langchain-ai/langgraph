import re
import sys
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import langsmith as ls
import pytest
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import Edge as DrawableEdge
from langchain_core.runnables.graph import Node as DrawableNode
from langgraph_sdk.schema import StreamPart
from typing_extensions import TypedDict

from langgraph.errors import GraphInterrupt
from langgraph.graph import StateGraph, add_messages
from langgraph.pregel import Pregel
from langgraph.pregel.remote import RemoteGraph
from langgraph.types import Interrupt, StateSnapshot
from tests.any_str import AnyStr
from tests.conftest import NO_DOCKER
from tests.example_app.example_graph import app

if NO_DOCKER:
    pytest.skip(
        "Skipping tests that require Docker. Unset NO_DOCKER to run them.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.anyio

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)

SKIP_PYTHON_314 = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="Not yet testing Python 3.14 with the server bc of dependency limits on the api side",
)


def test_with_config():
    # set up test
    remote_pregel = RemoteGraph(
        "test_graph_id",
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
                    "name": "agent_1",
                },
            },
        ],
        "edges": [
            {"source": "__start__", "target": "agent"},
            {"source": "agent", "target": "__end__"},
        ],
    }

    remote_pregel = RemoteGraph("test_graph_id", sync_client=mock_sync_client)

    # call method / assertions
    drawable_graph = remote_pregel.get_graph()

    assert drawable_graph.nodes == {
        "__start__": DrawableNode(
            id="__start__", name="__start__", data="__start__", metadata=None
        ),
        "__end__": DrawableNode(
            id="__end__", name="__end__", data="__end__", metadata=None
        ),
        "agent": DrawableNode(
            id="agent",
            name="agent_1",
            data={"id": ["langgraph", "utils", "RunnableCallable"], "name": "agent_1"},
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
                    "name": "agent_1",
                },
            },
        ],
        "edges": [
            {"source": "__start__", "target": "agent"},
            {"source": "agent", "target": "__end__"},
        ],
    }

    remote_pregel = RemoteGraph("test_graph_id", client=mock_async_client)

    # call method / assertions
    drawable_graph = await remote_pregel.aget_graph()

    assert drawable_graph.nodes == {
        "__start__": DrawableNode(
            id="__start__", name="__start__", data="__start__", metadata=None
        ),
        "__end__": DrawableNode(
            id="__end__", name="__end__", data="__end__", metadata=None
        ),
        "agent": DrawableNode(
            id="agent",
            name="agent_1",
            data={"id": ["langgraph", "utils", "RunnableCallable"], "name": "agent_1"},
            metadata=None,
        ),
    }

    assert drawable_graph.edges == [
        DrawableEdge(source="__start__", target="agent"),
        DrawableEdge(source="agent", target="__end__"),
    ]


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
    remote_pregel = RemoteGraph(
        "test_graph_id",
        sync_client=mock_sync_client,
    )

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
        interrupts=(),
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
    remote_pregel = RemoteGraph(
        "test_graph_id",
        client=mock_async_client,
    )

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
        interrupts=(),
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
    remote_pregel = RemoteGraph(
        "test_graph_id",
        sync_client=mock_sync_client,
    )

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
        interrupts=(),
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
    remote_pregel = RemoteGraph(
        "test_graph_id",
        client=mock_async_client,
    )

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
        interrupts=(),
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
    remote_pregel = RemoteGraph(
        "test_graph_id",
        sync_client=mock_sync_client,
    )

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
    remote_pregel = RemoteGraph(
        "test_graph_id",
        client=mock_async_client,
    )

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
        StreamPart(event="values", data={"chunk": "data1"}),
        StreamPart(event="values", data={"chunk": "data2"}),
        StreamPart(event="values", data={"chunk": "data3"}),
        StreamPart(event="updates", data={"chunk": "data4"}),
        StreamPart(
            event="messages",
            data=[
                {
                    "content": [{"text": "Hello", "type": "text", "index": 0}],
                    "type": "AIMessageChunk",
                },
                {
                    "langgraph_step": 1,
                    "langgraph_node": "call_llm",
                    "langgraph_triggers": ["branch:to:call_llm"],
                    "langgraph_path": ["__pregel_pull", "call_llm"],
                },
            ],
        ),
        StreamPart(
            event="updates",
            data={
                "__interrupt__": [
                    {
                        "value": {"question": "Does this look good?"},
                        "id": AnyStr(),
                    }
                ]
            },
        ),
    ]

    # call method / assertions
    remote_pregel = RemoteGraph(
        "test_graph_id",
        sync_client=mock_sync_client,
    )

    # test raising graph interrupt if invoked as a subgraph
    with pytest.raises(GraphInterrupt) as exc:
        for stream_part in remote_pregel.stream(
            {"input": "data"},
            # pretend we invoked this as a subgraph
            config={
                "configurable": {"thread_id": "thread_1", "checkpoint_ns": "some_ns"}
            },
            stream_mode="values",
        ):
            pass

    assert exc.value.args[0] == [
        Interrupt(
            value={"question": "Does this look good?"},
            id=AnyStr(),
        )
    ]

    # stream modes doesn't include 'updates'
    stream_parts = []
    for stream_part in remote_pregel.stream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        stream_mode="values",
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        {"chunk": "data1"},
        {"chunk": "data2"},
        {"chunk": "data3"},
    ]

    # stream_mode messages
    stream_parts = []
    for stream_part in remote_pregel.stream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        stream_mode="messages",
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        (
            {
                "content": [{"text": "Hello", "type": "text", "index": 0}],
                "type": "AIMessageChunk",
            },
            {
                "langgraph_step": 1,
                "langgraph_node": "call_llm",
                "langgraph_triggers": ["branch:to:call_llm"],
                "langgraph_path": ["__pregel_pull", "call_llm"],
            },
        ),
    ]

    mock_sync_client.runs.stream.return_value = [
        StreamPart(event="updates", data={"chunk": "data3"}),
        StreamPart(event="updates", data={"chunk": "data4"}),
        StreamPart(event="updates", data={"__interrupt__": ()}),
    ]

    # default stream_mode is updates
    stream_parts = []
    for stream_part in remote_pregel.stream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        {"chunk": "data3"},
        {"chunk": "data4"},
        {"__interrupt__": ()},
    ]

    # list stream_mode includes mode names
    stream_parts = []
    for stream_part in remote_pregel.stream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        stream_mode=["updates"],
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        ("updates", {"chunk": "data3"}),
        ("updates", {"chunk": "data4"}),
        ("updates", {"__interrupt__": ()}),
    ]

    # subgraphs + list modes
    stream_parts = []
    for stream_part in remote_pregel.stream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        stream_mode=["updates"],
        subgraphs=True,
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        ((), "updates", {"chunk": "data3"}),
        ((), "updates", {"chunk": "data4"}),
        ((), "updates", {"__interrupt__": ()}),
    ]

    # subgraphs + single mode
    stream_parts = []
    for stream_part in remote_pregel.stream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        subgraphs=True,
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        ((), {"chunk": "data3"}),
        ((), {"chunk": "data4"}),
        ((), {"__interrupt__": ()}),
    ]


@pytest.mark.anyio
async def test_astream():
    # set up test
    mock_async_client = MagicMock()
    async_iter = MagicMock()
    async_iter.__aiter__.return_value = [
        StreamPart(event="values", data={"chunk": "data1"}),
        StreamPart(event="values", data={"chunk": "data2"}),
        StreamPart(event="values", data={"chunk": "data3"}),
        StreamPart(event="updates", data={"chunk": "data4"}),
        StreamPart(
            event="messages",
            data=[
                {
                    "content": [{"text": "Hello", "type": "text", "index": 0}],
                    "type": "AIMessageChunk",
                },
                {
                    "langgraph_step": 1,
                    "langgraph_node": "call_llm",
                    "langgraph_triggers": ["branch:to:call_llm"],
                    "langgraph_path": ["__pregel_pull", "call_llm"],
                },
            ],
        ),
        StreamPart(
            event="updates",
            data={
                "__interrupt__": [
                    {
                        "value": {"question": "Does this look good?"},
                        "id": AnyStr(),
                    }
                ]
            },
        ),
    ]
    mock_async_client.runs.stream.return_value = async_iter

    # call method / assertions
    remote_pregel = RemoteGraph(
        "test_graph_id",
        client=mock_async_client,
    )

    # test raising graph interrupt if invoked as a subgraph
    with pytest.raises(GraphInterrupt) as exc:
        async for stream_part in remote_pregel.astream(
            {"input": "data"},
            # pretend we invoked this as a subgraph
            config={
                "configurable": {"thread_id": "thread_1", "checkpoint_ns": "some_ns"}
            },
            stream_mode="values",
        ):
            pass

    assert exc.value.args[0] == [
        Interrupt(
            value={"question": "Does this look good?"},
            id=AnyStr(),
        )
    ]

    # stream modes doesn't include 'updates'
    stream_parts = []
    async for stream_part in remote_pregel.astream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        stream_mode="values",
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        {"chunk": "data1"},
        {"chunk": "data2"},
        {"chunk": "data3"},
    ]

    # stream_mode messages
    stream_parts = []
    async for stream_part in remote_pregel.astream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        stream_mode="messages",
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        (
            {
                "content": [{"text": "Hello", "type": "text", "index": 0}],
                "type": "AIMessageChunk",
            },
            {
                "langgraph_step": 1,
                "langgraph_node": "call_llm",
                "langgraph_triggers": ["branch:to:call_llm"],
                "langgraph_path": ["__pregel_pull", "call_llm"],
            },
        ),
    ]

    async_iter = MagicMock()
    async_iter.__aiter__.return_value = [
        StreamPart(event="updates", data={"chunk": "data3"}),
        StreamPart(event="updates", data={"chunk": "data4"}),
        StreamPart(event="updates", data={"__interrupt__": ()}),
    ]
    mock_async_client.runs.stream.return_value = async_iter

    # default stream_mode is updates
    stream_parts = []
    async for stream_part in remote_pregel.astream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        {"chunk": "data3"},
        {"chunk": "data4"},
        {"__interrupt__": ()},
    ]

    # list stream_mode includes mode names
    stream_parts = []
    async for stream_part in remote_pregel.astream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        stream_mode=["updates"],
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        ("updates", {"chunk": "data3"}),
        ("updates", {"chunk": "data4"}),
        ("updates", {"__interrupt__": ()}),
    ]

    # subgraphs + list modes
    stream_parts = []
    async for stream_part in remote_pregel.astream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        stream_mode=["updates"],
        subgraphs=True,
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        ((), "updates", {"chunk": "data3"}),
        ((), "updates", {"chunk": "data4"}),
        ((), "updates", {"__interrupt__": ()}),
    ]

    # subgraphs + single mode
    stream_parts = []
    async for stream_part in remote_pregel.astream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        subgraphs=True,
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        ((), {"chunk": "data3"}),
        ((), {"chunk": "data4"}),
        ((), {"__interrupt__": ()}),
    ]

    async_iter = MagicMock()
    async_iter.__aiter__.return_value = [
        StreamPart(event="updates|my|subgraph", data={"chunk": "data3"}),
        StreamPart(event="updates|hello|subgraph", data={"chunk": "data4"}),
        StreamPart(event="updates|bye|subgraph", data={"__interrupt__": ()}),
    ]
    mock_async_client.runs.stream.return_value = async_iter

    # subgraphs + list modes
    stream_parts = []
    async for stream_part in remote_pregel.astream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        stream_mode=["updates"],
        subgraphs=True,
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        (("my", "subgraph"), "updates", {"chunk": "data3"}),
        (("hello", "subgraph"), "updates", {"chunk": "data4"}),
        (("bye", "subgraph"), "updates", {"__interrupt__": ()}),
    ]

    # subgraphs + single mode
    stream_parts = []
    async for stream_part in remote_pregel.astream(
        {"input": "data"},
        config={"configurable": {"thread_id": "thread_1"}},
        subgraphs=True,
    ):
        stream_parts.append(stream_part)

    assert stream_parts == [
        (("my", "subgraph"), {"chunk": "data3"}),
        (("hello", "subgraph"), {"chunk": "data4"}),
        (("bye", "subgraph"), {"__interrupt__": ()}),
    ]


def test_invoke():
    # set up test
    mock_sync_client = MagicMock()
    mock_sync_client.runs.stream.return_value = [
        StreamPart(event="values", data={"chunk": "data1"}),
        StreamPart(event="values", data={"chunk": "data2"}),
        StreamPart(
            event="values", data={"messages": [{"type": "human", "content": "world"}]}
        ),
    ]

    # call method / assertions
    remote_pregel = RemoteGraph(
        "test_graph_id",
        sync_client=mock_sync_client,
    )

    config = {"configurable": {"thread_id": "thread_1"}}
    result = remote_pregel.invoke(
        {"input": {"messages": [{"type": "human", "content": "hello"}]}}, config
    )

    assert result == {"messages": [{"type": "human", "content": "world"}]}


@pytest.mark.anyio
async def test_ainvoke():
    # set up test
    mock_async_client = MagicMock()
    async_iter = MagicMock()
    async_iter.__aiter__.return_value = [
        StreamPart(event="values", data={"chunk": "data1"}),
        StreamPart(event="values", data={"chunk": "data2"}),
        StreamPart(
            event="values", data={"messages": [{"type": "human", "content": "world"}]}
        ),
    ]
    mock_async_client.runs.stream.return_value = async_iter

    # call method / assertions
    remote_pregel = RemoteGraph(
        "test_graph_id",
        client=mock_async_client,
    )

    config = {"configurable": {"thread_id": "thread_1"}}
    result = await remote_pregel.ainvoke(
        {"input": {"messages": [{"type": "human", "content": "hello"}]}}, config
    )

    assert result == {"messages": [{"type": "human", "content": "world"}]}


@pytest.mark.skip(
    "Unskip this test to manually test the LangSmith Deployment integration"
)
@pytest.mark.anyio
async def test_langgraph_cloud_integration():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph_sdk.client import get_client, get_sync_client

    from langgraph.graph import END, START, MessagesState, StateGraph

    # create RemotePregel instance
    client = get_client()
    sync_client = get_sync_client()
    remote_pregel = RemoteGraph(
        "agent",
        client=client,
        sync_client=sync_client,
    )

    # define graph
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", remote_pregel)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    app = workflow.compile(checkpointer=InMemorySaver())

    # test invocation
    input = {
        "messages": [
            {
                "role": "human",
                "content": "What's the weather in SF?",
            }
        ]
    }

    # test invoke
    app.invoke(
        input,
        config={"configurable": {"thread_id": "39a6104a-34e7-4f83-929c-d9eb163003c9"}},
        interrupt_before=["agent"],
    )

    # test stream
    async for _ in app.astream(
        input,
        config={"configurable": {"thread_id": "2dc3e3e7-39ac-4597-aa57-4404b944e82a"}},
        subgraphs=True,
        stream_mode=["debug", "messages"],
    ):
        pass

    # test stream events
    async for chunk in remote_pregel.astream_events(
        input,
        config={"configurable": {"thread_id": "2dc3e3e7-39ac-4597-aa57-4404b944e82a"}},
        version="v2",
        subgraphs=True,
        stream_mode=[],
    ):
        pass

    # test get state
    await remote_pregel.aget_state(
        config={"configurable": {"thread_id": "2dc3e3e7-39ac-4597-aa57-4404b944e82a"}},
        subgraphs=True,
    )

    # test update state
    await remote_pregel.aupdate_state(
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

    # test get history
    async for state in remote_pregel.aget_state_history(
        config={"configurable": {"thread_id": "2dc3e3e7-39ac-4597-aa57-4404b944e82a"}},
    ):
        pass

    # test get graph
    remote_pregel.graph_id = "fe096781-5601-53d2-b2f6-0d3403f7e9ca"  # must be UUID
    await remote_pregel.aget_graph(xray=True)


def test_sanitize_config():
    # Create a test instance
    remote = RemoteGraph("test-graph")

    # Test 1: Basic config with primitives
    basic_config: RunnableConfig = {
        "recursion_limit": 10,
        "tags": ["tag1", "tag2"],
        "metadata": {"str_key": "value", "int_key": 42, "bool_key": True},
        "configurable": {"param1": "value1", "param2": 123},
    }
    sanitized = remote._sanitize_config(basic_config)
    assert sanitized["recursion_limit"] == 10
    assert sanitized["tags"] == ["tag1", "tag2"]
    assert sanitized["metadata"] == {
        "str_key": "value",
        "int_key": 42,
        "bool_key": True,
    }
    assert sanitized["configurable"] == {"param1": "value1", "param2": 123}

    # Test 2: Config with non-string tags and complex metadata
    complex_config: RunnableConfig = {
        "tags": ["tag1", 123, {"obj": "tag"}, "tag2"],  # Only string tags should remain
        "metadata": {
            "nested": {
                "key": "value",
                "num": 42,
                "invalid": lambda x: x,
            },  # Last item should be removed
            "list": [1, 2, "three"],
            "invalid": lambda x: x,  # Should be removed
            "tuple": (1, 2, 3),  # Should be converted to list
        },
    }
    sanitized = remote._sanitize_config(complex_config)
    assert sanitized["tags"] == ["tag1", "tag2"]
    assert sanitized["metadata"] == {
        "nested": {"key": "value", "num": 42},
        "list": [1, 2, "three"],
        "tuple": [1, 2, 3],
    }
    assert "invalid" not in sanitized["metadata"]

    # Test 3: Config with configurable fields that should be dropped
    config_with_drops: RunnableConfig = {
        "configurable": {
            "normal_param": "value",
            "checkpoint_map": {"key": "value"},  # Should be dropped
            "checkpoint_id": "123",  # Should be dropped
            "checkpoint_ns": "ns",  # Should be dropped
        }
    }
    sanitized = remote._sanitize_config(config_with_drops)
    assert sanitized["configurable"] == {"normal_param": "value"}
    assert "checkpoint_map" not in sanitized["configurable"]
    assert "checkpoint_id" not in sanitized["configurable"]
    assert "checkpoint_ns" not in sanitized["configurable"]

    # Test 4: Empty config
    empty_config: RunnableConfig = {}
    sanitized = remote._sanitize_config(empty_config)
    assert sanitized == {}

    # Test 5: Config with non-string keys in configurable
    invalid_keys_config: RunnableConfig = {
        "configurable": {
            "valid": "value",
            123: "invalid",  # Should be dropped
            ("tuple", "key"): "invalid",  # Should be dropped
        }
    }
    sanitized = remote._sanitize_config(invalid_keys_config)
    assert sanitized["configurable"] == {"valid": "value"}

    # Test 6: Deeply nested structures
    nested_config: RunnableConfig = {
        "metadata": {
            "level1": {
                "level2": {
                    "level3": {
                        "str": "value",
                        "list": [1, [2, [3]]],
                        "dict": {"a": {"b": {"c": "d"}}},
                    }
                }
            }
        }
    }
    sanitized = remote._sanitize_config(nested_config)
    assert sanitized["metadata"]["level1"]["level2"]["level3"]["str"] == "value"
    assert sanitized["metadata"]["level1"]["level2"]["level3"]["list"] == [1, [2, [3]]]
    assert sanitized["metadata"]["level1"]["level2"]["level3"]["dict"] == {
        "a": {"b": {"c": "d"}}
    }


"""Test RemoteGraph against an actual server."""


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


@NEEDS_CONTEXTVARS
@SKIP_PYTHON_314
async def test_remote_graph_basic_invoke(remote_graph: RemoteGraph) -> None:
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
@SKIP_PYTHON_314
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


@pytest.mark.anyio
@pytest.mark.parametrize("distributed_tracing", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("headers", [None, {"foo": "bar"}])
async def test_include_headers(
    distributed_tracing: bool, stream: bool, headers: dict[str, str] | None
):
    mock_async_client = MagicMock()
    async_iter = MagicMock()
    return_value = [
        StreamPart(event="values", data={"chunk": "data1"}),
    ]
    async_iter.__aiter__.return_value = return_value
    astream_mock = mock_async_client.runs.stream
    astream_mock.return_value = async_iter

    mock_sync_client = MagicMock()
    sync_iter = MagicMock()
    sync_iter.__iter__.return_value = return_value
    stream_mock = mock_sync_client.runs.stream
    stream_mock.return_value = async_iter

    remote_pregel = RemoteGraph(
        "test_graph_id",
        client=mock_async_client,
        sync_client=mock_sync_client,
        distributed_tracing=distributed_tracing,
    )

    config = {"configurable": {"thread_id": "thread_1"}}
    with ls.tracing_context(enabled=True, client=MagicMock()):
        with ls.trace("foo"):
            if stream:
                async for _ in remote_pregel.astream(
                    {"input": {"messages": [{"type": "human", "content": "hello"}]}},
                    config,
                    headers=headers,
                ):
                    pass

            else:
                await remote_pregel.ainvoke(
                    {"input": {"messages": [{"type": "human", "content": "hello"}]}},
                    config,
                    headers=headers,
                )
    expected = headers.copy() if headers else None
    if distributed_tracing:
        if expected is None:
            expected = {}
        expected["langsmith-trace"] = AnyStr()
        expected["baggage"] = AnyStr("langsmith-metadata=")

    assert astream_mock.call_args.kwargs["headers"] == expected
    stream_mock.assert_not_called()

    with ls.tracing_context(enabled=True, client=MagicMock()):
        with ls.trace("foo"):
            if stream:
                for _ in remote_pregel.stream(
                    {"input": {"messages": [{"type": "human", "content": "hello"}]}},
                    config,
                    headers=headers,
                ):
                    pass

            else:
                remote_pregel.invoke(
                    {"input": {"messages": [{"type": "human", "content": "hello"}]}},
                    config,
                    headers=headers,
                )
    assert stream_mock.call_args.kwargs["headers"] == expected
