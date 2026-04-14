from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.runnables.graph import Edge as DrawableEdge
from langchain_core.runnables.graph import Node as DrawableNode

from langgraph._internal._a2a import (
    A2AClientError,
    A2AMessage,
    A2ATask,
    AgentCard,
    AgentCapabilities,
    AgentInterface,
    Artifact,
    Part,
    Role,
    TaskState,
    TaskStatus,
)
from langgraph.errors import GraphInterrupt
from langgraph.pregel.a2a_remote import A2ARemoteGraph
from langgraph.types import Interrupt, StateSnapshot

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------------- #
#  Fixtures & helpers
# --------------------------------------------------------------------------- #


def _make_task(
    task_id: str = "task_1",
    context_id: str = "ctx_1",
    state: TaskState = TaskState.COMPLETED,
    agent_text: str = "Hello from A2A agent",
    artifacts: list[Artifact] | None = None,
) -> dict:
    """Build a raw A2A Task dict as returned by JSON-RPC."""
    status: dict = {"state": state.value}
    if agent_text:
        status["message"] = {
            "message_id": "msg_resp",
            "role": "ROLE_AGENT",
            "parts": [{"text": agent_text}],
        }
    d: dict = {
        "id": task_id,
        "context_id": context_id,
        "status": status,
        "artifacts": [],
        "history": [],
    }
    if artifacts:
        d["artifacts"] = [
            {
                "artifact_id": a.artifact_id,
                "parts": [p.to_dict() for p in a.parts],
            }
            for a in artifacts
        ]
    return d


def _make_graph(
    *,
    sync_client: MagicMock | None = None,
    async_client: AsyncMock | None = None,
    name: str = "test_a2a",
    agent_card: AgentCard | None = None,
) -> A2ARemoteGraph:
    return A2ARemoteGraph(
        "http://fake-a2a:8000",
        name=name,
        sync_client=sync_client or MagicMock(),
        client=async_client or AsyncMock(),
        agent_card=agent_card,
    )


# --------------------------------------------------------------------------- #
#  with_config
# --------------------------------------------------------------------------- #


def test_with_config():
    graph = _make_graph()
    graph.config = {"configurable": {"thread_id": "t1", "foo": "bar"}}
    copy = graph.with_config({"configurable": {"hello": "world"}})

    assert copy is not graph
    assert copy.config == {
        "configurable": {"thread_id": "t1", "foo": "bar", "hello": "world"}
    }


# --------------------------------------------------------------------------- #
#  get_graph / aget_graph
# --------------------------------------------------------------------------- #


def test_get_graph():
    graph = _make_graph(name="my_agent")
    drawable = graph.get_graph()

    assert "0" in drawable.nodes
    assert drawable.nodes["0"].name == "my_agent"
    assert DrawableEdge(source="__start__", target="0") in drawable.edges
    assert DrawableEdge(source="0", target="__end__") in drawable.edges


async def test_aget_graph():
    graph = _make_graph(name="my_agent")
    drawable = await graph.aget_graph()

    assert drawable.nodes["0"].name == "my_agent"


# --------------------------------------------------------------------------- #
#  get_state / aget_state
# --------------------------------------------------------------------------- #


def test_get_state():
    mock_sync = MagicMock()
    mock_sync.get_task.return_value = _make_task()

    graph = _make_graph(sync_client=mock_sync)
    config = {"configurable": {"thread_id": "ctx_1", "checkpoint_id": "task_1"}}
    snapshot = graph.get_state(config)

    assert isinstance(snapshot, StateSnapshot)
    assert snapshot.values["messages"][0]["content"] == "Hello from A2A agent"
    assert snapshot.config["configurable"]["checkpoint_id"] == "task_1"
    assert snapshot.next == ()  # COMPLETED → no next nodes
    mock_sync.get_task.assert_called_once_with("task_1")


def test_get_state_no_task_id_raises():
    graph = _make_graph()
    with pytest.raises(ValueError, match="No task ID found"):
        graph.get_state({"configurable": {"thread_id": "t1"}})


async def test_aget_state():
    mock_async = AsyncMock()
    mock_async.get_task.return_value = _make_task()

    graph = _make_graph(async_client=mock_async)
    config = {"configurable": {"thread_id": "ctx_1", "checkpoint_id": "task_1"}}
    snapshot = await graph.aget_state(config)

    assert snapshot.values["messages"][0]["content"] == "Hello from A2A agent"
    mock_async.get_task.assert_awaited_once_with("task_1")


def test_get_state_interrupted():
    mock_sync = MagicMock()
    mock_sync.get_task.return_value = _make_task(
        state=TaskState.INPUT_REQUIRED,
        agent_text="What is your name?",
    )

    graph = _make_graph(sync_client=mock_sync)
    config = {"configurable": {"thread_id": "ctx_1", "checkpoint_id": "task_1"}}
    snapshot = graph.get_state(config)

    assert snapshot.next == ()  # interrupted, not in-progress
    assert len(snapshot.interrupts) == 1
    assert "What is your name?" in snapshot.interrupts[0].value


def test_get_state_working():
    mock_sync = MagicMock()
    mock_sync.get_task.return_value = _make_task(
        state=TaskState.WORKING,
        agent_text="",
    )

    graph = _make_graph(sync_client=mock_sync, name="worker")
    config = {"configurable": {"thread_id": "ctx_1", "checkpoint_id": "task_1"}}
    snapshot = graph.get_state(config)

    assert snapshot.next == ("worker",)  # still has work to do


# --------------------------------------------------------------------------- #
#  get_state_history
# --------------------------------------------------------------------------- #


def test_get_state_history():
    mock_sync = MagicMock()
    mock_sync.get_task.return_value = _make_task()

    graph = _make_graph(sync_client=mock_sync)
    config = {"configurable": {"thread_id": "ctx_1", "checkpoint_id": "task_1"}}
    history = list(graph.get_state_history(config))

    assert len(history) == 1
    assert history[0].values["messages"][0]["content"] == "Hello from A2A agent"


# --------------------------------------------------------------------------- #
#  update_state — must raise
# --------------------------------------------------------------------------- #


def test_update_state_raises():
    graph = _make_graph()
    with pytest.raises(NotImplementedError, match="do not support direct state"):
        graph.update_state({"configurable": {"thread_id": "t1"}}, {"key": "val"})


async def test_aupdate_state_raises():
    graph = _make_graph()
    with pytest.raises(NotImplementedError):
        await graph.aupdate_state(
            {"configurable": {"thread_id": "t1"}}, {"key": "val"}
        )


# --------------------------------------------------------------------------- #
#  invoke / ainvoke
# --------------------------------------------------------------------------- #


def test_invoke():
    mock_sync = MagicMock()
    mock_sync.send_message.return_value = _make_task()

    card = AgentCard(
        name="test",
        description="test",
        version="1.0",
        supported_interfaces=[],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
    )
    graph = _make_graph(sync_client=mock_sync, agent_card=card)

    result = graph.invoke(
        {"messages": [("user", "hi")]},
        config={"configurable": {"thread_id": "t1"}},
    )

    assert result is not None
    assert result["messages"][0]["content"] == "Hello from A2A agent"
    mock_sync.send_message.assert_called_once()


async def test_ainvoke():
    mock_async = AsyncMock()
    mock_async.send_message.return_value = _make_task()

    card = AgentCard(
        name="test",
        description="test",
        version="1.0",
        supported_interfaces=[],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
    )
    graph = _make_graph(async_client=mock_async, agent_card=card)

    result = await graph.ainvoke(
        {"messages": [("user", "hi")]},
        config={"configurable": {"thread_id": "t1"}},
    )

    assert result is not None
    assert result["messages"][0]["content"] == "Hello from A2A agent"
    mock_async.send_message.assert_awaited_once()


def test_invoke_failed_raises():
    mock_sync = MagicMock()
    mock_sync.send_message.return_value = _make_task(
        state=TaskState.FAILED,
        agent_text="Something went wrong",
    )

    card = AgentCard(
        name="test",
        description="test",
        version="1.0",
        supported_interfaces=[],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
    )
    graph = _make_graph(sync_client=mock_sync, agent_card=card)

    with pytest.raises(A2AClientError, match="Something went wrong"):
        graph.invoke(
            {"messages": [("user", "hi")]},
            config={"configurable": {"thread_id": "t1"}},
        )


def test_invoke_interrupt_as_subgraph():
    mock_sync = MagicMock()
    mock_sync.send_message.return_value = _make_task(
        state=TaskState.INPUT_REQUIRED,
        agent_text="What is your name?",
    )

    card = AgentCard(
        name="test",
        description="test",
        version="1.0",
        supported_interfaces=[],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
    )
    graph = _make_graph(sync_client=mock_sync, agent_card=card)

    with pytest.raises(GraphInterrupt):
        graph.invoke(
            {"messages": [("user", "hi")]},
            config={
                "configurable": {"thread_id": "t1", "checkpoint_ns": "some_ns"}
            },
        )


# --------------------------------------------------------------------------- #
#  stream (sync, non-streaming agent → fallback)
# --------------------------------------------------------------------------- #


def test_stream_fallback():
    mock_sync = MagicMock()
    mock_sync.send_message.return_value = _make_task()

    card = AgentCard(
        name="test",
        description="test",
        version="1.0",
        supported_interfaces=[],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
    )
    graph = _make_graph(sync_client=mock_sync, agent_card=card)

    parts = list(
        graph.stream(
            {"messages": [("user", "hi")]},
            config={"configurable": {"thread_id": "t1"}},
            stream_mode="values",
        )
    )

    assert len(parts) == 1
    assert parts[0]["messages"][0]["content"] == "Hello from A2A agent"


# --------------------------------------------------------------------------- #
#  stream (sync, SSE streaming agent)
# --------------------------------------------------------------------------- #


def test_stream_sse():
    mock_sync = MagicMock()
    mock_sync.send_streaming_message.return_value = iter([
        {
            "result": {
                "task_id": "task_1",
                "context_id": "ctx_1",
                "status": {"state": "TASK_STATE_WORKING"},
            }
        },
        {
            "result": {
                "task_id": "task_1",
                "context_id": "ctx_1",
                "artifact": {
                    "artifact_id": "a1",
                    "parts": [{"text": "streaming chunk"}],
                },
            }
        },
        {
            "result": _make_task(
                state=TaskState.COMPLETED,
                agent_text="final answer",
            )
        },
    ])

    graph = _make_graph(sync_client=mock_sync)  # no card → assumes streaming

    parts = list(
        graph.stream(
            {"messages": [("user", "hi")]},
            config={"configurable": {"thread_id": "t1"}},
            stream_mode=["updates", "values"],
        )
    )

    assert len(parts) > 0
    # v1 multi-mode returns StreamPart NamedTuples with .event and .data
    events = {p.event for p in parts}
    assert "updates" in events or "values" in events


# --------------------------------------------------------------------------- #
#  stream v2 format
# --------------------------------------------------------------------------- #


def test_stream_v2():
    mock_sync = MagicMock()
    mock_sync.send_message.return_value = _make_task()

    card = AgentCard(
        name="test",
        description="test",
        version="1.0",
        supported_interfaces=[],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
    )
    graph = _make_graph(sync_client=mock_sync, agent_card=card)

    parts = list(
        graph.stream(
            {"messages": [("user", "hi")]},
            config={"configurable": {"thread_id": "t1"}},
            stream_mode="values",
            version="v2",
        )
    )

    assert len(parts) == 1
    assert parts[0]["type"] == "values"
    assert parts[0]["data"]["messages"][0]["content"] == "Hello from A2A agent"
    assert parts[0]["interrupts"] == ()


# --------------------------------------------------------------------------- #
#  astream
# --------------------------------------------------------------------------- #


async def test_astream_fallback():
    mock_async = AsyncMock()
    mock_async.send_message.return_value = _make_task()

    card = AgentCard(
        name="test",
        description="test",
        version="1.0",
        supported_interfaces=[],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
    )
    graph = _make_graph(async_client=mock_async, agent_card=card)

    parts = []
    async for chunk in graph.astream(
        {"messages": [("user", "hi")]},
        config={"configurable": {"thread_id": "t1"}},
        stream_mode="values",
    ):
        parts.append(chunk)

    assert len(parts) == 1
    assert parts[0]["messages"][0]["content"] == "Hello from A2A agent"


# --------------------------------------------------------------------------- #
#  polling backoff
# --------------------------------------------------------------------------- #


def test_poll_backoff():
    mock_sync = MagicMock()

    # first call returns WORKING, second returns COMPLETED
    mock_sync.get_task.side_effect = [
        _make_task(state=TaskState.WORKING, agent_text=""),
        _make_task(state=TaskState.COMPLETED, agent_text="done"),
    ]

    graph = _make_graph(sync_client=mock_sync)
    task = A2ATask.from_dict(_make_task(state=TaskState.WORKING, agent_text=""))

    with patch("langgraph.pregel.a2a_remote.time.sleep") as mock_sleep:
        result = graph._poll_until_done(task, initial_interval=1.0, backoff_factor=2.0)

    assert result.status.state == TaskState.COMPLETED
    assert mock_sync.get_task.call_count == 2
    # first sleep should be 1.0s (initial)
    mock_sleep.assert_called()
    assert mock_sleep.call_args_list[0].args[0] == 1.0


def test_poll_timeout():
    mock_sync = MagicMock()
    mock_sync.get_task.return_value = _make_task(
        state=TaskState.WORKING, agent_text=""
    )

    graph = _make_graph(sync_client=mock_sync)
    task = A2ATask.from_dict(_make_task(state=TaskState.WORKING, agent_text=""))

    with patch("langgraph.pregel.a2a_remote.time.sleep"):
        with pytest.raises(TimeoutError, match="did not complete"):
            graph._poll_until_done(task, max_elapsed=0.0)


# --------------------------------------------------------------------------- #
#  from_agent_card_sync
# --------------------------------------------------------------------------- #


def test_from_agent_card_sync():
    card_data = {
        "name": "Currency Agent",
        "description": "Converts currencies",
        "version": "1.0.0",
        "supported_interfaces": [
            {
                "url": "https://agent.example.com/jsonrpc",
                "protocol_binding": "JSONRPC",
                "protocol_version": "1.0",
            }
        ],
        "capabilities": {"streaming": True},
        "skills": [
            {
                "id": "convert",
                "name": "Convert Currency",
                "description": "Convert between currencies",
                "tags": ["currency"],
            }
        ],
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
    }

    card = AgentCard.from_dict(card_data)

    with patch(
        "langgraph.pregel.a2a_remote.SyncA2AClient"
    ) as MockSyncClient:
        mock_instance = MockSyncClient.return_value
        mock_instance.fetch_agent_card.return_value = card
        mock_instance.close.return_value = None

        graph = A2ARemoteGraph.from_agent_card_sync(
            "https://agent.example.com/.well-known/agent-card.json"
        )

    assert graph.name == "Currency Agent"
    assert graph.agent_card is not None
    assert graph.agent_card.capabilities.streaming is True
    assert graph.url == "https://agent.example.com/jsonrpc"


# --------------------------------------------------------------------------- #
#  sanitize_config
# --------------------------------------------------------------------------- #


def test_sanitize_config():
    graph = _make_graph()

    config = {
        "configurable": {
            "thread_id": "t1",
            "checkpoint_id": "c1",  # should be dropped
            "checkpoint_ns": "ns",  # should be dropped
            "custom_key": "value",
            "non_serializable": object(),  # should be dropped
        },
        "tags": ["tag1", "tag2"],
        "metadata": {"key": "val", "nested": {"a": 1}},
    }
    sanitized = graph._sanitize_config(config)

    assert "checkpoint_id" not in sanitized.get("configurable", {})
    assert "checkpoint_ns" not in sanitized.get("configurable", {})
    assert sanitized["configurable"]["thread_id"] == "t1"
    assert sanitized["configurable"]["custom_key"] == "value"
    assert "non_serializable" not in sanitized["configurable"]
    assert sanitized["tags"] == ["tag1", "tag2"]
    assert sanitized["metadata"]["nested"]["a"] == 1


# --------------------------------------------------------------------------- #
#  message conversion
# --------------------------------------------------------------------------- #


def test_messages_to_parts_dict():
    from langgraph.pregel.a2a_remote import _messages_to_parts

    parts = _messages_to_parts([{"role": "user", "content": "hello"}])
    assert len(parts) == 1
    assert parts[0].text == "hello"


def test_messages_to_parts_tuple():
    from langgraph.pregel.a2a_remote import _messages_to_parts

    parts = _messages_to_parts([("user", "hi there")])
    assert len(parts) == 1
    assert parts[0].text == "hi there"


def test_last_user_messages():
    from langgraph.pregel.a2a_remote import _last_user_messages

    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    result = _last_user_messages(messages)
    assert len(result) == 2
    assert result[0]["content"] == "second"
    assert result[1]["content"] == "third"


def test_task_to_ai_messages():
    from langgraph.pregel.a2a_remote import _task_to_ai_messages

    task = A2ATask.from_dict(_make_task(agent_text="hello world"))
    msgs = _task_to_ai_messages(task)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "hello world"
    assert msgs[0]["role"] == "assistant"


def test_task_to_ai_messages_with_artifacts():
    from langgraph.pregel.a2a_remote import _task_to_ai_messages

    task_dict = _make_task(
        agent_text="summary",
        artifacts=[Artifact(artifact_id="a1", parts=[Part(text="detailed result")])],
    )
    task = A2ATask.from_dict(task_dict)
    msgs = _task_to_ai_messages(task)

    contents = [m["content"] for m in msgs]
    assert "summary" in contents
    assert "detailed result" in contents


def test_task_to_ai_messages_deduplicates():
    from langgraph.pregel.a2a_remote import _task_to_ai_messages

    # same text in status message and artifact → should be deduplicated
    task_dict = _make_task(
        agent_text="same text",
        artifacts=[Artifact(artifact_id="a1", parts=[Part(text="same text")])],
    )
    task = A2ATask.from_dict(task_dict)
    msgs = _task_to_ai_messages(task)

    assert len(msgs) == 1
    assert msgs[0]["content"] == "same text"
