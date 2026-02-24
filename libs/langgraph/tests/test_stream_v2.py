"""Tests for v2 streaming format (StreamPart TypedDicts)."""

from __future__ import annotations

import operator
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.func import entrypoint
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import StreamPart
from tests.fake_chat import FakeChatModel


class SimpleState(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def _make_simple_graph() -> StateGraph:
    """Build a simple 2-node graph for testing."""

    def node_a(state: SimpleState) -> dict:
        return {"value": state["value"] + "_a", "items": ["a"]}

    def node_b(state: SimpleState) -> dict:
        return {"value": state["value"] + "_b", "items": ["b"]}

    builder = StateGraph(SimpleState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    return builder


def _make_messages_graph() -> StateGraph:
    """Build a graph that invokes a fake LLM for messages mode testing."""
    model = FakeChatModel(messages=[AIMessage(content="hello world")])

    def call_model(state: MessagesState) -> dict:
        return {"messages": model.invoke(state["messages"])}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    return builder


# Keys expected on each payload type
_CHECKPOINT_PAYLOAD_KEYS = {
    "config",
    "metadata",
    "values",
    "next",
    "parent_config",
    "tasks",
}
_TASK_START_KEYS = {"id", "name", "input", "triggers"}
_TASK_RESULT_KEYS = {"id", "name", "error", "interrupts", "result"}
_DEBUG_ENVELOPE_KEYS = {"step", "timestamp", "type", "payload"}


# --- v1 backwards compatibility ---


class TestV1BackwardsCompat:
    """Verify that default (v1) behavior is unchanged."""

    def test_stream_default_is_v1(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(graph.stream({"value": "x", "items": []}))
        # v1 default stream_mode is "values" — yields plain dicts
        for chunk in chunks:
            assert isinstance(chunk, dict)
            # v1 chunks should NOT have "type" or "ns" keys
            assert "type" not in chunk or "value" in chunk  # "type" not a stream key

    def test_stream_v1_updates_mode(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(graph.stream({"value": "x", "items": []}, stream_mode="updates"))
        # v1 updates mode yields plain dicts: {node_name: output}
        assert len(chunks) == 2
        assert "node_a" in chunks[0]
        assert "node_b" in chunks[1]

    def test_stream_v1_list_mode(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(
            graph.stream(
                {"value": "x", "items": []},
                stream_mode=["values", "updates"],
            )
        )
        # v1 list mode yields (mode, data) tuples
        for chunk in chunks:
            assert isinstance(chunk, tuple)
            assert len(chunk) == 2
            mode, data = chunk
            assert mode in ("values", "updates")

    def test_stream_v1_subgraphs(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(
            graph.stream(
                {"value": "x", "items": []},
                stream_mode="updates",
                subgraphs=True,
            )
        )
        # v1 subgraphs yields (namespace, data) tuples
        for chunk in chunks:
            assert isinstance(chunk, tuple)
            assert len(chunk) == 2
            ns, data = chunk
            assert isinstance(ns, tuple)


# --- v2 streaming ---


class TestV2Stream:
    """Test v2 streaming format."""

    def test_stream_v2_values(self) -> None:
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = list(
            graph.stream(
                {"value": "x", "items": []},
                stream_mode="values",
                version="v2",
            )
        )
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert chunk["type"] == "values"
            assert isinstance(chunk["ns"], tuple)
            assert chunk["ns"] == ()  # root graph
            assert isinstance(chunk["data"], dict)

    def test_stream_v2_updates(self) -> None:
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = list(
            graph.stream(
                {"value": "x", "items": []},
                stream_mode="updates",
                version="v2",
            )
        )
        assert len(chunks) == 2
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert chunk["type"] == "updates"
            assert chunk["ns"] == ()
        assert "node_a" in chunks[0]["data"]
        assert "node_b" in chunks[1]["data"]

    def test_stream_v2_messages(self) -> None:
        graph = _make_messages_graph().compile()
        chunks: list[StreamPart] = list(
            graph.stream(
                {"messages": "hi"},
                stream_mode="messages",
                version="v2",
            )
        )
        msg_chunks = [c for c in chunks if c["type"] == "messages"]
        assert len(msg_chunks) >= 1
        for chunk in msg_chunks:
            assert chunk["ns"] == ()
            data = chunk["data"]
            assert isinstance(data, tuple)
            assert len(data) == 2
            message, metadata = data
            assert isinstance(message, BaseMessage)
            assert isinstance(metadata, dict)
            # metadata should contain standard langgraph keys
            assert "langgraph_node" in metadata
            assert "langgraph_triggers" in metadata

    def test_stream_v2_custom(self) -> None:
        @entrypoint()
        def graph(inputs: dict, *, writer) -> dict:
            writer("hello")
            writer(42)
            return inputs

        chunks: list[StreamPart] = list(
            graph.stream(
                {"key": "val"},
                stream_mode="custom",
                version="v2",
            )
        )
        custom_chunks = [c for c in chunks if c["type"] == "custom"]
        assert len(custom_chunks) == 2
        assert custom_chunks[0]["data"] == "hello"
        assert custom_chunks[1]["data"] == 42

    def test_stream_v2_multiple_modes(self) -> None:
        """When stream_mode is a list, v2 still yields uniform StreamPart dicts."""
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = list(
            graph.stream(
                {"value": "x", "items": []},
                stream_mode=["values", "updates"],
                version="v2",
            )
        )
        assert len(chunks) > 0
        types_seen = set()
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert "type" in chunk
            assert "ns" in chunk
            assert "data" in chunk
            types_seen.add(chunk["type"])
        assert "values" in types_seen
        assert "updates" in types_seen

    def test_stream_v2_subgraphs_ns(self) -> None:
        """With subgraphs, ns should reflect the namespace for subgraph events."""
        inner = _make_simple_graph().compile()

        def outer_node(state: SimpleState) -> dict:
            result = inner.invoke(state)
            return {"value": result["value"], "items": result["items"]}

        outer_builder = StateGraph(SimpleState)
        outer_builder.add_node("inner", inner)
        outer_builder.add_edge(START, "inner")
        outer_builder.add_edge("inner", END)
        outer = outer_builder.compile()

        chunks: list[StreamPart] = list(
            outer.stream(
                {"value": "x", "items": []},
                stream_mode="updates",
                subgraphs=True,
                version="v2",
            )
        )
        # Should have events from both root and subgraph
        root_chunks = [c for c in chunks if c["ns"] == ()]
        sub_chunks = [c for c in chunks if c["ns"] != ()]
        assert len(root_chunks) >= 1
        assert len(sub_chunks) >= 1
        for chunk in sub_chunks:
            assert isinstance(chunk["ns"], tuple)
            assert len(chunk["ns"]) > 0

    def test_stream_v2_checkpoints_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-v2-ckpt"}}
        chunks: list[StreamPart] = list(
            graph.stream(
                {"value": "x", "items": []},
                config,
                stream_mode="checkpoints",
                version="v2",
            )
        )
        ckpt_chunks = [c for c in chunks if c["type"] == "checkpoints"]
        assert len(ckpt_chunks) >= 1
        for chunk in ckpt_chunks:
            assert chunk["ns"] == ()
            payload = chunk["data"]
            assert isinstance(payload, dict)
            # Validate all CheckpointPayload keys
            assert _CHECKPOINT_PAYLOAD_KEYS <= payload.keys(), (
                f"Missing keys: {_CHECKPOINT_PAYLOAD_KEYS - payload.keys()}"
            )
            assert isinstance(payload["values"], dict)
            assert isinstance(payload["next"], list)
            assert isinstance(payload["tasks"], list)
            assert isinstance(payload["metadata"], dict)

    def test_stream_v2_tasks_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-v2-tasks"}}
        chunks: list[StreamPart] = list(
            graph.stream(
                {"value": "x", "items": []},
                config,
                stream_mode="tasks",
                version="v2",
            )
        )
        task_chunks = [c for c in chunks if c["type"] == "tasks"]
        assert len(task_chunks) >= 2  # at least task start for node_a and node_b
        for chunk in task_chunks:
            assert chunk["ns"] == ()
            assert isinstance(chunk["data"], dict)
            assert "id" in chunk["data"]
            assert "name" in chunk["data"]

        # Separate task-start and task-result events by their keys
        start_chunks = [
            c for c in task_chunks if "input" in c["data"] and "triggers" in c["data"]
        ]
        result_chunks = [c for c in task_chunks if "result" in c["data"]]

        # Should have start events for both nodes
        assert len(start_chunks) >= 2
        for c in start_chunks:
            assert _TASK_START_KEYS <= c["data"].keys(), (
                f"Missing task start keys: {_TASK_START_KEYS - c['data'].keys()}"
            )
            assert isinstance(c["data"]["triggers"], (list, tuple))

        # Should have result events for both nodes
        assert len(result_chunks) >= 2
        for c in result_chunks:
            assert _TASK_RESULT_KEYS <= c["data"].keys(), (
                f"Missing task result keys: {_TASK_RESULT_KEYS - c['data'].keys()}"
            )
            assert isinstance(c["data"]["interrupts"], list)
            assert isinstance(c["data"]["result"], dict)

    def test_stream_v2_debug_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-v2-debug"}}
        chunks: list[StreamPart] = list(
            graph.stream(
                {"value": "x", "items": []},
                config,
                stream_mode="debug",
                version="v2",
            )
        )
        debug_chunks = [c for c in chunks if c["type"] == "debug"]
        assert len(debug_chunks) >= 1
        for chunk in debug_chunks:
            assert chunk["ns"] == ()
            envelope = chunk["data"]
            assert isinstance(envelope, dict)
            # Validate debug envelope keys
            assert _DEBUG_ENVELOPE_KEYS <= envelope.keys(), (
                f"Missing debug keys: {_DEBUG_ENVELOPE_KEYS - envelope.keys()}"
            )
            assert isinstance(envelope["step"], int)
            assert isinstance(envelope["timestamp"], str)
            assert envelope["type"] in ("checkpoint", "task", "task_result")
            assert isinstance(envelope["payload"], dict)

            # Validate inner payload based on debug event type
            if envelope["type"] == "checkpoint":
                assert _CHECKPOINT_PAYLOAD_KEYS <= envelope["payload"].keys()
            elif envelope["type"] == "task":
                assert _TASK_START_KEYS <= envelope["payload"].keys()
            elif envelope["type"] == "task_result":
                assert _TASK_RESULT_KEYS <= envelope["payload"].keys()

    def test_stream_v2_ignores_subgraphs_param(self) -> None:
        """In v2, subgraphs=True/False should not change the output format.
        ns is always present regardless."""
        graph = _make_simple_graph().compile()
        chunks_no_sub: list[StreamPart] = list(
            graph.stream(
                {"value": "x", "items": []},
                stream_mode="updates",
                subgraphs=False,
                version="v2",
            )
        )
        chunks_with_sub: list[StreamPart] = list(
            graph.stream(
                {"value": "x", "items": []},
                stream_mode="updates",
                subgraphs=True,
                version="v2",
            )
        )
        # Both should yield same structure: dict with type/ns/data
        for c in chunks_no_sub:
            assert "type" in c and "ns" in c and "data" in c
        for c in chunks_with_sub:
            assert "type" in c and "ns" in c and "data" in c


# --- v2 invoke ---


class TestV2Invoke:
    """Test v2 invoke format."""

    def test_invoke_v2_values_default(self) -> None:
        """v2 invoke with default stream_mode='values' returns the final state dict."""
        graph = _make_simple_graph().compile()
        result = graph.invoke({"value": "x", "items": []}, version="v2")
        assert isinstance(result, dict)
        assert result["value"] == "x_a_b"
        assert result["items"] == ["a", "b"]

    def test_invoke_v2_updates_mode(self) -> None:
        """v2 invoke with stream_mode='updates' returns list of StreamPart dicts."""
        graph = _make_simple_graph().compile()
        result = graph.invoke(
            {"value": "x", "items": []},
            stream_mode="updates",
            version="v2",
        )
        assert isinstance(result, list)
        assert len(result) == 2
        for chunk in result:
            assert isinstance(chunk, dict)
            assert chunk["type"] == "updates"
            assert isinstance(chunk["ns"], tuple)
            assert chunk["ns"] == ()
            assert isinstance(chunk["data"], dict)
        assert "node_a" in result[0]["data"]
        assert "node_b" in result[1]["data"]

    def test_invoke_v2_multiple_modes(self) -> None:
        """v2 invoke with list stream_mode returns list of StreamPart dicts."""
        graph = _make_simple_graph().compile()
        result = graph.invoke(
            {"value": "x", "items": []},
            stream_mode=["values", "updates"],
            version="v2",
        )
        assert isinstance(result, list)
        types_seen = set()
        for chunk in result:
            assert isinstance(chunk, dict)
            assert "type" in chunk
            assert "ns" in chunk
            assert "data" in chunk
            types_seen.add(chunk["type"])
        assert "values" in types_seen
        assert "updates" in types_seen

    def test_invoke_v1_default_unchanged(self) -> None:
        """Default invoke (v1) behavior is unchanged."""
        graph = _make_simple_graph().compile()
        result = graph.invoke({"value": "x", "items": []})
        assert isinstance(result, dict)
        assert result["value"] == "x_a_b"
        assert result["items"] == ["a", "b"]

    def test_invoke_v1_updates_unchanged(self) -> None:
        """Default invoke (v1) with updates returns v1-style chunks."""
        graph = _make_simple_graph().compile()
        result = graph.invoke(
            {"value": "x", "items": []},
            stream_mode="updates",
        )
        assert isinstance(result, list)
        # v1 chunks should not have "type"/"ns"/"data" structure
        for chunk in result:
            assert "node_a" in chunk or "node_b" in chunk


# --- async tests ---


class TestV2StreamAsync:
    @pytest.mark.anyio
    async def test_astream_v2_values(self) -> None:
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            {"value": "x", "items": []},
            stream_mode="values",
            version="v2",
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert chunk["type"] == "values"
            assert chunk["ns"] == ()
            assert isinstance(chunk["data"], dict)

    @pytest.mark.anyio
    async def test_astream_v2_updates(self) -> None:
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            {"value": "x", "items": []},
            stream_mode="updates",
            version="v2",
        ):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0]["type"] == "updates"
        assert "node_a" in chunks[0]["data"]
        assert chunks[1]["type"] == "updates"
        assert "node_b" in chunks[1]["data"]

    @pytest.mark.anyio
    async def test_astream_v2_messages(self) -> None:
        graph = _make_messages_graph().compile()
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            {"messages": "hi"},
            stream_mode="messages",
            version="v2",
        ):
            chunks.append(chunk)
        msg_chunks = [c for c in chunks if c["type"] == "messages"]
        assert len(msg_chunks) >= 1
        for chunk in msg_chunks:
            assert chunk["ns"] == ()
            data = chunk["data"]
            assert isinstance(data, tuple)
            assert len(data) == 2
            message, metadata = data
            assert isinstance(message, BaseMessage)
            assert isinstance(metadata, dict)
            assert "langgraph_node" in metadata

    @pytest.mark.anyio
    async def test_astream_v2_custom(self) -> None:
        @entrypoint()
        def graph(inputs: dict, *, writer) -> dict:
            writer("hello")
            writer(42)
            return inputs

        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            {"key": "val"},
            stream_mode="custom",
            version="v2",
        ):
            chunks.append(chunk)
        custom_chunks = [c for c in chunks if c["type"] == "custom"]
        assert len(custom_chunks) == 2
        assert custom_chunks[0]["data"] == "hello"
        assert custom_chunks[1]["data"] == 42

    @pytest.mark.anyio
    async def test_astream_v2_multiple_modes(self) -> None:
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            {"value": "x", "items": []},
            stream_mode=["values", "updates"],
            version="v2",
        ):
            chunks.append(chunk)
        types_seen = {c["type"] for c in chunks}
        assert "values" in types_seen
        assert "updates" in types_seen
        for chunk in chunks:
            assert "type" in chunk and "ns" in chunk and "data" in chunk

    @pytest.mark.anyio
    async def test_astream_v2_subgraphs_ns(self) -> None:
        inner = _make_simple_graph().compile()

        outer_builder = StateGraph(SimpleState)
        outer_builder.add_node("inner", inner)
        outer_builder.add_edge(START, "inner")
        outer_builder.add_edge("inner", END)
        outer = outer_builder.compile()

        chunks: list[StreamPart] = []
        async for chunk in outer.astream(
            {"value": "x", "items": []},
            stream_mode="updates",
            subgraphs=True,
            version="v2",
        ):
            chunks.append(chunk)
        root_chunks = [c for c in chunks if c["ns"] == ()]
        sub_chunks = [c for c in chunks if c["ns"] != ()]
        assert len(root_chunks) >= 1
        assert len(sub_chunks) >= 1
        for chunk in sub_chunks:
            assert isinstance(chunk["ns"], tuple)
            assert len(chunk["ns"]) > 0

    @pytest.mark.anyio
    async def test_astream_v2_checkpoints_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-v2-ckpt-async"}}
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            {"value": "x", "items": []},
            config,
            stream_mode="checkpoints",
            version="v2",
        ):
            chunks.append(chunk)
        ckpt_chunks = [c for c in chunks if c["type"] == "checkpoints"]
        assert len(ckpt_chunks) >= 1
        for chunk in ckpt_chunks:
            assert chunk["ns"] == ()
            payload = chunk["data"]
            assert isinstance(payload, dict)
            assert _CHECKPOINT_PAYLOAD_KEYS <= payload.keys()
            assert isinstance(payload["values"], dict)
            assert isinstance(payload["next"], list)
            assert isinstance(payload["tasks"], list)

    @pytest.mark.anyio
    async def test_astream_v2_tasks_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-v2-tasks-async"}}
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            {"value": "x", "items": []},
            config,
            stream_mode="tasks",
            version="v2",
        ):
            chunks.append(chunk)
        task_chunks = [c for c in chunks if c["type"] == "tasks"]
        assert len(task_chunks) >= 2

        start_chunks = [
            c for c in task_chunks if "input" in c["data"] and "triggers" in c["data"]
        ]
        result_chunks = [c for c in task_chunks if "result" in c["data"]]
        assert len(start_chunks) >= 2
        for c in start_chunks:
            assert _TASK_START_KEYS <= c["data"].keys()

        assert len(result_chunks) >= 2
        for c in result_chunks:
            assert _TASK_RESULT_KEYS <= c["data"].keys()

    @pytest.mark.anyio
    async def test_astream_v2_debug_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-v2-debug-async"}}
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            {"value": "x", "items": []},
            config,
            stream_mode="debug",
            version="v2",
        ):
            chunks.append(chunk)
        debug_chunks = [c for c in chunks if c["type"] == "debug"]
        assert len(debug_chunks) >= 1
        for chunk in debug_chunks:
            assert chunk["ns"] == ()
            envelope = chunk["data"]
            assert isinstance(envelope, dict)
            assert _DEBUG_ENVELOPE_KEYS <= envelope.keys()
            assert envelope["type"] in ("checkpoint", "task", "task_result")
            assert isinstance(envelope["payload"], dict)

            if envelope["type"] == "checkpoint":
                assert _CHECKPOINT_PAYLOAD_KEYS <= envelope["payload"].keys()
            elif envelope["type"] == "task":
                assert _TASK_START_KEYS <= envelope["payload"].keys()
            elif envelope["type"] == "task_result":
                assert _TASK_RESULT_KEYS <= envelope["payload"].keys()


# --- async v2 invoke ---


class TestV2InvokeAsync:
    @pytest.mark.anyio
    async def test_ainvoke_v2_values_default(self) -> None:
        """v2 ainvoke with default stream_mode='values' returns the final state dict."""
        graph = _make_simple_graph().compile()
        result = await graph.ainvoke({"value": "x", "items": []}, version="v2")
        assert isinstance(result, dict)
        assert result["value"] == "x_a_b"
        assert result["items"] == ["a", "b"]

    @pytest.mark.anyio
    async def test_ainvoke_v2_updates_mode(self) -> None:
        """v2 ainvoke with stream_mode='updates' returns list of StreamPart dicts."""
        graph = _make_simple_graph().compile()
        result = await graph.ainvoke(
            {"value": "x", "items": []},
            stream_mode="updates",
            version="v2",
        )
        assert isinstance(result, list)
        assert len(result) == 2
        for chunk in result:
            assert isinstance(chunk, dict)
            assert chunk["type"] == "updates"
            assert isinstance(chunk["ns"], tuple)
            assert chunk["ns"] == ()
            assert isinstance(chunk["data"], dict)
        assert "node_a" in result[0]["data"]
        assert "node_b" in result[1]["data"]

    @pytest.mark.anyio
    async def test_ainvoke_v2_multiple_modes(self) -> None:
        """v2 ainvoke with list stream_mode returns list of StreamPart dicts."""
        graph = _make_simple_graph().compile()
        result = await graph.ainvoke(
            {"value": "x", "items": []},
            stream_mode=["values", "updates"],
            version="v2",
        )
        assert isinstance(result, list)
        types_seen = set()
        for chunk in result:
            assert isinstance(chunk, dict)
            assert "type" in chunk
            assert "ns" in chunk
            assert "data" in chunk
            types_seen.add(chunk["type"])
        assert "values" in types_seen
        assert "updates" in types_seen
