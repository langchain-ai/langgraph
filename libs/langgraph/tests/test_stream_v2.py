"""Tests for v2 streaming format (StreamPart TypedDicts).

This file is checked by mypy directly — no subprocess workarounds.
Type-narrowing is validated via ``assert_type`` calls.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, assert_type

from langgraph.constants import END, START
from langgraph.func import entrypoint
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import (
    CheckpointPayload,
    CheckpointStreamPart,
    CustomStreamPart,
    DebugPayload,
    DebugStreamPart,
    MessagesStreamPart,
    StreamPart,
    StreamWriter,
    TaskPayload,
    TaskResultPayload,
    TasksStreamPart,
    UpdatesStreamPart,
    ValuesStreamPart,
)
from tests.fake_chat import FakeChatModel


class SimpleState(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def _make_simple_graph() -> StateGraph:
    """Build a simple 2-node graph for testing."""

    def node_a(state: SimpleState) -> dict[str, Any]:
        return {"value": state["value"] + "_a", "items": ["a"]}

    def node_b(state: SimpleState) -> dict[str, Any]:
        return {"value": state["value"] + "_b", "items": ["b"]}

    builder: StateGraph[SimpleState] = StateGraph(SimpleState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    return builder


def _make_messages_graph() -> StateGraph:
    """Build a graph that invokes a fake LLM for messages mode testing."""
    model = FakeChatModel(messages=[AIMessage(content="hello world")])

    def call_model(state: MessagesState) -> dict[str, Any]:
        return {"messages": model.invoke(state["messages"])}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    return builder


# --- shared helpers for data structure validation ---

_STREAM_PART_KEYS = {"type", "ns", "data"}


def _assert_stream_part_shape(part: StreamPart) -> None:
    """Assert that a stream part has the required keys and types."""
    assert isinstance(part, dict), f"Expected dict, got {type(part)}"
    assert _STREAM_PART_KEYS <= part.keys(), (
        f"Missing keys: {_STREAM_PART_KEYS - part.keys()}"
    )
    assert isinstance(part["type"], str), (
        f"type should be str, got {type(part['type'])}"
    )
    assert isinstance(part["ns"], tuple), f"ns should be tuple, got {type(part['ns'])}"
    for elem in part["ns"]:
        assert isinstance(elem, str), f"ns element should be str, got {type(elem)}"


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


def _assert_checkpoint_payload(payload: Any) -> None:
    """Validate all CheckpointPayload fields."""
    assert _CHECKPOINT_PAYLOAD_KEYS <= payload.keys(), (
        f"Missing keys: {_CHECKPOINT_PAYLOAD_KEYS - payload.keys()}"
    )
    assert isinstance(payload["values"], dict)
    assert isinstance(payload["next"], list)
    assert isinstance(payload["tasks"], list)
    assert isinstance(payload["metadata"], dict)


def _assert_task_start_payload(payload: Any) -> None:
    """Validate TaskPayload (task start) fields."""
    assert _TASK_START_KEYS <= payload.keys(), (
        f"Missing keys: {_TASK_START_KEYS - payload.keys()}"
    )
    assert isinstance(payload["id"], str)
    assert isinstance(payload["name"], str)
    assert isinstance(payload["triggers"], (list, tuple))


def _assert_task_result_payload(payload: Any) -> None:
    """Validate TaskResultPayload (task result) fields."""
    assert _TASK_RESULT_KEYS <= payload.keys(), (
        f"Missing keys: {_TASK_RESULT_KEYS - payload.keys()}"
    )
    assert isinstance(payload["id"], str)
    assert isinstance(payload["name"], str)
    assert isinstance(payload["interrupts"], list)
    assert isinstance(payload["result"], dict)


def _assert_debug_envelope(envelope: Any) -> None:
    """Validate DebugPayload outer envelope and inner payload."""
    assert _DEBUG_ENVELOPE_KEYS <= envelope.keys(), (
        f"Missing keys: {_DEBUG_ENVELOPE_KEYS - envelope.keys()}"
    )
    assert isinstance(envelope["step"], int)
    assert isinstance(envelope["timestamp"], str)
    assert envelope["type"] in ("checkpoint", "task", "task_result")
    assert isinstance(envelope["payload"], dict)

    if envelope["type"] == "checkpoint":
        _assert_checkpoint_payload(envelope["payload"])
    elif envelope["type"] == "task":
        _assert_task_start_payload(envelope["payload"])
    elif envelope["type"] == "task_result":
        _assert_task_result_payload(envelope["payload"])


# Use Any-typed inputs to avoid mypy's StateT binding issues with dict literals.
_SIMPLE_INPUT: Any = {"value": "x", "items": []}
_MSG_INPUT: Any = {"messages": "hi"}


# --- v1 backwards compatibility ---


class TestV1BackwardsCompat:
    """Verify that default (v1) behavior is unchanged."""

    def test_stream_default_is_v1(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(graph.stream(_SIMPLE_INPUT))
        # v1 default stream_mode is "values" — yields plain dicts
        for chunk in chunks:
            assert isinstance(chunk, dict)

    def test_stream_v1_updates_mode(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(graph.stream(_SIMPLE_INPUT, stream_mode="updates"))
        # v1 updates mode yields plain dicts: {node_name: output}
        assert len(chunks) == 2
        assert "node_a" in chunks[0]
        assert "node_b" in chunks[1]

    def test_stream_v1_list_mode(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode=["values", "updates"],
            )
        )
        # v1 list mode yields (mode, data) tuples
        for chunk in chunks:
            assert isinstance(chunk, tuple)
            assert len(chunk) == 2
            mode, _data = chunk
            assert mode in ("values", "updates")

    def test_stream_v1_subgraphs(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=True,
            )
        )
        # v1 subgraphs yields (namespace, data) tuples
        for chunk in chunks:
            assert isinstance(chunk, tuple)
            assert len(chunk) == 2
            ns, _data = chunk
            assert isinstance(ns, tuple)


# --- v2 streaming ---


class TestV2Stream:
    """Test v2 streaming format."""

    def test_stream_v2_values(self) -> None:
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode="values",
                stream_version="v2",
            )
        )
        assert len(chunks) >= 1
        for chunk in chunks:
            _assert_stream_part_shape(chunk)
            assert chunk["type"] == "values"
            assert chunk["ns"] == ()  # root graph
            assert isinstance(chunk["data"], dict)
            # type narrowing check
            if chunk["type"] == "values":
                assert_type(chunk, ValuesStreamPart)

    def test_stream_v2_updates(self) -> None:
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                stream_version="v2",
            )
        )
        assert len(chunks) == 2
        for chunk in chunks:
            _assert_stream_part_shape(chunk)
            assert chunk["type"] == "updates"
            assert chunk["ns"] == ()
            assert isinstance(chunk["data"], dict)
            if chunk["type"] == "updates":
                assert_type(chunk, UpdatesStreamPart)
        assert "node_a" in chunks[0]["data"]
        assert "node_b" in chunks[1]["data"]

    def test_stream_v2_messages(self) -> None:
        graph = _make_messages_graph().compile()
        chunks: list[StreamPart] = list(
            graph.stream(
                _MSG_INPUT,
                stream_mode="messages",
                stream_version="v2",
            )
        )
        msg_chunks = [c for c in chunks if c["type"] == "messages"]
        assert len(msg_chunks) >= 1
        for chunk in msg_chunks:
            _assert_stream_part_shape(chunk)
            assert chunk["ns"] == ()
            data = chunk["data"]
            assert isinstance(data, tuple)
            assert len(data) == 2
            message, metadata = data
            assert isinstance(message, BaseMessage)
            assert isinstance(metadata, dict)
            assert "langgraph_node" in metadata
            assert "langgraph_triggers" in metadata
            if chunk["type"] == "messages":
                assert_type(chunk, MessagesStreamPart)

    def test_stream_v2_custom(self) -> None:
        @entrypoint()
        def graph(inputs: Any, *, writer: StreamWriter) -> Any:
            writer("hello")
            writer(42)
            return inputs

        custom_input: Any = {"key": "val"}
        chunks: list[StreamPart] = list(
            graph.stream(
                custom_input,
                stream_mode="custom",
                stream_version="v2",
            )
        )
        custom_chunks = [c for c in chunks if c["type"] == "custom"]
        assert len(custom_chunks) == 2
        for chunk in custom_chunks:
            _assert_stream_part_shape(chunk)
            if chunk["type"] == "custom":
                assert_type(chunk, CustomStreamPart)
        assert custom_chunks[0]["data"] == "hello"
        assert custom_chunks[1]["data"] == 42

    def test_stream_v2_multiple_modes(self) -> None:
        """When stream_mode is a list, v2 still yields uniform StreamPart dicts."""
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode=["values", "updates"],
                stream_version="v2",
            )
        )
        assert len(chunks) > 0
        types_seen: set[str] = set()
        for chunk in chunks:
            _assert_stream_part_shape(chunk)
            types_seen.add(chunk["type"])
        assert "values" in types_seen
        assert "updates" in types_seen

    def test_stream_v2_subgraphs_ns(self) -> None:
        """With subgraphs, ns should reflect the namespace for subgraph events."""
        inner = _make_simple_graph().compile()

        outer_builder = StateGraph(SimpleState)
        outer_builder.add_node("inner", inner)
        outer_builder.add_edge(START, "inner")
        outer_builder.add_edge("inner", END)
        outer = outer_builder.compile()

        chunks: list[StreamPart] = list(
            outer.stream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=True,
                stream_version="v2",
            )
        )
        root_chunks = [c for c in chunks if c["ns"] == ()]
        sub_chunks = [c for c in chunks if c["ns"] != ()]
        assert len(root_chunks) >= 1
        assert len(sub_chunks) >= 1
        for chunk in chunks:
            _assert_stream_part_shape(chunk)
        for chunk in sub_chunks:
            assert len(chunk["ns"]) > 0

    def test_stream_v2_checkpoints_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config: Any = {"configurable": {"thread_id": "test-v2-ckpt"}}
        chunks: list[StreamPart] = list(
            graph.stream(
                _SIMPLE_INPUT,
                config,
                stream_mode="checkpoints",
                stream_version="v2",
            )
        )
        ckpt_chunks = [c for c in chunks if c["type"] == "checkpoints"]
        assert len(ckpt_chunks) >= 1
        for chunk in ckpt_chunks:
            _assert_stream_part_shape(chunk)
            assert chunk["ns"] == ()
            _assert_checkpoint_payload(chunk["data"])
            if chunk["type"] == "checkpoints":
                assert_type(chunk, CheckpointStreamPart)

    def test_stream_v2_tasks_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config: Any = {"configurable": {"thread_id": "test-v2-tasks"}}
        chunks: list[StreamPart] = list(
            graph.stream(
                _SIMPLE_INPUT,
                config,
                stream_mode="tasks",
                stream_version="v2",
            )
        )
        task_chunks = [c for c in chunks if c["type"] == "tasks"]
        assert len(task_chunks) >= 2
        for chunk in task_chunks:
            _assert_stream_part_shape(chunk)
            assert chunk["ns"] == ()
            assert isinstance(chunk["data"], dict)
            assert "id" in chunk["data"]
            assert "name" in chunk["data"]
            if chunk["type"] == "tasks":
                assert_type(chunk, TasksStreamPart)

        start_chunks = [
            c for c in task_chunks if "input" in c["data"] and "triggers" in c["data"]
        ]
        result_chunks = [c for c in task_chunks if "result" in c["data"]]

        assert len(start_chunks) >= 2
        for c in start_chunks:
            _assert_task_start_payload(c["data"])

        assert len(result_chunks) >= 2
        for c in result_chunks:
            _assert_task_result_payload(c["data"])

    def test_stream_v2_debug_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config: Any = {"configurable": {"thread_id": "test-v2-debug"}}
        chunks: list[StreamPart] = list(
            graph.stream(
                _SIMPLE_INPUT,
                config,
                stream_mode="debug",
                stream_version="v2",
            )
        )
        debug_chunks = [c for c in chunks if c["type"] == "debug"]
        assert len(debug_chunks) >= 1
        for chunk in debug_chunks:
            _assert_stream_part_shape(chunk)
            assert chunk["ns"] == ()
            _assert_debug_envelope(chunk["data"])
            if chunk["type"] == "debug":
                assert_type(chunk, DebugStreamPart)

    def test_stream_v2_ignores_subgraphs_param(self) -> None:
        """In v2, subgraphs=True/False should not change the output format.
        ns is always present regardless."""
        graph = _make_simple_graph().compile()
        chunks_no_sub: list[StreamPart] = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=False,
                stream_version="v2",
            )
        )
        chunks_with_sub: list[StreamPart] = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=True,
                stream_version="v2",
            )
        )
        for c in chunks_no_sub:
            _assert_stream_part_shape(c)
        for c in chunks_with_sub:
            _assert_stream_part_shape(c)


# --- v2 invoke ---


class TestV2Invoke:
    """Test v2 invoke format."""

    def test_invoke_v2_values_default(self) -> None:
        """v2 invoke with default stream_mode='values' returns the final state dict."""
        graph = _make_simple_graph().compile()
        result = graph.invoke(_SIMPLE_INPUT, stream_version="v2")
        assert isinstance(result, dict)
        assert result["value"] == "x_a_b"
        assert result["items"] == ["a", "b"]

    def test_invoke_v2_updates_mode(self) -> None:
        """v2 invoke with stream_mode='updates' returns list of StreamPart dicts."""
        graph = _make_simple_graph().compile()
        result = graph.invoke(
            _SIMPLE_INPUT,
            stream_mode="updates",
            stream_version="v2",
        )
        assert isinstance(result, list)
        assert len(result) == 2
        for chunk in result:
            _assert_stream_part_shape(chunk)
            assert chunk["type"] == "updates"
            assert chunk["ns"] == ()
            assert isinstance(chunk["data"], dict)
        assert "node_a" in result[0]["data"]
        assert "node_b" in result[1]["data"]

    def test_invoke_v2_multiple_modes(self) -> None:
        """v2 invoke with list stream_mode returns list of StreamPart dicts."""
        graph = _make_simple_graph().compile()
        modes: Any = ["values", "updates"]
        result = graph.invoke(
            _SIMPLE_INPUT,
            stream_mode=modes,
            stream_version="v2",
        )
        assert isinstance(result, list)
        types_seen: set[str] = set()
        for chunk in result:
            _assert_stream_part_shape(chunk)
            types_seen.add(chunk["type"])
        assert "values" in types_seen
        assert "updates" in types_seen

    def test_invoke_v1_default_unchanged(self) -> None:
        """Default invoke (v1) behavior is unchanged."""
        graph = _make_simple_graph().compile()
        result = graph.invoke(_SIMPLE_INPUT)
        assert isinstance(result, dict)
        assert result["value"] == "x_a_b"
        assert result["items"] == ["a", "b"]

    def test_invoke_v1_updates_unchanged(self) -> None:
        """Default invoke (v1) with updates returns v1-style chunks."""
        graph = _make_simple_graph().compile()
        result = graph.invoke(
            _SIMPLE_INPUT,
            stream_mode="updates",
        )
        assert isinstance(result, list)
        for chunk in result:
            assert "node_a" in chunk or "node_b" in chunk


# --- async tests ---


class TestV2StreamAsync:
    @pytest.mark.anyio
    async def test_astream_v2_values(self) -> None:
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            _SIMPLE_INPUT,
            stream_mode="values",
            stream_version="v2",
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1
        for chunk in chunks:
            _assert_stream_part_shape(chunk)
            assert chunk["type"] == "values"
            assert chunk["ns"] == ()
            assert isinstance(chunk["data"], dict)

    @pytest.mark.anyio
    async def test_astream_v2_updates(self) -> None:
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            _SIMPLE_INPUT,
            stream_mode="updates",
            stream_version="v2",
        ):
            chunks.append(chunk)
        assert len(chunks) == 2
        for chunk in chunks:
            _assert_stream_part_shape(chunk)
        assert chunks[0]["type"] == "updates"
        assert "node_a" in chunks[0]["data"]
        assert chunks[1]["type"] == "updates"
        assert "node_b" in chunks[1]["data"]

    @pytest.mark.anyio
    async def test_astream_v2_messages(self) -> None:
        graph = _make_messages_graph().compile()
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            _MSG_INPUT,
            stream_mode="messages",
            stream_version="v2",
        ):
            chunks.append(chunk)
        msg_chunks = [c for c in chunks if c["type"] == "messages"]
        assert len(msg_chunks) >= 1
        for chunk in msg_chunks:
            _assert_stream_part_shape(chunk)
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
        def graph(inputs: Any, *, writer: StreamWriter) -> Any:
            writer("hello")
            writer(42)
            return inputs

        custom_input: Any = {"key": "val"}
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            custom_input,
            stream_mode="custom",
            stream_version="v2",
        ):
            chunks.append(chunk)
        custom_chunks = [c for c in chunks if c["type"] == "custom"]
        assert len(custom_chunks) == 2
        for chunk in custom_chunks:
            _assert_stream_part_shape(chunk)
        assert custom_chunks[0]["data"] == "hello"
        assert custom_chunks[1]["data"] == 42

    @pytest.mark.anyio
    async def test_astream_v2_multiple_modes(self) -> None:
        graph = _make_simple_graph().compile()
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            _SIMPLE_INPUT,
            stream_mode=["values", "updates"],
            stream_version="v2",
        ):
            chunks.append(chunk)
        types_seen = {c["type"] for c in chunks}
        assert "values" in types_seen
        assert "updates" in types_seen
        for chunk in chunks:
            _assert_stream_part_shape(chunk)

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
            _SIMPLE_INPUT,
            stream_mode="updates",
            subgraphs=True,
            stream_version="v2",
        ):
            chunks.append(chunk)
        for chunk in chunks:
            _assert_stream_part_shape(chunk)
        root_chunks = [c for c in chunks if c["ns"] == ()]
        sub_chunks = [c for c in chunks if c["ns"] != ()]
        assert len(root_chunks) >= 1
        assert len(sub_chunks) >= 1
        for chunk in sub_chunks:
            assert len(chunk["ns"]) > 0

    @pytest.mark.anyio
    async def test_astream_v2_checkpoints_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config: Any = {"configurable": {"thread_id": "test-v2-ckpt-async"}}
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            _SIMPLE_INPUT,
            config,
            stream_mode="checkpoints",
            stream_version="v2",
        ):
            chunks.append(chunk)
        ckpt_chunks = [c for c in chunks if c["type"] == "checkpoints"]
        assert len(ckpt_chunks) >= 1
        for chunk in ckpt_chunks:
            _assert_stream_part_shape(chunk)
            assert chunk["ns"] == ()
            _assert_checkpoint_payload(chunk["data"])

    @pytest.mark.anyio
    async def test_astream_v2_tasks_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config: Any = {"configurable": {"thread_id": "test-v2-tasks-async"}}
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            _SIMPLE_INPUT,
            config,
            stream_mode="tasks",
            stream_version="v2",
        ):
            chunks.append(chunk)
        task_chunks = [c for c in chunks if c["type"] == "tasks"]
        assert len(task_chunks) >= 2
        for chunk in task_chunks:
            _assert_stream_part_shape(chunk)

        start_chunks = [
            c for c in task_chunks if "input" in c["data"] and "triggers" in c["data"]
        ]
        result_chunks = [c for c in task_chunks if "result" in c["data"]]
        assert len(start_chunks) >= 2
        for c in start_chunks:
            _assert_task_start_payload(c["data"])
        assert len(result_chunks) >= 2
        for c in result_chunks:
            _assert_task_result_payload(c["data"])

    @pytest.mark.anyio
    async def test_astream_v2_debug_mode(self) -> None:
        checkpointer = InMemorySaver()
        graph = _make_simple_graph().compile(checkpointer=checkpointer)
        config: Any = {"configurable": {"thread_id": "test-v2-debug-async"}}
        chunks: list[StreamPart] = []
        async for chunk in graph.astream(
            _SIMPLE_INPUT,
            config,
            stream_mode="debug",
            stream_version="v2",
        ):
            chunks.append(chunk)
        debug_chunks = [c for c in chunks if c["type"] == "debug"]
        assert len(debug_chunks) >= 1
        for chunk in debug_chunks:
            _assert_stream_part_shape(chunk)
            assert chunk["ns"] == ()
            _assert_debug_envelope(chunk["data"])


# --- async v2 invoke ---


class TestV2InvokeAsync:
    @pytest.mark.anyio
    async def test_ainvoke_v2_values_default(self) -> None:
        """v2 ainvoke with default stream_mode='values' returns the final state dict."""
        graph = _make_simple_graph().compile()
        result = await graph.ainvoke(_SIMPLE_INPUT, stream_version="v2")
        assert isinstance(result, dict)
        assert result["value"] == "x_a_b"
        assert result["items"] == ["a", "b"]

    @pytest.mark.anyio
    async def test_ainvoke_v2_updates_mode(self) -> None:
        """v2 ainvoke with stream_mode='updates' returns list of StreamPart dicts."""
        graph = _make_simple_graph().compile()
        result = await graph.ainvoke(
            _SIMPLE_INPUT,
            stream_mode="updates",
            stream_version="v2",
        )
        assert isinstance(result, list)
        assert len(result) == 2
        for chunk in result:
            _assert_stream_part_shape(chunk)
            assert chunk["type"] == "updates"
            assert chunk["ns"] == ()
            assert isinstance(chunk["data"], dict)
        assert "node_a" in result[0]["data"]
        assert "node_b" in result[1]["data"]

    @pytest.mark.anyio
    async def test_ainvoke_v2_multiple_modes(self) -> None:
        """v2 ainvoke with list stream_mode returns list of StreamPart dicts."""
        graph = _make_simple_graph().compile()
        modes: Any = ["values", "updates"]
        result = await graph.ainvoke(
            _SIMPLE_INPUT,
            stream_mode=modes,
            stream_version="v2",
        )
        assert isinstance(result, list)
        types_seen: set[str] = set()
        for chunk in result:
            _assert_stream_part_shape(chunk)
            types_seen.add(chunk["type"])
        assert "values" in types_seen
        assert "updates" in types_seen


# --- type narrowing compile-time checks ---
# These assert_type calls verify that mypy narrows the union correctly.


def _check_type_narrowing(part: StreamPart) -> None:
    """Compile-time type narrowing checks — never called at runtime."""
    if part["type"] == "values":
        assert_type(part, ValuesStreamPart)
        assert_type(part["data"], dict[str, Any])
    elif part["type"] == "updates":
        assert_type(part, UpdatesStreamPart)
        assert_type(part["data"], dict[str, Any])
    elif part["type"] == "messages":
        assert_type(part, MessagesStreamPart)
    elif part["type"] == "custom":
        assert_type(part, CustomStreamPart)
    elif part["type"] == "checkpoints":
        assert_type(part, CheckpointStreamPart)
        assert_type(part["data"], CheckpointPayload)
    elif part["type"] == "tasks":
        assert_type(part, TasksStreamPart)
        assert_type(part["data"], TaskPayload | TaskResultPayload)
    elif part["type"] == "debug":
        assert_type(part, DebugStreamPart)
        assert_type(part["data"], DebugPayload)
    # ns is always a tuple
    assert_type(part["ns"], tuple[str, ...])
