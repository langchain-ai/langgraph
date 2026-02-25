"""Tests for v2 streaming format (StreamPart TypedDicts).

This file is checked by mypy directly — no subprocess workarounds.
Type-narrowing is validated via `assert_type` calls in `_check_type_narrowing`.
"""

from __future__ import annotations

import operator
import sys
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

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)

# --- state and graph builders ---


class SimpleState(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


_SIMPLE_INPUT: SimpleState = {"value": "x", "items": []}
_MSG_INPUT: MessagesState = {"messages": "hi"}


def _make_simple_graph() -> StateGraph[SimpleState, None, SimpleState, SimpleState]:
    def node_a(state: SimpleState) -> dict[str, Any]:
        return {"value": state["value"] + "_a", "items": ["a"]}

    def node_b(state: SimpleState) -> dict[str, Any]:
        return {"value": state["value"] + "_b", "items": ["b"]}

    builder = StateGraph(SimpleState, input_schema=SimpleState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    return builder


def _make_messages_graph() -> StateGraph[
    MessagesState, None, MessagesState, MessagesState
]:
    model = FakeChatModel(messages=[AIMessage(content="hello world")])

    def call_model(state: MessagesState) -> dict[str, Any]:
        return {"messages": model.invoke(state["messages"])}

    builder = StateGraph(MessagesState, input_schema=MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    return builder


def _make_custom_graph() -> Any:
    @entrypoint()
    def graph(inputs: Any, *, writer: StreamWriter) -> Any:
        writer("hello")
        writer(42)
        return inputs

    return graph


def _make_subgraph() -> Any:
    inner = _make_simple_graph().compile()
    outer_builder = StateGraph(SimpleState, input_schema=SimpleState)
    outer_builder.add_node("inner", inner)
    outer_builder.add_edge(START, "inner")
    outer_builder.add_edge("inner", END)
    return outer_builder.compile()


# --- payload assertion helpers ---

_STREAM_PART_KEYS = {"type", "ns", "data"}


def _assert_stream_part_shape(part: StreamPart) -> None:
    assert isinstance(part, dict), f"Expected dict, got {type(part)}"
    assert _STREAM_PART_KEYS <= part.keys(), (
        f"Missing keys: {_STREAM_PART_KEYS - part.keys()}"
    )
    assert isinstance(part["type"], str)
    assert isinstance(part["ns"], tuple)
    for elem in part["ns"]:
        assert isinstance(elem, str)


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
    assert _CHECKPOINT_PAYLOAD_KEYS <= payload.keys()
    assert isinstance(payload["values"], dict)
    assert isinstance(payload["next"], list)
    assert isinstance(payload["tasks"], list)
    assert isinstance(payload["metadata"], dict)


def _assert_task_start_payload(payload: Any) -> None:
    assert _TASK_START_KEYS <= payload.keys()
    assert isinstance(payload["id"], str)
    assert isinstance(payload["name"], str)
    assert isinstance(payload["triggers"], (list, tuple))


def _assert_task_result_payload(payload: Any) -> None:
    assert _TASK_RESULT_KEYS <= payload.keys()
    assert isinstance(payload["id"], str)
    assert isinstance(payload["name"], str)
    assert isinstance(payload["interrupts"], list)
    assert isinstance(payload["result"], dict)


def _assert_debug_envelope(envelope: Any) -> None:
    assert _DEBUG_ENVELOPE_KEYS <= envelope.keys()
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


# --- chunk-level assertion helpers (shared by sync and async tests) ---


def _assert_values_chunks(chunks: list[StreamPart]) -> None:
    assert len(chunks) >= 1
    for c in chunks:
        _assert_stream_part_shape(c)
        assert c["type"] == "values"
        assert c["ns"] == ()
        assert isinstance(c["data"], dict)


def _assert_updates_chunks(chunks: list[StreamPart]) -> None:
    assert len(chunks) == 2
    for c in chunks:
        _assert_stream_part_shape(c)
        assert c["type"] == "updates"
        assert c["ns"] == ()
        assert isinstance(c["data"], dict)
    assert "node_a" in chunks[0]["data"]
    assert "node_b" in chunks[1]["data"]


def _assert_messages_chunks(chunks: list[StreamPart]) -> None:
    msg_chunks = [c for c in chunks if c["type"] == "messages"]
    assert len(msg_chunks) >= 1
    for c in msg_chunks:
        _assert_stream_part_shape(c)
        assert c["ns"] == ()
        data = c["data"]
        assert isinstance(data, tuple) and len(data) == 2
        message, metadata = data
        assert isinstance(message, BaseMessage)
        assert isinstance(metadata, dict)
        assert "langgraph_node" in metadata
        assert "langgraph_triggers" in metadata


def _assert_custom_chunks(chunks: list[StreamPart]) -> None:
    custom_chunks = [c for c in chunks if c["type"] == "custom"]
    assert len(custom_chunks) == 2
    for c in custom_chunks:
        _assert_stream_part_shape(c)
    assert custom_chunks[0]["data"] == "hello"
    assert custom_chunks[1]["data"] == 42


def _assert_multiple_modes_chunks(chunks: list[StreamPart]) -> None:
    assert len(chunks) > 0
    types_seen = {c["type"] for c in chunks}
    assert "values" in types_seen
    assert "updates" in types_seen
    for c in chunks:
        _assert_stream_part_shape(c)


def _assert_subgraph_chunks(chunks: list[StreamPart]) -> None:
    for c in chunks:
        _assert_stream_part_shape(c)
    root = [c for c in chunks if c["ns"] == ()]
    sub = [c for c in chunks if c["ns"] != ()]
    assert len(root) >= 1
    assert len(sub) >= 1
    for c in sub:
        assert len(c["ns"]) > 0


def _assert_checkpoint_chunks(chunks: list[StreamPart]) -> None:
    ckpt = [c for c in chunks if c["type"] == "checkpoints"]
    assert len(ckpt) >= 1
    for c in ckpt:
        _assert_stream_part_shape(c)
        assert c["ns"] == ()
        _assert_checkpoint_payload(c["data"])


def _assert_task_chunks(chunks: list[StreamPart]) -> None:
    tasks = [c for c in chunks if c["type"] == "tasks"]
    assert len(tasks) >= 2
    for c in tasks:
        _assert_stream_part_shape(c)
        assert c["ns"] == ()
        assert isinstance(c["data"], dict)
        assert "id" in c["data"]
        assert "name" in c["data"]
    starts = [c for c in tasks if "input" in c["data"] and "triggers" in c["data"]]
    results = [c for c in tasks if "result" in c["data"]]
    assert len(starts) >= 2
    for c in starts:
        _assert_task_start_payload(c["data"])
    assert len(results) >= 2
    for c in results:
        _assert_task_result_payload(c["data"])


def _assert_debug_chunks(chunks: list[StreamPart]) -> None:
    debug = [c for c in chunks if c["type"] == "debug"]
    assert len(debug) >= 1
    for c in debug:
        _assert_stream_part_shape(c)
        assert c["ns"] == ()
        _assert_debug_envelope(c["data"])


# --- v1 backwards compatibility ---


class TestV1BackwardsCompat:
    def test_stream_default_is_v1(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(graph.stream(_SIMPLE_INPUT))
        for chunk in chunks:
            assert isinstance(chunk, dict)

    def test_stream_v1_updates_mode(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(graph.stream(_SIMPLE_INPUT, stream_mode="updates"))
        assert len(chunks) == 2
        assert "node_a" in chunks[0]
        assert "node_b" in chunks[1]

    def test_stream_v1_list_mode(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(graph.stream(_SIMPLE_INPUT, stream_mode=["values", "updates"]))
        for chunk in chunks:
            assert isinstance(chunk, tuple) and len(chunk) == 2
            mode, _data = chunk
            assert mode in ("values", "updates")

    def test_stream_v1_subgraphs(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(
            graph.stream(_SIMPLE_INPUT, stream_mode="updates", subgraphs=True)
        )
        for chunk in chunks:
            assert isinstance(chunk, tuple) and len(chunk) == 2
            ns, _data = chunk
            assert isinstance(ns, tuple)


# --- v2 sync stream ---


class TestV2Stream:
    def test_values(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(
            graph.stream(_SIMPLE_INPUT, stream_mode="values", stream_version="v2")
        )
        _assert_values_chunks(chunks)

    def test_updates(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(
            graph.stream(_SIMPLE_INPUT, stream_mode="updates", stream_version="v2")
        )
        _assert_updates_chunks(chunks)

    def test_messages(self) -> None:
        graph = _make_messages_graph().compile()
        chunks = list(
            graph.stream(_MSG_INPUT, stream_mode="messages", stream_version="v2")
        )
        _assert_messages_chunks(chunks)

    def test_custom(self) -> None:
        graph = _make_custom_graph()
        chunks = list(
            graph.stream({"key": "val"}, stream_mode="custom", stream_version="v2")
        )
        _assert_custom_chunks(chunks)

    def test_multiple_modes(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode=["values", "updates"],
                stream_version="v2",
            )
        )
        _assert_multiple_modes_chunks(chunks)

    def test_subgraphs_ns(self) -> None:
        outer = _make_subgraph()
        chunks = list(
            outer.stream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=True,
                stream_version="v2",
            )
        )
        _assert_subgraph_chunks(chunks)

    def test_checkpoints(self) -> None:
        graph = _make_simple_graph().compile(checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "test-v2-ckpt"}}
        chunks = list(
            graph.stream(
                _SIMPLE_INPUT,
                config,
                stream_mode="checkpoints",
                stream_version="v2",
            )
        )
        _assert_checkpoint_chunks(chunks)

    def test_tasks(self) -> None:
        graph = _make_simple_graph().compile(checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "test-v2-tasks"}}
        chunks = list(
            graph.stream(
                _SIMPLE_INPUT,
                config,
                stream_mode="tasks",
                stream_version="v2",
            )
        )
        _assert_task_chunks(chunks)

    def test_debug(self) -> None:
        graph = _make_simple_graph().compile(checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "test-v2-debug"}}
        chunks = list(
            graph.stream(
                _SIMPLE_INPUT,
                config,
                stream_mode="debug",
                stream_version="v2",
            )
        )
        _assert_debug_chunks(chunks)

    def test_subgraphs_param_does_not_change_format(self) -> None:
        """In v2, subgraphs=True/False should not change the output format."""
        graph = _make_simple_graph().compile()
        chunks_no_sub = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=False,
                stream_version="v2",
            )
        )
        chunks_with_sub = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=True,
                stream_version="v2",
            )
        )
        for c in chunks_no_sub + chunks_with_sub:
            _assert_stream_part_shape(c)


# --- v2 sync invoke ---


class TestV2Invoke:
    def test_values_default(self) -> None:
        graph = _make_simple_graph().compile()
        result = graph.invoke(_SIMPLE_INPUT, stream_version="v2")
        assert isinstance(result, dict)
        assert result["value"] == "x_a_b"
        assert result["items"] == ["a", "b"]

    def test_updates_mode(self) -> None:
        graph = _make_simple_graph().compile()
        result = graph.invoke(_SIMPLE_INPUT, stream_mode="updates", stream_version="v2")
        assert isinstance(result, list) and len(result) == 2
        for chunk in result:
            _assert_stream_part_shape(chunk)
            assert chunk["type"] == "updates"
            assert chunk["ns"] == ()
        assert "node_a" in result[0]["data"]
        assert "node_b" in result[1]["data"]

    def test_multiple_modes(self) -> None:
        graph = _make_simple_graph().compile()
        modes: Any = ["values", "updates"]
        result = graph.invoke(_SIMPLE_INPUT, stream_mode=modes, stream_version="v2")
        assert isinstance(result, list)
        _assert_multiple_modes_chunks(result)

    def test_v1_default_unchanged(self) -> None:
        graph = _make_simple_graph().compile()
        result = graph.invoke(_SIMPLE_INPUT)
        assert isinstance(result, dict)
        assert result["value"] == "x_a_b"
        assert result["items"] == ["a", "b"]

    def test_v1_updates_unchanged(self) -> None:
        graph = _make_simple_graph().compile()
        result = graph.invoke(_SIMPLE_INPUT, stream_mode="updates")
        assert isinstance(result, list)
        for chunk in result:
            assert "node_a" in chunk or "node_b" in chunk


# --- v2 async stream ---


class TestV2StreamAsync:
    @pytest.mark.anyio
    async def test_values(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = [
            c
            async for c in graph.astream(
                _SIMPLE_INPUT, stream_mode="values", stream_version="v2"
            )
        ]
        _assert_values_chunks(chunks)

    @pytest.mark.anyio
    async def test_updates(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = [
            c
            async for c in graph.astream(
                _SIMPLE_INPUT, stream_mode="updates", stream_version="v2"
            )
        ]
        _assert_updates_chunks(chunks)

    @pytest.mark.anyio
    async def test_messages(self) -> None:
        graph = _make_messages_graph().compile()
        chunks = [
            c
            async for c in graph.astream(
                _MSG_INPUT, stream_mode="messages", stream_version="v2"
            )
        ]
        _assert_messages_chunks(chunks)

    @NEEDS_CONTEXTVARS
    @pytest.mark.anyio
    async def test_custom(self) -> None:
        graph = _make_custom_graph()
        chunks = [
            c
            async for c in graph.astream(
                {"key": "val"}, stream_mode="custom", stream_version="v2"
            )
        ]
        _assert_custom_chunks(chunks)

    @pytest.mark.anyio
    async def test_multiple_modes(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = [
            c
            async for c in graph.astream(
                _SIMPLE_INPUT,
                stream_mode=["values", "updates"],
                stream_version="v2",
            )
        ]
        _assert_multiple_modes_chunks(chunks)

    @pytest.mark.anyio
    async def test_subgraphs_ns(self) -> None:
        outer = _make_subgraph()
        chunks = [
            c
            async for c in outer.astream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=True,
                stream_version="v2",
            )
        ]
        _assert_subgraph_chunks(chunks)

    @pytest.mark.anyio
    async def test_checkpoints(self) -> None:
        graph = _make_simple_graph().compile(checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "test-v2-ckpt-async"}}
        chunks = [
            c
            async for c in graph.astream(
                _SIMPLE_INPUT,
                config,
                stream_mode="checkpoints",
                stream_version="v2",
            )
        ]
        _assert_checkpoint_chunks(chunks)

    @pytest.mark.anyio
    async def test_tasks(self) -> None:
        graph = _make_simple_graph().compile(checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "test-v2-tasks-async"}}
        chunks = [
            c
            async for c in graph.astream(
                _SIMPLE_INPUT,
                config,
                stream_mode="tasks",
                stream_version="v2",
            )
        ]
        _assert_task_chunks(chunks)

    @pytest.mark.anyio
    async def test_debug(self) -> None:
        graph = _make_simple_graph().compile(checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "test-v2-debug-async"}}
        chunks = [
            c
            async for c in graph.astream(
                _SIMPLE_INPUT,
                config,
                stream_mode="debug",
                stream_version="v2",
            )
        ]
        _assert_debug_chunks(chunks)


# --- v2 async invoke ---


class TestV2InvokeAsync:
    @pytest.mark.anyio
    async def test_values_default(self) -> None:
        graph = _make_simple_graph().compile()
        result = await graph.ainvoke(_SIMPLE_INPUT, stream_version="v2")
        assert isinstance(result, dict)
        assert result["value"] == "x_a_b"
        assert result["items"] == ["a", "b"]

    @pytest.mark.anyio
    async def test_updates_mode(self) -> None:
        graph = _make_simple_graph().compile()
        result = await graph.ainvoke(
            _SIMPLE_INPUT, stream_mode="updates", stream_version="v2"
        )
        assert isinstance(result, list) and len(result) == 2
        for chunk in result:
            _assert_stream_part_shape(chunk)
            assert chunk["type"] == "updates"
            assert chunk["ns"] == ()
        assert "node_a" in result[0]["data"]
        assert "node_b" in result[1]["data"]

    @pytest.mark.anyio
    async def test_multiple_modes(self) -> None:
        graph = _make_simple_graph().compile()
        modes: Any = ["values", "updates"]
        result = await graph.ainvoke(
            _SIMPLE_INPUT, stream_mode=modes, stream_version="v2"
        )
        assert isinstance(result, list)
        _assert_multiple_modes_chunks(result)


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
    assert_type(part["ns"], tuple[str, ...])
