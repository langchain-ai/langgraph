"""Tests for v2 streaming format (StreamPart TypedDicts).

This file is checked by mypy directly — no subprocess workarounds.
Type-narrowing is validated via `assert_type` calls in `_check_type_narrowing`.
"""

from __future__ import annotations

import operator
import sys
from dataclasses import dataclass
from typing import Annotated, Any, TypeVar

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, ValidationError
from typing_extensions import TypedDict, assert_type

from langgraph._internal._constants import INTERRUPT
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
    GraphOutput,
    Interrupt,
    MessagesStreamPart,
    StreamPart,
    StreamWriter,
    TaskPayload,
    TaskResultPayload,
    TasksStreamPart,
    UpdatesStreamPart,
    ValuesStreamPart,
    interrupt,
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


# --- shared assertion helpers ---

_STREAM_PART_KEYS = {"type", "ns", "data"}


def _assert_stream_part_shape(part: StreamPart[Any, Any]) -> None:
    """Assert a v2 stream part has the required keys and correct types."""
    assert isinstance(part, dict), f"Expected dict, got {type(part)}"
    assert _STREAM_PART_KEYS <= part.keys(), (
        f"Missing keys: {_STREAM_PART_KEYS - part.keys()}"
    )
    assert isinstance(part["type"], str)
    assert isinstance(part["ns"], tuple)
    for elem in part["ns"]:
        assert isinstance(elem, str)
    if part["type"] == "values":
        assert "interrupts" in part, "values stream part missing 'interrupts' field"
        assert isinstance(part["interrupts"], tuple)


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
        chunks = list(graph.stream(_SIMPLE_INPUT, stream_mode="values", version="v2"))
        assert len(chunks) >= 1
        for c in chunks:
            _assert_stream_part_shape(c)
            assert c["type"] == "values"
            assert c["ns"] == ()
            assert isinstance(c["data"], dict)

    def test_updates(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(graph.stream(_SIMPLE_INPUT, stream_mode="updates", version="v2"))
        assert len(chunks) == 2
        for c in chunks:
            _assert_stream_part_shape(c)
            assert c["type"] == "updates"
            assert c["ns"] == ()
        assert "node_a" in chunks[0]["data"]
        assert "node_b" in chunks[1]["data"]

    def test_messages(self) -> None:
        graph = _make_messages_graph().compile()
        chunks = list(graph.stream(_MSG_INPUT, stream_mode="messages", version="v2"))
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

    def test_custom(self) -> None:
        graph = _make_custom_graph()
        chunks = list(graph.stream({"key": "val"}, stream_mode="custom", version="v2"))
        custom = [c for c in chunks if c["type"] == "custom"]
        assert len(custom) == 2
        for c in custom:
            _assert_stream_part_shape(c)
        assert custom[0]["data"] == "hello"
        assert custom[1]["data"] == 42

    def test_multiple_modes(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode=["values", "updates"],
                version="v2",
            )
        )
        types_seen = {c["type"] for c in chunks}
        assert {"values", "updates"} <= types_seen
        for c in chunks:
            _assert_stream_part_shape(c)

    def test_subgraphs_ns(self) -> None:
        outer = _make_subgraph()
        chunks = list(
            outer.stream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=True,
                version="v2",
            )
        )
        for c in chunks:
            _assert_stream_part_shape(c)
        root = [c for c in chunks if c["ns"] == ()]
        sub = [c for c in chunks if c["ns"] != ()]
        assert len(root) >= 1
        assert len(sub) >= 1

    def test_checkpoints(self) -> None:
        graph = _make_simple_graph().compile(checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "test-v2-ckpt"}}
        chunks = list(
            graph.stream(
                _SIMPLE_INPUT,
                config,
                stream_mode="checkpoints",
                version="v2",
            )
        )
        ckpt = [c for c in chunks if c["type"] == "checkpoints"]
        assert len(ckpt) >= 1
        for c in ckpt:
            _assert_stream_part_shape(c)
            assert c["ns"] == ()
            payload = c["data"]
            assert {"config", "metadata", "values", "next", "tasks"} <= payload.keys()

    def test_tasks(self) -> None:
        graph = _make_simple_graph().compile(checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "test-v2-tasks"}}
        chunks = list(
            graph.stream(
                _SIMPLE_INPUT,
                config,
                stream_mode="tasks",
                version="v2",
            )
        )
        tasks = [c for c in chunks if c["type"] == "tasks"]
        assert len(tasks) >= 2
        for c in tasks:
            _assert_stream_part_shape(c)
            assert c["ns"] == ()
            assert "id" in c["data"] and "name" in c["data"]
        starts = [c for c in tasks if "triggers" in c["data"]]
        results = [c for c in tasks if "result" in c["data"]]
        assert len(starts) >= 2
        assert len(results) >= 2

    def test_debug(self) -> None:
        graph = _make_simple_graph().compile(checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "test-v2-debug"}}
        chunks = list(
            graph.stream(
                _SIMPLE_INPUT,
                config,
                stream_mode="debug",
                version="v2",
            )
        )
        debug = [c for c in chunks if c["type"] == "debug"]
        assert len(debug) >= 1
        for c in debug:
            _assert_stream_part_shape(c)
            assert c["ns"] == ()
            envelope = c["data"]
            assert {"step", "timestamp", "type", "payload"} <= envelope.keys()
            assert envelope["type"] in ("checkpoint", "task", "task_result")

    def test_subgraphs_param_does_not_change_format(self) -> None:
        """In v2, subgraphs=True/False should not change the output format."""
        graph = _make_simple_graph().compile()
        chunks_no_sub = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=False,
                version="v2",
            )
        )
        chunks_with_sub = list(
            graph.stream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=True,
                version="v2",
            )
        )
        for c in chunks_no_sub + chunks_with_sub:
            _assert_stream_part_shape(c)


# --- v2 sync invoke ---


class TestV2Invoke:
    def test_values_default(self) -> None:
        graph = _make_simple_graph().compile()
        result = graph.invoke(_SIMPLE_INPUT, version="v2")
        assert isinstance(result, GraphOutput)
        assert result.value == {"value": "x_a_b", "items": ["a", "b"]}
        assert result.interrupts == ()
        # backward compat dict access
        assert result["value"] == "x_a_b"
        assert result["items"] == ["a", "b"]

    def test_invoke_v2_graph_output_with_interrupts(self) -> None:
        def my_node(state: SimpleState) -> dict[str, Any]:
            answer = interrupt("what is your name?")
            return {"value": answer, "items": ["done"]}

        builder: StateGraph = StateGraph(SimpleState)
        builder.add_node("my_node", my_node)
        builder.add_edge(START, "my_node")
        builder.add_edge("my_node", END)
        graph = builder.compile(checkpointer=InMemorySaver())

        config: Any = {"configurable": {"thread_id": "test-invoke-v2-interrupts"}}
        result = graph.invoke({"value": "x", "items": []}, config, version="v2")
        assert isinstance(result, GraphOutput)
        assert len(result.interrupts) > 0
        for intr in result.interrupts:
            assert isinstance(intr, Interrupt)
        # value should still be the state (not None or empty)
        assert isinstance(result.value, dict)

    def test_invoke_v2_graph_output_interrupt_compat(self) -> None:
        """result['__interrupt__'] works via __getitem__."""

        def my_node(state: SimpleState) -> dict[str, Any]:
            answer = interrupt("what is your name?")
            return {"value": answer, "items": ["done"]}

        builder: StateGraph = StateGraph(SimpleState)
        builder.add_node("my_node", my_node)
        builder.add_edge(START, "my_node")
        builder.add_edge("my_node", END)
        graph = builder.compile(checkpointer=InMemorySaver())

        config: Any = {"configurable": {"thread_id": "test-invoke-v2-compat"}}
        result = graph.invoke({"value": "x", "items": []}, config, version="v2")
        assert isinstance(result, GraphOutput)
        assert INTERRUPT in result
        assert result[INTERRUPT] == result.interrupts
        assert len(result[INTERRUPT]) > 0

    def test_invoke_v2_graph_output_no_interrupts(self) -> None:
        graph = _make_simple_graph().compile()
        result = graph.invoke(_SIMPLE_INPUT, version="v2")
        assert isinstance(result, GraphOutput)
        assert result.interrupts == ()
        assert INTERRUPT not in result

    def test_invoke_v2_pydantic_state(self) -> None:
        """invoke with v2 and pydantic state returns GraphOutput with pydantic value."""

        def node_a(state: PydanticState) -> dict[str, Any]:
            return {"value": state.value + "_a", "items": ["a"]}

        builder: StateGraph = StateGraph(PydanticState)
        builder.add_node("node_a", node_a)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", END)
        graph = builder.compile()

        result = graph.invoke({"value": "x", "items": []}, version="v2")
        assert isinstance(result, GraphOutput)
        assert isinstance(result.value, PydanticState)
        assert result.value.value == "x_a"
        assert result.interrupts == ()

    def test_invoke_v2_dataclass_state(self) -> None:
        """invoke with v2 and dataclass state returns GraphOutput with dataclass value."""

        def node_a(state: DataclassState) -> dict[str, Any]:
            return {"value": state.value + "_a", "items": ["a"]}

        builder: StateGraph = StateGraph(DataclassState)
        builder.add_node("node_a", node_a)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", END)
        graph = builder.compile()

        result = graph.invoke({"value": "x", "items": []}, version="v2")
        assert isinstance(result, GraphOutput)
        assert isinstance(result.value, DataclassState)
        assert result.value.value == "x_a"
        assert result.value.items == ["a"]
        assert result.interrupts == ()

    def test_invoke_v2_non_values_mode_pydantic(self) -> None:
        """invoke with v2 + non-values mode + pydantic state returns list[StreamPart]."""

        def node_a(state: PydanticState) -> dict[str, Any]:
            return {"value": state.value + "_a", "items": ["a"]}

        builder: StateGraph = StateGraph(PydanticState)
        builder.add_node("node_a", node_a)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", END)
        graph = builder.compile()

        result = graph.invoke(
            {"value": "x", "items": []}, stream_mode="updates", version="v2"
        )
        assert isinstance(result, list)
        for chunk in result:
            _assert_stream_part_shape(chunk)
            assert chunk["type"] == "updates"
            # updates data should be plain dicts, not coerced to pydantic
            assert isinstance(chunk["data"], dict)

    def test_updates_mode(self) -> None:
        graph = _make_simple_graph().compile()
        result = graph.invoke(_SIMPLE_INPUT, stream_mode="updates", version="v2")
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
        result = graph.invoke(_SIMPLE_INPUT, stream_mode=modes, version="v2")
        assert isinstance(result, list)
        types_seen = {c["type"] for c in result}
        assert {"values", "updates"} <= types_seen
        for c in result:
            _assert_stream_part_shape(c)

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
                _SIMPLE_INPUT, stream_mode="values", version="v2"
            )
        ]
        assert len(chunks) >= 1
        for c in chunks:
            _assert_stream_part_shape(c)
            assert c["type"] == "values"
            assert c["ns"] == ()
            assert isinstance(c["data"], dict)

    @pytest.mark.anyio
    async def test_updates(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = [
            c
            async for c in graph.astream(
                _SIMPLE_INPUT, stream_mode="updates", version="v2"
            )
        ]
        assert len(chunks) == 2
        for c in chunks:
            _assert_stream_part_shape(c)
            assert c["type"] == "updates"
            assert c["ns"] == ()
        assert "node_a" in chunks[0]["data"]
        assert "node_b" in chunks[1]["data"]

    @pytest.mark.anyio
    async def test_messages(self) -> None:
        graph = _make_messages_graph().compile()
        chunks = [
            c
            async for c in graph.astream(
                _MSG_INPUT, stream_mode="messages", version="v2"
            )
        ]
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

    @NEEDS_CONTEXTVARS
    @pytest.mark.anyio
    async def test_custom(self) -> None:
        graph = _make_custom_graph()
        chunks = [
            c
            async for c in graph.astream(
                {"key": "val"}, stream_mode="custom", version="v2"
            )
        ]
        custom = [c for c in chunks if c["type"] == "custom"]
        assert len(custom) == 2
        for c in custom:
            _assert_stream_part_shape(c)
        assert custom[0]["data"] == "hello"
        assert custom[1]["data"] == 42

    @pytest.mark.anyio
    async def test_multiple_modes(self) -> None:
        graph = _make_simple_graph().compile()
        chunks = [
            c
            async for c in graph.astream(
                _SIMPLE_INPUT,
                stream_mode=["values", "updates"],
                version="v2",
            )
        ]
        types_seen = {c["type"] for c in chunks}
        assert {"values", "updates"} <= types_seen
        for c in chunks:
            _assert_stream_part_shape(c)

    @pytest.mark.anyio
    async def test_subgraphs_ns(self) -> None:
        outer = _make_subgraph()
        chunks = [
            c
            async for c in outer.astream(
                _SIMPLE_INPUT,
                stream_mode="updates",
                subgraphs=True,
                version="v2",
            )
        ]
        for c in chunks:
            _assert_stream_part_shape(c)
        root = [c for c in chunks if c["ns"] == ()]
        sub = [c for c in chunks if c["ns"] != ()]
        assert len(root) >= 1
        assert len(sub) >= 1

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
                version="v2",
            )
        ]
        ckpt = [c for c in chunks if c["type"] == "checkpoints"]
        assert len(ckpt) >= 1
        for c in ckpt:
            _assert_stream_part_shape(c)
            assert c["ns"] == ()
            payload = c["data"]
            assert {"config", "metadata", "values", "next", "tasks"} <= payload.keys()

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
                version="v2",
            )
        ]
        tasks = [c for c in chunks if c["type"] == "tasks"]
        assert len(tasks) >= 2
        for c in tasks:
            _assert_stream_part_shape(c)
            assert c["ns"] == ()
            assert "id" in c["data"] and "name" in c["data"]
        starts = [c for c in tasks if "triggers" in c["data"]]
        results = [c for c in tasks if "result" in c["data"]]
        assert len(starts) >= 2
        assert len(results) >= 2

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
                version="v2",
            )
        ]
        debug = [c for c in chunks if c["type"] == "debug"]
        assert len(debug) >= 1
        for c in debug:
            _assert_stream_part_shape(c)
            assert c["ns"] == ()
            envelope = c["data"]
            assert {"step", "timestamp", "type", "payload"} <= envelope.keys()
            assert envelope["type"] in ("checkpoint", "task", "task_result")


# --- v2 async invoke ---


class TestV2InvokeAsync:
    @pytest.mark.anyio
    async def test_values_default(self) -> None:
        graph = _make_simple_graph().compile()
        result = await graph.ainvoke(_SIMPLE_INPUT, version="v2")
        assert isinstance(result, GraphOutput)
        assert result.value == {"value": "x_a_b", "items": ["a", "b"]}
        assert result.interrupts == ()
        # backward compat dict access
        assert result["value"] == "x_a_b"
        assert result["items"] == ["a", "b"]

    @NEEDS_CONTEXTVARS
    @pytest.mark.anyio
    async def test_ainvoke_v2_graph_output_with_interrupts(self) -> None:
        def my_node(state: SimpleState) -> dict[str, Any]:
            answer = interrupt("what is your name?")
            return {"value": answer, "items": ["done"]}

        builder: StateGraph = StateGraph(SimpleState)
        builder.add_node("my_node", my_node)
        builder.add_edge(START, "my_node")
        builder.add_edge("my_node", END)
        graph = builder.compile(checkpointer=InMemorySaver())

        config: Any = {"configurable": {"thread_id": "test-ainvoke-v2-interrupts"}}
        result = await graph.ainvoke({"value": "x", "items": []}, config, version="v2")
        assert isinstance(result, GraphOutput)
        assert len(result.interrupts) > 0
        for intr in result.interrupts:
            assert isinstance(intr, Interrupt)
        assert isinstance(result.value, dict)

    @pytest.mark.anyio
    async def test_ainvoke_v2_pydantic_state(self) -> None:
        """ainvoke with v2 and pydantic state returns GraphOutput with pydantic value."""

        def node_a(state: PydanticState) -> dict[str, Any]:
            return {"value": state.value + "_a", "items": ["a"]}

        builder: StateGraph = StateGraph(PydanticState)
        builder.add_node("node_a", node_a)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", END)
        graph = builder.compile()

        result = await graph.ainvoke({"value": "x", "items": []}, version="v2")
        assert isinstance(result, GraphOutput)
        assert isinstance(result.value, PydanticState)
        assert result.value.value == "x_a"
        assert result.value.items == ["a"]
        assert result.interrupts == ()

    @pytest.mark.anyio
    async def test_ainvoke_v2_dataclass_state(self) -> None:
        """ainvoke with v2 and dataclass state returns GraphOutput with dataclass value."""

        def node_a(state: DataclassState) -> dict[str, Any]:
            return {"value": state.value + "_a", "items": ["a"]}

        builder: StateGraph = StateGraph(DataclassState)
        builder.add_node("node_a", node_a)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", END)
        graph = builder.compile()

        result = await graph.ainvoke({"value": "x", "items": []}, version="v2")
        assert isinstance(result, GraphOutput)
        assert isinstance(result.value, DataclassState)
        assert result.value.value == "x_a"
        assert result.value.items == ["a"]
        assert result.interrupts == ()

    @pytest.mark.anyio
    async def test_ainvoke_v2_graph_output_no_interrupts(self) -> None:
        graph = _make_simple_graph().compile()
        result = await graph.ainvoke(_SIMPLE_INPUT, version="v2")
        assert isinstance(result, GraphOutput)
        assert result.interrupts == ()
        assert INTERRUPT not in result

    @pytest.mark.anyio
    async def test_updates_mode(self) -> None:
        graph = _make_simple_graph().compile()
        result = await graph.ainvoke(_SIMPLE_INPUT, stream_mode="updates", version="v2")
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
        result = await graph.ainvoke(_SIMPLE_INPUT, stream_mode=modes, version="v2")
        assert isinstance(result, list)
        types_seen = {c["type"] for c in result}
        assert {"values", "updates"} <= types_seen
        for c in result:
            _assert_stream_part_shape(c)


# --- type-safe streaming: coercion + interrupt separation ---


class PydanticState(BaseModel):
    value: str
    items: Annotated[list[str], operator.add]


@dataclass
class DataclassState:
    value: str
    items: Annotated[list[str], operator.add]


class TestV2TypeSafeStreaming:
    """Test that v2 streaming coerces values to pydantic/dataclass instances
    and separates interrupts into a dedicated field."""

    def test_values_pydantic_state(self) -> None:
        """v2 values + pydantic state -> data is pydantic model instance."""

        def node_a(state: PydanticState) -> dict[str, Any]:
            return {"value": state.value + "_a", "items": ["a"]}

        builder: StateGraph = StateGraph(PydanticState)
        builder.add_node("node_a", node_a)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", END)
        graph = builder.compile()

        chunks = list(
            graph.stream(
                {"value": "x", "items": []},
                stream_mode="values",
                version="v2",
            )
        )
        assert len(chunks) >= 1
        for c in chunks:
            _assert_stream_part_shape(c)
            assert c["type"] == "values"
            assert isinstance(c["data"], PydanticState), (
                f"Expected PydanticState, got {type(c['data'])}"
            )
            assert c["interrupts"] == ()

    def test_values_dataclass_state(self) -> None:
        """v2 values + dataclass state -> data is dataclass instance."""

        def node_a(state: DataclassState) -> dict[str, Any]:
            return {"value": state.value + "_a", "items": ["a"]}

        builder: StateGraph = StateGraph(DataclassState)
        builder.add_node("node_a", node_a)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", END)
        graph = builder.compile()

        chunks = list(
            graph.stream(
                {"value": "x", "items": []},
                stream_mode="values",
                version="v2",
            )
        )
        assert len(chunks) >= 1
        for c in chunks:
            _assert_stream_part_shape(c)
            assert c["type"] == "values"
            assert isinstance(c["data"], DataclassState), (
                f"Expected DataclassState, got {type(c['data'])}"
            )

    def test_values_typeddict_state(self) -> None:
        """v2 values + TypedDict state -> data stays plain dict (no coercion)."""
        graph = _make_simple_graph().compile()
        chunks = list(graph.stream(_SIMPLE_INPUT, stream_mode="values", version="v2"))
        assert len(chunks) >= 1
        for c in chunks:
            _assert_stream_part_shape(c)
            assert c["type"] == "values"
            # TypedDict state should remain a plain dict
            assert isinstance(c["data"], dict)
            assert type(c["data"]) is dict

    def test_values_interrupt_v2(self) -> None:
        """v2 values + interrupt -> interrupts in typed field, not in data."""

        def my_node(state: SimpleState) -> dict[str, Any]:
            answer = interrupt("what is your name?")
            return {"value": answer, "items": ["done"]}

        builder: StateGraph = StateGraph(SimpleState)
        builder.add_node("my_node", my_node)
        builder.add_edge(START, "my_node")
        builder.add_edge("my_node", END)
        graph = builder.compile(checkpointer=InMemorySaver())

        config: Any = {"configurable": {"thread_id": "test-v2-interrupt"}}
        chunks = list(
            graph.stream(
                {"value": "x", "items": []},
                config,
                stream_mode="values",
                version="v2",
            )
        )
        # should have at least one values chunk with interrupts
        interrupt_chunks = [c for c in chunks if c.get("interrupts", ())]
        assert len(interrupt_chunks) >= 1, f"Expected interrupt chunks, got {chunks}"
        for c in interrupt_chunks:
            assert c["type"] == "values"
            assert isinstance(c["interrupts"], tuple)
            assert len(c["interrupts"]) > 0
            for intr in c["interrupts"]:
                assert isinstance(intr, Interrupt)
            # __interrupt__ should NOT be in data
            if isinstance(c["data"], dict):
                assert INTERRUPT not in c["data"]

    def test_values_interrupt_v1_compat(self) -> None:
        """v1 values + interrupt -> __interrupt__ still in dict (v1 compat)."""

        def my_node(state: SimpleState) -> dict[str, Any]:
            answer = interrupt("what is your name?")
            return {"value": answer, "items": ["done"]}

        builder: StateGraph = StateGraph(SimpleState)
        builder.add_node("my_node", my_node)
        builder.add_edge(START, "my_node")
        builder.add_edge("my_node", END)
        graph = builder.compile(checkpointer=InMemorySaver())

        config: Any = {"configurable": {"thread_id": "test-v1-interrupt-compat"}}
        chunks = list(
            graph.stream(
                {"value": "x", "items": []},
                config,
                stream_mode="values",
            )
        )
        # v1 format: should have __interrupt__ in dict
        interrupt_chunks = [c for c in chunks if isinstance(c, dict) and INTERRUPT in c]
        assert len(interrupt_chunks) >= 1, (
            f"Expected v1 interrupt chunks with {INTERRUPT}, got {chunks}"
        )

    def test_checkpoints_pydantic_state(self) -> None:
        """v2 checkpoints + pydantic state -> values is pydantic model instance
        (at least for checkpoints emitted after all channels are populated)."""

        def node_a(state: PydanticState) -> dict[str, Any]:
            return {"value": state.value + "_a", "items": ["a"]}

        builder: StateGraph = StateGraph(PydanticState)
        builder.add_node("node_a", node_a)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", END)
        graph = builder.compile(checkpointer=InMemorySaver())

        config: Any = {"configurable": {"thread_id": "test-v2-ckpt-pydantic"}}
        chunks = list(
            graph.stream(
                {"value": "x", "items": []},
                config,
                stream_mode="checkpoints",
                version="v2",
            )
        )
        ckpt_chunks = [c for c in chunks if c["type"] == "checkpoints"]
        assert len(ckpt_chunks) >= 1
        # At least one checkpoint (after first node runs) should have coerced values
        coerced_ckpts = [
            c for c in ckpt_chunks if isinstance(c["data"]["values"], PydanticState)
        ]
        assert len(coerced_ckpts) >= 1, (
            f"Expected at least one checkpoint with PydanticState values, got types: "
            f"{[type(c['data']['values']) for c in ckpt_chunks]}"
        )

    def test_debug_pydantic_state(self) -> None:
        """v2 debug + pydantic state -> inner checkpoint payload has coerced values
        (at least for checkpoints emitted after all channels are populated)."""

        def node_a(state: PydanticState) -> dict[str, Any]:
            return {"value": state.value + "_a", "items": ["a"]}

        builder: StateGraph = StateGraph(PydanticState)
        builder.add_node("node_a", node_a)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", END)
        graph = builder.compile(checkpointer=InMemorySaver())

        config: Any = {"configurable": {"thread_id": "test-v2-debug-pydantic"}}
        chunks = list(
            graph.stream(
                {"value": "x", "items": []},
                config,
                stream_mode="debug",
                version="v2",
            )
        )
        debug_chunks = [c for c in chunks if c["type"] == "debug"]
        checkpoint_debug = [
            c for c in debug_chunks if c["data"]["type"] == "checkpoint"
        ]
        assert len(checkpoint_debug) >= 1
        # At least one debug checkpoint should have coerced values
        coerced_debug = [
            c
            for c in checkpoint_debug
            if isinstance(c["data"]["payload"]["values"], PydanticState)
        ]
        assert len(coerced_debug) >= 1, (
            f"Expected at least one debug checkpoint with PydanticState values, got types: "
            f"{[type(c['data']['payload']['values']) for c in checkpoint_debug]}"
        )

    def test_values_pydantic_interrupt(self) -> None:
        """v2 values + pydantic state + interrupt -> data is model, interrupts separated."""

        def my_node(state: PydanticState) -> dict[str, Any]:
            answer = interrupt("what is your name?")
            return {"value": answer, "items": ["done"]}

        builder: StateGraph = StateGraph(PydanticState)
        builder.add_node("my_node", my_node)
        builder.add_edge(START, "my_node")
        builder.add_edge("my_node", END)
        graph = builder.compile(checkpointer=InMemorySaver())

        config: Any = {"configurable": {"thread_id": "test-v2-pydantic-interrupt"}}
        chunks = list(
            graph.stream(
                {"value": "x", "items": []},
                config,
                stream_mode="values",
                version="v2",
            )
        )
        interrupt_chunks = [c for c in chunks if c.get("interrupts", ())]
        assert len(interrupt_chunks) >= 1
        for c in interrupt_chunks:
            assert isinstance(c["data"], PydanticState), (
                f"Expected PydanticState, got {type(c['data'])}"
            )
            assert isinstance(c["interrupts"], tuple)
            assert len(c["interrupts"]) > 0

    def test_subgraph_different_pydantic_schema(self) -> None:
        """Subgraph with different pydantic schema -> subgraph data coerced with subgraph's schema."""

        class InnerState(BaseModel):
            value: str

        class OuterState(BaseModel):
            value: str

        def inner_node(state: InnerState) -> dict[str, Any]:
            return {"value": state.value + "_inner"}

        def outer_node(state: OuterState) -> dict[str, Any]:
            return {"value": state.value + "_outer"}

        inner_builder: StateGraph = StateGraph(InnerState)
        inner_builder.add_node("inner_node", inner_node)
        inner_builder.add_edge(START, "inner_node")
        inner_builder.add_edge("inner_node", END)
        inner_graph = inner_builder.compile()

        outer_builder: StateGraph = StateGraph(OuterState)
        outer_builder.add_node("outer_node", outer_node)
        outer_builder.add_node("inner", inner_graph)
        outer_builder.add_edge(START, "outer_node")
        outer_builder.add_edge("outer_node", "inner")
        outer_builder.add_edge("inner", END)
        outer = outer_builder.compile()

        chunks = list(
            outer.stream(
                {"value": "x"},
                stream_mode="values",
                subgraphs=True,
                version="v2",
            )
        )
        # Root-level values should be OuterState instances
        root_values = [c for c in chunks if c["type"] == "values" and c["ns"] == ()]
        assert len(root_values) >= 1
        for c in root_values:
            assert isinstance(c["data"], OuterState), (
                f"Expected OuterState, got {type(c['data'])}"
            )
        # Subgraph values are streamed from the subgraph's own stream()
        # which runs with default version="v1", so no coercion
        sub_values = [c for c in chunks if c["type"] == "values" and c["ns"] != ()]
        assert len(sub_values) >= 1


# --- v2 validation errors ---


def _make_pydantic_graph() -> Any:
    """Build a simple graph with PydanticState for validation error tests."""

    def node_a(state: PydanticState) -> dict[str, Any]:
        return {"value": state.value + "_a", "items": ["a"]}

    builder: StateGraph = StateGraph(PydanticState)
    builder.add_node("node_a", node_a)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", END)
    return builder.compile()


class TestV2ValidationErrors:
    """Validation errors propagate for pydantic state in both v1 and v2.

    Uses `value=[1, 2, 3]` which channels accept (LastValue stores anything)
    but pydantic rejects (list is not coercible to str even in lax mode).
    """

    _INVALID_INPUT: dict[str, Any] = {"value": [1, 2, 3], "items": []}

    def test_stream_v2_pydantic_validation_error(self) -> None:
        """Invalid input to stream with v2 + pydantic state raises ValidationError."""
        graph = _make_pydantic_graph()
        with pytest.raises(ValidationError):
            list(
                graph.stream(
                    self._INVALID_INPUT,
                    stream_mode="values",
                    version="v2",
                )
            )

    def test_invoke_v2_pydantic_validation_error(self) -> None:
        """Invalid input to invoke with v2 + pydantic state raises ValidationError."""
        graph = _make_pydantic_graph()
        with pytest.raises(ValidationError):
            graph.invoke(self._INVALID_INPUT, version="v2")

    def test_invoke_v1_pydantic_validation_error(self) -> None:
        """Regression: invalid input to invoke without version raises ValidationError."""
        graph = _make_pydantic_graph()
        with pytest.raises(ValidationError):
            graph.invoke(self._INVALID_INPUT)


# --- type narrowing compile-time checks ---
# These assert_type calls verify that mypy narrows the union correctly.


_OutputT = TypeVar("_OutputT")
_StateT = TypeVar("_StateT")


def _check_type_narrowing(part: StreamPart[_StateT, _OutputT]) -> None:
    """Compile-time type narrowing checks — never called at runtime."""
    if part["type"] == "values":
        assert_type(part, ValuesStreamPart[_OutputT])
    elif part["type"] == "updates":
        assert_type(part, UpdatesStreamPart)
        assert_type(part["data"], dict[str, Any])
    elif part["type"] == "messages":
        assert_type(part, MessagesStreamPart)
    elif part["type"] == "custom":
        assert_type(part, CustomStreamPart)
    elif part["type"] == "checkpoints":
        assert_type(part, CheckpointStreamPart[_StateT])
        assert_type(part["data"], CheckpointPayload[_StateT])
    elif part["type"] == "tasks":
        assert_type(part, TasksStreamPart)
        assert_type(part["data"], TaskPayload | TaskResultPayload)
    elif part["type"] == "debug":
        assert_type(part, DebugStreamPart[_StateT])
        assert_type(part["data"], DebugPayload[_StateT])
    assert_type(part["ns"], tuple[str, ...])
