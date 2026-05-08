"""Tests for StreamToolCallHandler and ToolRuntime.emit_output_delta.

These tests exercise the langgraph-core piece in isolation — the prebuilt
`ToolCallTransformer` has its own test file. Here we feed real graphs
through `Pregel.stream(stream_mode=["tools", ...])` and inspect the raw
`(ns, mode, payload)` tuples on the `tools` channel.
"""

from __future__ import annotations

import logging
import re
import base64
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, ToolRuntime
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.pregel._tools import _tool_call_writer

logger = logging.getLogger(__name__)

_DANGEROUS_PATTERN = re.compile(
    r"(base64|eval\(|exec\(|__import__|subprocess|os\.system|shell=True"
    r"|<script|javascript:|data:text/html"
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f])",
    re.IGNORECASE,
)

_INVISIBLE_PATTERN = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060-\u2064\ufeff]"
)


def _is_base64_encoded(value: str) -> bool:
    try:
        if len(value) % 4 == 0 and len(value) >= 8:
            decoded = base64.b64decode(value, validate=True)
            decoded_str = decoded.decode("utf-8", errors="ignore")
            if _DANGEROUS_PATTERN.search(decoded_str):
                return True
    except Exception:
        pass
    return False


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        if _DANGEROUS_PATTERN.search(value):
            raise ValueError(f"Potentially dangerous content detected in input: {value!r}")
        if _INVISIBLE_PATTERN.search(value):
            raise ValueError(f"Invisible/hidden characters detected in input: {value!r}")
        if _is_base64_encoded(value):
            raise ValueError(f"Potentially dangerous base64-encoded content detected: {value!r}")
    elif isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _sanitize_tool_args(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("tool_name must be a non-empty string")
    _sanitize_value(tool_name)
    return _sanitize_value(tool_args)


def _sanitize_graph_input(graph_input: dict[str, Any]) -> dict[str, Any]:
    return _sanitize_value(graph_input)


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _caller_sync(tool_name: str, tool_args: dict[str, Any], tc_id: str = "tc1"):
    sanitized_args = _sanitize_tool_args(tool_name, tool_args)

    def caller(state: _State) -> dict:
        ai_message = AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": sanitized_args, "id": tc_id}],
        )
        logger.info(
            "LLM interaction: tool_call name=%s args=%r tc_id=%s",
            tool_name,
            sanitized_args,
            tc_id,
        )
        return {"messages": [ai_message]}

    return caller


def _caller_async(tool_name: str, tool_args: dict[str, Any], tc_id: str = "tc1"):
    sanitized_args = _sanitize_tool_args(tool_name, tool_args)

    async def caller(state: _State) -> dict:
        ai_message = AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": sanitized_args, "id": tc_id}],
        )
        logger.info(
            "LLM interaction (async): tool_call name=%s args=%r tc_id=%s",
            tool_name,
            sanitized_args,
            tc_id,
        )
        return {"messages": [ai_message]}

    return caller


def _build_graph(caller, tools) -> Any:
    sg = StateGraph(_State)
    sg.add_node("caller", caller)
    sg.add_node("tools", ToolNode(tools))
    sg.add_edge(START, "caller")
    sg.add_edge("caller", "tools")
    sg.add_edge("tools", END)
    return sg.compile()


def _tool_events(stream) -> list[tuple[tuple[str, ...], dict]]:
    """Collect `(ns, payload)` for every `tools`-mode chunk."""
    out: list[tuple[tuple[str, ...], dict]] = []
    for ns, mode, payload in stream:
        if mode == "tools":
            out.append((tuple(ns), payload))
    return out


class TestSyncGraphSyncTool:
    def test_started_finished_cycle(self) -> None:
        @tool
        def echo(text: str) -> str:
            """echo."""
            return f"echoed:{text}"

        graph = _build_graph(_caller_sync("echo", {"text": "hi"}), [echo])
        sanitized_input = _sanitize_graph_input({"messages": []})
        events = _tool_events(
            graph.stream(
                sanitized_input,
                stream_mode=["tools"],
                subgraphs=True,
            )
        )

        assert [p["event"] for _, p in events] == [
            "tool-started",
            "tool-finished",
        ]
        assert events[0][1]["tool_call_id"] == "tc1"
        assert events[0][1]["tool_name"] == "echo"
        assert events[0][1]["input"] == {"text": "hi"}
        # ToolNode wraps the return in a ToolMessage.
        assert events[1][1]["tool_call_id"] == "tc1"

    def test_emit_output_delta_produces_delta_events(self) -> None:
        @tool
        def streaming_echo(text: str, runtime: ToolRuntime) -> str:
            """stream chunks."""
            for chunk in ("a", "b", "c"):
                runtime.emit_output_delta(chunk)
            return text

        graph = _build_graph(
            _caller_sync("streaming_echo", {"text": "x"}), [streaming_echo]
        )
        sanitized_input = _sanitize_graph_input({"messages": []})
        events = _tool_events(
            graph.stream(
                sanitized_input,
                stream_mode=["tools"],
                subgraphs=True,
            )
        )

        deltas = [p["delta"] for _, p in events if p["event"] == "tool-output-delta"]
        assert deltas == ["a", "b", "c"]
        # The deltas must be bracketed by started and finished.
        ordered = [p["event"] for _, p in events]
        assert ordered[0] == "tool-started"
        assert ordered[-1] == "tool-finished"

    def test_tool_error_event(self) -> None:
        @tool
        def boom() -> str:
            """raises."""
            raise ValueError("nope")

        graph = _build_graph(_caller_sync("boom", {}), [boom])
        sanitized_input = _sanitize_graph_input({"messages": []})
        events: list[tuple[tuple[str, ...], dict]] = []
        with pytest.raises(ValueError, match="nope"):
            for ns, mode, payload in graph.stream(
                sanitized_input,
                stream_mode=["tools"],
                subgraphs=True,
            ):
                if mode == "tools":
                    events.append((tuple(ns), payload))

        kinds = [p["event"] for _, p in events]
        assert kinds == ["tool-started", "tool-error"]
        assert events[1][1]["message"] == "nope"

    def test_writer_unset_outside_tool(self) -> None:
        # Outside any tool body the ContextVar that ToolRuntime reads
        # is unset — emitting from there would be a no-op.
        assert _tool_call_writer.get() is None

    def test_no_events_without_tools_mode(self) -> None:
        @tool
        def echo(text: str) -> str:
            """echo."""
            return text

        graph = _build_graph(_caller_sync("echo", {"text": "hi"}), [echo])
        sanitized_input = _sanitize_graph_input({"messages": []})
        # No "tools" in stream_mode — handler is not attached and zero
        # `tools`-method events fire.
        chunks = list(
            graph.stream(
                sanitized_input,
                stream_mode=["values"],
                subgraphs=True,
            )
        )
        assert all(
            not (isinstance(c, tuple) and len(c) == 3 and c[1] == "tools")
            for c in chunks
        )


class TestAsyncGraphAsyncTool:
    @pytest.mark.anyio
    async def test_async_tool_produces_events(self) -> None:
        @tool
        async def aecho(text: str, runtime: ToolRuntime) -> str:
            """async echo."""
            runtime.emit_output_delta(text)
            return f"got:{text}"

        graph = _build_graph(_caller_async("aecho", {"text": "hi"}), [aecho])
        sanitized_input = _sanitize_graph_input({"messages": []})
        events: list[tuple[tuple[str, ...], dict]] = []
        async for ns, mode, payload in graph.astream(
            sanitized_input,
            stream_mode=["tools"],
            subgraphs=True,
        ):
            if mode == "tools":
                events.append((tuple(ns), payload))

        kinds = [p["event"] for _, p in events]
        assert kinds == ["tool-started", "tool-output-delta", "tool-finished"]
        assert events[1][1]["delta"] == "hi"


class TestConcurrentToolCalls:
    def test_parallel_tool_calls_do_not_bleed(self) -> None:
        @tool
        def streamer(marker: str, runtime: ToolRuntime) -> str:
            """emits marker twice."""
            runtime.emit_output_delta(f"{marker}-1")
            runtime.emit_output_delta(f"{marker}-2")
            return marker

        def caller(state: _State) -> dict:
            ai_message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "streamer", "args": {"marker": "A"}, "id": "a"},
                    {"name": "streamer", "args": {"marker": "B"}, "id": "b"},
                ],
            )
            logger.info(
                "LLM interaction: parallel tool_calls streamer A and B"
            )
            return {"messages": [ai_message]}

        graph = _build_graph(caller, [streamer])
        sanitized_input = _sanitize_graph_input({"messages": []})
        events = _tool_events(
            graph.stream(
                sanitized_input,
                stream_mode=["tools"],
                subgraphs=True,
            )
        )

        # Group deltas by tool_call_id.
        by_id: dict[str, list[str]] = {}
        for _, p in events:
            if p["event"] == "tool-output-delta":
                by_id.setdefault(p["tool_call_id"], []).append(p["delta"])
        assert by_id["a"] == ["A-1", "A-2"]
        assert by_id["b"] == ["B-1", "B-2"]


class TestSubgraphNamespacePropagation:
    def test_tool_inside_subgraph_emits_with_subgraph_ns(self) -> None:
        @tool
        def inner_tool(text: str) -> str:
            """inner tool."""
            return text

        def sub_caller(state: _State) -> dict:
            ai_message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "inner_tool",
                        "args": {"text": "x"},
                        "id": "tc1",
                    }
                ],
            )
            logger.info(
                "LLM interaction: sub_caller tool_call inner_tool args={'text': 'x'} tc_id=tc1"
            )
            return {"messages": [ai_message]}

        inner = StateGraph(_State)
        inner.add_node("sub_caller", sub_caller)
        inner.add_node("sub_tools", ToolNode([inner_tool]))
        inner.add_edge(START, "sub_caller")
        inner.add_edge("sub_caller", "sub_tools")
        inner.add_edge("sub_tools", END)
        inner_graph = inner.compile()

        outer = StateGraph(_State)
        outer.add_node("sub", inner_graph)
        outer.add_edge(START, "sub")
        outer.add_edge("sub", END)
        graph = outer.compile()

        sanitized_input = _sanitize_graph_input({"messages": []})
        events = _tool_events(
            graph.stream(
                sanitized_input,
                stream_mode=["tools"],
                subgraphs=True,
            )
        )

        # All `tools` events should carry a non-empty namespace rooted
        # at the `sub` node.
        assert events, "expected at least one tools event"
        for ns, _ in events:
            assert ns  # non-empty
            assert ns[0].startswith("sub:")