"""Tests for StreamToolCallHandler and emit_tool_output_delta.

These tests exercise the langgraph-core piece in isolation — the prebuilt
`ToolCallTransformer` has its own test file. Here we feed real graphs
through `Pregel.stream(stream_mode=["tools", ...])` and inspect the raw
`(ns, mode, payload)` tuples on the `tools` channel.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from langgraph.config import emit_tool_output_delta
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _caller_sync(tool_name: str, tool_args: dict[str, Any], tc_id: str = "tc1"):
    def caller(state: _State) -> dict:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": tool_name, "args": tool_args, "id": tc_id}],
                )
            ]
        }

    return caller


def _caller_async(tool_name: str, tool_args: dict[str, Any], tc_id: str = "tc1"):
    async def caller(state: _State) -> dict:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": tool_name, "args": tool_args, "id": tc_id}],
                )
            ]
        }

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
        events = _tool_events(
            graph.stream(
                {"messages": []},
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

    def test_emit_tool_output_delta_produces_delta_events(self) -> None:
        @tool
        def streaming_echo(text: str) -> str:
            """stream chunks."""
            for chunk in ("a", "b", "c"):
                emit_tool_output_delta(chunk)
            return text

        graph = _build_graph(
            _caller_sync("streaming_echo", {"text": "x"}), [streaming_echo]
        )
        events = _tool_events(
            graph.stream(
                {"messages": []},
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
        events: list[tuple[tuple[str, ...], dict]] = []
        with pytest.raises(ValueError, match="nope"):
            for ns, mode, payload in graph.stream(
                {"messages": []},
                stream_mode=["tools"],
                subgraphs=True,
            ):
                if mode == "tools":
                    events.append((tuple(ns), payload))

        kinds = [p["event"] for _, p in events]
        assert kinds == ["tool-started", "tool-error"]
        assert events[1][1]["message"] == "nope"

    def test_emit_outside_tool_is_noop(self) -> None:
        # Called at import time (outside any tool body) — must not raise.
        emit_tool_output_delta("ignored")
        emit_tool_output_delta({"any": "payload"})

    def test_no_events_without_tools_mode(self) -> None:
        @tool
        def echo(text: str) -> str:
            """echo."""
            return text

        graph = _build_graph(_caller_sync("echo", {"text": "hi"}), [echo])
        # No "tools" in stream_mode — handler is not attached and zero
        # `tools`-method events fire.
        chunks = list(
            graph.stream(
                {"messages": []},
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
        async def aecho(text: str) -> str:
            """async echo."""
            emit_tool_output_delta(text)
            return f"got:{text}"

        graph = _build_graph(_caller_async("aecho", {"text": "hi"}), [aecho])
        events: list[tuple[tuple[str, ...], dict]] = []
        async for ns, mode, payload in graph.astream(
            {"messages": []},
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
        def streamer(marker: str) -> str:
            """emits marker twice."""
            emit_tool_output_delta(f"{marker}-1")
            emit_tool_output_delta(f"{marker}-2")
            return marker

        def caller(state: _State) -> dict:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "streamer", "args": {"marker": "A"}, "id": "a"},
                            {"name": "streamer", "args": {"marker": "B"}, "id": "b"},
                        ],
                    )
                ]
            }

        graph = _build_graph(caller, [streamer])
        events = _tool_events(
            graph.stream(
                {"messages": []},
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
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "inner_tool",
                                "args": {"text": "x"},
                                "id": "tc1",
                            }
                        ],
                    )
                ]
            }

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

        events = _tool_events(
            graph.stream(
                {"messages": []},
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
