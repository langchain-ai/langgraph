"""Tests for ToolCallTransformer and the ToolCallStream projection."""

from __future__ import annotations

import time
from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.stream_channel import StreamChannel
from langgraph.stream.transformers import (
    MessagesTransformer,
    ValuesTransformer,
)
from typing_extensions import TypedDict

from langgraph.prebuilt import (
    ToolCallTransformer,
    ToolNode,
    ToolRuntime,
)
from langgraph.prebuilt._tool_call_stream import ToolCallStream

TS = int(time.time() * 1000)


def _unstamped(items):
    """Strip push stamps from a StreamChannel's internal buffer."""
    return [item for _stamp, item in items]


def _tool_event(
    event: str,
    tool_call_id: str,
    *,
    tool_name: str = "",
    input: dict[str, Any] | None = None,
    delta: Any = None,
    output: Any = None,
    message: str = "",
    namespace: list[str] | None = None,
) -> ProtocolEvent:
    data: dict[str, Any] = {"event": event, "tool_call_id": tool_call_id}
    if event == "tool-started":
        data["tool_name"] = tool_name
        if input is not None:
            data["input"] = input
    elif event == "tool-output-delta":
        data["delta"] = delta
    elif event == "tool-finished":
        data["output"] = output
    elif event == "tool-error":
        data["message"] = message
    return {
        "type": "event",
        "method": "tools",
        "params": {
            "namespace": namespace or [],
            "timestamp": TS,
            "data": data,
        },
    }


def _subscribe(log: StreamChannel) -> None:
    log._subscribed = True


def _mux() -> tuple[StreamMux, ToolCallTransformer]:
    transformer = ToolCallTransformer()
    mux = StreamMux(
        [
            ValuesTransformer(),
            MessagesTransformer(),
            transformer,
        ],
        is_async=False,
    )
    _subscribe(transformer._log)
    return mux, transformer


class TestToolCallTransformerUnit:
    def test_required_stream_modes_declares_tools(self) -> None:
        assert ToolCallTransformer.required_stream_modes == ("tools",)

    def test_tool_started_yields_handle(self) -> None:
        mux, transformer = _mux()
        mux.push(
            _tool_event(
                "tool-started",
                "tc1",
                tool_name="echo",
                input={"text": "hi"},
            )
        )
        handles = _unstamped(transformer._log._items)
        assert len(handles) == 1
        h = handles[0]
        assert isinstance(h, ToolCallStream)
        assert h.tool_call_id == "tc1"
        assert h.tool_name == "echo"
        assert h.input == {"text": "hi"}
        assert h.completed is False

    def test_delta_accumulates_on_active_stream(self) -> None:
        mux, transformer = _mux()
        mux.push(_tool_event("tool-started", "tc1", tool_name="echo"))
        _subscribe(transformer._active["tc1"]._output_deltas)
        mux.push(_tool_event("tool-output-delta", "tc1", delta="a"))
        mux.push(_tool_event("tool-output-delta", "tc1", delta="b"))
        stream = transformer._active["tc1"]
        assert _unstamped(stream._output_deltas._items) == ["a", "b"]

    def test_finish_closes_stream(self) -> None:
        mux, transformer = _mux()
        mux.push(_tool_event("tool-started", "tc1", tool_name="echo"))
        stream = transformer._active["tc1"]
        mux.push(_tool_event("tool-finished", "tc1", output="done"))
        assert stream.completed is True
        assert stream.output == "done"
        assert stream.error is None
        assert "tc1" not in transformer._active

    def test_error_closes_stream(self) -> None:
        mux, transformer = _mux()
        mux.push(_tool_event("tool-started", "tc1", tool_name="boom"))
        stream = transformer._active["tc1"]
        mux.push(_tool_event("tool-error", "tc1", message="nope"))
        assert stream.completed is True
        assert stream.output is None
        assert stream.error == "nope"
        assert "tc1" not in transformer._active

    def test_concurrent_tool_calls_do_not_bleed(self) -> None:
        mux, transformer = _mux()
        mux.push(_tool_event("tool-started", "a", tool_name="t"))
        mux.push(_tool_event("tool-started", "b", tool_name="t"))
        for tc in ("a", "b"):
            _subscribe(transformer._active[tc]._output_deltas)
        mux.push(_tool_event("tool-output-delta", "a", delta="A1"))
        mux.push(_tool_event("tool-output-delta", "b", delta="B1"))
        mux.push(_tool_event("tool-output-delta", "a", delta="A2"))
        assert _unstamped(transformer._active["a"]._output_deltas._items) == [
            "A1",
            "A2",
        ]
        assert _unstamped(transformer._active["b"]._output_deltas._items) == ["B1"]

    def test_tools_event_passes_through_main_log(self) -> None:
        mux, transformer = _mux()
        _subscribe(mux._events)
        mux.push(_tool_event("tool-started", "tc1", tool_name="echo"))
        kept = [e for e in _unstamped(mux._events._items) if e["method"] == "tools"]
        assert len(kept) == 1


# ---------------------------------------------------------------------------
# End-to-end tests with a real graph
# ---------------------------------------------------------------------------


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _build_graph(caller, tools):
    sg = StateGraph(_State)
    sg.add_node("caller", caller)
    sg.add_node("tools", ToolNode(tools))
    sg.add_edge(START, "caller")
    sg.add_edge("caller", "tools")
    sg.add_edge("tools", END)
    return sg.compile()


class TestToolCallTransformerEndToEnd:
    def test_sync_streaming_tool_populates_tool_calls(self) -> None:
        @tool
        def streamer(text: str, runtime: ToolRuntime) -> str:
            """streams chunks."""
            for chunk in ("one", "two"):
                runtime.emit_output_delta(chunk)
            return text

        def caller(state: _State) -> dict:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "streamer", "args": {"text": "x"}, "id": "tc1"}
                        ],
                    )
                ]
            }

        graph = _build_graph(caller, [streamer])
        run = graph.stream_v2({"messages": []}, transformers=[ToolCallTransformer])

        tool_calls: list[ToolCallStream] = []
        for tc in run.tool_calls:
            tool_calls.append(tc)
            deltas = list(tc.output_deltas)
            assert deltas == ["one", "two"]
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc.tool_call_id == "tc1"
        assert tc.tool_name == "streamer"
        assert tc.completed is True
        assert tc.error is None

    def test_stream_modes_union_includes_tools(self) -> None:
        @tool
        def echo(text: str) -> str:
            """echo."""
            return text

        def caller(state: _State) -> dict:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "echo", "args": {"text": "x"}, "id": "tc1"}
                        ],
                    )
                ]
            }

        graph = _build_graph(caller, [echo])
        # Without ToolCallTransformer, no tool_calls projection is
        # exposed and no `tools` events flow through (required_stream_modes
        # omits it).
        run_no_tc = graph.stream_v2({"messages": []})
        assert "tool_calls" not in run_no_tc._mux.extensions  # type: ignore[attr-defined]

        # With ToolCallTransformer, the projection is present.
        run = graph.stream_v2({"messages": []}, transformers=[ToolCallTransformer])
        assert "tool_calls" in run._mux.extensions  # type: ignore[attr-defined]
        # Drain so the run closes cleanly.
        list(run.tool_calls)

    @pytest.mark.anyio
    async def test_async_streaming_tool_populates_tool_calls(self) -> None:
        @tool
        async def astreamer(text: str, runtime: ToolRuntime) -> str:
            """async streams."""
            runtime.emit_output_delta(text)
            runtime.emit_output_delta(text + "!")
            return text

        async def caller(state: _State) -> dict:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "astreamer", "args": {"text": "hi"}, "id": "tc1"}
                        ],
                    )
                ]
            }

        graph = _build_graph(caller, [astreamer])
        run = await graph.astream_v2(
            {"messages": []}, transformers=[ToolCallTransformer]
        )

        collected: list[ToolCallStream] = []
        async for tc in run.tool_calls:
            collected.append(tc)
            deltas = [d async for d in tc.output_deltas]
            assert deltas == ["hi", "hi!"]
        assert len(collected) == 1
        assert collected[0].completed is True
        assert collected[0].error is None

    def test_tool_error_populates_error_field(self) -> None:
        @tool
        def boom() -> str:
            """raises."""
            raise ValueError("nope")

        def caller(state: _State) -> dict:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "boom", "args": {}, "id": "tc1"}],
                    )
                ]
            }

        graph = _build_graph(caller, [boom])
        run = graph.stream_v2({"messages": []}, transformers=[ToolCallTransformer])

        collected: list[ToolCallStream] = []
        with pytest.raises(ValueError, match="nope"):
            for tc in run.tool_calls:
                collected.append(tc)
                # Drain deltas so the error field is populated before we
                # inspect it below.
                list(tc.output_deltas)

        assert len(collected) == 1
        assert collected[0].error == "nope"
        assert collected[0].output is None
        assert collected[0].completed is True
