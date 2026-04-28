"""Tests for MessagesTransformer: protocol event routing, whole-message fallback,
legacy v1 chunk filtering, and end-to-end via stream_v2 / astream_v2."""

from __future__ import annotations

import time
from typing import Any

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    ChatModelStream,
)
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.stream._event_log import EventLog
from langgraph.stream._mux import StreamMux
from langgraph.stream.run_stream import GraphRunStream
from langgraph.stream.transformers import MessagesTransformer, ValuesTransformer

TS = int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proto_event(
    event: dict[str, Any],
    *,
    run_id: str = "run-1",
    node: str = "llm",
) -> dict[str, Any]:
    """Build a messages ProtocolEvent carrying a protocol event dict (v2 path)."""
    return {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": [],
            "timestamp": TS,
            "data": (event, {"langgraph_node": node, "run_id": run_id}),
        },
    }


def _v1_chunk(
    text: str,
    msg_id: str = "msg-1",
    *,
    finish: bool = False,
    node: str = "llm",
) -> dict[str, Any]:
    """Build a messages ProtocolEvent carrying a v1 AIMessageChunk tuple."""
    rm: dict[str, Any] = {"finish_reason": "stop"} if finish else {}
    return {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": [],
            "timestamp": TS,
            "data": (
                AIMessageChunk(content=text, id=msg_id, response_metadata=rm),
                {"langgraph_node": node},
            ),
        },
    }


def _whole_msg(
    text: str,
    msg_id: str = "msg-10",
    *,
    node: str = "node",
) -> dict[str, Any]:
    """Build a messages ProtocolEvent carrying a completed AIMessage."""
    return {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": [],
            "timestamp": TS,
            "data": (AIMessage(content=text, id=msg_id), {"langgraph_node": node}),
        },
    }


def _make_sync_transformer() -> tuple[MessagesTransformer, EventLog[ChatModelStream]]:
    t = MessagesTransformer()
    log: EventLog[ChatModelStream] = t.init()["messages"]
    log._bind(is_async=False)
    # Subscribe up front so pushes during process() are retained.
    log._subscribed = True
    t._bind_pump(lambda: False)
    return t, log


def _make_async_transformer() -> tuple[MessagesTransformer, EventLog[ChatModelStream]]:
    t = MessagesTransformer()
    log: EventLog[ChatModelStream] = t.init()["messages"]
    log._bind(is_async=True)
    log._subscribed = True
    return t, log


def _lifecycle(
    *, text: str = "hello world", message_id: str = "run-1"
) -> list[dict[str, Any]]:
    """Produce a valid protocol event lifecycle: start, delta, finish."""
    half = len(text) // 2
    first, second = text[:half], text[half:]
    return [
        {"event": "message-start", "role": "ai", "message_id": message_id},
        {
            "event": "content-block-start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "event": "content-block-delta",
            "index": 0,
            "content_block": {"type": "text", "text": first},
        },
        {
            "event": "content-block-delta",
            "index": 0,
            "content_block": {"type": "text", "text": second},
        },
        {
            "event": "content-block-finish",
            "index": 0,
            "content_block": {"type": "text", "text": text},
        },
        {"event": "message-finish", "reason": "stop"},
    ]


def _simple_graph():
    def call_model(state: MessagesState) -> dict[str, Any]:
        model = GenericFakeChatModel(messages=iter(["hello world"]))
        stream = model.stream_v2(state["messages"])
        return {"messages": stream.output}

    return (
        StateGraph(MessagesState)
        .add_node("call_model", call_model)
        .add_edge(START, "call_model")
        .add_edge("call_model", END)
        .compile()
    )


# ---------------------------------------------------------------------------
# Protocol event routing
# ---------------------------------------------------------------------------


class TestProtocolEventRouting:
    def test_message_start_creates_stream(self) -> None:
        t, log = _make_sync_transformer()
        t.process(
            _proto_event(
                {"event": "message-start", "role": "ai", "message_id": "run-1"},
                run_id="run-1",
            )
        )
        log.close()
        (stream,) = list(log._items)
        assert isinstance(stream, ChatModelStream)
        assert stream.message_id == "run-1"

    def test_full_lifecycle_yields_done_stream(self) -> None:
        t, log = _make_sync_transformer()
        for evt in _lifecycle(text="hello world"):
            t.process(_proto_event(evt, run_id="run-1"))
        log.close()
        (stream,) = list(log._items)
        assert stream.done
        assert stream.output.text == "hello world"

    def test_message_finish_cleans_up_routing(self) -> None:
        t, log = _make_sync_transformer()
        for evt in _lifecycle():
            t.process(_proto_event(evt, run_id="run-1"))
        assert t._by_run == {}

    def test_events_without_prior_start_are_ignored(self) -> None:
        t, log = _make_sync_transformer()
        t.process(
            _proto_event(
                {
                    "event": "content-block-delta",
                    "index": 0,
                    "content_block": {"type": "text", "text": "orphan"},
                },
                run_id="unknown",
            )
        )
        log.close()
        assert list(log._items) == []

    def test_concurrent_streams_routed_by_run_id(self) -> None:
        t, log = _make_sync_transformer()
        life_a = _lifecycle(text="aaaa", message_id="run-a")
        life_b = _lifecycle(text="bbbb", message_id="run-b")
        for a, b in zip(life_a, life_b):
            t.process(_proto_event(a, run_id="run-a"))
            t.process(_proto_event(b, run_id="run-b"))
        log.close()
        streams = list(log._items)
        assert len(streams) == 2
        by_id = {s.message_id: s for s in streams}
        assert by_id["run-a"].output.text == "aaaa"
        assert by_id["run-b"].output.text == "bbbb"

    def test_text_deltas_accumulated_on_stream(self) -> None:
        t, log = _make_sync_transformer()
        for evt in _lifecycle(text="abcdef"):
            t.process(_proto_event(evt))
        log.close()
        (stream,) = list(log._items)
        assert "".join(stream._text_proj._deltas) == "abcdef"

    def test_stream_pushed_on_message_start_not_finish(self) -> None:
        # Consumer can see the stream before message-finish arrives.
        t, log = _make_sync_transformer()
        t.process(
            _proto_event(
                {"event": "message-start", "role": "ai", "message_id": "run-1"},
                run_id="run-1",
            )
        )
        assert len(log._items) == 1

    def test_node_metadata_set_on_stream(self) -> None:
        t, log = _make_sync_transformer()
        t.process(
            _proto_event(
                {"event": "message-start", "role": "ai", "message_id": "run-1"},
                run_id="run-1",
                node="my_llm",
            )
        )
        (stream,) = list(log._items)
        assert stream.node == "my_llm"


# ---------------------------------------------------------------------------
# Whole-message fallback
# ---------------------------------------------------------------------------


class TestWholeMessageFallback:
    def test_whole_ai_message_produces_complete_stream(self) -> None:
        t, log = _make_sync_transformer()
        t.process(_whole_msg("the full answer"))
        log.close()
        (stream,) = list(log._items)
        assert stream.done
        assert stream.output.text == "the full answer"

    def test_whole_message_has_full_lifecycle(self) -> None:
        t, log = _make_sync_transformer()
        t.process(_whole_msg("full"))
        log.close()
        (stream,) = list(log._items)
        assert [e["event"] for e in stream._events] == [
            "message-start",
            "content-block-start",
            "content-block-delta",
            "content-block-finish",
            "message-finish",
        ]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_non_messages_events_pass_through(self) -> None:
        t, _ = _make_sync_transformer()
        assert (
            t.process(
                {
                    "type": "event",
                    "method": "values",
                    "params": {"namespace": [], "timestamp": TS, "data": {"x": 1}},
                }
            )
            is True
        )

    def test_subgraph_namespace_dropped(self) -> None:
        t, log = _make_sync_transformer()
        t.process(
            {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": ["subgraph"],
                    "timestamp": TS,
                    "data": (
                        {"event": "message-start", "message_id": "run-x"},
                        {"run_id": "run-x"},
                    ),
                },
            }
        )
        log.close()
        assert list(log._items) == []

    def test_legacy_v1_chunks_ignored(self) -> None:
        # v1 AIMessageChunk tuples (from on_llm_new_token) are not streamed
        # into this projection; callers must migrate to stream_v2.
        t, log = _make_sync_transformer()
        t.process(_v1_chunk("hello"))
        t.process(_v1_chunk(" world", finish=True))
        log.close()
        assert list(log._items) == []


# ---------------------------------------------------------------------------
# Lifecycle: fail / finalize
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_fail_propagates_to_open_streams(self) -> None:
        t, log = _make_sync_transformer()
        t.process(
            _proto_event(
                {"event": "message-start", "message_id": "run-1"}, run_id="run-1"
            )
        )
        streams = list(log._items)
        err = RuntimeError("graph died")
        t.fail(err)
        assert t._by_run == {}
        assert streams[0]._error is err

    def test_finalize_clears_routing_state(self) -> None:
        t, _ = _make_sync_transformer()
        t.process(
            _proto_event(
                {"event": "message-start", "message_id": "run-1"}, run_id="run-1"
            )
        )
        assert "run-1" in t._by_run
        t.finalize()
        assert t._by_run == {}


# ---------------------------------------------------------------------------
# Async mode
# ---------------------------------------------------------------------------


class TestAsyncMode:
    def test_async_mode_creates_async_stream(self) -> None:
        t, log = _make_async_transformer()
        for evt in _lifecycle(text="async stream"):
            t.process(_proto_event(evt))
        assert isinstance(list(log._items)[0], AsyncChatModelStream)

    @pytest.mark.anyio
    async def test_text_projection_yields_deltas(self) -> None:
        t, log = _make_async_transformer()
        for evt in _lifecycle(text="hello world"):
            t.process(_proto_event(evt))
        (stream,) = list(log._items)
        assert isinstance(stream, AsyncChatModelStream)
        assert "".join([d async for d in stream.text]) == "hello world"

    @pytest.mark.anyio
    async def test_output_awaitable(self) -> None:
        t, log = _make_async_transformer()
        for evt in _lifecycle(text="async"):
            t.process(_proto_event(evt))
        (stream,) = list(log._items)
        assert (await stream.output).text == "async"


# ---------------------------------------------------------------------------
# GraphRunStream integration
# ---------------------------------------------------------------------------


class TestWireRequestMore:
    def test_bind_pump_called_on_wire(self) -> None:
        values_t = ValuesTransformer()
        messages_t = MessagesTransformer()
        mux = StreamMux([values_t, messages_t], is_async=False)

        assert messages_t._pump_fn is None
        run = GraphRunStream(iter([]), mux)
        assert messages_t._pump_fn is not None
        assert messages_t._pump_fn() is False
        assert run._exhausted

    def test_created_streams_have_request_more(self) -> None:
        values_t = ValuesTransformer()
        messages_t = MessagesTransformer()
        mux = StreamMux([values_t, messages_t], is_async=False)
        GraphRunStream(iter([]), mux)

        log: EventLog[ChatModelStream] = mux.extensions["messages"]
        log._subscribed = True
        for evt in _lifecycle():
            messages_t.process(_proto_event(evt))

        (stream,) = list(log._items)
        assert stream._request_more is messages_t._pump_fn


# ---------------------------------------------------------------------------
# End-to-end via StreamMux
# ---------------------------------------------------------------------------


class TestViaMux:
    def _make_mux(
        self,
    ) -> tuple[MessagesTransformer, StreamMux, EventLog[ChatModelStream]]:
        t = MessagesTransformer()
        v = ValuesTransformer()
        mux = StreamMux([v, t], is_async=False)
        t._bind_pump(lambda: False)
        log: EventLog[ChatModelStream] = mux.extensions["messages"]
        log._subscribed = True
        return t, mux, log

    def test_streaming_via_mux(self) -> None:
        t, mux, log = self._make_mux()
        for evt in _lifecycle(text="mux stream"):
            mux.push(_proto_event(evt))
        mux.close()
        (stream,) = list(log._items)
        assert stream.output.text == "mux stream"

    def test_whole_message_via_mux(self) -> None:
        t, mux, log = self._make_mux()
        mux.push(_whole_msg("result"))
        mux.close()
        (stream,) = list(log._items)
        assert stream.output.text == "result"

    @pytest.mark.anyio
    async def test_async_streaming_via_mux(self) -> None:
        t = MessagesTransformer()
        v = ValuesTransformer()
        mux = StreamMux([v, t], is_async=True)
        log: EventLog[ChatModelStream] = mux.extensions["messages"]
        log._subscribed = True

        for evt in _lifecycle(text="async mux"):
            await mux.apush(_proto_event(evt))

        (stream,) = list(log._items)
        assert (await stream.output).text == "async mux"
        await mux.aclose()


# ---------------------------------------------------------------------------
# End-to-end: graph → stream_v2 → run.messages (node calls stream_v2)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """stream_v2 path: node calls model.stream_v2() explicitly."""

    def test_node_calling_stream_v2_populates_messages(self) -> None:
        model = GenericFakeChatModel(messages=iter(["hello world"]))

        def call_model(state: MessagesState) -> dict[str, Any]:
            stream = model.stream_v2(state["messages"])
            return {"messages": stream.output}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        run = graph.stream_v2({"messages": "hi"})
        (stream,) = list(run.messages)
        assert isinstance(stream, ChatModelStream)
        assert stream.output.text == "hello world"

    def test_node_stream_v2_text_deltas_iterate(self) -> None:
        """Consumer can iterate `.text` on the streamed message in real time."""
        model = GenericFakeChatModel(messages=iter(["streamed answer"]))

        def call_model(state: MessagesState) -> dict[str, Any]:
            stream = model.stream_v2(state["messages"])
            return {"messages": stream.output}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        run = graph.stream_v2({"messages": "go"})
        (stream,) = list(run.messages)
        assert "".join(stream.text) == "streamed answer"

    def test_non_llm_message_returned_from_node(self) -> None:
        """Whole-message fallback: node returns a finalized AIMessage directly."""

        def return_message(state: MessagesState) -> dict[str, Any]:
            return {"messages": AIMessage(content="hardcoded", id="msg-abc")}

        graph = (
            StateGraph(MessagesState)
            .add_node("return_message", return_message)
            .add_edge(START, "return_message")
            .add_edge("return_message", END)
            .compile()
        )

        run = graph.stream_v2({"messages": "hi"})
        (stream,) = list(run.messages)
        assert stream.output.text == "hardcoded"

    @pytest.mark.anyio
    async def test_async_node_calling_astream_v2(self) -> None:
        model = GenericFakeChatModel(messages=iter(["async answer"]))

        async def call_model(state: MessagesState) -> dict[str, Any]:
            stream = await model.astream_v2(state["messages"])
            return {"messages": await stream}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        run = await graph.astream_v2({"messages": "hi"})
        streams = [s async for s in run.messages]
        assert len(streams) == 1
        assert isinstance(streams[0], AsyncChatModelStream)
        assert (await streams[0].output).text == "async answer"

    @pytest.mark.anyio
    async def test_nested_async_iteration_yields_text_deltas(self) -> None:
        """Inner stream.text drives the shared graph pump via the async pump binding."""
        import asyncio

        model = GenericFakeChatModel(messages=iter(["hello world"]))

        async def call_model(state: MessagesState) -> dict[str, Any]:
            stream = await model.astream_v2(state["messages"])
            return {"messages": await stream}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        run = await graph.astream_v2({"messages": "hi"})

        async def consume() -> list[str]:
            collected: list[str] = []
            async for stream in run.messages:
                async for delta in stream.text:
                    collected.append(delta)
            return collected

        assert "".join(await asyncio.wait_for(consume(), timeout=2.0)) == "hello world"


# ---------------------------------------------------------------------------
# End-to-end: graph → stream_v2 → run.messages (node calls invoke)
# ---------------------------------------------------------------------------


class TestEndToEndV2Invoke:
    """Auto-routing path: stream_v2 injects CONFIG_KEY_STREAM_MESSAGES_V2,
    causing BaseChatModel to drive the v2 protocol event generator even for
    model.invoke()."""

    def _graph(self, model):
        def call_model(state: MessagesState) -> dict[str, Any]:
            return {"messages": model.invoke(state["messages"])}

        return (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

    def test_invoke_populates_messages(self) -> None:
        run = self._graph(
            GenericFakeChatModel(messages=iter(["hello world"]))
        ).stream_v2({"messages": "hi"})
        (stream,) = list(run.messages)
        assert isinstance(stream, ChatModelStream)
        assert stream.output.text == "hello world"

    def test_invoke_emits_protocol_events(self) -> None:
        """Iterating the stream yields the full v2 lifecycle, not v1 chunks."""
        run = self._graph(
            GenericFakeChatModel(messages=iter(["streamed answer"]))
        ).stream_v2({"messages": "go"})
        (stream,) = list(run.messages)

        events = list(stream)
        event_types = [e.get("event") for e in events]
        assert "message-start" in event_types
        assert "content-block-start" in event_types
        assert "content-block-delta" in event_types
        assert "content-block-finish" in event_types
        assert "message-finish" in event_types
        # Sanity: every event is a dict carrying an "event" key — not an
        # AIMessageChunk tuple from the v1 path.
        for event in events:
            assert isinstance(event, dict)
            assert "event" in event
        # Typed projection still assembles the final text.
        assert stream.output.text == "streamed answer"

    def test_invoke_text_deltas_iterate(self) -> None:
        run = self._graph(
            GenericFakeChatModel(messages=iter(["delta streaming works"]))
        ).stream_v2({"messages": "hi"})
        (stream,) = list(run.messages)
        assert "".join(stream.text) == "delta streaming works"

    def test_invoke_two_nodes_two_streams(self) -> None:
        model_a = GenericFakeChatModel(messages=iter(["alpha"]))
        model_b = GenericFakeChatModel(messages=iter(["beta"]))

        def node_a(state: MessagesState) -> dict[str, Any]:
            return {"messages": model_a.invoke(state["messages"])}

        def node_b(state: MessagesState) -> dict[str, Any]:
            return {"messages": model_b.invoke(state["messages"])}

        graph = (
            StateGraph(MessagesState)
            .add_node("node_a", node_a)
            .add_node("node_b", node_b)
            .add_edge(START, "node_a")
            .add_edge("node_a", "node_b")
            .add_edge("node_b", END)
            .compile()
        )

        streams = list(graph.stream_v2({"messages": "hi"}).messages)
        assert len(streams) == 2
        assert {s.output.text for s in streams} == {"alpha", "beta"}

    def test_invoke_plus_constructed_message_two_streams(self) -> None:
        """Live-streamed node + constructed-message node → two ChatModelStreams."""
        model = GenericFakeChatModel(messages=iter(["live stream"]))

        def streaming_node(state: MessagesState) -> dict[str, Any]:
            return {"messages": model.invoke(state["messages"])}

        def constructed_node(state: MessagesState) -> dict[str, Any]:
            return {"messages": [AIMessage(content="hardcoded", id="constructed-1")]}

        graph = (
            StateGraph(MessagesState)
            .add_node("streaming_node", streaming_node)
            .add_node("constructed_node", constructed_node)
            .add_edge(START, "streaming_node")
            .add_edge("streaming_node", "constructed_node")
            .add_edge("constructed_node", END)
            .compile()
        )

        run = graph.stream_v2({"messages": "hi"})
        streams = list(run.messages)
        assert len(streams) == 2
        assert streams[0].node == "streaming_node"
        assert streams[0].output.text == "live stream"
        assert streams[1].node == "constructed_node"
        assert streams[1].output.text == "hardcoded"
        assert streams[1].message_id == "constructed-1"

    @pytest.mark.anyio
    async def test_ainvoke_populates_messages(self) -> None:
        model = GenericFakeChatModel(messages=iter(["async invoke"]))

        async def call_model(state: MessagesState) -> dict[str, Any]:
            return {"messages": await model.ainvoke(state["messages"])}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        run = await graph.astream_v2({"messages": "hi"})
        streams = [s async for s in run.messages]
        assert len(streams) == 1
        assert isinstance(streams[0], AsyncChatModelStream)
        assert (await streams[0].output).text == "async invoke"


# ---------------------------------------------------------------------------
# Regression: direct stream_mode="messages" must stay v1
# ---------------------------------------------------------------------------


class TestDirectMessagesModeStaysV1:
    def test_direct_graph_stream_messages_yields_ai_message_chunks(self) -> None:
        """graph.stream(stream_mode="messages") must not leak v2 event dicts —
        the v2 flag is only injected by stream_v2 / astream_v2."""
        model = GenericFakeChatModel(messages=iter(["legacy path"]))

        def call_model(state: MessagesState) -> dict[str, Any]:
            return {"messages": model.invoke(state["messages"])}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        parts = list(graph.stream({"messages": "hi"}, stream_mode="messages"))
        assert parts, "expected stream_mode='messages' to emit tuples"
        for payload, _metadata in parts:
            assert isinstance(payload, AIMessageChunk)
        assert (
            "".join(p[0].content for p in parts if isinstance(p[0].content, str))
            == "legacy path"
        )

    def test_nested_graph_stream_messages_stays_v1_under_outer_stream_v2(self) -> None:
        """An outer `stream_v2()` run must not flip an inner direct
        `stream_mode="messages"` call onto the v2 event protocol."""
        model = GenericFakeChatModel(messages=iter(["nested legacy path"]))

        def call_model(state: MessagesState) -> dict[str, Any]:
            return {"messages": model.invoke(state["messages"])}

        inner = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        class OuterState(TypedDict, total=False):
            saw_only_chunks: bool
            first_payload_type: str
            text: str

        def call_subgraph(state: OuterState, config: RunnableConfig) -> dict[str, Any]:
            parts = list(
                inner.stream(
                    {"messages": "hi"},
                    config,
                    stream_mode="messages",
                )
            )
            assert parts
            payloads = [payload for payload, _metadata in parts]
            return {
                "saw_only_chunks": all(
                    isinstance(payload, AIMessageChunk) for payload in payloads
                ),
                "first_payload_type": type(payloads[0]).__name__,
                "text": "".join(
                    payload.content
                    for payload in payloads
                    if isinstance(payload, AIMessageChunk)
                    and isinstance(payload.content, str)
                ),
            }

        outer = (
            StateGraph(OuterState)
            .add_node("call_subgraph", call_subgraph)
            .add_edge(START, "call_subgraph")
            .add_edge("call_subgraph", END)
            .compile()
        )

        result = outer.stream_v2({}).output

        assert result is not None
        assert result["saw_only_chunks"] is True
        assert result["first_payload_type"] == "AIMessageChunk"
        assert result["text"] == "nested legacy path"


# ---------------------------------------------------------------------------
# StreamMessagesHandlerV2 unit
# ---------------------------------------------------------------------------


class TestStreamMessagesHandlerV2Unit:
    def test_on_llm_new_token_is_noop(self) -> None:
        """v2 handler must not emit v1 chunks even when on_llm_new_token fires."""
        from uuid import uuid4

        from langchain_core.outputs import ChatGenerationChunk

        from langgraph.pregel._messages import StreamMessagesHandlerV2

        emitted: list[Any] = []
        handler = StreamMessagesHandlerV2(emitted.append, subgraphs=False)
        run_id = uuid4()
        handler.metadata[run_id] = ((), {"langgraph_node": "x"})

        handler.on_llm_new_token(
            "hello",
            chunk=ChatGenerationChunk(message=AIMessageChunk(content="hello")),
            run_id=run_id,
        )

        assert emitted == []

    def test_on_llm_end_dedupes_when_final_message_id_differs(self) -> None:
        """A streamed v2 message should not be emitted again from the final
        AIMessage fallback when its final id does not match `message-start`."""
        from uuid import uuid4

        from langchain_core.outputs import ChatGeneration, LLMResult

        from langgraph.pregel._messages import StreamMessagesHandlerV2

        emitted: list[Any] = []
        handler = StreamMessagesHandlerV2(emitted.append, subgraphs=False)
        run_id = uuid4()
        handler.metadata[run_id] = ((), {"langgraph_node": "x"})

        handler.on_stream_event(
            {"event": "message-start", "message_id": "stream-msg-1"},
            run_id=run_id,
        )
        handler.on_llm_end(
            LLMResult(
                generations=[
                    [
                        ChatGeneration(
                            message=AIMessage(content="hello", id="final-msg-1")
                        )
                    ]
                ]
            ),
            run_id=run_id,
        )

        assert len(emitted) == 1
