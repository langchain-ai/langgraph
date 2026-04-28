"""Tests for the MessagesTransformer content-block upgrade (B2).

Verifies that `MessagesTransformer` routes protocol events (emitted by
`stream_v2` via `on_stream_event`) to `ChatModelStream` objects keyed by
run_id, and replays whole `AIMessage` payloads via `message_to_events`.
Legacy v1 `AIMessageChunk` tuples (from `on_llm_new_token`) are ignored.
"""

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
    metadata: dict[str, Any] = {"langgraph_node": node, "run_id": run_id}
    return {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": [],
            "timestamp": TS,
            "data": (event, metadata),
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
    rm: dict[str, Any] = {}
    if finish:
        rm["finish_reason"] = "stop"
    message = AIMessageChunk(content=text, id=msg_id, response_metadata=rm)
    metadata: dict[str, Any] = {"langgraph_node": node}
    return {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": [],
            "timestamp": TS,
            "data": (message, metadata),
        },
    }


def _whole_msg(
    text: str,
    msg_id: str = "msg-10",
    *,
    node: str = "node",
) -> dict[str, Any]:
    """Build a messages ProtocolEvent carrying a completed AIMessage."""
    message = AIMessage(content=text, id=msg_id)
    metadata: dict[str, Any] = {"langgraph_node": node}
    return {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": [],
            "timestamp": TS,
            "data": (message, metadata),
        },
    }


def _make_sync_transformer() -> tuple[MessagesTransformer, EventLog[ChatModelStream]]:
    t = MessagesTransformer()
    proj = t.init()
    log: EventLog[ChatModelStream] = proj["messages"]
    log._bind(is_async=False)
    # Production subscribes via `iter(log)` from the graph consumer — do that
    # up front so `push` during `process` isn't a no-op. Tests read buffered
    # items via `log._items` directly rather than re-iterating.
    log._subscribed = True
    t._bind_pump(lambda: False)
    return t, log


def _make_async_transformer() -> tuple[MessagesTransformer, EventLog[ChatModelStream]]:
    t = MessagesTransformer()
    proj = t.init()
    log: EventLog[ChatModelStream] = proj["messages"]
    log._bind(is_async=True)
    log._subscribed = True
    return t, log


# Standard lifecycle events for one streaming LLM call.
def _lifecycle(
    *,
    text: str = "hello world",
    message_id: str = "run-1",
) -> list[dict[str, Any]]:
    """Produce a valid protocol event lifecycle: start, delta, finish, end."""
    # Split text into two deltas to exercise delta accumulation.
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


# ---------------------------------------------------------------------------
# Primary path: protocol event routing
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
        # Stream is in the log immediately.
        log.close()
        streams = list(log._items)
        assert len(streams) == 1
        assert isinstance(streams[0], ChatModelStream)
        assert streams[0].message_id == "run-1"

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
        """Orphan delta events (no preceding message-start) are dropped silently."""
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
        """Two interleaved LLM calls each produce their own stream."""
        t, log = _make_sync_transformer()
        # Interleave events from two different run_ids.
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
        deltas = list(stream._text_proj._deltas)
        assert "".join(deltas) == "abcdef"

    def test_stream_pushed_on_message_start_not_finish(self) -> None:
        """Consumer can see the stream before it finishes."""
        t, log = _make_sync_transformer()
        t.process(
            _proto_event(
                {"event": "message-start", "role": "ai", "message_id": "run-1"},
                run_id="run-1",
            )
        )
        # The log has the stream immediately — even though message-finish
        # hasn't arrived yet.
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
        (stream,) = [*log._items]
        assert stream.node == "my_llm"


# ---------------------------------------------------------------------------
# Non-streaming (whole AIMessage) fallback
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
        event_types = [e["event"] for e in stream._events]
        assert event_types == [
            "message-start",
            "content-block-start",
            "content-block-delta",
            "content-block-finish",
            "message-finish",
        ]


# ---------------------------------------------------------------------------
# Legacy v1 chunks are ignored (users must migrate to stream_v2)
# ---------------------------------------------------------------------------


class TestLegacyChunksIgnored:
    def test_aimessage_chunk_tuple_is_dropped(self) -> None:
        t, log = _make_sync_transformer()
        t.process(_v1_chunk("hello"))
        t.process(_v1_chunk(" world", finish=True))
        log.close()
        assert list(log._items) == []


# ---------------------------------------------------------------------------
# Filtering behaviors
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_non_messages_events_pass_through(self) -> None:
        t, _ = _make_sync_transformer()
        values_event = {
            "type": "event",
            "method": "values",
            "params": {"namespace": [], "timestamp": TS, "data": {"x": 1}},
        }
        assert t.process(values_event) is True

    def test_subgraph_namespace_dropped(self) -> None:
        """Root MessagesTransformer (via the mux) ignores non-root events."""
        from langgraph.stream._mux import StreamMux

        mux = StreamMux([MessagesTransformer()], is_async=False)
        t = mux.transformer_by_key("messages")
        assert isinstance(t, MessagesTransformer)
        t._log._subscribed = True
        t._bind_pump(lambda: False)

        mux.push(
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
        t._log.close()
        assert list(t._log._items) == []


# ---------------------------------------------------------------------------
# Lifecycle: finalize / fail
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_fail_propagates_to_open_streams(self) -> None:
        t, log = _make_sync_transformer()
        t.process(
            _proto_event(
                {"event": "message-start", "message_id": "run-1"},
                run_id="run-1",
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
                {"event": "message-start", "message_id": "run-1"},
                run_id="run-1",
            )
        )
        assert "run-1" in t._by_run
        t.finalize()
        assert t._by_run == {}


# ---------------------------------------------------------------------------
# Async mode (AsyncChatModelStream)
# ---------------------------------------------------------------------------


class TestAsyncMode:
    def test_async_mode_creates_async_stream(self) -> None:
        t, log = _make_async_transformer()
        for evt in _lifecycle(text="async stream"):
            t.process(_proto_event(evt))
        streams = list(log._items)
        assert len(streams) == 1
        assert isinstance(streams[0], AsyncChatModelStream)

    @pytest.mark.anyio
    async def test_async_text_projection_yields_deltas(self) -> None:
        t, log = _make_async_transformer()
        for evt in _lifecycle(text="hello world"):
            t.process(_proto_event(evt))
        (stream,) = list(log._items)
        assert isinstance(stream, AsyncChatModelStream)
        collected = []
        async for delta in stream.text:
            collected.append(delta)
        assert "".join(collected) == "hello world"

    @pytest.mark.anyio
    async def test_async_output_awaitable(self) -> None:
        t, log = _make_async_transformer()
        for evt in _lifecycle(text="async"):
            t.process(_proto_event(evt))
        (stream,) = list(log._items)
        msg = await stream.output
        assert msg.text == "async"


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
        # After wire, the transformer's pump callback is set.
        assert messages_t._pump_fn is not None
        # And calling it invokes GraphRunStream._pump_next (drains an empty
        # graph_iter, returns False).
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
        # Pump was threaded through: the stream's _request_more points at
        # the same callable the transformer was bound with.
        assert stream._request_more is messages_t._pump_fn


# ---------------------------------------------------------------------------
# End-to-end via StreamMux
# ---------------------------------------------------------------------------


class TestViaMux:
    def test_streaming_via_mux(self) -> None:
        t = MessagesTransformer()
        v = ValuesTransformer()
        mux = StreamMux([v, t], is_async=False)
        t._bind_pump(lambda: False)
        log: EventLog[ChatModelStream] = mux.extensions["messages"]
        # Simulate a consumer subscribing (as `run.messages` iteration would).
        log._subscribed = True

        for evt in _lifecycle(text="mux stream"):
            mux.push(_proto_event(evt))
        mux.close()

        (stream,) = list(log._items)
        assert stream.output.text == "mux stream"

    def test_whole_message_via_mux(self) -> None:
        t = MessagesTransformer()
        v = ValuesTransformer()
        mux = StreamMux([v, t], is_async=False)
        t._bind_pump(lambda: False)
        log: EventLog[ChatModelStream] = mux.extensions["messages"]
        log._subscribed = True

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

        streams = list(log._items)
        assert len(streams) == 1
        msg = await streams[0].output
        assert msg.text == "async mux"
        await mux.aclose()


# ---------------------------------------------------------------------------
# End-to-end: full graph → stream_v2 → run.messages
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Prove the full pipeline works when a node calls `model.stream_v2()`.

    These tests exercise the path that the new messages projection is
    designed for: a user node invokes `stream_v2` on a chat model,
    `on_stream_event` fires on `StreamMessagesHandler`, the handler
    forwards to the mux, and the transformer routes events into a
    `ChatModelStream` exposed on `run.messages`.

    Nothing in Pregel calls `stream_v2` automatically yet; the planned
    `graph.stream_v2()` API (B4) and the `create_react_agent`
    integration (C2) will wire that up. Until then, populating the
    messages projection is opt-in at the node level.
    """

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
        streams = list(run.messages)

        assert len(streams) == 1
        assert isinstance(streams[0], ChatModelStream)
        assert streams[0].output.text == "hello world"

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

        # Pull the stream handle out, then iterate its text deltas.
        (stream,) = list(run.messages)
        text = "".join(stream.text)
        assert text == "streamed answer"

    def test_non_llm_message_returned_from_node(self) -> None:
        """Node returns a finalized AIMessage directly — whole-message fallback."""

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
        streams = list(run.messages)

        assert len(streams) == 1
        assert streams[0].output.text == "hardcoded"

    @pytest.mark.anyio
    async def test_async_node_calling_astream_v2(self) -> None:
        model = GenericFakeChatModel(messages=iter(["async answer"]))

        async def call_model(state: MessagesState) -> dict[str, Any]:
            stream = await model.astream_v2(state["messages"])
            msg = await stream
            return {"messages": msg}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        run = await graph.astream_v2({"messages": "hi"})

        streams = []
        async for stream in run.messages:
            streams.append(stream)

        assert len(streams) == 1
        assert isinstance(streams[0], AsyncChatModelStream)
        msg = await streams[0].output
        assert msg.text == "async answer"

    @pytest.mark.anyio
    async def test_nested_async_iteration_yields_text_deltas(self) -> None:
        """Iterate `stream.text` inside `async for stream in run.messages`.

        The inner `stream.text` cursor drives the shared graph pump via
        `AsyncProjection._arequest_more`, wired by
        `MessagesTransformer._bind_apump` and
        `AsyncGraphRunStream._wire_arequest_more`.
        """
        import asyncio

        model = GenericFakeChatModel(messages=iter(["hello world"]))

        async def call_model(state: MessagesState) -> dict[str, Any]:
            stream = await model.astream_v2(state["messages"])
            msg = await stream
            return {"messages": msg}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        run = await graph.astream_v2({"messages": "hi"})

        async def consume_nested() -> list[str]:
            collected: list[str] = []
            async for stream in run.messages:
                async for delta in stream.text:
                    collected.append(delta)
            return collected

        deltas = await asyncio.wait_for(consume_nested(), timeout=2.0)
        assert "".join(deltas) == "hello world"


class TestEndToEndV2Invoke:
    """Nodes call `model.invoke()`; `stream_v2` routes through v2.

    Exercises the auto-routing path added in
    `feat(core): route invoke through v2 event path for
    _V2StreamingCallbackHandler`: `stream_v2` injects
    `CONFIG_KEY_STREAM_MESSAGES_V2` into the config, pregel attaches
    `StreamMessagesHandlerV2`, `BaseChatModel._should_stream_v2` sees the
    v2 marker and drives the protocol event generator, and
    `on_stream_event` forwards each event onto the messages channel.
    """

    def test_invoke_with_v2_marker_populates_messages(self) -> None:
        """Node calling `model.invoke()` produces one ChatModelStream with v2 events."""
        model = GenericFakeChatModel(messages=iter(["hello world"]))

        def call_model(state: MessagesState) -> dict[str, Any]:
            return {"messages": model.invoke(state["messages"])}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        run = graph.stream_v2({"messages": "hi"})
        streams = list(run.messages)

        assert len(streams) == 1, (
            "Expected exactly one ChatModelStream — the streamed invoke and "
            "the node's return of the same AIMessage must dedupe."
        )
        stream = streams[0]
        assert isinstance(stream, ChatModelStream)
        assert stream.output.text == "hello world"

    def test_invoke_v2_emits_protocol_events(self) -> None:
        """Iterating the stream yields the full v2 lifecycle (not v1 chunks)."""
        model = GenericFakeChatModel(messages=iter(["streamed answer"]))

        def call_model(state: MessagesState) -> dict[str, Any]:
            return {"messages": model.invoke(state["messages"])}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        run = graph.stream_v2({"messages": "go"})
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

    def test_invoke_text_deltas_iterate_live(self) -> None:
        """`.text` projection yields deltas in order."""
        model = GenericFakeChatModel(messages=iter(["delta streaming works"]))

        def call_model(state: MessagesState) -> dict[str, Any]:
            return {"messages": model.invoke(state["messages"])}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        run = graph.stream_v2({"messages": "hi"})
        (stream,) = list(run.messages)

        assembled = "".join(stream.text)
        assert assembled == "delta streaming works"

    def test_invoke_dedupe_survives_multi_node_graph(self) -> None:
        """Two model-invoking nodes produce exactly two streams, each once."""
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

        run = graph.stream_v2({"messages": "hi"})
        streams = list(run.messages)

        assert len(streams) == 2
        contents = {s.output.text for s in streams}
        assert contents == {"alpha", "beta"}

    def test_invoke_plus_constructed_message_two_streams(self) -> None:
        """A v2-streamed node + a node that returns a constructed AIMessage
        produces two ChatModelStreams — one from the live event lifecycle,
        one synthesized from the constructed message via `message_to_events`.
        """
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
    async def test_ainvoke_with_v2_marker_populates_messages(self) -> None:
        """Async mirror: `model.ainvoke()` + `astream_v2`."""
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

        streams = []
        async for stream in run.messages:
            streams.append(stream)

        assert len(streams) == 1
        assert isinstance(streams[0], AsyncChatModelStream)
        msg = await streams[0].output
        assert msg.text == "async invoke"


class TestDirectMessagesModeStaysV1:
    """Regression guard: direct `graph.stream(stream_mode="messages")`
    (no `stream_v2`) must keep the v1 `(AIMessageChunk, metadata)`
    tuple shape. The v2 flag is only injected by `stream_v2` / `astream_v2`.
    """

    def test_direct_graph_stream_messages_yields_ai_message_chunks(self) -> None:
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
        # Should have at least one streamed chunk; each part is
        # (AIMessageChunk, metadata) — not a v2 event dict.
        assert parts, "expected stream_mode='messages' to emit tuples"
        for part in parts:
            payload, _metadata = part
            assert isinstance(payload, AIMessageChunk), (
                "direct graph.stream(stream_mode='messages') leaked v2 "
                "event dicts — stream_v2 flag bled through."
            )
        assembled = "".join(
            p[0].content for p in parts if isinstance(p[0].content, str)
        )
        assert assembled == "legacy path"


class TestStreamMessagesHandlerV2Unit:
    """Unit tests on the handler class itself."""

    def test_on_llm_new_token_is_noop(self) -> None:
        """v2 handler must not emit v1 chunks even if `on_llm_new_token` fires
        (e.g. from a node calling `model.stream()` directly on a v2-flagged run).
        """
        from uuid import uuid4

        from langchain_core.outputs import ChatGenerationChunk

        from langgraph.pregel._messages import StreamMessagesHandlerV2

        emitted: list[Any] = []
        handler = StreamMessagesHandlerV2(emitted.append, subgraphs=False)
        run_id = uuid4()
        # Register a fake run so `self.metadata.get(run_id)` would succeed for
        # other callbacks — this makes sure the no-op is unconditional, not a
        # side effect of missing metadata.
        handler.metadata[run_id] = ((), {"langgraph_node": "x"})

        handler.on_llm_new_token(
            "hello",
            chunk=ChatGenerationChunk(message=AIMessageChunk(content="hello")),
            run_id=run_id,
        )

        assert emitted == [], (
            "StreamMessagesHandlerV2.on_llm_new_token must not push to the "
            "messages stream — it's the v2 marker's guarantee."
        )
