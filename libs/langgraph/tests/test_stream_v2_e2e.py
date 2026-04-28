"""End-to-end tests exercising all stream_v2 projections together.

Each test builds a realistic graph (subgraphs, LLM calls, custom writers,
interrupts) and verifies that every projection — values, messages, lifecycle,
subgraphs, raw events, output, interleave — produces correct, consistent
results through a single stream_v2 / astream_v2 run.
"""

from __future__ import annotations

import operator
import sys
from typing import Annotated, Any

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.language_models.chat_model_stream import (
    AsyncChatModelStream,
    ChatModelStream,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.stream import StreamChannel, StreamTransformer
from langgraph.stream._types import ProtocolEvent
from langgraph.types import StreamWriter, interrupt

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


# ---------------------------------------------------------------------------
# State and graph builders
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def _make_nested_graph():
    """Build a two-level graph with pure state transforms.

    Structure:
        outer:
            router_node  (state transform)
            inner_graph  (compiled subgraph)

        inner_graph:
            process_node  (state transform)
    """

    def process_node(state: AgentState) -> dict[str, Any]:
        return {"value": state["value"] + "_processed", "items": ["processed"]}

    inner_builder: StateGraph = StateGraph(AgentState, input_schema=AgentState)
    inner_builder.add_node("process_node", process_node)
    inner_builder.add_edge(START, "process_node")
    inner_builder.add_edge("process_node", END)
    inner_graph = inner_builder.compile()

    def router_node(state: AgentState) -> dict[str, Any]:
        return {"value": state["value"] + "_routed", "items": ["routed"]}

    outer_builder: StateGraph = StateGraph(AgentState, input_schema=AgentState)
    outer_builder.add_node("router", router_node)
    outer_builder.add_node("inner", inner_graph)
    outer_builder.add_edge(START, "router")
    outer_builder.add_edge("router", "inner")
    outer_builder.add_edge("inner", END)
    return outer_builder.compile()


def _make_messages_graph():
    """Flat graph with an LLM call for messages projection testing."""
    model = GenericFakeChatModel(messages=iter(["hello world"]))

    def call_model(state: MessagesState) -> dict[str, Any]:
        return {"messages": model.invoke(state["messages"])}

    return (
        StateGraph(MessagesState)
        .add_node("call_model", call_model)
        .add_edge(START, "call_model")
        .add_edge("call_model", END)
        .compile()
    )


def _make_messages_subgraph():
    """Outer graph with a MessagesState subgraph that returns an AIMessage.

    Uses the whole-message fallback path (node returns AIMessage directly)
    to exercise messages through a subgraph boundary.
    """

    def return_message(state: MessagesState) -> dict[str, Any]:
        return {"messages": AIMessage(content="from subgraph", id="sub-msg-1")}

    inner = (
        StateGraph(MessagesState)
        .add_node("return_message", return_message)
        .add_edge(START, "return_message")
        .add_edge("return_message", END)
        .compile()
    )

    class OuterState(TypedDict):
        messages: Annotated[list[Any], operator.add]
        done: bool

    def pre_node(state: OuterState) -> dict[str, Any]:
        return {"done": False}

    return (
        StateGraph(OuterState)
        .add_node("pre", pre_node)
        .add_node("inner", inner)
        .add_edge(START, "pre")
        .add_edge("pre", "inner")
        .add_edge("inner", END)
        .compile()
    )


def _make_custom_writer_graph():
    """Graph where a node emits custom stream events via StreamWriter."""

    def writer_node(state: AgentState, *, writer: StreamWriter) -> dict[str, Any]:
        writer({"step": "start", "detail": "beginning work"})
        writer({"step": "middle", "detail": "processing"})
        writer({"step": "end", "detail": "done"})
        return {"value": state["value"] + "_custom", "items": ["custom"]}

    builder = StateGraph(AgentState)
    builder.add_node("writer_node", writer_node)
    builder.add_edge(START, "writer_node")
    builder.add_edge("writer_node", END)
    return builder.compile()


def _make_interrupt_graph():
    """Graph that interrupts after the first node."""

    def step_one(state: AgentState) -> dict[str, Any]:
        return {"value": state["value"] + "_step1", "items": ["step1"]}

    def step_two(state: AgentState) -> dict[str, Any]:
        answer = interrupt("need approval")
        return {"value": state["value"] + f"_{answer}", "items": ["step2"]}

    builder = StateGraph(AgentState)
    builder.add_node("step_one", step_one)
    builder.add_node("step_two", step_two)
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", END)
    return builder.compile(checkpointer=InMemorySaver())


def _make_error_subgraph():
    """Graph with a subgraph that raises."""

    def failing_node(state: AgentState) -> dict[str, Any]:
        raise ValueError("subgraph explosion")

    inner_builder = StateGraph(AgentState)
    inner_builder.add_node("fail", failing_node)
    inner_builder.add_edge(START, "fail")
    inner_builder.add_edge("fail", END)
    inner = inner_builder.compile()

    outer_builder = StateGraph(AgentState)
    outer_builder.add_node("inner", inner)
    outer_builder.add_edge(START, "inner")
    outer_builder.add_edge("inner", END)
    return outer_builder.compile()


class _CustomPassthroughTransformer(StreamTransformer):
    required_stream_modes = ("custom",)

    def init(self) -> dict[str, Any]:
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        return True


class _CounterTransformer(StreamTransformer):
    """Custom transformer that counts values events via a StreamChannel."""

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._channel: StreamChannel[int] = StreamChannel("counter")
        self._count = 0

    def init(self) -> dict[str, Any]:
        return {"counter": self._channel}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "values":
            self._count += 1
            self._channel.push(self._count)
        return True


# ---------------------------------------------------------------------------
# Sync end-to-end: all projections on nested graph
# ---------------------------------------------------------------------------


class TestStreamV2E2ESync:
    def test_all_projections_nested_graph(self) -> None:
        """Run a nested graph through stream_v2 and verify values + lifecycle."""
        graph = _make_nested_graph()
        run = graph.stream_v2({"value": "x", "items": []})

        values_snapshots: list[dict[str, Any]] = []
        lifecycle_events: list[dict[str, Any]] = []
        for name, item in run.interleave("values", "lifecycle"):
            if name == "values":
                values_snapshots.append(item)
            elif name == "lifecycle":
                lifecycle_events.append(item)

        assert len(values_snapshots) >= 1
        final = values_snapshots[-1]
        assert "routed" in final["items"]
        assert "processed" in final["items"]
        assert "_routed" in final["value"]
        assert "_processed" in final["value"]

        assert len(lifecycle_events) >= 2
        started = [e for e in lifecycle_events if e["event"] == "started"]
        completed = [e for e in lifecycle_events if e["event"] == "completed"]
        assert len(started) >= 1
        assert len(completed) >= 1

    def test_subgraph_handles_with_drill_down(self) -> None:
        """Subgraph handles yield and support values drill-down."""
        graph = _make_nested_graph()
        run = graph.stream_v2({"value": "x", "items": []})

        handles = []
        for handle in run.subgraphs:
            child_values = list(handle.values)
            handles.append(
                {
                    "path": handle.path,
                    "graph_name": handle.graph_name,
                    "values_count": len(child_values),
                }
            )

        assert len(handles) >= 1
        assert handles[0]["values_count"] >= 1

        output = run.output
        assert output is not None
        assert "_routed" in output["value"]
        assert "_processed" in output["value"]

    def test_raw_events_have_monotonic_seq(self) -> None:
        """Raw protocol events have monotonically increasing seq numbers."""
        graph = _make_nested_graph()
        run = graph.stream_v2({"value": "x", "items": []})
        events = list(run)
        assert len(events) > 0

        seqs = [e["seq"] for e in events]
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], f"seq not monotonic at {i}: {seqs}"

        for event in events:
            assert event["type"] == "event"
            assert "method" in event
            assert isinstance(event["params"]["timestamp"], int)

    def test_output_matches_final_values_snapshot(self) -> None:
        """output property returns the same state as the last values snapshot."""
        run1 = _make_nested_graph().stream_v2({"value": "x", "items": []})
        snapshots = list(run1.values)
        final_via_values = snapshots[-1]

        run2 = _make_nested_graph().stream_v2({"value": "x", "items": []})
        final_via_output = run2.output

        assert final_via_values == final_via_output

    def test_context_manager_and_abort(self) -> None:
        """Context manager calls abort, marking the stream exhausted."""
        graph = _make_nested_graph()
        with graph.stream_v2({"value": "x", "items": []}) as run:
            first_val = next(iter(run.values))
            assert isinstance(first_val, dict)
        assert run._exhausted is True

    def test_extensions_has_all_native_keys(self) -> None:
        """Extensions dict exposes all native projection keys."""
        graph = _make_nested_graph()
        run = graph.stream_v2({"value": "x", "items": []})
        _ = run.output

        assert "values" in run.extensions
        assert "messages" in run.extensions
        assert "lifecycle" in run.extensions
        assert "subgraphs" in run.extensions
        assert run.values is run.extensions["values"]
        assert run.messages is run.extensions["messages"]
        assert run.lifecycle is run.extensions["lifecycle"]
        assert run.subgraphs is run.extensions["subgraphs"]


# ---------------------------------------------------------------------------
# Sync: messages projection
# ---------------------------------------------------------------------------


class TestStreamV2E2EMessages:
    def test_messages_projection_from_invoke(self) -> None:
        """Messages projection captures LLM calls via model.invoke() auto-routing."""
        graph = _make_messages_graph()
        run = graph.stream_v2({"messages": "hi"})
        streams = list(run.messages)

        assert len(streams) >= 1
        for stream in streams:
            assert isinstance(stream, ChatModelStream)
        assert streams[0].output.text == "hello world"

    def test_messages_text_deltas(self) -> None:
        """Text deltas from the messages projection concatenate correctly."""
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
        assert "".join(stream.text) == "streamed answer"

    def test_messages_from_whole_ai_message(self) -> None:
        """Node returning AIMessage directly produces a complete stream."""

        def return_msg(state: MessagesState) -> dict[str, Any]:
            return {"messages": AIMessage(content="hardcoded", id="msg-1")}

        graph = (
            StateGraph(MessagesState)
            .add_node("return_msg", return_msg)
            .add_edge(START, "return_msg")
            .add_edge("return_msg", END)
            .compile()
        )

        run = graph.stream_v2({"messages": "hi"})
        (stream,) = list(run.messages)
        assert stream.output.text == "hardcoded"
        assert stream.message_id == "msg-1"

    def test_root_messages_only_shows_root_scope(self) -> None:
        """Root messages projection doesn't surface subgraph-scoped messages."""
        graph = _make_messages_subgraph()
        run = graph.stream_v2({"messages": ["hi"], "done": False})
        root_streams = list(run.messages)
        # The message is emitted inside the subgraph, so the root
        # messages projection (scoped to root namespace) doesn't see it.
        assert root_streams == []

    def test_subgraph_handle_messages_drill_down(self) -> None:
        """Drilling into subgraph handle's messages surfaces subgraph messages."""
        graph = _make_messages_subgraph()
        run = graph.stream_v2({"messages": ["hi"], "done": False})

        found_messages = False
        for handle in run.subgraphs:
            child_messages = list(handle.messages)
            if child_messages:
                found_messages = True
                assert isinstance(child_messages[0], ChatModelStream)
                assert child_messages[0].output.text == "from subgraph"
        assert found_messages


# ---------------------------------------------------------------------------
# Sync: custom stream writer + custom transformer
# ---------------------------------------------------------------------------


class TestStreamV2E2ECustom:
    def test_custom_events_with_passthrough_transformer(self) -> None:
        """Custom StreamWriter events appear on the main log when a
        transformer declares the custom mode."""
        graph = _make_custom_writer_graph()
        run = graph.stream_v2(
            {"value": "x", "items": []},
            transformers=[_CustomPassthroughTransformer],
        )
        events = list(run)
        custom = [e for e in events if e["method"] == "custom"]
        assert len(custom) == 3
        steps = [e["params"]["data"]["step"] for e in custom]
        assert steps == ["start", "middle", "end"]

    def test_custom_events_suppressed_without_transformer(self) -> None:
        """Without a custom-mode transformer, custom events don't flow."""
        graph = _make_custom_writer_graph()
        run = graph.stream_v2({"value": "x", "items": []})
        events = list(run)
        custom = [e for e in events if e["method"] == "custom"]
        assert custom == []

    def test_custom_transformer_with_stream_channel(self) -> None:
        """A custom transformer with a StreamChannel produces extension data."""
        graph = _make_nested_graph()
        run = graph.stream_v2(
            {"value": "x", "items": []},
            transformers=[_CounterTransformer],
        )

        assert "counter" in run.extensions
        counter_iter = iter(run.extensions["counter"])
        _ = run.output
        counts = list(counter_iter)
        assert len(counts) >= 1
        assert all(isinstance(c, int) for c in counts)
        assert counts == sorted(counts)

    def test_custom_channel_events_on_main_log(self) -> None:
        """StreamChannel auto-forward injects custom:<name> events into the main log."""
        graph = _make_nested_graph()
        run = graph.stream_v2(
            {"value": "x", "items": []},
            transformers=[_CounterTransformer],
        )
        events = list(run)
        counter_events = [e for e in events if e["method"] == "custom:counter"]
        assert len(counter_events) >= 1
        assert all(isinstance(e["params"]["data"], int) for e in counter_events)


# ---------------------------------------------------------------------------
# Sync: interrupt handling
# ---------------------------------------------------------------------------


class TestStreamV2E2EInterrupt:
    def test_interrupt_sets_flags_and_surfaces_interrupts(self) -> None:
        """Interrupted run has correct flags and interrupt payloads."""
        graph = _make_interrupt_graph()
        config: dict[str, Any] = {"configurable": {"thread_id": "int-1"}}
        run = graph.stream_v2({"value": "x", "items": []}, config)

        output = run.output
        assert output is not None
        assert run.interrupted is True
        assert len(run.interrupts) > 0
        assert output["items"] == ["step1"]
        assert "_step1" in output["value"]

    def test_interrupt_values_snapshot_has_partial_state(self) -> None:
        """Values snapshots captured before the interrupt reflect partial state."""
        graph = _make_interrupt_graph()
        config: dict[str, Any] = {"configurable": {"thread_id": "int-2"}}
        run = graph.stream_v2({"value": "x", "items": []}, config)

        snapshots = list(run.values)
        assert len(snapshots) >= 1
        last = snapshots[-1]
        assert "step1" in last["items"]


# ---------------------------------------------------------------------------
# Sync: error propagation
# ---------------------------------------------------------------------------


class TestStreamV2E2EErrors:
    def test_subgraph_error_propagates_through_output(self) -> None:
        """Error in a subgraph propagates through output."""
        graph = _make_error_subgraph()
        run = graph.stream_v2({"value": "x", "items": []})

        with pytest.raises(ValueError, match="subgraph explosion"):
            _ = run.output

    def test_subgraph_error_propagates_through_raw_events(self) -> None:
        graph = _make_error_subgraph()
        run = graph.stream_v2({"value": "x", "items": []})

        with pytest.raises(ValueError, match="subgraph explosion"):
            list(run)

    def test_error_subgraph_handle_status(self) -> None:
        """Subgraph handle surfaces the error status."""
        graph = _make_error_subgraph()
        run = graph.stream_v2({"value": "x", "items": []})

        handle = next(iter(run.subgraphs))
        with pytest.raises(RuntimeError, match="subgraph explosion"):
            _ = handle.output
        assert handle.status == "failed"
        assert handle.error == "subgraph explosion"


# ---------------------------------------------------------------------------
# Async end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@NEEDS_CONTEXTVARS
class TestStreamV2E2EAsync:
    async def test_all_projections_async(self) -> None:
        """Async run exercises values projection."""
        graph = _make_nested_graph()
        run = await graph.astream_v2({"value": "x", "items": []})

        values_snapshots = [s async for s in run.values]
        assert len(values_snapshots) >= 1
        final = values_snapshots[-1]
        assert "_routed" in final["value"]
        assert "_processed" in final["value"]

    async def test_async_output(self) -> None:
        """Async output returns the final state."""
        graph = _make_nested_graph()
        run = await graph.astream_v2({"value": "x", "items": []})
        output = await run.output()
        assert output is not None
        assert output["value"] == "x_routed_processed"
        assert "routed" in output["items"]
        assert "processed" in output["items"]

    async def test_async_raw_events(self) -> None:
        """Async raw event iteration yields well-formed ProtocolEvents."""
        graph = _make_nested_graph()
        run = await graph.astream_v2({"value": "x", "items": []})
        events = [e async for e in run]
        assert len(events) > 0
        seqs = [e["seq"] for e in events]
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1]

    async def test_async_messages_projection(self) -> None:
        """Async messages projection captures LLM streams."""
        model = GenericFakeChatModel(messages=iter(["async answer"]))

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
        assert len(streams) >= 1
        for s in streams:
            assert isinstance(s, AsyncChatModelStream)
        assert (await streams[0].output).text == "async answer"

    async def test_async_interrupt(self) -> None:
        """Async interrupted run has correct flags."""
        graph = _make_interrupt_graph()
        config: dict[str, Any] = {"configurable": {"thread_id": "async-int-1"}}
        run = await graph.astream_v2({"value": "x", "items": []}, config)

        output = await run.output()
        assert output is not None
        assert await run.interrupted() is True
        assert len(await run.interrupts()) > 0

    async def test_async_error_propagation(self) -> None:
        """Async error from subgraph propagates through output."""
        graph = _make_error_subgraph()
        run = await graph.astream_v2({"value": "x", "items": []})
        with pytest.raises(ValueError, match="subgraph explosion"):
            await run.output()

    async def test_async_context_manager(self) -> None:
        """Async context manager calls abort on exit."""
        graph = _make_nested_graph()
        run = await graph.astream_v2({"value": "x", "items": []})
        async with run:
            _ = await anext(aiter(run.values))
        assert run._exhausted is True

    async def test_async_extensions_present(self) -> None:
        """Async run has all native extensions."""
        graph = _make_nested_graph()
        run = await graph.astream_v2({"value": "x", "items": []})
        _ = await run.output()
        assert "values" in run.extensions
        assert "messages" in run.extensions
        assert "lifecycle" in run.extensions
        assert "subgraphs" in run.extensions

    async def test_async_custom_transformer(self) -> None:
        """Async custom transformer with StreamChannel works."""
        graph = _make_nested_graph()
        run = await graph.astream_v2(
            {"value": "x", "items": []},
            transformers=[_CounterTransformer],
        )
        assert "counter" in run.extensions
        counter_cursor = aiter(run.extensions["counter"])
        _ = await run.output()
        counts = [c async for c in counter_cursor]
        assert len(counts) >= 1
        assert counts == sorted(counts)


# ---------------------------------------------------------------------------
# Sync: combined projections stress test
# ---------------------------------------------------------------------------


class TestStreamV2E2ECombined:
    def test_interleave_all_native_projections(self) -> None:
        """Interleave values + messages + lifecycle without deadlock."""
        graph = _make_nested_graph()
        run = graph.stream_v2({"value": "x", "items": []})

        seen_names: set[str] = set()
        for name, _item in run.interleave("values", "messages", "lifecycle"):
            seen_names.add(name)

        assert "values" in seen_names
        assert "lifecycle" in seen_names

    def test_multiple_custom_transformers(self) -> None:
        """Multiple custom transformers can coexist."""

        class TagTransformer(StreamTransformer):
            def __init__(self, scope: tuple[str, ...] = ()) -> None:
                super().__init__(scope)
                self._channel: StreamChannel[str] = StreamChannel("tags")

            def init(self) -> dict[str, Any]:
                return {"tags": self._channel}

            def process(self, event: ProtocolEvent) -> bool:
                if event["method"] == "values":
                    self._channel.push(
                        f"tag:{event['params']['data'].get('value', '')}"
                    )
                return True

        graph = _make_nested_graph()
        run = graph.stream_v2(
            {"value": "x", "items": []},
            transformers=[_CounterTransformer, TagTransformer],
        )

        assert "counter" in run.extensions
        assert "tags" in run.extensions

        counter_iter = iter(run.extensions["counter"])
        tags_iter = iter(run.extensions["tags"])
        _ = run.output
        counts = list(counter_iter)
        tags = list(tags_iter)

        assert len(counts) >= 1
        assert len(tags) >= 1
        assert all(t.startswith("tag:") for t in tags)

    def test_two_sibling_subgraphs_both_discoverable(self) -> None:
        """Two sequential subgraph invocations produce two handles."""

        class _S(TypedDict):
            items: Annotated[list[str], operator.add]

        def _item(name: str):
            def node(state: _S) -> dict[str, Any]:
                return {"items": [name]}

            return node

        inner_a = (
            StateGraph(_S)
            .add_node("add_a", _item("a"))
            .add_edge(START, "add_a")
            .add_edge("add_a", END)
            .compile()
        )
        inner_b = (
            StateGraph(_S)
            .add_node("add_b", _item("b"))
            .add_edge(START, "add_b")
            .add_edge("add_b", END)
            .compile()
        )

        outer = (
            StateGraph(_S)
            .add_node("sub_a", inner_a)
            .add_node("sub_b", inner_b)
            .add_edge(START, "sub_a")
            .add_edge("sub_a", "sub_b")
            .add_edge("sub_b", END)
            .compile()
        )

        run = outer.stream_v2({"items": []})
        handles = []
        for handle in run.subgraphs:
            list(handle.values)
            handles.append(handle)

        assert len(handles) == 2
        names = [h.graph_name for h in handles]
        assert "sub_a" in names
        assert "sub_b" in names
        assert all(h.status == "completed" for h in handles)

        output = run.output
        assert output is not None
        assert set(output["items"]) == {"a", "b"}

    def test_lifecycle_matches_subgraph_handles(self) -> None:
        """Lifecycle events and subgraph handles agree on discovered subgraphs."""
        run1 = _make_nested_graph().stream_v2({"value": "x", "items": []})
        handle_paths: list[tuple[str, ...]] = []
        for handle in run1.subgraphs:
            list(handle.values)
            handle_paths.append(handle.path)

        run2 = _make_nested_graph().stream_v2({"value": "x", "items": []})
        lifecycle = list(run2.lifecycle)

        started_ns = [
            tuple(e["namespace"]) for e in lifecycle if e["event"] == "started"
        ]
        # Handle paths use format "graph_name:call_id", lifecycle namespaces
        # use the same format. Both should have the same graph_name prefix.
        handle_prefixes = {p[0].split(":")[0] for p in handle_paths}
        lifecycle_prefixes = {ns[0].split(":")[0] for ns in started_ns}
        assert handle_prefixes == lifecycle_prefixes

    def test_values_plus_messages_plus_custom(self) -> None:
        """Values, messages, and a custom transformer all produce data in one run."""
        model = GenericFakeChatModel(messages=iter(["combined test"]))

        def call_model(state: MessagesState) -> dict[str, Any]:
            return {"messages": model.invoke(state["messages"])}

        graph = (
            StateGraph(MessagesState)
            .add_node("call_model", call_model)
            .add_edge(START, "call_model")
            .add_edge("call_model", END)
            .compile()
        )

        run = graph.stream_v2(
            {"messages": "hi"},
            transformers=[_CounterTransformer],
        )

        counter_iter = iter(run.extensions["counter"])
        values_iter = iter(run.values)
        messages_iter = iter(run.messages)

        values = list(values_iter)
        messages = list(messages_iter)
        counts = list(counter_iter)

        assert len(values) >= 1
        assert len(messages) >= 1
        assert len(counts) >= 1
        assert messages[0].output.text == "combined test"
