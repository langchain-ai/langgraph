"""End-to-end tests for `TracePolicy` input processing on node trace runs."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict

from langgraph._internal._runnable import (
    ASYNCIO_ACCEPTS_CONTEXT,
    RunnableCallable,
    RunnableSeq,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import TracePolicy
from tests.fake_tracer import FakeTracer, Run


class State(TypedDict):
    value: int


def _node_run(tracer: FakeTracer, name: str) -> Run:
    return next(r for r in tracer.flattened_runs() if r.name == name)


def _incr(state: State) -> State:
    return {"value": state["value"] + 1}


def test_trace_policy_transforms_recorded_inputs() -> None:
    seen: dict[str, Any] = {}

    def process_inputs(inp: Any) -> Any:
        seen["inputs"] = inp
        return {"scrubbed_in": True}

    graph = (
        StateGraph(State)
        .add_node("n", _incr, trace_policy=TracePolicy(process_inputs=process_inputs))
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile()
    )

    tracer = FakeTracer()
    # the real graph output is unaffected by the trace policy
    assert graph.invoke({"value": 1}, {"callbacks": [tracer]}) == {"value": 2}

    run = _node_run(tracer, "n")
    # the recorded input is transformed; the output is recorded as-is
    assert run.inputs == {"scrubbed_in": True}
    assert run.outputs == {"value": 2}
    # process_inputs observed the real, untransformed input
    assert seen["inputs"] == {"value": 1}


def test_trace_policy_transforms_recorded_outputs() -> None:
    seen: dict[str, Any] = {}

    def process_outputs(out: Any) -> Any:
        seen["outputs"] = out
        return {"scrubbed_out": True}

    graph = (
        StateGraph(State)
        .add_node("n", _incr, trace_policy=TracePolicy(process_outputs=process_outputs))
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile()
    )

    tracer = FakeTracer()
    # the real graph output is unaffected by the trace policy
    assert graph.invoke({"value": 1}, {"callbacks": [tracer]}) == {"value": 2}

    run = _node_run(tracer, "n")
    # the recorded output is transformed; the input is recorded as-is
    assert run.inputs == {"value": 1}
    assert run.outputs == {"scrubbed_out": True}
    # process_outputs observed the real, untransformed output
    assert seen["outputs"] == {"value": 2}


def test_trace_policy_none_records_real_payloads() -> None:
    graph = (
        StateGraph(State)
        .add_node("n", _incr)
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile()
    )

    tracer = FakeTracer()
    assert graph.invoke({"value": 1}, {"callbacks": [tracer]}) == {"value": 2}

    run = _node_run(tracer, "n")
    assert run.inputs == {"value": 1}
    assert run.outputs == {"value": 2}


def test_trace_policy_processor_error_safe_without_callbacks() -> None:
    def boom(_inp: Any) -> Any:
        raise RuntimeError("processor failed")

    graph = (
        StateGraph(State)
        .add_node("n", _incr, trace_policy=TracePolicy(process_inputs=boom))
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile()
    )

    # no callbacks: the processor still runs but is fail-open, so execution is unaffected
    assert graph.invoke({"value": 1}) == {"value": 2}


def test_trace_policy_processor_error_does_not_break_execution() -> None:
    def boom(_inp: Any) -> Any:
        raise RuntimeError("processor failed")

    graph = (
        StateGraph(State)
        .add_node("n", _incr, trace_policy=TracePolicy(process_inputs=boom))
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile()
    )

    tracer = FakeTracer()
    # a raising processor must not abort the node; the untransformed input is recorded
    assert graph.invoke({"value": 1}, {"callbacks": [tracer]}) == {"value": 2}
    assert _node_run(tracer, "n").inputs == {"value": 1}


@pytest.mark.anyio
async def test_trace_policy_transforms_recorded_inputs_async() -> None:
    graph = (
        StateGraph(State)
        .add_node(
            "n",
            _incr,
            trace_policy=TracePolicy(process_inputs=lambda _: {"scrubbed_in": True}),
        )
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile()
    )

    tracer = FakeTracer()
    assert await graph.ainvoke({"value": 5}, {"callbacks": [tracer]}) == {"value": 6}

    run = _node_run(tracer, "n")
    assert run.inputs == {"scrubbed_in": True}
    assert run.outputs == {"value": 6}


_RECORDING_POLICY = TracePolicy(
    process_inputs=lambda _: {"recorded": "inputs"},
    process_outputs=lambda _: {"recorded": "outputs"},
)


def _chat_graph() -> CompiledStateGraph:
    model = FakeMessagesListChatModel(
        responses=[AIMessage("model reply", id="ai-model")]
    )

    def chat(state: MessagesState) -> dict[str, Any]:
        reply = model.invoke(state["messages"])
        return {
            "messages": [
                *state["messages"],
                reply,
                AIMessage("from node", id="ai-node"),
            ]
        }

    return (
        StateGraph(MessagesState)
        .add_node("chat", chat, trace_policy=_RECORDING_POLICY)
        .add_edge(START, "chat")
        .compile()
    )


def _assert_scrubbed_node_run(tracer: FakeTracer, *, model_is_child: bool) -> None:
    run = _node_run(tracer, "chat")
    assert run.inputs == {"recorded": "inputs"}
    assert run.outputs == {"recorded": "outputs"}
    if model_is_child:
        assert [c.name for c in run.child_runs] == ["FakeMessagesListChatModel"]


def test_trace_policy_scrubs_tracer_not_messages_stream() -> None:
    tracer = FakeTracer()
    streamed = [
        message.content
        for message, _ in _chat_graph().stream(
            {"messages": [HumanMessage("hello", id="h-1")]},
            {"callbacks": [tracer]},
            stream_mode="messages",
        )
    ]

    assert streamed == ["model reply", "from node"]
    _assert_scrubbed_node_run(tracer, model_is_child=True)


@pytest.mark.anyio
async def test_trace_policy_scrubs_tracer_not_messages_stream_async() -> None:
    tracer = FakeTracer()
    streamed = [
        message.content
        async for message, _ in _chat_graph().astream(
            {"messages": [HumanMessage("hello", id="h-1")]},
            {"callbacks": [tracer]},
            stream_mode="messages",
        )
    ]

    assert streamed == ["model reply", "from node"]
    _assert_scrubbed_node_run(tracer, model_is_child=ASYNCIO_ACCEPTS_CONTEXT)


@pytest.mark.anyio
async def test_trace_policy_scrubs_tracer_not_astream_events() -> None:
    tracer = FakeTracer()
    chat_events = [
        event
        async for event in _chat_graph().astream_events(
            {"messages": [HumanMessage("hello", id="h-1")]},
            {"callbacks": [tracer]},
            version="v2",
        )
        if event["name"] == "chat"
        and event["event"] in ("on_chain_start", "on_chain_end")
    ]

    assert [e["event"] for e in chat_events] == ["on_chain_start", "on_chain_end"]
    start, end = (e["data"] for e in chat_events)
    assert [m.content for m in start["input"]["messages"]] == ["hello"]
    assert [m.content for m in end["output"]["messages"]] == [
        "hello",
        "model reply",
        "from node",
    ]
    _assert_scrubbed_node_run(tracer, model_is_child=ASYNCIO_ACCEPTS_CONTEXT)


class _ExporterWithTapMethods(BaseCallbackHandler):
    def __init__(self) -> None:
        self.node_payloads: list[Any] = []
        self.node_run_ids: set[UUID] = set()

    def _record(self, run_id: UUID, payload: Any) -> None:
        if run_id in self.node_run_ids:
            self.node_payloads.append(payload)

    def tap_output_iter(self, run_id: UUID, output: Iterator[Any]) -> Iterator[Any]:
        for chunk in output:
            self._record(run_id, chunk)
            yield chunk

    async def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[Any]
    ) -> AsyncIterator[Any]:
        async for chunk in output:
            self._record(run_id, chunk)
            yield chunk

    def on_chain_start(
        self, serialized: Any, inputs: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        if kwargs.get("name") == "n":
            self.node_run_ids.add(run_id)
            self._record(run_id, inputs)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._record(run_id, outputs)


@pytest.mark.anyio
@pytest.mark.parametrize("via_events", [False, True], ids=["ainvoke", "astream_events"])
async def test_trace_policy_scrubs_exporter_that_only_has_tap_methods(
    via_events: bool,
) -> None:
    graph = (
        StateGraph(State)
        .add_node("n", _incr, trace_policy=_RECORDING_POLICY)
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile()
    )
    exporter = _ExporterWithTapMethods()

    if via_events:
        async for _ in graph.astream_events(
            {"value": 1}, {"callbacks": [exporter]}, version="v2"
        ):
            pass
    else:
        assert await graph.ainvoke({"value": 1}, {"callbacks": [exporter]}) == {
            "value": 2
        }
    assert exporter.node_payloads == [{"recorded": "inputs"}, {"recorded": "outputs"}]


@pytest.mark.anyio
async def test_trace_policy_cancel_at_node_end_still_closes_tracer_run() -> None:
    def cancel_own_task(value: Any) -> Any:
        asyncio.current_task().cancel()
        return value

    tracer = FakeTracer()
    seq = RunnableSeq(
        RunnableCallable(lambda x: x, trace=False),
        RunnableCallable(cancel_own_task, trace=False),
        record_outputs=_RECORDING_POLICY.process_outputs,
    )
    with pytest.raises(asyncio.CancelledError):
        await asyncio.create_task(seq.ainvoke({}, {"callbacks": [tracer]}))
    for _ in range(100):
        if tracer.runs:
            break
        await asyncio.sleep(0.01)

    assert [r.outputs for r in tracer.runs] == [{"recorded": "outputs"}]
