"""End-to-end tests for `TracePolicy` input processing on node trace runs."""

from typing import Any

import pytest
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
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


def test_trace_policy_omit_outputs_keeps_messages_stream() -> None:
    """process_outputs must scrub traces without starving stream_mode='messages' (#9090)."""
    from typing import Annotated

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph.message import add_messages
    from langgraph.types import omit_payload
    from typing_extensions import TypedDict

    class MsgState(TypedDict):
        messages: Annotated[list, add_messages]

    def node(state: MsgState) -> dict:
        return {"messages": [AIMessage("from node", id="ai-1")]}

    graph = (
        StateGraph(MsgState)
        .add_node("node", node, trace_policy=TracePolicy(process_outputs=omit_payload))
        .add_edge(START, "node")
        .add_edge("node", END)
        .compile()
    )

    tracer = FakeTracer()
    seen = [
        msg.content
        for msg, _meta in graph.stream(
            {"messages": [HumanMessage("hello", id="h-1")]},
            {"callbacks": [tracer]},
            stream_mode="messages",
        )
    ]
    assert seen == ["from node"]
    run = _node_run(tracer, "node")
    assert run.outputs == {}


def test_trace_policy_omit_inputs_keeps_message_dedupe() -> None:
    """process_inputs must scrub traces without re-emitting pass-through inputs (#9090)."""
    from typing import Annotated

    from langchain_core.messages import HumanMessage
    from langgraph.graph.message import add_messages
    from langgraph.types import omit_payload
    from typing_extensions import TypedDict

    class MsgState(TypedDict):
        messages: Annotated[list, add_messages]

    def node(state: MsgState) -> dict:
        return {"messages": state["messages"]}

    graph = (
        StateGraph(MsgState)
        .add_node("node", node, trace_policy=TracePolicy(process_inputs=omit_payload))
        .add_edge(START, "node")
        .add_edge("node", END)
        .compile()
    )

    tracer = FakeTracer()
    seen = [
        msg.content
        for msg, _meta in graph.stream(
            {"messages": [HumanMessage("hello", id="h-1")]},
            {"callbacks": [tracer]},
            stream_mode="messages",
        )
    ]
    assert seen == []
    run = _node_run(tracer, "node")
    assert run.inputs == {}
