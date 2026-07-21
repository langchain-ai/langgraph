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
