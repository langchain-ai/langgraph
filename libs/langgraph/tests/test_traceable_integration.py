"""Tests for @traceable integration with LangGraph nodes.

These tests verify that LangGraph properly reads `__traceable_config__` from
decorated node functions and applies the configuration to tracing.

Note: Some tests manually set `__traceable_config__` to work independently of
the langsmith package version installed. When langsmith exposes this attribute
via @traceable decorator, the manual setting can be removed.
"""

from __future__ import annotations

from typing import Any

import pytest
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from tests.fake_tracer import FakeTracer

pytestmark = pytest.mark.anyio


class SimpleState(TypedDict):
    value: str


def _set_traceable_config(
    func: Any,
    *,
    process_inputs: Any = None,
    process_outputs: Any = None,
    enabled: bool | None = None,
) -> Any:
    """Helper to set __traceable_config__ on a function.

    This simulates what @traceable decorator should do when it exposes
    the __traceable_config__ attribute.

    Args:
        func: The function to decorate.
        process_inputs: Optional function to filter traced inputs.
        process_outputs: Optional function to filter traced outputs.
        enabled: Whether tracing is enabled. None means honor external context.
    """
    setattr(
        func,
        "__traceable_config__",
        {
            "process_inputs": process_inputs,
            "process_outputs": process_outputs,
            "enabled": enabled,
            "wrapped": func,
        },
    )
    return func


def test_traceable_config_process_inputs():
    """Test that __traceable_config__ with process_inputs filters inputs in trace."""
    tracer = FakeTracer()

    def filter_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        """Filter sensitive data from inputs."""
        return {"value": "[REDACTED]"}

    def my_node(state: SimpleState) -> SimpleState:
        return {"value": f"processed_{state['value']}"}

    # Manually set traceable config (simulating @traceable decorator)
    _set_traceable_config(my_node, process_inputs=filter_inputs)

    builder = StateGraph(SimpleState)
    builder.add_node("my_node", my_node)
    builder.add_edge("__start__", "my_node")
    graph = builder.compile()

    result = graph.invoke({"value": "secret_data"}, {"callbacks": [tracer]})

    assert result == {"value": "processed_secret_data"}

    # Find the node run (should be the RunnableSeq for my_node)
    runs = tracer.flattened_runs()
    node_runs = [r for r in runs if r.name == "my_node"]

    assert len(node_runs) == 1, f"Expected 1 my_node run, got {len(node_runs)}"

    # Verify inputs were filtered through process_inputs
    node_run = node_runs[0]
    assert node_run.inputs == {"value": "[REDACTED]"}, (
        f"Expected inputs to be filtered, got {node_run.inputs}"
    )


def test_traceable_config_process_outputs():
    """Test that __traceable_config__ with process_outputs filters outputs in trace."""
    tracer = FakeTracer()

    def filter_outputs(outputs: Any) -> Any:
        """Filter sensitive data from outputs."""
        if isinstance(outputs, dict) and "value" in outputs:
            return {"value": "[OUTPUT_REDACTED]"}
        return outputs

    def my_node(state: SimpleState) -> SimpleState:
        return {"value": "secret_result"}

    # Manually set traceable config
    _set_traceable_config(my_node, process_outputs=filter_outputs)

    builder = StateGraph(SimpleState)
    builder.add_node("my_node", my_node)
    builder.add_edge("__start__", "my_node")
    graph = builder.compile()

    result = graph.invoke({"value": "input"}, {"callbacks": [tracer]})

    # Result should be unfiltered (actual execution)
    assert result == {"value": "secret_result"}

    # Find the node run
    runs = tracer.flattened_runs()
    node_runs = [r for r in runs if r.name == "my_node"]

    assert len(node_runs) == 1, f"Expected 1 my_node run, got {len(node_runs)}"

    # Verify outputs were filtered through process_outputs
    node_run = node_runs[0]
    assert node_run.outputs == {"value": "[OUTPUT_REDACTED]"}, (
        f"Expected outputs to be filtered, got {node_run.outputs}"
    )


def test_traceable_config_enabled_false():
    """Test that __traceable_config__ with enabled=False skips trace creation."""
    tracer = FakeTracer()

    def hidden_node(state: SimpleState) -> SimpleState:
        return {"value": f"hidden_{state['value']}"}

    # Manually set traceable config with enabled=False
    _set_traceable_config(hidden_node, enabled=False)

    def visible_node(state: SimpleState) -> SimpleState:
        return {"value": f"visible_{state['value']}"}

    builder = StateGraph(SimpleState)
    builder.add_node("hidden_node", hidden_node)
    builder.add_node("visible_node", visible_node)
    builder.add_edge("__start__", "hidden_node")
    builder.add_edge("hidden_node", "visible_node")
    graph = builder.compile()

    result = graph.invoke({"value": "test"}, {"callbacks": [tracer]})

    assert result == {"value": "visible_hidden_test"}

    # Verify runs
    runs = tracer.flattened_runs()
    run_names = [r.name for r in runs]

    # visible_node should be traced
    assert "visible_node" in run_names, f"Expected visible_node in {run_names}"

    # hidden_node should NOT be traced (enabled=False)
    assert "hidden_node" not in run_names, (
        f"hidden_node should not be traced but found in {run_names}"
    )


def test_regular_node_tracing_unchanged():
    """Test that regular nodes (without __traceable_config__) trace normally."""
    tracer = FakeTracer()

    def regular_node(state: SimpleState) -> SimpleState:
        return {"value": f"regular_{state['value']}"}

    builder = StateGraph(SimpleState)
    builder.add_node("regular_node", regular_node)
    builder.add_edge("__start__", "regular_node")
    graph = builder.compile()

    result = graph.invoke({"value": "test"}, {"callbacks": [tracer]})

    assert result == {"value": "regular_test"}

    # Verify the node is traced
    runs = tracer.flattened_runs()
    node_runs = [r for r in runs if r.name == "regular_node"]

    assert len(node_runs) == 1, f"Expected 1 regular_node run, got {len(node_runs)}"

    # Verify inputs and outputs are captured normally (not filtered)
    node_run = node_runs[0]
    assert "value" in node_run.inputs
    assert node_run.inputs["value"] == "test"


def test_traceable_config_single_run_no_duplication():
    """Test that node with __traceable_config__ creates exactly one run."""
    tracer = FakeTracer()

    def my_node(state: SimpleState) -> SimpleState:
        return {"value": f"result_{state['value']}"}

    # Set traceable config (simulating @traceable decorator)
    _set_traceable_config(my_node)

    builder = StateGraph(SimpleState)
    builder.add_node("my_node", my_node)
    builder.add_edge("__start__", "my_node")
    graph = builder.compile()

    result = graph.invoke({"value": "test"}, {"callbacks": [tracer]})

    assert result == {"value": "result_test"}

    # Count runs with name my_node - should be exactly 1
    runs = tracer.flattened_runs()
    my_node_runs = [r for r in runs if r.name == "my_node"]

    assert len(my_node_runs) == 1, (
        f"Expected exactly 1 my_node run (no duplication), got {len(my_node_runs)}. "
        f"All run names: {[r.name for r in runs]}"
    )


async def test_traceable_config_async():
    """Test that async nodes with __traceable_config__ work correctly."""
    tracer = FakeTracer()

    def filter_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"value": "[ASYNC_REDACTED]"}

    async def async_node(state: SimpleState) -> SimpleState:
        return {"value": f"async_{state['value']}"}

    # Set traceable config (simulating @traceable decorator)
    _set_traceable_config(async_node, process_inputs=filter_inputs)

    builder = StateGraph(SimpleState)
    builder.add_node("async_node", async_node)
    builder.add_edge("__start__", "async_node")
    graph = builder.compile()

    result = await graph.ainvoke({"value": "secret"}, {"callbacks": [tracer]})

    assert result == {"value": "async_secret"}

    # Verify inputs were filtered
    runs = tracer.flattened_runs()
    node_runs = [r for r in runs if r.name == "async_node"]

    assert len(node_runs) == 1
    assert node_runs[0].inputs == {"value": "[ASYNC_REDACTED]"}


def test_traceable_config_both_inputs_and_outputs():
    """Test __traceable_config__ with both process_inputs and process_outputs."""
    tracer = FakeTracer()

    def filter_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        return {"value": "[INPUT_FILTERED]"}

    def filter_outputs(outputs: Any) -> Any:
        if isinstance(outputs, dict):
            return {"value": "[OUTPUT_FILTERED]"}
        return outputs

    def my_node(state: SimpleState) -> SimpleState:
        return {"value": f"result_{state['value']}"}

    # Set traceable config with both filters
    _set_traceable_config(
        my_node, process_inputs=filter_inputs, process_outputs=filter_outputs
    )

    builder = StateGraph(SimpleState)
    builder.add_node("my_node", my_node)
    builder.add_edge("__start__", "my_node")
    graph = builder.compile()

    result = graph.invoke({"value": "secret"}, {"callbacks": [tracer]})

    # Actual result should be unfiltered
    assert result == {"value": "result_secret"}

    # Trace should have filtered inputs and outputs
    runs = tracer.flattened_runs()
    node_runs = [r for r in runs if r.name == "my_node"]

    assert len(node_runs) == 1
    assert node_runs[0].inputs == {"value": "[INPUT_FILTERED]"}
    assert node_runs[0].outputs == {"value": "[OUTPUT_FILTERED]"}


def test_traceable_config_process_inputs_error_handling():
    """Test that errors in process_inputs don't leak PII."""
    tracer = FakeTracer()

    def bad_filter(inputs: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("filter crashed!")

    def my_node(state: SimpleState) -> SimpleState:
        return {"value": f"processed_{state['value']}"}

    _set_traceable_config(my_node, process_inputs=bad_filter)

    builder = StateGraph(SimpleState)
    builder.add_node("my_node", my_node)
    builder.add_edge("__start__", "my_node")
    graph = builder.compile()

    # Graph should still execute successfully
    result = graph.invoke({"value": "secret_pii_data"}, {"callbacks": [tracer]})
    assert result == {"value": "processed_secret_pii_data"}

    # Trace should show error placeholder, NOT the raw PII data
    runs = tracer.flattened_runs()
    node_runs = [r for r in runs if r.name == "my_node"]

    assert len(node_runs) == 1
    # Should NOT contain "secret_pii_data"
    assert node_runs[0].inputs == {"error": "<trace_inputs processing failed>"}


def test_traceable_config_process_outputs_error_handling():
    """Test that errors in process_outputs don't leak PII."""
    tracer = FakeTracer()

    def bad_filter(outputs: Any) -> Any:
        raise RuntimeError("output filter crashed!")

    def my_node(state: SimpleState) -> SimpleState:
        return {"value": "secret_output_pii"}

    _set_traceable_config(my_node, process_outputs=bad_filter)

    builder = StateGraph(SimpleState)
    builder.add_node("my_node", my_node)
    builder.add_edge("__start__", "my_node")
    graph = builder.compile()

    # Graph should still execute successfully
    result = graph.invoke({"value": "input"}, {"callbacks": [tracer]})
    assert result == {"value": "secret_output_pii"}

    # Trace should show error placeholder, NOT the raw PII data
    runs = tracer.flattened_runs()
    node_runs = [r for r in runs if r.name == "my_node"]

    assert len(node_runs) == 1
    # Should NOT contain "secret_output_pii"
    assert node_runs[0].outputs == {"error": "<trace_outputs processing failed>"}


def test_traceable_config_enabled_none_honors_external_context():
    """Test that enabled=None (default) honors external tracing context."""
    tracer = FakeTracer()

    def my_node(state: SimpleState) -> SimpleState:
        return {"value": f"result_{state['value']}"}

    # Set traceable config with enabled=None (the default)
    _set_traceable_config(my_node, enabled=None)

    builder = StateGraph(SimpleState)
    builder.add_node("my_node", my_node)
    builder.add_edge("__start__", "my_node")
    graph = builder.compile()

    result = graph.invoke({"value": "test"}, {"callbacks": [tracer]})
    assert result == {"value": "result_test"}

    # With enabled=None, tracing should proceed normally (honor external context)
    runs = tracer.flattened_runs()
    node_runs = [r for r in runs if r.name == "my_node"]

    # Node should be traced since we provided a tracer callback
    assert len(node_runs) == 1


async def test_traceable_context_propagates_to_children():
    """Traceable node should become the parent context for downstream nodes."""
    tracer = FakeTracer()

    async def traceable_node(state: SimpleState) -> dict:
        return await subgraph.ainvoke({"value": f"traceable_{state['value']}"})

    def child_node(state: SimpleState) -> SimpleState:
        return {"value": f"child_{state['value']}"}

    _set_traceable_config(traceable_node, process_inputs=lambda _: {"value": "[MASK]"})
    subgraph_builder = StateGraph(SimpleState)
    subgraph_builder.add_node("child_node", child_node)
    subgraph_builder.add_edge("__start__", "child_node")
    subgraph = subgraph_builder.compile(name="Subgraph")

    builder = StateGraph(SimpleState)
    builder.add_node("traceable_node", traceable_node)
    builder.add_edge("__start__", "traceable_node")
    graph = builder.compile()

    result = await graph.ainvoke({"value": "secret"}, {"callbacks": [tracer]})
    assert result == {"value": "child_traceable_secret"}

    runs = tracer.flattened_runs()
    traceable_runs = [r for r in runs if r.name == "traceable_node"]
    child_runs = [r for r in runs if r.name == "child_node"]
    subgraph_runs = [r for r in runs if r.name == "Subgraph"]

    assert len(traceable_runs) == 1
    assert len(child_runs) == 1
    assert len(subgraph_runs) == 1

    traceable_run = traceable_runs[0]
    child_run = child_runs[0]
    subgraph_run = subgraph_runs[0]

    assert child_run.trace_id == traceable_run.trace_id
    assert child_run.dotted_order.startswith(traceable_run.dotted_order)
    assert child_run.parent_run_id == subgraph_run.id
    assert subgraph_run.parent_run_id == traceable_run.id
