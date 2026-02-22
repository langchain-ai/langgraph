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

from langgraph._internal._runnable import ASYNCIO_ACCEPTS_CONTEXT
from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph
from tests.any_str import AnyStr
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
            "__unwrapped__": func,
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
    """Test that enabled=False skips LangSmith but custom handlers still receive events.

    Note: enabled=False filters out LangChainTracer (LangSmith) specifically,
    but custom callback handlers like FakeTracer still receive events.
    """
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

    # Both nodes should be traced to FakeTracer (a custom handler)
    # enabled=False only filters LangChainTracer, not custom handlers
    assert "visible_node" in run_names, f"Expected visible_node in {run_names}"
    assert "hidden_node" in run_names, (
        f"hidden_node should be traced to custom handlers, "
        f"enabled=False only filters LangSmith. Got: {run_names}"
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


@pytest.mark.skipif(
    not ASYNCIO_ACCEPTS_CONTEXT,
    reason="Requires Python 3.11+ for async context propagation",
)
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


# =============================================================================
# Functional API Tests (@entrypoint and @task)
# =============================================================================


def test_entrypoint_traceable_config_process_inputs():
    """Test that @entrypoint honors __traceable_config__ with process_inputs."""
    tracer = FakeTracer()

    def filter_inputs(inputs: Any) -> dict[str, Any]:
        return {"input": "[ENTRYPOINT_INPUT_REDACTED]"}

    def my_entrypoint(value: str) -> str:
        return f"processed_{value}"

    _set_traceable_config(my_entrypoint, process_inputs=filter_inputs)

    workflow = entrypoint()(my_entrypoint)

    result = workflow.invoke("secret_data", {"callbacks": [tracer]})

    assert result == "processed_secret_data"

    runs = tracer.flattened_runs()
    entrypoint_runs = [r for r in runs if r.name == "my_entrypoint"]

    assert len(entrypoint_runs) == 1, (
        f"Expected 1 my_entrypoint run, got {len(entrypoint_runs)}. "
        f"All runs: {[r.name for r in runs]}"
    )

    # Verify inputs were filtered
    assert entrypoint_runs[0].inputs == {"input": "[ENTRYPOINT_INPUT_REDACTED]"}, (
        f"Expected inputs to be filtered, got {entrypoint_runs[0].inputs}"
    )


def test_entrypoint_traceable_config_process_outputs():
    """Test that @entrypoint honors __traceable_config__ with process_outputs."""
    tracer = FakeTracer()

    def filter_outputs(outputs: Any) -> Any:
        return {"output": "[ENTRYPOINT_OUTPUT_REDACTED]"}

    def my_entrypoint(value: str) -> str:
        return f"secret_result_{value}"

    _set_traceable_config(my_entrypoint, process_outputs=filter_outputs)

    workflow = entrypoint()(my_entrypoint)

    result = workflow.invoke("input", {"callbacks": [tracer]})

    # Actual result should be unfiltered
    assert result == "secret_result_input"

    runs = tracer.flattened_runs()
    entrypoint_runs = [r for r in runs if r.name == "my_entrypoint"]

    assert len(entrypoint_runs) == 1

    # Verify outputs were filtered in trace
    assert entrypoint_runs[0].outputs == {"output": "[ENTRYPOINT_OUTPUT_REDACTED]"}, (
        f"Expected outputs to be filtered, got {entrypoint_runs[0].outputs}"
    )


def test_entrypoint_traceable_config_enabled_false():
    """Test that @entrypoint with enabled=False still fires custom callbacks.

    Note: enabled=False filters out LangChainTracer (LangSmith) specifically,
    but custom callback handlers like FakeTracer still receive events.
    """
    tracer = FakeTracer()

    def hidden_entrypoint(value: str) -> str:
        return f"hidden_{value}"

    _set_traceable_config(hidden_entrypoint, enabled=False)

    workflow = entrypoint()(hidden_entrypoint)

    result = workflow.invoke("test", {"callbacks": [tracer]})

    assert result == "hidden_test"

    runs = tracer.flattened_runs()
    run_names = [r.name for r in runs]

    # The entrypoint should still be traced to custom handlers
    # enabled=False only filters LangChainTracer, not custom handlers
    assert "hidden_entrypoint" in run_names, (
        f"hidden_entrypoint should be traced to custom handlers, "
        f"enabled=False only filters LangSmith. Got: {run_names}"
    )


async def test_entrypoint_traceable_config_async():
    """Test that async @entrypoint honors __traceable_config__."""
    tracer = FakeTracer()

    def filter_inputs(inputs: Any) -> dict[str, Any]:
        return {"input": "[ASYNC_ENTRYPOINT_REDACTED]"}

    async def async_entrypoint(value: str) -> str:
        return f"async_{value}"

    _set_traceable_config(async_entrypoint, process_inputs=filter_inputs)

    workflow = entrypoint()(async_entrypoint)

    result = await workflow.ainvoke("secret", {"callbacks": [tracer]})

    assert result == "async_secret"

    runs = tracer.flattened_runs()
    entrypoint_runs = [r for r in runs if r.name == "async_entrypoint"]

    assert len(entrypoint_runs) == 1
    assert entrypoint_runs[0].inputs == {"input": "[ASYNC_ENTRYPOINT_REDACTED]"}


def test_task_traceable_config_process_inputs():
    """Test that @task honors __traceable_config__ with process_inputs."""
    tracer = FakeTracer()

    def filter_inputs(inputs: Any) -> dict[str, Any]:
        return {"args": "[TASK_INPUT_REDACTED]"}

    def my_task_func(value: str) -> str:
        return f"task_{value}"

    _set_traceable_config(my_task_func, process_inputs=filter_inputs)

    my_task = task(my_task_func)

    @entrypoint()
    def workflow(value: str) -> str:
        future = my_task(value)
        return future.result()

    result = workflow.invoke("secret_data", {"callbacks": [tracer]})

    assert result == "task_secret_data"

    runs = tracer.flattened_runs()
    task_runs = [r for r in runs if r.name == "my_task_func"]

    assert len(task_runs) == 1, (
        f"Expected 1 my_task_func run, got {len(task_runs)}. "
        f"All runs: {[r.name for r in runs]}"
    )

    # Verify inputs were filtered
    assert task_runs[0].inputs == {"args": "[TASK_INPUT_REDACTED]"}, (
        f"Expected inputs to be filtered, got {task_runs[0].inputs}"
    )


def test_task_traceable_config_process_outputs():
    """Test that @task honors __traceable_config__ with process_outputs."""
    tracer = FakeTracer()

    def filter_outputs(outputs: Any) -> Any:
        return {"output": "[TASK_OUTPUT_REDACTED]"}

    def my_task_func(value: str) -> str:
        return f"secret_result_{value}"

    _set_traceable_config(my_task_func, process_outputs=filter_outputs)

    my_task = task(my_task_func)

    @entrypoint()
    def workflow(value: str) -> str:
        future = my_task(value)
        return future.result()

    result = workflow.invoke("input", {"callbacks": [tracer]})

    # Actual result should be unfiltered
    assert result == "secret_result_input"

    runs = tracer.flattened_runs()
    task_runs = [r for r in runs if r.name == "my_task_func"]

    assert len(task_runs) == 1

    # Verify outputs were filtered in trace
    assert task_runs[0].outputs == {"output": "[TASK_OUTPUT_REDACTED]"}, (
        f"Expected outputs to be filtered, got {task_runs[0].outputs}"
    )


def test_task_traceable_config_enabled_false():
    """Test that @task with enabled=False still fires custom callbacks.

    Note: enabled=False filters out LangChainTracer (LangSmith) specifically,
    but custom callback handlers like FakeTracer still receive events.
    """
    tracer = FakeTracer()

    def hidden_task_func(value: str) -> str:
        return f"hidden_{value}"

    _set_traceable_config(hidden_task_func, enabled=False)

    hidden_task = task(hidden_task_func)

    def visible_task_func(value: str) -> str:
        return f"visible_{value}"

    visible_task = task(visible_task_func)

    @entrypoint()
    def workflow(value: str) -> str:
        hidden_result = hidden_task(value).result()
        visible_result = visible_task(hidden_result).result()
        return visible_result

    result = workflow.invoke("test", {"callbacks": [tracer]})

    assert result == "visible_hidden_test"

    runs = tracer.flattened_runs()
    run_names = [r.name for r in runs]

    # Both tasks should be traced to FakeTracer (a custom handler)
    # enabled=False only filters LangChainTracer, not custom handlers
    assert "visible_task_func" in run_names, (
        f"Expected visible_task_func in {run_names}"
    )
    assert "hidden_task_func" in run_names, (
        f"hidden_task_func should be traced to custom handlers, "
        f"enabled=False only filters LangSmith. Got: {run_names}"
    )


@pytest.mark.skipif(
    not ASYNCIO_ACCEPTS_CONTEXT,
    reason="Requires Python 3.11+ for async context propagation",
)
async def test_task_traceable_config_async():
    """Test that async @task honors __traceable_config__."""
    tracer = FakeTracer()

    def filter_inputs(inputs: Any) -> dict[str, Any]:
        return {"args": "[ASYNC_TASK_REDACTED]"}

    async def async_task_func(value: str) -> str:
        return f"async_task_{value}"

    _set_traceable_config(async_task_func, process_inputs=filter_inputs)

    async_task = task(async_task_func)

    @entrypoint()
    async def workflow(value: str) -> str:
        result = await async_task(value)
        return result

    result = await workflow.ainvoke("secret", {"callbacks": [tracer]})

    assert result == "async_task_secret"

    runs = tracer.flattened_runs()
    task_runs = [r for r in runs if r.name == "async_task_func"]

    assert len(task_runs) == 1
    assert task_runs[0].inputs == {"args": "[ASYNC_TASK_REDACTED]"}


def test_entrypoint_with_traceable_tasks_context_propagation():
    """Test that trace context propagates from entrypoint through tasks."""
    tracer = FakeTracer()

    def filter_inputs(inputs: Any) -> dict[str, Any]:
        return {"input": "[MASKED]"}

    def my_task_func(value: str) -> str:
        return f"task_{value}"

    my_task = task(my_task_func)

    def my_entrypoint(value: str) -> str:
        future = my_task(value)
        return future.result()

    _set_traceable_config(my_entrypoint, process_inputs=filter_inputs)

    workflow = entrypoint()(my_entrypoint)

    result = workflow.invoke("test", {"callbacks": [tracer]})

    assert result == "task_test"

    runs = tracer.flattened_runs()
    entrypoint_runs = [r for r in runs if r.name == "my_entrypoint"]
    task_runs = [r for r in runs if r.name == "my_task_func"]

    assert len(entrypoint_runs) == 1
    assert len(task_runs) == 1

    entrypoint_run = entrypoint_runs[0]
    task_run = task_runs[0]

    # Verify same trace_id (in same trace)
    assert task_run.trace_id == entrypoint_run.trace_id

    # Verify task's dotted_order starts with entrypoint's (is a descendant)
    assert task_run.dotted_order.startswith(entrypoint_run.dotted_order)


def test_task_traceable_config_process_inputs_error_handling():
    """Test that errors in process_inputs don't leak PII for @task."""
    tracer = FakeTracer()

    def bad_filter(inputs: Any) -> Any:
        raise ValueError("filter crashed!")

    def my_task_func(value: str) -> str:
        return f"task_{value}"

    _set_traceable_config(my_task_func, process_inputs=bad_filter)

    my_task = task(my_task_func)

    @entrypoint()
    def workflow(value: str) -> str:
        future = my_task(value)
        return future.result()

    # Workflow should still execute successfully
    result = workflow.invoke("secret_pii_data", {"callbacks": [tracer]})
    assert result == "task_secret_pii_data"

    # Trace should show error placeholder, NOT the raw PII data
    runs = tracer.flattened_runs()
    task_runs = [r for r in runs if r.name == "my_task_func"]

    assert len(task_runs) == 1
    # Should NOT contain "secret_pii_data"
    assert task_runs[0].inputs == {"error": "<trace_inputs processing failed>"}


def test_task_traceable_config_process_outputs_error_handling():
    """Test that errors in process_outputs don't leak PII for @task."""
    tracer = FakeTracer()

    def bad_filter(outputs: Any) -> Any:
        raise RuntimeError("output filter crashed!")

    def my_task_func(value: str) -> str:
        return "secret_output_pii"

    _set_traceable_config(my_task_func, process_outputs=bad_filter)

    my_task = task(my_task_func)

    @entrypoint()
    def workflow(value: str) -> str:
        future = my_task(value)
        return future.result()

    # Workflow should still execute successfully
    result = workflow.invoke("input", {"callbacks": [tracer]})
    assert result == "secret_output_pii"

    # Trace should show error placeholder, NOT the raw PII data
    runs = tracer.flattened_runs()
    task_runs = [r for r in runs if r.name == "my_task_func"]

    assert len(task_runs) == 1
    # Should NOT contain "secret_output_pii"
    assert task_runs[0].outputs == {"error": "<trace_outputs processing failed>"}


def test_task_with_real_traceable_decorator():
    """Test @task with actual @ls.traceable decorator from langsmith."""
    import langsmith as ls

    tracer = FakeTracer()

    @ls.traceable(process_inputs=lambda inputs: {"value": "[LANGSMITH_REDACTED]"})
    def traceable_task_func(value: str) -> str:
        return f"result_{value}"

    # Skip if langsmith version doesn't expose __traceable_config__
    if not hasattr(traceable_task_func, "__traceable_config__"):
        pytest.skip("langsmith version doesn't expose __traceable_config__")

    my_task = task(traceable_task_func)

    @entrypoint()
    def workflow(value: str) -> str:
        future = my_task(value)
        return future.result()

    result = workflow.invoke("secret_data", {"callbacks": [tracer]})

    assert result == "result_secret_data"

    runs = tracer.flattened_runs()
    task_runs = [r for r in runs if r.name == "traceable_task_func"]

    assert len(task_runs) == 1, (
        f"Expected 1 traceable_task_func run, got {len(task_runs)}. "
        f"All runs: {[r.name for r in runs]}"
    )

    # Verify inputs were filtered by the real @traceable decorator's process_inputs
    assert task_runs[0].inputs == {"value": "[LANGSMITH_REDACTED]"}, (
        f"Expected inputs to be filtered, got {task_runs[0].inputs}"
    )


# =============================================================================
# astream_events Tests
# =============================================================================


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize an event for comparison by replacing dynamic values with AnyStr."""
    normalized = {}
    for key, value in event.items():
        if key in ("run_id", "parent_ids"):
            # These are always dynamic UUIDs
            if isinstance(value, str):
                normalized[key] = AnyStr()
            elif isinstance(value, list):
                normalized[key] = [AnyStr() for _ in value]
            else:
                normalized[key] = value
        elif key == "metadata" and isinstance(value, dict):
            # Normalize checkpoint_ns in metadata
            normalized[key] = {
                k: AnyStr() if k == "langgraph_checkpoint_ns" else v
                for k, v in value.items()
            }
        else:
            normalized[key] = value
    return normalized


def _create_graph_for_astream_events(use_traceable: bool) -> StateGraph:
    """Create a simple graph, optionally with traceable config."""

    def node_a(state: SimpleState) -> SimpleState:
        return {"value": f"a_{state['value']}"}

    def node_b(state: SimpleState) -> SimpleState:
        return {"value": f"b_{state['value']}"}

    if use_traceable:
        # Apply traceable config with process_inputs filter
        def filter_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
            return {"state": "[FILTERED]"}

        _set_traceable_config(node_a, process_inputs=filter_inputs)
        _set_traceable_config(node_b, process_inputs=filter_inputs)

    builder = StateGraph(SimpleState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge("__start__", "node_a")
    builder.add_edge("node_a", "node_b")

    return builder.compile()


@pytest.mark.skipif(
    not ASYNCIO_ACCEPTS_CONTEXT,
    reason="Requires Python 3.11+ for async context propagation",
)
@pytest.mark.parametrize("use_traceable", [False, True], ids=["plain", "traceable"])
async def test_astream_events_with_traceable_config(use_traceable: bool):
    """Test that astream_events produces consistent event structure with/without traceable."""
    graph = _create_graph_for_astream_events(use_traceable)

    events = [e async for e in graph.astream_events({"value": "test"}, version="v2")]

    # Should have chain events for the graph and both nodes
    chain_start_events = [e for e in events if e["event"] == "on_chain_start"]
    chain_end_events = [e for e in events if e["event"] == "on_chain_end"]

    # Verify we have start/end events for main graph and both nodes
    start_names = [e["name"] for e in chain_start_events]
    end_names = [e["name"] for e in chain_end_events]

    assert "LangGraph" in start_names, f"Missing LangGraph start event: {start_names}"
    assert "node_a" in start_names, f"Missing node_a start event: {start_names}"
    assert "node_b" in start_names, f"Missing node_b start event: {start_names}"

    assert "LangGraph" in end_names, f"Missing LangGraph end event: {end_names}"
    assert "node_a" in end_names, f"Missing node_a end event: {end_names}"
    assert "node_b" in end_names, f"Missing node_b end event: {end_names}"

    # Verify final output is correct (traceable shouldn't affect execution)
    final_event = next(
        e
        for e in reversed(events)
        if e["event"] == "on_chain_end" and e["name"] == "LangGraph"
    )
    assert final_event["data"]["output"] == {"value": "b_a_test"}


@pytest.mark.skipif(
    not ASYNCIO_ACCEPTS_CONTEXT,
    reason="Requires Python 3.11+ for async context propagation",
)
async def test_astream_events_traceable_vs_plain_structure_match():
    """Test that traceable config doesn't change the event structure, only input/output values."""
    plain_graph = _create_graph_for_astream_events(use_traceable=False)
    traceable_graph = _create_graph_for_astream_events(use_traceable=True)

    # Use FakeTracer to collect callback invocations
    plain_tracer = FakeTracer()
    traceable_tracer = FakeTracer()

    plain_events = [
        e
        async for e in plain_graph.astream_events(
            {"value": "test"}, version="v2", config={"callbacks": [plain_tracer]}
        )
    ]
    traceable_events = [
        e
        async for e in traceable_graph.astream_events(
            {"value": "test"}, version="v2", config={"callbacks": [traceable_tracer]}
        )
    ]

    # Verify callback handlers were invoked for both - get run names in order
    plain_runs = plain_tracer.flattened_runs()
    traceable_runs = traceable_tracer.flattened_runs()

    plain_run_names = [r.name for r in plain_runs]
    traceable_run_names = [r.name for r in traceable_runs]

    # Same runs should be recorded by the callback handler
    assert plain_run_names == traceable_run_names, (
        f"Callback run names mismatch:\nplain={plain_run_names}\n"
        f"traceable={traceable_run_names}"
    )

    # Verify we actually got runs for our nodes
    assert "node_a" in plain_run_names, f"node_a not in runs: {plain_run_names}"
    assert "node_b" in plain_run_names, f"node_b not in runs: {plain_run_names}"

    # Same number of events
    assert len(plain_events) == len(traceable_events), (
        f"Event count mismatch: plain={len(plain_events)}, "
        f"traceable={len(traceable_events)}"
    )

    # Same event types in same order
    plain_types = [(e["event"], e.get("name")) for e in plain_events]
    traceable_types = [(e["event"], e.get("name")) for e in traceable_events]
    assert plain_types == traceable_types, (
        f"Event types mismatch:\nplain={plain_types}\ntraceable={traceable_types}"
    )

    # For each event pair, compare structure (ignoring dynamic values and traced data)
    for i, (plain_e, traceable_e) in enumerate(zip(plain_events, traceable_events)):
        # Event type and name must match
        assert plain_e["event"] == traceable_e["event"], f"Event {i}: type mismatch"
        assert plain_e.get("name") == traceable_e.get("name"), (
            f"Event {i}: name mismatch"
        )

        # Tags should match
        assert plain_e.get("tags") == traceable_e.get("tags"), (
            f"Event {i}: tags mismatch"
        )

        # parent_ids length should match (IDs themselves will differ)
        plain_parents = plain_e.get("parent_ids", [])
        traceable_parents = traceable_e.get("parent_ids", [])
        assert len(plain_parents) == len(traceable_parents), (
            f"Event {i}: parent_ids length mismatch"
        )

    # Verify both graphs produce the same final result
    plain_final = next(
        e
        for e in reversed(plain_events)
        if e["event"] == "on_chain_end" and e["name"] == "LangGraph"
    )
    traceable_final = next(
        e
        for e in reversed(traceable_events)
        if e["event"] == "on_chain_end" and e["name"] == "LangGraph"
    )
    assert plain_final["data"]["output"] == traceable_final["data"]["output"]


@pytest.mark.skipif(
    not ASYNCIO_ACCEPTS_CONTEXT,
    reason="Requires Python 3.11+ for async context propagation",
)
async def test_astream_events_traceable_filters_traced_inputs():
    """Test that traceable process_inputs actually filters the traced input data."""
    traceable_graph = _create_graph_for_astream_events(use_traceable=True)

    events = [
        e
        async for e in traceable_graph.astream_events(
            {"value": "secret_data"}, version="v2"
        )
    ]

    # Find node_a start event - its input should be filtered
    node_a_start = next(
        e for e in events if e["event"] == "on_chain_start" and e["name"] == "node_a"
    )

    # The input should be filtered by process_inputs
    assert node_a_start["data"]["input"] == {"state": "[FILTERED]"}, (
        f"Expected filtered input, got {node_a_start['data']['input']}"
    )

    # But the actual execution should still work with real data
    final_event = next(
        e
        for e in reversed(events)
        if e["event"] == "on_chain_end" and e["name"] == "LangGraph"
    )
    assert final_event["data"]["output"] == {"value": "b_a_secret_data"}


# =============================================================================
# get_config() Compatibility Tests
# =============================================================================


def test_traceable_enabled_false_allows_get_config():
    """Test that nodes with enabled=False can still call get_config().

    This is a regression test for a bug where the trace=False path in
    RunnableSeq skipped set_config_context(), breaking get_config() calls.
    """
    from langgraph.config import get_config

    config_thread_id = None

    def my_node(state: SimpleState) -> SimpleState:
        nonlocal config_thread_id
        config = get_config()  # This should work even with enabled=False!
        config_thread_id = config.get("configurable", {}).get("thread_id")
        return {"value": f"got_config_{state['value']}"}

    _set_traceable_config(my_node, enabled=False)

    builder = StateGraph(SimpleState)
    builder.add_node("my_node", my_node)
    builder.add_edge("__start__", "my_node")
    graph = builder.compile()

    result = graph.invoke(
        {"value": "test"}, {"configurable": {"thread_id": "test-thread-123"}}
    )

    assert result == {"value": "got_config_test"}
    assert config_thread_id == "test-thread-123", (
        f"Expected thread_id 'test-thread-123', got {config_thread_id}"
    )


def test_traceable_enabled_false_still_fires_custom_callbacks():
    """Test that enabled=False skips LangSmith but fires callbacks to custom handlers.

    This verifies that on_chain_start and on_chain_end are still called for
    custom callback handlers even when the node has enabled=False in its
    traceable config.
    """
    tracer = FakeTracer()

    def my_node(state: SimpleState) -> SimpleState:
        return {"value": f"processed_{state['value']}"}

    _set_traceable_config(my_node, enabled=False)

    builder = StateGraph(SimpleState)
    builder.add_node("my_node", my_node)
    builder.add_edge("__start__", "my_node")
    graph = builder.compile()

    result = graph.invoke({"value": "test"}, {"callbacks": [tracer]})

    # Verify the graph executed correctly
    assert result == {"value": "processed_test"}

    # Verify custom callback handler received events for the node
    runs = tracer.flattened_runs()
    run_names = [r.name for r in runs]

    assert "LangGraph" in run_names, f"Missing LangGraph run: {run_names}"
    assert "my_node" in run_names, (
        f"enabled=False should still fire callbacks to custom handlers. "
        f"Got runs: {run_names}"
    )
