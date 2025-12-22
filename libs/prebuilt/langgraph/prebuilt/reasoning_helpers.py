"""Helper utilities for working with ReAct reasoning traces.

This module provides practical utilities for capturing, analyzing, and
rendering reasoning traces from agent executions.

Example:
    ```python
    from langgraph.prebuilt.reasoning_helpers import (
        capture_tool_observation,
        render_reasoning_trace,
        calculate_trace_metrics,
    )

    # In your tool execution code:
    observation = capture_tool_observation(
        tool_name="search",
        tool_input={"query": "weather"},
        tool_func=search_tool,
    )

    # After agent completes:
    print(render_reasoning_trace(state["reasoning_trace"]))
    metrics = calculate_trace_metrics(state["reasoning_trace"])
    ```
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.prebuilt.reasoning import Observation, ReActStep, Thought


def capture_tool_observation(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_func: Callable[..., Any],
) -> Observation:
    """Execute a tool and capture the result as an Observation.

    This helper wraps tool execution to automatically capture timing,
    success/failure status, and any errors.

    Args:
        tool_name: Name of the tool being executed
        tool_input: Input arguments for the tool
        tool_func: The tool function to call (will be called with **tool_input)

    Returns:
        Observation with captured execution details

    Example:
        ```python
        def search_web(query: str) -> str:
            # ... actual search implementation
            return "search results"

        obs = capture_tool_observation(
            tool_name="search_web",
            tool_input={"query": "LangGraph tutorial"},
            tool_func=search_web,
        )
        print(f"Success: {obs.success}, Duration: {obs.duration_ms}ms")
        ```
    """
    start_time = time.perf_counter()
    try:
        result = tool_func(**tool_input)
        duration_ms = (time.perf_counter() - start_time) * 1000
        return Observation(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=result,
            success=True,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return Observation(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=None,
            success=False,
            error=str(e),
            duration_ms=duration_ms,
        )


@dataclass
class TraceMetrics:
    """Metrics calculated from a reasoning trace."""

    total_steps: int
    tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    final_response_steps: int
    average_confidence: float | None
    total_duration_ms: float
    tool_names_used: list[str]


def calculate_trace_metrics(trace: list[ReActStep]) -> TraceMetrics:
    """Calculate metrics from a reasoning trace.

    Provides aggregated statistics about agent execution for monitoring,
    evaluation, and optimization.

    Args:
        trace: List of ReActStep objects from agent execution

    Returns:
        TraceMetrics with aggregated statistics

    Example:
        ```python
        metrics = calculate_trace_metrics(state["reasoning_trace"])
        print(f"Total steps: {metrics.total_steps}")
        print(f"Tool calls: {metrics.tool_calls}")
        print(f"Success rate: {metrics.successful_tool_calls / metrics.tool_calls:.1%}")
        print(f"Total duration: {metrics.total_duration_ms:.1f}ms")
        ```
    """
    if not trace:
        return TraceMetrics(
            total_steps=0,
            tool_calls=0,
            successful_tool_calls=0,
            failed_tool_calls=0,
            final_response_steps=0,
            average_confidence=None,
            total_duration_ms=0.0,
            tool_names_used=[],
        )

    tool_calls = [s for s in trace if s.is_tool_call()]
    final_responses = [s for s in trace if s.is_final_response()]

    successful = sum(1 for s in trace if s.observation and s.observation.success)
    failed = sum(1 for s in trace if s.observation and not s.observation.success)

    confidences = [
        s.thought.confidence
        for s in trace
        if s.thought and s.thought.confidence is not None
    ]
    avg_confidence = sum(confidences) / len(confidences) if confidences else None

    total_duration = sum(
        s.observation.duration_ms
        for s in trace
        if s.observation and s.observation.duration_ms
    )

    tool_names = list(
        dict.fromkeys(
            s.action["name"] for s in trace if s.action and "name" in s.action
        )
    )

    return TraceMetrics(
        total_steps=len(trace),
        tool_calls=len(tool_calls),
        successful_tool_calls=successful,
        failed_tool_calls=failed,
        final_response_steps=len(final_responses),
        average_confidence=avg_confidence,
        total_duration_ms=total_duration,
        tool_names_used=tool_names,
    )


def render_reasoning_trace(
    trace: list[ReActStep],
    *,
    show_timestamps: bool = False,
    show_metadata: bool = False,
) -> str:
    """Render a reasoning trace as human-readable text.

    Useful for debugging, logging, or displaying agent reasoning to users.

    Args:
        trace: List of ReActStep objects from agent execution
        show_timestamps: Include timestamps for each step
        show_metadata: Include thought metadata if available

    Returns:
        Formatted string representation of the trace

    Example:
        ```python
        print(render_reasoning_trace(state["reasoning_trace"]))
        # Output:
        # Step 0:
        #   Thought: I need to search for weather data
        #   Action: get_weather(city="Paris")
        #   Observation: 22°C, sunny (150ms, success)
        #
        # Step 1:
        #   Thought: Now I can respond to the user
        #   [Final Response]
        ```
    """
    if not trace:
        return "[Empty reasoning trace]"

    lines = []
    for step in trace:
        step_lines = [f"Step {step.step_number}:"]

        if show_timestamps:
            step_lines[0] += f" ({step.timestamp.isoformat()})"

        # Thought
        if step.thought:
            thought_line = f"  Thought: {step.thought.content}"
            if step.thought.confidence is not None:
                thought_line += f" (confidence: {step.thought.confidence:.0%})"
            step_lines.append(thought_line)

            if show_metadata and step.thought.metadata:
                step_lines.append(f"    Metadata: {step.thought.metadata}")
        else:
            step_lines.append("  Thought: [implicit]")

        # Action
        if step.action:
            action_name = step.action.get("name", "unknown")
            action_args = step.action.get("args", {})
            args_str = ", ".join(f'{k}="{v}"' for k, v in action_args.items())
            step_lines.append(f"  Action: {action_name}({args_str})")
        else:
            step_lines.append("  [Final Response]")

        # Observation
        if step.observation:
            obs = step.observation
            output_str = str(obs.tool_output)
            if len(output_str) > 100:
                output_str = output_str[:100] + "..."

            status = "success" if obs.success else f"failed: {obs.error}"
            timing = f", {obs.duration_ms:.0f}ms" if obs.duration_ms else ""
            step_lines.append(f"  Observation: {output_str} ({status}{timing})")

        lines.append("\n".join(step_lines))

    return "\n\n".join(lines)


def find_failed_steps(trace: list[ReActStep]) -> list[ReActStep]:
    """Find all steps with failed tool executions.

    Useful for debugging agent failures.

    Args:
        trace: List of ReActStep objects

    Returns:
        List of steps where the observation indicates failure
    """
    return [step for step in trace if step.observation and not step.observation.success]


def get_tool_usage_summary(trace: list[ReActStep]) -> dict[str, dict[str, Any]]:
    """Get usage summary per tool.

    Args:
        trace: List of ReActStep objects

    Returns:
        Dict mapping tool name to usage statistics

    Example:
        ```python
        summary = get_tool_usage_summary(trace)
        for tool, stats in summary.items():
            print(f"{tool}: {stats['calls']} calls, "
                  f"{stats['success_rate']:.0%} success rate")
        ```
    """
    tool_stats: dict[str, dict[str, Any]] = {}

    for step in trace:
        if not step.action or "name" not in step.action:
            continue

        tool_name = step.action["name"]
        if tool_name not in tool_stats:
            tool_stats[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_duration_ms": 0.0,
            }

        stats = tool_stats[tool_name]
        stats["calls"] += 1

        if step.observation:
            if step.observation.success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            if step.observation.duration_ms:
                stats["total_duration_ms"] += step.observation.duration_ms

    # Calculate derived metrics
    for stats in tool_stats.values():
        stats["success_rate"] = (
            stats["successes"] / stats["calls"] if stats["calls"] > 0 else 0.0
        )
        stats["avg_duration_ms"] = (
            stats["total_duration_ms"] / stats["calls"] if stats["calls"] > 0 else 0.0
        )

    return tool_stats


class ReasoningCapture:
    """Captures reasoning traces from ToolNode execution.

    This class provides a wrapper for ToolNode's `wrap_tool_call` parameter
    that automatically captures Observations and builds ReActSteps from
    tool executions.

    The captured steps can be retrieved and added to agent state after
    ToolNode execution completes.

    Example:
        ```python
        from langgraph.prebuilt import ToolNode, ReActStep, add_reasoning_steps
        from langgraph.prebuilt.reasoning_helpers import ReasoningCapture

        # Create capture instance
        capture = ReasoningCapture()

        # Use with ToolNode
        tool_node = ToolNode(
            tools=[my_tool],
            wrap_tool_call=capture.wrap_tool_call,
        )

        # In your graph node:
        def tools_node(state):
            # Clear previous captures
            capture.clear()

            # Execute tools
            result = tool_node.invoke(state)

            # Get captured steps and add to state
            steps = capture.get_steps()
            return {
                "messages": result["messages"],
                "reasoning_trace": steps,
            }
        ```

    For more control, you can also set thoughts before tool execution:
        ```python
        capture.set_pending_thought(Thought(content="I need to search..."))
        result = tool_node.invoke(state)
        ```
    """

    def __init__(self) -> None:
        """Initialize the reasoning capture."""
        self._steps: list[ReActStep] = []
        self._pending_thought: Thought | None = None
        self._step_counter: int = 0

    def clear(self) -> None:
        """Clear all captured steps and reset state.

        Call this before each ToolNode invocation to start fresh.
        """
        self._steps = []
        self._pending_thought = None
        self._step_counter = 0

    def set_pending_thought(self, thought: Thought) -> None:
        """Set a thought to be associated with the next tool call.

        Args:
            thought: The Thought to associate with the next captured step.
        """
        self._pending_thought = thought

    def get_steps(self) -> list[ReActStep]:
        """Get all captured ReActSteps.

        Returns:
            List of captured ReActStep objects, in execution order.
        """
        return list(self._steps)

    def wrap_tool_call(
        self,
        request: Any,  # ToolCallRequest
        execute: Callable[
            [Any], Any
        ],  # Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> Any:  # ToolMessage | Command
        """Wrapper function for ToolNode.wrap_tool_call.

        This captures timing and result information to create Observations
        and ReActSteps automatically.

        Args:
            request: The ToolCallRequest from ToolNode
            execute: The execute callable to invoke the tool

        Returns:
            The ToolMessage or Command from tool execution
        """
        from datetime import datetime

        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("args", {})

        # Capture the pending thought
        thought = self._pending_thought
        self._pending_thought = None

        # Execute with timing
        start_time = time.perf_counter()
        error_msg: str | None = None
        success = True
        tool_output: Any = None

        try:
            result = execute(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Extract output from ToolMessage if applicable
            if hasattr(result, "content"):
                tool_output = result.content
                # Check for error status
                if hasattr(result, "status") and result.status == "error":
                    success = False
                    error_msg = str(result.content)
            else:
                tool_output = result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            success = False
            error_msg = str(e)
            raise

        finally:
            # Always capture the observation (even on error)
            observation = Observation(
                tool_name=tool_name,
                tool_input=dict(tool_args) if tool_args else {},
                tool_output=tool_output,
                success=success,
                error=error_msg,
                duration_ms=duration_ms,
            )

            step = ReActStep(
                thought=thought,
                action={
                    "name": tool_name,
                    "args": dict(tool_args) if tool_args else {},
                    "id": tool_call.get("id"),
                },
                observation=observation,
                step_number=self._step_counter,
                timestamp=datetime.now(),
            )
            self._steps.append(step)
            self._step_counter += 1

        return result


__all__ = [
    "capture_tool_observation",
    "calculate_trace_metrics",
    "render_reasoning_trace",
    "find_failed_steps",
    "get_tool_usage_summary",
    "TraceMetrics",
    "ReasoningCapture",
]
