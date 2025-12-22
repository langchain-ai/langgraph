"""True ReAct primitives for explicit reasoning traces.

This module provides first-class types for capturing the Thought → Action → Observation
cycle that defines the ReAct (Reasoning + Acting) pattern. Unlike implicit reasoning
hidden in LLM outputs, these primitives make agent reasoning observable, debuggable,
and learnable.

Example usage:
    ```python
    from typing import Annotated
    from langgraph.prebuilt.reasoning import (
        Thought,
        Observation,
        ReActStep,
        add_reasoning_steps,
    )
    from typing_extensions import TypedDict

    class AgentStateWithReasoning(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        reasoning_trace: Annotated[list[ReActStep], add_reasoning_steps]
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Thought:
    """Represents explicit reasoning before taking an action.

    Captures the agent's thought process as structured data rather than
    implicit text in LLM outputs. This enables:
    - Debugging: See exactly what the agent was thinking
    - Evaluation: Assess reasoning quality programmatically
    - Transparency: Show users the decision-making process

    Attributes:
        content: The reasoning text (e.g., "I need to search for weather data")
        confidence: Optional confidence score from 0.0 to 1.0, if the model provides it
        metadata: Additional context (e.g., model name, token count, latency)

    Example:
        ```python
        thought = Thought(
            content="The user wants weather info. I should use the weather API.",
            confidence=0.95,
            metadata={"model": "claude-3-opus", "tokens": 42}
        )
        ```
    """

    content: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence is in valid range if provided."""
        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    f"confidence must be between 0.0 and 1.0, got {self.confidence}"
                )


@dataclass
class Observation:
    """Represents the result of an action (tool execution).

    Provides richer information than raw tool results, including metadata
    about execution success, timing, and errors. This enables:
    - Monitoring: Track tool performance and failure rates
    - Debugging: Understand why tools failed
    - Analysis: Measure execution patterns over time

    Attributes:
        tool_name: Name of the tool that was executed
        tool_input: The input arguments passed to the tool
        tool_output: The result returned by the tool (can be any type)
        success: Whether the tool executed successfully
        error: Error message if the tool failed (None if successful)
        duration_ms: Execution time in milliseconds (None if not measured)

    Example:
        ```python
        observation = Observation(
            tool_name="get_weather",
            tool_input={"city": "Paris"},
            tool_output="22°C, sunny",
            success=True,
            duration_ms=150.5,
        )
        ```
    """

    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Any
    success: bool
    error: str | None = None
    duration_ms: float | None = None


@dataclass
class ReActStep:
    """Represents one complete Thought → Action → Observation cycle.

    This is the core primitive for capturing reasoning traces. Each step
    represents a complete iteration of the ReAct loop:
    1. Thought: Why the agent decided to take an action
    2. Action: The tool call (or None for final response)
    3. Observation: The result of the action

    The `action` field stores the tool call information as a dict with
    standard fields: `name`, `args`, and optionally `id`.

    Attributes:
        thought: The reasoning before this action (may be None if implicit)
        action: The tool call dict, or None for final response steps
        observation: The tool result, or None for final response steps
        step_number: Zero-indexed position in the reasoning trace
        timestamp: When this step was recorded

    Example:
        ```python
        step = ReActStep(
            thought=Thought(content="I need to find the weather"),
            action={"name": "get_weather", "args": {"city": "Paris"}, "id": "call_123"},
            observation=Observation(
                tool_name="get_weather",
                tool_input={"city": "Paris"},
                tool_output="22°C, sunny",
                success=True,
            ),
            step_number=0,
            timestamp=datetime.now(),
        )
        ```

    For final response steps (no tool call):
        ```python
        final_step = ReActStep(
            thought=Thought(content="I have the information to respond"),
            action=None,  # No tool call
            observation=None,  # No tool result
            step_number=1,
            timestamp=datetime.now(),
        )
        ```
    """

    thought: Thought | None
    action: dict[str, Any] | None  # Tool call dict with name, args, id
    observation: Observation | None
    step_number: int
    timestamp: datetime = field(default_factory=datetime.now)

    def is_tool_call(self) -> bool:
        """Check if this step involves a tool call."""
        return self.action is not None

    def is_final_response(self) -> bool:
        """Check if this is a final response step (no tool call)."""
        return self.action is None


def add_reasoning_steps(
    existing: list[ReActStep] | None,
    new: list[ReActStep] | None,
) -> list[ReActStep]:
    """Reducer for accumulating reasoning steps in agent state.

    This function is designed to be used with LangGraph's Annotated type
    for state fields. It accumulates reasoning steps across graph iterations,
    building a complete trace of the agent's reasoning.

    Args:
        existing: The current list of reasoning steps (may be None)
        new: New steps to add (may be None)

    Returns:
        Combined list of all reasoning steps

    Example:
        ```python
        from typing import Annotated
        from typing_extensions import TypedDict

        class AgentState(TypedDict):
            messages: Annotated[list[BaseMessage], add_messages]
            reasoning_trace: Annotated[list[ReActStep], add_reasoning_steps]

        # In a graph node:
        def my_node(state: AgentState) -> dict:
            new_step = ReActStep(...)
            return {"reasoning_trace": [new_step]}  # Will be accumulated
        ```
    """
    existing = existing or []
    new = new or []
    return existing + new


__all__ = [
    "Thought",
    "Observation",
    "ReActStep",
    "add_reasoning_steps",
]
