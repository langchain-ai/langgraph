"""Tests for True ReAct primitives (Thought, Observation, ReActStep)."""

import json
from datetime import datetime
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langgraph.prebuilt import (
    Observation,
    ReActStep,
    Thought,
    add_reasoning_steps,
    calculate_trace_metrics,
    capture_tool_observation,
    find_failed_steps,
    get_tool_usage_summary,
    render_reasoning_trace,
)


class TestThought:
    """Tests for the Thought dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic Thought."""
        thought = Thought(content="I need to search for weather data")
        assert thought.content == "I need to search for weather data"
        assert thought.confidence is None
        assert thought.metadata == {}

    def test_with_confidence(self) -> None:
        """Test creating a Thought with confidence score."""
        thought = Thought(
            content="I should use the calculator tool",
            confidence=0.95,
        )
        assert thought.content == "I should use the calculator tool"
        assert thought.confidence == 0.95

    def test_with_metadata(self) -> None:
        """Test creating a Thought with metadata."""
        thought = Thought(
            content="Let me analyze this data",
            metadata={"model": "claude-3-opus", "tokens": 42, "latency_ms": 150},
        )
        assert thought.metadata["model"] == "claude-3-opus"
        assert thought.metadata["tokens"] == 42

    def test_confidence_validation_too_high(self) -> None:
        """Test that confidence > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            Thought(content="test", confidence=1.5)

    def test_confidence_validation_too_low(self) -> None:
        """Test that confidence < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            Thought(content="test", confidence=-0.1)

    def test_confidence_boundary_values(self) -> None:
        """Test confidence at boundary values (0.0 and 1.0)."""
        thought_zero = Thought(content="uncertain", confidence=0.0)
        assert thought_zero.confidence == 0.0

        thought_one = Thought(content="certain", confidence=1.0)
        assert thought_one.confidence == 1.0


class TestObservation:
    """Tests for the Observation dataclass."""

    def test_successful_observation(self) -> None:
        """Test creating a successful observation."""
        obs = Observation(
            tool_name="get_weather",
            tool_input={"city": "Paris"},
            tool_output="22°C, sunny",
            success=True,
        )
        assert obs.tool_name == "get_weather"
        assert obs.tool_input == {"city": "Paris"}
        assert obs.tool_output == "22°C, sunny"
        assert obs.success is True
        assert obs.error is None
        assert obs.duration_ms is None

    def test_failed_observation(self) -> None:
        """Test creating a failed observation."""
        obs = Observation(
            tool_name="search_database",
            tool_input={"query": "SELECT * FROM users"},
            tool_output=None,
            success=False,
            error="Connection timeout",
        )
        assert obs.success is False
        assert obs.error == "Connection timeout"
        assert obs.tool_output is None

    def test_with_timing(self) -> None:
        """Test observation with execution timing."""
        obs = Observation(
            tool_name="slow_api",
            tool_input={"endpoint": "/data"},
            tool_output={"status": "ok"},
            success=True,
            duration_ms=1523.45,
        )
        assert obs.duration_ms == 1523.45

    def test_complex_output(self) -> None:
        """Test observation with complex output types."""
        complex_output = {
            "results": [1, 2, 3],
            "nested": {"key": "value"},
            "count": 42,
        }
        obs = Observation(
            tool_name="complex_tool",
            tool_input={},
            tool_output=complex_output,
            success=True,
        )
        assert obs.tool_output == complex_output
        assert obs.tool_output["nested"]["key"] == "value"


class TestReActStep:
    """Tests for the ReActStep dataclass."""

    def test_complete_step(self) -> None:
        """Test creating a complete ReActStep with thought, action, observation."""
        thought = Thought(content="I need weather data")
        observation = Observation(
            tool_name="get_weather",
            tool_input={"city": "Paris"},
            tool_output="22°C, sunny",
            success=True,
        )
        step = ReActStep(
            thought=thought,
            action={"name": "get_weather", "args": {"city": "Paris"}, "id": "call_123"},
            observation=observation,
            step_number=0,
        )
        assert step.thought == thought
        assert step.action["name"] == "get_weather"
        assert step.observation == observation
        assert step.step_number == 0
        assert isinstance(step.timestamp, datetime)

    def test_final_response_step(self) -> None:
        """Test creating a final response step (no tool call)."""
        thought = Thought(content="I have all the information to respond")
        step = ReActStep(
            thought=thought,
            action=None,
            observation=None,
            step_number=1,
        )
        assert step.action is None
        assert step.observation is None
        assert step.is_final_response()
        assert not step.is_tool_call()

    def test_is_tool_call(self) -> None:
        """Test is_tool_call() method."""
        tool_step = ReActStep(
            thought=None,
            action={"name": "search", "args": {}},
            observation=None,
            step_number=0,
        )
        assert tool_step.is_tool_call()
        assert not tool_step.is_final_response()

    def test_implicit_thought(self) -> None:
        """Test step with implicit (None) thought."""
        obs = Observation(
            tool_name="calculator",
            tool_input={"expr": "2+2"},
            tool_output=4,
            success=True,
        )
        step = ReActStep(
            thought=None,  # Implicit reasoning
            action={"name": "calculator", "args": {"expr": "2+2"}},
            observation=obs,
            step_number=0,
        )
        assert step.thought is None
        assert step.action is not None

    def test_custom_timestamp(self) -> None:
        """Test step with custom timestamp."""
        custom_time = datetime(2024, 1, 15, 10, 30, 0)
        step = ReActStep(
            thought=Thought(content="test"),
            action=None,
            observation=None,
            step_number=0,
            timestamp=custom_time,
        )
        assert step.timestamp == custom_time


class TestAddReasoningSteps:
    """Tests for the add_reasoning_steps reducer."""

    def test_add_to_empty(self) -> None:
        """Test adding steps to empty list."""
        new_steps = [
            ReActStep(
                thought=Thought(content="First thought"),
                action=None,
                observation=None,
                step_number=0,
            )
        ]
        result = add_reasoning_steps(None, new_steps)
        assert len(result) == 1
        assert result[0].thought.content == "First thought"

    def test_add_to_existing(self) -> None:
        """Test adding steps to existing list."""
        existing = [
            ReActStep(
                thought=Thought(content="First"),
                action=None,
                observation=None,
                step_number=0,
            )
        ]
        new = [
            ReActStep(
                thought=Thought(content="Second"),
                action=None,
                observation=None,
                step_number=1,
            )
        ]
        result = add_reasoning_steps(existing, new)
        assert len(result) == 2
        assert result[0].thought.content == "First"
        assert result[1].thought.content == "Second"

    def test_add_none_to_existing(self) -> None:
        """Test adding None to existing list."""
        existing = [
            ReActStep(
                thought=Thought(content="Existing"),
                action=None,
                observation=None,
                step_number=0,
            )
        ]
        result = add_reasoning_steps(existing, None)
        assert len(result) == 1
        assert result[0].thought.content == "Existing"

    def test_both_none(self) -> None:
        """Test when both inputs are None."""
        result = add_reasoning_steps(None, None)
        assert result == []

    def test_accumulation_preserves_order(self) -> None:
        """Test that accumulation preserves step order."""
        steps = []
        for i in range(5):
            new_step = [
                ReActStep(
                    thought=Thought(content=f"Step {i}"),
                    action=None,
                    observation=None,
                    step_number=i,
                )
            ]
            steps = add_reasoning_steps(steps, new_step)

        assert len(steps) == 5
        for i, step in enumerate(steps):
            assert step.thought.content == f"Step {i}"
            assert step.step_number == i


class TestIntegration:
    """Integration tests showing typical usage patterns."""

    def test_state_with_reasoning_trace(self) -> None:
        """Test using reasoning trace in a TypedDict state."""

        class AgentState(TypedDict):
            reasoning_trace: Annotated[list[ReActStep], add_reasoning_steps]

        # Simulate graph node returning new step
        def agent_node(state: AgentState) -> dict:
            step = ReActStep(
                thought=Thought(content="Analyzing user request"),
                action={"name": "search", "args": {"query": "test"}},
                observation=Observation(
                    tool_name="search",
                    tool_input={"query": "test"},
                    tool_output=["result1", "result2"],
                    success=True,
                    duration_ms=100.0,
                ),
                step_number=len(state.get("reasoning_trace", [])),
            )
            return {"reasoning_trace": [step]}

        # Initial state
        state: AgentState = {"reasoning_trace": []}

        # Simulate first iteration
        update1 = agent_node(state)
        state["reasoning_trace"] = add_reasoning_steps(
            state["reasoning_trace"], update1["reasoning_trace"]
        )

        # Simulate second iteration
        update2 = agent_node(state)
        state["reasoning_trace"] = add_reasoning_steps(
            state["reasoning_trace"], update2["reasoning_trace"]
        )

        assert len(state["reasoning_trace"]) == 2
        assert state["reasoning_trace"][0].step_number == 0
        assert state["reasoning_trace"][1].step_number == 1

    def test_debugging_failed_step(self) -> None:
        """Test using reasoning trace to debug a failed tool call."""
        trace = [
            ReActStep(
                thought=Thought(content="Need to fetch data from API"),
                action={"name": "api_call", "args": {"endpoint": "/users"}},
                observation=Observation(
                    tool_name="api_call",
                    tool_input={"endpoint": "/users"},
                    tool_output=None,
                    success=False,
                    error="Connection refused",
                    duration_ms=5000.0,
                ),
                step_number=0,
            ),
        ]

        # Find failed steps
        failed_steps = [s for s in trace if s.observation and not s.observation.success]
        assert len(failed_steps) == 1
        assert failed_steps[0].observation.error == "Connection refused"
        assert failed_steps[0].thought.content == "Need to fetch data from API"

    def test_calculate_metrics(self) -> None:
        """Test calculating metrics from reasoning trace."""
        trace = [
            ReActStep(
                thought=Thought(content="Step 1", confidence=0.9),
                action={"name": "tool1", "args": {}},
                observation=Observation(
                    tool_name="tool1",
                    tool_input={},
                    tool_output="ok",
                    success=True,
                    duration_ms=100.0,
                ),
                step_number=0,
            ),
            ReActStep(
                thought=Thought(content="Step 2", confidence=0.8),
                action={"name": "tool2", "args": {}},
                observation=Observation(
                    tool_name="tool2",
                    tool_input={},
                    tool_output="ok",
                    success=True,
                    duration_ms=200.0,
                ),
                step_number=1,
            ),
            ReActStep(
                thought=Thought(content="Final", confidence=0.95),
                action=None,
                observation=None,
                step_number=2,
            ),
        ]

        # Calculate metrics
        total_steps = len(trace)
        tool_calls = sum(1 for s in trace if s.is_tool_call())
        avg_confidence = sum(
            s.thought.confidence for s in trace if s.thought and s.thought.confidence
        ) / len([s for s in trace if s.thought and s.thought.confidence])
        total_duration = sum(
            s.observation.duration_ms
            for s in trace
            if s.observation and s.observation.duration_ms
        )

        assert total_steps == 3
        assert tool_calls == 2
        assert avg_confidence == pytest.approx(0.883, rel=0.01)
        assert total_duration == 300.0


class TestCaptureToolObservation:
    """Tests for capture_tool_observation helper."""

    def test_successful_capture(self) -> None:
        """Test capturing successful tool execution."""

        def add(a: int, b: int) -> int:
            return a + b

        obs = capture_tool_observation(
            tool_name="add",
            tool_input={"a": 2, "b": 3},
            tool_func=add,
        )
        assert obs.tool_name == "add"
        assert obs.tool_output == 5
        assert obs.success is True
        assert obs.error is None
        assert obs.duration_ms is not None
        assert obs.duration_ms >= 0

    def test_failed_capture(self) -> None:
        """Test capturing failed tool execution."""

        def failing_tool(x: int) -> int:
            raise ValueError("Something went wrong")

        obs = capture_tool_observation(
            tool_name="failing_tool",
            tool_input={"x": 1},
            tool_func=failing_tool,
        )
        assert obs.tool_name == "failing_tool"
        assert obs.success is False
        assert obs.error == "Something went wrong"
        assert obs.tool_output is None


class TestCalculateTraceMetrics:
    """Tests for calculate_trace_metrics helper."""

    def test_empty_trace(self) -> None:
        """Test metrics for empty trace."""
        metrics = calculate_trace_metrics([])
        assert metrics.total_steps == 0
        assert metrics.tool_calls == 0
        assert metrics.average_confidence is None

    def test_full_trace(self) -> None:
        """Test metrics for full trace."""
        trace = [
            ReActStep(
                thought=Thought(content="Step 1", confidence=0.9),
                action={"name": "search", "args": {}},
                observation=Observation(
                    tool_name="search",
                    tool_input={},
                    tool_output="result",
                    success=True,
                    duration_ms=100.0,
                ),
                step_number=0,
            ),
            ReActStep(
                thought=Thought(content="Step 2", confidence=0.8),
                action={"name": "calculator", "args": {}},
                observation=Observation(
                    tool_name="calculator",
                    tool_input={},
                    tool_output=None,
                    success=False,
                    error="Division by zero",
                    duration_ms=50.0,
                ),
                step_number=1,
            ),
            ReActStep(
                thought=Thought(content="Final"),
                action=None,
                observation=None,
                step_number=2,
            ),
        ]
        metrics = calculate_trace_metrics(trace)
        assert metrics.total_steps == 3
        assert metrics.tool_calls == 2
        assert metrics.successful_tool_calls == 1
        assert metrics.failed_tool_calls == 1
        assert metrics.final_response_steps == 1
        assert metrics.average_confidence == pytest.approx(0.85)
        assert metrics.total_duration_ms == 150.0
        assert metrics.tool_names_used == ["search", "calculator"]


class TestRenderReasoningTrace:
    """Tests for render_reasoning_trace helper."""

    def test_empty_trace(self) -> None:
        """Test rendering empty trace."""
        result = render_reasoning_trace([])
        assert result == "[Empty reasoning trace]"

    def test_basic_render(self) -> None:
        """Test basic rendering."""
        trace = [
            ReActStep(
                thought=Thought(content="I need to search"),
                action={"name": "search", "args": {"query": "test"}},
                observation=Observation(
                    tool_name="search",
                    tool_input={"query": "test"},
                    tool_output="results",
                    success=True,
                    duration_ms=100.0,
                ),
                step_number=0,
            ),
        ]
        result = render_reasoning_trace(trace)
        assert "Step 0:" in result
        assert "I need to search" in result
        assert "search" in result
        assert "success" in result

    def test_render_with_confidence(self) -> None:
        """Test rendering with confidence scores."""
        trace = [
            ReActStep(
                thought=Thought(content="I'm sure", confidence=0.95),
                action=None,
                observation=None,
                step_number=0,
            ),
        ]
        result = render_reasoning_trace(trace)
        assert "95%" in result


class TestFindFailedSteps:
    """Tests for find_failed_steps helper."""

    def test_no_failures(self) -> None:
        """Test trace with no failures."""
        trace = [
            ReActStep(
                thought=None,
                action={"name": "tool", "args": {}},
                observation=Observation(
                    tool_name="tool",
                    tool_input={},
                    tool_output="ok",
                    success=True,
                ),
                step_number=0,
            ),
        ]
        failed = find_failed_steps(trace)
        assert len(failed) == 0

    def test_with_failures(self) -> None:
        """Test trace with failures."""
        trace = [
            ReActStep(
                thought=None,
                action={"name": "good_tool", "args": {}},
                observation=Observation(
                    tool_name="good_tool",
                    tool_input={},
                    tool_output="ok",
                    success=True,
                ),
                step_number=0,
            ),
            ReActStep(
                thought=None,
                action={"name": "bad_tool", "args": {}},
                observation=Observation(
                    tool_name="bad_tool",
                    tool_input={},
                    tool_output=None,
                    success=False,
                    error="Failed!",
                ),
                step_number=1,
            ),
        ]
        failed = find_failed_steps(trace)
        assert len(failed) == 1
        assert failed[0].observation.error == "Failed!"


class TestGetToolUsageSummary:
    """Tests for get_tool_usage_summary helper."""

    def test_summary(self) -> None:
        """Test tool usage summary."""
        trace = [
            ReActStep(
                thought=None,
                action={"name": "search", "args": {}},
                observation=Observation(
                    tool_name="search",
                    tool_input={},
                    tool_output="r1",
                    success=True,
                    duration_ms=100.0,
                ),
                step_number=0,
            ),
            ReActStep(
                thought=None,
                action={"name": "search", "args": {}},
                observation=Observation(
                    tool_name="search",
                    tool_input={},
                    tool_output=None,
                    success=False,
                    duration_ms=50.0,
                ),
                step_number=1,
            ),
            ReActStep(
                thought=None,
                action={"name": "calculator", "args": {}},
                observation=Observation(
                    tool_name="calculator",
                    tool_input={},
                    tool_output=42,
                    success=True,
                    duration_ms=10.0,
                ),
                step_number=2,
            ),
        ]
        summary = get_tool_usage_summary(trace)

        assert "search" in summary
        assert summary["search"]["calls"] == 2
        assert summary["search"]["successes"] == 1
        assert summary["search"]["failures"] == 1
        assert summary["search"]["success_rate"] == 0.5

        assert "calculator" in summary
        assert summary["calculator"]["calls"] == 1
        assert summary["calculator"]["success_rate"] == 1.0


class TestLangGraphIntegration:
    """Integration tests with LangGraph StateGraph."""

    def test_reasoning_trace_in_state_graph(self) -> None:
        """Test that reasoning_trace works as a proper LangGraph state channel."""

        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            reasoning_trace: Annotated[list[ReActStep], add_reasoning_steps]

        def agent_node(state: AgentState) -> dict:
            """Simulates agent deciding to call a tool."""
            step = ReActStep(
                thought=Thought(
                    content="I need to search for information",
                    confidence=0.9,
                ),
                action={"name": "search", "args": {"query": "test"}, "id": "call_1"},
                observation=None,  # Will be filled by tool node
                step_number=len(state.get("reasoning_trace", [])),
            )
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search",
                                "args": {"query": "test"},
                                "id": "call_1",
                            }
                        ],
                    )
                ],
                "reasoning_trace": [step],
            }

        def tool_node(state: AgentState) -> dict:
            """Simulates tool execution and updates reasoning trace."""
            # Get the last step - in a real implementation we'd update it
            trace = state.get("reasoning_trace", [])
            if trace:
                _ = trace[-1]  # Access to verify it exists
                return {
                    "messages": [
                        {
                            "role": "tool",
                            "content": "Search results: found 10 items",
                            "tool_call_id": "call_1",
                        }
                    ],
                    "reasoning_trace": [],  # No new steps in this simple test
                }
            return {"messages": [], "reasoning_trace": []}

        # Build graph
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", "tools")
        graph.add_edge("tools", END)

        app = graph.compile()

        # Run graph
        result = app.invoke(
            {
                "messages": [HumanMessage(content="Search for test")],
                "reasoning_trace": [],
            }
        )

        # Verify reasoning trace was accumulated
        assert "reasoning_trace" in result
        assert len(result["reasoning_trace"]) == 1
        step = result["reasoning_trace"][0]
        assert step.thought.content == "I need to search for information"
        assert step.thought.confidence == 0.9
        assert step.action["name"] == "search"

    def test_multi_step_reasoning_accumulation(self) -> None:
        """Test that multiple steps accumulate correctly across graph iterations."""

        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            reasoning_trace: Annotated[list[ReActStep], add_reasoning_steps]
            iteration: int

        def agent_node(state: AgentState) -> dict:
            """Agent that runs multiple iterations."""
            iteration = state.get("iteration", 0)
            step = ReActStep(
                thought=Thought(content=f"Iteration {iteration} reasoning"),
                action={"name": f"tool_{iteration}", "args": {}},
                observation=Observation(
                    tool_name=f"tool_{iteration}",
                    tool_input={},
                    tool_output=f"Result {iteration}",
                    success=True,
                    duration_ms=50.0 * (iteration + 1),
                ),
                step_number=iteration,
            )
            return {
                "reasoning_trace": [step],
                "iteration": iteration + 1,
            }

        def should_continue(state: AgentState) -> str:
            if state.get("iteration", 0) >= 3:
                return END
            return "agent"

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue)

        app = graph.compile()

        result = app.invoke(
            {
                "messages": [],
                "reasoning_trace": [],
                "iteration": 0,
            }
        )

        # Verify all 3 steps accumulated
        assert len(result["reasoning_trace"]) == 3
        for i, step in enumerate(result["reasoning_trace"]):
            assert step.thought.content == f"Iteration {i} reasoning"
            assert step.step_number == i
            assert step.observation.tool_name == f"tool_{i}"
            assert step.observation.duration_ms == 50.0 * (i + 1)

    def test_full_react_loop_simulation(self) -> None:
        """Simulate a complete ReAct loop: Thought → Action → Observation → Thought → Response."""

        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            reasoning_trace: Annotated[list[ReActStep], add_reasoning_steps]
            phase: str

        def think_node(state: AgentState) -> dict:
            """Agent thinks about what to do."""
            phase = state.get("phase", "initial")

            if phase == "initial":
                thought = Thought(
                    content="User wants weather info. I should call the weather API.",
                    confidence=0.95,
                )
                step = ReActStep(
                    thought=thought,
                    action={
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "id": "w1",
                    },
                    observation=None,
                    step_number=0,
                )
                return {
                    "reasoning_trace": [step],
                    "phase": "called_tool",
                }
            else:
                # After tool call, prepare final response
                thought = Thought(
                    content="Got weather data. Now I can respond to the user.",
                    confidence=0.98,
                )
                step = ReActStep(
                    thought=thought,
                    action=None,  # No more tool calls
                    observation=None,
                    step_number=len(state.get("reasoning_trace", [])),
                )
                return {
                    "reasoning_trace": [step],
                    "phase": "done",
                    "messages": [
                        AIMessage(content="The weather in Paris is 22°C and sunny!")
                    ],
                }

        def act_node(state: AgentState) -> dict:
            """Execute tool and record observation."""
            trace = state.get("reasoning_trace", [])
            if not trace:
                return {"reasoning_trace": []}

            # Find the step that needs observation
            pending_step = trace[-1]
            if pending_step.action and pending_step.observation is None:
                # In a real impl, we'd update the step with observation
                # For this test, we just verify the step exists and move on
                _ = pending_step.action["name"]  # Verify action data
                return {
                    "reasoning_trace": [],
                    "phase": "observed",
                }
            return {"reasoning_trace": []}

        def route(state: AgentState) -> str:
            phase = state.get("phase", "initial")
            if phase == "done":
                return END
            elif phase == "called_tool":
                return "act"
            elif phase == "observed":
                return "think"
            return "think"

        graph = StateGraph(AgentState)
        graph.add_node("think", think_node)
        graph.add_node("act", act_node)
        graph.set_entry_point("think")
        graph.add_conditional_edges("think", route)
        graph.add_conditional_edges("act", route)

        app = graph.compile()

        result = app.invoke(
            {
                "messages": [HumanMessage(content="What's the weather in Paris?")],
                "reasoning_trace": [],
                "phase": "initial",
            }
        )

        # Verify complete ReAct trace
        trace = result["reasoning_trace"]
        assert len(trace) == 2

        # First step: Thought + Action
        assert (
            trace[0].thought.content
            == "User wants weather info. I should call the weather API."
        )
        assert trace[0].action["name"] == "get_weather"
        assert trace[0].action["args"] == {"city": "Paris"}

        # Second step: Final thought (no action)
        assert (
            trace[1].thought.content
            == "Got weather data. Now I can respond to the user."
        )
        assert trace[1].action is None
        assert trace[1].is_final_response()


class TestSerialization:
    """Tests for JSON serialization of reasoning types."""

    def test_thought_to_dict(self) -> None:
        """Test Thought can be converted to dict for serialization."""
        thought = Thought(
            content="I need to search",
            confidence=0.85,
            metadata={"model": "claude-3"},
        )
        # Dataclass provides __dict__ access
        data = {
            "content": thought.content,
            "confidence": thought.confidence,
            "metadata": thought.metadata,
        }
        json_str = json.dumps(data)
        restored = json.loads(json_str)
        assert restored["content"] == "I need to search"
        assert restored["confidence"] == 0.85
        assert restored["metadata"]["model"] == "claude-3"

    def test_observation_to_dict(self) -> None:
        """Test Observation can be converted to dict for serialization."""
        obs = Observation(
            tool_name="search",
            tool_input={"query": "test"},
            tool_output=["result1", "result2"],
            success=True,
            error=None,
            duration_ms=100.5,
        )
        data = {
            "tool_name": obs.tool_name,
            "tool_input": obs.tool_input,
            "tool_output": obs.tool_output,
            "success": obs.success,
            "error": obs.error,
            "duration_ms": obs.duration_ms,
        }
        json_str = json.dumps(data)
        restored = json.loads(json_str)
        assert restored["tool_name"] == "search"
        assert restored["tool_output"] == ["result1", "result2"]
        assert restored["success"] is True

    def test_react_step_to_dict(self) -> None:
        """Test ReActStep can be converted to dict for serialization."""
        step = ReActStep(
            thought=Thought(content="thinking"),
            action={"name": "tool", "args": {"x": 1}},
            observation=Observation(
                tool_name="tool",
                tool_input={"x": 1},
                tool_output="done",
                success=True,
            ),
            step_number=0,
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
        )
        data = {
            "thought": {"content": step.thought.content} if step.thought else None,
            "action": step.action,
            "observation": {
                "tool_name": step.observation.tool_name,
                "success": step.observation.success,
            }
            if step.observation
            else None,
            "step_number": step.step_number,
            "timestamp": step.timestamp.isoformat(),
        }
        json_str = json.dumps(data)
        restored = json.loads(json_str)
        assert restored["thought"]["content"] == "thinking"
        assert restored["action"]["name"] == "tool"
        assert restored["step_number"] == 0


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_empty_thought_content(self) -> None:
        """Test thought with empty content."""
        thought = Thought(content="")
        assert thought.content == ""

    def test_very_long_thought_content(self) -> None:
        """Test thought with very long content."""
        long_content = "x" * 10000
        thought = Thought(content=long_content)
        assert len(thought.content) == 10000

    def test_observation_with_none_output(self) -> None:
        """Test successful observation with None output (valid for some tools)."""
        obs = Observation(
            tool_name="void_tool",
            tool_input={},
            tool_output=None,
            success=True,
        )
        assert obs.success is True
        assert obs.tool_output is None

    def test_react_step_all_none(self) -> None:
        """Test ReActStep with minimal data."""
        step = ReActStep(
            thought=None,
            action=None,
            observation=None,
            step_number=0,
        )
        assert step.is_final_response()
        assert not step.is_tool_call()

    def test_reducer_with_empty_lists(self) -> None:
        """Test reducer handles empty lists correctly."""
        result = add_reasoning_steps([], [])
        assert result == []

    def test_observation_with_complex_error(self) -> None:
        """Test observation with complex error message."""
        obs = Observation(
            tool_name="api",
            tool_input={"url": "https://api.example.com"},
            tool_output=None,
            success=False,
            error="ConnectionError: HTTPSConnectionPool(host='api.example.com', port=443): Max retries exceeded with url: / (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x...>': Failed to establish a new connection: [Errno 61] Connection refused'))",
        )
        assert "ConnectionError" in obs.error
        assert obs.success is False

    def test_multiple_tools_same_name(self) -> None:
        """Test trace with multiple calls to same tool."""
        trace = [
            ReActStep(
                thought=Thought(content=f"Call {i}"),
                action={"name": "search", "args": {"page": i}},
                observation=Observation(
                    tool_name="search",
                    tool_input={"page": i},
                    tool_output=f"Page {i} results",
                    success=True,
                    duration_ms=100.0,
                ),
                step_number=i,
            )
            for i in range(5)
        ]
        summary = get_tool_usage_summary(trace)
        assert summary["search"]["calls"] == 5
        assert summary["search"]["success_rate"] == 1.0
        assert summary["search"]["total_duration_ms"] == 500.0


class TestReasoningCapture:
    """Tests for the ReasoningCapture wrapper class."""

    def test_basic_capture(self) -> None:
        """Test basic tool execution capture."""
        from langgraph.prebuilt import ReasoningCapture

        capture = ReasoningCapture()

        # Simulate a ToolCallRequest-like object
        class MockRequest:
            def __init__(self) -> None:
                self.tool_call = {
                    "name": "search",
                    "args": {"query": "weather"},
                    "id": "call_1",
                }

        class MockToolMessage:
            def __init__(self, content: str) -> None:
                self.content = content

        def mock_execute(request: MockRequest) -> MockToolMessage:
            return MockToolMessage(content="Search results: sunny, 22°C")

        request = MockRequest()
        result = capture.wrap_tool_call(request, mock_execute)

        assert result.content == "Search results: sunny, 22°C"

        steps = capture.get_steps()
        assert len(steps) == 1
        step = steps[0]
        assert step.action["name"] == "search"
        assert step.action["args"] == {"query": "weather"}
        assert step.observation.success is True
        assert step.observation.tool_output == "Search results: sunny, 22°C"
        assert step.observation.duration_ms is not None
        assert step.observation.duration_ms >= 0

    def test_capture_with_thought(self) -> None:
        """Test capturing with a pending thought."""
        from langgraph.prebuilt import ReasoningCapture, Thought

        capture = ReasoningCapture()

        # Set a thought before execution
        capture.set_pending_thought(Thought(content="I need to search for weather"))

        class MockRequest:
            def __init__(self) -> None:
                self.tool_call = {
                    "name": "weather",
                    "args": {"city": "Paris"},
                    "id": "w1",
                }

        class MockToolMessage:
            def __init__(self, content: str) -> None:
                self.content = content

        def mock_execute(request: MockRequest) -> MockToolMessage:
            return MockToolMessage(content="22°C, sunny")

        capture.wrap_tool_call(MockRequest(), mock_execute)

        steps = capture.get_steps()
        assert len(steps) == 1
        assert steps[0].thought is not None
        assert steps[0].thought.content == "I need to search for weather"

    def test_capture_failure(self) -> None:
        """Test capturing a failed tool execution."""
        from langgraph.prebuilt import ReasoningCapture

        capture = ReasoningCapture()

        class MockRequest:
            def __init__(self) -> None:
                self.tool_call = {"name": "api_call", "args": {}, "id": "a1"}

        class MockToolMessage:
            def __init__(self, content: str, status: str = "success") -> None:
                self.content = content
                self.status = status

        def mock_execute(request: MockRequest) -> MockToolMessage:
            return MockToolMessage(content="Error: timeout", status="error")

        capture.wrap_tool_call(MockRequest(), mock_execute)

        steps = capture.get_steps()
        assert len(steps) == 1
        assert steps[0].observation.success is False
        assert "timeout" in steps[0].observation.error

    def test_capture_exception(self) -> None:
        """Test capturing when execute raises an exception."""
        from langgraph.prebuilt import ReasoningCapture

        capture = ReasoningCapture()

        class MockRequest:
            def __init__(self) -> None:
                self.tool_call = {"name": "bad_tool", "args": {}, "id": "b1"}

        def mock_execute(request: MockRequest) -> None:
            raise ValueError("Tool crashed!")

        with pytest.raises(ValueError, match="Tool crashed!"):
            capture.wrap_tool_call(MockRequest(), mock_execute)

        # Even on exception, observation should be captured
        steps = capture.get_steps()
        assert len(steps) == 1
        assert steps[0].observation.success is False
        assert steps[0].observation.error == "Tool crashed!"

    def test_multiple_captures(self) -> None:
        """Test capturing multiple tool calls in sequence."""
        from langgraph.prebuilt import ReasoningCapture

        capture = ReasoningCapture()

        class MockRequest:
            def __init__(self, name: str, call_id: str) -> None:
                self.tool_call = {"name": name, "args": {}, "id": call_id}

        class MockToolMessage:
            def __init__(self, content: str) -> None:
                self.content = content

        def mock_execute(request: MockRequest) -> MockToolMessage:
            return MockToolMessage(content=f"Result from {request.tool_call['name']}")

        # Execute multiple tools
        capture.wrap_tool_call(MockRequest("tool1", "c1"), mock_execute)
        capture.wrap_tool_call(MockRequest("tool2", "c2"), mock_execute)
        capture.wrap_tool_call(MockRequest("tool3", "c3"), mock_execute)

        steps = capture.get_steps()
        assert len(steps) == 3
        assert steps[0].action["name"] == "tool1"
        assert steps[0].step_number == 0
        assert steps[1].action["name"] == "tool2"
        assert steps[1].step_number == 1
        assert steps[2].action["name"] == "tool3"
        assert steps[2].step_number == 2

    def test_clear_resets_state(self) -> None:
        """Test that clear() resets all capture state."""
        from langgraph.prebuilt import ReasoningCapture, Thought

        capture = ReasoningCapture()

        class MockRequest:
            def __init__(self) -> None:
                self.tool_call = {"name": "tool", "args": {}, "id": "t1"}

        class MockToolMessage:
            def __init__(self, content: str) -> None:
                self.content = content

        def mock_execute(request: MockRequest) -> MockToolMessage:
            return MockToolMessage(content="result")

        # Capture some data
        capture.set_pending_thought(Thought(content="test"))
        capture.wrap_tool_call(MockRequest(), mock_execute)
        assert len(capture.get_steps()) == 1

        # Clear and verify
        capture.clear()
        assert len(capture.get_steps()) == 0

        # New captures should start at step 0
        capture.wrap_tool_call(MockRequest(), mock_execute)
        assert capture.get_steps()[0].step_number == 0

    def test_get_steps_returns_copy(self) -> None:
        """Test that get_steps returns a copy, not the internal list."""
        from langgraph.prebuilt import ReasoningCapture

        capture = ReasoningCapture()

        class MockRequest:
            def __init__(self) -> None:
                self.tool_call = {"name": "tool", "args": {}, "id": "t1"}

        class MockToolMessage:
            def __init__(self, content: str) -> None:
                self.content = content

        def mock_execute(request: MockRequest) -> MockToolMessage:
            return MockToolMessage(content="result")

        capture.wrap_tool_call(MockRequest(), mock_execute)

        steps1 = capture.get_steps()
        steps2 = capture.get_steps()

        assert steps1 is not steps2
        assert len(steps1) == len(steps2)

    def test_reasoning_capture_with_tool_node_integration(self) -> None:
        """Integration test: ReasoningCapture with real ToolNode in StateGraph."""
        from langchain_core.tools import tool

        from langgraph.prebuilt import (
            ReasoningCapture,
            ToolNode,
            add_reasoning_steps,
            calculate_trace_metrics,
        )

        # Define a simple tool
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: 22°C, sunny"

        @tool
        def calculate(expression: str) -> str:
            """Calculate a math expression."""
            return f"Result: {eval(expression)}"

        # Create capture and ToolNode
        capture = ReasoningCapture()
        tool_node = ToolNode(
            tools=[get_weather, calculate],
            wrap_tool_call=capture.wrap_tool_call,
        )

        # Define state with reasoning trace
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            reasoning_trace: Annotated[list[ReActStep], add_reasoning_steps]

        def agent_node(state: AgentState) -> dict:
            """Agent that calls tools."""
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_weather",
                                "args": {"city": "Paris"},
                                "id": "call_weather",
                            },
                            {
                                "name": "calculate",
                                "args": {"expression": "2 + 2"},
                                "id": "call_calc",
                            },
                        ],
                    )
                ],
            }

        def tools_with_capture(state: AgentState) -> dict:
            """Execute tools and capture reasoning trace."""
            capture.clear()
            result = tool_node.invoke(state)
            steps = capture.get_steps()
            return {
                "messages": result["messages"],
                "reasoning_trace": steps,
            }

        # Build and run graph
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tools_with_capture)
        graph.set_entry_point("agent")
        graph.add_edge("agent", "tools")
        graph.add_edge("tools", END)

        app = graph.compile()

        result = app.invoke(
            {
                "messages": [HumanMessage(content="What's the weather and 2+2?")],
                "reasoning_trace": [],
            }
        )

        # Verify reasoning trace was captured
        trace = result["reasoning_trace"]
        assert len(trace) == 2

        # Verify first tool call (get_weather)
        weather_step = next(s for s in trace if s.action["name"] == "get_weather")
        assert weather_step.observation.success is True
        assert "22°C" in weather_step.observation.tool_output
        assert weather_step.observation.duration_ms is not None
        assert weather_step.observation.duration_ms >= 0

        # Verify second tool call (calculate)
        calc_step = next(s for s in trace if s.action["name"] == "calculate")
        assert calc_step.observation.success is True
        assert "4" in calc_step.observation.tool_output

        # Verify metrics work on captured trace
        metrics = calculate_trace_metrics(trace)
        assert metrics.total_steps == 2
        assert metrics.tool_calls == 2
        assert metrics.successful_tool_calls == 2
        assert metrics.failed_tool_calls == 0
        assert set(metrics.tool_names_used) == {"get_weather", "calculate"}


class TestCreateReactAgentWithReasoning:
    """Tests for create_react_agent with reasoning=True parameter.

    These tests use FakeToolCallingModel following the repository's standard
    testing pattern for fast, deterministic testing without API calls.

    Note: The reasoning=True feature requires actual tool execution through
    ToolNode's wrap_tool_call mechanism. FakeToolCallingModel only simulates
    tool call messages without executing them, so these tests focus on:
    - State schema correctness
    - Error handling
    - Backward compatibility
    """

    def test_reasoning_true_has_correct_state_schema(self) -> None:
        """Test that reasoning=True creates agent with reasoning_trace in state."""
        from langchain_core.tools import tool
        from tests.model import FakeToolCallingModel

        from langgraph.prebuilt import create_react_agent

        @tool
        def get_weather(city: str) -> str:
            """Get weather."""
            return "22°C"

        model = FakeToolCallingModel(tool_calls=[[]])

        # Create agent with reasoning=True
        agent = create_react_agent(model=model, tools=[get_weather], reasoning=True)

        # Invoke returns reasoning_trace field (even if empty with FakeModel)
        result = agent.invoke({"messages": [HumanMessage("Test")]})
        assert "reasoning_trace" in result
        assert isinstance(result["reasoning_trace"], list)
        assert "messages" in result

    def test_reasoning_false_no_trace_in_state(self) -> None:
        """Test that reasoning=False (default) doesn't add reasoning_trace to state."""
        from langchain_core.tools import tool
        from tests.model import FakeToolCallingModel

        from langgraph.prebuilt import create_react_agent

        @tool
        def get_weather(city: str) -> str:
            """Get weather."""
            return "22°C"

        model = FakeToolCallingModel(tool_calls=[[]])

        # Create agent WITHOUT reasoning (default behavior)
        agent = create_react_agent(model=model, tools=[get_weather])

        # Invoke does not return reasoning_trace
        result = agent.invoke({"messages": [HumanMessage("Test")]})
        assert "reasoning_trace" not in result
        assert "messages" in result

    def test_reasoning_with_response_format_has_both_fields(self) -> None:
        """Test reasoning=True with response_format has both fields in state."""
        from langchain_core.tools import tool
        from pydantic import BaseModel
        from tests.model import FakeToolCallingModel

        from langgraph.prebuilt import create_react_agent

        class WeatherResponse(BaseModel):
            """Structured response."""

            city: str

        @tool
        def get_weather(city: str) -> str:
            """Get weather."""
            return "22°C"

        model = FakeToolCallingModel(
            tool_calls=[[]],
            structured_response=WeatherResponse(city="Paris"),
        )

        # Create agent with BOTH reasoning=True AND response_format
        agent = create_react_agent(
            model=model,
            tools=[get_weather],
            reasoning=True,
            response_format=WeatherResponse,
        )

        # Invoke returns both fields
        result = agent.invoke({"messages": [HumanMessage("Test")]})
        assert "reasoning_trace" in result
        assert "structured_response" in result
        assert "messages" in result

    def test_reasoning_true_rejects_prebuilt_tool_node(self) -> None:
        """Test that reasoning=True raises ValueError for pre-built ToolNode."""
        from langchain_core.tools import tool
        from tests.model import FakeToolCallingModel

        from langgraph.prebuilt import ToolNode, create_react_agent

        @tool
        def get_weather(city: str) -> str:
            """Get weather."""
            return "22°C"

        model = FakeToolCallingModel(tool_calls=[[]])

        # Create a pre-built ToolNode
        tool_node = ToolNode([get_weather])

        # Attempting to use reasoning=True with pre-built ToolNode should fail
        with pytest.raises(
            ValueError,
            match="Cannot use reasoning=True with a pre-built ToolNode",
        ):
            create_react_agent(model=model, tools=tool_node, reasoning=True)

    def test_backward_compatibility_reasoning_false(self) -> None:
        """Test that existing code works unchanged (reasoning defaults to False)."""
        from langchain_core.tools import tool
        from tests.model import FakeToolCallingModel

        from langgraph.prebuilt import create_react_agent

        @tool
        def get_weather(city: str) -> str:
            """Get weather."""
            return "22°C"

        model = FakeToolCallingModel(
            tool_calls=[
                [
                    {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
                [],
            ]
        )

        # Create agent without specifying reasoning parameter
        agent = create_react_agent(model=model, tools=[get_weather])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        # Should work exactly as before - no reasoning_trace
        assert "messages" in result
        assert "reasoning_trace" not in result
        assert len(result["messages"]) >= 2  # Input + at least one response
