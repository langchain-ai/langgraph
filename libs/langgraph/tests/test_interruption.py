import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Durability

pytestmark = pytest.mark.anyio


def test_interruption_without_state_updates(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    """Test interruption without state updates. This test confirms that
    interrupting doesn't require a state key having been updated in the prev step"""

    class State(TypedDict):
        input: str

    def noop(_state):
        pass

    builder = StateGraph(State)
    builder.add_node("step_1", noop)
    builder.add_node("step_2", noop)
    builder.add_node("step_3", noop)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)

    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_after="*")

    initial_input = {"input": "hello world"}
    thread = {"configurable": {"thread_id": "1"}}

    graph.invoke(initial_input, thread, durability=durability)
    assert graph.get_state(thread).next == ("step_2",)
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (3 if durability != "exit" else 1)

    graph.invoke(None, thread, durability=durability)
    assert graph.get_state(thread).next == ("step_3",)
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (4 if durability != "exit" else 2)

    graph.invoke(None, thread, durability=durability)
    assert graph.get_state(thread).next == ()
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (5 if durability != "exit" else 3)


async def test_interruption_without_state_updates_async(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    """Test interruption without state updates. This test confirms that
    interrupting doesn't require a state key having been updated in the prev step"""

    class State(TypedDict):
        input: str

    async def noop(_state):
        pass

    builder = StateGraph(State)
    builder.add_node("step_1", noop)
    builder.add_node("step_2", noop)
    builder.add_node("step_3", noop)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)

    graph = builder.compile(checkpointer=async_checkpointer, interrupt_after="*")

    initial_input = {"input": "hello world"}
    thread = {"configurable": {"thread_id": "1"}}

    await graph.ainvoke(initial_input, thread, durability=durability)
    assert (await graph.aget_state(thread)).next == ("step_2",)
    n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
    assert n_checkpoints == (3 if durability != "exit" else 1)

    await graph.ainvoke(None, thread, durability=durability)
    assert (await graph.aget_state(thread)).next == ("step_3",)
    n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
    assert n_checkpoints == (4 if durability != "exit" else 2)

    await graph.ainvoke(None, thread, durability=durability)
    assert (await graph.aget_state(thread)).next == ()
    n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
    assert n_checkpoints == (5 if durability != "exit" else 3)


def test_multiple_interrupts_in_node_have_unique_ids(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test that multiple interrupt() calls within the same node get unique IDs.

    This is a regression test for https://github.com/langchain-ai/langgraph/issues/6626
    where multiple interrupt() calls in the same namespace would get identical IDs,
    making it impossible to resume each interrupt with its own value.
    """
    from uuid import uuid4

    from langgraph.types import Command, Interrupt, interrupt

    class State(TypedDict):
        value: str
        results: list[str]

    def node_with_multiple_interrupts(state: State):
        """A node that calls interrupt() multiple times."""
        # Each interrupt should get a unique ID
        response_a = interrupt({"question": "First question?"})
        response_b = interrupt({"question": "Second question?"})
        response_c = interrupt({"question": "Third question?"})
        return {"results": [response_a, response_b, response_c]}

    graph = StateGraph(State)
    graph.add_node("ask", node_with_multiple_interrupts)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)

    workflow = graph.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}

    # First invocation - triggers first interrupt
    workflow.invoke({"value": "start", "results": []}, config)

    state = workflow.get_state(config)
    all_interrupts: list[Interrupt] = []
    for task in state.tasks:
        if task.interrupts:
            all_interrupts.extend(task.interrupts)

    # First run should have 1 interrupt (first interrupt() call)
    assert len(all_interrupts) == 1
    first_interrupt = all_interrupts[0]
    assert first_interrupt.value == {"question": "First question?"}

    # Resume first interrupt
    workflow.invoke(Command(resume="answer_1"), config)

    state = workflow.get_state(config)
    all_interrupts = []
    for task in state.tasks:
        if task.interrupts:
            all_interrupts.extend(task.interrupts)

    # Second run should have 1 interrupt (second interrupt() call)
    assert len(all_interrupts) == 1
    second_interrupt = all_interrupts[0]
    assert second_interrupt.value == {"question": "Second question?"}

    # CRITICAL: The second interrupt should have a DIFFERENT ID than the first
    # This is the fix for issue #6626
    assert first_interrupt.id != second_interrupt.id, (
        f"Interrupt IDs should be unique! "
        f"First: {first_interrupt.id}, Second: {second_interrupt.id}"
    )

    # Resume second interrupt
    workflow.invoke(Command(resume="answer_2"), config)

    state = workflow.get_state(config)
    all_interrupts = []
    for task in state.tasks:
        if task.interrupts:
            all_interrupts.extend(task.interrupts)

    # Third run should have 1 interrupt (third interrupt() call)
    assert len(all_interrupts) == 1
    third_interrupt = all_interrupts[0]
    assert third_interrupt.value == {"question": "Third question?"}

    # All three interrupts should have unique IDs
    assert third_interrupt.id != first_interrupt.id
    assert third_interrupt.id != second_interrupt.id

    # Resume third interrupt and complete
    result = workflow.invoke(Command(resume="answer_3"), config)

    # Verify all responses were collected correctly
    assert result["results"] == ["answer_1", "answer_2", "answer_3"]


def test_parallel_tool_interrupts_have_unique_ids(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Integration test for issue #6626: parallel tools with interrupts.

    This test verifies that when multiple tools are called in parallel via Send(),
    each interrupt gets a unique ID, and we can resume using Command(resume={id: value}).

    This follows the same pattern as test_parallel_interrupts in test_pregel.py.
    """
    import operator
    import re
    from typing import Annotated
    from uuid import uuid4

    from pydantic import BaseModel, Field

    from langgraph.types import Command, Interrupt, Send, interrupt

    # Child graph that represents a tool requiring human approval
    class ToolState(BaseModel):
        tool_name: str = Field(..., description="Name of the tool")
        approval: str | None = Field(None, description="Human approval response")
        results: Annotated[list[str], operator.add] = Field(
            default_factory=list, description="Results to pass back to parent"
        )

    def execute_tool(state: ToolState):
        """Execute a tool that requires human approval."""
        approval = interrupt({"tool": state.tool_name})
        return {
            "approval": approval,
            "results": [f"{state.tool_name}: {approval}"],
        }

    child_builder = StateGraph(ToolState)
    child_builder.add_node("execute", execute_tool)
    child_builder.add_edge(START, "execute")
    child_builder.add_edge("execute", END)
    child_graph = child_builder.compile()

    # Parent graph that dispatches multiple tools in parallel
    class ParentState(BaseModel):
        tool_names: list[str] = Field(..., description="List of tool names")
        results: Annotated[list[str], operator.add] = Field(
            default_factory=list, description="Results from all tools"
        )

    def dispatch_tools(state: ParentState):
        return [
            Send("tool_executor", ToolState(tool_name=name))
            for name in state.tool_names
        ]

    def collect_results(state: ParentState):
        assert len(state.results) == len(state.tool_names)

    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("tool_executor", child_graph)
    parent_builder.add_node("collect", collect_results)
    parent_builder.add_conditional_edges(START, dispatch_tools, ["tool_executor"])
    parent_builder.add_edge("tool_executor", "collect")
    parent_builder.add_edge("collect", END)

    workflow = parent_builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}

    current_input: dict | Command = {"tool_names": ["tool_a", "tool_b"], "results": []}
    resumed_interrupt_ids: set[str] = set()
    all_interrupt_ids: set[str] = set()

    invokes = 0
    while invokes < 10:
        invokes += 1
        current_interrupts: list[Interrupt] = []

        # Stream to collect interrupt events
        for event in workflow.stream(current_input, config, stream_mode="updates"):
            if "__interrupt__" in event:
                current_interrupts.extend(event["__interrupt__"])

        if current_interrupts:
            # Collect all seen interrupt IDs (may see same one multiple times)
            for intr in current_interrupts:
                # Verify ID has correct format: {32-hex-chars}:{idx}
                assert re.match(r"[0-9a-f]{32}:\d+", intr.id), (
                    f"Interrupt ID has unexpected format: {intr.id}"
                )
                all_interrupt_ids.add(intr.id)

            # Resume one interrupt at a time using ID-based resume
            intr = current_interrupts[0]
            tool_name = intr.value.get("tool", "unknown")

            # Track which IDs we've resumed
            assert intr.id not in resumed_interrupt_ids, (
                f"Already resumed interrupt {intr.id}"
            )
            resumed_interrupt_ids.add(intr.id)

            current_input = Command(resume={intr.id: f"approved_{tool_name}"})
        else:
            break
    else:
        raise AssertionError("Detected infinite loop")

    # Should complete in 3 invocations (initial + 2 resumes for 2 tools)
    assert invokes == 3, f"Expected 3 invocations, got {invokes}"

    # CRITICAL TEST: Verify we saw 2 UNIQUE interrupt IDs (one per parallel tool)
    # This is the fix for issue #6626 - before the fix, both tools would have the same ID
    assert len(all_interrupt_ids) == 2, (
        f"Expected 2 unique interrupt IDs, got {len(all_interrupt_ids)}: {all_interrupt_ids}"
    )

    # Verify we resumed both
    assert len(resumed_interrupt_ids) == 2

    # Verify final results - each tool got its correct resume value
    final_state = workflow.get_state(config)
    assert final_state.next == ()
    results = set(final_state.values["results"])
    assert "tool_a: approved_tool_a" in results
    assert "tool_b: approved_tool_b" in results
