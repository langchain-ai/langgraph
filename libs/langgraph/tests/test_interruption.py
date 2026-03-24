import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Durability, interrupt

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


def test_command_resume_none_no_unbound_error(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Regression test for https://github.com/langchain-ai/langgraph/issues/7034.

    Command(resume=None) used to raise UnboundLocalError because `resume_is_map`
    was only assigned inside the `if resume is not None` block but referenced
    unconditionally afterwards.  It should instead raise EmptyInputError (or
    succeed) without an UnboundLocalError.
    """
    from langgraph.errors import EmptyInputError

    class State(TypedDict):
        result: str

    def checkpoint_node(state: State):
        interrupt(None)
        return {}

    def work_node(state: State):
        return {"result": "done"}

    builder = StateGraph(State)
    builder.add_node("checkpoint", checkpoint_node)
    builder.add_node("work", work_node)
    builder.add_edge(START, "checkpoint")
    builder.add_edge("checkpoint", "work")
    builder.add_edge("work", END)

    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "resume-none-test"}}

    # First invocation — runs until the interrupt
    graph.invoke({"result": ""}, config)

    # Second invocation with Command(resume=None) — must NOT raise UnboundLocalError.
    # resume=None is the sentinel for "no resume value provided", so it should
    # raise EmptyInputError rather than an unrelated UnboundLocalError.
    with pytest.raises(EmptyInputError):
        graph.invoke(Command(resume=None), config)


async def test_command_resume_none_no_unbound_error_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Async variant of test_command_resume_none_no_unbound_error (issue #7034)."""
    from langgraph.errors import EmptyInputError

    class State(TypedDict):
        result: str

    async def checkpoint_node(state: State):
        interrupt(None)
        return {}

    async def work_node(state: State):
        return {"result": "done"}

    builder = StateGraph(State)
    builder.add_node("checkpoint", checkpoint_node)
    builder.add_node("work", work_node)
    builder.add_edge(START, "checkpoint")
    builder.add_edge("checkpoint", "work")
    builder.add_edge("work", END)

    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "resume-none-test-async"}}

    await graph.ainvoke({"result": ""}, config)

    with pytest.raises(EmptyInputError):
        await graph.ainvoke(Command(resume=None), config)
