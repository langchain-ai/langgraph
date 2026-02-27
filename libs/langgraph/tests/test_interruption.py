import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Durability
from langgraph.types import interrupt as lg_interrupt

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


def test_get_state_next_after_double_interrupt(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Regression test for #6956.

    get_state().next must not be empty after resuming from the first of two
    consecutive interrupt() calls inside the same node.
    """

    def ask_twice(state):
        lg_interrupt("first question")
        lg_interrupt("second question")

    builder = StateGraph(dict)
    builder.add_node("ask_twice", ask_twice)
    builder.add_edge(START, "ask_twice")
    builder.add_edge("ask_twice", END)

    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "double-interrupt-sync"}}

    # First invoke hits the first interrupt
    graph.invoke({}, config)
    assert graph.get_state(config).next == ("ask_twice",)

    # Resume from first interrupt — hits the second interrupt
    graph.invoke(Command(resume="ans1"), config)
    # Must still show the node as pending, not an empty tuple
    assert graph.get_state(config).next == ("ask_twice",)

    # Resume from second interrupt — node completes
    graph.invoke(Command(resume="ans2"), config)
    assert graph.get_state(config).next == ()


async def test_get_state_next_after_double_interrupt_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    """Async regression test for #6956."""

    async def ask_twice(state):
        lg_interrupt("first question")
        lg_interrupt("second question")

    builder = StateGraph(dict)
    builder.add_node("ask_twice", ask_twice)
    builder.add_edge(START, "ask_twice")
    builder.add_edge("ask_twice", END)

    graph = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "double-interrupt-async"}}

    await graph.ainvoke({}, config)
    assert (await graph.aget_state(config)).next == ("ask_twice",)

    await graph.ainvoke(Command(resume="ans1"), config)
    assert (await graph.aget_state(config)).next == ("ask_twice",)

    await graph.ainvoke(Command(resume="ans2"), config)
    assert (await graph.aget_state(config)).next == ()
