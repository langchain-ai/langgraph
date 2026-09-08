import operator
from typing import Annotated

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


def test_consumed_resume_not_redelivered_on_null_invoke(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    class S(TypedDict):
        log: Annotated[list[str], operator.add]

    def node(state: S) -> dict:
        first = interrupt("i1")
        second = interrupt("i2")
        return {"log": [f"got:{first!r}+{second!r}"]}

    builder = StateGraph(S)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    app = builder.compile(checkpointer=sync_checkpointer)
    thread = {"configurable": {"thread_id": "test_stale_resume"}}

    # Step 1: Initial invoke pauses at i1
    app.invoke({"log": []}, thread)
    state = app.get_state(thread)
    assert [i.value for t in state.tasks for i in t.interrupts] == ["i1"]

    # Step 2: Resume i1 with 'first'; pauses at i2
    app.invoke(Command(resume="first"), thread)
    state = app.get_state(thread)
    assert [i.value for t in state.tasks for i in t.interrupts] == ["i2"]

    # Step 3: Call invoke(None); i2 must remain pending and NOT be auto-answered
    app.invoke(None, thread)
    state = app.get_state(thread)
    assert [i.value for t in state.tasks for i in t.interrupts] == ["i2"]
    assert state.values.get("log") == [] or "log" not in state.values

    # Step 4: Resume i2 with 'second'; execution finishes
    app.invoke(Command(resume="second"), thread)
    state = app.get_state(thread)
    assert state.next == ()
    assert state.values["log"] == ["got:'first'+'second'"]
