import pytest
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from tests.conftest import (
    REGULAR_CHECKPOINTERS_ASYNC,
    REGULAR_CHECKPOINTERS_SYNC,
    awith_checkpointer,
)

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("checkpoint_during", [True, False])
@pytest.mark.parametrize("checkpointer_name", REGULAR_CHECKPOINTERS_SYNC)
def test_interruption_without_state_updates(
    request: pytest.FixtureRequest, checkpointer_name: str, checkpoint_during: bool
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

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    graph = builder.compile(checkpointer=checkpointer, interrupt_after="*")

    initial_input = {"input": "hello world"}
    thread = {"configurable": {"thread_id": "1"}}

    graph.invoke(initial_input, thread, checkpoint_during=checkpoint_during)
    assert graph.get_state(thread).next == ("step_2",)
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (3 if checkpoint_during else 1)

    graph.invoke(None, thread, checkpoint_during=checkpoint_during)
    assert graph.get_state(thread).next == ("step_3",)
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (4 if checkpoint_during else 2)

    graph.invoke(None, thread, checkpoint_during=checkpoint_during)
    assert graph.get_state(thread).next == ()
    n_checkpoints = len([c for c in graph.get_state_history(thread)])
    assert n_checkpoints == (5 if checkpoint_during else 3)


@pytest.mark.parametrize("checkpoint_during", [True, False])
@pytest.mark.parametrize("checkpointer_name", REGULAR_CHECKPOINTERS_ASYNC)
async def test_interruption_without_state_updates_async(
    checkpointer_name: str, checkpoint_during: bool
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

    async with awith_checkpointer(checkpointer_name) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer, interrupt_after="*")

        initial_input = {"input": "hello world"}
        thread = {"configurable": {"thread_id": "1"}}

        await graph.ainvoke(initial_input, thread, checkpoint_during=checkpoint_during)
        assert (await graph.aget_state(thread)).next == ("step_2",)
        n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
        assert n_checkpoints == (3 if checkpoint_during else 1)

        await graph.ainvoke(None, thread, checkpoint_during=checkpoint_during)
        assert (await graph.aget_state(thread)).next == ("step_3",)
        n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
        assert n_checkpoints == (4 if checkpoint_during else 2)

        await graph.ainvoke(None, thread, checkpoint_during=checkpoint_during)
        assert (await graph.aget_state(thread)).next == ()
        n_checkpoints = len([c async for c in graph.aget_state_history(thread)])
        assert n_checkpoints == (5 if checkpoint_during else 3)
