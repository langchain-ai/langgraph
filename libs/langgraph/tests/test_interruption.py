from typing import TypedDict

import pytest
from pytest_mock import MockerFixture

from langgraph.graph import END, START, StateGraph
from tests.conftest import (
    ALL_CHECKPOINTERS_ASYNC,
    ALL_CHECKPOINTERS_SYNC,
    awith_checkpointer,
)

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_interruption_without_state_updates(
    request: pytest.FixtureRequest, checkpointer_name: str, mocker: MockerFixture
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

    graph.invoke(initial_input, thread, debug=True)
    assert graph.get_state(thread).next == ("step_2",)

    graph.invoke(None, thread, debug=True)
    assert graph.get_state(thread).next == ("step_3",)

    graph.invoke(None, thread, debug=True)
    assert graph.get_state(thread).next == ()


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_interruption_without_state_updates_async(
    checkpointer_name: str, mocker: MockerFixture
):
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

        await graph.ainvoke(initial_input, thread, debug=True)
        assert (await graph.aget_state(thread)).next == ("step_2",)

        await graph.ainvoke(None, thread, debug=True)
        assert (await graph.aget_state(thread)).next == ("step_3",)

        await graph.ainvoke(None, thread, debug=True)
        assert (await graph.aget_state(thread)).next == ()
