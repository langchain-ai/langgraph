from typing import TypedDict

import pytest

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from tests.conftest import DEFAULT_POSTGRES_URI
from tests.memory_assert import MemorySaverAssertImmutable


@pytest.mark.parametrize(
    "checkpointer",
    [
        MemorySaverAssertImmutable(),
        SqliteSaver.from_conn_string(":memory:"),
        PostgresSaver.from_conn_string(DEFAULT_POSTGRES_URI),
        PostgresSaver.from_conn_string(DEFAULT_POSTGRES_URI, pipeline=True),
    ],
    ids=["memory", "sqlite", "postgres", "postgres_pipeline"],
)
def test_interruption_without_state_updates(checkpointer: BaseCheckpointSaver):
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

    with checkpointer as checkpointer:
        graph = builder.compile(checkpointer=checkpointer, interrupt_after="*")

        initial_input = {"input": "hello world"}
        thread = {"configurable": {"thread_id": "1"}}

        graph.invoke(initial_input, thread, debug=True)
        assert graph.get_state(thread).next == ("step_2",)

        graph.invoke(None, thread, debug=True)
        assert graph.get_state(thread).next == ("step_3",)

        graph.invoke(None, thread, debug=True)
        assert graph.get_state(thread).next == ()


@pytest.mark.parametrize(
    "checkpointer",
    [
        MemorySaverAssertImmutable(),
        AsyncSqliteSaver.from_conn_string(":memory:"),
        AsyncPostgresSaver.from_conn_string(DEFAULT_POSTGRES_URI),
        AsyncPostgresSaver.from_conn_string(DEFAULT_POSTGRES_URI, pipeline=True),
    ],
    ids=["memory", "sqlite", "postgres", "postgres_pipeline"],
)
async def test_interruption_without_state_updates_async(
    checkpointer: BaseCheckpointSaver,
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

    async with checkpointer as checkpointer:
        graph = builder.compile(checkpointer=checkpointer, interrupt_after="*")

        initial_input = {"input": "hello world"}
        thread = {"configurable": {"thread_id": "1"}}

        await graph.ainvoke(initial_input, thread, debug=True)
        assert (await graph.aget_state(thread)).next == ("step_2",)

        await graph.ainvoke(None, thread, debug=True)
        assert (await graph.aget_state(thread)).next == ("step_3",)

        await graph.ainvoke(None, thread, debug=True)
        assert (await graph.aget_state(thread)).next == ()
