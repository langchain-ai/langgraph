"""Regression test for #7034: UnboundLocalError on resume_is_map
when using Command(resume=None).

Command(resume=None) must raise EmptyInputError, not UnboundLocalError.
"""

import pytest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import EmptyInputError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(dict):
    result: str


def test_command_resume_none_raises_empty_input_error() -> None:
    """Command(resume=None) should raise EmptyInputError, not crash."""

    def checkpoint_node(state: State) -> dict:
        interrupt(None)
        return {}

    def work_node(state: State) -> dict:
        return {"result": "done"}

    builder = StateGraph(State)
    builder.add_node("checkpoint", checkpoint_node)
    builder.add_node("work", work_node)
    builder.add_edge(START, "checkpoint")
    builder.add_edge("checkpoint", "work")
    builder.add_edge("work", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"result": ""}, config)

    with pytest.raises(EmptyInputError):
        graph.invoke(Command(resume=None), config)
