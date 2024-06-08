from typing import Annotated as Annotated2
from typing import Any

import pytest
from pydantic.v1 import BaseModel
from typing_extensions import Annotated, TypedDict

from langgraph.graph.state import StateGraph, _warn_invalid_state_schema


class State(BaseModel):
    foo: str
    bar: int


class State2(TypedDict):
    foo: str
    bar: int


@pytest.mark.parametrize(
    "schema",
    [
        {"foo": "bar"},
        ["hi", lambda x, y: x + y],
        State(foo="bar", bar=1),
        State2(foo="bar", bar=1),
    ],
)
def test_warns_invalid_schema(schema: Any):
    with pytest.warns(UserWarning):
        _warn_invalid_state_schema(schema)


@pytest.mark.parametrize(
    "schema",
    [
        Annotated[dict, lambda x, y: y],
        Annotated2[list, lambda x, y: y],
        dict,
        State,
        State2,
    ],
)
def test_doesnt_warn_valid_schema(schema: Any):
    # Assert the function does not raise a warning
    with pytest.warns(None):
        _warn_invalid_state_schema(schema)


def test_unknown_start_raises_error():
    graph = StateGraph(State)
    graph.add_node("start", lambda x: x)
    graph.add_edge("__start__", "start")
    graph.add_edge("unknown", "start")
    graph.add_edge("start", "__end__")
    with pytest.raises(ValueError, match="Found edge starting at unknown node "):
        graph.compile()


def test_unset_end_accepted():
    graph = StateGraph(State)
    graph.add_node("start", lambda x: x)
    graph.add_edge("__start__", "start")
    graph.add_edge("unknown", "start")
    graph.compile()
