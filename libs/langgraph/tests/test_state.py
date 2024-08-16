from typing import Annotated as Annotated2
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
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


def test_state_schema_with_type_hint():
    class InputState(TypedDict):
        question: str

    class OutputState(TypedDict):
        input_state: InputState

    def complete_hint(state: InputState) -> OutputState:
        return {"input_state": state}

    def miss_first_hint(state, config: RunnableConfig) -> OutputState:
        return {"input_state": state}

    def only_return_hint(state, config) -> OutputState:
        return {"input_state": state}

    def miss_all_hint(state, config):
        return {"input_state": state}

    graph = StateGraph(input=InputState, output=OutputState)
    actions = [complete_hint, miss_first_hint, only_return_hint, miss_all_hint]

    for action in actions:
        graph.add_node(action)

    graph.set_entry_point(actions[0].__name__)
    for i in range(len(actions) - 1):
        graph.add_edge(actions[i].__name__, actions[i + 1].__name__)
    graph.set_finish_point(actions[-1].__name__)

    graph = graph.compile()

    input_state = InputState(question="Hello World!")
    output_state = OutputState(input_state=input_state)
    for i, c in enumerate(graph.stream(input_state, stream_mode="updates")):
        node_name = actions[i].__name__
        assert c[node_name] == output_state
