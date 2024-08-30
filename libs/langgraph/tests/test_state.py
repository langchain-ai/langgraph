from typing import Annotated as Annotated2
from typing import Any, Optional

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic.v1 import BaseModel
from typing_extensions import Annotated, NotRequired, Required, TypedDict

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


@pytest.mark.parametrize("total_", [True, False])
def test_state_schema_optional_values(total_: bool):
    class SomeParentState(TypedDict):
        val0a: str
        val0b: Optional[str]

    class InputState(SomeParentState, total=total_):  # type: ignore
        val1: str
        val2: Optional[str]
        val3: Required[str]
        val4: NotRequired[dict]
        val5: Annotated[Required[str], "foo"]
        val6: Annotated[NotRequired[str], "bar"]

    class State(InputState):  # this would be ignored
        val4: dict

    builder = StateGraph(State, input=InputState)
    builder.add_node("n", lambda x: x)
    builder.add_edge("__start__", "n")
    graph = builder.compile()
    model = graph.input_schema
    json_schema = model.schema()

    if total_ is False:
        expected_required = set()
        expected_optional = {"val2", "val1"}
    else:
        expected_required = {"val1"}

        expected_optional = {"val2"}

    # The others should always have precedence based on the required annotation
    expected_required |= {"val0a", "val3", "val5"}
    expected_optional |= {"val0b", "val4", "val6"}

    assert set(json_schema.get("required", set())) == expected_required
    assert (
        set(json_schema["properties"].keys()) == expected_required | expected_optional
    )
