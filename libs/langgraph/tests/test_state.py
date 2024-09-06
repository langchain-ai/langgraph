import inspect
import warnings
from dataclasses import dataclass, field
from typing import Annotated as Annotated2
from typing import Any, Optional

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic.v1 import BaseModel
from typing_extensions import Annotated, NotRequired, Required, TypedDict

from langgraph.graph.state import StateGraph, _warn_invalid_state_schema
from langgraph.managed.shared_value import SharedValue


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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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

    class OutputState(SomeParentState, total=total_):  # type: ignore
        out_val1: str
        out_val2: Optional[str]
        out_val3: Required[str]
        out_val4: NotRequired[dict]
        out_val5: Annotated[Required[str], "foo"]
        out_val6: Annotated[NotRequired[str], "bar"]

    class State(InputState):  # this would be ignored
        val4: dict
        some_shared_channel: Annotated[str, SharedValue.on("assistant_id")] = field(
            default="foo"
        )

    builder = StateGraph(State, input=InputState, output=OutputState)
    builder.add_node("n", lambda x: x)
    builder.add_edge("__start__", "n")
    graph = builder.compile()
    model = graph.get_input_schema()
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

    # Check output schema. Should be the same process
    output_schema = graph.get_output_schema().schema()
    if total_ is False:
        expected_required = set()
        expected_optional = {"out_val2", "out_val1"}
    else:
        expected_required = {"out_val1"}
        expected_optional = {"out_val2"}

    expected_required |= {"val0a", "out_val3", "out_val5"}
    expected_optional |= {"val0b", "out_val4", "out_val6"}

    assert set(output_schema.get("required", set())) == expected_required
    assert (
        set(output_schema["properties"].keys()) == expected_required | expected_optional
    )


@pytest.mark.parametrize("kw_only_", [False, True])
def test_state_schema_default_values(kw_only_: bool):
    kwargs = {}
    if "kw_only" in inspect.signature(dataclass).parameters:
        kwargs = {"kw_only": kw_only_}

    @dataclass(**kwargs)
    class InputState:
        val1: str
        val2: Optional[int]
        val3: Annotated[Optional[float], "optional annotated"]
        val4: Optional[str] = None
        val5: list[int] = field(default_factory=lambda: [1, 2, 3])
        val6: dict[str, int] = field(default_factory=lambda: {"a": 1})
        val7: str = field(default=...)
        val8: Annotated[int, "some metadata"] = 42
        val9: Annotated[str, "more metadata"] = field(default="some foo")
        val10: str = "default"
        val11: Annotated[list[str], "annotated list"] = field(
            default_factory=lambda: ["a", "b"]
        )
        some_shared_channel: Annotated[str, SharedValue.on("assistant_id")] = field(
            default="foo"
        )

    builder = StateGraph(InputState)
    builder.add_node("n", lambda x: x)
    builder.add_edge("__start__", "n")
    graph = builder.compile()
    for model in [graph.get_input_schema(), graph.get_output_schema()]:
        json_schema = model.schema()

        expected_required = {"val1", "val7"}
        expected_optional = {
            "val2",
            "val3",
            "val4",
            "val5",
            "val6",
            "val8",
            "val9",
            "val10",
            "val11",
        }

        assert set(json_schema.get("required", set())) == expected_required
        assert (
            set(json_schema["properties"].keys())
            == expected_required | expected_optional
        )
