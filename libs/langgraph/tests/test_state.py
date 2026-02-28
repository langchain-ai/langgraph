import inspect
import operator
import warnings
from dataclasses import dataclass, field
from typing import Annotated, Any, Union
from typing import Annotated as Annotated2

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from typing_extensions import NotRequired, Required, TypedDict

from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.state import (
    StateGraph,
    _get_node_name,
    _is_field_channel,
    _warn_invalid_state_schema,
)


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

    class FooState(InputState):
        foo: str

    def complete_hint(state: InputState) -> OutputState:
        return {"input_state": state}

    def miss_first_hint(state, config: RunnableConfig) -> OutputState:
        return {"input_state": state}

    def only_return_hint(state, config) -> OutputState:
        return {"input_state": state}

    def miss_all_hint(state, config):
        return {"input_state": state}

    def pre_foo(_) -> FooState:
        return {"foo": "bar"}

    def pre_bar(_) -> FooState:
        return {"foo": "bar"}

    class Foo:
        def __call__(self, state: FooState) -> OutputState:
            assert state.pop("foo") == "bar"
            return {"input_state": state}

    class Bar:
        def my_node(self, state: FooState) -> OutputState:
            assert state.pop("foo") == "bar"
            return {"input_state": state}

    graph = StateGraph(InputState, output_schema=OutputState)
    actions = [
        complete_hint,
        miss_first_hint,
        only_return_hint,
        miss_all_hint,
        pre_foo,
        Foo(),
        pre_bar,
        Bar().my_node,
    ]

    for action in actions:
        graph.add_node(action)

    def get_name(action) -> str:
        return getattr(action, "__name__", action.__class__.__name__)

    graph.set_entry_point(get_name(actions[0]))
    for i in range(len(actions) - 1):
        graph.add_edge(get_name(actions[i]), get_name(actions[i + 1]))
    graph.set_finish_point(get_name(actions[-1]))

    graph = graph.compile()

    input_state = InputState(question="Hello World!")
    output_state = OutputState(input_state=input_state)
    foo_state = FooState(foo="bar")
    for i, c in enumerate(graph.stream(input_state, stream_mode="updates")):
        node_name = get_name(actions[i])
        if node_name in {"pre_foo", "pre_bar"}:
            assert c[node_name] == foo_state
        else:
            assert c[node_name] == output_state


@pytest.mark.parametrize("total_", [True, False])
def test_state_schema_optional_values(total_: bool):
    class SomeParentState(TypedDict):
        val0a: str
        val0b: str | None

    class InputState(SomeParentState, total=total_):  # type: ignore
        val1: str
        val2: str | None
        val3: Required[Annotated[dict, operator.or_]]
        val4: NotRequired[dict]
        val5: Annotated[Required[str], "foo"]
        val6: Annotated[NotRequired[str], "bar"]

    class OutputState(SomeParentState, total=total_):  # type: ignore
        out_val1: str
        out_val2: str | None
        out_val3: Required[str]
        out_val4: NotRequired[dict]
        out_val5: Annotated[Required[str], "foo"]
        out_val6: Annotated[NotRequired[str], "bar"]

    class State(InputState):  # this would be ignored
        val4: dict

    builder = StateGraph(State, input_schema=InputState, output_schema=OutputState)
    builder.add_node("n", lambda x: x)
    builder.add_edge("__start__", "n")
    graph = builder.compile()
    json_schema = graph.get_input_jsonschema()

    assert isinstance(graph.channels["val3"], BinaryOperatorAggregate)

    if total_ is False:
        expected_required = set()
        expected_optional = {"val2", "val1"}
    else:
        expected_required = {"val1", "val2"}
        expected_optional = set()

    # The others should always have precedence based on the required annotation
    expected_required |= {"val0a", "val0b", "val3", "val5"}
    expected_optional |= {"val4", "val6"}

    assert set(json_schema.get("required", set())) == expected_required
    assert (
        set(json_schema["properties"].keys()) == expected_required | expected_optional
    )

    # Check output schema. Should be the same process
    output_schema = graph.get_output_jsonschema()
    if total_ is False:
        expected_required = set()
        expected_optional = {"out_val2", "out_val1"}
    else:
        expected_required = {"out_val1", "out_val2"}
        expected_optional = set()

    expected_required |= {"val0a", "val0b", "out_val3", "out_val5"}
    expected_optional |= {"out_val4", "out_val6"}

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
        val2: int | None
        val3: Annotated[float | None, "optional annotated"]
        val4: str | None = None
        val5: list[int] = field(default_factory=lambda: [1, 2, 3])
        val6: dict[str, int] = field(default_factory=lambda: {"a": 1})
        val7: str = field(default=...)
        val8: Annotated[int, "some metadata"] = 42
        val9: Annotated[str, "more metadata"] = field(default="some foo")
        val10: str = "default"
        val11: Annotated[list[str], "annotated list"] = field(
            default_factory=lambda: ["a", "b"]
        )

    builder = StateGraph(InputState)
    builder.add_node("n", lambda x: x)
    builder.add_edge("__start__", "n")
    graph = builder.compile()
    for json_schema in [graph.get_input_jsonschema(), graph.get_output_jsonschema()]:
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
        set(json_schema["properties"].keys()) == expected_required | expected_optional
    )


def test__get_node_name() -> None:
    # lambda
    assert _get_node_name(lambda x: x) == "<lambda>"

    # regular function
    def func(state):
        return

    assert _get_node_name(func) == "func"

    class MyClass:
        def __call__(self, state):
            return

        def class_method(self, state):
            return

    # callable class
    assert _get_node_name(MyClass()) == "MyClass"

    # class method
    assert _get_node_name(MyClass().class_method) == "class_method"


def test_input_schema_conditional_edge():
    class OverallState(TypedDict):
        foo: Annotated[int, operator.add]
        bar: str

    class PrivateState(TypedDict):
        baz: str

    builder = StateGraph(OverallState)

    def node_1(state: OverallState):
        return {"foo": 1, "baz": "bar"}

    def node_2(state: PrivateState):
        return {"foo": 1, "bar": state["baz"], "something_else": "meow"}

    def node_3(state: OverallState):
        return {"foo": 1}

    def router(state: OverallState):
        assert state == {"foo": 2, "bar": "bar"}
        if state["foo"] == 2:
            return "node_3"
        else:
            return "__end__"

    builder.add_node(node_1)
    builder.add_node(node_2)
    builder.add_node(node_3)
    builder.add_conditional_edges("node_2", router)
    builder.add_edge("__start__", "node_1")
    builder.add_edge("node_1", "node_2")
    graph = builder.compile()
    assert graph.invoke({"foo": 0}) == {"foo": 3, "bar": "bar"}


def test_private_input_schema_conditional_edge():
    class OverallState(TypedDict):
        foo: Annotated[int, operator.add]
        bar: str

    class RouterState(TypedDict):
        baz: str

    class Node2State(TypedDict):
        foo: Annotated[int, operator.add]
        baz: str

    builder = StateGraph(OverallState)

    def node_1(state: OverallState):
        return {"foo": 1, "baz": "meow"}

    def node_2(state: Node2State):
        return {"foo": 1, "bar": state["baz"]}

    def router(state: RouterState):
        assert state == {"baz": "meow"}
        if state["baz"] == "meow":
            return "node_2"
        else:
            return "__end__"

    builder.add_node(node_1)
    builder.add_node(node_2)
    builder.add_conditional_edges("node_1", router)
    builder.add_edge("__start__", "node_1")
    graph = builder.compile()
    assert graph.invoke({"foo": 0}) == {"foo": 2, "bar": "meow"}


def test_is_field_channel() -> None:
    """Test channel detection across all scenarios."""
    # Basic detection
    result = _is_field_channel(Annotated[int, EphemeralValue])
    assert isinstance(result, EphemeralValue) and result.typ is int

    # Main fix: handles extraneous annotations
    result = _is_field_channel(Annotated[str, "metadata", EphemeralValue, "more"])
    assert isinstance(result, EphemeralValue) and result.typ is str

    # Complex types work
    union_type = Union[int, str]  # noqa: UP007
    result = _is_field_channel(Annotated[union_type, EphemeralValue])
    assert isinstance(result, EphemeralValue) and result.typ is union_type

    # Pre-instantiated channels
    instantiated = EphemeralValue(int)
    result = _is_field_channel(Annotated[int, instantiated])
    assert result is instantiated

    # Pre-instantiated channels with multiple annotations
    instantiated = EphemeralValue(int)
    result = _is_field_channel(Annotated[int, "metadata", instantiated, "more"])
    assert result is instantiated

    # No channel cases
    assert _is_field_channel(int) is None
    assert _is_field_channel(Annotated[int, "just_metadata"]) is None
