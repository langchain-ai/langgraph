from dataclasses import dataclass
from operator import add
from typing import Annotated, Any

import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.pregel.remote import RemoteGraph
from langgraph.types import Command, GraphOutput, StreamPart


def test_typed_dict_state() -> None:
    class TypedDictState(TypedDict):
        info: Annotated[list[str], add]

    graph_builder = StateGraph(TypedDictState)

    def valid(state: TypedDictState) -> Any: ...

    def valid_with_config(state: TypedDictState, config: RunnableConfig) -> Any: ...

    def invalid() -> Any: ...

    def invalid_node() -> Any: ...

    graph_builder.add_node("valid", valid)
    graph_builder.add_node("invalid", valid_with_config)
    graph_builder.add_node("invalid_node", invalid_node)  # type: ignore[call-overload]
    graph_builder.set_entry_point("valid")
    graph = graph_builder.compile()

    graph.invoke({"info": ["hello", "world"]})
    graph.invoke({"invalid": "lalala"})  # type: ignore[arg-type]


def test_dataclass_state() -> None:
    @dataclass
    class DataclassState:
        info: Annotated[list[str], add]

    def valid(state: DataclassState) -> Any: ...

    def valid_with_config(state: DataclassState, config: RunnableConfig) -> Any: ...

    def invalid() -> Any: ...

    graph_builder = StateGraph(DataclassState)
    graph_builder.add_node("valid", valid)
    graph_builder.add_node("invalid", valid_with_config)
    graph_builder.add_node("invalid_node", invalid)  # type: ignore[call-overload]

    graph_builder.set_entry_point("valid")
    graph = graph_builder.compile()

    graph.invoke(DataclassState(info=["hello", "world"]))
    graph.invoke({"invalid": 1})  # type: ignore[arg-type]
    graph.invoke({"info": ["hello", "world"]})  # type: ignore[arg-type]


def test_base_model_state() -> None:
    class PydanticState(BaseModel):
        info: Annotated[list[str], add]

    def valid(state: PydanticState) -> Any: ...

    def valid_with_config(state: PydanticState, config: RunnableConfig) -> Any: ...

    def invalid() -> Any: ...

    graph_builder = StateGraph(PydanticState)
    graph_builder.add_node("valid", valid)
    graph_builder.add_node("invalid", valid_with_config)
    graph_builder.add_node("invalid_node", invalid)  # type: ignore[call-overload]

    graph_builder.set_entry_point("valid")
    graph = graph_builder.compile()

    graph.invoke(PydanticState(info=["hello", "world"]))
    graph.invoke({"invalid": 1})  # type: ignore[arg-type]
    graph.invoke({"info": ["hello", "world"]})  # type: ignore[arg-type]


def test_plain_class_not_allowed() -> None:
    class NotAllowed:
        info: Annotated[list[str], add]

    StateGraph(NotAllowed)  # type: ignore[type-var]


def test_input_state_specified() -> None:
    class InputState(TypedDict):
        something: int

    class State(InputState):
        info: Annotated[list[str], add]

    def valid(state: State) -> Any: ...

    new_builder = StateGraph(State, input_schema=InputState)
    new_builder.add_node("valid", valid)
    new_builder.set_entry_point("valid")
    new_graph = new_builder.compile()

    new_graph.invoke({"something": 1})
    new_graph.invoke({"something": 2, "info": ["hello", "world"]})  # type: ignore[arg-type]


@pytest.mark.skip("Purely for type checking")
def test_invoke_with_all_valid_types() -> None:
    class State(TypedDict):
        a: int

    def a(state: State) -> Any: ...

    graph = StateGraph(State).add_node("a", a).set_entry_point("a").compile()
    graph.invoke({"a": 1})
    graph.invoke(None)
    graph.invoke(Command())


def test_add_node_with_explicit_input_schema() -> None:
    class A(TypedDict):
        a1: int
        a2: str

    class B(TypedDict):
        b1: int
        b2: str

    class ANarrow(TypedDict):
        a1: int

    class BNarrow(TypedDict):
        b1: int

    class State(A, B): ...

    def a(state: A) -> Any: ...

    def b(state: B) -> Any: ...

    workflow = StateGraph(State)
    # input schema matches typed schemas
    workflow.add_node("a", a, input_schema=A)
    workflow.add_node("b", b, input_schema=B)

    # input schema does not match typed schemas
    workflow.add_node("a_wrong", a, input_schema=B)  # type: ignore[arg-type]
    workflow.add_node("b_wrong", b, input_schema=A)  # type: ignore[arg-type]

    # input schema is more broad than the typed schemas, which is allowed
    # by the principles of contravariance
    workflow.add_node("a_inclusive", a, input_schema=State)
    workflow.add_node("b_inclusive", b, input_schema=State)

    # input schema is more narrow than the typed schemas, which is not allowed
    # because it violates the principles of contravariance
    workflow.add_node("a_narrow", a, input_schema=ANarrow)  # type: ignore[arg-type]
    workflow.add_node("b_narrow", b, input_schema=BNarrow)  # type: ignore[arg-type]


@pytest.mark.skip("Purely for type checking")
def test_remote_graph_generics_typed_dict() -> None:
    """RemoteGraph parameterized with TypedDict should propagate types."""

    class MyState(TypedDict):
        messages: list[str]

    rg: RemoteGraph[MyState, None, MyState, MyState] = RemoteGraph(
        "test", url="http://localhost:8123"
    )

    # v2 invoke should return GraphOutput[MyState]
    result: GraphOutput[MyState] = rg.invoke({"messages": ["hi"]}, version="v2")
    _val: MyState = result.value

    # v1 invoke should return dict[str, Any] | Any
    _v1_result: dict[str, Any] | Any = rg.invoke({"messages": ["hi"]})

    # v2 stream should yield StreamPart[MyState, MyState]
    for part in rg.stream({"messages": ["hi"]}, version="v2"):
        _part: StreamPart[MyState, MyState] = part

    # input should accept the state type
    rg.invoke({"messages": ["hi"]}, version="v2")

    # input should also accept Command
    rg.invoke(Command(), version="v2")

    # input should also accept None
    rg.invoke(None, version="v2")


@pytest.mark.skip("Purely for type checking")
def test_remote_graph_generics_pydantic() -> None:
    """RemoteGraph parameterized with Pydantic model should propagate types."""

    class PydanticState(BaseModel):
        messages: list[str]

    rg: RemoteGraph[PydanticState, None, PydanticState, PydanticState] = RemoteGraph(
        "test", url="http://localhost:8123"
    )

    result: GraphOutput[PydanticState] = rg.invoke(
        PydanticState(messages=["hi"]), version="v2"
    )
    _val: PydanticState = result.value


@pytest.mark.skip("Purely for type checking")
def test_remote_graph_separate_input_output() -> None:
    """RemoteGraph with different input/output schemas."""

    class InputState(TypedDict):
        query: str

    class OutputState(TypedDict):
        answer: str

    class FullState(InputState, OutputState): ...

    rg: RemoteGraph[FullState, None, InputState, OutputState] = RemoteGraph(
        "test", url="http://localhost:8123"
    )

    result: GraphOutput[OutputState] = rg.invoke({"query": "hi"}, version="v2")
    _val: OutputState = result.value

    # wrong input type should fail type checking
    rg.invoke({"answer": "wrong"}, version="v2")  # type: ignore[call-overload]
