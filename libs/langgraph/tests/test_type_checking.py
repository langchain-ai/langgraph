from dataclasses import dataclass
from operator import add
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.graph import StateGraph


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
