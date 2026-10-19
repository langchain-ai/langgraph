from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, NewType, Optional, Union

import pytest
from pydantic import BaseModel
from typing_extensions import NotRequired, Required, TypedDict

from langgraph._internal._serde import (
    collect_allowlist_from_schemas,
    collect_control_types_from_callables,
    curated_core_allowlist,
)
from langgraph.types import Send


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class InnerDataclass:
    value: int


class InnerModel(BaseModel):
    name: str


@dataclass
class Node:
    value: int
    child: Node | None = None


if TYPE_CHECKING:

    class MissingType:
        pass


@dataclass
class MissingRefDataclass:
    payload: MissingType


class Payload(TypedDict):
    item: InnerDataclass
    maybe: NotRequired[InnerModel]
    required: Required[str]


@dataclass
class NestedDataclass:
    inner: InnerDataclass
    items: list[InnerModel]
    mapping: dict[str, InnerDataclass]
    optional: InnerModel | None
    union: InnerDataclass | InnerModel
    queue: deque[InnerDataclass]
    frozen: frozenset[InnerModel]


AnnotatedList = Annotated[list[InnerDataclass], "meta"]
UserId = NewType("UserId", int)


class DummyChannel:
    @property
    def ValueType(self) -> type[InnerDataclass]:
        return InnerDataclass

    @property
    def UpdateType(self) -> type[InnerModel]:
        return InnerModel


class FanOutState(TypedDict):
    xs: list[str]
    out: list[str]


def test_curated_core_allowlist_includes_messages() -> None:
    try:
        from langchain_core.messages import BaseMessage  # noqa: PLC0415
    except Exception:
        pytest.skip("langchain_core not available")
    allowlist = curated_core_allowlist()
    assert (BaseMessage.__module__, BaseMessage.__name__) in allowlist


def test_collect_allowlist_basic_models() -> None:
    allowlist = collect_allowlist_from_schemas(
        schemas=[InnerDataclass, InnerModel, Color]
    )
    assert (InnerDataclass.__module__, InnerDataclass.__name__) in allowlist
    assert (InnerModel.__module__, InnerModel.__name__) in allowlist
    assert (Color.__module__, Color.__name__) in allowlist


def test_collect_allowlist_nested_containers() -> None:
    allowlist = collect_allowlist_from_schemas(schemas=[NestedDataclass])
    assert (NestedDataclass.__module__, NestedDataclass.__name__) in allowlist
    assert (InnerDataclass.__module__, InnerDataclass.__name__) in allowlist
    assert (InnerModel.__module__, InnerModel.__name__) in allowlist


def test_collect_allowlist_annotated_and_union() -> None:
    allowlist = collect_allowlist_from_schemas(
        schemas=[AnnotatedList, InnerModel | None, InnerDataclass | None]
    )
    assert (InnerDataclass.__module__, InnerDataclass.__name__) in allowlist
    assert (InnerModel.__module__, InnerModel.__name__) in allowlist


def test_collect_allowlist_literal_and_any() -> None:
    allowlist = collect_allowlist_from_schemas(schemas=[Any, Literal["a"]])
    assert allowlist == set()


def test_collect_allowlist_typeddict_fields_only() -> None:
    allowlist = collect_allowlist_from_schemas(schemas=[Payload])
    assert (InnerDataclass.__module__, InnerDataclass.__name__) in allowlist
    assert (InnerModel.__module__, InnerModel.__name__) in allowlist
    assert (Payload.__module__, Payload.__name__) not in allowlist


def test_collect_allowlist_forward_refs() -> None:
    allowlist = collect_allowlist_from_schemas(schemas=[Node])
    assert (Node.__module__, Node.__name__) in allowlist


def test_collect_allowlist_missing_forward_ref() -> None:
    allowlist = collect_allowlist_from_schemas(schemas=[MissingRefDataclass])
    assert allowlist == {(MissingRefDataclass.__module__, MissingRefDataclass.__name__)}


def test_collect_allowlist_newtype_supertype() -> None:
    allowlist = collect_allowlist_from_schemas(schemas=[UserId])
    assert allowlist == set()


def test_collect_allowlist_channels() -> None:
    channels = {"a": DummyChannel(), "b": DummyChannel()}
    allowlist = collect_allowlist_from_schemas(channels=channels)
    assert (InnerDataclass.__module__, InnerDataclass.__name__) in allowlist
    assert (InnerModel.__module__, InnerModel.__name__) in allowlist


def test_collect_allowlist_pep604_union() -> None:
    schema = InnerDataclass | InnerModel
    allowlist = collect_allowlist_from_schemas(schemas=[schema])
    assert (InnerDataclass.__module__, InnerDataclass.__name__) in allowlist
    assert (InnerModel.__module__, InnerModel.__name__) in allowlist


def test_collect_allowlist_typing_union_optional() -> None:
    typing_optional = Optional[InnerDataclass]  # noqa: UP045
    typing_union = Union[InnerDataclass, InnerModel]  # noqa: UP007
    allowlist = collect_allowlist_from_schemas(schemas=[typing_optional, typing_union])
    assert (InnerDataclass.__module__, InnerDataclass.__name__) in allowlist
    assert (InnerModel.__module__, InnerModel.__name__) in allowlist


# --- control-type detection for the compile-time derived allowlist ---


def test_control_types_detected_from_router() -> None:
    from langgraph.types import (  # noqa: PLC0415
        Command,
        Send,
    )

    def fan_out(state: Any) -> list[Send]:
        return [Send("work", {"k": v}) for v in state["xs"]]

    assert collect_control_types_from_callables([fan_out]) == {
        ("langgraph.types", "Send")
    }

    def both(state: Any) -> list[Any]:
        return [Send("2", Command(goto=Send("2", 3)))]

    assert collect_control_types_from_callables([both]) == {
        ("langgraph.types", "Send"),
        ("langgraph.types", "Command"),
    }


def test_control_types_detected_from_module_attribute_access() -> None:
    import langgraph.types as lt  # noqa: PLC0415

    def fan_out(state: Any) -> list[Any]:
        return [lt.Send("work", 1)]

    assert collect_control_types_from_callables([fan_out]) == {
        ("langgraph.types", "Send")
    }


def test_control_types_detected_from_alias_and_closure() -> None:
    from langgraph.types import Send as FanOut  # noqa: PLC0415

    def make_router():
        def fan_out(state: Any) -> list[Any]:
            return [FanOut("work", 1)]

        return fan_out

    assert collect_control_types_from_callables([make_router()]) == {
        ("langgraph.types", "Send")
    }


def test_control_types_not_detected_from_plain_callables() -> None:
    def plain(state: Any) -> dict:
        return {"a": state["a"] + 1}

    class CallableNode:
        def __call__(self, state: Any) -> dict:
            return {"a": 1}

    assert collect_control_types_from_callables([plain, CallableNode()]) == set()
    assert collect_control_types_from_callables(None) == set()


def test_control_types_detected_through_compiled_graph() -> None:
    from langgraph.graph import END, START, StateGraph  # noqa: PLC0415

    def router(state: FanOutState) -> list[Send]:
        return [Send("work", x) for x in state["xs"]]

    def work(state: Any) -> dict:
        return {"out": [state]}

    builder = StateGraph(FanOutState)
    builder.add_node("source", lambda state: {"xs": state["xs"]})
    builder.add_node("work", work)
    builder.add_edge(START, "source")
    builder.add_conditional_edges("source", router)
    builder.add_edge("work", END)
    graph = builder.compile()

    sources = [spec.runnable for spec in builder.nodes.values()] + [
        branch.path
        for branch_specs in builder.branches.values()
        for branch in branch_specs.values()
    ]
    assert collect_control_types_from_callables(sources) == {
        ("langgraph.types", "Send")
    }
    # the compiled graph object itself is a valid scan source (subgraph nodes)
    assert ("langgraph.types", "Send") in collect_control_types_from_callables([graph])
