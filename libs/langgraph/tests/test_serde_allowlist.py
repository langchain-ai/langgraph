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
    curated_core_allowlist,
)


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


def test_curated_core_allowlist_includes_messages() -> None:
    try:
        from langchain_core.messages import BaseMessage
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
