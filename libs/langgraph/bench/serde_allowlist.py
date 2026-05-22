from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict

from langgraph._internal._serde import collect_allowlist_from_schemas


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class InnerDataclass:
    value: int


class InnerModel(BaseModel):
    name: str


class InnerTyped(TypedDict):
    payload: InnerDataclass
    optional: NotRequired[InnerModel]


@dataclass
class Node:
    value: int
    child: Node | None = None


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


class DummyChannel:
    @property
    def ValueType(self) -> type[InnerDataclass]:
        return InnerDataclass

    @property
    def UpdateType(self) -> type[InnerModel]:
        return InnerModel


SCHEMAS_SMALL = [InnerDataclass, InnerModel, Color]
SCHEMAS_LARGE = [
    InnerDataclass,
    InnerModel,
    Color,
    InnerTyped,
    Node,
    NestedDataclass,
    AnnotatedList,
]
CHANNELS = {"a": DummyChannel(), "b": DummyChannel()}


def collect_allowlist_small() -> None:
    collect_allowlist_from_schemas(schemas=SCHEMAS_SMALL, channels=CHANNELS)


def collect_allowlist_large() -> None:
    collect_allowlist_from_schemas(schemas=SCHEMAS_LARGE, channels=CHANNELS)
