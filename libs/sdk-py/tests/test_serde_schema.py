from dataclasses import dataclass
from typing import runtime_checkable

from pydantic import BaseModel

from langgraph_sdk.schema import (
    _BaseModelLike,
    _DataclassLike,
)


def rc(cls: type) -> type:
    return runtime_checkable(cls)


class MyModel(BaseModel):
    foo: str


def test_base_model_like():
    assert isinstance(MyModel(foo="test"), rc(_BaseModelLike))


@dataclass
class MyDataclass:
    foo: str


def test_dataclass_like():
    assert isinstance(MyDataclass(foo="test"), rc(_DataclassLike))
