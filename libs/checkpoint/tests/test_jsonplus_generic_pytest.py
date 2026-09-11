"""Regression tests for GH #6102: JsonPlusSerializer round-trips of
parameterized pydantic v2 generic models."""

from typing import Generic, TypeVar

import ormsgpack
from pydantic import BaseModel

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

TInner = TypeVar("TInner", bound=BaseModel)
TKey = TypeVar("TKey")


class MyModel(BaseModel):
    hello: str


class MyGeneric(BaseModel, Generic[TInner]):
    inner: TInner


class NestedGeneric(BaseModel, Generic[TInner]):
    items: list[TInner]


class TwoParamGeneric(BaseModel, Generic[TKey, TInner]):
    key: str
    inner: TInner


class OptionalGeneric(BaseModel, Generic[TInner]):
    inner: TInner | None


def test_roundtrip_parameterized_generic():
    instance = MyGeneric[MyModel](inner=MyModel(hello="hello"))
    serde = JsonPlusSerializer()
    dumped = serde.dumps_typed(instance)
    assert dumped[0] == "msgpack"
    result = serde.loads_typed(dumped)
    assert instance == result


def test_roundtrip_nested_generic():
    instance = NestedGeneric[MyModel](items=[MyModel(hello="a"), MyModel(hello="b")])
    serde = JsonPlusSerializer()
    result = serde.loads_typed(serde.dumps_typed(instance))
    assert instance == result
    assert all(isinstance(item, MyModel) for item in result.items)


def test_roundtrip_two_param_generic():
    instance = TwoParamGeneric[str, MyModel](key="k", inner=MyModel(hello="x"))
    serde = JsonPlusSerializer()
    result = serde.loads_typed(serde.dumps_typed(instance))
    assert instance == result


def test_roundtrip_optional_generic():
    instance = OptionalGeneric[MyModel](inner=MyModel(hello="y"))
    serde = JsonPlusSerializer()
    result = serde.loads_typed(serde.dumps_typed(instance))
    assert instance == result


def test_plain_model_still_roundtrips():
    instance = MyModel(hello="plain")
    serde = JsonPlusSerializer()
    result = serde.loads_typed(serde.dumps_typed(instance))
    assert instance == result


def test_unresolvable_param_degrades_to_dict():
    """If a type arg can't be resolved (class renamed since encode time),
    we can't rebuild the parameterized class safely; degrade to the raw
    kwargs dict, exactly like any other type we can't reconstruct."""
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("tests.test_jsonplus_generic_pytest", "MyGeneric[DoesNotExist]")
        ]
    )
    instance = MyGeneric[MyModel](inner=MyModel(hello="z"))
    dumped = serde.dumps_typed(instance)
    assert dumped[0] == "msgpack"
    raw = ormsgpack.unpackb(dumped[1], ext_hook=lambda code, data: (code, data))
    code, payload = raw
    tup = ormsgpack.unpackb(payload)
    assert code == 5  # EXT_PYDANTIC_V2
    broken = ormsgpack.packb((tup[0], "MyGeneric[DoesNotExist]", tup[2], tup[3]))
    revived = ormsgpack.unpackb(
        ormsgpack.packb(ormsgpack.Ext(code, broken)), ext_hook=serde._unpack_ext_hook
    )
    # Same graceful degradation as unknown non-generic types: raw kwargs.
    assert revived == {"inner": {"hello": "z"}}
