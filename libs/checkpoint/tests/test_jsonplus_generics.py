"""Round-tripping parametrized pydantic v2 generics through msgpack (#6102)."""

from typing import Generic, TypeVar

import ormsgpack
import pytest
from pydantic import BaseModel

from langgraph.checkpoint.serde import _msgpack as _lg_msgpack
from langgraph.checkpoint.serde.jsonplus import (
    EXT_PYDANTIC_V2,
    JsonPlusSerializer,
    _msgpack_enc,
    _warned_blocked_types,
    _warned_unregistered_types,
)

T = TypeVar("T")
U = TypeVar("U")

MOD = __name__


class Inner(BaseModel):
    hello: str


class Other(BaseModel):
    n: int


class Box(BaseModel, Generic[T]):
    inner: T


class Pair(BaseModel, Generic[T, U]):
    left: T
    right: U


# Parametrizations are built inside functions on purpose: at module scope
# (including in annotations) pydantic binds ``Box[Inner]`` as a module
# attribute, which makes the legacy name lookup succeed and hides the bug.
def make_box():  # no return annotation, see above
    return Box[Inner](inner=Inner(hello="hi"))


def test_parametrization_is_not_a_module_attribute() -> None:
    """Guard the premise of this file: the legacy lookup must not succeed."""
    make_box()
    assert "Box[Inner]" not in globals()


@pytest.fixture(autouse=True)
def _reset_state():
    strict = _lg_msgpack.STRICT_MSGPACK_ENABLED
    _warned_unregistered_types.clear()
    _warned_blocked_types.clear()
    yield
    _lg_msgpack.STRICT_MSGPACK_ENABLED = strict


def roundtrip(serde: JsonPlusSerializer, obj: object) -> object:
    return serde.loads_typed(serde.dumps_typed(obj))


def test_generic_roundtrip_default_mode() -> None:
    _lg_msgpack.STRICT_MSGPACK_ENABLED = False
    obj = make_box()
    result = roundtrip(JsonPlusSerializer(), obj)
    assert type(result) is type(obj)
    assert result == obj


def test_generic_roundtrip_strict_with_origin_and_args_allowed() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=[(MOD, "Box"), (MOD, "Inner")])
    obj = make_box()
    result = roundtrip(serde, obj)
    assert type(result) is type(obj)
    assert result == obj


def test_generic_blocked_when_type_arg_not_allowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=[(MOD, "Box")])
    obj = make_box()
    result = roundtrip(serde, obj)
    assert result == obj.model_dump()
    assert f"{MOD}.Inner" in caplog.text


def test_generic_blocked_when_origin_not_allowed() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=[(MOD, "Inner")])
    obj = make_box()
    assert roundtrip(serde, obj) == obj.model_dump()


def test_nested_and_multi_arg_generics() -> None:
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            (MOD, "Box"),
            (MOD, "Pair"),
            (MOD, "Inner"),
            (MOD, "Other"),
        ]
    )
    obj = Box[Pair[Inner, Other]](
        inner=Pair[Inner, Other](left=Inner(hello="a"), right=Other(n=1))
    )
    result = roundtrip(serde, obj)
    assert type(result) is type(obj)
    assert type(result.inner) is type(obj.inner)
    assert result == obj


def test_builtin_type_args_need_no_allowlist_entry() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=[(MOD, "Box")])
    obj = Box[int](inner=3)
    result = roundtrip(serde, obj)
    assert type(result) is type(obj)
    assert result == obj


def test_legacy_payload_without_generic_spec_still_loads() -> None:
    """Checkpoints written before this change carry only ``module, name``."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=[(MOD, "Inner")])
    legacy = ormsgpack.packb(
        ormsgpack.Ext(
            EXT_PYDANTIC_V2,
            _msgpack_enc((MOD, "Inner", {"hello": "hi"}, "model_validate_json")),
        )
    )
    assert serde.loads_typed(("msgpack", legacy)) == Inner(hello="hi")


def test_non_generic_payload_is_unchanged() -> None:
    """Plain models keep the 4-field encoding, so old readers are unaffected."""
    _, data = JsonPlusSerializer().dumps_typed(Inner(hello="hi"))
    ext = ormsgpack.unpackb(data, ext_hook=lambda code, raw: ormsgpack.unpackb(raw))
    assert len(ext) == 4
