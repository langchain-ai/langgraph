import dataclasses
import json
import logging
import pathlib
import re
import sys
import uuid
from collections import deque
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address
from zoneinfo import ZoneInfo

import dataclasses_json
import numpy as np
import ormsgpack
import pandas as pd
import pytest
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, SecretStr
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import SecretStr as SecretStrV1

from langgraph.checkpoint.serde import _msgpack as _lg_msgpack
from langgraph.checkpoint.serde._msgpack import AllowedMsgpackModules
from langgraph.checkpoint.serde.event_hooks import (
    SerdeEvent,
    register_serde_event_listener,
)
from langgraph.checkpoint.serde.jsonplus import (
    EXT_METHOD_SINGLE_ARG,
    InvalidModuleError,
    JsonPlusSerializer,
    _msgpack_enc,
    _msgpack_ext_hook_to_json,
)
from langgraph.store.base import Item


class InnerPydantic(BaseModel):
    hello: str


class MyPydantic(BaseModel):
    foo: str
    bar: int
    inner: InnerPydantic


class AnotherPydantic(BaseModel):
    foo: str


class InnerPydanticV1(BaseModelV1):
    hello: str


class MyPydanticV1(BaseModelV1):
    foo: str
    bar: int
    inner: InnerPydanticV1


@dataclasses.dataclass
class InnerDataclass:
    hello: str


@dataclasses.dataclass
class MyDataclass:
    foo: str
    bar: int
    inner: InnerDataclass

    def something(self) -> None:
        pass


@dataclasses.dataclass(slots=True)
class MyDataclassWSlots:
    foo: str
    bar: int
    inner: InnerDataclass

    def something(self) -> None:
        pass


class MyEnum(Enum):
    FOO = "foo"
    BAR = "bar"


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Person:
    name: str


def test_serde_jsonplus() -> None:
    uid = uuid.UUID(int=1)
    deque_instance = deque([1, 2, 3])
    tzn = ZoneInfo("America/New_York")
    ip4 = IPv4Address("192.168.0.1")
    current_date = date(2024, 4, 19)
    current_time = time(23, 4, 57, 51022, timezone.max)
    current_timestamp = datetime(2024, 4, 19, 23, 4, 57, 51022, timezone.max)

    to_serialize = {
        "path": pathlib.Path("foo", "bar"),
        "re": re.compile(r"foo", re.DOTALL),
        "decimal": Decimal("1.10101"),
        "set": {1, 2, frozenset({1, 2})},
        "frozen_set": frozenset({1, 2, 3}),
        "ip4": ip4,
        "deque": deque_instance,
        "tzn": tzn,
        "date": current_date,
        "time": current_time,
        "uid": uid,
        "timestamp": current_timestamp,
        "my_rich_dict": {(1, 2, 3): 45},
        "my_slotted_class": MyDataclassWSlots("bar", 2, InnerDataclass("hello")),
        "my_dataclass": MyDataclass("foo", 1, InnerDataclass("hello")),
        "my_enum": MyEnum.FOO,
        "my_pydantic": MyPydantic(foo="foo", bar=1, inner=InnerPydantic(hello="hello")),
        "my_secret_str": SecretStr("meow"),
        "person": Person(name="foo"),
        "a_bool": True,
        "a_none": None,
        "a_str": "foo",
        "a_str_nuc": "foo\u0000",
        "a_str_uc": "foo ⛰️",
        "a_str_ucuc": "foo \u26f0\ufe0f\u0000",
        "a_str_ucucuc": "foo \\u26f0\\ufe0f",
        "an_int": 1,
        "a_float": 1.1,
        "a_bytes": b"my bytes",
        "a_bytearray": bytearray([42]),
        "my_item": Item(
            value={},
            key="my-key",
            namespace=("a", "name", " "),
            created_at=datetime(2024, 9, 24, 17, 29, 10, 128397),
            updated_at=datetime(2024, 9, 24, 17, 29, 11, 128397),
        ),
    }

    if sys.version_info < (3, 14):
        to_serialize["my_pydantic_v1"] = MyPydanticV1(
            foo="foo", bar=1, inner=InnerPydanticV1(hello="hello")
        )
        to_serialize["my_secret_str_v1"] = SecretStrV1("meow")

    allowed_msgpack_modules: AllowedMsgpackModules = [
        InnerDataclass,
        MyDataclass,
        MyDataclassWSlots,
        MyEnum,
        InnerPydantic,
        MyPydantic,
        # Testing that it supports both.
        (Person.__module__, Person.__name__),
        (SecretStr.__module__, SecretStr.__name__),
    ]
    if sys.version_info < (3, 14):
        allowed_msgpack_modules.extend(  # type: ignore
            [
                (InnerPydanticV1.__module__, InnerPydanticV1.__name__),
                (MyPydanticV1.__module__, MyPydanticV1.__name__),
                (SecretStrV1.__module__, SecretStrV1.__name__),
            ]
        )

    serde = JsonPlusSerializer(allowed_msgpack_modules=allowed_msgpack_modules)

    dumped = serde.dumps_typed(to_serialize)

    assert dumped[0] == "msgpack"
    assert serde.loads_typed(dumped) == to_serialize

    for value in to_serialize.values():
        assert serde.loads_typed(serde.dumps_typed(value)) == value

    surrogates = [
        "Hello??",
        "Python??",
        "Surrogate??",
        "Example??",
        "String??",
        "With??",
        "Surrogates??",
        "Embedded??",
        "In??",
        "The??",
        "Text??",
        "收花🙄·到",
    ]
    serde = JsonPlusSerializer(pickle_fallback=False)

    assert serde.loads_typed(serde.dumps_typed(surrogates)) == surrogates


def test_serde_jsonplus_json_mode() -> None:
    uid = uuid.UUID(int=1)
    deque_instance = deque([1, 2, 3])
    tzn = ZoneInfo("America/New_York")
    ip4 = IPv4Address("192.168.0.1")
    current_date = date(2024, 4, 19)
    current_time = time(23, 4, 57, 51022, timezone.max)
    current_timestamp = datetime(2024, 4, 19, 23, 4, 57, 51022, timezone.max)

    to_serialize = {
        "path": pathlib.Path("foo", "bar"),
        "re": re.compile(r"foo", re.DOTALL),
        "decimal": Decimal("1.10101"),
        "set": {1, 2, frozenset({1, 2})},
        "frozen_set": frozenset({1, 2, 3}),
        "ip4": ip4,
        "deque": deque_instance,
        "tzn": tzn,
        "date": current_date,
        "time": current_time,
        "uid": uid,
        "timestamp": current_timestamp,
        "my_slotted_class": MyDataclassWSlots("bar", 2, InnerDataclass("hello")),
        "my_dataclass": MyDataclass("foo", 1, InnerDataclass("hello")),
        "my_enum": MyEnum.FOO,
        "my_pydantic": MyPydantic(foo="foo", bar=1, inner=InnerPydantic(hello="hello")),
        "my_secret_str": SecretStr("meow"),
        "person": Person(name="foo"),
        "a_bool": True,
        "a_none": None,
        "a_str": "foo",
        "a_str_nuc": "foo\u0000",
        "a_str_uc": "foo ⛰️",
        "a_str_ucuc": "foo \u26f0\ufe0f\u0000",
        "a_str_ucucuc": "foo \\u26f0\\ufe0f",
        "an_int": 1,
        "a_float": 1.1,
        "a_bytes": b"my bytes",
        "a_bytearray": bytearray([42]),
        "my_item": Item(
            value={},
            key="my-key",
            namespace=("a", "name", " "),
            created_at=datetime(2024, 9, 24, 17, 29, 10, 128397),
            updated_at=datetime(2024, 9, 24, 17, 29, 11, 128397),
        ),
    }

    if sys.version_info < (3, 14):
        to_serialize["my_pydantic_v1"] = MyPydanticV1(
            foo="foo", bar=1, inner=InnerPydanticV1(hello="hello")
        )
        to_serialize["my_secret_str_v1"] = SecretStrV1("meow")

    serde = JsonPlusSerializer(__unpack_ext_hook__=_msgpack_ext_hook_to_json)

    dumped = serde.dumps_typed(to_serialize)

    assert dumped[0] == "msgpack"
    result = serde.loads_typed(dumped)

    expected_result = {
        "path": ["foo", "bar"],
        "re": ["foo", 48],
        "decimal": "1.10101",
        "set": [1, 2, [1, 2]],
        "frozen_set": [1, 2, 3],
        "ip4": "192.168.0.1",
        "deque": [1, 2, 3],
        "tzn": "America/New_York",
        "date": [2024, 4, 19],
        "time": {
            "hour": 23,
            "minute": 4,
            "second": 57,
            "microsecond": 51022,
            "tzinfo": [[0, 86340, 0]],
            "fold": 0,
        },
        "uid": "00000000-0000-0000-0000-000000000001",
        "timestamp": "2024-04-19T23:04:57.051022+23:59",
        "my_slotted_class": {"foo": "bar", "bar": 2, "inner": {"hello": "hello"}},
        "my_dataclass": {"foo": "foo", "bar": 1, "inner": {"hello": "hello"}},
        "my_enum": "foo",
        "my_pydantic": {"foo": "foo", "bar": 1, "inner": {"hello": "hello"}},
        "my_secret_str": "meow",
        "person": {"name": "foo"},
        "a_bool": True,
        "a_none": None,
        "a_str": "foo",
        "a_str_nuc": "foo\x00",
        "a_str_uc": "foo ⛰️",
        "a_str_ucuc": "foo ⛰️\x00",
        "a_str_ucucuc": "foo \\u26f0\\ufe0f",
        "an_int": 1,
        "a_float": 1.1,
        "a_bytes": b"my bytes",
        "a_bytearray": b"*",
        "my_item": {
            "namespace": ["a", "name", " "],
            "key": "my-key",
            "value": {},
            "created_at": "2024-09-24T17:29:10.128397",
            "updated_at": "2024-09-24T17:29:11.128397",
        },
    }

    if sys.version_info < (3, 14):
        expected_result["my_pydantic_v1"] = {
            "foo": "foo",
            "bar": 1,
            "inner": {"hello": "hello"},
        }
        expected_result["my_secret_str_v1"] = "meow"

    assert result == expected_result


def test_serde_jsonplus_bytes() -> None:
    serde = JsonPlusSerializer()

    some_bytes = b"my bytes"
    dumped = serde.dumps_typed(some_bytes)

    assert dumped == ("bytes", some_bytes)
    assert serde.loads_typed(dumped) == some_bytes


def test_deserde_invalid_module() -> None:
    serde = JsonPlusSerializer()
    load = {
        "lc": 2,
        "type": "constructor",
        "id": ["pprint", "pprint"],
        "kwargs": {"object": "HELLO"},
    }
    with pytest.raises(InvalidModuleError):
        serde._revive_lc2(load)
    serde = JsonPlusSerializer(allowed_json_modules=[("pprint", "pprint")])
    serde.loads_typed(("json", json.dumps(load).encode("utf-8")))


def test_serde_jsonplus_bytearray() -> None:
    serde = JsonPlusSerializer()

    some_bytearray = bytearray([42])
    dumped = serde.dumps_typed(some_bytearray)

    assert dumped == ("bytearray", some_bytearray)
    assert serde.loads_typed(dumped) == some_bytearray


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(9, dtype=np.int32).reshape(3, 3),
        np.asfortranarray(np.arange(9, dtype=np.float64).reshape(3, 3)),
        np.arange(12, dtype=np.int16)[::2].reshape(3, 2),
    ],
)
def test_serde_jsonplus_numpy_array(arr: np.ndarray) -> None:
    serde = JsonPlusSerializer()

    dumped = serde.dumps_typed(arr)
    assert dumped[0] == "msgpack"
    result = serde.loads_typed(dumped)
    assert isinstance(result, np.ndarray)
    assert result.dtype == arr.dtype
    assert np.array_equal(result, arr)


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(6, dtype=np.float32).reshape(2, 3),
        np.asfortranarray(np.arange(4, dtype=np.complex128).reshape(2, 2)),
    ],
)
def test_serde_jsonplus_numpy_array_json_hook(arr: np.ndarray) -> None:
    serde = JsonPlusSerializer(__unpack_ext_hook__=_msgpack_ext_hook_to_json)
    dumped = serde.dumps_typed(arr)
    assert dumped[0] == "msgpack"
    result = serde.loads_typed(dumped)
    assert isinstance(result, list)
    assert result == arr.tolist()


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame({"int_col": [1, 2, 3]}),
        pd.DataFrame({"float_col": [1.1, 2.2, 3.3]}),
        pd.DataFrame({"str_col": ["a", "b", "c"]}),
        pd.DataFrame({"bool_col": [True, False, True]}),
        pd.DataFrame(
            {
                "datetime_col": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                ]
            }
        ),
        pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
            }
        ),
        pd.DataFrame(
            {
                "int_col": [1, 2, None],
                "float_col": [1.1, None, 3.3],
                "str_col": ["a", None, "c"],
            }
        ),
        pytest.param(
            pd.DataFrame({"cat_col": pd.Categorical(["a", "b", "a", "c"])}),
            marks=pytest.mark.skipif(
                sys.version_info >= (3, 14), reason="NotImplementedError on Python 3.14"
            ),
        ),
        pd.DataFrame(
            {
                "int8": pd.array([1, 2, 3], dtype="int8"),
                "int16": pd.array([10, 20, 30], dtype="int16"),
                "int32": pd.array([100, 200, 300], dtype="int32"),
                "int64": pd.array([1000, 2000, 3000], dtype="int64"),
                "float32": pd.array([1.1, 2.2, 3.3], dtype="float32"),
                "float64": pd.array([10.1, 20.2, 30.3], dtype="float64"),
            }
        ),
        pd.DataFrame({"value": [1, 2, 3]}, index=["x", "y", "z"]),
        pd.DataFrame(
            [[1, 2, 3, 4]],
            columns=pd.MultiIndex.from_tuples(
                [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")]
            ),
        ),
        pd.DataFrame(
            {"value": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="D")
        ),
        pd.DataFrame(
            {
                "col1": range(1000),
                "col2": [f"str_{i}" for i in range(1000)],
                "col3": np.random.rand(1000),
            }
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "tz_datetime": pd.date_range(
                        "2024-01-01", periods=3, freq="D", tz="UTC"
                    )
                }
            ),
            marks=pytest.mark.skipif(
                sys.version_info >= (3, 14), reason="NotImplementedError on Python 3.14"
            ),
        ),
        pd.DataFrame({"timedelta": pd.to_timedelta([1, 2, 3], unit="D")}),
        pytest.param(
            pd.DataFrame({"period": pd.period_range("2024-01", periods=3, freq="M")}),
            marks=pytest.mark.skipif(
                sys.version_info >= (3, 14), reason="NotImplementedError on Python 3.14"
            ),
        ),
        pd.DataFrame({"interval": pd.interval_range(start=0, end=3, periods=3)}),
        pd.DataFrame({"unicode": ["Hello 🌍", "Python 🐍", "Data 📊"]}),
        pd.DataFrame({"mixed": [1, "string", [1, 2, 3], {"key": "value"}]}),
        pd.DataFrame({"a": [1], "b": ["test"], "c": [3.14]}),
        pd.DataFrame({"single": [42]}),
        pd.DataFrame(
            {
                "small": [sys.float_info.min, 0, sys.float_info.max],
                "large_int": [-(2**63), 0, 2**63 - 1],
            }
        ),
        pd.DataFrame({"special_strings": ["", "null", "None", "NaN", "inf", "-inf"]}),
        pd.DataFrame({"bytes_col": [b"hello", b"world", b"\x00\x01\x02"]}),
    ],
)
def test_serde_jsonplus_pandas_dataframe(df: pd.DataFrame) -> None:
    serde = JsonPlusSerializer(pickle_fallback=True)

    dumped = serde.dumps_typed(df)
    assert dumped[0] == "pickle"
    result = serde.loads_typed(dumped)
    assert result.equals(df)


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([]),
        pd.Series([1, 2, 3]),
        pd.Series([1.1, 2.2, 3.3]),
        pd.Series(["a", "b", "c"]),
        pd.Series([True, False, True]),
        pd.Series([datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]),
        pd.Series([1, 2, None]),
        pd.Series([1.1, None, 3.3]),
        pd.Series(["a", None, "c"]),
        pytest.param(
            pd.Series(pd.Categorical(["a", "b", "a", "c"])),
            marks=pytest.mark.skipif(
                sys.version_info >= (3, 14), reason="NotImplementedError on Python 3.14"
            ),
        ),
        pd.Series([1, 2, 3], dtype="int8"),
        pd.Series([10, 20, 30], dtype="int16"),
        pd.Series([100, 200, 300], dtype="int32"),
        pd.Series([1000, 2000, 3000], dtype="int64"),
        pd.Series([1.1, 2.2, 3.3], dtype="float32"),
        pd.Series([10.1, 20.2, 30.3], dtype="float64"),
        pd.Series([1, 2, 3], index=["x", "y", "z"]),
        pd.Series([1, 2, 3], index=pd.date_range("2024-01-01", periods=3, freq="D")),
        pd.Series(range(1000)),
        pd.Series(pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")),
        pd.Series(pd.to_timedelta([1, 2, 3], unit="D")),
        pd.Series(pd.period_range("2024-01", periods=3, freq="M")),
        pd.Series(pd.interval_range(start=0, end=3, periods=3)),
        pd.Series(["Hello 🌍", "Python 🐍", "Data 📊"]),
        pd.Series([1, "string", [1, 2, 3], {"key": "value"}]),
        pd.Series([42], name="single"),
        pd.Series([sys.float_info.min, 0, sys.float_info.max]),
        pd.Series([-(2**63), 0, 2**63 - 1]),
        pd.Series(["", "null", "None", "NaN", "inf", "-inf"]),
        pd.Series([b"hello", b"world", b"\x00\x01\x02"]),
        pd.Series([1, 2, 3], name="named_series"),
        pd.Series(
            [10, 20],
            index=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["x", "y"]),
        ),
    ],
)
def test_serde_jsonplus_pandas_series(series: pd.Series) -> None:
    serde = JsonPlusSerializer(pickle_fallback=True)
    dumped = serde.dumps_typed(series)

    assert dumped[0] == "pickle"
    result = serde.loads_typed(dumped)
    assert result.equals(series)


def test_msgpack_safe_types_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test safe types deserialize without warnings."""

    serde = JsonPlusSerializer()

    safe_objects = [
        datetime.now(),
        date.today(),
        time(12, 30),
        timezone.utc,
        uuid.uuid4(),
        Decimal("123.45"),
        {1, 2, 3},
        frozenset([1, 2, 3]),
        deque([1, 2, 3]),
        IPv4Address("192.168.1.1"),
        pathlib.Path("/tmp/test"),
    ]

    for obj in safe_objects:
        caplog.clear()
        dumped = serde.dumps_typed(obj)
        result = serde.loads_typed(dumped)
        assert "unregistered type" not in caplog.text.lower(), (
            f"Unexpected warning for {type(obj)}"
        )
        assert result is not None


def test_msgpack_pydantic_warns_by_default(caplog: pytest.LogCaptureFixture) -> None:
    """Pydantic models not in allowlist should log warning but still deserialize."""
    current = _lg_msgpack.STRICT_MSGPACK_ENABLED
    _lg_msgpack.STRICT_MSGPACK_ENABLED = False
    serde = JsonPlusSerializer()

    obj = MyPydantic(foo="test", bar=42, inner=InnerPydantic(hello="world"))

    caplog.clear()
    dumped = serde.dumps_typed(obj)
    result = serde.loads_typed(dumped)

    assert "unregistered type" in caplog.text.lower()
    assert "allowed_msgpack_modules" in caplog.text
    assert result == obj
    _lg_msgpack.STRICT_MSGPACK_ENABLED = current


def test_msgpack_env_strict_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Strict msgpack env should default to blocking unregistered types."""
    current = _lg_msgpack.STRICT_MSGPACK_ENABLED
    _lg_msgpack.STRICT_MSGPACK_ENABLED = True
    serde = JsonPlusSerializer()

    obj = MyPydantic(foo="test", bar=42, inner=InnerPydantic(hello="world"))

    caplog.clear()
    dumped = serde.dumps_typed(obj)
    result = serde.loads_typed(dumped)

    assert "blocked" in caplog.text.lower()
    assert result == obj.model_dump()
    _lg_msgpack.STRICT_MSGPACK_ENABLED = current


def test_msgpack_allowlist_silences_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Types in allowed_msgpack_modules should deserialize without warnings."""

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("tests.test_jsonplus", "MyPydantic"),
            ("tests.test_jsonplus", "InnerPydantic"),
        ]
    )

    obj = MyPydantic(foo="test", bar=42, inner=InnerPydantic(hello="world"))

    caplog.clear()
    dumped = serde.dumps_typed(obj)
    result = serde.loads_typed(dumped)

    assert "unregistered type" not in caplog.text.lower()
    assert result == obj


def test_msgpack_none_blocks_unregistered(caplog: pytest.LogCaptureFixture) -> None:
    """allowed_msgpack_modules=None should block unregistered types."""

    serde = JsonPlusSerializer(allowed_msgpack_modules=None)

    obj = MyPydantic(foo="test", bar=42, inner=InnerPydantic(hello="world"))

    caplog.clear()
    dumped = serde.dumps_typed(obj)
    result = serde.loads_typed(dumped)

    assert "blocked" in caplog.text.lower()
    expected = obj.model_dump()
    assert result == expected


def test_msgpack_allowlist_blocks_non_listed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Allowlists should block unregistered types even if msgpack is enabled."""

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("tests.test_jsonplus", "MyPydantic")]
    )

    obj = AnotherPydantic(foo="nope")

    caplog.clear()
    dumped = serde.dumps_typed(obj)
    result = serde.loads_typed(dumped)

    assert "blocked" in caplog.text.lower()
    expected = obj.model_dump()
    # It's not allowed, so we just leave it as a dict
    assert result == expected


def test_msgpack_blocked_emits_event() -> None:
    events: list[SerdeEvent] = []
    unregister = register_serde_event_listener(events.append)
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=None)
        obj = AnotherPydantic(foo="nope")
        serde.loads_typed(serde.dumps_typed(obj))
    finally:
        unregister()

    assert {
        "kind": "msgpack_blocked",
        "module": "tests.test_jsonplus",
        "name": "AnotherPydantic",
    } in events


def test_msgpack_unregistered_allowed_emits_event() -> None:
    events: list[SerdeEvent] = []
    unregister = register_serde_event_listener(events.append)
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=True)
        obj = AnotherPydantic(foo="ok")
        serde.loads_typed(serde.dumps_typed(obj))
    finally:
        unregister()

    assert {
        "kind": "msgpack_unregistered_allowed",
        "module": "tests.test_jsonplus",
        "name": "AnotherPydantic",
    } in events


def test_msgpack_strict_allows_safe_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Safe types should still deserialize in strict mode without warnings."""

    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    safe = uuid.uuid4()

    caplog.clear()
    dumped = serde.dumps_typed(safe)
    result = serde.loads_typed(dumped)

    assert "blocked" not in caplog.text.lower()
    assert result == safe


def test_msgpack_strict_allows_core_langchain_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    msg = HumanMessage(content="hello")

    caplog.clear()
    result = serde.loads_typed(serde.dumps_typed(msg))

    assert "blocked" not in caplog.text.lower()
    assert "unregistered" not in caplog.text.lower()
    assert isinstance(result, HumanMessage)
    assert result == msg


def test_msgpack_strict_allows_langchain_document(
    caplog: pytest.LogCaptureFixture,
) -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    doc = Document(page_content="hello", metadata={"k": "v"})

    caplog.clear()
    result = serde.loads_typed(serde.dumps_typed(doc))

    assert "blocked" not in caplog.text.lower()
    assert "unregistered" not in caplog.text.lower()
    assert isinstance(result, Document)
    assert result == doc


def test_msgpack_regex_safe_type(caplog: pytest.LogCaptureFixture) -> None:
    """re.compile patterns should deserialize without warnings as a safe type."""

    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    pattern = re.compile(r"foo.*bar", re.IGNORECASE | re.DOTALL)

    caplog.clear()
    dumped = serde.dumps_typed(pattern)
    result = serde.loads_typed(dumped)

    assert "blocked" not in caplog.text.lower()
    assert "unregistered" not in caplog.text.lower()
    assert result.pattern == pattern.pattern
    assert result.flags == pattern.flags


def test_msgpack_method_pathlib_blocked_in_strict(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "secret.txt"
    target.write_text("secret")
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    payload = ormsgpack.packb(
        ormsgpack.Ext(
            EXT_METHOD_SINGLE_ARG,
            _msgpack_enc(("pathlib", "Path", target, "read_text")),
        ),
        option=ormsgpack.OPT_NON_STR_KEYS,
    )

    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    caplog.clear()
    result = serde.loads_typed(("msgpack", payload))

    assert result == target
    assert "blocked deserialization of method call pathlib.path.read_text" in (
        caplog.text.lower()
    )


def test_msgpack_method_pathlib_blocked_default_mode(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "secret.txt"
    target.write_text("secret")
    serde = JsonPlusSerializer(allowed_msgpack_modules=True)
    payload = ormsgpack.packb(
        ormsgpack.Ext(
            EXT_METHOD_SINGLE_ARG,
            _msgpack_enc(("pathlib", "Path", target, "read_text")),
        ),
        option=ormsgpack.OPT_NON_STR_KEYS,
    )

    caplog.set_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus")
    caplog.clear()
    result = serde.loads_typed(("msgpack", payload))

    assert result == target
    assert "blocked deserialization of method call pathlib.path.read_text" in (
        caplog.text.lower()
    )


def test_msgpack_regex_still_works_strict(caplog: pytest.LogCaptureFixture) -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    pattern = re.compile(r"pattern", re.IGNORECASE | re.MULTILINE)

    caplog.clear()
    result = serde.loads_typed(serde.dumps_typed(pattern))

    assert "blocked" not in caplog.text.lower()
    assert result.pattern == pattern.pattern
    assert result.flags == pattern.flags


def test_msgpack_path_constructor_still_works() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    path_obj = pathlib.Path("/tmp/foo")

    result = serde.loads_typed(serde.dumps_typed(path_obj))

    assert result == path_obj


def test_with_msgpack_allowlist_noop_returns_same_instance() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)

    result = serde.with_msgpack_allowlist(())

    assert result is serde


def test_with_msgpack_allowlist_supports_subclass_without_init_kwargs() -> None:
    class CustomSerializer(JsonPlusSerializer):
        def __init__(self) -> None:
            super().__init__(allowed_msgpack_modules=None)

    serde = CustomSerializer()
    result = serde.with_msgpack_allowlist([MyDataclass])

    assert isinstance(result, CustomSerializer)
    assert result is not serde
    assert serde._allowed_msgpack_modules is None
    assert result._allowed_msgpack_modules == {
        (MyDataclass.__module__, MyDataclass.__name__)
    }


def test_with_msgpack_allowlist_rebuilds_default_unpack_hook() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    original_hook = serde._unpack_ext_hook

    result = serde.with_msgpack_allowlist([MyDataclass])

    assert result._unpack_ext_hook is not original_hook


def test_with_msgpack_allowlist_preserves_custom_unpack_hook() -> None:
    def custom_hook(code: int, data: bytes) -> None:
        return None

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=None, __unpack_ext_hook__=custom_hook
    )
    result = serde.with_msgpack_allowlist([MyDataclass])

    assert result._unpack_ext_hook is custom_hook


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="pydantic v1 not on 3.14+")
def test_msgpack_pydantic_v1_allowlist(caplog: pytest.LogCaptureFixture) -> None:
    """Pydantic v1 models in allowlist should deserialize without warnings."""

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("tests.test_jsonplus", "MyPydanticV1"),
            ("tests.test_jsonplus", "InnerPydanticV1"),
        ]
    )

    obj = MyPydanticV1(foo="test", bar=42, inner=InnerPydanticV1(hello="world"))

    caplog.clear()
    dumped = serde.dumps_typed(obj)
    result = serde.loads_typed(dumped)

    assert "unregistered type" not in caplog.text.lower()
    assert "blocked" not in caplog.text.lower()
    assert result == obj


def test_msgpack_dataclass_allowlist(caplog: pytest.LogCaptureFixture) -> None:
    """Dataclasses in allowlist should deserialize without warnings."""

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("tests.test_jsonplus", "MyDataclass"),
            ("tests.test_jsonplus", "InnerDataclass"),
        ]
    )

    obj = MyDataclass(foo="test", bar=42, inner=InnerDataclass(hello="world"))

    caplog.clear()
    dumped = serde.dumps_typed(obj)
    result = serde.loads_typed(dumped)

    assert "unregistered type" not in caplog.text.lower()
    assert "blocked" not in caplog.text.lower()
    assert result == obj


def test_msgpack_safe_types_value_equality(caplog: pytest.LogCaptureFixture) -> None:
    """Verify safe types are correctly restored with proper values."""

    serde = JsonPlusSerializer(allowed_msgpack_modules=None)

    test_cases = [
        datetime(2024, 1, 15, 12, 30, 45, 123456),
        date(2024, 6, 15),
        time(14, 30, 0),
        uuid.UUID("12345678-1234-5678-1234-567812345678"),
        Decimal("123.456789"),
        {1, 2, 3, 4, 5},
        frozenset(["a", "b", "c"]),
        deque([1, 2, 3]),
        IPv4Address("10.0.0.1"),
        pathlib.Path("/some/test/path"),
        re.compile(r"\d+", re.MULTILINE),
    ]

    for obj in test_cases:
        caplog.clear()
        dumped = serde.dumps_typed(obj)
        result = serde.loads_typed(dumped)

        assert "blocked" not in caplog.text.lower(), f"Blocked for {type(obj)}"
        # For regex patterns, compare pattern and flags
        if isinstance(obj, re.Pattern):
            assert result.pattern == obj.pattern
            assert result.flags == obj.flags
        else:
            assert result == obj, f"Value mismatch for {type(obj)}: {result} != {obj}"


def test_msgpack_nested_pydantic_serializes_as_dict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nested Pydantic models are serialized via model_dump() as dicts.

    This means nested models don't go through the ext hook and don't need
    to be in the allowlist - only the outer type does.
    """

    # Only allow outer type - inner is serialized as dict via model_dump()
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("tests.test_jsonplus", "MyPydantic")]
    )

    obj = MyPydantic(foo="test", bar=42, inner=InnerPydantic(hello="world"))

    caplog.clear()
    dumped = serde.dumps_typed(obj)
    result = serde.loads_typed(dumped)

    # No blocking should occur - inner is serialized as dict, not ext
    assert "blocked" not in caplog.text.lower()
    assert result == obj
