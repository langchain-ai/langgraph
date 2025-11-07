import dataclasses
import json
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
import pandas as pd
import pytest
from pydantic import BaseModel, SecretStr
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import SecretStr as SecretStrV1

from langgraph.checkpoint.serde.jsonplus import (
    InvalidModuleError,
    JsonPlusSerializer,
    _msgpack_ext_hook_to_json,
)
from langgraph.store.base import Item


class InnerPydantic(BaseModel):
    hello: str


class MyPydantic(BaseModel):
    foo: str
    bar: int
    inner: InnerPydantic


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

    serde = JsonPlusSerializer()

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
