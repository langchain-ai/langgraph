import dataclasses
import pathlib
import re
import sys
import uuid
from collections import deque
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address

import dataclasses_json
from pydantic import BaseModel, SecretStr
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import SecretStr as SecretStrV1
from zoneinfo import ZoneInfo

from langgraph.checkpoint.serde.jsonplus import (
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


if sys.version_info < (3, 10):

    class MyDataclassWSlots(MyDataclass):
        pass

else:

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
        "my_pydantic_v1": MyPydanticV1(
            foo="foo", bar=1, inner=InnerPydanticV1(hello="hello")
        ),
        "my_secret_str": SecretStr("meow"),
        "my_secret_str_v1": SecretStrV1("meow"),
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

    serde = JsonPlusSerializer()

    dumped = serde.dumps_typed(to_serialize)

    assert dumped[0] == "msgpack"
    assert serde.loads_typed(dumped) == to_serialize

    for value in to_serialize.values():
        assert serde.loads_typed(serde.dumps_typed(value)) == value

    surrogates = [
        "Hello\ud83d\ude00",
        "Python\ud83d\udc0d",
        "Surrogate\ud834\udd1e",
        "Example\ud83c\udf89",
        "String\ud83c\udfa7",
        "With\ud83c\udf08",
        "Surrogates\ud83d\ude0e",
        "Embedded\ud83d\udcbb",
        "In\ud83c\udf0e",
        "The\ud83d\udcd6",
        "Text\ud83d\udcac",
        "收花🙄·到",
    ]

    assert serde.loads_typed(serde.dumps_typed(surrogates)) == [
        v.encode("utf-8", "ignore").decode() for v in surrogates
    ]


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
        "my_pydantic_v1": MyPydanticV1(
            foo="foo", bar=1, inner=InnerPydanticV1(hello="hello")
        ),
        "my_secret_str": SecretStr("meow"),
        "my_secret_str_v1": SecretStrV1("meow"),
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

    serde = JsonPlusSerializer(__unpack_ext_hook__=_msgpack_ext_hook_to_json)

    dumped = serde.dumps_typed(to_serialize)

    assert dumped[0] == "msgpack"
    result = serde.loads_typed(dumped)
    assert result == {
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
        "my_pydantic_v1": {"foo": "foo", "bar": 1, "inner": {"hello": "hello"}},
        "my_secret_str": "meow",
        "my_secret_str_v1": "meow",
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


def test_serde_jsonplus_bytes() -> None:
    serde = JsonPlusSerializer()

    some_bytes = b"my bytes"
    dumped = serde.dumps_typed(some_bytes)

    assert dumped == ("bytes", some_bytes)
    assert serde.loads_typed(dumped) == some_bytes


def test_serde_jsonplus_bytearray() -> None:
    serde = JsonPlusSerializer()

    some_bytearray = bytearray([42])
    dumped = serde.dumps_typed(some_bytearray)

    assert dumped == ("bytearray", some_bytearray)
    assert serde.loads_typed(dumped) == some_bytearray


def test_loads_cannot_find() -> None:
    serde = JsonPlusSerializer()

    dumped = (
        "json",
        b'{"lc": 2, "type": "constructor", "id": ["tests", "test_jsonplus", "MyPydanticccc"], "method": null, "args": [], "kwargs": {"foo": "foo", "bar": 1}}',
    )

    assert serde.loads_typed(dumped) is None, "Should return None if cannot find class"

    dumped = (
        "json",
        b'{"lc": 2, "type": "constructor", "id": ["tests", "test_jsonpluss", "MyPydantic"], "method": null, "args": [], "kwargs": {"foo": "foo", "bar": 1}}',
    )

    assert serde.loads_typed(dumped) is None, "Should return None if cannot find module"
