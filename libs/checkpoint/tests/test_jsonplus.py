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
from langchain_core.runnables import RunnableMap
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from zoneinfo import ZoneInfo

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class MyPydantic(BaseModel):
    foo: str
    bar: int


class MyFunnyPydantic(BaseModelV1):
    foo: str
    bar: int


@dataclasses.dataclass
class MyDataclass:
    foo: str
    bar: int

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
        "ip4": ip4,
        "deque": deque_instance,
        "tzn": tzn,
        "date": current_date,
        "time": current_time,
        "uid": uid,
        "timestamp": current_timestamp,
        "my_slotted_class": MyDataclassWSlots("bar", 2),
        "my_dataclass": MyDataclass("foo", 1),
        "my_enum": MyEnum.FOO,
        "my_pydantic": MyPydantic(foo="foo", bar=1),
        "my_funny_pydantic": MyFunnyPydantic(foo="foo", bar=1),
        "person": Person(name="foo"),
        "a_bool": True,
        "a_none": None,
        "a_str": "foo",
        "a_str_nuc": "foo\u0000",
        "a_str_uc": "foo ⛰️",
        "a_str_ucuc": "foo \u26f0\ufe0f\u0000",
        "a_str_ucucuc": "foo \\u26f0\\ufe0f",
        "text": [
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
        ],
        "an_int": 1,
        "a_float": 1.1,
        "runnable_map": RunnableMap({}),
    }

    serde = JsonPlusSerializer()

    dumped = serde.dumps_typed(to_serialize)

    assert dumped == (
        "json",
        b"""{"path": {"lc": 2, "type": "constructor", "id": ["pathlib", "Path"], "args": ["foo", "bar"]}, "re": {"lc": 2, "type": "constructor", "id": ["re", "compile"], "args": ["foo", 48]}, "decimal": {"lc": 2, "type": "constructor", "id": ["decimal", "Decimal"], "args": ["1.10101"]}, "ip4": {"lc": 2, "type": "constructor", "id": ["ipaddress", "IPv4Address"], "args": ["192.168.0.1"]}, "deque": {"lc": 2, "type": "constructor", "id": ["collections", "deque"], "args": [[1, 2, 3]]}, "tzn": {"lc": 2, "type": "constructor", "id": ["zoneinfo", "ZoneInfo"], "args": ["America/New_York"]}, "date": {"lc": 2, "type": "constructor", "id": ["datetime", "date"], "args": [2024, 4, 19]}, "time": {"lc": 2, "type": "constructor", "id": ["datetime", "time"], "args": [23, 4, 57, 51022, {"lc": 2, "type": "constructor", "id": ["datetime", "timezone"], "args": [{"lc": 2, "type": "constructor", "id": ["datetime", "timedelta"], "args": [0, 86340, 0]}]}], "kwargs": {"fold": 0}}, "uid": {"lc": 2, "type": "constructor", "id": ["uuid", "UUID"], "args": ["00000000000000000000000000000001"]}, "timestamp": {"lc": 2, "type": "constructor", "id": ["datetime", "datetime"], "method": "fromisoformat", "args": ["2024-04-19T23:04:57.051022+23:59"]}, "my_slotted_class": {"lc": 2, "type": "constructor", "id": ["tests", "test_jsonplus", "MyDataclassWSlots"], "kwargs": {"foo": "bar", "bar": 2}}, "my_dataclass": {"lc": 2, "type": "constructor", "id": ["tests", "test_jsonplus", "MyDataclass"], "kwargs": {"foo": "foo", "bar": 1}}, "my_enum": {"lc": 2, "type": "constructor", "id": ["tests", "test_jsonplus", "MyEnum"], "args": ["foo"]}, "my_pydantic": {"lc": 2, "type": "constructor", "id": ["tests", "test_jsonplus", "MyPydantic"], "method": [null, "model_construct"], "kwargs": {"foo": "foo", "bar": 1}}, "my_funny_pydantic": {"lc": 2, "type": "constructor", "id": ["tests", "test_jsonplus", "MyFunnyPydantic"], "method": [null, "construct"], "kwargs": {"foo": "foo", "bar": 1}}, "person": {"lc": 2, "type": "constructor", "id": ["tests", "test_jsonplus", "Person"], "kwargs": {"name": "foo"}}, "a_bool": true, "a_none": null, "a_str": "foo", "a_str_nuc": "foo\\u0000", "a_str_uc": "foo \xe2\x9b\xb0\xef\xb8\x8f", "a_str_ucuc": "foo \xe2\x9b\xb0\xef\xb8\x8f\\u0000", "a_str_ucucuc": "foo \\\\u26f0\\\\ufe0f", "text": ["Hello", "Python", "Surrogate", "Example", "String", "With", "Surrogates", "Embedded", "In", "The", "Text", "\xe6\x94\xb6\xe8\x8a\xb1\xf0\x9f\x99\x84\xc2\xb7\xe5\x88\xb0"], "an_int": 1, "a_float": 1.1, "runnable_map": {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "runnable", "RunnableParallel"], "kwargs": {"steps__": {}}, "name": "RunnableParallel<>", "graph": {"nodes": [{"id": 0, "type": "schema", "data": "Parallel<>Input"}, {"id": 1, "type": "schema", "data": "Parallel<>Output"}], "edges": []}}}""",
    )

    assert serde.loads_typed(dumped) == {
        **to_serialize,
        "text": [v.encode("utf-8", "ignore").decode() for v in to_serialize["text"]],
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
