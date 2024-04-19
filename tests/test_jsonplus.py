import dataclasses
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum

import dataclasses_json
import pytest
from langchain_core.pydantic_v1 import BaseModel as LcBaseModel
from langchain_core.runnables import RunnableMap
from pydantic import BaseModel

from langgraph.serde.jsonplus import JsonPlusSerializer


class MyPydantic(BaseModel):
    foo: str
    bar: int


class MyFunnyPydantic(LcBaseModel):
    foo: str
    bar: int


@dataclasses.dataclass
class MyDataclass:
    foo: str
    bar: int

    def something(self) -> None:
        pass


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


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_serde_jsonplus() -> None:
    uid = uuid.UUID(int=1)
    current_time = datetime(2024, 4, 19, 23, 4, 57, 51022, timezone.max)

    to_serialize = {
        "uid": uid,
        "time": current_time,
        "my_slotted_class": MyDataclassWSlots("bar", 2),
        "my_dataclass": MyDataclass("foo", 1),
        "my_enum": MyEnum.FOO,
        "my_pydantic": MyPydantic(foo="foo", bar=1),
        "my_funny_pydantic": MyFunnyPydantic(foo="foo", bar=1),
        "person": Person(name="foo"),
        "a_bool": True,
        "a_none": None,
        "a_str": "foo",
        "an_int": 1,
        "a_float": 1.1,
        "runnable_map": RunnableMap({}),
    }

    serde = JsonPlusSerializer()

    dumped = serde.dumps(to_serialize)

    assert (
        dumped
        == '{"a_bool": true, "a_float": 1.1, "a_none": null, "a_str": "foo", "an_int": 1, "my_dataclass": {"args": [], "id": ["tests", "test_jsonplus", "MyDataclass"], "kwargs": {"bar": 1, "foo": "foo"}, "lc": 2, "method": null, "type": "constructor"}, "my_enum": {"args": ["foo"], "id": ["tests", "test_jsonplus", "MyEnum"], "kwargs": {}, "lc": 2, "method": null, "type": "constructor"}, "my_funny_pydantic": {"args": [], "id": ["tests", "test_jsonplus", "MyFunnyPydantic"], "kwargs": {"bar": 1, "foo": "foo"}, "lc": 2, "method": null, "type": "constructor"}, "my_pydantic": {"args": [], "id": ["tests", "test_jsonplus", "MyPydantic"], "kwargs": {"bar": 1, "foo": "foo"}, "lc": 2, "method": null, "type": "constructor"}, "my_slotted_class": {"args": [], "id": ["tests", "test_jsonplus", "MyDataclassWSlots"], "kwargs": {"bar": 2, "foo": "bar"}, "lc": 2, "method": null, "type": "constructor"}, "person": {"args": [], "id": ["tests", "test_jsonplus", "Person"], "kwargs": {"name": "foo"}, "lc": 2, "method": null, "type": "constructor"}, "runnable_map": {"graph": {"edges": [], "nodes": [{"data": "Parallel<>Input", "id": 0, "type": "schema"}, {"data": "Parallel<>Output", "id": 1, "type": "schema"}]}, "id": ["langchain", "schema", "runnable", "RunnableParallel"], "kwargs": {"steps": {}}, "lc": 1, "name": "RunnableParallel<>", "type": "constructor"}, "time": {"args": ["2024-04-19T23:04:57.051022+23:59"], "id": ["datetime", "datetime"], "kwargs": {}, "lc": 2, "method": "fromisoformat", "type": "constructor"}, "uid": {"args": ["00000000000000000000000000000001"], "id": ["uuid", "UUID"], "kwargs": {}, "lc": 2, "method": null, "type": "constructor"}}'
    )

    assert serde.loads(dumped) == to_serialize
