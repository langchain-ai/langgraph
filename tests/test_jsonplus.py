import dataclasses
import uuid
from datetime import datetime, timezone
from enum import Enum

import dataclasses_json
from langchain_core.pydantic_v1 import BaseModel as LcBaseModel
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


def test_serde_jsonplus() -> None:
    uid = uuid.uuid4()
    current_time = datetime.now(timezone.max)

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
    }

    serde = JsonPlusSerializer()

    assert serde.loads(serde.dumps(to_serialize)) == to_serialize
