import pytest
from typing import List, Dict, Set, Tuple, Optional, Union, TypeVar, Generic, Literal

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from typing_extensions import Annotated
from pydantic import BaseModel, Field, Discriminator, Tag

from langgraph.graph.schema_utils import SchemaCoercionMapper


def test_any_message():
    class MyMessage(BaseModel):
        msg: List[AnyMessage]


    data = {
        "msg": [
            {
                "type": "human",
                "content": "Hello"
            },
            {
                "type": "ai",
                "content": "Hi there!"
            }
        ]
    }

    MyMessage.model_validate(data)

    mapper = SchemaCoercionMapper(MyMessage)
    result = mapper(data)
    assert isinstance(result, MyMessage)
    assert isinstance(result.msg, list)
    assert len(result.msg) == 2
    assert isinstance(result.msg[0], (HumanMessage))
    assert isinstance(result.msg[1], (AIMessage))

# ==== 基础模型 ====
class SimpleModel(BaseModel):
    name: str
    age: int


def test_simple_model():
    data = {"name": "Alice", "age": 30}
    mapper = SchemaCoercionMapper(SimpleModel)
    result = mapper(data)
    assert isinstance(result, SimpleModel)
    assert result.name == "Alice"
    assert result.age == 30


# ==== 容器类型 ====
class ContainerModel(BaseModel):
    items: List[int]
    mapping: Dict[str, float]
    tags: Set[str]
    coords: Tuple[int, int]


def test_container_model():
    data = {
        "items": [1, 2, 3],
        "mapping": {"a": 1.1},
        "tags": ["x", "y"],
        "coords": [10, 20],
    }
    mapper = SchemaCoercionMapper(ContainerModel)
    result = mapper(data)
    assert isinstance(result.items, list)
    assert isinstance(result.mapping, dict)
    assert isinstance(result.tags, set)
    assert isinstance(result.coords, tuple)


# ==== 泛型 ====
T = TypeVar("T")


class Wrapper(BaseModel, Generic[T]):
    value: T


def test_generic_model():
    class IntWrapper(Wrapper[int]):
        pass

    data = {"value": 123}
    mapper = SchemaCoercionMapper(IntWrapper)
    result = mapper(data)
    assert result.value == 123


# ==== Union 类型 ====
class Dog(BaseModel):
    type: Literal["dog"]
    age: int


class Cat(BaseModel):
    type: Literal["cat"]
    name: str


Pet = Union[Dog, Cat]


class Owner(BaseModel):
    pet: Pet


def test_union_type():
    data = {"pet": {"type": "dog", "age": 5}}
    mapper = SchemaCoercionMapper(Owner)
    result = mapper(data)
    assert isinstance(result.pet, Dog)


# ==== Annotated + Tag + discriminator ====
TaggedPet = Annotated[
    Union[
        Annotated[Dog, Tag(tag="dog")],
        Annotated[Cat, Tag(tag="cat")],
    ],
    Field(discriminator="type"),
]


class TaggedOwner(BaseModel):
    pet: TaggedPet


def test_tagged_union():
    data = {"pet": {"type": "cat", "name": "Mimi"}}
    mapper = SchemaCoercionMapper(TaggedOwner)
    result = mapper(data)
    assert isinstance(result.pet, Cat)


# ==== Annotated + Field(discriminator=Discriminator(func)) ====
def _get_type(obj):
    return obj.get("type")


FuncDiscriminatorPet = Annotated[
    Union[
        Annotated[Dog, Tag(tag="dog")],
        Annotated[Cat, Tag(tag="cat")],
    ],
    Field(discriminator=Discriminator(_get_type)),
]


class FuncOwner(BaseModel):
    pet: FuncDiscriminatorPet


def test_func_discriminator():
    data = {"pet": {"type": "dog", "age": 9}}
    mapper = SchemaCoercionMapper(FuncOwner)
    result = mapper(data)
    assert isinstance(result.pet, Dog)


# ==== Optional + 泛型 + 多态嵌套 ====
class Box(BaseModel, Generic[T]):
    content: Optional[T]


class Crate(BaseModel, Generic[T]):
    payload: Box[T]


class Zoo(BaseModel):
    animal: Box[TaggedPet]


class Warehouse(BaseModel):
    cage: Crate[TaggedPet]


def test_nested_optional_generic_union():
    # Box[TaggedPet]
    data1 = {
        "animal": {
            "content": {
                "type": "cat",
                "name": "Kitty"
            }
        }
    }
    mapper1 = SchemaCoercionMapper(Zoo)
    result1 = mapper1(data1)
    assert isinstance(result1.animal.content, Cat)

    # Crate[TaggedPet]
    data2 = {
        "cage": {
            "payload": {
                "content": {
                    "type": "dog",
                    "age": 8
                }
            }
        }
    }
    mapper2 = SchemaCoercionMapper(Warehouse)
    result2 = mapper2(data2)
    assert isinstance(result2.cage.payload.content, Dog)

    # Optional None
    data3 = {"animal": {"content": None}}
    result3 = mapper1(data3)
    assert result3.animal.content is None

    # deeply nested Optional
    data4 = {"cage": {"payload": {"content": None}}}
    result4 = mapper2(data4)
    assert result4.cage.payload.content is None