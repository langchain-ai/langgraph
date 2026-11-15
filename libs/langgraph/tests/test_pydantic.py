import datetime
import decimal
import ipaddress
import pathlib
import re
import sys
import typing
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from threading import Event
from typing import Annotated, Literal, Optional

import pydantic
import pytest
import typing_extensions
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import (
    BaseModel,
    ByteSize,
    Field,
    SecretStr,
    confloat,
    conint,
    conlist,
    constr,
    field_validator,
    model_validator,
)

from langgraph._internal._pydantic import create_model, is_supported_by_pydantic
from langgraph.constants import END, START
from langgraph.func import entrypoint, task
from langgraph.graph.state import StateGraph
from langgraph.types import Command, Interrupt, interrupt
from tests.any_str import AnyStr


@pytest.mark.parametrize("root_model", [False, True], ids=["fields", "root"])
@pytest.mark.parametrize("default", [None, []], ids=["cached", "uncached"])
def test_create_model_preserves_concurrent_warnings(root_model, default) -> None:
    schema_started = Event()
    warning_emitted = Event()

    class PausedSchema:
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            schema_started.set()
            assert warning_emitted.wait(timeout=10)
            return handler(source)

    type_ = Annotated[list[int] | None, PausedSchema]
    kwargs = (
        {"root": (type_, default)}
        if root_model
        else {"field_definitions": {"model_values": (type_, default)}}
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(create_model, "ConcurrentSchema", **kwargs)
            try:
                assert schema_started.wait(timeout=10)
                warnings.warn("unrelated concurrent warning", UserWarning)
                warnings.simplefilter("error", DeprecationWarning)
            finally:
                warning_emitted.set()
            model = future.result(timeout=10)

        assert [str(warning.message) for warning in captured] == [
            "unrelated concurrent warning"
        ]
        with pytest.raises(DeprecationWarning, match="retained warning filter"):
            warnings.warn("retained warning filter", DeprecationWarning)

    instance = model.model_validate(["1"] if root_model else {"model_values": ["1"]})
    assert instance.model_dump() == ([1] if root_model else {"model_values": [1]})


@pytest.mark.parametrize("default", [0, []], ids=["cached", "uncached"])
def test_create_model_model_fields(default) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = create_model(
            "ModelFields",
            field_definitions={
                "model_id": (int, ...),
                "model_name": (str, "default"),
                "model_dump": (str, ...),
                "_private": (int, 0),
                "model_value": (int | list[int], default),
            },
        )

    instance = model.model_validate(
        {"model_id": "1", "model_dump": "data", "_private": "2"}
    )
    assert model.__name__ == "ModelFields"
    assert instance.model_dump(by_alias=True) == {
        "model_id": 1,
        "model_name": "default",
        "model_dump": "data",
        "_private": 2,
        "model_value": default,
    }
    with pytest.raises(pydantic.ValidationError, match="model_id"):
        model.model_validate({"model_id": "invalid", "model_dump": "data"})
    with pytest.raises(pydantic.ValidationError, match="frozen_instance"):
        instance.model_id = 2

    schema = model.model_json_schema()
    assert schema["title"] == "ModelFields"
    assert schema["required"] == ["model_id", "model_dump"]
    assert set(schema["properties"]) == {
        "model_id",
        "model_name",
        "model_dump",
        "_private",
        "model_value",
    }
    assert schema["properties"]["model_dump"] == {
        "title": "Model Dump",
        "type": "string",
    }
    assert schema["properties"]["model_value"]["default"] == default


@pytest.mark.parametrize("default", [0, []], ids=["cached", "uncached"])
def test_create_root_model_defaults(default) -> None:
    model = create_model("RootSchema", root=(int | list[int], default))
    assert model.__name__ == "RootSchema"
    assert model().model_dump() == default
    assert model.model_validate("2").model_dump() == 2
    schema = model.model_json_schema()
    assert schema == {
        "anyOf": [{"type": "integer"}, {"items": {"type": "integer"}, "type": "array"}],
        "default": default,
        "title": "RootSchema",
    }
    with pytest.warns(pydantic.PydanticDeprecatedSince20, match="deprecated"):
        assert model.schema() == schema


def test_create_model_preserves_schema_warnings() -> None:
    class WarningSchema:
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            warnings.warn("custom schema warning", UserWarning)
            return handler(source)

    with pytest.warns(UserWarning, match="custom schema warning"):
        create_model(
            "WarningSchema", field_definitions={"model_value": (WarningSchema, ...)}
        )


def test_is_supported_by_pydantic() -> None:
    """Test if types are supported by pydantic."""

    class TypedDictExtensions(typing_extensions.TypedDict):
        x: int

    assert is_supported_by_pydantic(TypedDictExtensions) is True

    class VanillaClass:
        x: int

    assert is_supported_by_pydantic(VanillaClass) is False

    class BuiltinTypedDict(typing.TypedDict):  # noqa: TID251
        x: int

    if sys.version_info >= (3, 12):
        assert is_supported_by_pydantic(BuiltinTypedDict) is True
    else:
        assert is_supported_by_pydantic(BuiltinTypedDict) is False

    class PydanticModel(pydantic.BaseModel):
        x: int

    assert is_supported_by_pydantic(PydanticModel) is True


def test_nested_pydantic_models() -> None:
    """Test that nested Pydantic models are properly constructed from leaf nodes up."""

    class NestedModel(BaseModel):
        value: int
        name: str

    # For constrained types
    PositiveInt = Annotated[int, Field(gt=0)]
    NonNegativeFloat = Annotated[float, Field(ge=0)]

    # Enum type
    class UserRole(Enum):
        ADMIN = "admin"
        USER = "user"
        GUEST = "guest"

    # Forward reference model
    class RecursiveModel(BaseModel):
        value: str
        child: Optional["RecursiveModel"] = None

    # Discriminated union models
    class Cat(BaseModel):
        pet_type: Literal["cat"]
        meow: str

    class Dog(BaseModel):
        pet_type: Literal["dog"]
        bark: str

    # Cyclic reference model
    class Person(BaseModel):
        id: str
        name: str
        friends: list[str] = Field(default_factory=list)  # IDs of friends

    conlist_type = conlist(item_type=int, min_length=2, max_length=5)

    class State(BaseModel):
        # Basic nested model tests
        top_level: str
        auuid: uuid.UUID
        nested: NestedModel
        optional_nested: Annotated[NestedModel | None, lambda x, y: y, "Foo"]
        dict_nested: dict[str, NestedModel]
        simple_str_list: list[str]
        list_nested: Annotated[
            dict | list[dict[str, NestedModel]], lambda x, y: (x or []) + [y]
        ]
        tuple_nested: tuple[str, NestedModel]
        tuple_list_nested: list[tuple[int, NestedModel]]
        complex_tuple: tuple[str, dict[str, tuple[int, NestedModel]]]

        # Forward reference test
        recursive: RecursiveModel

        # Discriminated union test
        pet: Cat | Dog

        # Cyclic reference test
        people: dict[str, Person]  # Map of ID -> Person

        # Rich type adapters
        ip_address: ipaddress.IPv4Address
        ip_address_v6: ipaddress.IPv6Address
        amount: decimal.Decimal
        file_path: pathlib.Path
        timestamp: datetime.datetime
        date_only: datetime.date
        time_only: datetime.time
        duration: datetime.timedelta
        immutable_set: frozenset[int]
        binary_data: bytes
        pattern: re.Pattern
        secret: SecretStr
        file_size: ByteSize

        # Constrained types
        positive_value: PositiveInt
        non_negative: NonNegativeFloat
        limited_string: constr(min_length=3, max_length=10)
        bounded_int: conint(ge=10, le=100)
        restricted_float: confloat(gt=0, lt=1)
        required_list: conlist_type

        # Enum & Literal
        role: UserRole
        status: Literal["active", "inactive", "pending"]

        # Annotated & NewType
        validated_age: Annotated[int, Field(gt=0, lt=120)]

        # Generic containers with validators
        decimal_list: list[decimal.Decimal]
        id_tuple: tuple[uuid.UUID, uuid.UUID]

    inputs = {
        # Basic nested models
        "top_level": "initial",
        "auuid": str(uuid.uuid4()),
        "nested": {"value": 42, "name": "test"},
        "optional_nested": {"value": 10, "name": "optional"},
        "dict_nested": {"a": {"value": 5, "name": "a"}},
        "list_nested": [{"a": {"value": 6, "name": "b"}}],
        "tuple_nested": ["tuple-key", {"value": 7, "name": "tuple-value"}],
        "tuple_list_nested": [[1, {"value": 8, "name": "tuple-in-list"}]],
        "simple_str_list": ["siss", "boom", "bah"],
        "complex_tuple": [
            "complex",
            {"nested": [9, {"value": 10, "name": "deep"}]},
        ],
        # Forward reference
        "recursive": {"value": "parent", "child": {"value": "child", "child": None}},
        # Discriminated union (using a cat in this case)
        "pet": {"pet_type": "cat", "meow": "meow!"},
        # Cyclic references
        "people": {
            "1": {
                "id": "1",
                "name": "Alice",
                "friends": ["2", "3"],  # Alice is friends with Bob and Charlie
            },
            "2": {
                "id": "2",
                "name": "Bob",
                "friends": ["1"],  # Bob is friends with Alice
            },
            "3": {
                "id": "3",
                "name": "Charlie",
                "friends": ["1", "2"],  # Charlie is friends with Alice and Bob
            },
        },
        # Rich type adapters
        "ip_address": "192.168.1.1",
        "ip_address_v6": "2001:db8::1",
        "amount": "123.45",
        "file_path": "/tmp/test.txt",
        "timestamp": "2025-04-07T10:58:04",
        "date_only": "2025-04-07",
        "time_only": "10:58:04",
        "duration": 3600,  # seconds
        "immutable_set": [1, 2, 3, 4],
        "binary_data": b"hello world",
        "pattern": "^test$",
        "secret": "password123",
        "file_size": 1024,
        # Constrained types
        "positive_value": 42,
        "non_negative": 0.0,
        "limited_string": "test",
        "bounded_int": 50,
        "restricted_float": 0.5,
        "required_list": [10, 20, 30],
        # Enum & Literal
        "role": "admin",
        "status": "active",
        # Annotated & NewType
        "validated_age": 30,
        # Generic containers with validators
        "decimal_list": ["10.5", "20.75", "30.25"],
        "id_tuple": [str(uuid.uuid4()), str(uuid.uuid4())],
    }

    update = {"top_level": "updated", "nested": {"value": 100, "name": "updated"}}

    expected = State(**inputs)

    def node_fn(state: State) -> dict:
        # Basic assertions
        assert isinstance(state.auuid, uuid.UUID)
        assert state == expected

        # Rich type assertions
        assert isinstance(state.ip_address, ipaddress.IPv4Address)
        assert isinstance(state.ip_address_v6, ipaddress.IPv6Address)
        assert isinstance(state.amount, decimal.Decimal)
        assert isinstance(state.file_path, pathlib.Path)
        assert isinstance(state.timestamp, datetime.datetime)
        assert isinstance(state.date_only, datetime.date)
        assert isinstance(state.time_only, datetime.time)
        assert isinstance(state.duration, datetime.timedelta)
        assert isinstance(state.immutable_set, frozenset)
        assert isinstance(state.binary_data, bytes)
        assert isinstance(state.pattern, re.Pattern)

        # Constrained types
        assert state.positive_value > 0
        assert state.non_negative >= 0
        assert 3 <= len(state.limited_string) <= 10
        assert 10 <= state.bounded_int <= 100
        assert 0 < state.restricted_float < 1
        assert 2 <= len(state.required_list) <= 5

        # Enum & Literal
        assert state.role == UserRole.ADMIN
        assert state.status == "active"

        # Annotated
        assert 0 < state.validated_age < 120

        # Generic containers
        assert len(state.decimal_list) == 3
        assert len(state.id_tuple) == 2

        return update

    builder = StateGraph(State)
    builder.add_node("process", node_fn)
    builder.set_entry_point("process")
    builder.set_finish_point("process")
    graph = builder.compile()

    result = graph.invoke(inputs.copy())

    assert result == {**inputs, **update}

    new_inputs = inputs.copy()
    new_inputs["list_nested"] = {"foo": "bar"}
    expected = State(**new_inputs)
    assert {**new_inputs, **update} == graph.invoke(new_inputs.copy())


def test_pydantic_state_field_validator():
    class State(BaseModel):
        name: str
        text: str = ""
        only_root: int = 13

        @field_validator("name", mode="after")
        @classmethod
        def validate_name(cls, value):
            if value[0].islower():
                raise ValueError("Name must start with a capital letter")
            return "Validated " + value

        @model_validator(mode="before")
        @classmethod
        def validate_amodel(cls, values: "State"):
            return values | {"only_root": 392}

    input_state = {"name": "John"}

    def process_node(state: State):
        assert State.model_validate(input_state) == state
        return {"text": "Hello, " + state.name + "!"}

    builder = StateGraph(state_schema=State)
    builder.add_node("process", process_node)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    g = builder.compile()
    res = g.invoke(input_state)
    assert res["text"] == "Hello, Validated John!"


class FunctionalState(BaseModel):
    a: str
    b: str | None = None


def test_interrupt_functional_pydantic(sync_checkpointer: BaseCheckpointSaver) -> None:
    called_count = 0

    @task
    def foo(state: FunctionalState) -> FunctionalState:
        nonlocal called_count
        called_count += 1
        return FunctionalState(**{"a": state.a + "foo"})

    @task
    def bar(state: FunctionalState) -> dict:
        return {"a": state.a + "bar", "b": state.b}

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(inputs: FunctionalState) -> FunctionalState:
        fut_foo = foo(inputs)
        value = interrupt("Provide value for bar:")
        foo_res = fut_foo.result()
        assert isinstance(foo_res, FunctionalState)
        bar_input = FunctionalState(a=foo_res.a, b=value)
        fut_bar = bar(bar_input)
        return fut_bar.result()

    config = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert graph.invoke(FunctionalState(a=""), config) == {
        "__interrupt__": [
            Interrupt(
                value="Provide value for bar:",
                id=AnyStr(),
            )
        ]
    }
    # Resume with an answer
    res = graph.invoke(Command(resume="bar"), config)
    assert res == {"a": "foobar", "b": "bar"}
    assert called_count == 1
