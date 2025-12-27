import datetime
import decimal
import ipaddress
import pathlib
import re
import sys
import uuid
from enum import Enum
from typing import Annotated, Literal, Optional

import pytest
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

from langgraph._internal._pydantic import is_supported_by_pydantic
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph


def test_is_supported_by_pydantic() -> None:
    """Test if types are supported by pydantic."""
    import typing

    import pydantic
    import typing_extensions

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


def test_pydantic_aliased_fields():
    """Test that Pydantic state with aliased fields works correctly.

    When using Field(alias=...), the graph should accept input using either
    the alias or the field name.

    This addresses GitHub issue #2555.
    """

    class State(BaseModel):
        foo: str = Field(alias="bar")
        other: str = Field(default="default")

    def node_fn(state: State) -> dict:
        # The state should have the correct value
        assert state.foo == "hello"
        assert state.other == "default"
        return {"other": "updated"}

    builder = StateGraph(State)
    builder.add_node("process", node_fn)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    graph = builder.compile()

    # Test with alias key
    result = graph.invoke({"bar": "hello"})
    assert result["foo"] == "hello"
    assert result["other"] == "updated"


def test_pydantic_multiple_aliased_fields():
    """Test multiple aliased fields work correctly."""

    class State(BaseModel):
        first_name: str = Field(alias="firstName")
        last_name: str = Field(alias="lastName")
        age: int = Field(alias="userAge")

    def node_fn(state: State) -> dict:
        assert state.first_name == "John"
        assert state.last_name == "Doe"
        assert state.age == 30
        return {"age": 31}

    builder = StateGraph(State)
    builder.add_node("process", node_fn)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    graph = builder.compile()

    # Test with alias keys (camelCase)
    result = graph.invoke({"firstName": "John", "lastName": "Doe", "userAge": 30})
    assert result["first_name"] == "John"
    assert result["last_name"] == "Doe"
    assert result["age"] == 31


def test_pydantic_mixed_alias_and_fieldname():
    """Test mixing aliased and non-aliased fields."""

    class State(BaseModel):
        aliased_field: str = Field(alias="aliasedField")
        normal_field: str

    def node_fn(state: State) -> dict:
        assert state.aliased_field == "alias_value"
        assert state.normal_field == "normal_value"
        return {}

    builder = StateGraph(State)
    builder.add_node("process", node_fn)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    graph = builder.compile()

    # Test with alias for aliased field and field name for normal field
    result = graph.invoke(
        {"aliasedField": "alias_value", "normal_field": "normal_value"}
    )
    assert result["aliased_field"] == "alias_value"
    assert result["normal_field"] == "normal_value"


def test_pydantic_alias_with_reducer():
    """Test aliased fields work with reducers."""

    def concat_reducer(a: list[str], b: str | list[str]) -> list[str]:
        if isinstance(b, list):
            return a + b
        return a + [b]

    class State(BaseModel):
        messages: Annotated[list[str], concat_reducer] = Field(
            default_factory=list, alias="msgs"
        )

    def node_fn(state: State) -> dict:
        return {"messages": "processed"}

    builder = StateGraph(State)
    builder.add_node("process", node_fn)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    graph = builder.compile()

    # Test with alias key and list initial value
    result = graph.invoke({"msgs": ["hello"]})
    assert result["messages"] == ["hello", "processed"]


def test_pydantic_aliased_fields_multinode_with_checkpointer():
    """Integration test: aliased fields with multiple nodes and checkpointing."""
    from langgraph.checkpoint.memory import InMemorySaver

    class State(BaseModel):
        user_name: str = Field(alias="userName")
        step_count: int = Field(default=0, alias="stepCount")
        processed_by: Annotated[list[str], lambda a, b: a + [b]] = Field(
            default_factory=list, alias="processedBy"
        )

    def node_a(state: State) -> dict:
        assert state.user_name == "Alice"
        return {"step_count": state.step_count + 1, "processed_by": "node_a"}

    def node_b(state: State) -> dict:
        assert state.user_name == "Alice"
        assert state.step_count == 1
        assert "node_a" in state.processed_by
        return {"step_count": state.step_count + 1, "processed_by": "node_b"}

    def node_c(state: State) -> dict:
        assert state.step_count == 2
        assert state.processed_by == ["node_a", "node_b"]
        return {"step_count": state.step_count + 1, "processed_by": "node_c"}

    builder = StateGraph(State)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", "node_c")
    builder.add_edge("node_c", END)

    # Test without checkpointer - full run
    graph = builder.compile()
    result = graph.invoke({"userName": "Alice"})
    assert result["user_name"] == "Alice"
    assert result["step_count"] == 3
    assert result["processed_by"] == ["node_a", "node_b", "node_c"]

    # Test with checkpointer and interrupt
    checkpointer = InMemorySaver()
    graph_with_cp = builder.compile(
        checkpointer=checkpointer, interrupt_after=["node_a"]
    )

    config = {"configurable": {"thread_id": "test-1"}}

    # First invocation - should stop after node_a
    result1 = graph_with_cp.invoke({"userName": "Alice"}, config)
    assert result1["step_count"] == 1
    assert result1["processed_by"] == ["node_a"]

    # Check state
    state = graph_with_cp.get_state(config)
    assert state.next == ("node_b",)

    # Resume - should complete
    result2 = graph_with_cp.invoke(None, config)
    assert result2["step_count"] == 3
    assert result2["processed_by"] == ["node_a", "node_b", "node_c"]


@pytest.mark.anyio
async def test_pydantic_aliased_fields_async():
    """Integration test: aliased fields with async graph."""

    class State(BaseModel):
        query: str = Field(alias="q")
        result: str = Field(default="", alias="res")

    async def process(state: State) -> dict:
        assert state.query == "hello"
        return {"result": f"processed: {state.query}"}

    builder = StateGraph(State)
    builder.add_node("process", process)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    graph = builder.compile()

    # Test async invoke with alias
    result = await graph.ainvoke({"q": "hello"})
    assert result["query"] == "hello"
    assert result["result"] == "processed: hello"
