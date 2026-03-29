"""Regression tests for recursive message detection in stream_mode=messages.

Before this fix, ``_find_and_emit_messages`` and ``on_chain_start`` only scanned
two levels deep, missing messages nested inside Pydantic models, dicts-of-dicts,
or dataclass fields.
"""

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from langgraph.pregel._messages import _find_messages_recursive

# --- test fixtures ---


class InnerModel(BaseModel):
    messages: list[BaseMessage]


class OuterModel(BaseModel):
    inner: InnerModel
    label: str = "outer"


@dataclass
class DataContainer:
    name: str
    messages: list[BaseMessage] = field(default_factory=list)


# --- tests ---


def test_flat_list() -> None:
    """Messages in a flat list are found."""
    msgs = [HumanMessage(content="hi", id="1"), AIMessage(content="hey", id="2")]
    result = _find_messages_recursive(msgs)
    assert len(result) == 2
    assert result[0].id == "1"
    assert result[1].id == "2"


def test_single_message() -> None:
    """A single BaseMessage is returned directly."""
    msg = HumanMessage(content="hello", id="solo")
    result = _find_messages_recursive(msg)
    assert result == [msg]


def test_dict_values() -> None:
    """Messages inside dict values are found."""
    data = {
        "messages": [HumanMessage(content="a", id="1")],
        "other": "not a message",
    }
    result = _find_messages_recursive(data)
    assert len(result) == 1
    assert result[0].id == "1"


def test_nested_dict_of_dicts() -> None:
    """Messages inside dict → dict → list are found."""
    data = {"level1": {"level2": {"msgs": [HumanMessage(content="deep", id="deep")]}}}
    result = _find_messages_recursive(data)
    assert len(result) == 1
    assert result[0].id == "deep"


def test_pydantic_model_fields() -> None:
    """Messages inside Pydantic model fields are found."""
    model = InnerModel(messages=[HumanMessage(content="in model", id="m1")])
    result = _find_messages_recursive(model)
    assert len(result) == 1
    assert result[0].id == "m1"


def test_nested_pydantic_models() -> None:
    """Messages inside nested Pydantic model fields are found."""
    inner = InnerModel(messages=[AIMessage(content="nested", id="n1")])
    outer = OuterModel(inner=inner)
    result = _find_messages_recursive(outer)
    assert len(result) == 1
    assert result[0].id == "n1"


def test_dataclass_fields() -> None:
    """Messages inside dataclass fields are found."""
    dc = DataContainer(
        name="test",
        messages=[HumanMessage(content="dc msg", id="dc1")],
    )
    result = _find_messages_recursive(dc)
    assert len(result) == 1
    assert result[0].id == "dc1"


def test_dict_with_nested_pydantic_and_dataclass() -> None:
    """Mixed nesting: dict → Pydantic → messages, dict → dataclass → messages."""
    data = {
        "model": InnerModel(messages=[HumanMessage(content="from model", id="pm1")]),
        "container": DataContainer(
            name="dc",
            messages=[AIMessage(content="from dc", id="dc2")],
        ),
        "flat": [HumanMessage(content="flat", id="f1")],
    }
    result = _find_messages_recursive(data)
    ids = {m.id for m in result}
    assert ids == {"pm1", "dc2", "f1"}


def test_empty_and_none() -> None:
    """Edge cases: empty structures and non-container types return no messages."""
    assert _find_messages_recursive({}) == []
    assert _find_messages_recursive([]) == []
    assert _find_messages_recursive("a string") == []
    assert _find_messages_recursive(42) == []
    assert _find_messages_recursive(None) == []


def test_depth_limit() -> None:
    """Recursion stops at depth > 10 to prevent infinite loops."""
    # Build a dict nested 15 levels deep with a message at the bottom
    data: dict = {"msg": HumanMessage(content="very deep", id="deep")}
    for _ in range(15):
        data = {"nested": data}
    result = _find_messages_recursive(data)
    # The message is at depth 16 — beyond the limit of 10
    assert len(result) == 0


def test_messages_not_duplicated() -> None:
    """Same message appearing in multiple locations is returned each time."""
    msg = HumanMessage(content="shared", id="shared")
    data = {"a": [msg], "b": InnerModel(messages=[msg])}
    result = _find_messages_recursive(data)
    # Should find both occurrences — dedup is the caller's responsibility
    assert len(result) == 2


def test_mixed_message_types() -> None:
    """Both HumanMessage and AIMessage are found in the same structure."""
    data = {
        "human": HumanMessage(content="question", id="h1"),
        "ai": AIMessage(content="answer", id="a1"),
    }
    result = _find_messages_recursive(data)
    ids = {m.id for m in result}
    assert ids == {"h1", "a1"}
