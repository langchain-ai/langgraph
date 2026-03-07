"""Regression tests for reducer detection with non-trailing metadata.

Before this fix, `_is_field_binop` only checked the *last* metadata item in an
`Annotated` type.  If additional metadata (doc strings, Pydantic `Field`, etc.)
appeared after the reducer callable, the reducer was silently ignored and the
field fell back to `LastValue` — losing aggregation semantics.
"""

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from langgraph.graph import END, START, StateGraph

# --- helpers ---


def _add_lists(a: list, b: list) -> list:
    return a + b


# --- tests ---


def test_reducer_as_last_metadata() -> None:
    """Baseline: reducer is the last (only) metadata item — always worked."""

    class State(BaseModel):
        items: Annotated[list[str], operator.add]

    graph = StateGraph(State)
    graph.add_node("step", lambda s: {"items": ["b"]})
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    result = graph.compile().invoke({"items": ["a"]})
    assert result["items"] == ["a", "b"]


def test_reducer_before_string_annotation() -> None:
    """Reducer callable precedes a doc-string annotation — must still be found."""

    class State(BaseModel):
        items: Annotated[list[str], operator.add, "item accumulator"]

    graph = StateGraph(State)
    graph.add_node("step", lambda s: {"items": ["b"]})
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    result = graph.compile().invoke({"items": ["a"]})
    assert result["items"] == ["a", "b"]


def test_reducer_between_metadata() -> None:
    """Reducer sits between two non-callable metadata items."""

    class State(BaseModel):
        items: Annotated[list[str], "doc string", _add_lists, "another annotation"]

    graph = StateGraph(State)
    graph.add_node("step", lambda s: {"items": ["b"]})
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    result = graph.compile().invoke({"items": ["a"]})
    assert result["items"] == ["a", "b"]


def test_reducer_with_field_metadata_after() -> None:
    """Reducer followed by Pydantic Field — previously lost the reducer."""

    class State(BaseModel):
        items: Annotated[
            list[str], operator.add, Field(description="accumulated items")
        ]

    graph = StateGraph(State)
    graph.add_node("step", lambda s: {"items": ["b"]})
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    result = graph.compile().invoke({"items": ["a"]})
    assert result["items"] == ["a", "b"]


def test_custom_reducer_not_last() -> None:
    """Custom 2-arg reducer that is not the last metadata item."""

    def sum_ints(a: int, b: int) -> int:
        return a + b

    class State(BaseModel):
        total: Annotated[int, sum_ints, "running total"]

    graph = StateGraph(State)
    graph.add_node("step", lambda s: {"total": 5})
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    result = graph.compile().invoke({"total": 10})
    assert result["total"] == 15


def test_multiple_nodes_with_non_trailing_reducer() -> None:
    """Multi-step graph: reducer works correctly across node boundaries."""

    class State(BaseModel):
        log: Annotated[list[str], operator.add, "audit log"]

    graph = StateGraph(State)
    graph.add_node("a", lambda s: {"log": ["from_a"]})
    graph.add_node("b", lambda s: {"log": ["from_b"]})
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    result = graph.compile().invoke({"log": ["init"]})
    assert result["log"] == ["init", "from_a", "from_b"]


def test_no_reducer_still_falls_through() -> None:
    """Field with only non-callable metadata falls back to LastValue."""

    class State(BaseModel):
        name: Annotated[str, "just a doc string"]

    graph = StateGraph(State)
    graph.add_node("step", lambda s: {"name": "updated"})
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    result = graph.compile().invoke({"name": "original"})
    assert result["name"] == "updated"
