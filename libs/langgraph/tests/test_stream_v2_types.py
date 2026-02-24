"""Type-checking tests for v2 streaming.

This file is meant to be checked by mypy/pyright, not executed as pytest.
It verifies that type narrowing works correctly with the StreamPart union.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, assert_type  # noqa: F401

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import (
    CheckpointPayload,
    DebugPayload,
    TaskPayload,
    TaskResultPayload,
)


class State(TypedDict):
    value: str
    items: Annotated[list[str], operator.add]


def check_type_narrowing() -> None:
    """Verify that type narrowing works with part["type"] checks."""
    builder = StateGraph(State)

    def node(state: State) -> dict:
        return {"value": "x", "items": ["a"]}

    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile()

    for part in graph.stream({"value": "", "items": []}, version="v2"):
        # part should be StreamPart
        assert_type(part["ns"], tuple[str, ...])

        if part["type"] == "values":
            data: dict[str, Any] = part["data"]
            assert isinstance(data, dict)
        elif part["type"] == "updates":
            data_u: dict[str, Any] = part["data"]
            assert isinstance(data_u, dict)
        elif part["type"] == "messages":
            data_m: tuple[AnyMessage, dict[str, Any]] = part["data"]
            assert isinstance(data_m, tuple)
        elif part["type"] == "custom":
            data_c: Any = part["data"]
            _ = data_c
        elif part["type"] == "checkpoints":
            data_ck: CheckpointPayload = part["data"]
            assert isinstance(data_ck, dict)
        elif part["type"] == "tasks":
            data_t: TaskPayload | TaskResultPayload = part["data"]
            assert isinstance(data_t, dict)
        elif part["type"] == "debug":
            data_d: DebugPayload = part["data"]
            assert isinstance(data_d, dict)
