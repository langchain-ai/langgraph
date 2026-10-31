"""Functional API with postponed (string) annotations.

The types below must live at module level so that the string annotations
produced by `from __future__ import annotations` can be resolved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph._internal import _serde
from langgraph.func import entrypoint

if TYPE_CHECKING:
    from decimal import Decimal


class InputState(TypedDict):
    a: int


class OutputModel(BaseModel):
    b: str


def test_entrypoint_schemas_with_postponed_annotations() -> None:
    @entrypoint()
    def wf(inputs: InputState) -> OutputModel:
        return OutputModel(b=str(inputs["a"]))

    assert wf.get_input_jsonschema() == {
        "$defs": {
            "InputState": {
                "properties": {"a": {"title": "A", "type": "integer"}},
                "required": ["a"],
                "title": "InputState",
                "type": "object",
            }
        },
        "$ref": "#/$defs/InputState",
        "title": "LangGraphInput",
    }
    assert wf.get_output_jsonschema() == {
        "properties": {"b": {"title": "B", "type": "string"}},
        "required": ["b"],
        "title": "OutputModel",
        "type": "object",
    }
    assert wf.invoke({"a": 1}) == OutputModel(b="1")


@pytest.mark.anyio
async def test_entrypoint_schemas_with_postponed_annotations_async() -> None:
    @entrypoint()
    async def wf(inputs: InputState) -> OutputModel:
        return OutputModel(b=str(inputs["a"]))

    assert wf.get_output_jsonschema()["title"] == "OutputModel"
    assert await wf.ainvoke({"a": 1}) == OutputModel(b="1")


def test_entrypoint_final_with_postponed_annotations() -> None:
    previous_values = []

    @entrypoint(checkpointer=InMemorySaver())
    def wf(inputs: int, *, previous: Any) -> entrypoint.final[int, OutputModel]:
        previous_values.append(previous)
        return entrypoint.final(value=inputs, save=OutputModel(b=str(inputs)))

    assert wf.get_output_jsonschema() == {
        "title": "LangGraphOutput",
        "type": "integer",
    }

    config = {"configurable": {"thread_id": "1"}}
    assert wf.invoke(1, config) == 1
    assert wf.invoke(2, config) == 2
    assert previous_values == [None, OutputModel(b="1")]


def test_entrypoint_serde_allowlist_with_postponed_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_serde, "STRICT_MSGPACK_ENABLED", True)

    @entrypoint()
    def wf(inputs: InputState, *, previous: Any) -> OutputModel:
        return OutputModel(b=str(inputs["a"]))

    assert (__name__, "OutputModel") in wf._serde_allowlist


def test_entrypoint_unresolvable_postponed_annotations() -> None:
    # annotations that can't be resolved fall back to the unresolved signature
    @entrypoint()
    def wf(inputs: Decimal) -> int:
        return 1

    assert wf.invoke(1) == 1
