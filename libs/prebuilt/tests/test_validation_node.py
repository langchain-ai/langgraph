import sys
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool as dec_tool
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

from langgraph.prebuilt import ValidationNode

pytestmark = pytest.mark.anyio


def my_function(some_val: int, some_other_val: str) -> str:
    return f"{some_val} - {some_other_val}"


class MyModel(BaseModel):
    some_val: int
    some_other_val: str


class MyModelV1(BaseModelV1):
    some_val: int
    some_other_val: str


@dec_tool
def my_tool(some_val: int, some_other_val: str) -> str:
    """Cool."""
    return f"{some_val} - {some_other_val}"


@pytest.mark.parametrize(
    "tool_schema",
    [
        my_function,
        MyModel,
        pytest.param(
            MyModelV1,
            marks=pytest.mark.skipif(
                sys.version_info >= (3, 14),
                reason="Pydantic v1 not supported in Python 3.14+",
            ),
        ),
        my_tool,
    ],
)
@pytest.mark.parametrize("use_message_key", [True, False])
async def test_validation_node(tool_schema: Any, use_message_key: bool):
    validation_node = ValidationNode([tool_schema])
    tool_name = getattr(tool_schema, "name", getattr(tool_schema, "__name__", None))
    inputs = [
        AIMessage(
            "hi?",
            tool_calls=[
                {
                    "name": tool_name,
                    "args": {"some_val": 1, "some_other_val": "foo"},
                    "id": "some 0",
                },
                {
                    "name": tool_name,
                    # Wrong type for some_val
                    "args": {"some_val": "bar", "some_other_val": "foo"},
                    "id": "some 1",
                },
            ],
        ),
    ]
    if use_message_key:
        inputs = {"messages": inputs}
    result = await validation_node.ainvoke(inputs)
    if use_message_key:
        result = result["messages"]

    def check_results(messages: list):
        assert len(messages) == 2
        assert all(m.type == "tool" for m in messages)
        assert not messages[0].additional_kwargs.get("is_error")
        assert messages[1].additional_kwargs.get("is_error")

    check_results(result)
    result_sync = validation_node.invoke(inputs)
    if use_message_key:
        result_sync = result_sync["messages"]
    check_results(result_sync)


async def test_validation_node_unknown_tool_name():
    """A tool call naming an unknown tool must produce an error ToolMessage
    instead of raising an uncaught KeyError."""
    validation_node = ValidationNode([MyModel])
    inputs = [
        AIMessage(
            "hi?",
            tool_calls=[
                {
                    "name": "not_a_tool",
                    "args": {"some_val": 1},
                    "id": "call 0",
                },
                {
                    "name": "MyModel",
                    "args": {"some_val": 1, "some_other_val": "foo"},
                    "id": "call 1",
                },
            ],
        ),
    ]

    def check_results(messages: list):
        assert len(messages) == 2
        assert all(m.type == "tool" for m in messages)
        error_msg = messages[0]
        assert error_msg.status == "error"
        assert error_msg.additional_kwargs.get("is_error")
        assert error_msg.name == "not_a_tool"
        assert error_msg.tool_call_id == "call 0"
        assert "not_a_tool is not a valid tool" in error_msg.content
        assert "MyModel" in error_msg.content
        # The valid call is still validated normally.
        assert not messages[1].additional_kwargs.get("is_error")

    check_results(validation_node.invoke(inputs))
    check_results(await validation_node.ainvoke(inputs))
