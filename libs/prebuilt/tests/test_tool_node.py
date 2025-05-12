from typing import (
    Annotated,
    Any,
    Union,
)

import pytest
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.tools import tool as dec_tool
from pydantic import BaseModel, ValidationError
from pydantic.v1 import ValidationError as ValidationErrorV1

from langgraph.errors import NodeInterrupt
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import TOOL_CALL_ERROR_TEMPLATE
from langgraph.types import Command, Send
from tests.conftest import IS_LANGCHAIN_CORE_030_OR_GREATER

pytestmark = pytest.mark.anyio


def tool1(some_val: int, some_other_val: str) -> str:
    """Tool 1 docstring."""
    if some_val == 0:
        raise ValueError("Test error")
    return f"{some_val} - {some_other_val}"


async def tool2(some_val: int, some_other_val: str) -> str:
    """Tool 2 docstring."""
    if some_val == 0:
        raise ToolException("Test error")
    return f"tool2: {some_val} - {some_other_val}"


async def tool3(some_val: int, some_other_val: str) -> str:
    """Tool 3 docstring."""
    return [
        {"key_1": some_val, "key_2": "foo"},
        {"key_1": some_other_val, "key_2": "baz"},
    ]


async def tool4(some_val: int, some_other_val: str) -> str:
    """Tool 4 docstring."""
    return [
        {"type": "image_url", "image_url": {"url": "abdc"}},
    ]


@dec_tool
def tool5(some_val: int):
    """Tool 5 docstring."""
    raise ToolException("Test error")


tool5.handle_tool_error = "foo"


async def test_tool_node():
    result = ToolNode([tool1]).invoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool1",
                            "args": {"some_val": 1, "some_other_val": "foo"},
                            "id": "some 0",
                        }
                    ],
                )
            ]
        }
    )

    tool_message: ToolMessage = result["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.content == "1 - foo"
    assert tool_message.tool_call_id == "some 0"

    result2 = await ToolNode([tool2]).ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool2",
                            "args": {"some_val": 2, "some_other_val": "bar"},
                            "id": "some 1",
                        }
                    ],
                )
            ]
        }
    )

    tool_message: ToolMessage = result2["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.content == "tool2: 2 - bar"

    # list of dicts tool content
    result3 = await ToolNode([tool3]).ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool3",
                            "args": {"some_val": 2, "some_other_val": "bar"},
                            "id": "some 2",
                        }
                    ],
                )
            ]
        }
    )
    tool_message: ToolMessage = result3["messages"][-1]
    assert tool_message.type == "tool"
    assert (
        tool_message.content
        == '[{"key_1": 2, "key_2": "foo"}, {"key_1": "bar", "key_2": "baz"}]'
    )
    assert tool_message.tool_call_id == "some 2"

    # list of content blocks tool content
    result4 = await ToolNode([tool4]).ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool4",
                            "args": {"some_val": 2, "some_other_val": "bar"},
                            "id": "some 3",
                        }
                    ],
                )
            ]
        }
    )
    tool_message: ToolMessage = result4["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.content == [{"type": "image_url", "image_url": {"url": "abdc"}}]
    assert tool_message.tool_call_id == "some 3"


async def test_tool_node_tool_call_input():
    # Single tool call
    tool_call_1 = {
        "name": "tool1",
        "args": {"some_val": 1, "some_other_val": "foo"},
        "id": "some 0",
        "type": "tool_call",
    }
    result = ToolNode([tool1]).invoke([tool_call_1])
    assert result["messages"] == [
        ToolMessage(content="1 - foo", tool_call_id="some 0", name="tool1"),
    ]

    # Multiple tool calls
    tool_call_2 = {
        "name": "tool1",
        "args": {"some_val": 2, "some_other_val": "bar"},
        "id": "some 1",
        "type": "tool_call",
    }
    result = ToolNode([tool1]).invoke([tool_call_1, tool_call_2])
    assert result["messages"] == [
        ToolMessage(content="1 - foo", tool_call_id="some 0", name="tool1"),
        ToolMessage(content="2 - bar", tool_call_id="some 1", name="tool1"),
    ]

    # Test with unknown tool
    tool_call_3 = tool_call_1.copy()
    tool_call_3["name"] = "tool2"
    result = ToolNode([tool1]).invoke([tool_call_1, tool_call_3])
    assert result["messages"] == [
        ToolMessage(content="1 - foo", tool_call_id="some 0", name="tool1"),
        ToolMessage(
            content="Error: tool2 is not a valid tool, try one of [tool1].",
            name="tool2",
            tool_call_id="some 0",
            status="error",
        ),
    ]


async def test_tool_node_error_handling():
    def handle_all(e: Union[ValueError, ToolException, ValidationError]):
        return TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))

    # test catching all exceptions, via:
    # - handle_tool_errors = True
    # - passing a tuple of all exceptions
    # - passing a callable with all exceptions in the signature
    for handle_tool_errors in (
        True,
        (ValueError, ToolException, ValidationError),
        handle_all,
    ):
        result_error = await ToolNode(
            [tool1, tool2, tool3], handle_tool_errors=handle_tool_errors
        ).ainvoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "tool1",
                                "args": {"some_val": 0, "some_other_val": "foo"},
                                "id": "some id",
                            },
                            {
                                "name": "tool2",
                                "args": {"some_val": 0, "some_other_val": "bar"},
                                "id": "some other id",
                            },
                            {
                                "name": "tool3",
                                "args": {"some_val": 0},
                                "id": "another id",
                            },
                        ],
                    )
                ]
            }
        )

        assert all(m.type == "tool" for m in result_error["messages"])
        assert all(m.status == "error" for m in result_error["messages"])
        assert (
            result_error["messages"][0].content
            == f"Error: {repr(ValueError('Test error'))}\n Please fix your mistakes."
        )
        assert (
            result_error["messages"][1].content
            == f"Error: {repr(ToolException('Test error'))}\n Please fix your mistakes."
        )
        assert (
            "ValidationError" in result_error["messages"][2].content
            or "validation error" in result_error["messages"][2].content
        )

        assert result_error["messages"][0].tool_call_id == "some id"
        assert result_error["messages"][1].tool_call_id == "some other id"
        assert result_error["messages"][2].tool_call_id == "another id"


async def test_tool_node_error_handling_callable():
    def handle_value_error(e: ValueError):
        return "Value error"

    def handle_tool_exception(e: ToolException):
        return "Tool exception"

    for handle_tool_errors in ("Value error", handle_value_error):
        result_error = await ToolNode(
            [tool1], handle_tool_errors=handle_tool_errors
        ).ainvoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "tool1",
                                "args": {"some_val": 0, "some_other_val": "foo"},
                                "id": "some id",
                            },
                        ],
                    )
                ]
            }
        )
        tool_message: ToolMessage = result_error["messages"][-1]
        assert tool_message.type == "tool"
        assert tool_message.status == "error"
        assert tool_message.content == "Value error"

    # test raising for an unhandled exception, via:
    # - passing a tuple of all exceptions
    # - passing a callable with all exceptions in the signature
    for handle_tool_errors in ((ValueError,), handle_value_error):
        with pytest.raises(ToolException) as exc_info:
            await ToolNode(
                [tool1, tool2], handle_tool_errors=handle_tool_errors
            ).ainvoke(
                {
                    "messages": [
                        AIMessage(
                            "hi?",
                            tool_calls=[
                                {
                                    "name": "tool1",
                                    "args": {"some_val": 0, "some_other_val": "foo"},
                                    "id": "some id",
                                },
                                {
                                    "name": "tool2",
                                    "args": {"some_val": 0, "some_other_val": "bar"},
                                    "id": "some other id",
                                },
                            ],
                        )
                    ]
                }
            )
        assert str(exc_info.value) == "Test error"

    for handle_tool_errors in ((ToolException,), handle_tool_exception):
        with pytest.raises(ValueError) as exc_info:
            await ToolNode(
                [tool1, tool2], handle_tool_errors=handle_tool_errors
            ).ainvoke(
                {
                    "messages": [
                        AIMessage(
                            "hi?",
                            tool_calls=[
                                {
                                    "name": "tool1",
                                    "args": {"some_val": 0, "some_other_val": "foo"},
                                    "id": "some id",
                                },
                                {
                                    "name": "tool2",
                                    "args": {"some_val": 0, "some_other_val": "bar"},
                                    "id": "some other id",
                                },
                            ],
                        )
                    ]
                }
            )
        assert str(exc_info.value) == "Test error"


async def test_tool_node_handle_tool_errors_false():
    with pytest.raises(ValueError) as exc_info:
        ToolNode([tool1], handle_tool_errors=False).invoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "tool1",
                                "args": {"some_val": 0, "some_other_val": "foo"},
                                "id": "some id",
                            }
                        ],
                    )
                ]
            }
        )

    assert str(exc_info.value) == "Test error"

    with pytest.raises(ToolException):
        await ToolNode([tool2], handle_tool_errors=False).ainvoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "tool2",
                                "args": {"some_val": 0, "some_other_val": "bar"},
                                "id": "some id",
                            }
                        ],
                    )
                ]
            }
        )

    assert str(exc_info.value) == "Test error"

    # test validation errors get raised if handle_tool_errors is False
    with pytest.raises((ValidationError, ValidationErrorV1)):
        ToolNode([tool1], handle_tool_errors=False).invoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "tool1",
                                "args": {"some_val": 0},
                                "id": "some id",
                            }
                        ],
                    )
                ]
            }
        )


def test_tool_node_individual_tool_error_handling():
    # test error handling on individual tools (and that it overrides overall error handling!)
    result_individual_tool_error_handler = ToolNode(
        [tool5], handle_tool_errors="bar"
    ).invoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool5",
                            "args": {"some_val": 0},
                            "id": "some 0",
                        }
                    ],
                )
            ]
        }
    )

    tool_message: ToolMessage = result_individual_tool_error_handler["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.status == "error"
    assert tool_message.content == "foo"
    assert tool_message.tool_call_id == "some 0"


def test_tool_node_incorrect_tool_name():
    result_incorrect_name = ToolNode([tool1, tool2]).invoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool3",
                            "args": {"some_val": 1, "some_other_val": "foo"},
                            "id": "some 0",
                        }
                    ],
                )
            ]
        }
    )

    tool_message: ToolMessage = result_incorrect_name["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.status == "error"
    assert (
        tool_message.content
        == "Error: tool3 is not a valid tool, try one of [tool1, tool2]."
    )
    assert tool_message.tool_call_id == "some 0"


def test_tool_node_node_interrupt():
    def tool_interrupt(some_val: int) -> str:
        """Tool docstring."""
        raise NodeInterrupt("foo")

    def handle(e: NodeInterrupt):
        return "handled"

    for handle_tool_errors in (True, (NodeInterrupt,), "handled", handle, False):
        node = ToolNode([tool_interrupt], handle_tool_errors=handle_tool_errors)
        with pytest.raises(NodeInterrupt) as exc_info:
            node.invoke(
                {
                    "messages": [
                        AIMessage(
                            "hi?",
                            tool_calls=[
                                {
                                    "name": "tool_interrupt",
                                    "args": {"some_val": 0},
                                    "id": "some 0",
                                }
                            ],
                        )
                    ]
                }
            )
            assert exc_info.value == "foo"


@pytest.mark.skipif(
    not IS_LANGCHAIN_CORE_030_OR_GREATER,
    reason="Langchain core 0.3.0 or greater is required",
)
@pytest.mark.parametrize("input_type", ["dict", "tool_calls"])
async def test_tool_node_command(input_type: str):
    from langchain_core.tools.base import InjectedToolCallId

    @dec_tool
    def transfer_to_bob(tool_call_id: Annotated[str, InjectedToolCallId]):
        """Transfer to Bob"""
        return Command(
            update={
                "messages": [
                    ToolMessage(content="Transferred to Bob", tool_call_id=tool_call_id)
                ]
            },
            goto="bob",
            graph=Command.PARENT,
        )

    @dec_tool
    async def async_transfer_to_bob(tool_call_id: Annotated[str, InjectedToolCallId]):
        """Transfer to Bob"""
        return Command(
            update={
                "messages": [
                    ToolMessage(content="Transferred to Bob", tool_call_id=tool_call_id)
                ]
            },
            goto="bob",
            graph=Command.PARENT,
        )

    class CustomToolSchema(BaseModel):
        tool_call_id: Annotated[str, InjectedToolCallId]

    class MyCustomTool(BaseTool):
        def _run(*args: Any, **kwargs: Any):
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Transferred to Bob",
                            tool_call_id=kwargs["tool_call_id"],
                        )
                    ]
                },
                goto="bob",
                graph=Command.PARENT,
            )

        async def _arun(*args: Any, **kwargs: Any):
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Transferred to Bob",
                            tool_call_id=kwargs["tool_call_id"],
                        )
                    ]
                },
                goto="bob",
                graph=Command.PARENT,
            )

    custom_tool = MyCustomTool(
        name="custom_transfer_to_bob",
        description="Transfer to bob",
        args_schema=CustomToolSchema,
    )
    async_custom_tool = MyCustomTool(
        name="async_custom_transfer_to_bob",
        description="Transfer to bob",
        args_schema=CustomToolSchema,
    )

    # test mixing regular tools and tools returning commands
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    tool_calls = [
        {"args": {"a": 1, "b": 2}, "id": "1", "name": "add", "type": "tool_call"},
        {"args": {}, "id": "2", "name": "transfer_to_bob", "type": "tool_call"},
    ]
    if input_type == "dict":
        input_ = {"messages": [AIMessage("", tool_calls=tool_calls)]}
    elif input_type == "tool_calls":
        input_ = tool_calls
    result = ToolNode([add, transfer_to_bob]).invoke(input_)

    assert result == [
        {
            "messages": [
                ToolMessage(
                    content="3",
                    tool_call_id="1",
                    name="add",
                )
            ]
        },
        Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id="2",
                        name="transfer_to_bob",
                    )
                ]
            },
            goto="bob",
            graph=Command.PARENT,
        ),
    ]

    # test tools returning commands

    # test sync tools
    for tool in [transfer_to_bob, custom_tool]:
        result = ToolNode([tool]).invoke(
            {
                "messages": [
                    AIMessage(
                        "", tool_calls=[{"args": {}, "id": "1", "name": tool.name}]
                    )
                ]
            }
        )
        assert result == [
            Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Transferred to Bob",
                            tool_call_id="1",
                            name=tool.name,
                        )
                    ]
                },
                goto="bob",
                graph=Command.PARENT,
            )
        ]

    # test async tools
    for tool in [async_transfer_to_bob, async_custom_tool]:
        result = await ToolNode([tool]).ainvoke(
            {
                "messages": [
                    AIMessage(
                        "", tool_calls=[{"args": {}, "id": "1", "name": tool.name}]
                    )
                ]
            }
        )
        assert result == [
            Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Transferred to Bob",
                            tool_call_id="1",
                            name=tool.name,
                        )
                    ]
                },
                goto="bob",
                graph=Command.PARENT,
            )
        ]

    # test multiple commands
    result = ToolNode([transfer_to_bob, custom_tool]).invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[
                        {"args": {}, "id": "1", "name": "transfer_to_bob"},
                        {"args": {}, "id": "2", "name": "custom_transfer_to_bob"},
                    ],
                )
            ]
        }
    )
    assert result == [
        Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id="1",
                        name="transfer_to_bob",
                    )
                ]
            },
            goto="bob",
            graph=Command.PARENT,
        ),
        Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id="2",
                        name="custom_transfer_to_bob",
                    )
                ]
            },
            goto="bob",
            graph=Command.PARENT,
        ),
    ]

    # test validation (mismatch between input type and command.update type)
    with pytest.raises(ValueError):

        @dec_tool
        def list_update_tool(tool_call_id: Annotated[str, InjectedToolCallId]):
            """My tool"""
            return Command(
                update=[ToolMessage(content="foo", tool_call_id=tool_call_id)]
            )

        ToolNode([list_update_tool]).invoke(
            {
                "messages": [
                    AIMessage(
                        "",
                        tool_calls=[
                            {"args": {}, "id": "1", "name": "list_update_tool"}
                        ],
                    )
                ]
            }
        )

    # test validation (missing tool message in the update for current graph)
    with pytest.raises(ValueError):

        @dec_tool
        def no_update_tool():
            """My tool"""
            return Command(update={"messages": []})

        ToolNode([no_update_tool]).invoke(
            {
                "messages": [
                    AIMessage(
                        "",
                        tool_calls=[{"args": {}, "id": "1", "name": "no_update_tool"}],
                    )
                ]
            }
        )

    # test validation (tool message with a wrong tool call ID)
    with pytest.raises(ValueError):

        @dec_tool
        def mismatching_tool_call_id_tool():
            """My tool"""
            return Command(
                update={"messages": [ToolMessage(content="foo", tool_call_id="2")]}
            )

        ToolNode([mismatching_tool_call_id_tool]).invoke(
            {
                "messages": [
                    AIMessage(
                        "",
                        tool_calls=[
                            {
                                "args": {},
                                "id": "1",
                                "name": "mismatching_tool_call_id_tool",
                            }
                        ],
                    )
                ]
            }
        )

    # test validation (missing tool message in the update for parent graph is OK)
    @dec_tool
    def node_update_parent_tool():
        """No update"""
        return Command(update={"messages": []}, graph=Command.PARENT)

    assert ToolNode([node_update_parent_tool]).invoke(
        {
            "messages": [
                AIMessage(
                    "",
                    tool_calls=[
                        {"args": {}, "id": "1", "name": "node_update_parent_tool"}
                    ],
                )
            ]
        }
    ) == [Command(update={"messages": []}, graph=Command.PARENT)]


@pytest.mark.skipif(
    not IS_LANGCHAIN_CORE_030_OR_GREATER,
    reason="Langchain core 0.3.0 or greater is required",
)
async def test_tool_node_command_list_input():
    from langchain_core.tools.base import InjectedToolCallId

    @dec_tool
    def transfer_to_bob(tool_call_id: Annotated[str, InjectedToolCallId]):
        """Transfer to Bob"""
        return Command(
            update=[
                ToolMessage(content="Transferred to Bob", tool_call_id=tool_call_id)
            ],
            goto="bob",
            graph=Command.PARENT,
        )

    @dec_tool
    async def async_transfer_to_bob(tool_call_id: Annotated[str, InjectedToolCallId]):
        """Transfer to Bob"""
        return Command(
            update=[
                ToolMessage(content="Transferred to Bob", tool_call_id=tool_call_id)
            ],
            goto="bob",
            graph=Command.PARENT,
        )

    class CustomToolSchema(BaseModel):
        tool_call_id: Annotated[str, InjectedToolCallId]

    class MyCustomTool(BaseTool):
        def _run(*args: Any, **kwargs: Any):
            return Command(
                update=[
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id=kwargs["tool_call_id"],
                    )
                ],
                goto="bob",
                graph=Command.PARENT,
            )

        async def _arun(*args: Any, **kwargs: Any):
            return Command(
                update=[
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id=kwargs["tool_call_id"],
                    )
                ],
                goto="bob",
                graph=Command.PARENT,
            )

    custom_tool = MyCustomTool(
        name="custom_transfer_to_bob",
        description="Transfer to bob",
        args_schema=CustomToolSchema,
    )
    async_custom_tool = MyCustomTool(
        name="async_custom_transfer_to_bob",
        description="Transfer to bob",
        args_schema=CustomToolSchema,
    )

    # test mixing regular tools and tools returning commands
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    result = ToolNode([add, transfer_to_bob]).invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"args": {"a": 1, "b": 2}, "id": "1", "name": "add"},
                    {"args": {}, "id": "2", "name": "transfer_to_bob"},
                ],
            )
        ]
    )

    assert result == [
        [
            ToolMessage(
                content="3",
                tool_call_id="1",
                name="add",
            )
        ],
        Command(
            update=[
                ToolMessage(
                    content="Transferred to Bob",
                    tool_call_id="2",
                    name="transfer_to_bob",
                )
            ],
            goto="bob",
            graph=Command.PARENT,
        ),
    ]

    # test tools returning commands

    # test sync tools
    for tool in [transfer_to_bob, custom_tool]:
        result = ToolNode([tool]).invoke(
            [AIMessage("", tool_calls=[{"args": {}, "id": "1", "name": tool.name}])]
        )
        assert result == [
            Command(
                update=[
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id="1",
                        name=tool.name,
                    )
                ],
                goto="bob",
                graph=Command.PARENT,
            )
        ]

    # test async tools
    for tool in [async_transfer_to_bob, async_custom_tool]:
        result = await ToolNode([tool]).ainvoke(
            [AIMessage("", tool_calls=[{"args": {}, "id": "1", "name": tool.name}])]
        )
        assert result == [
            Command(
                update=[
                    ToolMessage(
                        content="Transferred to Bob",
                        tool_call_id="1",
                        name=tool.name,
                    )
                ],
                goto="bob",
                graph=Command.PARENT,
            )
        ]

    # test multiple commands
    result = ToolNode([transfer_to_bob, custom_tool]).invoke(
        [
            AIMessage(
                "",
                tool_calls=[
                    {"args": {}, "id": "1", "name": "transfer_to_bob"},
                    {"args": {}, "id": "2", "name": "custom_transfer_to_bob"},
                ],
            )
        ]
    )
    assert result == [
        Command(
            update=[
                ToolMessage(
                    content="Transferred to Bob",
                    tool_call_id="1",
                    name="transfer_to_bob",
                )
            ],
            goto="bob",
            graph=Command.PARENT,
        ),
        Command(
            update=[
                ToolMessage(
                    content="Transferred to Bob",
                    tool_call_id="2",
                    name="custom_transfer_to_bob",
                )
            ],
            goto="bob",
            graph=Command.PARENT,
        ),
    ]

    # test validation (mismatch between input type and command.update type)
    with pytest.raises(ValueError):

        @dec_tool
        def list_update_tool(tool_call_id: Annotated[str, InjectedToolCallId]):
            """My tool"""
            return Command(
                update={
                    "messages": [ToolMessage(content="foo", tool_call_id=tool_call_id)]
                }
            )

        ToolNode([list_update_tool]).invoke(
            [
                AIMessage(
                    "",
                    tool_calls=[{"args": {}, "id": "1", "name": "list_update_tool"}],
                )
            ]
        )

    # test validation (missing tool message in the update for current graph)
    with pytest.raises(ValueError):

        @dec_tool
        def no_update_tool():
            """My tool"""
            return Command(update=[])

        ToolNode([no_update_tool]).invoke(
            [
                AIMessage(
                    "",
                    tool_calls=[{"args": {}, "id": "1", "name": "no_update_tool"}],
                )
            ]
        )

    # test validation (tool message with a wrong tool call ID)
    with pytest.raises(ValueError):

        @dec_tool
        def mismatching_tool_call_id_tool():
            """My tool"""
            return Command(update=[ToolMessage(content="foo", tool_call_id="2")])

        ToolNode([mismatching_tool_call_id_tool]).invoke(
            [
                AIMessage(
                    "",
                    tool_calls=[
                        {"args": {}, "id": "1", "name": "mismatching_tool_call_id_tool"}
                    ],
                )
            ]
        )

    # test validation (missing tool message in the update for parent graph is OK)
    @dec_tool
    def node_update_parent_tool():
        """No update"""
        return Command(update=[], graph=Command.PARENT)

    assert ToolNode([node_update_parent_tool]).invoke(
        [
            AIMessage(
                "",
                tool_calls=[{"args": {}, "id": "1", "name": "node_update_parent_tool"}],
            )
        ]
    ) == [Command(update=[], graph=Command.PARENT)]


def test_tool_node_parent_command_with_send():
    from langchain_core.tools.base import InjectedToolCallId

    @dec_tool
    def transfer_to_alice(tool_call_id: Annotated[str, InjectedToolCallId]):
        """Transfer to Alice"""
        return Command(
            goto=[
                Send(
                    "alice",
                    {
                        "messages": [
                            ToolMessage(
                                content="Transferred to Alice",
                                name="transfer_to_alice",
                                tool_call_id=tool_call_id,
                            )
                        ]
                    },
                )
            ],
            graph=Command.PARENT,
        )

    @dec_tool
    def transfer_to_bob(tool_call_id: Annotated[str, InjectedToolCallId]):
        """Transfer to Bob"""
        return Command(
            goto=[
                Send(
                    "bob",
                    {
                        "messages": [
                            ToolMessage(
                                content="Transferred to Bob",
                                name="transfer_to_bob",
                                tool_call_id=tool_call_id,
                            )
                        ]
                    },
                )
            ],
            graph=Command.PARENT,
        )

    tool_calls = [
        {"args": {}, "id": "1", "name": "transfer_to_alice", "type": "tool_call"},
        {"args": {}, "id": "2", "name": "transfer_to_bob", "type": "tool_call"},
    ]

    result = ToolNode([transfer_to_alice, transfer_to_bob]).invoke(
        [AIMessage("", tool_calls=tool_calls)]
    )

    assert result == [
        Command(
            goto=[
                Send(
                    "alice",
                    {
                        "messages": [
                            ToolMessage(
                                content="Transferred to Alice",
                                name="transfer_to_alice",
                                tool_call_id="1",
                            )
                        ]
                    },
                ),
                Send(
                    "bob",
                    {
                        "messages": [
                            ToolMessage(
                                content="Transferred to Bob",
                                name="transfer_to_bob",
                                tool_call_id="2",
                            )
                        ]
                    },
                ),
            ],
            graph=Command.PARENT,
        )
    ]
