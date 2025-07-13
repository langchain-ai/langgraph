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
from langgraph.store.base import BaseStore
from langgraph.types import Command, Send

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


def test_tool_node_runtime_tools_from_config():
    """ToolNode should use tools provided in runtime config when not defined at init."""

    node = ToolNode(config_tools_key="extra_tools")

    # Using tool1 which expects (some_val: int, some_other_val: str)
    tool_call = {
        "name": "tool1",
        "args": {"some_val": 5, "some_other_val": "test"},
        "id": "call_1",
        "type": "tool_call",
    }
    messages = [AIMessage("", tool_calls=[tool_call])]

    result = node.invoke(
        {"messages": messages},
        config={"configurable": {"extra_tools": [tool1]}},
    )

    tool_messages = result["messages"]
    assert isinstance(tool_messages[0], ToolMessage)
    assert tool_messages[0].content == "5 - test"


async def test_tool_node_merge_init_and_runtime_tools():
    """Runtime tools should merge with init-time tools."""

    # Start with tool1 at compile time
    node = ToolNode(tools=[tool1], config_tools_key="tools")

    # Runtime provides tool2 (async tool)
    tool_call = {
        "name": "tool2",
        "args": {"some_val": 3, "some_other_val": "bar"},
        "id": "call_2",
        "type": "tool_call",
    }
    messages = [AIMessage("", tool_calls=[tool_call])]

    result = await node.ainvoke(
        {"messages": messages},
        config={"configurable": {"tools": [tool2]}},
    )

    tool_messages = result["messages"]
    assert tool_messages[0].content == "tool2: 3 - bar"


def test_tool_node_constructor_requires_tool_source():
    """It should be an error to create ToolNode without any tool source."""

    with pytest.raises(ValueError) as exc_info:
        ToolNode(config_tools_key=None)
    
    assert "requires at least one source of tools" in str(exc_info.value)


def test_tool_node_runtime_tools_with_injected_state():
    """Test that runtime tools with InjectedState work correctly."""
    from langgraph.prebuilt import InjectedState
    
    @dec_tool
    def get_user_info(query: str, state: Annotated[dict, InjectedState]) -> str:
        """Get info based on user's name from state"""
        user_name = state.get("user_name", "Unknown")
        return f"User {user_name} asked: {query}"
    
    # Create ToolNode with runtime tools only
    node = ToolNode(config_tools_key="tools")
    
    # Create tool call - note that LLM only generates 'query' arg, not 'state'
    tool_call = {
        "name": "get_user_info",
        "args": {"query": "what's the weather?"},  # No 'state' arg here
        "id": "call_1",
        "type": "tool_call",
    }
    
    state = {
        "messages": [AIMessage("", tool_calls=[tool_call])],
        "user_name": "Alice"
    }
    
    # Verify that runtime tools with InjectedState are handled correctly
    result = node.invoke(
        state,
        config={"configurable": {"tools": [get_user_info]}}
    )
    
    # Assertion verifies correct behavior
    tool_message = result["messages"][0]
    assert tool_message.content == "User Alice asked: what's the weather?"


async def test_tool_node_runtime_tools_with_injected_store():
    """Test that runtime tools with InjectedStore work correctly."""
    from langgraph.prebuilt import InjectedStore
    from langgraph.store.memory import InMemoryStore
    
    @dec_tool
    def save_data(key: str, value: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
        """Save data to store"""
        store.put(("data",), key, {"value": value})
        return f"Saved {value} to {key}"
    
    # Create store and ToolNode with runtime tools only
    store = InMemoryStore()
    node = ToolNode(config_tools_key="tools")
    
    # Create tool call - note that LLM only generates 'key' and 'value' args, not 'store'
    tool_call = {
        "name": "save_data",
        "args": {"key": "test_key", "value": "test_value"},  # No 'store' arg
        "id": "call_1",
        "type": "tool_call",
    }
    
    messages = [AIMessage("", tool_calls=[tool_call])]
    
    # Verify that runtime tools with InjectedStore are handled correctly
    result = await node.ainvoke(
        {"messages": messages},
        config={"configurable": {"tools": [save_data]}},
        store=store
    )
    
    # Verify the result
    tool_message = result["messages"][0]
    assert tool_message.content == "Saved test_value to test_key"
    
    # Verify data was actually saved to store
    stored = store.get(("data",), "test_key")
    assert stored.value["value"] == "test_value"


def test_tool_node_runtime_tools_comprehensive():
    """Test comprehensive scenarios for runtime tools."""
    from langgraph.prebuilt import InjectedState, InjectedStore
    from langgraph.store.memory import InMemoryStore
    from langchain_core.tools.base import InjectedToolCallId
    
    # Tool with multiple injections
    @dec_tool
    def complex_tool(
        x: int,
        state: Annotated[dict, InjectedState],
        store: Annotated[BaseStore, InjectedStore()],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """Complex tool with multiple injections"""
        user_name = state.get("user_name", "Unknown")
        store.put(("calls",), tool_call_id, {"user": user_name, "x": x})
        return f"User {user_name} called with x={x}, saved to {tool_call_id}"
    
    # Tool that will be overridden
    @dec_tool
    def simple_tool(x: int) -> str:
        """Simple tool - compile time version"""
        return f"Compile-time: x={x}"
    
    # Runtime version of the same tool
    @dec_tool 
    def simple_tool_runtime(x: int) -> str:
        """Simple tool - runtime version"""
        return f"Runtime: x={x}"
    simple_tool_runtime.name = "simple_tool"  # Same name to test override
    
    # Create node with compile-time tool
    store = InMemoryStore()
    node = ToolNode(tools=[simple_tool], config_tools_key="tools")
    
    # Test 1: Runtime tool with multiple injections
    tool_call_complex = {
        "name": "complex_tool",
        "args": {"x": 42},
        "id": "complex_call_1",
        "type": "tool_call",
    }
    
    state = {
        "messages": [AIMessage("", tool_calls=[tool_call_complex])],
        "user_name": "Bob"
    }
    
    result = node.invoke(
        state,
        config={"configurable": {"tools": [complex_tool]}},
        store=store
    )
    
    assert result["messages"][0].content == "User Bob called with x=42, saved to complex_call_1"
    
    # Verify store was updated
    stored_call = store.get(("calls",), "complex_call_1")
    assert stored_call.value == {"user": "Bob", "x": 42}
    
    # Test 2: Runtime tool overrides compile-time tool
    tool_call_simple = {
        "name": "simple_tool",
        "args": {"x": 10},
        "id": "simple_call_1",
        "type": "tool_call",
    }
    
    # Without runtime override
    result1 = node.invoke(
        {"messages": [AIMessage("", tool_calls=[tool_call_simple])]}
    )
    assert result1["messages"][0].content == "Compile-time: x=10"
    
    # With runtime override
    result2 = node.invoke(
        {"messages": [AIMessage("", tool_calls=[tool_call_simple])]},
        config={"configurable": {"tools": [simple_tool_runtime]}}
    )
    assert result2["messages"][0].content == "Runtime: x=10"
    
    # Test 3: Mix of compile-time and runtime tools
    tool_calls_mixed = [
        {"name": "simple_tool", "args": {"x": 1}, "id": "id1", "type": "tool_call"},
        {"name": "complex_tool", "args": {"x": 2}, "id": "id2", "type": "tool_call"},
    ]
    
    result3 = node.invoke(
        {
            "messages": [AIMessage("", tool_calls=tool_calls_mixed)],
            "user_name": "Alice"
        },
        config={"configurable": {"tools": [complex_tool]}},
        store=store
    )
    
    # First call uses compile-time tool
    assert result3["messages"][0].content == "Compile-time: x=1"
    # Second call uses runtime tool with injections
    assert result3["messages"][1].content == "User Alice called with x=2, saved to id2"
