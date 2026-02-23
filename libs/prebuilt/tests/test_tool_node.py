import contextlib
import dataclasses
import json
import sys
from functools import partial
from typing import (
    Annotated,
    Any,
    NoReturn,
    TypeVar,
)
from unittest.mock import Mock

import pytest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from langchain_core.tools import tool as dec_tool
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Send
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import TypedDict

from langgraph.prebuilt import (
    InjectedState,
    InjectedStore,
    ToolNode,
)
from langgraph.prebuilt.tool_node import (
    TOOL_CALL_ERROR_TEMPLATE,
    ToolInvocationError,
    ToolRuntime,
    tools_condition,
)

from .messages import _AnyIdHumanMessage, _AnyIdToolMessage
from .model import FakeToolCallingModel

pytestmark = pytest.mark.anyio


def _create_mock_runtime(store: BaseStore | None = None) -> Mock:
    """Create a mock Runtime object for testing ToolNode outside of graph context.

    This helper is needed because ToolNode._func expects a Runtime parameter
    which is injected by RunnableCallable from config["configurable"]["__pregel_runtime"].
    When testing ToolNode directly (outside a graph), we need to provide this manually.
    """
    mock_runtime = Mock()
    mock_runtime.store = store
    mock_runtime.context = None
    mock_runtime.stream_writer = lambda *args, **kwargs: None
    return mock_runtime


def _create_config_with_runtime(store: BaseStore | None = None) -> RunnableConfig:
    """Create a RunnableConfig with mock Runtime for testing ToolNode.

    Returns:
        RunnableConfig with __pregel_runtime in configurable dict.
    """
    return {"configurable": {"__pregel_runtime": _create_mock_runtime(store)}}


def tool1(some_val: int, some_other_val: str) -> str:
    """Tool 1 docstring."""
    if some_val == 0:
        msg = "Test error"
        raise ValueError(msg)
    return f"{some_val} - {some_other_val}"


async def tool2(some_val: int, some_other_val: str) -> str:
    """Tool 2 docstring."""
    if some_val == 0:
        msg = "Test error"
        raise ToolException(msg)
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
def tool5(some_val: int) -> NoReturn:
    """Tool 5 docstring."""
    msg = "Test error"
    raise ToolException(msg)


tool5.handle_tool_error = "foo"


async def test_tool_node() -> None:
    """Test tool node."""
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
        },
        config=_create_config_with_runtime(),
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
        },
        config=_create_config_with_runtime(),
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
        },
        config=_create_config_with_runtime(),
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
        },
        config=_create_config_with_runtime(),
    )
    tool_message: ToolMessage = result4["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.content == [{"type": "image_url", "image_url": {"url": "abdc"}}]
    assert tool_message.tool_call_id == "some 3"


async def test_tool_node_tool_call_input() -> None:
    # Single tool call
    tool_call_1 = {
        "name": "tool1",
        "args": {"some_val": 1, "some_other_val": "foo"},
        "id": "some 0",
        "type": "tool_call",
    }
    result = ToolNode([tool1]).invoke(
        [tool_call_1], config=_create_config_with_runtime()
    )
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
    result = ToolNode([tool1]).invoke(
        [tool_call_1, tool_call_2], config=_create_config_with_runtime()
    )
    assert result["messages"] == [
        ToolMessage(content="1 - foo", tool_call_id="some 0", name="tool1"),
        ToolMessage(content="2 - bar", tool_call_id="some 1", name="tool1"),
    ]

    # Test with unknown tool
    tool_call_3 = tool_call_1.copy()
    tool_call_3["name"] = "tool2"
    result = ToolNode([tool1]).invoke(
        [tool_call_1, tool_call_3], config=_create_config_with_runtime()
    )
    assert result["messages"] == [
        ToolMessage(content="1 - foo", tool_call_id="some 0", name="tool1"),
        ToolMessage(
            content="Error: tool2 is not a valid tool, try one of [tool1].",
            name="tool2",
            tool_call_id="some 0",
            status="error",
        ),
    ]


def test_tool_node_error_handling_default_invocation() -> None:
    tn = ToolNode([tool1])
    result = tn.invoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "tool1",
                            "args": {"invalid": 0, "args": "foo"},
                            "id": "some id",
                        },
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    assert all(m.type == "tool" for m in result["messages"])
    assert all(m.status == "error" for m in result["messages"])
    assert (
        "Error invoking tool 'tool1' with kwargs {'invalid': 0, 'args': 'foo'} with error:\n"
        in result["messages"][0].content
    )


def test_tool_node_error_handling_default_exception() -> None:
    tn = ToolNode([tool1])
    with pytest.raises(ValueError):
        tn.invoke(
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
            },
            config=_create_config_with_runtime(),
        )


async def test_tool_node_error_handling() -> None:
    def handle_all(e: ValueError | ToolException | ToolInvocationError):
        return TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))

    # test catching all exceptions, via:
    # - handle_tool_errors = True
    # - passing a tuple of all exceptions
    # - passing a callable with all exceptions in the signature
    for handle_tool_errors in (
        True,
        (ValueError, ToolException, ToolInvocationError),
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
            },
            config=_create_config_with_runtime(),
        )

        assert all(m.type == "tool" for m in result_error["messages"])
        assert all(m.status == "error" for m in result_error["messages"])
        assert (
            result_error["messages"][0].content
            == f"Error: {ValueError('Test error')!r}\n Please fix your mistakes."
        )
        assert (
            result_error["messages"][1].content
            == f"Error: {ToolException('Test error')!r}\n Please fix your mistakes."
        )
        # Check that the validation error contains the field name
        assert "some_other_val" in result_error["messages"][2].content

        assert result_error["messages"][0].tool_call_id == "some id"
        assert result_error["messages"][1].tool_call_id == "some other id"
        assert result_error["messages"][2].tool_call_id == "another id"


async def test_tool_node_error_handling_callable() -> None:
    def handle_value_error(e: ValueError) -> str:
        return "Value error"

    def handle_tool_exception(e: ToolException) -> str:
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
            },
            config=_create_config_with_runtime(),
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
                },
                config=_create_config_with_runtime(),
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
                },
                config=_create_config_with_runtime(),
            )
        assert str(exc_info.value) == "Test error"


async def test_tool_node_handle_tool_errors_false() -> None:
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
            },
            config=_create_config_with_runtime(),
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
            },
            config=_create_config_with_runtime(),
        )

    assert str(exc_info.value) == "Test error"

    # test validation errors get raised if handle_tool_errors is False
    with pytest.raises(ToolInvocationError):
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
            },
            config=_create_config_with_runtime(),
        )


def test_tool_node_individual_tool_error_handling() -> None:
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
        },
        config=_create_config_with_runtime(),
    )

    tool_message: ToolMessage = result_individual_tool_error_handler["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.status == "error"
    assert tool_message.content == "foo"
    assert tool_message.tool_call_id == "some 0"


def test_tool_node_incorrect_tool_name() -> None:
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
        },
        config=_create_config_with_runtime(),
    )

    tool_message: ToolMessage = result_incorrect_name["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.status == "error"
    assert (
        tool_message.content
        == "Error: tool3 is not a valid tool, try one of [tool1, tool2]."
    )
    assert tool_message.tool_call_id == "some 0"


def test_tool_node_node_interrupt() -> None:
    def tool_interrupt(some_val: int) -> None:
        """Tool docstring."""
        msg = "foo"
        raise GraphBubbleUp(msg)

    def handle(e: GraphInterrupt) -> str:
        return "handled"

    for handle_tool_errors in (True, (GraphBubbleUp,), "handled", handle, False):
        node = ToolNode([tool_interrupt], handle_tool_errors=handle_tool_errors)
        with pytest.raises(GraphBubbleUp) as exc_info:
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
                },
                config=_create_config_with_runtime(),
            )
            assert exc_info.value == "foo"


@pytest.mark.parametrize("input_type", ["dict", "tool_calls"])
async def test_tool_node_command(input_type: str) -> None:
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
    result = ToolNode([add, transfer_to_bob]).invoke(
        input_, config=_create_config_with_runtime()
    )

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
            },
            config=_create_config_with_runtime(),
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
            },
            config=_create_config_with_runtime(),
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
        },
        config=_create_config_with_runtime(),
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
            },
            config=_create_config_with_runtime(),
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
            },
            config=_create_config_with_runtime(),
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
            },
            config=_create_config_with_runtime(),
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
        },
        config=_create_config_with_runtime(),
    ) == [Command(update={"messages": []}, graph=Command.PARENT)]


async def test_tool_node_command_list_input() -> None:
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
        ],
        config=_create_config_with_runtime(),
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
            [AIMessage("", tool_calls=[{"args": {}, "id": "1", "name": tool.name}])],
            config=_create_config_with_runtime(),
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
            [AIMessage("", tool_calls=[{"args": {}, "id": "1", "name": tool.name}])],
            config=_create_config_with_runtime(),
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
        ],
        config=_create_config_with_runtime(),
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
            ],
            config=_create_config_with_runtime(),
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
            ],
            config=_create_config_with_runtime(),
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
            ],
            config=_create_config_with_runtime(),
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
        ],
        config=_create_config_with_runtime(),
    ) == [Command(update=[], graph=Command.PARENT)]


def test_tool_node_parent_command_with_send() -> None:
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
        [AIMessage("", tool_calls=tool_calls)],
        config=_create_config_with_runtime(),
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


async def test_tool_node_command_remove_all_messages() -> None:
    from langchain_core.tools.base import InjectedToolCallId

    @dec_tool
    def remove_all_messages_tool(tool_call_id: Annotated[str, InjectedToolCallId]):
        """A tool that removes all messages."""
        return Command(update={"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})

    tool_node = ToolNode([remove_all_messages_tool])
    tool_call = {
        "name": "remove_all_messages_tool",
        "args": {},
        "id": "tool_call_123",
    }
    result = await tool_node.ainvoke(
        {"messages": [AIMessage(content="", tool_calls=[tool_call])]},
        config=_create_config_with_runtime(),
    )

    assert isinstance(result, list)
    assert len(result) == 1
    command = result[0]
    assert isinstance(command, Command)
    assert command.update == {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}


class _InjectStateSchema(TypedDict):
    messages: list
    foo: str


class _InjectedStatePydanticV2Schema(BaseModel):
    messages: list
    foo: str


@dataclasses.dataclass
class _InjectedStateDataclassSchema:
    messages: list
    foo: str


_INJECTED_STATE_SCHEMAS = [
    _InjectStateSchema,
    _InjectedStatePydanticV2Schema,
    _InjectedStateDataclassSchema,
]

if sys.version_info < (3, 14):

    class _InjectedStatePydanticSchema(BaseModelV1):
        messages: list
        foo: str

    _INJECTED_STATE_SCHEMAS.append(_InjectedStatePydanticSchema)

T = TypeVar("T")


@pytest.mark.parametrize("schema_", _INJECTED_STATE_SCHEMAS)
def test_tool_node_inject_state(schema_: type[T]) -> None:
    def tool1(some_val: int, state: Annotated[T, InjectedState]) -> str:
        """Tool 1 docstring."""
        if isinstance(state, dict):
            return state["foo"]
        return state.foo

    def tool2(some_val: int, state: Annotated[T, InjectedState()]) -> str:
        """Tool 2 docstring."""
        if isinstance(state, dict):
            return state["foo"]
        return state.foo

    def tool3(
        some_val: int,
        foo: Annotated[str, InjectedState("foo")],
        msgs: Annotated[list[AnyMessage], InjectedState("messages")],
    ) -> str:
        """Tool 1 docstring."""
        return foo

    def tool4(
        some_val: int, msgs: Annotated[list[AnyMessage], InjectedState("messages")]
    ) -> str:
        """Tool 1 docstring."""
        return msgs[0].content

    node = ToolNode([tool1, tool2, tool3, tool4], handle_tool_errors=True)
    for tool_name in ("tool1", "tool2", "tool3"):
        tool_call = {
            "name": tool_name,
            "args": {"some_val": 1},
            "id": "some 0",
            "type": "tool_call",
        }
        msg = AIMessage("hi?", tool_calls=[tool_call])
        result = node.invoke(
            schema_(messages=[msg], foo="bar"), config=_create_config_with_runtime()
        )
        tool_message = result["messages"][-1]
        assert tool_message.content == "bar", f"Failed for tool={tool_name}"

        if tool_name == "tool3":
            failure_input = None
            with contextlib.suppress(Exception):
                failure_input = schema_(messages=[msg], notfoo="bar")
            if failure_input is not None:
                with pytest.raises(KeyError):
                    node.invoke(failure_input, config=_create_config_with_runtime())

                with pytest.raises(ValueError):
                    node.invoke([msg], config=_create_config_with_runtime())
        else:
            failure_input = None
            try:
                failure_input = schema_(messages=[msg], notfoo="bar")
            except Exception:
                # We'd get a validation error from pydantic state and wouldn't make it to the node
                # anyway
                pass
            if failure_input is not None:
                messages_ = node.invoke(
                    failure_input, config=_create_config_with_runtime()
                )
                tool_message = messages_["messages"][-1]
                assert "KeyError" in tool_message.content
                tool_message = node.invoke([msg], config=_create_config_with_runtime())[
                    -1
                ]
                assert "KeyError" in tool_message.content

    tool_call = {
        "name": "tool4",
        "args": {"some_val": 1},
        "id": "some 0",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])
    result = node.invoke(
        schema_(messages=[msg], foo=""), config=_create_config_with_runtime()
    )
    tool_message = result["messages"][-1]
    assert tool_message.content == "hi?"

    result = node.invoke([msg], config=_create_config_with_runtime())
    tool_message = result[-1]
    assert tool_message.content == "hi?"


def test_tool_node_inject_store() -> None:
    store = InMemoryStore()
    namespace = ("test",)

    def tool1(some_val: int, store: Annotated[BaseStore, InjectedStore()]) -> str:
        """Tool 1 docstring."""
        store_val = store.get(namespace, "test_key").value["foo"]
        return f"Some val: {some_val}, store val: {store_val}"

    def tool2(some_val: int, store: Annotated[BaseStore, InjectedStore()]) -> str:
        """Tool 2 docstring."""
        store_val = store.get(namespace, "test_key").value["foo"]
        return f"Some val: {some_val}, store val: {store_val}"

    def tool3(
        some_val: int,
        bar: Annotated[str, InjectedState("bar")],
        store: Annotated[BaseStore, InjectedStore()],
    ) -> str:
        """Tool 3 docstring."""
        store_val = store.get(namespace, "test_key").value["foo"]
        return f"Some val: {some_val}, store val: {store_val}, state val: {bar}"

    node = ToolNode([tool1, tool2, tool3], handle_tool_errors=True)
    store.put(namespace, "test_key", {"foo": "bar"})

    class State(MessagesState):
        bar: str

    builder = StateGraph(State)
    builder.add_node("tools", node)
    builder.add_edge(START, "tools")
    graph = builder.compile(store=store)

    for tool_name in ("tool1", "tool2"):
        tool_call = {
            "name": tool_name,
            "args": {"some_val": 1},
            "id": "some 0",
            "type": "tool_call",
        }
        msg = AIMessage("hi?", tool_calls=[tool_call])
        node_result = node.invoke(
            {"messages": [msg]}, config=_create_config_with_runtime(store=store)
        )
        graph_result = graph.invoke({"messages": [msg]})
        for result in (node_result, graph_result):
            result["messages"][-1]
            tool_message = result["messages"][-1]
            assert tool_message.content == "Some val: 1, store val: bar", (
                f"Failed for tool={tool_name}"
            )

    tool_call = {
        "name": "tool3",
        "args": {"some_val": 1},
        "id": "some 0",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])
    node_result = node.invoke(
        {"messages": [msg], "bar": "baz"},
        config=_create_config_with_runtime(store=store),
    )
    graph_result = graph.invoke({"messages": [msg], "bar": "baz"})
    for result in (node_result, graph_result):
        result["messages"][-1]
        tool_message = result["messages"][-1]
        assert tool_message.content == "Some val: 1, store val: bar, state val: baz", (
            f"Failed for tool={tool_name}"
        )

    # test injected store without passing store to compiled graph
    failing_graph = builder.compile()
    with pytest.raises(ValueError):
        failing_graph.invoke({"messages": [msg], "bar": "baz"})


def test_tool_node_ensure_utf8() -> None:
    @dec_tool
    def get_day_list(days: list[str]) -> list[str]:
        """choose days"""
        return days

    data = ["星期一", "水曜日", "목요일", "Friday"]
    tools = [get_day_list]
    tool_calls = [ToolCall(name=get_day_list.name, args={"days": data}, id="test_id")]
    outputs: list[ToolMessage] = ToolNode(tools).invoke(
        [AIMessage(content="", tool_calls=tool_calls)],
        config=_create_config_with_runtime(),
    )
    assert outputs[0].content == json.dumps(data, ensure_ascii=False)


def test_tool_node_messages_key() -> None:
    @dec_tool
    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    model = FakeToolCallingModel(
        tool_calls=[[ToolCall(name=add.name, args={"a": 1, "b": 2}, id="test_id")]]
    )

    class State(TypedDict):
        subgraph_messages: Annotated[list[AnyMessage], add_messages]

    def call_model(state: State) -> dict[str, Any]:
        response = model.invoke(state["subgraph_messages"])
        model.tool_calls = []
        return {"subgraph_messages": response}

    builder = StateGraph(State)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode([add], messages_key="subgraph_messages"))
    builder.add_conditional_edges(
        "agent", partial(tools_condition, messages_key="subgraph_messages")
    )
    builder.add_edge(START, "agent")
    builder.add_edge("tools", "agent")

    graph = builder.compile()
    result = graph.invoke({"subgraph_messages": [HumanMessage(content="hi")]})
    assert result["subgraph_messages"] == [
        _AnyIdHumanMessage(content="hi"),
        AIMessage(
            content="hi",
            id="0",
            tool_calls=[ToolCall(name=add.name, args={"a": 1, "b": 2}, id="test_id")],
        ),
        _AnyIdToolMessage(content="3", name=add.name, tool_call_id="test_id"),
        AIMessage(content="hi-hi-3", id="1"),
    ]


def test_tool_node_stream_writer() -> None:
    @dec_tool
    def streaming_tool(x: int) -> str:
        """Do something with writer."""
        my_writer = get_stream_writer()
        for value in ["foo", "bar", "baz"]:
            my_writer({"custom_tool_value": value})

        return x

    tool_node = ToolNode([streaming_tool])
    graph = (
        StateGraph(MessagesState)
        .add_node("tools", tool_node)
        .add_edge(START, "tools")
        .compile()
    )

    tool_call = {
        "name": "streaming_tool",
        "args": {"x": 1},
        "id": "1",
        "type": "tool_call",
    }
    inputs = {
        "messages": [AIMessage("", tool_calls=[tool_call])],
    }

    assert list(graph.stream(inputs, stream_mode="custom")) == [
        {"custom_tool_value": "foo"},
        {"custom_tool_value": "bar"},
        {"custom_tool_value": "baz"},
    ]
    assert list(graph.stream(inputs, stream_mode=["custom", "updates"])) == [
        ("custom", {"custom_tool_value": "foo"}),
        ("custom", {"custom_tool_value": "bar"}),
        ("custom", {"custom_tool_value": "baz"}),
        (
            "updates",
            {
                "tools": {
                    "messages": [
                        _AnyIdToolMessage(
                            content="1",
                            name="streaming_tool",
                            tool_call_id="1",
                        ),
                    ],
                },
            },
        ),
    ]


def test_tool_call_request_setattr_deprecation_warning():
    """Test that ToolCallRequest raises a deprecation warning on direct attribute modification."""
    import warnings

    from langgraph.prebuilt.tool_node import ToolCallRequest

    # Create a mock ToolCall
    tool_call = {"name": "test", "args": {"a": 1}, "id": "call_1", "type": "tool_call"}

    # Create a ToolCallRequest
    request = ToolCallRequest(
        tool_call=tool_call,
        tool=None,
        state={"messages": []},
        runtime=None,
    )

    # Test 1: Direct attribute assignment should raise deprecation warning but still work
    with pytest.warns(DeprecationWarning, match="deprecated.*override"):
        request.tool_call = {"name": "other", "args": {}, "id": "call_2"}

    # Verify the attribute was actually modified
    assert request.tool_call == {"name": "other", "args": {}, "id": "call_2"}

    # Reset for further tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        request.tool_call = tool_call

    # Test 2: override method should work without warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        new_tool_call = {
            "name": "new_tool",
            "args": {"b": 2},
            "id": "call_3",
            "type": "tool_call",
        }
        new_request = request.override(tool_call=new_tool_call)

        # Verify no warning was raised
        assert len(w) == 0

    # Verify original is unchanged
    assert request.tool_call == tool_call

    # Verify new request has updated values
    assert new_request.tool_call == new_tool_call

    # Test 3: Initialization should not trigger warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ToolCallRequest(
            tool_call=tool_call,
            tool=None,
            state={"messages": []},
            runtime=None,
        )
        # Verify no warning was raised during initialization
        assert len(w) == 0


async def test_tool_node_inject_async_all_types_signature_only() -> None:
    """Test all injection types without @tool decorator."""
    store = InMemoryStore()
    namespace = ("test",)
    store.put(namespace, "test_key", {"store_data": "from_store"})

    class TestState(TypedDict):
        messages: list
        foo: str
        bar: int

    async def comprehensive_async_tool(
        x: int,
        whole_state: Annotated[TestState, InjectedState],
        foo_field: Annotated[str, InjectedState("foo")],
        store: Annotated[BaseStore, InjectedStore()],
        runtime: ToolRuntime,
    ) -> str:
        """Async tool that uses all injection types."""
        bar_from_whole = whole_state["bar"]
        foo_value = foo_field
        store_val = store.get(namespace, "test_key").value["store_data"]
        foo_from_runtime = runtime.state["foo"]
        tool_call_id = runtime.tool_call_id

        return (
            f"x={x}, "
            f"bar_from_whole={bar_from_whole}, "
            f"foo_field={foo_value}, "
            f"store={store_val}, "
            f"foo_from_runtime={foo_from_runtime}, "
            f"tool_call_id={tool_call_id}"
        )

    node = ToolNode([comprehensive_async_tool], handle_tool_errors=True)
    tool_call = {
        "name": "comprehensive_async_tool",
        "args": {"x": 42},
        "id": "test_call_123",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])

    config = _create_config_with_runtime(store=store)
    result = await node.ainvoke(
        {"messages": [msg], "foo": "foo_value", "bar": 99}, config=config
    )

    tool_message = result["messages"][-1]
    assert tool_message.content == (
        "x=42, "
        "bar_from_whole=99, "
        "foo_field=foo_value, "
        "store=from_store, "
        "foo_from_runtime=foo_value, "
        "tool_call_id=test_call_123"
    )


async def test_tool_node_inject_async_all_types_with_decorator() -> None:
    """Test all injection types with @tool decorator."""
    store = InMemoryStore()
    namespace = ("test",)
    store.put(namespace, "test_key", {"store_data": "from_store"})

    class TestState(TypedDict):
        messages: list
        foo: str
        bar: int

    @dec_tool
    async def comprehensive_async_tool(
        x: int,
        whole_state: Annotated[TestState, InjectedState],
        foo_field: Annotated[str, InjectedState("foo")],
        store: Annotated[BaseStore, InjectedStore()],
        runtime: ToolRuntime,
    ) -> str:
        """Async tool that uses all injection types."""
        bar_from_whole = whole_state["bar"]
        foo_value = foo_field
        store_val = store.get(namespace, "test_key").value["store_data"]
        foo_from_runtime = runtime.state["foo"]
        tool_call_id = runtime.tool_call_id

        return (
            f"x={x}, "
            f"bar_from_whole={bar_from_whole}, "
            f"foo_field={foo_value}, "
            f"store={store_val}, "
            f"foo_from_runtime={foo_from_runtime}, "
            f"tool_call_id={tool_call_id}"
        )

    node = ToolNode([comprehensive_async_tool], handle_tool_errors=True)
    tool_call = {
        "name": "comprehensive_async_tool",
        "args": {"x": 42},
        "id": "test_call_456",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])

    config = _create_config_with_runtime(store=store)
    result = await node.ainvoke(
        {"messages": [msg], "foo": "foo_value", "bar": 99}, config=config
    )

    tool_message = result["messages"][-1]
    assert tool_message.content == (
        "x=42, "
        "bar_from_whole=99, "
        "foo_field=foo_value, "
        "store=from_store, "
        "foo_from_runtime=foo_value, "
        "tool_call_id=test_call_456"
    )


async def test_tool_node_inject_async_all_types_with_schema() -> None:
    """Test all injection types with explicit schema."""
    store = InMemoryStore()
    namespace = ("test",)
    store.put(namespace, "test_key", {"store_data": "from_store"})

    class TestState(TypedDict):
        messages: list
        foo: str
        bar: int

    class ComprehensiveToolSchema(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        x: int
        whole_state: Annotated[TestState, InjectedState]
        foo_field: Annotated[str, InjectedState("foo")]
        store: Annotated[BaseStore, InjectedStore()]
        runtime: ToolRuntime

    @dec_tool(args_schema=ComprehensiveToolSchema)
    async def comprehensive_async_tool(
        x: int,
        whole_state: Annotated[TestState, InjectedState],
        foo_field: Annotated[str, InjectedState("foo")],
        store: Annotated[BaseStore, InjectedStore()],
        runtime: ToolRuntime,
    ) -> str:
        """Async tool that uses all injection types."""
        bar_from_whole = whole_state["bar"]
        foo_value = foo_field
        store_val = store.get(namespace, "test_key").value["store_data"]
        foo_from_runtime = runtime.state["foo"]
        tool_call_id = runtime.tool_call_id

        return (
            f"x={x}, "
            f"bar_from_whole={bar_from_whole}, "
            f"foo_field={foo_value}, "
            f"store={store_val}, "
            f"foo_from_runtime={foo_from_runtime}, "
            f"tool_call_id={tool_call_id}"
        )

    node = ToolNode([comprehensive_async_tool], handle_tool_errors=True)
    tool_call = {
        "name": "comprehensive_async_tool",
        "args": {"x": 42},
        "id": "test_call_789",
        "type": "tool_call",
    }
    msg = AIMessage("hi?", tool_calls=[tool_call])

    config = _create_config_with_runtime(store=store)
    result = await node.ainvoke(
        {"messages": [msg], "foo": "foo_value", "bar": 99}, config=config
    )

    tool_message = result["messages"][-1]
    assert tool_message.content == (
        "x=42, "
        "bar_from_whole=99, "
        "foo_field=foo_value, "
        "store=from_store, "
        "foo_from_runtime=foo_value, "
        "tool_call_id=test_call_789"
    )


async def test_tool_node_tool_runtime_generic() -> None:
    """Test that ToolRuntime with generic type arguments is correctly injected."""

    @dataclasses.dataclass
    class MyContext:
        some_info: str

    @dec_tool
    def get_info(rt: ToolRuntime[MyContext]):
        """This tool returns info from context."""
        return rt.context.some_info

    # Create a mock runtime with context
    mock_runtime = _create_mock_runtime()
    mock_runtime.context = MyContext(some_info="test_info")

    config = {"configurable": {"__pregel_runtime": mock_runtime}}

    result = await ToolNode([get_info]).ainvoke(
        {
            "messages": [
                AIMessage(
                    "call tool",
                    tool_calls=[
                        {
                            "name": "get_info",
                            "args": {},
                            "id": "call_1",
                        }
                    ],
                )
            ]
        },
        config=config,
    )

    tool_message = result["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.content == "test_info"
    assert tool_message.tool_call_id == "call_1"


def test_tool_node_inject_runtime_dynamic_tool_via_wrap_tool_call() -> None:
    """Test that ToolRuntime is injected for dynamically registered tools.

    Regression test for https://github.com/langchain-ai/langchain/issues/35305.
    When a tool is dynamically provided via wrap_tool_call (not registered at
    ToolNode init time), ToolRuntime should still be injected into the tool.
    """

    @dec_tool
    def static_tool(x: int) -> str:
        """A static tool registered at init."""
        return f"static: {x}"

    @dec_tool
    def dynamic_tool_with_runtime(x: int, runtime: ToolRuntime) -> str:
        """A dynamic tool that needs ToolRuntime injection."""
        return f"dynamic: x={x}, tool_call_id={runtime.tool_call_id}"

    def wrap_tool_call(request, execute):
        """Middleware that swaps in a dynamic tool."""
        if request.tool_call["name"] == "dynamic_tool_with_runtime":
            # Override tool to the dynamic one (not registered at init)
            new_request = request.override(tool=dynamic_tool_with_runtime)
            return execute(new_request)
        return execute(request)

    # ToolNode only knows about static_tool at init time
    tool_node = ToolNode(
        [static_tool],
        wrap_tool_call=wrap_tool_call,
    )

    # Verify the dynamic tool is NOT in the tool node's registered tools
    assert "dynamic_tool_with_runtime" not in tool_node.tools_by_name

    # Call the dynamic tool
    tool_call = {
        "name": "dynamic_tool_with_runtime",
        "args": {"x": 42},
        "id": "call_dynamic_1",
        "type": "tool_call",
    }
    msg = AIMessage("", tool_calls=[tool_call])
    result = tool_node.invoke(
        {"messages": [msg]},
        config=_create_config_with_runtime(),
    )

    # ToolRuntime should be injected and the tool should execute successfully
    tool_message = result["messages"][-1]
    assert tool_message.content == "dynamic: x=42, tool_call_id=call_dynamic_1"
    assert tool_message.tool_call_id == "call_dynamic_1"


async def test_tool_node_inject_runtime_dynamic_tool_via_wrap_tool_call_async() -> None:
    """Test that ToolRuntime is injected for dynamically registered tools (async).

    Async version of the regression test for
    https://github.com/langchain-ai/langchain/issues/35305.
    """

    @dec_tool
    def static_tool(x: int) -> str:
        """A static tool registered at init."""
        return f"static: {x}"

    @dec_tool
    async def dynamic_tool_with_runtime(x: int, runtime: ToolRuntime) -> str:
        """A dynamic async tool that needs ToolRuntime injection."""
        return f"dynamic: x={x}, tool_call_id={runtime.tool_call_id}"

    async def awrap_tool_call(request, execute):
        """Async middleware that swaps in a dynamic tool."""
        if request.tool_call["name"] == "dynamic_tool_with_runtime":
            new_request = request.override(tool=dynamic_tool_with_runtime)
            return await execute(new_request)
        return await execute(request)

    # ToolNode only knows about static_tool at init time
    tool_node = ToolNode(
        [static_tool],
        awrap_tool_call=awrap_tool_call,
    )

    # Verify the dynamic tool is NOT in the tool node's registered tools
    assert "dynamic_tool_with_runtime" not in tool_node.tools_by_name

    # Call the dynamic tool
    tool_call = {
        "name": "dynamic_tool_with_runtime",
        "args": {"x": 42},
        "id": "call_dynamic_2",
        "type": "tool_call",
    }
    msg = AIMessage("", tool_calls=[tool_call])
    result = await tool_node.ainvoke(
        {"messages": [msg]},
        config=_create_config_with_runtime(),
    )

    # ToolRuntime should be injected and the tool should execute successfully
    tool_message = result["messages"][-1]
    assert tool_message.content == "dynamic: x=42, tool_call_id=call_dynamic_2"
    assert tool_message.tool_call_id == "call_dynamic_2"
