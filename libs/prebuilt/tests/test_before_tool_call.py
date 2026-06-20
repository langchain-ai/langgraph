"""Unit tests for ToolNode before_tool_call hook."""

import asyncio
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore

from langgraph.prebuilt.tool_node import ToolNode

pytestmark = pytest.mark.anyio


def _create_mock_runtime(store: BaseStore | None = None) -> Mock:
    mock_runtime = Mock()
    mock_runtime.store = store
    mock_runtime.context = None
    mock_runtime.stream_writer = lambda _: None
    return mock_runtime


def _create_config_with_runtime(store: BaseStore | None = None) -> RunnableConfig:
    return {"configurable": {"__pregel_runtime": _create_mock_runtime(store)}}


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_before_tool_call_blocks_tool_execution() -> None:
    calls = {"count": 0}

    @tool
    def tracked_add(a: int, b: int) -> int:
        """Add two numbers with invocation tracking."""
        calls["count"] += 1
        return a + b

    def before_tool_call(_tool_name: str, _tool_input: dict, _graph_state: dict) -> dict:
        return {"action": "BLOCK", "reason": "policy denied"}

    tool_node = ToolNode([tracked_add], before_tool_call=before_tool_call)
    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "tracked_add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_block",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "error"
    assert tool_message.content == "Blocked: policy denied"
    assert calls["count"] == 0


def test_before_tool_call_modifies_input() -> None:
    def before_tool_call(_tool_name: str, _tool_input: dict, _graph_state: dict) -> dict:
        return {"action": "MODIFY", "tool_input": {"a": 5, "b": 7}}

    tool_node = ToolNode([add], before_tool_call=before_tool_call)
    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_modify",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "12"


def test_before_tool_call_modify_requires_tool_input() -> None:
    def before_tool_call(_tool_name: str, _tool_input: dict, _graph_state: dict) -> dict:
        return {"action": "MODIFY"}

    tool_node = ToolNode([add], before_tool_call=before_tool_call)

    with pytest.raises(ValueError, match="MODIFY without tool_input"):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "adding",
                        tool_calls=[
                            {
                                "name": "add",
                                "args": {"a": 1, "b": 2},
                                "id": "call_bad_modify",
                            }
                        ],
                    )
                ]
            },
            config=_create_config_with_runtime(),
        )


def test_before_tool_call_sync_invoke_rejects_async_hook() -> None:
    async def before_tool_call(
        _tool_name: str, _tool_input: dict, _graph_state: dict
    ) -> dict:
        return {"action": "ALLOW"}

    tool_node = ToolNode([add], before_tool_call=before_tool_call)

    with pytest.raises(TypeError, match="returned an awaitable during sync execution"):
        tool_node.invoke(
            {
                "messages": [
                    AIMessage(
                        "adding",
                        tool_calls=[
                            {
                                "name": "add",
                                "args": {"a": 1, "b": 2},
                                "id": "call_async_sync",
                            }
                        ],
                    )
                ]
            },
            config=_create_config_with_runtime(),
        )


async def test_before_tool_call_async_hook_blocks_execution() -> None:
    calls = {"count": 0}

    @tool
    def tracked_add(a: int, b: int) -> int:
        """Add two numbers with invocation tracking."""
        calls["count"] += 1
        return a + b

    async def before_tool_call(
        _tool_name: str, _tool_input: dict, _graph_state: dict
    ) -> dict:
        await asyncio.sleep(0)
        return {"action": "BLOCK", "reason": "async policy denied"}

    tool_node = ToolNode([tracked_add], before_tool_call=before_tool_call)
    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "adding",
                    tool_calls=[
                        {
                            "name": "tracked_add",
                            "args": {"a": 1, "b": 2},
                            "id": "call_async_block",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "error"
    assert tool_message.content == "Blocked: async policy denied"
    assert calls["count"] == 0
