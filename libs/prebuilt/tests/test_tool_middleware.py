"""Unit tests for composable ToolNode middleware."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore

from langgraph.prebuilt import (
    ToolNode,
    build_tool_call_key,
    chain_async_tool_wrappers,
    chain_tool_wrappers,
    deduplicate_tool_calls,
    deduplicate_tool_calls_in_state,
)
from langgraph.prebuilt.tool_node import ToolCallRequest

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


def test_build_tool_call_key_is_stable_for_equivalent_args() -> None:
    first = {"name": "search", "args": {"term": "milk", "banner": "loblaw"}}
    second = {"name": "search", "args": {"banner": "loblaw", "term": "milk"}}

    assert build_tool_call_key(first) == build_tool_call_key(second)


def test_deduplicate_tool_calls_removes_exact_duplicates() -> None:
    messages = [
        HumanMessage(content="find milk"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search", "id": "call_1", "args": {"term": "milk"}},
                {"name": "search", "id": "call_2", "args": {"term": "milk"}},
                {"name": "read_cart", "id": "call_3", "args": {}},
            ],
        ),
    ]

    deduplicated = deduplicate_tool_calls(messages)

    tool_calls = deduplicated[-1].tool_calls
    assert len(tool_calls) == 2
    assert [call["name"] for call in tool_calls] == ["search", "read_cart"]
    assert [call["id"] for call in tool_calls] == ["call_1", "call_3"]


def test_deduplicate_tool_calls_treats_reordered_args_as_duplicates() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "id": "call_1",
                    "args": {"term": "milk", "banner": "loblaw"},
                },
                {
                    "name": "search",
                    "id": "call_2",
                    "args": {"banner": "loblaw", "term": "milk"},
                },
            ],
        )
    ]

    deduplicated = deduplicate_tool_calls(messages)

    assert len(deduplicated[-1].tool_calls) == 1
    assert deduplicated[-1].tool_calls[0]["id"] == "call_1"


def test_deduplicate_tool_calls_returns_new_list_when_nothing_changes() -> None:
    messages = [
        HumanMessage(content="hello"),
        AIMessage(content="hi there"),
    ]

    deduplicated = deduplicate_tool_calls(messages)

    assert deduplicated == messages
    assert deduplicated is not messages


def test_deduplicate_tool_calls_in_state_updates_messages_key() -> None:
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "id": "call_1", "args": {"term": "milk"}},
                    {"name": "search", "id": "call_2", "args": {"term": "milk"}},
                ],
            )
        ],
        "metadata": [],
    }

    updated = deduplicate_tool_calls_in_state(state)

    assert len(updated["messages"][-1].tool_calls) == 1
    assert updated["metadata"] == []
    assert len(state["messages"][-1].tool_calls) == 2


def test_chain_tool_wrappers_runs_outer_then_inner() -> None:
    order: list[str] = []

    def outer(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        order.append("outer")
        result = execute(request)
        order.append("outer_after")
        return result

    def inner(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        order.append("inner")
        return execute(request)

    tool_node = ToolNode([add], wrap_tool_call=chain_tool_wrappers(outer, inner))

    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "add", "id": "call_1", "args": {"a": 1, "b": 2}}
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    assert order == ["outer", "inner", "outer_after"]
    assert result["messages"][-1].content == "3"


def test_chain_tool_wrappers_requires_at_least_one_wrapper() -> None:
    with pytest.raises(ValueError, match="requires at least one wrapper"):
        chain_tool_wrappers()


def test_chain_async_tool_wrappers_requires_at_least_one_wrapper() -> None:
    with pytest.raises(ValueError, match="requires at least one wrapper"):
        chain_async_tool_wrappers()


async def test_chain_async_tool_wrappers_supports_sync_wrapper() -> None:
    seen: list[str] = []

    def sync_wrapper(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        seen.append("sync")
        return execute(request)

    tool_node = ToolNode([add], awrap_tool_call=chain_async_tool_wrappers(sync_wrapper))

    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "add", "id": "call_1", "args": {"a": 3, "b": 5}}
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    assert seen == ["sync"]
    assert result["messages"][-1].content == "8"


async def test_chain_async_tool_wrappers_runs_outer_then_inner() -> None:
    order: list[str] = []

    async def outer(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage:
        order.append("outer")
        result = await execute(request)
        order.append("outer_after")
        return result

    def inner(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        order.append("inner")
        return execute(request)

    tool_node = ToolNode([add], awrap_tool_call=chain_async_tool_wrappers(outer, inner))

    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "add", "id": "call_1", "args": {"a": 1, "b": 2}}
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    assert order == ["outer", "inner", "outer_after"]
    assert result["messages"][-1].content == "3"


async def test_chain_async_tool_wrappers_supports_async_wrapper() -> None:
    seen: list[str] = []

    async def async_wrapper(
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage:
        seen.append("async")
        return await execute(request)

    tool_node = ToolNode(
        [add], awrap_tool_call=chain_async_tool_wrappers(async_wrapper)
    )

    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "add", "id": "call_1", "args": {"a": 2, "b": 4}}
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    assert seen == ["async"]
    assert result["messages"][-1].content == "6"


def test_tool_node_with_deduplicated_state_executes_each_unique_call_once() -> None:
    execution_count = {"add": 0}

    @tool
    def counting_add(a: int, b: int) -> int:
        """Add while counting invocations."""
        execution_count["add"] += 1
        return a + b

    tool_node = ToolNode([counting_add])
    state = deduplicate_tool_calls_in_state(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "counting_add",
                            "id": "call_1",
                            "args": {"a": 1, "b": 1},
                        },
                        {
                            "name": "counting_add",
                            "id": "call_2",
                            "args": {"a": 1, "b": 1},
                        },
                        {
                            "name": "counting_add",
                            "id": "call_3",
                            "args": {"a": 2, "b": 3},
                        },
                    ],
                )
            ]
        }
    )

    result = tool_node.invoke(state, config=_create_config_with_runtime())

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    assert execution_count["add"] == 2
