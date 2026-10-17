from __future__ import annotations

from collections.abc import Callable
from typing import Union

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph.prebuilt import ToolNode

pytestmark = pytest.mark.anyio


def handle_pipe(error: ValueError | TypeError) -> str:
    return f"Handled: {error}"


def handle_union(error: Union[ValueError, TypeError]) -> str:  # noqa: UP007 - Test postponed typing.Union.
    return f"Handled: {error}"


@pytest.mark.parametrize("handler", [handle_pipe, handle_union], ids=["pipe", "union"])
@pytest.mark.parametrize("error_type", [ValueError, TypeError, KeyError])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_tool_error_handler_with_postponed_union(
    handler: Callable[..., str], error_type: type[Exception], asynchronous: bool
) -> None:
    @tool
    def failing_tool() -> str:
        """Raise an error during tool execution."""
        raise error_type("tool failed")

    graph = (
        StateGraph(MessagesState)
        .add_node("tools", ToolNode([failing_tool], handle_tool_errors=handler))
        .add_edge(START, "tools")
        .add_edge("tools", END)
        .compile()
    )
    inputs = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "failing_tool", "args": {}, "id": "call-1"}],
            )
        ]
    }

    if error_type is KeyError:
        with pytest.raises(KeyError, match="tool failed"):
            if asynchronous:
                await graph.ainvoke(inputs)
            else:
                graph.invoke(inputs)
        return

    result = await graph.ainvoke(inputs) if asynchronous else graph.invoke(inputs)
    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.content == "Handled: tool failed"
    assert message.status == "error"
    assert message.tool_call_id == "call-1"
