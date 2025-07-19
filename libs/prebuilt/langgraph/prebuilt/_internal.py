from typing import Any, Literal, TypedDict

from langchain_core.messages import ToolCall


class ToolCallWithContext(TypedDict):
    """ToolCall with additional context for graph state.

    This is an internal data-structure meant to help the ToolNode accept
    tools calls with additional context (e.g. state) when dispatched using the
    `Send` API.

    The Send API is used in create_react_agent to be able to distribute the tool
    calls in parallel and support human-in-the-loop workflows where graph execution
    may be paused for an indefinite time.
    """

    tool_call: ToolCall
    __type: Literal["tool_call_with_context"]
    """Type to parameterize the payload.
    
    Using "__" as a prefix to be defensive against potential name collisions with
    regular user state.
    """
    state: Any
    """The state is provided as additional context."""
