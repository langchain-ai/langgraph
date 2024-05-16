"""langgraph.prebuilt exposes a higher-level API for creating and executing agents and tools."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph.prebuilt import chat_agent_executor
    from langgraph.prebuilt.agent_executor import create_agent_executor
    from langgraph.prebuilt.chat_agent_executor import create_react_agent
    from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
    from langgraph.prebuilt.tool_node import ToolNode, tools_condition
    from langgraph.prebuilt.tool_validator import ValidationNode


def __getattr__(name: str) -> Any:
    if name == "chat_agent_executor":
        from langgraph.prebuilt import chat_agent_executor

        return chat_agent_executor
    if name == "create_agent_executor":
        from langgraph.prebuilt.agent_executor import create_agent_executor

        return create_agent_executor
    if name == "create_react_agent":
        from langgraph.prebuilt.chat_agent_executor import create_react_agent

        return create_react_agent
    if name == "ToolExecutor":
        from langgraph.prebuilt.tool_executor import ToolExecutor

        return ToolExecutor
    if name == "ToolInvocation":
        from langgraph.prebuilt.tool_executor import ToolInvocation

        return ToolInvocation
    if name == "ToolNode":
        from langgraph.prebuilt.tool_node import ToolNode

        return ToolNode
    if name == "tools_condition":
        from langgraph.prebuilt.tool_node import tools_condition

        return tools_condition
    if name == "ValidationNode":
        from langgraph.prebuilt.tool_validator import ValidationNode

        return ValidationNode
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "create_agent_executor",
    "chat_agent_executor",
    "create_react_agent",
    "ToolExecutor",
    "ToolInvocation",
    "ToolNode",
    "tools_condition",
    "ValidationNode",
]
