"""langgraph.prebuilt exposes a higher-level API for creating and executing agents and tools."""

from typing import Any

from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.prebuilt.tool_node import (
    InjectedState,
    InjectedStore,
    ToolNode,
    ToolRuntime,
    tools_condition,
)
from langgraph.prebuilt.tool_validator import ValidationNode


def __getattr__(name: str) -> Any:  # noqa: C901
    # Lazy imports for optional dns-aid integration
    _dns_aid_names = {
        "DiscoveredAgent",
        "discover_agents",
        "discover_tools",
        "publish_graph",
        "unpublish_graph",
    }
    _dns_aid_node_names = {
        "DnsAidResolverNode",
        "ResolverResult",
        "resolve_and_dispatch",
    }

    if name in _dns_aid_names:
        from langgraph.prebuilt import dns_aid

        return getattr(dns_aid, name)
    if name in _dns_aid_node_names:
        from langgraph.prebuilt import dns_aid_node

        return getattr(dns_aid_node, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_react_agent",
    "DiscoveredAgent",
    "discover_agents",
    "discover_tools",
    "publish_graph",
    "unpublish_graph",
    "DnsAidResolverNode",
    "ResolverResult",
    "resolve_and_dispatch",
    "ToolNode",
    "tools_condition",
    "ValidationNode",
    "InjectedState",
    "InjectedStore",
    "ToolRuntime",
]
