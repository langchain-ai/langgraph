"""DNS-AID agent discovery utilities for LangGraph.

Discover remote AI agents via DNS using the DNS-AID protocol (SVCB + TXT
records) and create LangChain tools from them for use in LangGraph agents.

This enables multi-agent systems where agents discover collaborators
dynamically at runtime rather than hardcoding agent references.

Setup:
    Install the ``dns-aid`` and ``httpx`` packages:

    .. code-block:: bash

        pip install dns-aid httpx

Usage with create_react_agent:
    .. code-block:: python

        from langgraph.prebuilt import create_react_agent
        from langgraph.prebuilt.dns_aid import discover_tools

        # Discover agents published at a domain
        tools = await discover_tools("agents.example.com", protocol="mcp")

        # Use discovered tools in a ReAct agent
        agent = create_react_agent(model, tools=tools)
        result = await agent.ainvoke({
            "messages": [("user", "Search for recent papers")]
        })

Usage for agent publishing:
    .. code-block:: python

        from langgraph.prebuilt.dns_aid import publish_graph, unpublish_graph

        # Publish a compiled graph as a discoverable agent
        await publish_graph(
            graph=compiled_agent,
            name="research-agent",
            domain="agents.example.com",
            endpoint="research.example.com",
            capabilities=["search", "summarize"],
            backend_name="route53",
        )
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _import_dns_aid() -> Any:
    """Lazy import dns_aid, raising a clear error if not installed."""
    try:
        import dns_aid

        return dns_aid
    except ImportError:
        raise ImportError(
            "The dns-aid package is required for DNS-AID discovery. "
            "Install it with: pip install dns-aid"
        )


def _import_create_backend() -> Any:
    """Lazy import the dns_aid backend factory."""
    try:
        from dns_aid.backends import create_backend

        return create_backend
    except ImportError:
        raise ImportError(
            "The dns-aid package is required for DNS-AID discovery. "
            "Install it with: pip install dns-aid"
        )


def _import_httpx() -> Any:
    """Lazy import httpx for HTTP-based agent invocation."""
    try:
        import httpx

        return httpx
    except ImportError:
        raise ImportError(
            "httpx is required for invoking discovered agents. "
            "Install it with: pip install httpx"
        )


class AgentInvokeInput(BaseModel):
    """Input schema for invoking a discovered agent."""

    query: str = Field(
        ..., description="The query or message to send to the agent"
    )


@dataclass
class DiscoveredAgent:
    """Metadata about a discovered DNS-AID agent."""

    name: str
    endpoint: str
    port: int
    protocol: str
    capabilities: list[str] = field(default_factory=list)
    version: str = ""
    description: str = ""


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def _create_agent_tool(agent: DiscoveredAgent) -> BaseTool:
    """Create a LangChain tool that invokes a discovered agent via HTTP.

    Args:
        agent: Discovered agent metadata.

    Returns:
        A BaseTool that sends queries to the agent's endpoint.
    """
    httpx = _import_httpx()

    caps_str = ", ".join(agent.capabilities) if agent.capabilities else ""
    desc_parts = [f"Invoke the '{agent.name}' agent"]
    if agent.description:
        desc_parts.append(f"({agent.description})")
    if caps_str:
        desc_parts.append(f"Capabilities: {caps_str}.")
    description = " ".join(desc_parts)

    scheme = "https" if agent.port == 443 else "http"
    base_url = f"{scheme}://{agent.endpoint}:{agent.port}"

    async def _ainvoke_agent(query: str) -> str:
        """Send a query to the discovered agent."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Try LangServe invoke endpoint first
            try:
                response = await client.post(
                    f"{base_url}/invoke",
                    json={"input": query},
                )
                if response.status_code == 200:
                    return json.dumps(response.json())
            except Exception:
                pass

            # Fallback: try A2A-style message
            try:
                response = await client.post(
                    f"{base_url}",
                    json={
                        "jsonrpc": "2.0",
                        "method": "message/send",
                        "params": {
                            "message": {
                                "parts": [
                                    {
                                        "kind": "text",
                                        "text": query,
                                    }
                                ],
                                "role": "user",
                            }
                        },
                    },
                )
                if response.status_code == 200:
                    return json.dumps(response.json())
            except Exception:
                pass

            return json.dumps(
                {
                    "error": f"Failed to invoke agent '{agent.name}' "
                    f"at {base_url}"
                }
            )

    def _invoke_agent(query: str) -> str:
        """Sync wrapper for agent invocation."""
        return _run_async(_ainvoke_agent(query))

    tool_name = f"invoke_{agent.name.replace('-', '_')}"

    return StructuredTool.from_function(
        func=_invoke_agent,
        coroutine=_ainvoke_agent,
        name=tool_name,
        description=description,
        args_schema=AgentInvokeInput,
    )


async def discover_agents(
    domain: str,
    *,
    protocol: str | None = None,
    name: str | None = None,
    require_dnssec: bool = False,
) -> list[DiscoveredAgent]:
    """Discover agents at a domain via DNS-AID.

    Queries DNS SVCB + TXT records to find published agents.

    Args:
        domain: Domain to search (e.g. 'agents.example.com').
        protocol: Filter by protocol ('a2a', 'mcp', 'https').
        name: Filter by specific agent name.
        require_dnssec: Require DNSSEC-validated responses.

    Returns:
        List of DiscoveredAgent metadata objects.
    """
    dns_aid = _import_dns_aid()
    result = await dns_aid.discover(
        domain=domain,
        protocol=protocol,
        name=name,
        require_dnssec=require_dnssec,
    )

    agents = []
    result_dict = result.model_dump()
    for agent_data in result_dict.get("agents", []):
        agents.append(
            DiscoveredAgent(
                name=agent_data.get("name", "unknown"),
                endpoint=agent_data.get("endpoint", ""),
                port=agent_data.get("port", 443),
                protocol=agent_data.get("protocol", "https"),
                capabilities=agent_data.get("capabilities", []),
                version=agent_data.get("version", ""),
                description=agent_data.get("description", ""),
            )
        )

    return agents


async def discover_tools(
    domain: str,
    *,
    protocol: str | None = None,
    name: str | None = None,
    require_dnssec: bool = False,
) -> list[BaseTool]:
    """Discover agents via DNS-AID and create tools for invoking them.

    Each discovered agent becomes a LangChain tool that can be used
    in ``create_react_agent`` or ``ToolNode``.

    Args:
        domain: Domain to search (e.g. 'agents.example.com').
        protocol: Filter by protocol ('a2a', 'mcp', 'https').
        name: Filter by specific agent name.
        require_dnssec: Require DNSSEC-validated responses.

    Returns:
        List of BaseTool instances, one per discovered agent.

    Example:
        .. code-block:: python

            tools = await discover_tools(
                "agents.example.com", protocol="mcp"
            )
            agent = create_react_agent(model, tools=tools)
    """
    agents = await discover_agents(
        domain,
        protocol=protocol,
        name=name,
        require_dnssec=require_dnssec,
    )

    tools = []
    for agent in agents:
        try:
            tool = _create_agent_tool(agent)
            tools.append(tool)
            logger.info(
                "DNS-AID: Created tool for agent '%s' at %s:%d",
                agent.name,
                agent.endpoint,
                agent.port,
            )
        except Exception:
            logger.exception(
                "DNS-AID: Failed to create tool for agent '%s'",
                agent.name,
            )

    return tools


async def publish_graph(
    graph: Any,
    *,
    name: str,
    domain: str,
    endpoint: str,
    protocol: str = "https",
    port: int = 443,
    capabilities: list[str] | None = None,
    version: str = "1.0.0",
    description: str | None = None,
    ttl: int = 3600,
    backend_name: str | None = None,
    backend: Any = None,
) -> dict[str, Any]:
    """Publish a compiled LangGraph agent as a DNS-AID discoverable service.

    Creates SVCB + TXT records so the agent can be discovered by
    other agents querying DNS.

    Args:
        graph: A compiled LangGraph (CompiledStateGraph).
            Used for metadata extraction; the graph itself is not
            modified.
        name: Agent identifier in DNS label format
            (e.g. 'research-agent').
        domain: Domain to publish under
            (e.g. 'agents.example.com').
        endpoint: Hostname where the agent is reachable.
        protocol: Protocol: 'a2a', 'mcp', or 'https'.
        port: Port number.
        capabilities: List of agent capabilities.
        version: Agent version string.
        description: Human-readable description.
        ttl: DNS record TTL in seconds.
        backend_name: DNS backend name
            (e.g. 'route53', 'cloudflare').
        backend: Pre-configured DNSBackend instance.

    Returns:
        Publish result dict.

    Example:
        .. code-block:: python

            agent = create_react_agent(model, tools=[...])
            result = await publish_graph(
                agent,
                name="research-agent",
                domain="agents.example.com",
                endpoint="research.example.com",
                capabilities=["search", "summarize"],
                backend_name="route53",
            )
    """
    dns_aid = _import_dns_aid()

    resolved_backend = backend
    if resolved_backend is None and backend_name:
        create_backend_fn = _import_create_backend()
        resolved_backend = create_backend_fn(backend_name)

    # Extract description from graph name if not provided
    if description is None and hasattr(graph, "name") and graph.name:
        description = f"LangGraph agent: {graph.name}"

    result = await dns_aid.publish(
        name=name,
        domain=domain,
        protocol=protocol,
        endpoint=endpoint,
        port=port,
        capabilities=capabilities,
        version=version,
        description=description,
        ttl=ttl,
        backend=resolved_backend,
    )

    logger.info(
        "DNS-AID: Published graph '%s' at %s", name, domain
    )
    return result.model_dump()


async def unpublish_graph(
    *,
    name: str,
    domain: str,
    protocol: str = "https",
    backend_name: str | None = None,
    backend: Any = None,
) -> bool:
    """Remove a LangGraph agent's DNS-AID records.

    Args:
        name: Agent identifier to remove.
        domain: Domain the agent is published under.
        protocol: Protocol used when publishing.
        backend_name: DNS backend name.
        backend: Pre-configured DNSBackend instance.

    Returns:
        True if records were deleted, False if not found.
    """
    dns_aid = _import_dns_aid()

    resolved_backend = backend
    if resolved_backend is None and backend_name:
        create_backend_fn = _import_create_backend()
        resolved_backend = create_backend_fn(backend_name)

    deleted = await dns_aid.unpublish(
        name=name,
        domain=domain,
        protocol=protocol,
        backend=resolved_backend,
    )

    if deleted:
        logger.info(
            "DNS-AID: Unpublished graph '%s' from %s", name, domain
        )
    else:
        logger.warning(
            "DNS-AID: Graph '%s' not found at %s", name, domain
        )

    return deleted
