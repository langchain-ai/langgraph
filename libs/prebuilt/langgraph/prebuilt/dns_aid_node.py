"""DNS-AID resolver node for LangGraph.

A LangGraph node that discovers agents via DNS-AID, filters by required
capabilities, and dispatches queries to the best-matching agent. Supports
TTL-aware caching to minimize DNS lookups.

Usage in a StateGraph:
    .. code-block:: python

        from langgraph.graph import StateGraph, MessagesState
        from langgraph.prebuilt.dns_aid_node import DnsAidResolverNode

        resolver = DnsAidResolverNode(
            domain="agents.example.com",
            required_capabilities=["search"],
        )

        graph = StateGraph(MessagesState)
        graph.add_node("resolve", resolver)
        graph.add_node("agent", agent_node)
        graph.add_edge("resolve", "agent")

Usage as a standalone function:
    .. code-block:: python

        from langgraph.prebuilt.dns_aid_node import resolve_and_dispatch

        result = await resolve_and_dispatch(
            query="Find papers on transformers",
            domain="agents.example.com",
            required_capabilities=["search", "summarize"],
        )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def _import_dns_aid() -> Any:
    """Lazy import dns_aid."""
    try:
        import dns_aid

        return dns_aid
    except ImportError:
        raise ImportError(
            "The dns-aid package is required for DNS-AID resolver node. "
            "Install it with: pip install dns-aid"
        )


def _import_httpx() -> Any:
    """Lazy import httpx."""
    try:
        import httpx

        return httpx
    except ImportError:
        raise ImportError(
            "httpx is required for DNS-AID resolver node. "
            "Install it with: pip install httpx"
        )


@dataclass
class CachedDiscovery:
    """A cached DNS-AID discovery result with TTL tracking."""

    agents: list[dict[str, Any]]
    discovered_at: float
    ttl: int = 3600

    @property
    def is_expired(self) -> bool:
        return time.monotonic() - self.discovered_at > self.ttl


@dataclass
class ResolverResult:
    """Result from the resolver node."""

    agent_name: str
    agent_endpoint: str
    agent_protocol: str
    capabilities: list[str]
    response: str | None = None
    error: str | None = None
    success: bool = True


class DnsAidResolverNode:
    """LangGraph node that resolves agents via DNS-AID and dispatches queries.

    This node:
    1. Discovers agents at a domain via DNS-AID SVCB+TXT records
    2. Filters discovered agents by required capabilities
    3. Selects the best-matching agent
    4. Optionally dispatches the query to the selected agent

    Args:
        domain: DNS domain to discover agents from.
        required_capabilities: Capabilities the target agent must have.
            If empty, all agents match.
        protocol: Filter by protocol ('a2a', 'mcp', 'https', or None for any).
        require_dnssec: Require DNSSEC-validated responses.
        cache_ttl: How long to cache discovery results (seconds).
            Set to 0 to disable caching.
        auto_dispatch: If True, automatically invoke the best agent.
            If False, only resolve and return agent metadata.
        dispatch_timeout: HTTP timeout for agent invocation (seconds).
    """

    def __init__(
        self,
        domain: str,
        *,
        required_capabilities: Sequence[str] | None = None,
        protocol: str | None = None,
        require_dnssec: bool = False,
        cache_ttl: int = 300,
        auto_dispatch: bool = True,
        dispatch_timeout: float = 30.0,
    ) -> None:
        self.domain = domain
        self.required_capabilities = list(required_capabilities or [])
        self.protocol = protocol
        self.require_dnssec = require_dnssec
        self.cache_ttl = cache_ttl
        self.auto_dispatch = auto_dispatch
        self.dispatch_timeout = dispatch_timeout
        self._cache: dict[str, CachedDiscovery] = {}

    def _cache_key(self) -> str:
        """Generate cache key from discovery parameters."""
        return f"{self.domain}:{self.protocol or 'any'}"

    async def _discover(self) -> list[dict[str, Any]]:
        """Discover agents, using cache if available."""
        key = self._cache_key()
        cached = self._cache.get(key)

        if cached and not cached.is_expired:
            logger.debug("DNS-AID resolver: using cached discovery for %s", key)
            return cached.agents

        dns_aid = _import_dns_aid()
        result = await dns_aid.discover(
            domain=self.domain,
            protocol=self.protocol,
            require_dnssec=self.require_dnssec,
        )

        agents = []
        result_dict = result.model_dump()
        for agent_data in result_dict.get("agents", []):
            agents.append({
                "name": agent_data.get("name", "unknown"),
                "target_host": agent_data.get("target_host", ""),
                "port": agent_data.get("port", 443),
                "protocol": agent_data.get("protocol", "https"),
                "capabilities": agent_data.get("capabilities", []),
                "version": agent_data.get("version", ""),
                "description": agent_data.get("description", ""),
            })

        if self.cache_ttl > 0:
            self._cache[key] = CachedDiscovery(
                agents=agents,
                discovered_at=time.monotonic(),
                ttl=self.cache_ttl,
            )

        logger.info(
            "DNS-AID resolver: discovered %d agents at %s",
            len(agents),
            self.domain,
        )
        return agents

    def _filter_by_capabilities(
        self, agents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter agents that have all required capabilities."""
        if not self.required_capabilities:
            return agents

        required = set(self.required_capabilities)
        matching = []
        for agent in agents:
            agent_caps = set(agent.get("capabilities", []))
            if required.issubset(agent_caps):
                matching.append(agent)

        return matching

    def _rank_agents(
        self, agents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rank matching agents by capability coverage (more caps = better match).

        Agents with more capabilities matching the required set are ranked higher.
        Ties are broken by total capability count (prefer specialists over generalists).
        """
        if not self.required_capabilities:
            return agents

        required = set(self.required_capabilities)

        def score(agent: dict[str, Any]) -> tuple[int, int]:
            caps = set(agent.get("capabilities", []))
            overlap = len(caps & required)
            # Prefer agents with fewer extra capabilities (specialists)
            excess = len(caps - required)
            return (overlap, -excess)

        return sorted(agents, key=score, reverse=True)

    async def _dispatch(
        self, agent: dict[str, Any], query: str
    ) -> str:
        """Invoke the selected agent with a query."""
        httpx = _import_httpx()

        host = agent.get("target_host", "")
        port = agent.get("port", 443)
        protocol = agent.get("protocol", "https")
        scheme = "https" if port == 443 else "http"
        base_url = f"{scheme}://{host}:{port}"

        async with httpx.AsyncClient(timeout=self.dispatch_timeout) as client:
            # Try LangServe /invoke first
            try:
                response = await client.post(
                    f"{base_url}/invoke",
                    json={"input": query},
                )
                if response.status_code == 200:
                    return json.dumps(response.json())
            except Exception:
                pass

            # Try A2A message/send
            if protocol == "a2a":
                try:
                    response = await client.post(
                        base_url,
                        json={
                            "jsonrpc": "2.0",
                            "method": "message/send",
                            "params": {
                                "message": {
                                    "parts": [{"kind": "text", "text": query}],
                                    "role": "user",
                                }
                            },
                        },
                    )
                    if response.status_code == 200:
                        return json.dumps(response.json())
                except Exception:
                    pass

            return json.dumps({
                "error": f"Failed to invoke agent '{agent.get('name')}' at {base_url}"
            })

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the resolver node.

        Expects state with 'messages' key (list of message dicts/tuples).
        Returns state update with resolver results.
        """
        # Extract query from the last user message
        messages = state.get("messages", [])
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, dict):
                if msg.get("role") == "user" or msg.get("type") == "human":
                    query = msg.get("content", "")
                    break
            elif isinstance(msg, tuple) and len(msg) == 2:
                if msg[0] in ("user", "human"):
                    query = msg[1]
                    break
            elif hasattr(msg, "content") and hasattr(msg, "type"):
                if msg.type in ("human",):
                    query = msg.content
                    break

        if not query:
            return {
                "dns_aid_result": ResolverResult(
                    agent_name="",
                    agent_endpoint="",
                    agent_protocol="",
                    capabilities=[],
                    error="No user query found in messages",
                    success=False,
                ),
            }

        # Discover and filter
        agents = await self._discover()
        matching = self._filter_by_capabilities(agents)
        ranked = self._rank_agents(matching)

        if not ranked:
            caps_str = ", ".join(self.required_capabilities)
            return {
                "dns_aid_result": ResolverResult(
                    agent_name="",
                    agent_endpoint="",
                    agent_protocol="",
                    capabilities=[],
                    error=f"No agents found with capabilities: {caps_str}",
                    success=False,
                ),
            }

        best = ranked[0]
        result = ResolverResult(
            agent_name=best.get("name", ""),
            agent_endpoint=f"https://{best.get('target_host')}:{best.get('port')}",
            agent_protocol=best.get("protocol", ""),
            capabilities=best.get("capabilities", []),
        )

        if self.auto_dispatch:
            response = await self._dispatch(best, query)
            result.response = response

        return {"dns_aid_result": result}

    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._cache.clear()


async def resolve_and_dispatch(
    query: str,
    domain: str,
    *,
    required_capabilities: Sequence[str] | None = None,
    protocol: str | None = None,
    require_dnssec: bool = False,
    timeout: float = 30.0,
) -> ResolverResult:
    """Convenience function to resolve an agent and dispatch a query.

    Args:
        query: The query to send to the discovered agent.
        domain: DNS domain to discover agents from.
        required_capabilities: Required agent capabilities.
        protocol: Filter by protocol.
        require_dnssec: Require DNSSEC validation.
        timeout: HTTP timeout for dispatch.

    Returns:
        ResolverResult with agent metadata and response.
    """
    resolver = DnsAidResolverNode(
        domain=domain,
        required_capabilities=required_capabilities,
        protocol=protocol,
        require_dnssec=require_dnssec,
        auto_dispatch=True,
        dispatch_timeout=timeout,
        cache_ttl=0,  # No caching for one-shot usage
    )

    state = {"messages": [("user", query)]}
    result_state = await resolver(state)
    return result_state["dns_aid_result"]
