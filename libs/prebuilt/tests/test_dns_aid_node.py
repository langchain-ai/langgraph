"""Tests for DNS-AID resolver node."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.prebuilt.dns_aid_node import (
    CachedDiscovery,
    DnsAidResolverNode,
    ResolverResult,
)


def _mock_discovery_result(agents: list[dict]) -> MagicMock:
    """Create a mock DiscoveryResult."""
    result = MagicMock()
    result.model_dump.return_value = {"agents": agents}
    return result


SAMPLE_AGENTS = [
    {
        "name": "search-agent",
        "endpoint": "search.example.com",
        "port": 443,
        "protocol": "a2a",
        "capabilities": ["search", "summarize"],
        "version": "1.0.0",
        "description": "Search and summarize",
    },
    {
        "name": "code-agent",
        "endpoint": "code.example.com",
        "port": 443,
        "protocol": "mcp",
        "capabilities": ["code", "debug", "test"],
        "version": "2.0.0",
        "description": "Code assistant",
    },
    {
        "name": "general-agent",
        "endpoint": "general.example.com",
        "port": 443,
        "protocol": "https",
        "capabilities": ["search", "code", "summarize", "translate"],
        "version": "1.0.0",
        "description": "General purpose",
    },
]


class TestCachedDiscovery:
    """Tests for discovery cache."""

    def test_not_expired(self):
        cached = CachedDiscovery(
            agents=[],
            discovered_at=time.monotonic(),
            ttl=300,
        )
        assert cached.is_expired is False

    def test_expired(self):
        cached = CachedDiscovery(
            agents=[],
            discovered_at=time.monotonic() - 400,
            ttl=300,
        )
        assert cached.is_expired is True


class TestCapabilityFiltering:
    """Tests for capability-based agent filtering."""

    def test_no_requirements_returns_all(self):
        resolver = DnsAidResolverNode(domain="example.com")
        result = resolver._filter_by_capabilities(SAMPLE_AGENTS)
        assert len(result) == 3

    def test_filter_by_single_capability(self):
        resolver = DnsAidResolverNode(
            domain="example.com",
            required_capabilities=["search"],
        )
        result = resolver._filter_by_capabilities(SAMPLE_AGENTS)
        names = {a["name"] for a in result}
        assert names == {"search-agent", "general-agent"}

    def test_filter_by_multiple_capabilities(self):
        resolver = DnsAidResolverNode(
            domain="example.com",
            required_capabilities=["search", "summarize"],
        )
        result = resolver._filter_by_capabilities(SAMPLE_AGENTS)
        names = {a["name"] for a in result}
        assert names == {"search-agent", "general-agent"}

    def test_no_matching_agents(self):
        resolver = DnsAidResolverNode(
            domain="example.com",
            required_capabilities=["nonexistent"],
        )
        result = resolver._filter_by_capabilities(SAMPLE_AGENTS)
        assert len(result) == 0


class TestAgentRanking:
    """Tests for agent ranking."""

    def test_specialist_ranked_higher(self):
        resolver = DnsAidResolverNode(
            domain="example.com",
            required_capabilities=["search", "summarize"],
        )
        matching = resolver._filter_by_capabilities(SAMPLE_AGENTS)
        ranked = resolver._rank_agents(matching)
        # search-agent has exactly the required caps (specialist)
        # general-agent has extra caps (generalist)
        assert ranked[0]["name"] == "search-agent"

    def test_no_requirements_preserves_order(self):
        resolver = DnsAidResolverNode(domain="example.com")
        ranked = resolver._rank_agents(SAMPLE_AGENTS)
        assert len(ranked) == 3


class TestResolverNodeCall:
    """Tests for the __call__ method."""

    @pytest.mark.asyncio
    async def test_resolve_with_dispatch(self):
        resolver = DnsAidResolverNode(
            domain="agents.example.com",
            required_capabilities=["search"],
            auto_dispatch=True,
            cache_ttl=0,
        )

        mock_dns_aid = MagicMock()
        mock_dns_aid.discover = AsyncMock(
            return_value=_mock_discovery_result(SAMPLE_AGENTS)
        )

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "search results"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.AsyncClient.return_value = mock_client

        with (
            patch(
                "langgraph.prebuilt.dns_aid_node._import_dns_aid",
                return_value=mock_dns_aid,
            ),
            patch(
                "langgraph.prebuilt.dns_aid_node._import_httpx",
                return_value=mock_httpx,
            ),
        ):
            state = {"messages": [("user", "find papers on AI")]}
            result_state = await resolver(state)

        result = result_state["dns_aid_result"]
        assert isinstance(result, ResolverResult)
        assert result.success is True
        assert result.agent_name == "search-agent"
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_resolve_without_dispatch(self):
        resolver = DnsAidResolverNode(
            domain="agents.example.com",
            required_capabilities=["code"],
            auto_dispatch=False,
            cache_ttl=0,
        )

        mock_dns_aid = MagicMock()
        mock_dns_aid.discover = AsyncMock(
            return_value=_mock_discovery_result(SAMPLE_AGENTS)
        )

        with patch(
            "langgraph.prebuilt.dns_aid_node._import_dns_aid",
            return_value=mock_dns_aid,
        ):
            state = {"messages": [("user", "fix this bug")]}
            result_state = await resolver(state)

        result = result_state["dns_aid_result"]
        assert result.success is True
        assert result.agent_name == "code-agent"
        assert result.response is None  # No dispatch

    @pytest.mark.asyncio
    async def test_no_matching_agents_returns_error(self):
        resolver = DnsAidResolverNode(
            domain="agents.example.com",
            required_capabilities=["quantum-computing"],
            cache_ttl=0,
        )

        mock_dns_aid = MagicMock()
        mock_dns_aid.discover = AsyncMock(
            return_value=_mock_discovery_result(SAMPLE_AGENTS)
        )

        with patch(
            "langgraph.prebuilt.dns_aid_node._import_dns_aid",
            return_value=mock_dns_aid,
        ):
            state = {"messages": [("user", "run quantum sim")]}
            result_state = await resolver(state)

        result = result_state["dns_aid_result"]
        assert result.success is False
        assert "quantum-computing" in result.error

    @pytest.mark.asyncio
    async def test_no_query_returns_error(self):
        resolver = DnsAidResolverNode(
            domain="agents.example.com",
            cache_ttl=0,
        )

        state = {"messages": []}
        result_state = await resolver(state)
        result = result_state["dns_aid_result"]
        assert result.success is False
        assert "No user query" in result.error

    @pytest.mark.asyncio
    async def test_dict_messages_supported(self):
        resolver = DnsAidResolverNode(
            domain="agents.example.com",
            auto_dispatch=False,
            cache_ttl=0,
        )

        mock_dns_aid = MagicMock()
        mock_dns_aid.discover = AsyncMock(
            return_value=_mock_discovery_result(SAMPLE_AGENTS)
        )

        with patch(
            "langgraph.prebuilt.dns_aid_node._import_dns_aid",
            return_value=mock_dns_aid,
        ):
            state = {
                "messages": [
                    {"role": "user", "content": "hello"},
                ]
            }
            result_state = await resolver(state)

        result = result_state["dns_aid_result"]
        assert result.success is True


class TestCaching:
    """Tests for discovery result caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        resolver = DnsAidResolverNode(
            domain="agents.example.com",
            auto_dispatch=False,
            cache_ttl=300,
        )

        mock_dns_aid = MagicMock()
        mock_dns_aid.discover = AsyncMock(
            return_value=_mock_discovery_result(SAMPLE_AGENTS)
        )

        with patch(
            "langgraph.prebuilt.dns_aid_node._import_dns_aid",
            return_value=mock_dns_aid,
        ):
            state = {"messages": [("user", "test")]}
            await resolver(state)
            await resolver(state)

        # discover should only be called once (second call uses cache)
        assert mock_dns_aid.discover.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        resolver = DnsAidResolverNode(
            domain="agents.example.com",
            auto_dispatch=False,
            cache_ttl=0,
        )

        mock_dns_aid = MagicMock()
        mock_dns_aid.discover = AsyncMock(
            return_value=_mock_discovery_result(SAMPLE_AGENTS)
        )

        with patch(
            "langgraph.prebuilt.dns_aid_node._import_dns_aid",
            return_value=mock_dns_aid,
        ):
            state = {"messages": [("user", "test")]}
            await resolver(state)
            await resolver(state)

        assert mock_dns_aid.discover.call_count == 2

    def test_clear_cache(self):
        resolver = DnsAidResolverNode(
            domain="agents.example.com",
            cache_ttl=300,
        )
        resolver._cache["test"] = CachedDiscovery(
            agents=[], discovered_at=time.monotonic(), ttl=300
        )
        assert len(resolver._cache) == 1
        resolver.clear_cache()
        assert len(resolver._cache) == 0
