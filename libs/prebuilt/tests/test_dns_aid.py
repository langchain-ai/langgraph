"""Unit tests for DNS-AID discovery utilities."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.prebuilt.dns_aid import (
    DiscoveredAgent,
    _create_agent_tool,
    discover_agents,
    discover_tools,
    publish_graph,
    unpublish_graph,
)


class TestDiscoveredAgent:
    """Tests for DiscoveredAgent dataclass."""

    def test_defaults(self) -> None:
        agent = DiscoveredAgent(
            name="test", endpoint="example.com", port=443, protocol="https"
        )
        assert agent.name == "test"
        assert agent.capabilities == []
        assert agent.version == ""
        assert agent.description == ""

    def test_with_all_fields(self) -> None:
        agent = DiscoveredAgent(
            name="search-agent",
            endpoint="search.example.com",
            port=8080,
            protocol="mcp",
            capabilities=["search", "summarize"],
            version="2.0.0",
            description="A search agent",
        )
        assert agent.name == "search-agent"
        assert agent.capabilities == ["search", "summarize"]
        assert agent.version == "2.0.0"


class TestCreateAgentTool:
    """Tests for _create_agent_tool."""

    def test_creates_tool_with_correct_name(self) -> None:
        agent = DiscoveredAgent(
            name="search-agent",
            endpoint="search.example.com",
            port=443,
            protocol="https",
        )
        tool = _create_agent_tool(agent)
        assert tool.name == "invoke_search_agent"

    def test_creates_tool_with_description(self) -> None:
        agent = DiscoveredAgent(
            name="search-agent",
            endpoint="search.example.com",
            port=443,
            protocol="https",
            capabilities=["search", "summarize"],
            description="A powerful search agent",
        )
        tool = _create_agent_tool(agent)
        assert "search-agent" in tool.description
        assert "search, summarize" in tool.description
        assert "A powerful search agent" in tool.description

    def test_creates_tool_with_schema(self) -> None:
        agent = DiscoveredAgent(
            name="test", endpoint="example.com", port=443, protocol="https"
        )
        tool = _create_agent_tool(agent)
        schema = tool.args_schema.model_json_schema()
        assert "query" in schema["properties"]


class TestDiscoverAgents:
    """Tests for discover_agents."""

    @pytest.mark.asyncio
    async def test_returns_discovered_agents(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "query": "_agents.example.com",
            "domain": "example.com",
            "agents": [
                {
                    "name": "agent-1",
                    "endpoint": "agent1.example.com",
                    "port": 443,
                    "protocol": "mcp",
                    "capabilities": ["search"],
                    "version": "1.0.0",
                    "description": "First agent",
                },
                {
                    "name": "agent-2",
                    "endpoint": "agent2.example.com",
                    "port": 8080,
                    "protocol": "a2a",
                    "capabilities": [],
                    "version": "2.0.0",
                    "description": "",
                },
            ],
            "count": 2,
        }

        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.discover = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            agents = await discover_agents("example.com", protocol="mcp")

        assert len(agents) == 2
        assert agents[0].name == "agent-1"
        assert agents[0].endpoint == "agent1.example.com"
        assert agents[0].capabilities == ["search"]
        assert agents[1].name == "agent-2"
        assert agents[1].port == 8080

    @pytest.mark.asyncio
    async def test_passes_query_params(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"agents": []}

        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.discover = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            await discover_agents(
                "example.com",
                protocol="mcp",
                name="specific-agent",
                require_dnssec=True,
            )

        call_kwargs = mock_dns_aid.discover.call_args.kwargs
        assert call_kwargs["domain"] == "example.com"
        assert call_kwargs["protocol"] == "mcp"
        assert call_kwargs["name"] == "specific-agent"
        assert call_kwargs["require_dnssec"] is True

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_agents(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"agents": []}

        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.discover = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            agents = await discover_agents("empty.example.com")

        assert agents == []


class TestDiscoverTools:
    """Tests for discover_tools."""

    @pytest.mark.asyncio
    async def test_creates_tools_from_discovered_agents(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "agents": [
                {
                    "name": "search-agent",
                    "endpoint": "search.example.com",
                    "port": 443,
                    "protocol": "https",
                    "capabilities": ["search"],
                    "version": "1.0.0",
                    "description": "Search agent",
                },
            ],
        }

        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.discover = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            tools = await discover_tools("example.com")

        assert len(tools) == 1
        assert tools[0].name == "invoke_search_agent"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_agents(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"agents": []}

        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.discover = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            tools = await discover_tools("empty.example.com")

        assert tools == []


class TestPublishGraph:
    """Tests for publish_graph."""

    @pytest.mark.asyncio
    async def test_publishes_graph(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "success": True,
            "agent": {"name": "my-graph"},
        }

        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            mock_graph = MagicMock()
            mock_graph.name = "test-graph"

            result = await publish_graph(
                mock_graph,
                name="my-graph",
                domain="agents.example.com",
                endpoint="api.example.com",
                capabilities=["search", "summarize"],
            )

        assert result["success"] is True
        call_kwargs = mock_dns_aid.publish.call_args.kwargs
        assert call_kwargs["name"] == "my-graph"
        assert call_kwargs["domain"] == "agents.example.com"
        assert call_kwargs["endpoint"] == "api.example.com"
        assert call_kwargs["capabilities"] == ["search", "summarize"]

    @pytest.mark.asyncio
    async def test_uses_graph_name_for_description(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"success": True}

        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            mock_graph = MagicMock()
            mock_graph.name = "research-assistant"

            await publish_graph(
                mock_graph,
                name="research",
                domain="agents.example.com",
                endpoint="api.example.com",
            )

        call_kwargs = mock_dns_aid.publish.call_args.kwargs
        assert "research-assistant" in call_kwargs["description"]

    @pytest.mark.asyncio
    async def test_uses_explicit_description(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"success": True}

        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            mock_graph = MagicMock()
            mock_graph.name = "research-assistant"

            await publish_graph(
                mock_graph,
                name="research",
                domain="agents.example.com",
                endpoint="api.example.com",
                description="Custom description",
            )

        call_kwargs = mock_dns_aid.publish.call_args.kwargs
        assert call_kwargs["description"] == "Custom description"

    @pytest.mark.asyncio
    async def test_passes_backend(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"success": True}
        mock_backend = MagicMock()

        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid

            await publish_graph(
                MagicMock(),
                name="agent",
                domain="example.com",
                endpoint="api.example.com",
                backend=mock_backend,
            )

        call_kwargs = mock_dns_aid.publish.call_args.kwargs
        assert call_kwargs["backend"] is mock_backend

    @pytest.mark.asyncio
    async def test_resolves_backend_name(self) -> None:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"success": True}
        mock_backend_instance = MagicMock()

        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import, patch(
            "langgraph.prebuilt.dns_aid._import_create_backend"
        ) as mock_create:
            mock_dns_aid = MagicMock()
            mock_dns_aid.publish = AsyncMock(return_value=mock_result)
            mock_import.return_value = mock_dns_aid
            mock_create.return_value = MagicMock(
                return_value=mock_backend_instance
            )

            await publish_graph(
                MagicMock(),
                name="agent",
                domain="example.com",
                endpoint="api.example.com",
                backend_name="route53",
            )

        call_kwargs = mock_dns_aid.publish.call_args.kwargs
        assert call_kwargs["backend"] is mock_backend_instance


class TestUnpublishGraph:
    """Tests for unpublish_graph."""

    @pytest.mark.asyncio
    async def test_unpublishes_successfully(self) -> None:
        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.unpublish = AsyncMock(return_value=True)
            mock_import.return_value = mock_dns_aid

            result = await unpublish_graph(
                name="my-graph",
                domain="agents.example.com",
            )

        assert result is True
        call_kwargs = mock_dns_aid.unpublish.call_args.kwargs
        assert call_kwargs["name"] == "my-graph"
        assert call_kwargs["domain"] == "agents.example.com"

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self) -> None:
        with patch(
            "langgraph.prebuilt.dns_aid._import_dns_aid"
        ) as mock_import:
            mock_dns_aid = MagicMock()
            mock_dns_aid.unpublish = AsyncMock(return_value=False)
            mock_import.return_value = mock_dns_aid

            result = await unpublish_graph(
                name="missing",
                domain="agents.example.com",
            )

        assert result is False


class TestImportErrors:
    """Tests for lazy import error handling."""

    def test_import_dns_aid_raises_when_missing(self) -> None:
        from langgraph.prebuilt.dns_aid import _import_dns_aid

        with patch.dict("sys.modules", {"dns_aid": None}):
            with pytest.raises(ImportError, match="dns-aid"):
                _import_dns_aid()

    def test_import_create_backend_raises_when_missing(self) -> None:
        from langgraph.prebuilt.dns_aid import _import_create_backend

        with patch.dict(
            "sys.modules",
            {"dns_aid": None, "dns_aid.backends": None},
        ):
            with pytest.raises(ImportError, match="dns-aid"):
                _import_create_backend()

    def test_import_httpx_raises_when_missing(self) -> None:
        from langgraph.prebuilt.dns_aid import _import_httpx

        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx"):
                _import_httpx()
