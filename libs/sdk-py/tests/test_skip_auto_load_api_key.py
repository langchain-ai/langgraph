"""Tests for api_key parameter behavior."""

import pytest

from langgraph_sdk import get_client, get_sync_client


class TestSkipAutoLoadApiKey:
    """Test the api_key parameter's auto-loading behavior."""

    @pytest.mark.asyncio
    async def test_get_client_loads_from_env_by_default(self, monkeypatch):
        """Test that API key is loaded from environment by default."""
        monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key-from-env")

        client = get_client(url="http://localhost:8123")
        assert "x-api-key" in client.http.client.headers
        assert client.http.client.headers["x-api-key"] == "test-key-from-env"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_get_client_skips_env_when_sentinel_used(self, monkeypatch):
        """Test that API key is not loaded from environment when None is explicitly passed."""
        monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key-from-env")

        client = get_client(url="http://localhost:8123", api_key=None)
        assert "x-api-key" not in client.http.client.headers
        await client.aclose()

    @pytest.mark.asyncio
    async def test_get_client_uses_explicit_key_when_provided(self, monkeypatch):
        """Test that explicit API key takes precedence over environment."""
        monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key-from-env")

        client = get_client(
            url="http://localhost:8123",
            api_key="explicit-key",
        )
        assert "x-api-key" in client.http.client.headers
        assert client.http.client.headers["x-api-key"] == "explicit-key"
        await client.aclose()

    def test_get_sync_client_loads_from_env_by_default(self, monkeypatch):
        """Test that sync client loads API key from environment by default."""
        monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key-from-env")

        client = get_sync_client(url="http://localhost:8123")
        assert "x-api-key" in client.http.client.headers
        assert client.http.client.headers["x-api-key"] == "test-key-from-env"
        client.close()

    def test_get_sync_client_skips_env_when_sentinel_used(self, monkeypatch):
        """Test that sync client doesn't load from environment when None is explicitly passed."""
        monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key-from-env")

        client = get_sync_client(url="http://localhost:8123", api_key=None)
        assert "x-api-key" not in client.http.client.headers
        client.close()

    def test_get_sync_client_uses_explicit_key_when_provided(self, monkeypatch):
        """Test that sync client uses explicit API key when provided."""
        monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key-from-env")

        client = get_sync_client(
            url="http://localhost:8123",
            api_key="explicit-key",
        )
        assert "x-api-key" in client.http.client.headers
        assert client.http.client.headers["x-api-key"] == "explicit-key"
        client.close()
