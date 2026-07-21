"""Test that langsmith_tracing parameter is correctly mapped to langsmith_tracer in payloads."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from langgraph_sdk.schema import LangSmithTracing


@pytest.fixture
def tracing_config() -> LangSmithTracing:
    return LangSmithTracing(
        project_name="my-project",
        example_id="example-123",
    )


class TestLangSmithTracingPayload:
    """Verify langsmith_tracing param maps to langsmith_tracer in request payload."""

    @pytest.mark.asyncio
    async def test_async_create_includes_langsmith_tracer(self, tracing_config):
        """Test that async create sends langsmith_tracer in payload."""
        from langgraph_sdk._async.runs import RunsClient

        captured: dict[str, Any] = {}

        async def mock_post(_path, *, json=None, **_kwargs):
            captured["json"] = json
            return {"run_id": "r1", "status": "pending"}

        http = MagicMock()
        http.post = AsyncMock(side_effect=mock_post)
        client = RunsClient(http)

        await client.create(
            thread_id="t1",
            assistant_id="a1",
            langsmith_tracing=tracing_config,
        )

        assert "langsmith_tracer" in captured["json"]
        assert captured["json"]["langsmith_tracer"] == {
            "project_name": "my-project",
            "example_id": "example-123",
        }

    def test_sync_create_includes_langsmith_tracer(self, tracing_config):
        """Test that sync create sends langsmith_tracer in payload."""
        from langgraph_sdk._sync.runs import SyncRunsClient

        captured: dict[str, Any] = {}

        def mock_post(_path, *, json=None, **_kwargs):
            captured["json"] = json
            return {"run_id": "r1", "status": "pending"}

        http = MagicMock()
        http.post = MagicMock(side_effect=mock_post)
        client = SyncRunsClient(http)

        client.create(
            thread_id="t1",
            assistant_id="a1",
            langsmith_tracing=tracing_config,
        )

        assert "langsmith_tracer" in captured["json"]
        assert captured["json"]["langsmith_tracer"] == {
            "project_name": "my-project",
            "example_id": "example-123",
        }

    def test_sync_wait_includes_langsmith_tracer(self, tracing_config):
        """Test that sync wait sends langsmith_tracer in payload."""
        from langgraph_sdk._sync.runs import SyncRunsClient

        captured: dict[str, Any] = {}

        def mock_request_reconnect(_path, _method, *, json=None, **_kwargs):
            captured["json"] = json
            return {"messages": []}

        http = MagicMock()
        http.request_reconnect = MagicMock(side_effect=mock_request_reconnect)
        client = SyncRunsClient(http)

        client.wait(
            thread_id="t1",
            assistant_id="a1",
            langsmith_tracing=tracing_config,
        )

        assert "langsmith_tracer" in captured["json"]
        assert captured["json"]["langsmith_tracer"] == {
            "project_name": "my-project",
            "example_id": "example-123",
        }

    def test_create_without_langsmith_tracing_excludes_key(self):
        """Test that langsmith_tracer is not in payload when not provided."""
        from langgraph_sdk._sync.runs import SyncRunsClient

        captured: dict[str, Any] = {}

        def mock_post(_path, *, json=None, **_kwargs):
            captured["json"] = json
            return {"run_id": "r1", "status": "pending"}

        http = MagicMock()
        http.post = MagicMock(side_effect=mock_post)
        client = SyncRunsClient(http)

        client.create(
            thread_id="t1",
            assistant_id="a1",
        )

        assert "langsmith_tracer" not in captured["json"]

    def test_langsmith_tracing_project_name_only(self):
        """Test that langsmith_tracing works with only project_name."""
        from langgraph_sdk._sync.runs import SyncRunsClient

        captured: dict[str, Any] = {}

        def mock_post(_path, *, json=None, **_kwargs):
            captured["json"] = json
            return {"run_id": "r1", "status": "pending"}

        http = MagicMock()
        http.post = MagicMock(side_effect=mock_post)
        client = SyncRunsClient(http)

        client.create(
            thread_id="t1",
            assistant_id="a1",
            langsmith_tracing={"project_name": "my-project"},
        )

        assert captured["json"]["langsmith_tracer"] == {
            "project_name": "my-project",
        }
