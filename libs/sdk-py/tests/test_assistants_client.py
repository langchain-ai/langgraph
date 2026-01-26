from __future__ import annotations

import httpx
import pytest

from langgraph_sdk.client import (
    AssistantsClient,
    HttpClient,
    SyncAssistantsClient,
    SyncHttpClient,
)


def _assistant_payload() -> dict[str, object]:
    return {
        "assistant_id": "asst_123",
        "graph_id": "graph_123",
        "config": {"configurable": {"foo": "bar"}},
        "context": {"foo": "bar"},
        "created_at": "2024-01-01T00:00:00Z",
        "metadata": {"env": "test"},
        "version": 1,
        "name": "My Assistant",
        "description": "Example",
        "updated_at": "2024-01-02T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_assistants_search_returns_list_by_default():
    assistant = _assistant_payload()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/assistants/search"
        return httpx.Response(200, json=[assistant])

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        assistants_client = AssistantsClient(http_client)
        result = await assistants_client.search(limit=3)

    assert result == [assistant]


@pytest.mark.asyncio
async def test_assistants_search_can_return_object_with_pagination_metadata():
    assistant = _assistant_payload()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/assistants/search"
        return httpx.Response(
            200,
            headers={"X-Pagination-Next": "42"},
            json=[assistant],
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        assistants_client = AssistantsClient(http_client)
        result = await assistants_client.search(response_format="object")

    assert result == {"assistants": [assistant], "next": "42"}


def test_sync_assistants_search_can_return_object_with_pagination_metadata():
    assistant = _assistant_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/assistants/search"
        return httpx.Response(
            200,
            headers={"X-Pagination-Next": "84"},
            json=[assistant],
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        assistants_client = SyncAssistantsClient(http_client)
        result = assistants_client.search(response_format="object")

    assert result == {"assistants": [assistant], "next": "84"}
