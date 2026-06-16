from __future__ import annotations

import json

import httpx
import pytest

from langgraph_sdk.client import (
    HttpClient,
    SyncHttpClient,
    SyncThreadsClient,
    ThreadsClient,
)


@pytest.mark.asyncio
async def test_async_threads_update_return_minimal():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/threads/thread_123"
        assert request.headers["Prefer"] == "return=minimal"
        assert json.loads(request.content) == {"metadata": {"foo": "bar"}}
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        threads_client = ThreadsClient(http_client)
        result = await threads_client.update(
            "thread_123",
            metadata={"foo": "bar"},
            return_minimal=True,
        )

    assert result is None


def test_sync_threads_update_return_minimal():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/threads/thread_123"
        assert request.headers["Prefer"] == "return=minimal"
        assert json.loads(request.content) == {"metadata": {"foo": "bar"}}
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        threads_client = SyncThreadsClient(http_client)
        result = threads_client.update(
            "thread_123",
            metadata={"foo": "bar"},
            return_minimal=True,
        )

    assert result is None
