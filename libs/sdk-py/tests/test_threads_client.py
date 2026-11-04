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
from langgraph_sdk.schema import Checkpoint

CHECKPOINT: Checkpoint = {
    "thread_id": "thread_123",
    "checkpoint_ns": "",
    "checkpoint_id": "cp_1",
    "checkpoint_map": None,
}


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


@pytest.mark.asyncio
async def test_async_threads_get_state_select():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        threads_client = ThreadsClient(HttpClient(client))
        await threads_client.get_state("thread_123", select=["values.messages", "next"])
        await threads_client.get_state(
            "thread_123", checkpoint_id="cp_1", select=["values.messages"]
        )
        await threads_client.get_state(
            "thread_123",
            checkpoint=CHECKPOINT,
            select=["values.messages"],
        )
        await threads_client.get_state("thread_123")

    latest, by_id, by_checkpoint, default = requests
    assert latest.method == "GET"
    assert latest.url.path == "/threads/thread_123/state"
    assert latest.url.params.get_list("select") == ["values.messages", "next"]
    assert by_id.method == "GET"
    assert by_id.url.path == "/threads/thread_123/state/cp_1"
    assert by_id.url.params.get_list("select") == ["values.messages"]
    assert by_checkpoint.method == "POST"
    assert by_checkpoint.url.path == "/threads/thread_123/state/checkpoint"
    assert json.loads(by_checkpoint.content) == {
        "checkpoint": CHECKPOINT,
        "subgraphs": False,
        "select": ["values.messages"],
    }
    assert "select" not in default.url.params


@pytest.mark.asyncio
async def test_async_threads_get_history_select():
    bodies: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/threads/thread_123/history"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        threads_client = ThreadsClient(HttpClient(client))
        await threads_client.get_history("thread_123", select=["values.messages"])
        await threads_client.get_history("thread_123")

    assert bodies == [{"limit": 10, "select": ["values.messages"]}, {"limit": 10}]


def test_sync_threads_get_state_select():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        threads_client = SyncThreadsClient(SyncHttpClient(client))
        threads_client.get_state("thread_123", select=["values.messages", "next"])
        threads_client.get_state(
            "thread_123", checkpoint_id="cp_1", select=["values.messages"]
        )
        threads_client.get_state(
            "thread_123",
            checkpoint=CHECKPOINT,
            select=["values.messages"],
        )
        threads_client.get_state("thread_123")

    latest, by_id, by_checkpoint, default = requests
    assert latest.method == "GET"
    assert latest.url.path == "/threads/thread_123/state"
    assert latest.url.params.get_list("select") == ["values.messages", "next"]
    assert by_id.method == "GET"
    assert by_id.url.path == "/threads/thread_123/state/cp_1"
    assert by_id.url.params.get_list("select") == ["values.messages"]
    assert by_checkpoint.method == "POST"
    assert by_checkpoint.url.path == "/threads/thread_123/state/checkpoint"
    assert json.loads(by_checkpoint.content) == {
        "checkpoint": CHECKPOINT,
        "subgraphs": False,
        "select": ["values.messages"],
    }
    assert "select" not in default.url.params


def test_sync_threads_get_history_select():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/threads/thread_123/history"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        threads_client = SyncThreadsClient(SyncHttpClient(client))
        threads_client.get_history("thread_123", select=["values.messages"])
        threads_client.get_history("thread_123")

    assert bodies == [{"limit": 10, "select": ["values.messages"]}, {"limit": 10}]
