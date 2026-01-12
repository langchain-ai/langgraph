"""Tests for the crons client."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from langgraph_sdk.client import (
    CronClient,
    HttpClient,
    SyncCronClient,
    SyncHttpClient,
)


def _cron_payload() -> dict[str, object]:
    """Return a mock cron response payload."""
    return {
        "run_id": "run_123",
        "thread_id": "thread_123",
        "assistant_id": "asst_123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "status": "success",
        "metadata": {},
        "multitask_strategy": "reject",
    }


@pytest.mark.asyncio
async def test_async_create_for_thread():
    """Test that CronClient.create_for_thread works without end_time."""
    cron = _cron_payload()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/threads/thread_123/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 0 * * *"
        assert body["assistant_id"] == "asst_123"
        assert "end_time" not in body  # Should be filtered out by the None check

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        cron_client = CronClient(http_client)
        result = await cron_client.create_for_thread(
            thread_id="thread_123",
            assistant_id="asst_123",
            schedule="0 0 * * *",
        )

    assert result == cron


@pytest.mark.asyncio
async def test_async_create_for_thread_with_end_time():
    """Test that CronClient.create_for_thread includes end_time in the payload."""
    cron = _cron_payload()
    end_time = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/threads/thread_123/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 0 * * *"
        assert body["assistant_id"] == "asst_123"
        assert body["end_time"] == "2025-12-31T23:59:59+00:00"

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        cron_client = CronClient(http_client)
        result = await cron_client.create_for_thread(
            thread_id="thread_123",
            assistant_id="asst_123",
            schedule="0 0 * * *",
            end_time=end_time,
        )

    assert result == cron


@pytest.mark.asyncio
async def test_async_create():
    """Test that CronClient.create works without end_time."""
    cron = _cron_payload()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 12 * * *"
        assert body["assistant_id"] == "asst_456"
        assert "end_time" not in body  # Should be filtered out by the None check

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        cron_client = CronClient(http_client)
        result = await cron_client.create(
            assistant_id="asst_456",
            schedule="0 12 * * *",
        )

    assert result == cron


@pytest.mark.asyncio
async def test_async_create_with_end_time():
    """Test that CronClient.create includes end_time in the payload."""
    cron = _cron_payload()
    end_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 12 * * *"
        assert body["assistant_id"] == "asst_456"
        assert body["end_time"] == "2025-06-15T12:00:00+00:00"

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        cron_client = CronClient(http_client)
        result = await cron_client.create(
            assistant_id="asst_456",
            schedule="0 12 * * *",
            end_time=end_time,
        )

    assert result == cron


def test_sync_create_for_thread():
    """Test that SyncCronClient.create_for_thread works without end_time."""
    cron = _cron_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/threads/thread_123/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 0 * * *"
        assert body["assistant_id"] == "asst_123"
        assert "end_time" not in body  # Should be filtered out by the None check

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.create_for_thread(
            thread_id="thread_123",
            assistant_id="asst_123",
            schedule="0 0 * * *",
        )

    assert result == cron


def test_sync_create_for_thread_with_end_time():
    """Test that SyncCronClient.create_for_thread includes end_time in the payload."""
    cron = _cron_payload()
    end_time = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/threads/thread_123/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 0 * * *"
        assert body["assistant_id"] == "asst_123"
        assert body["end_time"] == "2025-12-31T23:59:59+00:00"

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.create_for_thread(
            thread_id="thread_123",
            assistant_id="asst_123",
            schedule="0 0 * * *",
            end_time=end_time,
        )

    assert result == cron


def test_sync_create():
    """Test that SyncCronClient.create works without end_time."""
    cron = _cron_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 12 * * *"
        assert body["assistant_id"] == "asst_456"
        assert "end_time" not in body  # Should be filtered out by the None check

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.create(
            assistant_id="asst_456",
            schedule="0 12 * * *",
        )

    assert result == cron


def test_sync_create_with_end_time():
    """Test that SyncCronClient.create includes end_time in the payload."""
    cron = _cron_payload()
    end_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/runs/crons"

        # Parse the request body
        body = json.loads(request.content)
        assert body["schedule"] == "0 12 * * *"
        assert body["assistant_id"] == "asst_456"
        assert body["end_time"] == "2025-06-15T12:00:00+00:00"

        return httpx.Response(200, json=cron)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        cron_client = SyncCronClient(http_client)
        result = cron_client.create(
            assistant_id="asst_456",
            schedule="0 12 * * *",
            end_time=end_time,
        )

    assert result == cron
