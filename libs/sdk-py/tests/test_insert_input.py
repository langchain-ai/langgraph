"""Unit tests for RunsClient.insert_input / SyncRunsClient.insert_input.

These tests use httpx.MockTransport so no server is required.  They verify:
- The correct HTTP method and path are used.
- The input payload is serialised as {"input": <value>} in the request body.
- The decoded Run dict is returned on success.
- An httpx.HTTPStatusError is raised on a 409 Conflict (run not active).
"""

from __future__ import annotations

import json

import httpx
import pytest

from langgraph_sdk.client import HttpClient, RunsClient, SyncHttpClient, SyncRunsClient

THREAD_ID = "thread-abc"
RUN_ID = "run-xyz"
EXPECTED_PATH = f"/threads/{THREAD_ID}/runs/{RUN_ID}/input"

RUN_PAYLOAD = {
    "run_id": RUN_ID,
    "thread_id": THREAD_ID,
    "assistant_id": "asst-1",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:01Z",
    "status": "running",
    "metadata": {},
    "multitask_strategy": "reject",
}


@pytest.mark.asyncio
class TestAsyncInsertInput:
    async def test_posts_to_correct_path(self) -> None:
        captured: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=RUN_PAYLOAD)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs = RunsClient(HttpClient(client))
            await runs.insert_input(
                THREAD_ID, RUN_ID, input={"messages": [{"role": "user", "content": "hi"}]}
            )

        assert len(captured) == 1
        req = captured[0]
        assert req.method == "POST"
        assert req.url.path == EXPECTED_PATH

    async def test_input_serialised_as_json_body(self) -> None:
        captured: list[httpx.Request] = []
        user_input = {"messages": [{"role": "user", "content": "steer me"}]}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=RUN_PAYLOAD)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs = RunsClient(HttpClient(client))
            await runs.insert_input(THREAD_ID, RUN_ID, input=user_input)

        body = json.loads(captured[0].content)
        assert body == {"input": user_input}

    async def test_returns_run_dict(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=RUN_PAYLOAD)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs = RunsClient(HttpClient(client))
            result = await runs.insert_input(THREAD_ID, RUN_ID, input={})

        assert result["run_id"] == RUN_ID
        assert result["thread_id"] == THREAD_ID
        assert result["status"] == "running"

    async def test_conflict_raises_http_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(409, json={"detail": "Run is not active"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs = RunsClient(HttpClient(client))
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await runs.insert_input(THREAD_ID, RUN_ID, input={})

        assert exc_info.value.response.status_code == 409

    async def test_custom_headers_forwarded(self) -> None:
        captured: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=RUN_PAYLOAD)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs = RunsClient(HttpClient(client))
            await runs.insert_input(
                THREAD_ID, RUN_ID, input={}, headers={"X-Custom": "value"}
            )

        assert captured[0].headers.get("x-custom") == "value"


class TestSyncInsertInput:
    def test_posts_to_correct_path(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=RUN_PAYLOAD)

        transport = httpx.MockTransport(handler)
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs = SyncRunsClient(SyncHttpClient(client))
            runs.insert_input(
                THREAD_ID, RUN_ID, input={"messages": [{"role": "user", "content": "hi"}]}
            )

        assert len(captured) == 1
        req = captured[0]
        assert req.method == "POST"
        assert req.url.path == EXPECTED_PATH

    def test_input_serialised_as_json_body(self) -> None:
        captured: list[httpx.Request] = []
        user_input = {"messages": [{"role": "user", "content": "steer me"}]}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=RUN_PAYLOAD)

        transport = httpx.MockTransport(handler)
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs = SyncRunsClient(SyncHttpClient(client))
            runs.insert_input(THREAD_ID, RUN_ID, input=user_input)

        body = json.loads(captured[0].content)
        assert body == {"input": user_input}

    def test_returns_run_dict(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=RUN_PAYLOAD)

        transport = httpx.MockTransport(handler)
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs = SyncRunsClient(SyncHttpClient(client))
            result = runs.insert_input(THREAD_ID, RUN_ID, input={})

        assert result["run_id"] == RUN_ID
        assert result["thread_id"] == THREAD_ID
        assert result["status"] == "running"

    def test_conflict_raises_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(409, json={"detail": "Run is not active"})

        transport = httpx.MockTransport(handler)
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs = SyncRunsClient(SyncHttpClient(client))
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                runs.insert_input(THREAD_ID, RUN_ID, input={})

        assert exc_info.value.response.status_code == 409

    def test_custom_headers_forwarded(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=RUN_PAYLOAD)

        transport = httpx.MockTransport(handler)
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            runs = SyncRunsClient(SyncHttpClient(client))
            runs.insert_input(
                THREAD_ID, RUN_ID, input={}, headers={"X-Custom": "value"}
            )

        assert captured[0].headers.get("x-custom") == "value"
