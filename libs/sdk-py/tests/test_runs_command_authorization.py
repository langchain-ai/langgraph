from __future__ import annotations

from typing import Any, cast

import httpx
import orjson
import pytest

from langgraph_sdk.client import HttpClient, RunsClient, SyncHttpClient, SyncRunsClient


def _valid_command() -> dict[str, Any]:
    return {
        "resume": {"approved": True},
        "resume_authorization": {
            "actor_id": "approver-1",
            "token": "opaque-token",
            "signature": "signed-token",
            "issuer": "policy-service",
        },
    }


@pytest.mark.asyncio
async def test_async_runs_create_allows_valid_resume_authorization():
    seen_json: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_json.update(cast(dict[str, Any], orjson.loads(await request.aread())))
        return httpx.Response(200, json={"run_id": "run-1"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        runs_client = RunsClient(HttpClient(client))
        await runs_client.create(
            thread_id=None,
            assistant_id="assistant-1",
            command=_valid_command(),
        )

    assert seen_json["command"]["resume_authorization"]["actor_id"] == "approver-1"


def test_sync_runs_wait_rejects_resume_authorization_without_resume():
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP request should not be sent for invalid command")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        runs_client = SyncRunsClient(SyncHttpClient(client))
        with pytest.raises(
            ValueError,
            match=r"`command\.resume_authorization` requires `command\.resume`\.",
        ):
            runs_client.wait(
                thread_id=None,
                assistant_id="assistant-1",
                command={
                    "resume_authorization": {
                        "actor_id": "approver-1",
                        "token": "opaque-token",
                        "signature": "signed-token",
                    }
                },
            )


@pytest.mark.asyncio
async def test_async_runs_create_rejects_invalid_resume_authorization_shape():
    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP request should not be sent for invalid command")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        runs_client = RunsClient(HttpClient(client))
        with pytest.raises(
            ValueError,
            match="missing required key\\(s\\): signature",
        ):
            await runs_client.create(
                thread_id=None,
                assistant_id="assistant-1",
                command={
                    "resume": {"approved": True},
                    "resume_authorization": {
                        "actor_id": "approver-1",
                        "token": "opaque-token",
                    },
                },
            )


def test_sync_runs_create_batch_validates_resume_authorization():
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP request should not be sent for invalid command")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        runs_client = SyncRunsClient(SyncHttpClient(client))
        with pytest.raises(
            ValueError,
            match="contains unknown key\\(s\\): unexpected",
        ):
            runs_client.create_batch(
                cast(
                    Any,
                    [
                        {
                            "thread_id": None,
                            "assistant_id": "assistant-1",
                            "command": {
                                "resume": {"approved": True},
                                "resume_authorization": {
                                    "actor_id": "approver-1",
                                    "token": "opaque-token",
                                    "signature": "signed-token",
                                    "unexpected": "oops",
                                },
                            },
                        }
                    ],
                )
            )
