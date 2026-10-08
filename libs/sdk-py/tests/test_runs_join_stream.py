from __future__ import annotations

import json

import httpx
import pytest

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._async.runs import RunsClient
from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk._sync.runs import SyncRunsClient


def test_sync_join_stream_encodes_stream_modes_as_json() -> None:
    observed: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.url.params.get("stream_mode"))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b""
        )

    with httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as raw_client:
        runs = SyncRunsClient(SyncHttpClient(raw_client))
        list(runs.join_stream("thread", "run", stream_mode=["values", "updates"]))

    assert observed == [json.dumps(["values", "updates"])]


@pytest.mark.asyncio
async def test_async_join_stream_encodes_stream_modes_as_json() -> None:
    observed: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.url.params.get("stream_mode"))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b""
        )

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as raw_client:
        runs = RunsClient(HttpClient(raw_client))
        async for _ in runs.join_stream(
            "thread", "run", stream_mode=["values", "updates"]
        ):
            pass

    assert observed == [json.dumps(["values", "updates"])]
