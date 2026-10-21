from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from langgraph_sdk.client import SyncLangGraphClient


@pytest.mark.parametrize(
    ("thread_id", "options"), [(None, {}), ("thread-1", {"raise_error": True})]
)
def test_sync_wait_raises_on_failed_run(
    thread_id: str | None, options: dict[str, Any]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        expected_path = f"/threads/{thread_id}/runs/wait" if thread_id else "/runs/wait"
        assert request.url.path == expected_path
        return httpx.Response(
            200, json={"__error__": {"error": "RuntimeError", "message": "boom"}}
        )

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://test"
    ) as http:
        client = SyncLangGraphClient(http)
        with pytest.raises(Exception, match="RuntimeError: boom"):
            client.runs.wait(thread_id, "agent", **options)


@pytest.mark.parametrize("thread_id", [None, "thread-1"])
def test_sync_wait_can_return_failed_run(thread_id: str | None) -> None:
    response = {"__error__": {"error": "RuntimeError", "message": "boom"}}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://test"
    ) as http:
        client = SyncLangGraphClient(http)
        assert client.runs.wait(thread_id, "agent", raise_error=False) == response


@pytest.mark.parametrize("thread_id", [None, "thread-1"])
@pytest.mark.parametrize(
    "response", [{"answer": 42}, [1, 2], {"__error__": "not an error object"}]
)
def test_sync_wait_returns_successful_run(
    thread_id: str | None, response: dict[str, Any] | list[int]
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://test"
    ) as http:
        client = SyncLangGraphClient(http)
        assert client.runs.wait(thread_id, "agent") == response


@pytest.mark.parametrize("thread_id", [None, "thread-1"])
@pytest.mark.parametrize("raise_error", [False, True])
def test_sync_wait_does_not_send_raise_error(
    thread_id: str | None, raise_error: bool
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": 42})

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://test"
    ) as http:
        client = SyncLangGraphClient(http)
        client.runs.wait(thread_id, "agent", raise_error=raise_error)

    assert requests == [{"assistant_id": "agent"}]
