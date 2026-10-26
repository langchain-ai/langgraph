from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest

from langgraph_sdk.client import (
    HttpClient,
    RunsClient,
    SyncHttpClient,
    SyncRunsClient,
)


class _BodyDropsMidRead(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield b'{"partial'
        raise httpx.ReadError("connection dropped")

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"partial'
        raise httpx.ReadError("connection dropped")


def _wait_handler(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                headers={"Location": "/threads/t1/runs/r1/join"},
                stream=_BodyDropsMidRead(),
            )
        return httpx.Response(200, json={"done": True})

    return handler


def _assert_bodyless_get(request: httpx.Request) -> None:
    assert request.method == "GET"
    assert request.url.path == "/threads/t1/runs/r1/join"
    assert "content-length" not in request.headers, request.headers
    assert "content-type" not in request.headers, request.headers
    assert request.read() == b""


def test_sync_wait_reconnects_with_a_bodyless_get():
    requests: list[httpx.Request] = []
    transport = httpx.MockTransport(_wait_handler(requests))
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        runs = SyncRunsClient(SyncHttpClient(client))
        with pytest.warns(UserWarning, match="attempting reconnect"):
            result = runs.wait("t1", "agent", input={"question": "hi"})

    assert result == {"done": True}
    _assert_bodyless_get(requests[1])


async def test_async_wait_reconnects_with_a_bodyless_get():
    requests: list[httpx.Request] = []
    transport = httpx.MockTransport(_wait_handler(requests))
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        runs = RunsClient(HttpClient(client))
        with pytest.warns(UserWarning, match="attempting reconnect"):
            result = await runs.wait("t1", "agent", input={"question": "hi"})

    assert result == {"done": True}
    _assert_bodyless_get(requests[1])


def test_sync_wait_raises_the_run_error():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"__error__": {"error": "RuntimeError", "message": "boom"}}
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        runs = SyncRunsClient(SyncHttpClient(client))
        with pytest.raises(Exception, match="RuntimeError: boom"):
            runs.wait("t1", "agent", input={}, raise_error=True)

    assert "raise_error" not in json.loads(requests[0].content)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_wait_returns_the_run_error_without_raise_error(asynchronous):
    requests: list[httpx.Request] = []
    run_error = {"__error__": {"error": "RuntimeError", "message": "boom"}}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=run_error)

    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            result = await RunsClient(HttpClient(client)).wait(
                "t1", "agent", input={}, raise_error=False
            )
    else:
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            result = SyncRunsClient(SyncHttpClient(client)).wait(
                "t1", "agent", input={}, raise_error=False
            )

    assert result == run_error
    assert "raise_error" not in json.loads(requests[0].content)
