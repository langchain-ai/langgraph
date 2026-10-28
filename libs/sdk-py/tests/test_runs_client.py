from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any

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


def _done(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"done": True})


def _wait_handler(
    requests: list[httpx.Request],
    reconnects: Sequence[Callable[[httpx.Request], httpx.Response]] = (_done,),
):
    """POST drops mid-body; the reconnects answer from `reconnects` in order."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                headers={"Location": "/threads/t1/runs/r1/join"},
                stream=_BodyDropsMidRead(),
            )
        return reconnects[len(requests) - 2](request)

    return handler


def _assert_bodyless_get(request: httpx.Request) -> None:
    assert request.method == "GET"
    assert request.url.path == "/threads/t1/runs/r1/join"
    assert "content-length" not in request.headers, request.headers
    assert "content-type" not in request.headers, request.headers
    assert request.read() == b""


@pytest.mark.usefixtures("no_sleep")
def test_sync_wait_reconnects_with_a_bodyless_get():
    requests: list[httpx.Request] = []
    transport = httpx.MockTransport(_wait_handler(requests))
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        runs = SyncRunsClient(SyncHttpClient(client))
        with pytest.warns(UserWarning, match="attempting reconnect"):
            result = runs.wait("t1", "agent", input={"question": "hi"})

    assert result == {"done": True}
    _assert_bodyless_get(requests[1])


@pytest.mark.usefixtures("no_sleep")
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


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("[Errno 61] Connection refused", request=request)


async def _wait(
    handler: Callable[[httpx.Request], httpx.Response], *, asynchronous: bool
) -> Any:
    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            return await RunsClient(HttpClient(client)).wait("t1", "agent", input={})
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        return SyncRunsClient(SyncHttpClient(client)).wait("t1", "agent", input={})


# What a server restart or a deploy looks like to the reconnect GET: each one is
# retried after a backoff.
RECONNECT_FAILURES = {
    "429": lambda request: httpx.Response(429, request=request),
    "503": lambda request: httpx.Response(503, request=request),
    "connect_error": _connect_error,
    "body_drops": lambda _request: httpx.Response(
        200,
        headers={"Location": "/threads/t1/runs/r1/join"},
        stream=_BodyDropsMidRead(),
    ),
}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "failure", RECONNECT_FAILURES.values(), ids=RECONNECT_FAILURES.keys()
)
async def test_wait_retries_a_failed_reconnect_after_a_backoff(
    failure, asynchronous, no_sleep
):
    requests: list[httpx.Request] = []
    with pytest.warns(UserWarning, match="attempting reconnect"):
        result = await _wait(
            _wait_handler(requests, [failure, _done]), asynchronous=asynchronous
        )

    assert result == {"done": True}
    assert len(requests) == 3
    assert len(no_sleep) == 2 and min(no_sleep) >= 1, no_sleep


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_wait_raises_after_five_failed_reconnects(asynchronous):
    requests: list[httpx.Request] = []
    with (
        pytest.warns(UserWarning, match="attempting reconnect"),
        pytest.raises(httpx.HTTPStatusError, match="503"),
    ):
        await _wait(
            _wait_handler(requests, [RECONNECT_FAILURES["503"]] * 5),
            asynchronous=asynchronous,
        )

    assert len(requests) == 6


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_wait_does_not_retry_a_client_error(asynchronous):
    requests: list[httpx.Request] = []
    with (
        pytest.warns(UserWarning, match="attempting reconnect"),
        pytest.raises(httpx.HTTPStatusError, match="404"),
    ):
        await _wait(
            _wait_handler(
                requests, [lambda request: httpx.Response(404, request=request)]
            ),
            asynchronous=asynchronous,
        )

    assert len(requests) == 2


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
