from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from langgraph_sdk._async.client import get_client
from langgraph_sdk._shared.retry import AsyncRetryTransport, RetryTransport
from langgraph_sdk._sync.client import get_sync_client

IDEMPOTENT_METHODS = ("GET", "HEAD", "OPTIONS", "PUT", "DELETE")


def _retrying(asynchronous, handler, sleep):
    inner = httpx.MockTransport(handler)
    if asynchronous:
        return AsyncRetryTransport(inner, sleep=sleep)
    return RetryTransport(inner, sleep=sleep)


async def _send(transport, request):
    if isinstance(transport, AsyncRetryTransport):
        return await transport.handle_async_request(request)
    return transport.handle_request(request)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("method", [*IDEMPOTENT_METHODS, "POST", "PATCH"])
@pytest.mark.parametrize(
    "statuses",
    [(200,), (429,), (500,), (502, 200), (503, 200), (504, 200), (504, 504)],
)
async def test_gateway_failures_are_resent_once(asynchronous, method, statuses):
    requests: list[httpx.Request] = []
    responses: list[httpx.Response] = []

    def handler(request):
        requests.append(request)
        responses.append(
            httpx.Response(
                statuses[len(requests) - 1], stream=httpx.ByteStream(b'{"ok": true}')
            )
        )
        return responses[-1]

    def wait(delay):
        assert responses[-1].is_closed, "the failed response is released first"
        assert 1.0 <= delay <= 1.25

    sleep = AsyncMock(side_effect=wait) if asynchronous else Mock(side_effect=wait)
    transport = _retrying(asynchronous, handler, sleep)
    request = httpx.Request(
        method,
        "https://example.com/test?q=value",
        headers={"X-Test": "value"},
        content=b'{"input":"value"}',
        extensions={"timeout": {"connect": 5, "read": 60, "write": 60, "pool": 5}},
    )
    response = await _send(transport, request)

    resent = method in IDEMPOTENT_METHODS and statuses[0] in (502, 503, 504)
    assert len(requests) == (2 if resent else 1)
    assert sleep.call_count == (1 if resent else 0)
    assert all(sent is request for sent in requests)
    assert request.content == b'{"input":"value"}'
    assert request.extensions["timeout"]["read"] == 60
    assert response is responses[-1]
    assert response.status_code == (statuses[1] if resent else statuses[0])
    if asynchronous:
        assert await response.aread() == b'{"ok": true}'
    else:
        assert response.read() == b'{"ok": true}'


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_streamed_upload_is_not_resent(asynchronous):
    async def chunks():
        yield b"payload"

    handler = Mock(return_value=httpx.Response(504))
    sleep = AsyncMock() if asynchronous else Mock()
    content = chunks() if asynchronous else iter([b"payload"])
    request = httpx.Request("PUT", "https://example.com/test", content=content)

    response = await _send(_retrying(asynchronous, handler, sleep), request)

    assert response.status_code == 504
    handler.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_transport_errors_are_not_resent(asynchronous):
    handler = Mock(side_effect=httpx.ReadTimeout("read timed out"))
    sleep = AsyncMock() if asynchronous else Mock()
    transport = _retrying(asynchronous, handler, sleep)

    with pytest.raises(httpx.ReadTimeout):
        await _send(transport, httpx.Request("GET", "https://example.com/test"))

    handler.assert_called_once()
    sleep.assert_not_called()


class BrokenByteStream(httpx.ByteStream):
    def __iter__(self):
        yield b"partial response"
        raise httpx.ReadError("disconnected")

    async def __aiter__(self):
        for chunk in self:
            yield chunk


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_started_response_is_not_resent(asynchronous):
    handler = Mock(return_value=httpx.Response(200, stream=BrokenByteStream(b"")))
    sleep = AsyncMock() if asynchronous else Mock()
    transport = _retrying(asynchronous, handler, sleep)

    with pytest.raises(httpx.ReadError):
        if asynchronous:
            async with httpx.AsyncClient(transport=transport) as client:
                await client.get("https://example.com/test")
        else:
            with httpx.Client(transport=transport) as client:
                client.get("https://example.com/test")

    handler.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_close_reaches_the_wrapped_transport(asynchronous):
    if asynchronous:
        inner = Mock(spec=httpx.AsyncBaseTransport, aclose=AsyncMock())
        await AsyncRetryTransport(inner).aclose()
        inner.aclose.assert_awaited_once()
    else:
        inner = Mock(spec=httpx.BaseTransport)
        RetryTransport(inner).close()
        inner.close.assert_called_once()


async def test_remote_clients_use_the_retry_transport():
    async with get_client(url="https://example.com", api_key=None) as client:
        transport = client.http.client._transport
        assert isinstance(transport, AsyncRetryTransport)
        assert isinstance(transport._transport, httpx.AsyncHTTPTransport)
    with get_sync_client(url="https://example.com", api_key=None) as sync_client:
        transport = sync_client.http.client._transport
        assert isinstance(transport, RetryTransport)
        assert isinstance(transport._transport, httpx.HTTPTransport)


async def test_in_process_client_is_not_wrapped(monkeypatch):
    monkeypatch.setenv("__LANGGRAPH_DEFER_LOOPBACK_TRANSPORT", "true")
    monkeypatch.setattr("langgraph_sdk._async.client._registered_transports", [])
    monkeypatch.setattr(
        "langgraph_sdk._async.client.get_asgi_transport", lambda: httpx.ASGITransport
    )
    async with get_client(api_key=None) as client:
        assert isinstance(client.http.client._transport, httpx.ASGITransport)
