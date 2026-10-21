import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from langgraph_sdk._async.client import get_client
from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._shared.retry import AsyncRetryTransport, RetryPolicy, RetryTransport
from langgraph_sdk._sync.client import get_sync_client
from langgraph_sdk._sync.http import SyncHttpClient
from langgraph_sdk.stream.transport.http import ProtocolSseTransport


async def _handle(asynchronous, transport, request):
    if asynchronous:
        return await transport.handle_async_request(request)
    return transport.handle_request(request)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "method", ["GET", "HEAD", "OPTIONS", "PUT", "DELETE", "POST", "PATCH", "TRACE"]
)
@pytest.mark.parametrize(
    "statuses",
    [
        (200,),
        (400,),
        (429,),
        (500,),
        (502, 200),
        (503, 200),
        (504, 200),
        (502, 502),
        (503, 503),
        (504, 504),
    ],
)
async def test_gateway_retry_policy(asynchronous, method, statuses):
    requests: list[httpx.Request] = []
    responses: list[httpx.Response] = []

    def wait(delay):
        assert responses[-1].is_closed
        assert 0.5 <= delay <= 1.5

    sleep = AsyncMock(side_effect=wait) if asynchronous else Mock(side_effect=wait)

    def handler(request):
        requests.append(request)
        response = httpx.Response(
            statuses[len(requests) - 1], stream=httpx.ByteStream(b'{"ok": true}')
        )
        responses.append(response)
        return response

    inner = httpx.MockTransport(handler)
    transport = (
        AsyncRetryTransport(inner, sleep=sleep)
        if asynchronous
        else RetryTransport(inner, sleep=sleep)
    )
    request = httpx.Request(
        method,
        "https://example.com/test?q=value",
        headers={"X-Test": "value"},
        content=b'{"input":"value"}',
        extensions={"timeout": {"connect": 5, "read": 60, "write": 60, "pool": 5}},
    )
    expected_headers = request.headers.copy()
    response = await _handle(asynchronous, transport, request)
    retry = method in {"GET", "HEAD", "OPTIONS", "PUT", "DELETE"} and statuses[0] in {
        502,
        503,
        504,
    }
    assert len(requests) == (2 if retry else 1)
    assert sleep.call_count == int(retry)
    assert response is responses[-1]
    assert response.status_code == (statuses[-1] if retry else statuses[0])
    assert all(sent is request for sent in requests)
    assert request.url.params["q"] == "value"
    assert request.headers == expected_headers
    assert request.content == b'{"input":"value"}'
    assert request.extensions["timeout"]["read"] == 60
    assert not response.is_closed
    if isinstance(transport, AsyncRetryTransport):
        assert await response.aread() == b'{"ok": true}'
        await transport.aclose()
    else:
        assert response.read() == b'{"ok": true}'
        transport.close()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_one_shot_upload_not_replayed(asynchronous):
    async def chunks():
        yield b"payload"

    handler = Mock(return_value=httpx.Response(504))
    sleep = AsyncMock() if asynchronous else Mock()
    inner = httpx.MockTransport(handler)
    transport = (
        AsyncRetryTransport(inner, sleep=sleep)
        if asynchronous
        else RetryTransport(inner, sleep=sleep)
    )
    request = httpx.Request(
        "PUT",
        "https://example.com/test",
        content=chunks() if asynchronous else iter([b"payload"]),
    )
    response = await _handle(asynchronous, transport, request)
    assert response.status_code == 504
    handler.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_transport_errors_not_retried(asynchronous):
    handler = Mock(side_effect=httpx.ReadTimeout("read timed out"))
    sleep = AsyncMock() if asynchronous else Mock()
    inner = httpx.MockTransport(handler)
    transport = (
        AsyncRetryTransport(inner, sleep=sleep)
        if asynchronous
        else RetryTransport(inner, sleep=sleep)
    )
    with pytest.raises(httpx.ReadTimeout):
        await _handle(
            asynchronous, transport, httpx.Request("GET", "https://example.com/test")
        )
    handler.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_transport_close_delegates(asynchronous):
    if asynchronous:
        inner = Mock(spec=httpx.AsyncBaseTransport)
        inner.aclose = AsyncMock()
        await AsyncRetryTransport(inner).aclose()
        inner.aclose.assert_awaited_once()
    else:
        inner = Mock(spec=httpx.BaseTransport)
        RetryTransport(inner).close()
        inner.close.assert_called_once()


async def test_cancellation_during_backoff_closes_failed_response():
    response = httpx.Response(504, stream=httpx.ByteStream(b"timeout"))
    handler = Mock(return_value=response)
    sleep = AsyncMock(side_effect=asyncio.CancelledError)
    transport = AsyncRetryTransport(httpx.MockTransport(handler), sleep=sleep)
    with pytest.raises(asyncio.CancelledError):
        await transport.handle_async_request(
            httpx.Request("GET", "https://example.com/test")
        )
    assert response.is_closed
    handler.assert_called_once()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_network_factories_install_retry_transport(asynchronous):
    if asynchronous:
        async with get_client(url="https://example.com", api_key=None) as client:
            assert isinstance(client.http.client._transport, AsyncRetryTransport)
            assert isinstance(
                client.http.client._transport._transport, httpx.AsyncHTTPTransport
            )
    else:
        with get_sync_client(url="https://example.com", api_key=None) as client:
            assert isinstance(client.http.client._transport, RetryTransport)
            assert isinstance(
                client.http.client._transport._transport, httpx.HTTPTransport
            )


async def test_in_process_factory_does_not_install_retries(monkeypatch):
    monkeypatch.setenv("__LANGGRAPH_DEFER_LOOPBACK_TRANSPORT", "true")
    monkeypatch.setattr("langgraph_sdk._async.client._registered_transports", [])
    monkeypatch.setattr(
        "langgraph_sdk._async.client.get_asgi_transport", lambda: httpx.ASGITransport
    )
    async with get_client(api_key=None) as client:
        assert isinstance(client.http.client._transport, httpx.ASGITransport)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_custom_http_clients_do_not_retry(asynchronous):
    handler = Mock(return_value=httpx.Response(504, json={"error": "timeout"}))
    if asynchronous:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://example.com"
        ) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await HttpClient(client).get("/test")
    else:
        with (
            httpx.Client(
                transport=httpx.MockTransport(handler), base_url="https://example.com"
            ) as client,
            pytest.raises(httpx.HTTPStatusError),
        ):
            SyncHttpClient(client).get("/test")
    handler.assert_called_once()


class BrokenByteStream(httpx.ByteStream):
    def __iter__(self):
        yield b"partial response"
        raise httpx.ReadError("disconnected")

    async def __aiter__(self):
        for chunk in self:
            yield chunk


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_started_response_is_not_replayed(asynchronous):
    handler = Mock(return_value=httpx.Response(200, stream=BrokenByteStream(b"")))
    sleep = AsyncMock() if asynchronous else Mock()
    inner = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(
            transport=AsyncRetryTransport(inner, sleep=sleep)
        ) as client:
            with pytest.raises(httpx.ReadError):
                await client.get("https://example.com/test")
    else:
        with (
            httpx.Client(transport=RetryTransport(inner, sleep=sleep)) as client,
            pytest.raises(httpx.ReadError),
        ):
            client.get("https://example.com/test")
    handler.assert_called_once()
    sleep.assert_not_called()


async def test_v3_command_is_not_retried():
    handler = Mock(return_value=httpx.Response(504))
    sleep = AsyncMock()
    transport = AsyncRetryTransport(httpx.MockTransport(handler), sleep=sleep)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        protocol = ProtocolSseTransport(client=client, thread_id="thread")
        with pytest.raises(httpx.HTTPStatusError):
            await protocol.send_command({"id": 1, "method": "run.start", "params": {}})
    handler.assert_called_once()
    sleep.assert_not_called()


def test_retry_policy_is_shared():
    assert (
        RetryTransport(httpx.MockTransport(Mock()))._policy
        == AsyncRetryTransport(httpx.MockTransport(Mock()))._policy
        == RetryPolicy()
    )
