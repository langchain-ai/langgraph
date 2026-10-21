from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest

from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._sync.http import SyncHttpClient


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "operation",
    ["get", "post", "put", "patch", "delete", "request_reconnect", "stream"],
)
@pytest.mark.parametrize(
    "statuses",
    [
        (200,),
        (504, 504, 200),
        (504, 504, 504, 504),
        (504, 400),
        (400,),
        (500,),
        (502,),
        (503,),
    ],
)
async def test_gateway_timeout_retries(monkeypatch, asynchronous, operation, statuses):
    requests: list[httpx.Request] = []
    responses: list[httpx.Response] = []
    sleep = AsyncMock() if asynchronous else Mock()
    monkeypatch.setattr(
        "langgraph_sdk._async.http.asyncio.sleep"
        if asynchronous
        else "langgraph_sdk._sync.http.time.sleep",
        sleep,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert all(response.is_closed for response in responses)
        response = httpx.Response(
            statuses[len(requests) - 1],
            headers={"content-type": "text/event-stream"}
            if operation == "stream"
            else {},
            stream=httpx.ByteStream(
                b'event: values\ndata: {"ok": true}\n\n'
                if operation == "stream" and statuses[len(requests) - 1] == 200
                else b'{"ok": true}'
            ),
        )
        responses.append(response)
        return response

    kwargs = {"params": {"q": "value"}, "headers": {"X-Test": "value"}}
    if operation not in ("get", "delete"):
        kwargs["json"] = {"input": "value"}
    if operation in ("request_reconnect", "stream"):
        kwargs["method"] = "POST"
    callback = Mock()
    kwargs["on_response"] = callback

    async def invoke():
        if asynchronous:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="https://example.com"
            ) as client:
                result = getattr(HttpClient(client), operation)("/test", **kwargs)
                return (
                    [part async for part in result]
                    if operation == "stream"
                    else await result
                )
        with httpx.Client(
            transport=httpx.MockTransport(handler), base_url="https://example.com"
        ) as client:
            result = getattr(SyncHttpClient(client), operation)("/test", **kwargs)
            return list(result) if operation == "stream" else result

    if statuses[-1] == 200:
        result = await invoke()
        if operation == "stream":
            assert result[0].data == {"ok": True}
        else:
            assert result == (None if operation == "delete" else {"ok": True})
    else:
        with pytest.raises(httpx.HTTPStatusError) as exc:
            await invoke()
        assert exc.value.response.status_code == statuses[-1]
    assert len(requests) == len(statuses)
    assert sleep.call_args_list == [call(0.5 * 2**i) for i in range(len(statuses) - 1)]
    callback.assert_called_once_with(responses[-1])
    assert all(request.url == requests[0].url for request in requests)
    assert all(request.headers == requests[0].headers for request in requests)
    assert all(request.content == requests[0].content for request in requests)
    assert all(request.method == requests[0].method for request in requests)
    assert requests[0].url.params["q"] == "value"
    assert requests[0].headers["X-Test"] == "value"
    if "json" in kwargs:
        assert requests[0].content == b'{"input":"value"}'
