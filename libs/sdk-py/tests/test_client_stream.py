from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import orjson
import pytest
from typing_extensions import assert_type

from langgraph_sdk._async.runs import _wrap_stream_v2
from langgraph_sdk._shared.utilities import _sse_to_v2_dict
from langgraph_sdk._sync.runs import _wrap_stream_v2_sync
from langgraph_sdk.client import HttpClient, SyncHttpClient
from langgraph_sdk.errors import InternalServerError, NotFoundError
from langgraph_sdk.schema import (
    CheckpointPayload,
    CheckpointsStreamPart,
    CustomStreamPart,
    DebugPayload,
    DebugStreamPart,
    MetadataStreamPart,
    RunMetadataPayload,
    StreamPart,
    StreamPartV2,
    TaskPayload,
    TaskResultPayload,
    TasksStreamPart,
    UpdatesStreamPart,
    ValuesStreamPart,
)
from langgraph_sdk.sse import BytesLike, BytesLineDecoder, SSEDecoder

with open(Path(__file__).parent / "fixtures" / "response.txt", "rb") as f:
    RESPONSE_PAYLOAD = f.read()


# --- test helpers ---


class AsyncListByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: Sequence[bytes], exc: Exception | None = None) -> None:
        self._chunks = list(chunks)
        self._exc = exc

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        if self._exc is not None:
            raise self._exc

    async def aclose(self) -> None:
        return None


class ListByteStream(httpx.ByteStream):
    def __init__(self, chunks: Sequence[bytes], exc: Exception | None = None) -> None:
        self._chunks = list(chunks)
        self._exc = exc

    def __iter__(self):
        yield from self._chunks
        if self._exc is not None:
            raise self._exc

    def close(self) -> None:
        return None


def iter_lines_raw(payload: list[bytes]) -> Iterator[BytesLike]:
    decoder = BytesLineDecoder()
    for part in payload:
        yield from decoder.decode(part)
    yield from decoder.flush()


_V2_REQUIRED_KEYS = {"type", "ns", "data"}


def _assert_v2_shape(part: Any) -> None:
    """Assert a v2 stream part has the required keys and types."""
    assert isinstance(part, dict), f"Expected dict, got {type(part)}"
    assert part.keys() >= _V2_REQUIRED_KEYS, (
        f"Missing keys: {_V2_REQUIRED_KEYS - part.keys()}"
    )
    assert isinstance(part["type"], str)
    assert isinstance(part["ns"], list)
    for elem in part["ns"]:
        assert isinstance(elem, str)


# --- SSE parsing ---


def test_stream_sse():
    for groups in (
        [RESPONSE_PAYLOAD],
        RESPONSE_PAYLOAD.splitlines(keepends=True),
    ):
        parts: list[StreamPart] = []

        decoder = SSEDecoder()
        for line in iter_lines_raw(groups):
            sse = decoder.decode(line=line.rstrip(b"\n"))  # type: ignore
            if sse is not None:
                parts.append(sse)
        if sse := decoder.decode(b""):
            parts.append(sse)

        assert decoder.decode(b"") is None
        assert len(parts) == 79


def test_sse_decoder_joins_data_lines_with_a_newline():
    decoder = SSEDecoder()
    for line in (b"data: [1,", b"data: 2]"):
        decoder.decode(line)
    assert decoder.decode(b"") == StreamPart(event="", data=[1, 2], id=None)  # ty: ignore[invalid-argument-type]

    for line in (b"data: [1", b"data: 0]"):
        decoder.decode(line)
    with pytest.raises(orjson.JSONDecodeError):
        decoder.decode(b"")


def test_sse_decoder_strips_a_leading_bom():
    decoder = SSEDecoder()
    for line in (b"\xef\xbb\xbfevent: values", b"data: 1"):
        decoder.decode(line)
    assert decoder.decode(b"") == StreamPart(event="values", data=1, id=None)  # ty: ignore[invalid-argument-type]


def test_control_blocks_do_not_repeat_after_a_cursor_update():
    decoder = SSEDecoder()
    decoder.decode(b"id: cursor")
    assert decoder.decode(b"") == StreamPart(event="", data=None, id="cursor")  # ty: ignore[invalid-argument-type]
    decoder.decode(b": heartbeat")
    assert decoder.decode(b"") is None
    assert decoder.decode(b"") is None
    decoder.decode(b"id:")
    assert decoder.decode(b"") == StreamPart(event="", data=None, id=None)  # ty: ignore[invalid-argument-type]
    assert decoder.last_event_id == ""
    assert decoder.decode(b"") is None


def test_cr_lines_are_emitted_without_waiting_for_another_chunk():
    decoder = BytesLineDecoder()
    assert decoder.decode(b"data: 1\r\r") == [b"data: 1", b""]
    assert decoder.decode(b"\n") == []
    assert decoder.decode(b"data: 2\r") == [b"data: 2"]
    assert decoder.decode(b"") == []
    assert decoder.decode(b"\n\r") == [b""]
    assert decoder.flush() == []


def test_empty_data_lines_preserve_their_newlines():
    decoder = SSEDecoder()
    for line in (b"data", b"data: 1", b"data:"):
        decoder.decode(line)
    assert decoder._data == b"\n1\n\n"
    assert decoder.decode(b"") == StreamPart(event="", data=1, id=None)  # ty: ignore[invalid-argument-type]
    decoder.decode(b"data:")
    decoder.decode(b"data:")
    assert decoder.decode(b"") == StreamPart(event="", data=None, id=None)  # ty: ignore[invalid-argument-type]
    assert decoder.has_data
    assert decoder.decode(b"") is None
    assert not decoder.has_data


def test_malformed_utf8_is_replaced_in_fields_and_json_strings():
    decoder = SSEDecoder()
    for line in (b"event: \xff", b"id: \xff", b'data: "\xff"'):
        decoder.decode(line)
    assert decoder.decode(b"") == StreamPart(event="\ufffd", data="\ufffd", id="\ufffd")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_null_and_empty_data_are_not_dropped(asynchronous: bool):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=b"data: null\n\ndata:\n\n",
        )

    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            parts = [part async for part in HttpClient(client).stream("/stream", "GET")]
    else:
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            parts = list(SyncHttpClient(client).stream("/stream", "GET"))
    assert parts == [StreamPart(event="", data=None), StreamPart(event="", data=None)]  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_cursor_is_inherited_after_reconnect(asynchronous: bool):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            stream_cls = AsyncListByteStream if asynchronous else ListByteStream
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream", "Location": "/reconnect"},
                stream=stream_cls(
                    [b"id: cursor\nevent: values\ndata: 1\n\n"],
                    httpx.RemoteProtocolError("incomplete chunked read"),
                ),
            )
        assert (b"Last-Event-ID", b"cursor") in request.headers.raw
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=b"event: values\ndata: 2\n\n",
        )

    transport = httpx.MockTransport(handler)
    if asynchronous:
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.com"
        ) as client:
            parts = [part async for part in HttpClient(client).stream("/stream", "GET")]
    else:
        with httpx.Client(
            transport=transport, base_url="https://example.com"
        ) as client:
            parts = list(SyncHttpClient(client).stream("/stream", "GET"))
    assert [part.id for part in parts] == ["cursor", "cursor"]


# --- HTTP client streaming ---


@pytest.mark.asyncio
async def test_http_client_stream_flushes_trailing_event():
    payload = b'event: foo\ndata: {"bar": 1}\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "text/event-stream"
        assert request.headers["cache-control"] == "no-store"
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=payload,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        parts = [part async for part in http_client.stream("/stream", "GET")]

    assert parts == [StreamPart(event="foo", data={"bar": 1})]


def test_sync_http_client_stream_flushes_trailing_event():
    payload = b'event: foo\ndata: {"bar": 1}\n'

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "text/event-stream"
        assert request.headers["cache-control"] == "no-store"
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=payload,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        parts = list(http_client.stream("/stream", "GET"))

    assert parts == [StreamPart(event="foo", data={"bar": 1})]


@pytest.mark.usefixtures("no_sleep")
def test_sync_http_client_stream_recovers_after_disconnect():
    reconnect_path = "/reconnect"
    first_chunks = [
        b"id: 1\n",
        b"event: values\n",
        b'data: {"step": 1}\n\n',
    ]
    second_chunks = [
        b"id: 2\n",
        b"event: values\n",
        b'data: {"step": 2}\n\n',
        b"event: end\n",
        b"data: null\n\n",
    ]
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            assert request.method == "POST"
            assert request.url.path == "/stream"
            assert request.headers["accept"] == "text/event-stream"
            assert request.headers["cache-control"] == "no-store"
            assert "last-event-id" not in {
                k.lower(): v for k, v in request.headers.items()
            }
            assert request.read()
            return httpx.Response(
                200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Location": reconnect_path,
                },
                stream=ListByteStream(
                    first_chunks,
                    httpx.RemoteProtocolError("incomplete chunked read"),
                ),
            )
        if call_count == 2:
            assert request.method == "GET"
            assert request.url.path == reconnect_path
            assert request.headers["Last-Event-ID"] == "1"
            assert request.read() == b""
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                stream=ListByteStream(second_chunks),
            )
        raise AssertionError("unexpected request")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        http_client = SyncHttpClient(client)
        parts = list(http_client.stream("/stream", "POST", json={"payload": "value"}))

    assert call_count == 2
    assert parts == [
        StreamPart(event="values", data={"step": 1}, id="1"),
        StreamPart(event="values", data={"step": 2}, id="2"),
        StreamPart(event="end", data=None, id="2"),  # ty: ignore[invalid-argument-type]
    ]


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_sleep")
async def test_http_client_stream_recovers_after_disconnect():
    reconnect_path = "/reconnect"
    first_chunks = [
        b"id: 1\n",
        b"event: values\n",
        b'data: {"step": 1}\n\n',
    ]
    second_chunks = [
        b"id: 2\n",
        b"event: values\n",
        b'data: {"step": 2}\n\n',
        b"event: end\n",
        b"data: null\n\n",
    ]
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            assert request.method == "POST"
            assert request.url.path == "/stream"
            assert request.headers["accept"] == "text/event-stream"
            assert request.headers["cache-control"] == "no-store"
            assert "last-event-id" not in {
                k.lower(): v for k, v in request.headers.items()
            }
            assert await request.aread()
            return httpx.Response(
                200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Location": reconnect_path,
                },
                stream=AsyncListByteStream(
                    first_chunks,
                    httpx.RemoteProtocolError("incomplete chunked read"),
                ),
            )
        if call_count == 2:
            assert request.method == "GET"
            assert request.url.path == reconnect_path
            assert request.headers["Last-Event-ID"] == "1"
            assert await request.aread() == b""
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                stream=AsyncListByteStream(second_chunks),
            )
        raise AssertionError("unexpected request")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        http_client = HttpClient(client)
        parts = [
            part
            async for part in http_client.stream(
                "/stream", "POST", json={"payload": "value"}
            )
        ]

    assert call_count == 2
    assert parts == [
        StreamPart(event="values", data={"step": 1}, id="1"),
        StreamPart(event="values", data={"step": 2}, id="2"),
        StreamPart(event="end", data=None, id="2"),  # ty: ignore[invalid-argument-type]
    ]


def _drop_after(chunks: list[bytes], stream_cls: type, reconnects: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream", "Location": "/reconnect"},
                stream=stream_cls(
                    chunks, httpx.RemoteProtocolError("incomplete chunked read")
                ),
            )
        reconnects.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=stream_cls([b"event: end\ndata: null\n\n"]),
        )

    return handler


def _sync_reconnect(
    chunks: list[bytes], headers: dict[str, str] | None
) -> httpx.Request:
    reconnects: list[httpx.Request] = []
    transport = httpx.MockTransport(_drop_after(chunks, ListByteStream, reconnects))
    with httpx.Client(transport=transport, base_url="https://example.com") as client:
        list(SyncHttpClient(client).stream("/stream", "POST", json={}, headers=headers))
    return reconnects[0]


async def _async_reconnect(
    chunks: list[bytes], headers: dict[str, str] | None
) -> httpx.Request:
    reconnects: list[httpx.Request] = []
    transport = httpx.MockTransport(
        _drop_after(chunks, AsyncListByteStream, reconnects)
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="https://example.com"
    ) as client:
        [
            part
            async for part in HttpClient(client).stream(
                "/stream", "POST", json={}, headers=headers
            )
        ]
    return reconnects[0]


_EMPTY_IDS = pytest.mark.parametrize(
    "empty_id",
    [b"id:\nevent: values\ndata: 2\n\n", b"id:\n\n"],
    ids=["with-an-event", "alone"],
)
_CALLER_CURSORS = pytest.mark.parametrize(
    "headers",
    [None, {"Last-Event-ID": "5"}, {"last-event-id": "5"}],
    ids=["no-caller-cursor", "caller-cursor", "lowercase-caller-cursor"],
)


@_EMPTY_IDS
@_CALLER_CURSORS
@pytest.mark.usefixtures("no_sleep")
def test_sync_stream_reconnects_without_a_cursor_after_an_empty_id(
    empty_id: bytes, headers: dict[str, str] | None
):
    with pytest.warns(UserWarning, match="Last-Event-ID"):
        reconnect = _sync_reconnect(
            [b"id: 1\nevent: values\ndata: 1\n\n", empty_id], headers
        )

    assert "last-event-id" not in reconnect.headers


@_EMPTY_IDS
@_CALLER_CURSORS
@pytest.mark.usefixtures("no_sleep")
async def test_async_stream_reconnects_without_a_cursor_after_an_empty_id(
    empty_id: bytes, headers: dict[str, str] | None
):
    with pytest.warns(UserWarning, match="Last-Event-ID"):
        reconnect = await _async_reconnect(
            [b"id: 1\nevent: values\ndata: 1\n\n", empty_id], headers
        )

    assert "last-event-id" not in reconnect.headers


@pytest.mark.usefixtures("no_sleep")
def test_sync_stream_id_replaces_a_lowercase_caller_cursor():
    reconnect = _sync_reconnect(
        [b"id: 1\nevent: values\ndata: 1\n\n"], {"last-event-id": "5"}
    )

    assert reconnect.headers.get_list("last-event-id") == ["1"]


@pytest.mark.usefixtures("no_sleep")
async def test_async_stream_id_replaces_a_lowercase_caller_cursor():
    reconnect = await _async_reconnect(
        [b"id: 1\nevent: values\ndata: 1\n\n"], {"last-event-id": "5"}
    )

    assert reconnect.headers.get_list("last-event-id") == ["1"]


@pytest.mark.usefixtures("no_sleep")
def test_sync_stream_keeps_the_caller_cursor_until_the_stream_sends_an_id():
    reconnect = _sync_reconnect([b"event: values\ndata: 1\n\n"], {"Last-Event-ID": "5"})

    assert reconnect.headers.get_list("last-event-id") == ["5"]


@pytest.mark.usefixtures("no_sleep")
async def test_async_stream_keeps_the_caller_cursor_until_the_stream_sends_an_id():
    reconnect = await _async_reconnect(
        [b"event: values\ndata: 1\n\n"], {"Last-Event-ID": "5"}
    )

    assert reconnect.headers.get_list("last-event-id") == ["5"]


# A reconnect that sends no Last-Event-ID can't replay what was sent while it was
# disconnected, so it warns.
_RECONNECT_CURSORS = pytest.mark.parametrize(
    ("chunks", "headers", "warns"),
    [
        ([b"event: values\ndata: 1\n\n"], None, True),
        ([b"id: 1\nevent: values\ndata: 1\n\n"], None, False),
        ([b"event: values\ndata: 1\n\n"], {"Last-Event-ID": "5"}, False),
    ],
    ids=["no-cursor", "stream-cursor", "caller-cursor"],
)


@_RECONNECT_CURSORS
@pytest.mark.usefixtures("no_sleep")
def test_sync_reconnect_warns_only_without_a_cursor(
    chunks: list[bytes],
    headers: dict[str, str] | None,
    warns: bool,
    recwarn: pytest.WarningsRecorder,
):
    _sync_reconnect(chunks, headers)

    assert any("Last-Event-ID" in str(w.message) for w in recwarn) is warns


@_RECONNECT_CURSORS
@pytest.mark.usefixtures("no_sleep")
async def test_async_reconnect_warns_only_without_a_cursor(
    chunks: list[bytes],
    headers: dict[str, str] | None,
    warns: bool,
    recwarn: pytest.WarningsRecorder,
):
    await _async_reconnect(chunks, headers)

    assert any("Last-Event-ID" in str(w.message) for w in recwarn) is warns


# --- reconnects survive a server restart or a deploy ---

SSE = {"Content-Type": "text/event-stream"}
DROP = httpx.RemoteProtocolError("incomplete chunked read")


def _event(i: int) -> bytes:
    return f"id: {i}\nevent: values\ndata: {i}\n\n".encode()


def _scripted(steps: list[Callable[[httpx.Request], httpx.Response]]):
    """Handler that plays one step per request and records each request."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) > len(steps):
            raise AssertionError("unexpected request")
        return steps[len(seen) - 1](request)

    return handler, seen


def _ok(stream_cls, chunks, *, drop=False, location=None):
    headers = dict(SSE)
    if location:
        headers["Location"] = location

    def step(_request):
        return httpx.Response(
            200, headers=headers, stream=stream_cls(chunks, DROP if drop else None)
        )

    return step


def _status(code):
    return lambda request: httpx.Response(
        code, json={"detail": "upstream"}, request=request
    )


def _connect_error(request):
    raise httpx.ConnectError("[Errno 61] Connection refused", request=request)


def _not_sse(request):
    return httpx.Response(
        200, headers={"Content-Type": "text/html"}, text="<html>", request=request
    )


def _transient_steps(stream_cls, failure):
    """POST drops after event 1; first reconnect fails; second succeeds."""
    return [
        _ok(stream_cls, [_event(1)], drop=True, location="/reconnect"),
        failure,
        _ok(stream_cls, [_event(2), b"event: end\ndata: null\n\n"]),
    ]


def _long_stream_steps(stream_cls, drops):
    """POST + `drops` reconnects, each delivering one event then dropping."""
    steps = [_ok(stream_cls, [_event(0)], drop=True, location="/reconnect")]
    for i in range(1, drops):
        steps.append(_ok(stream_cls, [_event(i)], drop=True))
    steps.append(_ok(stream_cls, [_event(drops), b"event: end\ndata: null\n\n"]))
    return steps


def _run_sync(steps, headers=None):
    handler, seen = _scripted(steps)
    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://example.com"
    ) as client:
        parts = list(
            SyncHttpClient(client).stream("/stream", "POST", json={}, headers=headers)
        )
    return parts, seen


async def _run_async(steps, headers=None):
    sync_handler, seen = _scripted(steps)

    async def handler(request):
        return sync_handler(request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.com"
    ) as client:
        parts = [
            p
            async for p in HttpClient(client).stream(
                "/stream", "POST", json={}, headers=headers
            )
        ]
    return parts, seen


FAILURES = {
    "429": _status(429),
    "502": _status(502),
    "503": _status(503),
    "connect_error": _connect_error,
    "not_sse": _not_sse,
}


# A reconnect that hits a 429 / 502 / 503 / connect error / non-SSE body, then
# succeeds, should recover.


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("failure", FAILURES.values(), ids=FAILURES.keys())
def test_sync_reconnect_recovers_from_transient_failure(failure):
    parts, seen = _run_sync(_transient_steps(ListByteStream, failure))
    assert [p.event for p in parts] == ["values", "values", "end"]
    assert len(seen) == 3


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("failure", FAILURES.values(), ids=FAILURES.keys())
async def test_async_reconnect_recovers_from_transient_failure(failure):
    parts, seen = await _run_async(_transient_steps(AsyncListByteStream, failure))
    assert [p.event for p in parts] == ["values", "values", "end"]
    assert len(seen) == 3


# Any other 4xx on a reconnect won't go away by retrying, so it raises.


def _client_error_steps(stream_cls):
    return [
        _ok(stream_cls, [_event(1)], drop=True, location="/reconnect"),
        _status(404),
    ]


@pytest.mark.usefixtures("no_sleep")
def test_sync_reconnect_does_not_retry_a_client_error():
    with pytest.raises(NotFoundError):
        _run_sync(_client_error_steps(ListByteStream))


@pytest.mark.usefixtures("no_sleep")
async def test_async_reconnect_does_not_retry_a_client_error():
    with pytest.raises(NotFoundError):
        await _run_async(_client_error_steps(AsyncListByteStream))


# The first request isn't a reconnect, so it raises even when it names a Location.


def _first_request_503(request):
    return httpx.Response(
        503,
        headers={"Location": "/reconnect"},
        json={"detail": "upstream"},
        request=request,
    )


@pytest.mark.usefixtures("no_sleep")
def test_sync_first_request_failure_is_not_retried():
    with pytest.raises(InternalServerError):
        _run_sync([_first_request_503])


@pytest.mark.usefixtures("no_sleep")
async def test_async_first_request_failure_is_not_retried():
    with pytest.raises(InternalServerError):
        await _run_async([_first_request_503])


# A server that stays down gets five reconnects, backing off 1, 2, 4, 5 and 5s
# (each plus up to 25% jitter), so about 20s, before the stream gives up. The
# error it gives up with is caused by the last failure, so callers see why.

GIVE_UP = "Exceeded maximum SSE reconnection attempts"


def _server_down_steps(stream_cls):
    return [
        _ok(stream_cls, [_event(1)], drop=True, location="/reconnect"),
        *[_status(502)] * 5,
    ]


def _assert_backed_off_for_about_20s(sleeps: list[float]) -> None:
    assert len(sleeps) == 5
    for slept, delay in zip(sleeps, [1, 2, 4, 5, 5], strict=True):
        assert delay <= slept <= delay * 1.25


def test_sync_reconnect_gives_up_after_five_backed_off_attempts(no_sleep):
    with pytest.raises(httpx.TransportError, match=GIVE_UP) as exc_info:
        _run_sync(_server_down_steps(ListByteStream))
    assert isinstance(exc_info.value.__cause__, InternalServerError)
    _assert_backed_off_for_about_20s(no_sleep)


async def test_async_reconnect_gives_up_after_five_backed_off_attempts(no_sleep):
    with pytest.raises(httpx.TransportError, match=GIVE_UP) as exc_info:
        await _run_async(_server_down_steps(AsyncListByteStream))
    assert isinstance(exc_info.value.__cause__, InternalServerError)
    _assert_backed_off_for_about_20s(no_sleep)


# A long stream that drops 6 times, each reconnect succeeding and making
# progress, should not run out of attempts.


@pytest.mark.usefixtures("no_sleep")
def test_sync_long_stream_survives_many_spaced_out_drops():
    parts, _ = _run_sync(_long_stream_steps(ListByteStream, drops=6))
    assert [p.data for p in parts] == [0, 1, 2, 3, 4, 5, 6, None]


@pytest.mark.usefixtures("no_sleep")
async def test_async_long_stream_survives_many_spaced_out_drops():
    parts, _ = await _run_async(_long_stream_steps(AsyncListByteStream, drops=6))
    assert [p.data for p in parts] == [0, 1, 2, 3, 4, 5, 6, None]


# Each reconnect resumes after the newest event the stream got, so a reconnect
# loop on a resumable run replays only what it missed. A failed reconnect keeps
# the cursor for the next one.
RECONNECT_LOOPS = {
    "progress_between_drops": (
        lambda stream_cls: _long_stream_steps(stream_cls, drops=3),
        [None, "0", "1", "2"],
    ),
    "failed_reconnect": (
        lambda stream_cls: _transient_steps(stream_cls, _status(502)),
        [None, "1", "1"],
    ),
}


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize(
    ("steps", "cursors"), RECONNECT_LOOPS.values(), ids=RECONNECT_LOOPS.keys()
)
def test_sync_reconnect_loop_resumes_after_the_newest_event(steps, cursors):
    _, seen = _run_sync(steps(ListByteStream))
    assert [r.headers.get("last-event-id") for r in seen] == cursors


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize(
    ("steps", "cursors"), RECONNECT_LOOPS.values(), ids=RECONNECT_LOOPS.keys()
)
async def test_async_reconnect_loop_resumes_after_the_newest_event(steps, cursors):
    _, seen = await _run_async(steps(AsyncListByteStream))
    assert [r.headers.get("last-event-id") for r in seen] == cursors


# Reconnects should back off rather than hammer the server.


def test_sync_reconnect_backs_off(no_sleep):
    _run_sync(_long_stream_steps(ListByteStream, drops=2))
    assert len(no_sleep) == 2 and all(d > 0 for d in no_sleep)


async def test_async_reconnect_backs_off(no_sleep):
    await _run_async(_long_stream_steps(AsyncListByteStream, drops=2))
    assert len(no_sleep) == 2 and all(d > 0 for d in no_sleep)


# --- a run stream that closes cleanly before the run ends ---

RUN_STREAM = "/threads/t1/runs/r1/stream"
RUN = "/threads/t1/runs/r1"
END = b"event: end\ndata: null\n\n"

Step = Callable[[httpx.Request], httpx.Response]


def _run_status(status: str) -> Step:
    return lambda request: httpx.Response(200, json={"status": status}, request=request)


def _clean_close_steps(stream_cls: type, closes: int, check: Step) -> list[Step]:
    steps = [_ok(stream_cls, [_event(0)], location=RUN_STREAM)]
    for i in range(1, closes + 1):
        steps += [check, _ok(stream_cls, [_event(i)])]
    return [*steps, _run_status("success")]


def _stalled_steps(stream_cls: type) -> list[Step]:
    return [
        _ok(stream_cls, [_event(1)], location=RUN_STREAM),
        *[_run_status("running"), _ok(stream_cls, [])] * 5,
        _run_status("running"),
    ]


def _requests(seen: list[httpx.Request]) -> list[tuple[str, str, str | None]]:
    return [(r.method, r.url.path, r.headers.get("last-event-id")) for r in seen]


RESUMED = [
    ("POST", "/stream", None),
    ("GET", RUN, None),
    ("GET", RUN_STREAM, "0"),
    ("GET", RUN, None),
]


CHECKS_THAT_RECONNECT = {
    "pending": _run_status("pending"),
    "running": _run_status("running"),
    "429": _status(429),
    "503": _status(503),
    "connect_error": _connect_error,
}


@pytest.mark.parametrize(
    "check", CHECKS_THAT_RECONNECT.values(), ids=CHECKS_THAT_RECONNECT.keys()
)
def test_sync_clean_close_reconnects_after_check(check: Step, no_sleep: list[float]):
    parts, seen = _run_sync(_clean_close_steps(ListByteStream, 1, check))
    assert [p.data for p in parts] == [0, 1]
    assert _requests(seen) == RESUMED
    assert len(no_sleep) == 1


@pytest.mark.parametrize(
    "check", CHECKS_THAT_RECONNECT.values(), ids=CHECKS_THAT_RECONNECT.keys()
)
async def test_async_clean_close_reconnects_after_check(
    check: Step, no_sleep: list[float]
):
    parts, seen = await _run_async(_clean_close_steps(AsyncListByteStream, 1, check))
    assert [p.data for p in parts] == [0, 1]
    assert _requests(seen) == RESUMED
    assert len(no_sleep) == 1


@pytest.mark.usefixtures("no_sleep")
def test_sync_clean_closes_with_progress_do_not_run_out_of_attempts():
    parts, _ = _run_sync(_clean_close_steps(ListByteStream, 6, _run_status("running")))
    assert [p.data for p in parts] == [0, 1, 2, 3, 4, 5, 6]


@pytest.mark.usefixtures("no_sleep")
async def test_async_clean_closes_with_progress_do_not_run_out_of_attempts():
    parts, _ = await _run_async(
        _clean_close_steps(AsyncListByteStream, 6, _run_status("running"))
    )
    assert [p.data for p in parts] == [0, 1, 2, 3, 4, 5, 6]


CHECKS_THAT_END = {
    "success": _run_status("success"),
    "error": _run_status("error"),
    "timeout": _run_status("timeout"),
    "interrupted": _run_status("interrupted"),
    "403": _status(403),
    "404": _status(404),
    "not_json": _not_sse,
    "json_without_status": lambda request: httpx.Response(
        200, json={}, request=request
    ),
}
CHECKED = [("POST", "/stream", None), ("GET", RUN, None)]


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("check", CHECKS_THAT_END.values(), ids=CHECKS_THAT_END.keys())
def test_sync_clean_close_ends_the_stream_after_check(check: Step):
    parts, seen = _run_sync(
        [_ok(ListByteStream, [_event(1)], location=RUN_STREAM), check]
    )
    assert [p.data for p in parts] == [1]
    assert _requests(seen) == CHECKED


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("check", CHECKS_THAT_END.values(), ids=CHECKS_THAT_END.keys())
async def test_async_clean_close_ends_the_stream_after_check(check: Step):
    parts, seen = await _run_async(
        [_ok(AsyncListByteStream, [_event(1)], location=RUN_STREAM), check]
    )
    assert [p.data for p in parts] == [1]
    assert _requests(seen) == CHECKED


NO_CHECK = {
    "end": lambda s: [_ok(s, [_event(1), END], location=RUN_STREAM)],
    "error": lambda s: [
        _ok(s, [_event(1), b"event: error\ndata: {}\n\n"], location=RUN_STREAM)
    ],
    "unterminated_end": lambda s: [
        _ok(s, [_event(1), b"event: end\ndata: null\n"], location=RUN_STREAM)
    ],
    "end_on_an_earlier_connection": lambda s: [
        _ok(s, [_event(1), END], drop=True, location=RUN_STREAM),
        _ok(s, []),
    ],
    "thread_stream": lambda s: [_ok(s, [_event(1)], location="/threads/t1/stream")],
    "threadless_run_location": lambda s: [
        _ok(s, [_event(1)], location="/runs/r1/stream")
    ],
}


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("steps", NO_CHECK.values(), ids=NO_CHECK.keys())
def test_sync_stream_ends_without_a_run_check(steps: Callable[[type], list[Step]]):
    _, seen = _run_sync(steps(ListByteStream))
    assert [r.url.path for r in seen if not r.url.path.endswith("/stream")] == []


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("steps", NO_CHECK.values(), ids=NO_CHECK.keys())
async def test_async_stream_ends_without_a_run_check(
    steps: Callable[[type], list[Step]],
):
    _, seen = await _run_async(steps(AsyncListByteStream))
    assert [r.url.path for r in seen if not r.url.path.endswith("/stream")] == []


def test_sync_clean_closes_without_progress_give_up_after_five_attempts(
    no_sleep: list[float],
):
    with pytest.raises(httpx.TransportError, match=GIVE_UP) as exc_info:
        _run_sync(_stalled_steps(ListByteStream))
    assert "still pending or running" in str(exc_info.value.__cause__)
    _assert_backed_off_for_about_20s(no_sleep)


async def test_async_clean_closes_without_progress_give_up_after_five_attempts(
    no_sleep: list[float],
):
    with pytest.raises(httpx.TransportError, match=GIVE_UP) as exc_info:
        await _run_async(_stalled_steps(AsyncListByteStream))
    assert "still pending or running" in str(exc_info.value.__cause__)
    _assert_backed_off_for_about_20s(no_sleep)


CHECK_URLS = {
    "path": ("/threads/t1/runs/r1/stream", "https://example.com/threads/t1/runs/r1"),
    "prefix_and_query": (
        "/api/threads/t1/runs/r1/stream?stream_mode=values",
        "https://example.com/api/threads/t1/runs/r1",
    ),
    "absolute": (
        "https://example.com/threads/t1/runs/r1/stream",
        "https://example.com/threads/t1/runs/r1",
    ),
}


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize(
    ("location", "url"), CHECK_URLS.values(), ids=CHECK_URLS.keys()
)
def test_sync_run_check_gets_the_run_behind_the_stream_location(
    location: str, url: str
):
    _, seen = _run_sync(
        [_ok(ListByteStream, [_event(1)], location=location), _run_status("success")]
    )
    assert str(seen[-1].url) == url


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize(
    ("location", "url"), CHECK_URLS.values(), ids=CHECK_URLS.keys()
)
async def test_async_run_check_gets_the_run_behind_the_stream_location(
    location: str, url: str
):
    _, seen = await _run_async(
        [
            _ok(AsyncListByteStream, [_event(1)], location=location),
            _run_status("success"),
        ]
    )
    assert str(seen[-1].url) == url


CALLER_HEADERS = {"Authorization": "Bearer token", "Last-Event-ID": "0"}
BODY_AND_CURSOR_HEADERS = {"content-type", "content-length", "last-event-id"}


def _assert_run_check_headers(seen: list[httpx.Request]) -> None:
    check = seen[-1]
    assert check.url.path == RUN
    assert check.headers["authorization"] == "Bearer token"
    assert check.headers["accept"] == "application/json"
    assert check.headers.keys() & BODY_AND_CURSOR_HEADERS == set()


@pytest.mark.usefixtures("no_sleep")
def test_sync_run_check_sends_caller_headers_and_accepts_json_without_body_or_cursor():
    _, seen = _run_sync(
        [_ok(ListByteStream, [_event(1)], location=RUN_STREAM), _run_status("success")],
        headers=CALLER_HEADERS,
    )
    _assert_run_check_headers(seen)


@pytest.mark.usefixtures("no_sleep")
async def test_async_run_check_sends_caller_headers_and_accepts_json_without_body_or_cursor():
    _, seen = await _run_async(
        [
            _ok(AsyncListByteStream, [_event(1)], location=RUN_STREAM),
            _run_status("success"),
        ],
        headers=CALLER_HEADERS,
    )
    _assert_run_check_headers(seen)


# --- _sse_to_v2_dict conversion ---


def test_sse_to_v2_dict_basic() -> None:
    result = _sse_to_v2_dict("values", {"messages": [{"role": "user"}]})
    assert result is not None
    _assert_v2_shape(result)
    assert result == {
        "type": "values",
        "ns": [],
        "data": {"messages": [{"role": "user"}]},
        "interrupts": [],
    }


def test_sse_to_v2_dict_with_namespace() -> None:
    result = _sse_to_v2_dict("updates|sub:abc", {"key": "val"})
    assert result is not None
    _assert_v2_shape(result)
    assert result == {
        "type": "updates",
        "ns": ["sub:abc"],
        "data": {"key": "val"},
        "interrupts": [],
    }


def test_sse_to_v2_dict_with_multiple_ns() -> None:
    result = _sse_to_v2_dict("custom|parent|child:123", "hello")
    assert result is not None
    _assert_v2_shape(result)
    assert result == {
        "type": "custom",
        "ns": ["parent", "child:123"],
        "data": "hello",
        "interrupts": [],
    }


def test_sse_to_v2_dict_end_event() -> None:
    assert _sse_to_v2_dict("end", None) is None


def test_sse_to_v2_dict_metadata_event() -> None:
    result = _sse_to_v2_dict("metadata", {"run_id": "abc-123"})
    assert result is not None
    _assert_v2_shape(result)
    assert result == {
        "type": "metadata",
        "ns": [],
        "data": {"run_id": "abc-123"},
        "interrupts": [],
    }


def test_sse_to_v2_dict_messages_partial() -> None:
    result = _sse_to_v2_dict("messages/partial", [{"type": "ai", "content": "hi"}])
    assert result is not None
    _assert_v2_shape(result)
    assert result == {
        "type": "messages/partial",
        "ns": [],
        "data": [{"type": "ai", "content": "hi"}],
        "interrupts": [],
    }


def test_sse_to_v2_dict_values_with_interrupts() -> None:
    data = {
        "messages": [{"role": "user"}],
        "__interrupt__": [{"value": "confirm?", "resumable": True}],
    }
    result = _sse_to_v2_dict("values", data)
    assert result is not None
    _assert_v2_shape(result)
    assert result == {
        "type": "values",
        "ns": [],
        "data": {"messages": [{"role": "user"}]},
        "interrupts": [{"value": "confirm?", "resumable": True}],
    }
    # __interrupt__ should be popped from data
    assert "__interrupt__" not in result["data"]


# --- client-side v2 stream wrapping ---


@pytest.mark.asyncio
async def test_async_stream_v2_client_side_conversion() -> None:

    async def mock_stream() -> Any:
        yield StreamPart(event="metadata", data={"run_id": "r1"})
        yield StreamPart(
            event="values", data={"messages": [{"role": "user", "content": "hi"}]}
        )
        yield StreamPart(event="updates|sub:abc", data={"node": {"out": 1}})
        yield StreamPart(event="end", data=None)  # ty: ignore[invalid-argument-type]

    parts: list[StreamPartV2] = [part async for part in _wrap_stream_v2(mock_stream())]
    assert len(parts) == 3
    for part in parts:
        _assert_v2_shape(part)
    assert parts[0] == {
        "type": "metadata",
        "ns": [],
        "data": {"run_id": "r1"},
        "interrupts": [],
    }
    assert parts[1] == {
        "type": "values",
        "ns": [],
        "data": {"messages": [{"role": "user", "content": "hi"}]},
        "interrupts": [],
    }
    assert parts[2] == {
        "type": "updates",
        "ns": ["sub:abc"],
        "data": {"node": {"out": 1}},
        "interrupts": [],
    }


def test_sync_stream_v2_client_side_conversion() -> None:

    def mock_stream() -> Any:
        yield StreamPart(event="metadata", data={"run_id": "r1"})
        yield StreamPart(event="values", data={"state": "full"})
        yield StreamPart(event="end", data=None)  # ty: ignore[invalid-argument-type]

    parts: list[StreamPartV2] = list(_wrap_stream_v2_sync(mock_stream()))
    assert len(parts) == 2
    for part in parts:
        _assert_v2_shape(part)
    assert parts[0] == {
        "type": "metadata",
        "ns": [],
        "data": {"run_id": "r1"},
        "interrupts": [],
    }
    assert parts[1] == {
        "type": "values",
        "ns": [],
        "data": {"state": "full"},
        "interrupts": [],
    }


# --- type narrowing compile-time checks ---


def _check_v2_type_narrowing(part: StreamPartV2) -> None:
    """Compile-time type narrowing checks."""
    if part["type"] == "values":
        assert_type(part, ValuesStreamPart)
        assert_type(part["data"], dict[str, Any])
    elif part["type"] == "updates":
        assert_type(part, UpdatesStreamPart)
        assert_type(part["data"], dict[str, Any])
    elif part["type"] == "custom":
        assert_type(part, CustomStreamPart)
    elif part["type"] == "checkpoints":
        assert_type(part, CheckpointsStreamPart)
        assert_type(part["data"], CheckpointPayload)
    elif part["type"] == "tasks":
        assert_type(part, TasksStreamPart)
        assert_type(part["data"], TaskPayload | TaskResultPayload)
    elif part["type"] == "debug":
        assert_type(part, DebugStreamPart)
        assert_type(part["data"], DebugPayload)
    elif part["type"] == "metadata":
        assert_type(part, MetadataStreamPart)
        assert_type(part["data"], RunMetadataPayload)
