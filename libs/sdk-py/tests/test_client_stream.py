from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from typing_extensions import assert_type

from langgraph_sdk._shared.utilities import _sse_to_v2_dict
from langgraph_sdk.client import HttpClient, SyncHttpClient
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
        StreamPart(event="end", data=None, id="2"),
    ]


@pytest.mark.asyncio
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
        StreamPart(event="end", data=None, id="2"),
    ]


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
    from langgraph_sdk._async.runs import _wrap_stream_v2

    async def mock_stream() -> Any:
        yield StreamPart(event="metadata", data={"run_id": "r1"})
        yield StreamPart(
            event="values", data={"messages": [{"role": "user", "content": "hi"}]}
        )
        yield StreamPart(event="updates|sub:abc", data={"node": {"out": 1}})
        yield StreamPart(event="end", data=None)  # type: ignore[arg-type]

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
    from langgraph_sdk._sync.runs import _wrap_stream_v2_sync

    def mock_stream() -> Any:
        yield StreamPart(event="metadata", data={"run_id": "r1"})
        yield StreamPart(event="values", data={"state": "full"})
        yield StreamPart(event="end", data=None)  # type: ignore[arg-type]

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
    """Compile-time type narrowing checks — validates mypy narrows the union."""
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
