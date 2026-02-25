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


# --- v2 streaming tests ---

_V2_REQUIRED_KEYS = {"type", "ns", "data"}


def _assert_v2_shape(part: Any) -> None:
    """Assert a v2 stream part has the required keys and types."""
    assert isinstance(part, dict), f"Expected dict, got {type(part)}"
    assert part.keys() >= _V2_REQUIRED_KEYS, (
        f"Missing keys: {_V2_REQUIRED_KEYS - part.keys()}"
    )
    assert isinstance(part["type"], str), (
        f"type should be str, got {type(part['type'])}"
    )
    assert isinstance(part["ns"], list), f"ns should be list, got {type(part['ns'])}"
    for elem in part["ns"]:
        assert isinstance(elem, str), f"ns element should be str, got {type(elem)}"


def test_sse_to_v2_dict_basic() -> None:
    """Test basic conversion of SSE event to v2 dict."""
    result = _sse_to_v2_dict("values", {"messages": [{"role": "user"}]})
    assert result is not None
    _assert_v2_shape(result)
    assert result == {
        "type": "values",
        "ns": [],
        "data": {"messages": [{"role": "user"}]},
    }


def test_sse_to_v2_dict_with_namespace() -> None:
    """Test namespace parsing from pipe-separated event name."""
    result = _sse_to_v2_dict("updates|sub:abc", {"key": "val"})
    assert result is not None
    _assert_v2_shape(result)
    assert result == {
        "type": "updates",
        "ns": ["sub:abc"],
        "data": {"key": "val"},
    }


def test_sse_to_v2_dict_with_multiple_ns() -> None:
    """Test multiple namespace parts."""
    result = _sse_to_v2_dict("custom|parent|child:123", "hello")
    assert result is not None
    _assert_v2_shape(result)
    assert result == {
        "type": "custom",
        "ns": ["parent", "child:123"],
        "data": "hello",
    }


def test_sse_to_v2_dict_end_event() -> None:
    """Test that end events return None (stop iteration)."""
    assert _sse_to_v2_dict("end", None) is None


def test_sse_to_v2_dict_metadata_event() -> None:
    """Test metadata control event."""
    result = _sse_to_v2_dict("metadata", {"run_id": "abc-123"})
    assert result is not None
    _assert_v2_shape(result)
    assert result == {
        "type": "metadata",
        "ns": [],
        "data": {"run_id": "abc-123"},
    }


def test_sse_to_v2_dict_messages_partial() -> None:
    """Test messages/partial event type."""
    result = _sse_to_v2_dict("messages/partial", [{"type": "ai", "content": "hi"}])
    assert result is not None
    _assert_v2_shape(result)
    assert result == {
        "type": "messages/partial",
        "ns": [],
        "data": [{"type": "ai", "content": "hi"}],
    }


@pytest.mark.asyncio
async def test_async_stream_v2_client_side_conversion() -> None:
    """Test v2 wrapping with client-side conversion of v1-format SSE events."""
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
    assert parts[0] == {"type": "metadata", "ns": [], "data": {"run_id": "r1"}}
    assert parts[1] == {
        "type": "values",
        "ns": [],
        "data": {"messages": [{"role": "user", "content": "hi"}]},
    }
    assert parts[2] == {
        "type": "updates",
        "ns": ["sub:abc"],
        "data": {"node": {"out": 1}},
    }


def test_sync_stream_v2_client_side_conversion() -> None:
    """Test v2 wrapping with sync generator."""
    from langgraph_sdk._sync.runs import _wrap_stream_v2_sync

    def mock_stream() -> Any:
        yield StreamPart(event="metadata", data={"run_id": "r1"})
        yield StreamPart(event="values", data={"state": "full"})
        yield StreamPart(event="end", data=None)  # type: ignore[arg-type]

    parts: list[StreamPartV2] = list(_wrap_stream_v2_sync(mock_stream()))
    assert len(parts) == 2
    for part in parts:
        _assert_v2_shape(part)
    assert parts[0] == {"type": "metadata", "ns": [], "data": {"run_id": "r1"}}
    assert parts[1] == {"type": "values", "ns": [], "data": {"state": "full"}}


# --- data structure validation tests ---


def test_v2_values_data_structure() -> None:
    """Validate ValuesStreamPart data structure."""
    result = _sse_to_v2_dict("values", {"messages": [{"role": "user"}], "count": 1})
    assert result is not None
    _assert_v2_shape(result)
    assert result["type"] == "values"
    assert isinstance(result["data"], dict)
    # values data should be the full state dict
    assert "messages" in result["data"]
    assert "count" in result["data"]


def test_v2_updates_data_structure() -> None:
    """Validate UpdatesStreamPart data structure."""
    result = _sse_to_v2_dict("updates", {"my_node": {"output": "value"}})
    assert result is not None
    _assert_v2_shape(result)
    assert result["type"] == "updates"
    assert isinstance(result["data"], dict)
    assert "my_node" in result["data"]


def test_v2_tasks_start_data_structure() -> None:
    """Validate TaskPayload (task start) data structure."""
    task_data: dict[str, Any] = {
        "id": "task-123",
        "name": "my_node",
        "input": {"key": "value"},
        "triggers": ["start:my_node"],
    }
    result = _sse_to_v2_dict("tasks", task_data)
    assert result is not None
    _assert_v2_shape(result)
    assert result["type"] == "tasks"
    data = result["data"]
    assert isinstance(data, dict)
    assert isinstance(data["id"], str)
    assert isinstance(data["name"], str)
    assert isinstance(data["triggers"], list)
    assert all(isinstance(t, str) for t in data["triggers"])


def test_v2_tasks_result_data_structure() -> None:
    """Validate TaskResultPayload (task result) data structure."""
    result_data: dict[str, Any] = {
        "id": "task-123",
        "name": "my_node",
        "error": None,
        "interrupts": [],
        "result": {"output": "value"},
    }
    result = _sse_to_v2_dict("tasks", result_data)
    assert result is not None
    _assert_v2_shape(result)
    data = result["data"]
    assert isinstance(data, dict)
    assert isinstance(data["id"], str)
    assert isinstance(data["name"], str)
    assert data["error"] is None
    assert isinstance(data["interrupts"], list)
    assert isinstance(data["result"], dict)


def test_v2_checkpoints_data_structure() -> None:
    """Validate CheckpointPayload data structure."""
    ckpt_data: dict[str, Any] = {
        "config": {"configurable": {"thread_id": "t1"}},
        "metadata": {"step": 1},
        "values": {"count": 42},
        "next": ["my_node"],
        "parent_config": None,
        "tasks": [
            {
                "id": "task-1",
                "name": "my_node",
                "error": None,
                "interrupts": [],
                "state": None,
            }
        ],
    }
    result = _sse_to_v2_dict("checkpoints", ckpt_data)
    assert result is not None
    _assert_v2_shape(result)
    data = result["data"]
    assert isinstance(data, dict)
    assert isinstance(data["config"], dict)
    assert isinstance(data["metadata"], dict)
    assert isinstance(data["values"], dict)
    assert isinstance(data["next"], list)
    assert all(isinstance(n, str) for n in data["next"])
    assert isinstance(data["tasks"], list)
    assert len(data["tasks"]) == 1
    task = data["tasks"][0]
    assert isinstance(task["id"], str)
    assert isinstance(task["name"], str)


def test_v2_debug_data_structure() -> None:
    """Validate DebugPayload data structure."""
    debug_data: dict[str, Any] = {
        "step": 1,
        "timestamp": "2025-01-01T00:00:00Z",
        "type": "task",
        "payload": {
            "id": "task-1",
            "name": "my_node",
            "input": {"key": "val"},
            "triggers": ["start:my_node"],
        },
    }
    result = _sse_to_v2_dict("debug", debug_data)
    assert result is not None
    _assert_v2_shape(result)
    data = result["data"]
    assert isinstance(data, dict)
    assert isinstance(data["step"], int)
    assert isinstance(data["timestamp"], str)
    assert data["type"] in ("checkpoint", "task", "task_result")
    assert isinstance(data["payload"], dict)


def test_v2_metadata_data_structure() -> None:
    """Validate RunMetadataPayload data structure."""
    result = _sse_to_v2_dict("metadata", {"run_id": "abc-123"})
    assert result is not None
    _assert_v2_shape(result)
    assert result["type"] == "metadata"
    data = result["data"]
    assert isinstance(data, dict)
    assert isinstance(data["run_id"], str)


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
