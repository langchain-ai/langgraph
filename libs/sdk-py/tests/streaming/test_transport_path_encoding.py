"""Regression tests for #7953: v3 stream transports must percent-encode
`thread_id` in their default paths so a value containing reserved characters
or dot-segments stays an opaque identifier under `/threads/{thread_id}/...`
instead of being normalized into a different resource path by the HTTP stack.
"""

from __future__ import annotations

import httpx
import pytest

from langgraph_sdk.stream.transport.base import build_websocket_url
from langgraph_sdk.stream.transport.http import ProtocolSseTransport
from langgraph_sdk.stream.transport.sync_http import SyncProtocolSseTransport
from langgraph_sdk.stream.transport.sync_ws import SyncProtocolWebSocketTransport
from langgraph_sdk.stream.transport.ws import ProtocolWebSocketTransport

# A thread_id that escapes the /threads/ namespace if interpolated raw: an HTTP
# client collapses `/threads/../assistants/abc/...` to `/assistants/abc/...`.
TRAVERSAL_THREAD_ID = "../assistants/abc"
ENCODED_COMMANDS_PATH = "/threads/..%2Fassistants%2Fabc/commands"
ENCODED_STREAM_PATH = "/threads/..%2Fassistants%2Fabc/stream/events"


@pytest.mark.anyio
async def test_async_sse_default_paths_encode_thread_id():
    transport = ProtocolSseTransport(
        client=httpx.AsyncClient(), thread_id=TRAVERSAL_THREAD_ID
    )
    assert transport._commands_url == ENCODED_COMMANDS_PATH
    assert transport._stream_url == ENCODED_STREAM_PATH


def test_sync_sse_default_paths_encode_thread_id():
    transport = SyncProtocolSseTransport(
        client=httpx.Client(), thread_id=TRAVERSAL_THREAD_ID
    )
    assert transport._commands_url == ENCODED_COMMANDS_PATH
    assert transport._stream_url == ENCODED_STREAM_PATH


@pytest.mark.anyio
async def test_async_ws_default_paths_encode_thread_id():
    transport = ProtocolWebSocketTransport(
        client=httpx.AsyncClient(), thread_id=TRAVERSAL_THREAD_ID
    )
    assert transport._commands_url == ENCODED_COMMANDS_PATH
    assert transport._stream_path == ENCODED_STREAM_PATH


def test_sync_ws_default_paths_encode_thread_id():
    transport = SyncProtocolWebSocketTransport(
        client=httpx.Client(), thread_id=TRAVERSAL_THREAD_ID
    )
    assert transport._commands_url == ENCODED_COMMANDS_PATH
    assert transport._stream_path == ENCODED_STREAM_PATH


async def test_async_sse_wire_path_stays_under_threads_namespace():
    """The path that actually goes on the wire must not be normalized away."""
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.raw_path.decode("ascii"))
        return httpx.Response(202)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://example.com",
        trust_env=False,
    ) as client:
        transport = ProtocolSseTransport(client=client, thread_id=TRAVERSAL_THREAD_ID)
        await transport.send_command({"id": 1, "method": "noop", "params": {}})

    assert seen[0] == ENCODED_COMMANDS_PATH


def test_sync_sse_wire_path_stays_under_threads_namespace():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.raw_path.decode("ascii"))
        return httpx.Response(202)

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://example.com",
        trust_env=False,
    ) as client:
        transport = SyncProtocolSseTransport(
            client=client, thread_id=TRAVERSAL_THREAD_ID
        )
        transport.send_command({"id": 1, "method": "noop", "params": {}})

    assert seen[0] == ENCODED_COMMANDS_PATH


@pytest.mark.anyio
async def test_async_ws_url_stays_under_threads_namespace():
    transport = ProtocolWebSocketTransport(
        client=httpx.AsyncClient(base_url="https://example.com/api"),
        thread_id=TRAVERSAL_THREAD_ID,
    )
    url = build_websocket_url(transport._client.base_url, transport._stream_path)
    assert url == "wss://example.com/api/threads/..%2Fassistants%2Fabc/stream/events"


def test_sync_ws_url_stays_under_threads_namespace():
    transport = SyncProtocolWebSocketTransport(
        client=httpx.Client(base_url="https://example.com/api"),
        thread_id=TRAVERSAL_THREAD_ID,
    )
    url = build_websocket_url(transport._client.base_url, transport._stream_path)
    assert url == "wss://example.com/api/threads/..%2Fassistants%2Fabc/stream/events"


@pytest.mark.anyio
async def test_explicit_path_overrides_are_left_untouched():
    """Callers passing explicit paths opt out of default encoding entirely."""
    sse = ProtocolSseTransport(
        client=httpx.AsyncClient(),
        thread_id=TRAVERSAL_THREAD_ID,
        commands_path="/custom/commands",
        stream_path="/custom/events",
    )
    assert sse._commands_url == "/custom/commands"
    assert sse._stream_url == "/custom/events"

    ws = ProtocolWebSocketTransport(
        client=httpx.AsyncClient(),
        thread_id=TRAVERSAL_THREAD_ID,
        commands_path="/custom/commands",
        stream_path="/custom/events",
    )
    assert ws._commands_url == "/custom/commands"
    assert ws._stream_path == "/custom/events"
