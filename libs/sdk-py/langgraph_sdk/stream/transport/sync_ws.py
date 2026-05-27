"""Sync WebSocket transport for the v3 thread-centric protocol."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from typing import Any, cast

import httpx
import orjson
from langchain_protocol import Event
from websockets.sync.client import connect as websocket_connect

from langgraph_sdk.stream.transport.base import (
    SyncEventStreamHandle,
    build_event_stream_body,
    build_websocket_url,
    websocket_headers,
)


class SyncProtocolWebSocketTransport:
    """Sync v3 protocol transport using HTTP commands and WebSocket events."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        thread_id: str,
        commands_path: str | None = None,
        stream_path: str | None = None,
        headers: Mapping[str, str] | None = None,
        connect: Callable[..., Any] = websocket_connect,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = 20.0,
    ) -> None:
        self._client = client
        self.thread_id = thread_id
        self._commands_url = commands_path or f"/threads/{thread_id}/commands"
        self._stream_path = stream_path or f"/threads/{thread_id}/stream/events"
        self._default_headers: dict[str, str] = dict(headers or {})
        self._connect = connect
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._closed = False

    def send_command(self, command: dict[str, Any]) -> dict[str, Any] | None:
        if self._closed:
            raise RuntimeError("Protocol transport is closed.")
        merged_headers = {**self._default_headers, "content-type": "application/json"}
        response = self._client.post(
            self._commands_url,
            content=orjson.dumps(command),
            headers=merged_headers,
        )
        response.raise_for_status()
        if response.status_code in (202, 204):
            return None
        payload = orjson.loads(response.content)
        if not isinstance(payload, dict) or "id" not in payload:
            raise RuntimeError("Protocol command did not return a valid response.")
        return payload

    def open_event_stream(self, params: dict[str, Any]) -> SyncEventStreamHandle:
        if self._closed:
            raise RuntimeError("Protocol transport is closed.")
        closed = False
        stream_error: BaseException | None = None

        url = build_websocket_url(self._client.base_url, self._stream_path)
        handshake_headers = list(websocket_headers(self._default_headers))
        cookie_header = _cookie_header(self._client, self._stream_path)
        if cookie_header:
            handshake_headers.append(("Cookie", cookie_header))
        # Pre-enter the WebSocket context manager so close() can reach the socket
        # immediately, even before the caller has started iterating events().
        ws_cm = self._connect(
            url,
            additional_headers=handshake_headers,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
        )
        websocket = ws_cm.__enter__()

        def events() -> Iterator[Event]:
            nonlocal stream_error
            try:
                # Wrap the initial subscribe in a ``subscription.subscribe``
                # Protocol command envelope so the server's WS endpoint
                # (see ``langgraph-api`` ``api/event_streaming.py``
                # ``_thread_websocket``) accepts it. Bare subscribe bodies
                # are rejected with ``invalid_argument``.
                subscribe_command = {
                    "id": 1,
                    "method": "subscription.subscribe",
                    "params": build_event_stream_body(params),
                }
                websocket.send(orjson.dumps(subscribe_command).decode())
                for raw in websocket:
                    if closed:
                        return
                    payload = _decode_frame(raw)
                    if isinstance(payload, dict):
                        yield cast("Event", payload)
            except BaseException as exc:
                if not closed:
                    stream_error = exc
                raise
            finally:
                with contextlib.suppress(Exception):
                    ws_cm.__exit__(None, None, None)

        def error() -> BaseException | None:
            return stream_error

        def close() -> None:
            nonlocal closed
            closed = True
            with contextlib.suppress(Exception):
                websocket.close()

        return SyncEventStreamHandle(events=events(), error=error, close=close)

    def close(self) -> None:
        self._closed = True


def _decode_frame(raw: str | bytes | bytearray | memoryview) -> Any:
    if isinstance(raw, str):
        return orjson.loads(raw.encode())
    return orjson.loads(bytes(raw))


def _cookie_header(client: httpx.Client, path: str) -> str | None:
    """Build a `Cookie` header for the WebSocket handshake.

    Why pass `path`: `dict(client.cookies)` flattens the entire jar without
    domain/path filtering, so cookies set by responses from other origins would
    leak to the WS server. We delegate to `httpx.Cookies.set_cookie_header`,
    which applies the same `CookieJar` rules httpx uses for regular HTTP
    requests, scoping the result to `client.base_url` + `path`.
    """
    if not list(client.cookies.jar):
        return None
    target = client.base_url.copy_with(path=path)
    request = httpx.Request("GET", target)
    client.cookies.set_cookie_header(request)
    return request.headers.get("Cookie")
