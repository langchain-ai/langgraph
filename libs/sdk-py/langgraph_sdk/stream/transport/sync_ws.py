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
    ) -> None:
        self._client = client
        self.thread_id = thread_id
        self._commands_url = commands_path or f"/threads/{thread_id}/commands"
        self._stream_path = stream_path or f"/threads/{thread_id}/stream/events"
        self._default_headers: dict[str, str] = dict(headers or {})
        self._connect = connect
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
        # Pre-enter the WebSocket context manager so close() can reach the socket
        # immediately, even before the caller has started iterating events().
        ws_cm = self._connect(
            url,
            additional_headers=websocket_headers(self._default_headers),
        )
        websocket = ws_cm.__enter__()

        def events() -> Iterator[Event]:
            nonlocal stream_error
            try:
                websocket.send(orjson.dumps(build_event_stream_body(params)).decode())
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
