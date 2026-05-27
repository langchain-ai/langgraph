"""Synchronous HTTP/SSE transport for the v3 thread-centric protocol."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any, cast

import httpx
import orjson
from langchain_protocol import Event

from langgraph_sdk.sse import BytesLineDecoder, SSEDecoder
from langgraph_sdk.stream.transport.base import (
    SyncEventStreamHandle,
    build_event_stream_body,
)


class SyncProtocolSseTransport:
    """Sync v3 protocol transport bound to one thread id."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        thread_id: str,
        commands_path: str | None = None,
        stream_path: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._client = client
        self.thread_id = thread_id
        self._commands_url = commands_path or f"/threads/{thread_id}/commands"
        self._stream_url = stream_path or f"/threads/{thread_id}/stream/events"
        self._default_headers: dict[str, str] = dict(headers or {})
        self._closed = False
        self._open_responses: list[httpx.Response] = []

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
        sse_headers = {
            **self._default_headers,
            "content-type": "application/json",
            "accept": "text/event-stream",
            "cache-control": "no-store",
        }
        request = self._client.build_request(
            "POST",
            self._stream_url,
            content=orjson.dumps(build_event_stream_body(params)),
            headers=sse_headers,
        )
        stream_cm = self._client.send(request, stream=True)
        stream_cm.raise_for_status()
        content_type = stream_cm.headers.get("content-type", "").partition(";")[0]
        if "text/event-stream" not in content_type:
            stream_cm.close()
            raise httpx.TransportError(
                "Expected response header Content-Type to contain "
                f"'text/event-stream', got {content_type!r}"
            )
        self._open_responses.append(stream_cm)
        closed = False
        stream_error: BaseException | None = None

        def events() -> Iterator[Event]:
            nonlocal stream_error
            line_decoder = BytesLineDecoder()
            sse_decoder = SSEDecoder()
            try:
                for chunk in stream_cm.iter_bytes():
                    if closed:
                        return
                    for line in line_decoder.decode(chunk):
                        part = sse_decoder.decode(bytes(line))
                        if part is not None and isinstance(part.data, dict):
                            yield cast("Event", part.data)
                for line in line_decoder.flush():
                    part = sse_decoder.decode(bytes(line))
                    if part is not None and isinstance(part.data, dict):
                        yield cast("Event", part.data)
                part = sse_decoder.decode(b"")
                if part is not None and isinstance(part.data, dict):
                    yield cast("Event", part.data)
            except BaseException as exc:
                if not closed:
                    stream_error = exc
                raise
            finally:
                with contextlib.suppress(ValueError):
                    self._open_responses.remove(stream_cm)
                stream_cm.close()

        def error() -> BaseException | None:
            return stream_error

        def close() -> None:
            nonlocal closed
            closed = True
            with contextlib.suppress(Exception):
                stream_cm.close()

        return SyncEventStreamHandle(events=events(), error=error, close=close)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for response in list(self._open_responses):
            with contextlib.suppress(Exception):
                response.close()
        self._open_responses.clear()
