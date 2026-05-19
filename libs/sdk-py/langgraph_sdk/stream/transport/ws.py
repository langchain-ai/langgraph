"""Async WebSocket transport for the v3 thread-centric protocol."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, cast

import httpx
import orjson
from langchain_protocol import Event
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from langgraph_sdk.stream.transport.base import (
    EventStreamHandle,
    build_event_stream_body,
    build_websocket_url,
    websocket_headers,
)


class ProtocolWebSocketTransport:
    """v3 protocol transport using HTTP commands and WebSocket events."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        thread_id: str,
        commands_path: str | None = None,
        stream_path: str | None = None,
        headers: Mapping[str, str] | None = None,
        connect: Callable[..., Any] = websocket_connect,
        max_queue_size: int = 1024,
    ) -> None:
        self._client = client
        self.thread_id = thread_id
        self._commands_url = commands_path or f"/threads/{thread_id}/commands"
        self._stream_path = stream_path or f"/threads/{thread_id}/stream/events"
        self._default_headers: dict[str, str] = dict(headers or {})
        self._connect = connect
        self._max_queue_size = max_queue_size
        self._closed = False
        self._event_streams: set[asyncio.Task[None]] = set()

    async def send_command(self, command: dict[str, Any]) -> dict[str, Any] | None:
        if self._closed:
            raise RuntimeError("Protocol transport is closed.")
        merged_headers = {**self._default_headers, "content-type": "application/json"}
        response = await self._client.post(
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

    def open_event_stream(self, params: dict[str, Any]) -> EventStreamHandle:
        if self._closed:
            raise RuntimeError("Protocol transport is closed.")

        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()
        done: asyncio.Future[BaseException | None] = loop.create_future()
        queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=self._max_queue_size)
        cancel_event = asyncio.Event()

        async def pump() -> None:
            try:
                url = build_websocket_url(self._client.base_url, self._stream_path)
                async with self._connect(
                    url,
                    additional_headers=websocket_headers(self._default_headers),
                ) as websocket:
                    await websocket.send(
                        orjson.dumps(build_event_stream_body(params)).decode()
                    )
                    if not ready.done():
                        ready.set_result(None)
                    async for raw in websocket:
                        if cancel_event.is_set():
                            break
                        payload = _decode_frame(raw, done)
                        if payload is not None:
                            await queue.put(cast("Event", payload))
            except asyncio.CancelledError as err:
                if not done.done():
                    done.set_result(err)
                raise
            except ConnectionClosedOK:
                # Server sent close code 1000 — clean end, not an error.
                if not done.done():
                    done.set_result(None)
            except ConnectionClosedError as err:
                # Abnormal close (1006) or application error (4xxx).
                if not ready.done():
                    ready.set_exception(err)
                if not done.done():
                    done.set_result(err)
            except Exception as err:
                if not ready.done():
                    ready.set_exception(err)
                if not done.done():
                    done.set_result(err)
            finally:
                if not done.done():
                    done.set_result(None)
                await queue.put(None)

        task = asyncio.create_task(pump())
        self._event_streams.add(task)
        task.add_done_callback(self._event_streams.discard)

        async def aiter() -> AsyncIterator[Event]:
            while True:
                item = await queue.get()
                if item is None or cancel_event.is_set():
                    return
                yield item

        async def close() -> None:
            cancel_event.set()
            queue.put_nowait(None)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        return EventStreamHandle(events=aiter(), ready=ready, done=done, close=close)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = list(self._event_streams)
        for task in tasks:
            task.cancel()
        if tasks:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.gather(*tasks, return_exceptions=True)


def _decode_frame(
    raw: str | bytes | bytearray | memoryview,
    done: asyncio.Future[BaseException | None],
) -> dict[str, Any] | None:
    """Decode a raw WS frame into an Event dict.

    Returns None and sets `done` if the frame is invalid JSON or not a JSON object.
    """
    try:
        payload = orjson.loads(raw.encode() if isinstance(raw, str) else bytes(raw))
    except orjson.JSONDecodeError as err:
        if not done.done():
            done.set_result(RuntimeError(f"WS frame is not valid JSON: {err!r}"))
        return None
    if not isinstance(payload, dict):
        if not done.done():
            done.set_result(
                RuntimeError(f"WS frame is not a JSON object: {type(payload).__name__}")
            )
        return None
    return payload
