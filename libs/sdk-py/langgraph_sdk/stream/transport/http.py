"""HTTP/SSE transport for the v3 thread-centric protocol.

Direct port of `libs/sdk/src/client/stream/transport/http.ts`.

`ProtocolSseTransport` is bound to a single `thread_id` at construction. Commands
go to `POST /threads/{thread_id}/commands` (JSON in, JSON out). Each
`open_event_stream(params)` opens an independent filtered SSE connection at
`POST /threads/{thread_id}/stream/events` with the `SubscribeParams` in the
request body.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import httpx
import orjson
from langchain_protocol import Event

from langgraph_sdk._shared.utilities import _quote_path_param
from langgraph_sdk.sse import BytesLineDecoder, SSEDecoder
from langgraph_sdk.stream.transport.base import (
    EventStreamHandle,
    build_event_stream_body,
)

_build_event_stream_body = build_event_stream_body


class ProtocolSseTransport:
    """v3 protocol transport bound to a single `thread_id`.

    Commands go to `POST /threads/{thread_id}/commands` (JSON in, JSON out).
    `open_event_stream` opens filtered SSE streams against
    `POST /threads/{thread_id}/stream/events`.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        thread_id: str,
        commands_path: str | None = None,
        stream_path: str | None = None,
        headers: Mapping[str, str] | None = None,
        max_queue_size: int = 1024,
    ) -> None:
        self._client = client
        self.thread_id = thread_id
        self._commands_url = (
            commands_path or f"/threads/{_quote_path_param(thread_id)}/commands"
        )
        self._stream_url = (
            stream_path or f"/threads/{_quote_path_param(thread_id)}/stream/events"
        )
        self._default_headers: dict[str, str] = dict(headers or {})
        self._max_queue_size = max_queue_size
        self._closed = False
        self._event_streams: set[asyncio.Task[None]] = set()

    async def send_command(self, command: dict[str, Any]) -> dict[str, Any] | None:
        """POST a command. Returns the response JSON, or `None` for 202/204.

        Raises:
            httpx.HTTPStatusError: server returned >= 400.
            RuntimeError: the transport has been closed via `close()`.
            RuntimeError: server returned a response missing the protocol envelope.
        """
        if self._closed:
            raise RuntimeError("Protocol transport is closed.")
        # Merge default headers first so content-type always wins.
        merged_headers = {**self._default_headers, "content-type": "application/json"}
        response = await self._client.post(
            self._commands_url,
            content=orjson.dumps(command),
            headers=merged_headers,
        )
        response.raise_for_status()
        if response.status_code in (202, 204):
            return None
        if not response.content:
            raise RuntimeError("Protocol command did not return a valid response.")
        try:
            payload = orjson.loads(response.content)
        except orjson.JSONDecodeError as err:
            raise RuntimeError(
                "Protocol command did not return a valid response."
            ) from err
        if not isinstance(payload, dict) or "id" not in payload:
            raise RuntimeError("Protocol command did not return a valid response.")
        return payload

    def open_event_stream(self, params: dict[str, Any]) -> EventStreamHandle:
        """Open an independent filtered SSE event stream.

        Posts `params` as a SubscribeParams body to `/threads/{thread_id}/stream/events`.
        Returns an `EventStreamHandle` whose `events` async iterator yields typed
        `Event` dicts as the server emits them. `handle.ready` resolves on a 2xx
        response (rejects on HTTP error or transport failure before headers).

        Reconnect: pass `params["since"]` to filter outbound seqs server-side. The
        cursor goes in the request body, not as a `Last-Event-ID` header.
        """
        if self._closed:
            raise RuntimeError("Protocol transport is closed.")

        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()
        done: asyncio.Future[BaseException | None] = loop.create_future()
        queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=self._max_queue_size)
        cancel_event = asyncio.Event()

        async def pump() -> None:
            try:
                # Merge default headers first so fixed SSE headers always win.
                sse_headers = {
                    **self._default_headers,
                    "content-type": "application/json",
                    "accept": "text/event-stream",
                    "cache-control": "no-store",
                }
                async with self._client.stream(
                    "POST",
                    self._stream_url,
                    content=orjson.dumps(build_event_stream_body(params)),
                    headers=sse_headers,
                ) as response:
                    response.raise_for_status()
                    if not ready.done():
                        ready.set_result(None)
                    line_decoder = BytesLineDecoder()
                    sse_decoder = SSEDecoder()
                    async for chunk in response.aiter_bytes():
                        if cancel_event.is_set():
                            break
                        for line in line_decoder.decode(chunk):
                            part = sse_decoder.decode(bytes(line))
                            if part is None:
                                continue
                            if isinstance(part.data, dict):
                                await queue.put(cast("Event", part.data))
                    # Drain any trailing buffered line, then fire any pending event.
                    if not cancel_event.is_set():
                        for line in line_decoder.flush():
                            part = sse_decoder.decode(bytes(line))
                            if part is not None and isinstance(part.data, dict):
                                await queue.put(cast("Event", part.data))
                        part = sse_decoder.decode(b"")
                        if part is not None and isinstance(part.data, dict):
                            await queue.put(cast("Event", part.data))
            except asyncio.CancelledError as err:
                if not done.done():
                    done.set_result(err)
                raise
            except BaseException as err:
                if not ready.done():
                    ready.set_exception(err)
                if not done.done():
                    done.set_result(err)
            finally:
                if not done.done():
                    done.set_result(None)
                await queue.put(None)  # sentinel: end of stream

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
            # Why: pump may be mid-`finally`; ensure consumer unblocks.
            queue.put_nowait(None)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        return EventStreamHandle(events=aiter(), ready=ready, done=done, close=close)

    async def close(self) -> None:
        """Cancel any open event streams and mark the transport closed. Idempotent."""
        if self._closed:
            return
        self._closed = True
        tasks = list(self._event_streams)
        for task in tasks:
            task.cancel()
        if tasks:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.gather(*tasks, return_exceptions=True)
