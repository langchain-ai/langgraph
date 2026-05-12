"""HTTP/SSE transport for the v3 thread-centric protocol.

Direct port of `libs/sdk/src/client/stream/transport/http.ts`.

`ProtocolSseTransport` (landing in subsequent tasks) is bound to a single
`thread_id` at construction. Commands go to `POST /threads/{thread_id}/commands`
(JSON in, JSON out). Each `open_event_stream(params)` opens an independent
filtered SSE connection at `POST /threads/{thread_id}/stream/events` with the
`SubscribeParams` in the request body.

`ProtocolSseTransport` is present (commands). `open_event_stream` lands in
Task 8.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import orjson
from langchain_protocol import Event


@dataclass
class EventStreamHandle:
    """Handle for one filtered SSE stream.

    Attributes:
        events: async iterator of typed `Event`s. Exhausts when the
            stream closes (server hangup or `close()`).
        ready: resolves once HTTP response headers arrive; rejects on
            connection failure before headers.
        close: invoke to cancel the underlying task and free the
            connection.
    """

    events: AsyncIterator[Event]
    ready: asyncio.Future[None]
    close: Callable[[], Awaitable[None]]


class ProtocolSseTransport:
    """v3 protocol transport bound to a single `thread_id`.

    Commands go to `POST /threads/{thread_id}/commands` (JSON in, JSON out).
    `open_event_stream` (landing in Task 8) opens filtered SSE streams against
    `POST /threads/{thread_id}/stream/events`.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        thread_id: str,
        commands_path: str | None = None,
        stream_path: str | None = None,
    ) -> None:
        self._client = client
        self.thread_id = thread_id
        self._commands_url = commands_path or f"/threads/{thread_id}/commands"
        # Used by open_event_stream in Task 8.
        self._stream_url = stream_path or f"/threads/{thread_id}/stream/events"
        self._closed = False

    async def send_command(self, command: dict[str, Any]) -> dict[str, Any] | None:
        """POST a command. Returns the response JSON, or `None` for 202/204.

        Raises:
            httpx.HTTPStatusError: server returned >= 400.
            RuntimeError: the transport has been closed via `close()`.
            RuntimeError: server returned a response missing the protocol envelope.
        """
        if self._closed:
            raise RuntimeError("Protocol transport is closed.")
        response = await self._client.post(
            self._commands_url,
            content=orjson.dumps(command),
            headers={"content-type": "application/json"},
        )
        response.raise_for_status()
        if response.status_code in (202, 204):
            return None
        payload = orjson.loads(response.content)
        if not isinstance(payload, dict) or "command_id" not in payload:
            raise RuntimeError("Protocol command did not return a valid response.")
        return payload

    async def close(self) -> None:
        """Mark the transport closed. Idempotent."""
        self._closed = True
