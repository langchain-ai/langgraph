"""HTTP/SSE transport for the v3 thread-centric protocol.

Direct port of `libs/sdk/src/client/stream/transport/http.ts`.

`ProtocolSseTransport` (landing in subsequent tasks) is bound to a single
`thread_id` at construction. Commands go to `POST /threads/{thread_id}/commands`
(JSON in, JSON out). Each `open_event_stream(params)` opens an independent
filtered SSE connection at `POST /threads/{thread_id}/stream/events` with the
`SubscribeParams` in the request body.

This file currently only ships the `EventStreamHandle` return type; the
transport class itself lands in Tasks 7 and 8.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

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
