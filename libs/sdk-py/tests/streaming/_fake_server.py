"""In-process ASGI fake of the v3 protocol endpoints.

Used by transport and thread-streaming tests. Mirrors the production endpoints
just closely enough to validate the client:

  - POST /threads/{thread_id}/commands
  - POST /threads/{thread_id}/stream/events
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import orjson
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


class FakeServer:
    """Holds scripted state for tests and exposes a Starlette app.

    Attributes:
        received_commands: every command body posted to /commands, in order.
        scripted_events: events the next /stream/events call will replay.
        stream_request_bodies: bodies posted to /stream/events, in order.
        command_request_headers: headers from each POST to /commands, in order.
        stream_request_headers_list: headers from each POST to /stream/events, in order.
    """

    def __init__(self) -> None:
        self.received_commands: list[dict[str, Any]] = []
        self.scripted_events: list[dict[str, Any]] = []
        self.stream_request_bodies: list[dict[str, Any]] = []
        self.command_request_headers: list[dict[str, str]] = []
        self.stream_request_headers_list: list[dict[str, str]] = []
        self._stream_delay: float = 0.0
        self._app: Starlette | None = None
        self.open_event_streams = 0
        self._open_event_streams_max = 0

    def script(self, events: list[dict[str, Any]], *, delay: float = 0.0) -> None:
        """Set the events the next /stream/events call will replay."""
        self.scripted_events = list(events)
        self._stream_delay = delay

    @property
    def app(self) -> Starlette:
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def _build_app(self) -> Starlette:
        async def commands(request: Request) -> Response:
            body = orjson.loads(await request.body())
            self.received_commands.append(body)
            self.command_request_headers.append(dict(request.headers))
            command_id = body.get("id")
            return JSONResponse(
                {
                    "type": "success",
                    "id": command_id,
                    "result": {"run_id": "run-1"},
                }
            )

        async def stream_events(request: Request) -> Response:
            self.stream_request_bodies.append(orjson.loads(await request.body()))
            self.stream_request_headers_list.append(dict(request.headers))
            return StreamingResponse(
                self._sse_body(),
                media_type="text/event-stream",
            )

        return Starlette(
            routes=[
                Route("/threads/{thread_id}/commands", commands, methods=["POST"]),
                Route(
                    "/threads/{thread_id}/stream/events",
                    stream_events,
                    methods=["POST"],
                ),
            ]
        )

    async def _sse_body(self) -> AsyncIterator[bytes]:
        self.open_event_streams += 1
        self._open_event_streams_max = max(
            self._open_event_streams_max, self.open_event_streams
        )
        try:
            # Why: script() rebinds scripted_events; in-flight iterators retain
            # a reference to the prior list and are unaffected by later
            # script() calls.
            for event in self.scripted_events:
                if self._stream_delay:
                    await asyncio.sleep(self._stream_delay)
                payload = orjson.dumps(event).decode()
                yield f"id: {event.get('event_id', '')}\n".encode()
                yield f"event: message\ndata: {payload}\n\n".encode()
        finally:
            self.open_event_streams -= 1

    @property
    def peak_open_event_streams(self) -> int:
        return self._open_event_streams_max
