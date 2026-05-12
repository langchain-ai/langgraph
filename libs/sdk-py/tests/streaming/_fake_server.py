"""In-process ASGI fake of the v3 protocol endpoints.

Used by Phase 1 transport tests. Mirrors the production endpoints just
closely enough to validate the client:

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
    """

    def __init__(self) -> None:
        self.received_commands: list[dict[str, Any]] = []
        self.scripted_events: list[dict[str, Any]] = []
        self.stream_request_bodies: list[dict[str, Any]] = []
        self._stream_delay: float = 0.0

    def script(self, events: list[dict[str, Any]], *, delay: float = 0.0) -> None:
        """Set the events the next /stream/events call will replay."""
        self.scripted_events = list(events)
        self._stream_delay = delay

    @property
    def app(self) -> Starlette:
        async def commands(request: Request) -> Response:
            body = orjson.loads(await request.body())
            self.received_commands.append(body)
            command_id = body.get("command_id")
            return JSONResponse(
                {"command_id": command_id, "result": {"run_id": "run-1"}}
            )

        async def stream_events(request: Request) -> Response:
            self.stream_request_bodies.append(orjson.loads(await request.body()))
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
        for event in self.scripted_events:
            if self._stream_delay:
                await asyncio.sleep(self._stream_delay)
            payload = orjson.dumps(event).decode()
            yield f"id: {event.get('id', '')}\n".encode()
            yield f"event: message\ndata: {payload}\n\n".encode()
