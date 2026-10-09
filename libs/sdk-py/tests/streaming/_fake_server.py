"""In-process ASGI fake of the v3 protocol endpoints.

Mirrors the production endpoints just closely enough to validate the client:

  - POST /threads/{thread_id}/commands
  - POST /threads/{thread_id}/stream/events
  - GET /threads/{thread_id}/state
  - GET /assistants/{assistant_id}/graph
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import orjson
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


class _AsyncSseByteStream(httpx.AsyncByteStream):
    """Async SSE byte stream that supports mid-stream errors via `fail_after`."""

    def __init__(self, script: _StreamScript) -> None:
        self._script = script

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, event in enumerate(self._script.events, start=1):
            if self._script.delay:
                await asyncio.sleep(self._script.delay)
            payload = orjson.dumps(event).decode()
            yield f"id: {event.get('event_id', '')}\n".encode()
            yield f"event: message\ndata: {payload}\n\n".encode()
            if self._script.fail_after is not None and index >= self._script.fail_after:
                raise httpx.ReadError("scripted async stream failure")


class _CountedAsyncSseByteStream(httpx.AsyncByteStream):
    """Wraps `_AsyncSseByteStream` and decrements the server's open stream counter."""

    def __init__(self, script: _StreamScript, server: Any) -> None:
        self._inner = _AsyncSseByteStream(script)
        self._server = server

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._inner:
                yield chunk
        finally:
            self._server.open_event_streams -= 1


@dataclass
class _StreamScript:
    events: list[dict[str, Any]]
    delay: float = 0.0
    fail_after: int | None = None


class FakeServer:
    """Holds scripted state for tests and exposes a Starlette app.

    Attributes:
        received_commands: every command body posted to /commands, in order.
        scripted_events: events the next /stream/events call will replay.
        stream_request_bodies: bodies posted to /stream/events, in order.
        command_request_headers: headers from each POST to /commands, in order.
        stream_request_headers_list: headers from each POST to /stream/events, in order.
        state: the `ThreadState`-shaped dict returned by GET /threads/{thread_id}/state.
        state_request_count: number of times the state endpoint has been called.
        state_request_headers: headers from each GET to /threads/{thread_id}/state, in order.
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
        self.state: dict[str, Any] = {}
        self.state_request_count: int = 0
        self.state_request_headers: list[dict[str, str]] = []
        self._stream_scripts: list[_StreamScript] = []
        self._command_response: dict[str, Any] | None = None
        self.transport: httpx.MockTransport = httpx.MockTransport(self._handle_request)
        self.graph_response: dict[str, Any] = {
            "nodes": [{"id": "agent", "type": "runnable", "data": {"name": "agent"}}],
            "edges": [],
        }
        self.graph_request_params: list[dict[str, str]] = []
        self.graph_request_headers: list[dict[str, str]] = []

    def script(
        self,
        events: list[dict[str, Any]],
        *,
        delay: float = 0.0,
        fail_after: int | None = None,
    ) -> None:
        """Set the events the next /stream/events calls will replay."""
        self.scripted_events = list(events)
        self._stream_delay = delay
        self._stream_scripts = [
            _StreamScript(events=list(events), delay=delay, fail_after=fail_after)
        ]

    def script_sequence(self, scripts: list[_StreamScript]) -> None:
        """Set per-open stream scripts consumed in order by /stream/events."""
        self._stream_scripts = list(scripts)
        self.scripted_events = []

    def script_command_response(self, response: dict[str, Any]) -> None:
        """Set the command envelope returned by /commands."""
        self._command_response = dict(response)

    def set_graph(self, graph: dict[str, Any]) -> None:
        """Store the graph returned by GET /assistants/{assistant_id}/graph."""
        self.graph_response = dict(graph)

    def set_state(
        self,
        values: dict[str, Any],
        next: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a `ThreadState`-shaped dict for GET /threads/{thread_id}/state."""
        self.state = {
            "values": values,
            "next": next if next is not None else [],
            "tasks": [],
            "metadata": metadata if metadata is not None else {},
            "checkpoint": None,
            "created_at": None,
        }

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
            if self._command_response is not None:
                response = dict(self._command_response)
                response["id"] = command_id
                return JSONResponse(response)
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

        async def thread_state(request: Request) -> Response:
            self.state_request_count += 1
            self.state_request_headers.append(dict(request.headers))
            return JSONResponse(self.state)

        async def assistant_graph(request: Request) -> Response:
            self.graph_request_params.append(dict(request.query_params))
            self.graph_request_headers.append(dict(request.headers))
            return JSONResponse(self.graph_response)

        return Starlette(
            routes=[
                Route("/threads/{thread_id}/commands", commands, methods=["POST"]),
                Route(
                    "/threads/{thread_id}/stream/events",
                    stream_events,
                    methods=["POST"],
                ),
                Route(
                    "/threads/{thread_id}/state",
                    thread_state,
                    methods=["GET"],
                ),
                Route(
                    "/assistants/{assistant_id}/graph",
                    assistant_graph,
                    methods=["GET"],
                ),
            ]
        )

    async def _sse_body(self) -> AsyncIterator[bytes]:
        self.open_event_streams += 1
        self._open_event_streams_max = max(
            self._open_event_streams_max, self.open_event_streams
        )
        script = (
            self._stream_scripts.pop(0)
            if self._stream_scripts
            else _StreamScript(
                events=list(self.scripted_events), delay=self._stream_delay
            )
        )
        try:
            for index, event in enumerate(script.events, start=1):
                if script.delay:
                    await asyncio.sleep(script.delay)
                payload = orjson.dumps(event).decode()
                yield f"id: {event.get('event_id', '')}\n".encode()
                yield f"event: message\ndata: {payload}\n\n".encode()
                if script.fail_after is not None and index >= script.fail_after:
                    raise RuntimeError("scripted async stream failure")
        finally:
            self.open_event_streams -= 1

    @property
    def peak_open_event_streams(self) -> int:
        return self._open_event_streams_max

    async def _handle_request(self, request: httpx.Request) -> httpx.Response:
        """Async handler for `httpx.MockTransport` — supports proper streaming failures."""
        path = request.url.path
        if path.endswith("/commands"):
            body = orjson.loads(request.content)
            self.received_commands.append(body)
            self.command_request_headers.append(dict(request.headers))
            command_id = body.get("id")
            if self._command_response is not None:
                response = dict(self._command_response)
                response["id"] = command_id
                return httpx.Response(200, json=response)
            return httpx.Response(
                200,
                json={
                    "type": "success",
                    "id": command_id,
                    "result": {"run_id": "run-1"},
                },
            )
        if path.endswith("/stream/events"):
            self.stream_request_bodies.append(orjson.loads(request.content))
            self.stream_request_headers_list.append(dict(request.headers))
            script = (
                self._stream_scripts.pop(0)
                if self._stream_scripts
                else _StreamScript(
                    events=list(self.scripted_events), delay=self._stream_delay
                )
            )
            self.open_event_streams += 1
            self._open_event_streams_max = max(
                self._open_event_streams_max, self.open_event_streams
            )
            # Wrap the stream to decrement open_event_streams on exhaustion.
            stream = _CountedAsyncSseByteStream(script, self)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        if path.endswith("/state"):
            self.state_request_count += 1
            self.state_request_headers.append(dict(request.headers))
            return httpx.Response(200, json=self.state)
        return httpx.Response(404, json={"error": f"unexpected path: {path}"})
