"""Synchronous fake v3 protocol server for sync streaming tests."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import orjson


@dataclass
class SyncStreamScript:
    events: list[dict[str, Any]]
    fail_after: int | None = None


class _SseByteStream(httpx.SyncByteStream):
    def __init__(self, script: SyncStreamScript) -> None:
        self._script = script

    def __iter__(self) -> Iterator[bytes]:
        for index, event in enumerate(self._script.events, start=1):
            payload = orjson.dumps(event).decode()
            yield f"id: {event.get('event_id', '')}\n".encode()
            yield f"event: message\ndata: {payload}\n\n".encode()
            if self._script.fail_after is not None and index >= self._script.fail_after:
                raise httpx.ReadError("scripted sync stream failure")


class SyncFakeServer:
    """Synchronous fake for `/commands`, `/stream/events`, and `/state`."""

    def __init__(self) -> None:
        self.received_commands: list[dict[str, Any]] = []
        self.stream_request_bodies: list[dict[str, Any]] = []
        self.command_request_headers: list[dict[str, str]] = []
        self.stream_request_headers_list: list[dict[str, str]] = []
        self.state_request_headers: list[dict[str, str]] = []
        self.state_request_count = 0
        self.scripted_events: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {}
        self.transport = httpx.MockTransport(self._handle)
        self._stream_scripts: list[SyncStreamScript] = []
        self._command_response: dict[str, Any] | None = None
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
        fail_after: int | None = None,
    ) -> None:
        self.scripted_events = list(events)
        self._stream_scripts = [
            SyncStreamScript(events=list(events), fail_after=fail_after)
        ]

    def script_sequence(self, scripts: list[SyncStreamScript]) -> None:
        self._stream_scripts = list(scripts)
        self.scripted_events = []

    def set_graph(self, graph: dict[str, Any]) -> None:
        self.graph_response = dict(graph)

    def script_command_response(self, response: dict[str, Any]) -> None:
        self._command_response = dict(response)

    def set_state(
        self,
        values: dict[str, Any],
        next: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.state = {
            "values": values,
            "next": next if next is not None else [],
            "tasks": [],
            "metadata": metadata if metadata is not None else {},
            "checkpoint": None,
            "created_at": None,
        }

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/commands"):
            body = orjson.loads(request.content)
            self.received_commands.append(body)
            self.command_request_headers.append(dict(request.headers))
            if self._command_response is not None:
                response = dict(self._command_response)
                response["id"] = body.get("id")
                return httpx.Response(200, json=response)
            return httpx.Response(
                200,
                json={
                    "type": "success",
                    "id": body.get("id"),
                    "result": {"run_id": "run-1"},
                },
            )
        if path.endswith("/stream/events"):
            self.stream_request_bodies.append(orjson.loads(request.content))
            self.stream_request_headers_list.append(dict(request.headers))
            script = (
                self._stream_scripts.pop(0)
                if self._stream_scripts
                else SyncStreamScript(events=list(self.scripted_events))
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_SseByteStream(script),
            )
        if path.endswith("/graph") and "/assistants/" in path:
            self.graph_request_params.append(dict(request.url.params))
            self.graph_request_headers.append(dict(request.headers))
            return httpx.Response(200, json=self.graph_response)
        if path.endswith("/state"):
            self.state_request_count += 1
            self.state_request_headers.append(dict(request.headers))
            return httpx.Response(200, json=self.state)
        return httpx.Response(404, json={"error": f"unexpected path: {path}"})
