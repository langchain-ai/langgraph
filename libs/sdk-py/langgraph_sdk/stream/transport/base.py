"""Shared transport contracts for v3 thread-centric streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from langchain_protocol import Event


@dataclass
class EventStreamHandle:
    """Handle for one async filtered event stream."""

    events: AsyncIterator[Event]
    ready: asyncio.Future[None]
    done: asyncio.Future[BaseException | None]
    close: Callable[[], Awaitable[None]]


@dataclass
class SyncEventStreamHandle:
    """Handle for one sync filtered event stream."""

    events: Iterator[Event]
    error: Callable[[], BaseException | None]
    close: Callable[[], None]


class AsyncProtocolTransport(Protocol):
    """Protocol implemented by async SSE and WebSocket transports."""

    thread_id: str

    async def send_command(self, command: dict[str, Any]) -> dict[str, Any] | None: ...

    def open_event_stream(self, params: dict[str, Any]) -> EventStreamHandle: ...

    async def close(self) -> None: ...


class SyncProtocolTransport(Protocol):
    """Protocol implemented by sync SSE and WebSocket transports."""

    thread_id: str

    def send_command(self, command: dict[str, Any]) -> dict[str, Any] | None: ...

    def open_event_stream(self, params: dict[str, Any]) -> SyncEventStreamHandle: ...

    def close(self) -> None: ...


def build_event_stream_body(params: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {"channels": params["channels"]}
    if params.get("namespaces") is not None:
        body["namespaces"] = params["namespaces"]
    if params.get("depth") is not None:
        body["depth"] = params["depth"]
    since = params.get("since")
    if isinstance(since, int):
        body["since"] = since
    return body


def build_websocket_url(base_url: httpx.URL, path: str) -> str:
    """Convert an HTTP base URL plus API path into a WebSocket URL."""
    scheme = "wss" if base_url.scheme == "https" else "ws"
    base_path = base_url.path.rstrip("/")
    stream_path = path if path.startswith("/") else f"/{path}"
    full_path = f"{base_path}{stream_path}" if base_path else stream_path
    return str(base_url.copy_with(scheme=scheme, path=full_path, query=None))


def websocket_headers(headers: Mapping[str, str] | None) -> list[tuple[str, str]]:
    return list(dict(headers or {}).items())
