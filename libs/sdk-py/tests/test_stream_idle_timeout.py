"""A streaming run must be boundable in wall-clock time, not in bytes.

The server sends an SSE heartbeat comment every few seconds for as long as the
response is open (``langgraph_api/sse.py``: ``ServerSentEvent(comment="heartbeat")``).
A comment is bytes on the wire but not an event, so it resets httpx's per-read
timeout without making progress: ``client.runs.stream(...)`` with
``timeout=30`` can block for hours. ``stream_idle_timeout`` is the opt-in bound;
these tests pin its three behaviours — off by default, heartbeats excluded from
progress, real events refreshing the clock — without needing a server.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk._async.http import HttpClient
from langgraph_sdk._sync.http import SyncHttpClient

HEARTBEAT = b": heartbeat\n\n"
EVENT = b"event: metadata\ndata: {}\n\n"


class _TickingStream(httpx.SyncByteStream):
    """Heartbeats only, unless ``event_every`` asks for a real event now and then."""

    def __init__(self, ticks: int, interval: float, event_every: int | None = None):
        self.ticks = ticks
        self.interval = interval
        self.event_every = event_every

    def __iter__(self):
        for i in range(self.ticks):
            if self.event_every is not None and i % self.event_every == 0:
                yield EVENT
            else:
                yield HEARTBEAT
            time.sleep(self.interval)

    def close(self) -> None:
        return None


class _TickingAsyncStream(httpx.AsyncByteStream):
    def __init__(self, ticks: int, interval: float, event_every: int | None = None):
        self.ticks = ticks
        self.interval = interval
        self.event_every = event_every

    async def __aiter__(self):
        for i in range(self.ticks):
            if self.event_every is not None and i % self.event_every == 0:
                yield EVENT
            else:
                yield HEARTBEAT
            await asyncio.sleep(self.interval)

    async def aclose(self) -> None:
        return None


def _sync_http(stream: httpx.SyncByteStream, idle: float | None) -> SyncHttpClient:
    client = httpx.Client(
        base_url="http://api",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        ),
    )
    return SyncHttpClient(client, stream_idle_timeout=idle)


def _async_http(stream: httpx.AsyncByteStream, idle: float | None) -> HttpClient:
    client = httpx.AsyncClient(
        base_url="http://api",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        ),
    )
    return HttpClient(client, stream_idle_timeout=idle)


def test_heartbeats_alone_trip_the_idle_bound():
    http = _sync_http(_TickingStream(ticks=60, interval=0.05), idle=0.3)
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        list(http.stream("/threads/t/runs/stream", "POST", json={}))
    elapsed = time.monotonic() - started
    assert elapsed < 5, f"idle bound did not fire in time: {elapsed:.1f}s"


def test_the_bound_is_off_by_default():
    """Same stream, no bound asked for: it ends normally, exactly as today."""
    http = _sync_http(_TickingStream(ticks=10, interval=0.01), idle=None)
    assert list(http.stream("/threads/t/runs/stream", "POST", json={})) == []


def test_real_events_reset_the_idle_clock():
    http = _sync_http(_TickingStream(ticks=20, interval=0.05, event_every=2), idle=0.25)
    parts = list(http.stream("/threads/t/runs/stream", "POST", json={}))
    assert len(parts) >= 10


async def test_heartbeats_alone_trip_the_idle_bound_async():
    http = _async_http(_TickingAsyncStream(ticks=60, interval=0.05), idle=0.3)
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        async for _ in http.stream("/threads/t/runs/stream", "POST", json={}):
            pass
    elapsed = time.monotonic() - started
    assert elapsed < 5, f"idle bound did not fire in time: {elapsed:.1f}s"


async def test_the_bound_is_off_by_default_async():
    http = _async_http(_TickingAsyncStream(ticks=10, interval=0.01), idle=None)
    parts = [p async for p in http.stream("/threads/t/runs/stream", "POST", json={})]
    assert parts == []


def test_factories_accept_the_parameter():
    assert (
        get_sync_client(
            url="http://api", stream_idle_timeout=12
        ).http.stream_idle_timeout
        == 12
    )
    assert (
        get_client(url="http://api", stream_idle_timeout=12).http.stream_idle_timeout
        == 12
    )
