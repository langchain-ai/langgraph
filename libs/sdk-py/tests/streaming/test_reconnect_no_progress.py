"""Post-ready drops that deliver nothing must not reconnect without bound.

`_reconnect_shared_stream` returns as soon as a new stream reaches `ready`, so a server that
keeps accepting the connection and then dropping it without sending anything resets the attempt
budget on every cycle. Once each read is bounded (an idle stream raises instead of hanging), that
turns a silent hang into a reconnect loop that never stops.

The fakes cap how many streams they will open. Without a cap the buggy path never suspends — the
fake resolves `ready` synchronously — so the test hangs instead of failing.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest

from langgraph_sdk.stream.controller import StreamController
from langgraph_sdk.stream.sync_controller import SyncStreamController
from langgraph_sdk.stream.transport.base import (
    EventStreamHandle,
    SyncEventStreamHandle,
)
from streaming._events import values_event

MAX_RECONNECT_ATTEMPTS = 3
MAX_OPENS = 20


class ScriptedAsyncTransport:
    """Every stream opens, yields `events_per_open` events, then drops with a read timeout."""

    thread_id = "t-1"

    def __init__(self, *, events_per_open: int = 0, opens_cap: int = MAX_OPENS) -> None:
        self.opens = 0
        self.events_per_open = events_per_open
        self.opens_cap = opens_cap

    async def send_command(self, command: dict[str, Any]) -> dict[str, Any] | None:  # noqa: ARG002
        return None

    async def close(self) -> None:
        return None

    def open_event_stream(self, params: dict[str, Any]) -> EventStreamHandle:  # noqa: ARG002
        self.opens += 1
        if self.opens > self.opens_cap:
            raise RuntimeError(f"opened more than {self.opens_cap} streams")
        open_index = self.opens
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()
        ready.set_result(None)
        done: asyncio.Future[BaseException | None] = loop.create_future()
        done.set_result(httpx.ReadTimeout("read operation timed out"))

        async def aiter() -> AsyncIterator[Any]:
            for index in range(self.events_per_open):
                yield values_event(seq=open_index * 1000 + index, counter=index)

        async def close() -> None:
            return None

        return EventStreamHandle(events=aiter(), ready=ready, done=done, close=close)


class ScriptedSyncTransport:
    """Sync twin: opens, yields nothing, drops with a read timeout."""

    thread_id = "t-1"

    def __init__(self, opens_cap: int = MAX_OPENS) -> None:
        self.opens = 0
        self.opens_cap = opens_cap

    def send_command(self, command: dict[str, Any]) -> dict[str, Any] | None:  # noqa: ARG002
        return None

    def close(self) -> None:
        return None

    def open_event_stream(self, params: dict[str, Any]) -> SyncEventStreamHandle:  # noqa: ARG002
        self.opens += 1
        if self.opens > self.opens_cap:
            raise RuntimeError(f"opened more than {self.opens_cap} streams")

        def aiter() -> Iterator[Any]:
            return iter(())

        return SyncEventStreamHandle(
            events=aiter(),
            error=lambda: httpx.ReadTimeout("read operation timed out"),
            close=lambda: None,
        )


def _controller(transport: Any) -> StreamController:
    return StreamController(
        transport=transport,
        max_reconnect_attempts=MAX_RECONNECT_ATTEMPTS,
        reconnect_backoff_base=0.001,
        reconnect_backoff_cap=0.002,
    )


@pytest.mark.anyio
async def test_streams_that_keep_dropping_empty_stop_after_the_attempt_budget() -> None:
    """A stream that is accepted and then dropped empty over and over must give up."""

    transport = ScriptedAsyncTransport(events_per_open=0)
    controller = _controller(transport)
    seen: list[Any] = []

    async def consume() -> None:
        async for event in controller.subscribe(["values"]):
            seen.append(event)

    await asyncio.wait_for(consume(), timeout=5.0)

    assert seen == []
    assert transport.opens <= 1 + MAX_RECONNECT_ATTEMPTS, (
        f"reconnected {transport.opens} times without receiving a single event"
    )


@pytest.mark.anyio
async def test_a_stream_that_delivers_events_is_still_reconnected() -> None:
    """Progress resets the budget: a stream that drops after an event may reconnect."""

    transport = ScriptedAsyncTransport(events_per_open=1, opens_cap=10_000)
    controller = _controller(transport)
    seen: list[Any] = []

    async def consume() -> None:
        async for event in controller.subscribe(["values"]):
            seen.append(event)
            if len(seen) >= 10:
                await controller.close()
                return

    await asyncio.wait_for(consume(), timeout=5.0)

    assert len(seen) >= 10
    assert transport.opens > 1 + MAX_RECONNECT_ATTEMPTS, (
        "the stream delivered events before dropping, which must keep reconnecting"
    )


def test_sync_streams_that_keep_dropping_empty_stop_after_the_attempt_budget() -> None:
    """The sync controller shares the budget rule."""

    transport = ScriptedSyncTransport()
    controller = SyncStreamController(
        transport,
        max_reconnect_attempts=MAX_RECONNECT_ATTEMPTS,
        reconnect_backoff_base=0.001,
        reconnect_backoff_cap=0.002,
    )
    controller.register_subscription({"channels": ["values"]})
    controller.reconcile_stream({"channels": ["values"]})

    def drive() -> None:
        controller.ensure_fanout_running()

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    controller.close()

    assert not thread.is_alive(), "sync fanout never stopped"
    assert transport.opens <= 1 + MAX_RECONNECT_ATTEMPTS, (
        f"reconnected {transport.opens} times without receiving a single event"
    )
