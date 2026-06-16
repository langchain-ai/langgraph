"""Public conformance helper for the transport replay contract.

Usage:

    from tests.streaming.assert_transport_replays import assert_transport_replays

    async def test_my_transport():
        async with my_transport_factory() as harness:
            await assert_transport_replays(harness)

The helper publishes a few events into a transport's underlying buffer
(via whatever side-channel the implementation exposes — typically by
scripting the fake server) and verifies that a fresh `open_event_stream`
yields them all before any new live events.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from langchain_protocol import Event

from langgraph_sdk.stream.transport.http import ProtocolSseTransport
from streaming._events import lifecycle_event


class _ReplayableHarness(Protocol):
    transport: ProtocolSseTransport

    def script_buffered(self, events: list[dict]) -> None: ...


async def assert_transport_replays(
    harness: _ReplayableHarness,
    *,
    buffered_count: int = 3,
    timeout: float = 1.0,
) -> None:
    """Assert that `harness.transport` replays buffered events on subscribe.

    Args:
        harness: object exposing an open `ProtocolSseTransport` plus a
            `script_buffered(events)` method that queues events as if they
            were buffered server-side before the subscription opens.
        buffered_count: how many synthetic events to script.
        timeout: per-step await timeout in seconds.

    Raises:
        AssertionError: when fewer than `buffered_count` events arrive (or
            arrive out of order) on the fresh stream before it closes.
    """
    events = [lifecycle_event(seq=i) for i in range(buffered_count)]
    harness.script_buffered(events)
    handle = harness.transport.open_event_stream({"channels": ["lifecycle"]})
    await asyncio.wait_for(handle.ready, timeout=timeout)

    received: list[Event] = []

    async def drain() -> None:
        async for event in handle.events:
            received.append(event)

    try:
        await asyncio.wait_for(drain(), timeout=timeout)
    finally:
        await handle.close()

    seqs = [e["seq"] for e in received]
    assert seqs == list(range(buffered_count)), (
        f"transport did not replay buffered events: expected "
        f"{list(range(buffered_count))}, got {seqs}"
    )
