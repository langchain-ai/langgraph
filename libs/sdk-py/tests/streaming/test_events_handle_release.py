"""Regression test for #9620: completed event handles must not be retained."""

from __future__ import annotations

import asyncio

import pytest

from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk.stream.transport import EventStreamHandle

pytestmark = pytest.mark.anyio


async def test_events_handles_released_after_completion():
    stream = AsyncThreadStream(
        http=None, thread_id="example", assistant_id="example"
    )

    async def empty_events():
        if False:
            yield {}

    async def close_handle():
        pass

    class Transport:
        def open_event_stream(self, params):
            loop = asyncio.get_running_loop()
            return EventStreamHandle(
                empty_events(),
                loop.create_future(),
                loop.create_future(),
                close_handle,
            )

        async def close(self):
            pass

    stream._transport = Transport()

    for _ in range(1000):
        async for _event in stream.events:
            pass

    # Completed iterators must release their handles immediately.
    assert len(stream._open_handles) == 0

    await stream.close()
    # Closing the session must release any remaining handles.
    assert len(stream._open_handles) == 0
