from __future__ import annotations

import asyncio

from langgraph_sdk.stream.transport.http import EventStreamHandle


async def test_event_stream_handle_constructs_with_open_state():
    ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    closed = False

    async def aiter_events():
        if False:
            yield  # pragma: no cover

    async def closer():
        nonlocal closed
        closed = True

    handle = EventStreamHandle(events=aiter_events(), ready=ready, close=closer)
    assert handle.ready is ready
    await handle.close()
    assert closed is True
