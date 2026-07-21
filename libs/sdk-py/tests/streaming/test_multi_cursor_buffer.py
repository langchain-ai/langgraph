from __future__ import annotations

import asyncio

from langgraph_sdk.stream.multi_cursor_buffer import MultiCursorBuffer


async def _drain(buf: MultiCursorBuffer[int]) -> list[int]:
    return [item async for item in buf]


async def test_late_subscriber_replays_from_index_zero():
    buf: MultiCursorBuffer[int] = MultiCursorBuffer()
    buf.push(1)
    buf.push(2)
    buf.push(3)
    buf.close()
    assert await _drain(buf) == [1, 2, 3]


async def test_two_iterators_each_get_full_log():
    buf: MultiCursorBuffer[int] = MultiCursorBuffer()
    buf.push(1)
    buf.push(2)
    buf.close()
    a, b = await asyncio.gather(_drain(buf), _drain(buf))
    assert a == [1, 2]
    assert b == [1, 2]


async def test_iterator_waits_for_new_items():
    buf: MultiCursorBuffer[int] = MultiCursorBuffer()
    drain_task = asyncio.create_task(_drain(buf))
    # Yield so the drain task starts and parks at the tail.
    await asyncio.sleep(0)
    assert len(buf._wakeups) == 1, "cursor must have suspended before push"
    buf.push(10)
    buf.push(20)
    buf.close()
    assert await drain_task == [10, 20]


async def test_close_releases_waiting_iterators():
    buf: MultiCursorBuffer[int] = MultiCursorBuffer()
    drain_task = asyncio.create_task(_drain(buf))
    await asyncio.sleep(0)
    buf.close()
    assert await asyncio.wait_for(drain_task, timeout=1.0) == []


async def test_len_reports_buffered_count():
    buf: MultiCursorBuffer[int] = MultiCursorBuffer()
    assert len(buf) == 0
    buf.push(1)
    buf.push(2)
    assert len(buf) == 2
