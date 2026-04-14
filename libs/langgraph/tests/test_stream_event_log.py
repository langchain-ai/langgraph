import asyncio

import pytest

from langgraph.stream._event_log import EventLog


@pytest.mark.anyio
async def test_push_and_iterate_in_order():
    log = EventLog()
    log.append("a")
    log.append("b")
    log.append("c")
    log.close()
    items = [item async for item in log.subscribe(0)]
    assert items == ["a", "b", "c"]


@pytest.mark.anyio
async def test_multiple_independent_cursors():
    log = EventLog()
    log.append("x")
    log.append("y")
    log.close()
    items1 = [item async for item in log.subscribe(0)]
    items2 = [item async for item in log.subscribe(0)]
    assert items1 == ["x", "y"]
    assert items2 == ["x", "y"]


@pytest.mark.anyio
async def test_close_ends_iteration():
    log = EventLog()
    log.close()
    items = [item async for item in log.subscribe(0)]
    assert items == []


@pytest.mark.anyio
async def test_fail_raises_error():
    log = EventLog()
    log.fail(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in log.subscribe(0):
            pass


@pytest.mark.anyio
async def test_nonzero_offset():
    log = EventLog()
    log.append("x")
    log.append("y")
    log.append("z")
    log.close()
    items = [item async for item in log.subscribe(2)]
    assert items == ["z"]


@pytest.mark.anyio
async def test_concurrent_push_and_iterate():
    log = EventLog()
    received = []

    async def consumer():
        async for item in log.subscribe(0):
            received.append(item)

    async def producer():
        for i in range(5):
            log.append(i)
            await asyncio.sleep(0.01)
        log.close()

    await asyncio.gather(producer(), consumer())
    assert received == [0, 1, 2, 3, 4]


@pytest.mark.anyio
async def test_items_before_cursor_visible():
    log = EventLog()
    log.append("a")
    log.append("b")
    cursor = log.subscribe(0)
    log.append("c")
    log.close()
    items = [item async for item in cursor]
    assert items == ["a", "b", "c"]


@pytest.mark.anyio
async def test_empty_log_closed_yields_nothing():
    log = EventLog()
    log.close()
    items = [item async for item in log.subscribe(0)]
    assert items == []


@pytest.mark.anyio
async def test_fail_mid_iteration():
    """A cursor that has consumed some items should raise when fail() is called."""
    log = EventLog()
    received = []

    async def consumer():
        async for item in log.subscribe(0):
            received.append(item)

    async def producer():
        log.append("a")
        log.append("b")
        await asyncio.sleep(0.02)
        log.fail(RuntimeError("mid-stream error"))

    with pytest.raises(RuntimeError, match="mid-stream error"):
        await asyncio.gather(producer(), consumer())
    assert received == ["a", "b"]


@pytest.mark.anyio
async def test_abandoned_cursor_cleans_up_waiters():
    """Abandoned async cursors should not leave stale futures in the
    EventLog waiter list.

    When a cursor's __anext__ is cancelled (e.g. consumer breaks out of
    ``async for``), the Future it registered in ``_waiters`` should be
    cleaned up.  Otherwise the list grows without bound until the next
    append/close/fail triggers ``_wake_all()``.
    """
    log: EventLog[str] = EventLog()

    for _ in range(10):
        cursor = log.subscribe(0)
        task = asyncio.ensure_future(cursor.__anext__())
        await asyncio.sleep(0)  # let task register its waiter
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert len(log._waiters) == 0, (
        f"Expected 0 waiters after abandoning 10 cursors, "
        f"got {len(log._waiters)}.  Abandoned cursors leak futures."
    )
