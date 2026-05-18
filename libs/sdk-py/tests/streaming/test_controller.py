"""Tests for StreamController, _SeenEventIds, and related stream/controller.py types."""

from __future__ import annotations

import pytest

from langgraph_sdk.stream.controller import StreamController, _SeenEventIds

# ---------------------------------------------------------------------------
# Task 3.1: bounded subscription queues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscription_queue_bounded_by_max_queue_size():
    """`StreamController` must create per-subscription queues bounded by `max_queue_size`."""
    import httpx

    from langgraph_sdk.stream.transport.http import ProtocolSseTransport

    transport = ProtocolSseTransport(
        client=httpx.AsyncClient(base_url="http://test"),
        thread_id="t-1",
    )
    controller = StreamController(transport=transport, max_queue_size=4)
    sub = controller._register_subscription({"channels": ["values"]})
    assert sub.queue.maxsize == 4


@pytest.mark.asyncio
async def test_subscription_queue_default_max_queue_size_is_1024():
    """`StreamController` default `max_queue_size` is 1024."""
    import httpx

    from langgraph_sdk.stream.transport.http import ProtocolSseTransport

    transport = ProtocolSseTransport(
        client=httpx.AsyncClient(base_url="http://test"),
        thread_id="t-1",
    )
    controller = StreamController(transport=transport)
    sub = controller._register_subscription({"channels": ["values"]})
    assert sub.queue.maxsize == 1024


# ---------------------------------------------------------------------------
# Task 3.2: bounded LRU seen-event-ids
# ---------------------------------------------------------------------------


def test_seen_event_ids_is_bounded_lru():
    """`_SeenEventIds` must evict oldest entries when capacity is exceeded.

    Default cap is 10_000; explicit kwarg overrides.
    """
    seen = _SeenEventIds(maxsize=3)
    seen.add("a")
    seen.add("b")
    seen.add("c")
    assert "a" in seen
    seen.add("d")
    assert "a" not in seen  # evicted
    assert {"b", "c", "d"} <= set(seen)


def test_seen_event_ids_move_to_end_on_re_add():
    """`_SeenEventIds.add` of an existing key must promote it (LRU move-to-end)."""
    seen = _SeenEventIds(maxsize=3)
    seen.add("a")
    seen.add("b")
    seen.add("c")
    # Re-adding "a" should promote it so "b" is evicted next.
    seen.add("a")
    seen.add("d")
    assert "b" not in seen  # "b" was the oldest, "a" was promoted
    assert "a" in seen


def test_seen_event_ids_default_maxsize_is_10000():
    """Default `_SeenEventIds` max is 10_000."""
    seen = _SeenEventIds()
    # Add 10_000 + 1 entries.
    for i in range(10_001):
        seen.add(str(i))
    # "0" (the first added) should have been evicted.
    assert "0" not in seen
    assert "10000" in seen


def test_seen_event_ids_contains_false_for_missing():
    seen = _SeenEventIds(maxsize=10)
    assert "missing" not in seen


def test_seen_event_ids_iter_returns_keys():
    seen = _SeenEventIds(maxsize=10)
    seen.add("x")
    seen.add("y")
    assert set(seen) == {"x", "y"}


# ---------------------------------------------------------------------------
# Task 3.3: close() awaits pending rotation closes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_awaits_pending_rotation_closes():
    """When a rotation is mid-flight, controller.close() must await the old
    stream close before returning."""
    import asyncio as _asyncio

    import httpx

    from langgraph_sdk.stream.controller import _close_after
    from langgraph_sdk.stream.transport.http import ProtocolSseTransport

    rotation_close_done = _asyncio.Event()

    class _SlowHandle:
        """A fake EventStreamHandle whose close() takes a moment."""

        def __init__(self):
            self.events = self._empty()
            loop = _asyncio.get_running_loop()
            self.ready: _asyncio.Future[None] = loop.create_future()
            self.ready.set_result(None)
            self.done: _asyncio.Future[None] = loop.create_future()

        async def _empty(self):
            if False:
                yield  # pragma: no cover

        async def close(self):
            await _asyncio.sleep(0.05)
            rotation_close_done.set()

    transport = ProtocolSseTransport(
        client=httpx.AsyncClient(base_url="http://test"),
        thread_id="t-1",
    )
    controller = StreamController(transport=transport)

    # Simulate a mid-flight rotation close by directly injecting a task.
    slow_handle = _SlowHandle()
    task = _asyncio.create_task(
        _close_after(slow_handle)  # ty: ignore[invalid-argument-type]
    )
    controller._rotation_close_tasks.add(task)
    task.add_done_callback(controller._rotation_close_tasks.discard)

    # close() must block until the rotation close completes.
    await controller.close()
    assert rotation_close_done.is_set()
