"""Tests for StreamController, _SeenEventIds, and related stream/controller.py types."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from langgraph_sdk.stream.controller import StreamController, _SeenEventIds
from langgraph_sdk.stream.transport.http import EventStreamHandle

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


# ---------------------------------------------------------------------------
# Reconnect helpers
# ---------------------------------------------------------------------------


def _make_handle(
    *,
    error: BaseException | None = None,
) -> EventStreamHandle:
    """Build a minimal EventStreamHandle that closes immediately."""
    loop = asyncio.get_running_loop()
    ready: asyncio.Future[None] = loop.create_future()
    ready.set_result(None)
    done: asyncio.Future[BaseException | None] = loop.create_future()
    done.set_result(error)

    async def _aiter() -> AsyncIterator[Any]:
        if False:
            yield  # pragma: no cover

    return EventStreamHandle(
        events=_aiter(),
        ready=ready,
        done=done,
        close=AsyncMock(),
    )


def _always_error_transport(error_type: type[Exception] = RuntimeError) -> Any:
    """Return a fake transport whose open_event_stream always raises."""

    class _Transport:
        def open_event_stream(self, _params: dict[str, Any]) -> EventStreamHandle:
            raise error_type("scripted transport error")

    return _Transport()


def _error_then_succeed_transport(fail_count: int) -> Any:
    """Return a fake transport that fails *fail_count* times then succeeds."""
    calls = [0]

    class _Transport:
        def open_event_stream(self, _params: dict[str, Any]) -> EventStreamHandle:
            calls[0] += 1
            if calls[0] <= fail_count:
                raise RuntimeError(f"scripted error #{calls[0]}")
            return _make_handle()

    return _Transport()


# ---------------------------------------------------------------------------
# Task 8.1: Exp+jitter backoff
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_reconnect_uses_exp_backoff_with_jitter(monkeypatch):
    """Reconnect attempts should sleep increasing durations with jitter,
    not a fixed 50ms."""
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    transport = _always_error_transport()
    controller = StreamController(
        transport=transport,
        run_start_gate=AsyncMock(),
        max_reconnect_attempts=3,
        reconnect_backoff_base=0.1,
        reconnect_backoff_cap=2.0,
    )
    # Seed filter so reconnect doesn't bail early.
    controller._shared_stream_filter = {"channels": ["lifecycle"]}

    await controller._reconnect_shared_stream()

    # Should have slept once per attempt.
    assert len(sleeps) == 3
    # All sleeps within [base, cap + 25% jitter].
    assert all(0.1 <= s <= 2.5 for s in sleeps)


@pytest.mark.anyio
async def test_reconnect_accepts_backoff_kwargs():
    """StreamController must accept reconnect_backoff_base and _cap kwargs."""
    controller = StreamController(
        transport=_always_error_transport(),
        run_start_gate=AsyncMock(),
        max_reconnect_attempts=1,
        reconnect_backoff_base=0.05,
        reconnect_backoff_cap=1.0,
    )
    assert controller._reconnect_backoff_base == 0.05
    assert controller._reconnect_backoff_cap == 1.0


# ---------------------------------------------------------------------------
# Task 8.2: Close old handle before reconnect
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_transport_drop_exception_logged_with_type(monkeypatch, caplog):
    """Bare `pass` discarded exception types; the drop should at least log."""
    import logging

    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    loop = asyncio.get_running_loop()
    ready: asyncio.Future[None] = loop.create_future()
    ready.set_result(None)
    done: asyncio.Future[BaseException | None] = loop.create_future()
    done.set_result(RuntimeError("transport drop"))

    async def _raises() -> AsyncIterator[Any]:
        raise RuntimeError("transport drop")
        yield  # pragma: no cover

    old_handle = EventStreamHandle(
        events=_raises(),
        ready=ready,
        done=done,
        close=AsyncMock(),
    )

    transport = _always_error_transport()
    controller = StreamController(
        transport=transport,
        run_start_gate=AsyncMock(),
        max_reconnect_attempts=1,
        reconnect_backoff_base=0.0,
        reconnect_backoff_cap=0.0,
    )
    controller._shared_stream = old_handle
    controller._shared_stream_filter = {"channels": ["lifecycle"]}

    with caplog.at_level(logging.DEBUG, logger="langgraph_sdk.stream.controller"):
        await controller._fanout()

    assert any("transport drop" in rec.message for rec in caplog.records)


@pytest.mark.anyio
async def test_reconnect_closes_old_handle_before_opening_new(monkeypatch):
    """When the shared stream errors and triggers reconnect, the old
    EventStreamHandle's close() must be called."""
    # Suppress actual sleeps.
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    close_calls: list[str] = []

    loop = asyncio.get_running_loop()
    old_ready: asyncio.Future[None] = loop.create_future()
    old_ready.set_result(None)
    # done resolves with an error to trigger the reconnect path in _fanout.
    old_done: asyncio.Future[BaseException | None] = loop.create_future()
    old_done.set_result(RuntimeError("transport drop"))

    async def _empty() -> AsyncIterator[Any]:
        # Raise on the first iteration so _fanout exits the inner loop.
        raise RuntimeError("transport drop")
        yield  # pragma: no cover

    old_handle = EventStreamHandle(
        events=_empty(),
        ready=old_ready,
        done=old_done,
        close=AsyncMock(side_effect=lambda: close_calls.append("old_closed")),
    )

    # Transport always errors so reconnect exhausts all attempts and _fanout exits.
    transport = _always_error_transport()
    controller = StreamController(
        transport=transport,
        run_start_gate=AsyncMock(),
        max_reconnect_attempts=1,
        reconnect_backoff_base=0.0,
        reconnect_backoff_cap=0.0,
    )
    controller._shared_stream = old_handle
    controller._shared_stream_filter = {"channels": ["lifecycle"]}

    # _fanout drives reconnect; wait for it to complete.
    await controller._fanout()

    assert "old_closed" in close_calls
