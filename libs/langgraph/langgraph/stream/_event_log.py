from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class EventLog(Generic[T]):
    """Append-only buffer with multi-cursor sync and async iteration.

    Supports multiple independent consumers iterating the same log
    concurrently. Each call to ``__iter__`` or ``__aiter__`` creates a
    new cursor starting from the beginning.

    A given instance should be used in either sync or async mode — the
    two notification paths are independent and do not interfere, but
    mixing them on one instance is not tested.

    Producer API (thread-safe):
        push(item)  — append an item, notify all waiting cursors
        close()     — mark the log as done
        fail(err)   — mark the log as errored

    Consumer API:
        __iter__()  — new sync cursor (blocks via threading.Condition)
        __aiter__() — new async cursor (awaits via asyncio.Future)
    """

    def __init__(self) -> None:
        self._items: list[T] = []
        self._closed = False
        self._error: BaseException | None = None
        # Sync notification
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # Async notification — futures created lazily by async cursors
        self._async_waiters: list[asyncio.Future[None]] = []

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def push(self, item: T) -> None:
        """Append *item* and wake all waiting cursors."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot push to a closed EventLog")
            self._items.append(item)
            self._cond.notify_all()
        self._wake_async()

    def close(self) -> None:
        """Mark the log as complete — open cursors will finish cleanly."""
        with self._lock:
            self._closed = True
            self._cond.notify_all()
        self._wake_async()

    def fail(self, err: BaseException) -> None:
        """Mark the log as errored — open cursors will raise *err*."""
        with self._lock:
            self._error = err
            self._closed = True
            self._cond.notify_all()
        self._wake_async()

    # ------------------------------------------------------------------
    # Sync iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[T]:
        """Return a new independent sync cursor over the log."""
        return self._sync_cursor()

    def _sync_cursor(self) -> Iterator[T]:
        cursor = 0
        while True:
            with self._lock:
                # Wait until data is available or the log is done.
                while cursor >= len(self._items) and not self._closed:
                    self._cond.wait()
                # Yield available items before raising errors, matching
                # the async cursor's behavior.
                if cursor < len(self._items):
                    item = self._items[cursor]
                    cursor += 1
                elif self._error is not None:
                    raise self._error
                else:
                    # closed and no more items
                    return
            yield item

    # ------------------------------------------------------------------
    # Async iteration
    # ------------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[T]:
        """Return a new independent async cursor over the log."""
        return self._async_cursor()

    async def _async_cursor(self) -> AsyncIterator[T]:
        cursor = 0
        while True:
            if cursor < len(self._items):
                yield self._items[cursor]
                cursor += 1
            elif self._closed:
                if self._error is not None:
                    raise self._error
                return
            else:
                # Wait for notification from push/close/fail.
                loop = asyncio.get_running_loop()
                fut: asyncio.Future[None] = loop.create_future()
                self._async_waiters.append(fut)
                await fut

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wake_async(self) -> None:
        """Resolve all pending async futures (safe from any thread)."""
        waiters = self._async_waiters
        if not waiters:
            return
        self._async_waiters = []
        for fut in waiters:
            if not fut.done():
                try:
                    fut.get_loop().call_soon_threadsafe(fut.set_result, None)
                except RuntimeError:
                    # Event loop already closed — nothing to notify.
                    pass
